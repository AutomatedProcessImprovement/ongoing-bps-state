import datetime
import json
import logging
import multiprocessing
import os
import re
import secrets
import shutil
import threading
import time
from pathlib import Path
from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool
from lxml import etree
import pandas as pd
from ongoing_process_state.utils import read_bpmn_model
from pix_framework.io.event_log import read_csv_log, EventLogIDs
from pydantic import BaseModel

from src.compute_bps_state_and_run_simulation import compute_bps_state_and_run_simulation, compute_bps_resumed_state, \
    generate_events_with_token_movements, ongoing_case_ids, rerun_from_process_state, write_empty_process_state
from src.compute_frontend_events_from_trace import sim_log_ids
from src.run_metrics import paired_change, resource_timeline, summarize_log
from src.scenario import InvalidEdit, apply_edits, can_edit_pools, pool_size

from fastapi import Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from db.database import get_db
from db.ngram_repository import delete_ngrams_by_process_id

app = FastAPI()
logger = logging.getLogger(__name__)

PROCESSES_DIR = Path("./processes")
PROCESS_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}")


def process_folder_of(process_id: str) -> Path:
    if not PROCESS_ID_PATTERN.fullmatch(process_id):
        raise HTTPException(status_code=400, detail="Invalid process id")
    return PROCESSES_DIR / process_id


RUN_TTL_SECONDS = 3600
SCENARIO_REPEATS = 8
REPEAT_BUDGET_SECONDS = float(os.getenv("SCENARIO_REPEAT_BUDGET_SECONDS", "60"))
PROCESS_STATE_FILE = "process_state.pkl"
RUN_SETTINGS_FILE = "run.json"
runs: dict = {}
multiprocessing.set_start_method("forkserver", force=True)
multiprocessing.set_forkserver_preload(["pix_framework.enhancement.concurrency_oracle"])

DISCOVER_ENABLEMENT = "__DISCOVER__"
BPMN_NAMESPACE = "http://www.omg.org/spec/BPMN/20100524/MODEL"
SILENT_NODES = ("exclusiveGateway", "inclusiveGateway", "parallelGateway", "intermediateCatchEvent", "intermediateThrowEvent")

runs_lock = threading.Lock()
simulation_lock = threading.Lock()


def set_stage(process_id: str, stage: str):
    now = time.time()
    with runs_lock:
        for stale_id in [key for key, run in runs.items() if now - run["updated"] > RUN_TTL_SECONDS]:
            del runs[stale_id]
        run = runs.setdefault(process_id, {"stage": None, "stages": {}, "error": None, "updated": now})
        if run["stage"] in run["stages"]:
            run["stages"][run["stage"]]["finished"] = now
        run["stage"] = stage
        run["stages"][stage] = {"started": now, "finished": None}
        run["updated"] = now


def finish_run(process_id: str, error: str = None) -> str:
    now = time.time()
    with runs_lock:
        run = runs.get(process_id)
        if run is None:
            return None
        stage = run["stage"]
        if stage in run["stages"]:
            run["stages"][stage]["finished"] = now
        run["error"] = error
        run["stage"] = "failed" if error else "done"
        run["updated"] = now
        return stage


@app.get("/progress/{process_id}")
def run_progress(process_id: str):
    now = time.time()
    with runs_lock:
        run = runs.get(process_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Unknown run")
        return {
            "stage": run["stage"],
            "error": run["error"],
            "durations": {
                stage: round((times["finished"] or now) - times["started"], 2)
                for stage, times in run["stages"].items()
            },
        }


@app.get("/health")
def health(db: Session = Depends(get_db)):
    db.execute(text("SELECT 1"))
    return {"status": "ok"}

def log_ids_of(column_mapping: dict) -> EventLogIDs:
    return EventLogIDs(
        case=column_mapping.get("case", "CaseId"),
        activity=column_mapping.get("activity", "Activity"),
        resource=column_mapping.get("resource", "Resource"),
        start_time=column_mapping.get("start", "StartTime"),
        end_time=column_mapping.get("end", "EndTime"),
        enabled_time=column_mapping.get("enablement", "enabled_time"),
    )


def state_column_mapping(log_ids: EventLogIDs) -> str:
    mapping = {
        log_ids.case: "CaseId",
        log_ids.activity: "Activity",
        log_ids.resource: "Resource",
        log_ids.start_time: "StartTime",
        log_ids.end_time: "EndTime",
    }
    if log_ids.enabled_time != DISCOVER_ENABLEMENT:
        mapping[log_ids.enabled_time] = "enabled_time"
    return json.dumps(mapping)


def name_model_nodes(bpmn_model_path: Path):
    tree = etree.parse(str(bpmn_model_path))
    names = {node.get("name") for node in tree.iter() if isinstance(node.tag, str)}
    changed = False
    for event, name in (("startEvent", "Process start"), ("endEvent", "Process end")):
        for node in tree.iter(f"{{{BPMN_NAMESPACE}}}{event}"):
            if not (node.get("name") or "").strip() and name not in names:
                node.set("name", name)
                changed = True
    for tag in SILENT_NODES:
        for node in tree.iter(f"{{{BPMN_NAMESPACE}}}{tag}"):
            if node.get("name") != node.get("id"):
                node.set("name", node.get("id"))
                changed = True
    if changed:
        tree.write(str(bpmn_model_path), xml_declaration=True, encoding="UTF-8")


def read_run_settings(process_folder: Path):
    settings_path = process_folder / RUN_SETTINGS_FILE
    return json.loads(settings_path.read_text()) if settings_path.exists() else None


def note_run_seconds(process_id: str, process_folder: Path):
    with runs_lock:
        stages = dict(runs.get(process_id, {}).get("stages", {}))
    seconds = {
        f"{stage}_seconds": round(times["finished"] - times["started"], 2)
        for stage, times in stages.items()
        if stage in ("state", "simulation") and times["finished"]
    }
    settings_path = process_folder / RUN_SETTINGS_FILE
    settings_path.write_text(json.dumps({**read_run_settings(process_folder), **seconds}))


def simulate(
        process_id: str,
        process_folder: Path,
        ongoing_log_path: Path,
        ongoing_log_ids: EventLogIDs,
        bpmn_model_path: Path,
        bpmn_parameters_path: Path,
        start_time: str,
        simulation_horizon: str,
        seed: int,
):
    with simulation_lock:
        # Path to files to store intermediate objects
        short_term_simulated_log_path = process_folder / "short-term-simulation.csv"

        # Compute the initial frame and run short-term simulation
        frame, short_term_simulated_log_path = compute_bps_state_and_run_simulation(
            ongoing_log_path=ongoing_log_path,
            bpmn_model_path=bpmn_model_path,
            bpmn_parameters_path=bpmn_parameters_path,
            start_time=start_time,
            simulation_horizon=simulation_horizon,
            short_term_simulated_log_path=short_term_simulated_log_path,
            column_mapping=state_column_mapping(ongoing_log_ids),
            on_stage=lambda stage: set_stage(process_id, stage),
            process_state_path=process_folder / PROCESS_STATE_FILE,
            seed=seed,
        )
        set_stage(process_id, "playback")
        return prepare_playback(
            process_folder, frame, ongoing_log_path, ongoing_log_ids, bpmn_model_path, start_time,
            short_term_simulated_log_path,
        )


def prepare_playback(
        process_folder: Path,
        frame: list,
        ongoing_log_path: Path,
        ongoing_log_ids: EventLogIDs,
        bpmn_model_path: Path,
        start_time: str,
        short_term_simulated_log_path: Path,
):
    post_processed_log_path = process_folder / "short-term-simulation-processed.csv"
    reachability_graph_path = process_folder / "complete_reachability_graph.tgf"

    # Post-process simulated log by:
    # - Adding the ongoing case prefixes (activity instances prior to [start_time])
    # - Adding the start event prior to the first event of each case.
    # - Adding the end event after the last event of each case.
    post_process_simulated_log(
        ongoing_log_path=ongoing_log_path,
        ongoing_log_ids=ongoing_log_ids,
        start_time=pd.Timestamp(start_time),
        simulated_log_path=short_term_simulated_log_path,
        post_processed_log_path=post_processed_log_path,
        bpmn_model_path=bpmn_model_path,
        ongoing_case_ids=[case['case_id'] for case in frame],
    )

    # Compute token events with token movements for front-end
    events = generate_events_with_token_movements(
        bpmn_model_path=bpmn_model_path,
        start_timestamp=pd.Timestamp(start_time),
        short_term_simulation_path=post_processed_log_path,
        reachability_graph_path=reachability_graph_path,
        frame=frame,
    )

    for event in events:
        event['timestamp'] = event['timestamp'].strftime("%Y-%m-%dT%H:%M:%S.%f%z")

    arrival_times = compute_ongoing_case_arrivals(
        ongoing_log_path=ongoing_log_path,
        ongoing_log_ids=ongoing_log_ids,
        ongoing_case_ids=[case['case_id'] for case in frame],
    )
    task_times = compute_ongoing_task_times(
        simulated_log_path=short_term_simulated_log_path,
        bpmn_model_path=bpmn_model_path,
        start_time=pd.Timestamp(start_time),
        ongoing_case_ids=[case['case_id'] for case in frame],
    )
    for case in frame:
        case['arrival_time'] = arrival_times.get(str(case['case_id']))
        case['task_times'] = task_times.get(str(case['case_id']), {})

    return {
        "frames": frame,
        "events": events,
    }


def simulate_scenario(process_id: str, base_process_id: str, edits: dict):
    base_folder = process_folder_of(base_process_id)
    settings = read_run_settings(base_folder)
    if settings is None or not (base_folder / PROCESS_STATE_FILE).exists():
        raise InvalidEdit("This simulation was created before what-if runs existed. Run it again to use them.")

    parameters = apply_edits(json.loads((base_folder / "json_parameters.json").read_text()), edits)

    process_folder = process_folder_of(process_id)
    process_folder.mkdir(parents=True, exist_ok=True)
    bpmn_model_path = process_folder / "bpmn_model.bpmn"
    bpmn_parameters_path = process_folder / "json_parameters.json"
    bpmn_parameters_path.write_text(json.dumps(parameters))
    for name in ["bpmn_model.bpmn", "complete_reachability_graph.tgf", "reachability_graph_with_events.tgf",
                 PROCESS_STATE_FILE]:
        if (base_folder / name).exists():
            shutil.copy(base_folder / name, process_folder / name)
    (process_folder / RUN_SETTINGS_FILE).write_text(json.dumps({
        **settings, "base_process_id": base_process_id, "edits": edits,
    }))

    seed = settings["seed"]
    start_time = settings["start_time"]
    simulation_horizon = settings["simulation_horizon"]
    ongoing_log_path = base_folder / "ongoing_event_log.csv"
    ongoing_log_ids = log_ids_of(settings["column_mapping"])
    window = pd.Timestamp(start_time), pd.Timestamp(simulation_horizon)
    log_path = process_folder / "short-term-simulation.csv"

    as_is_state_path = base_folder / PROCESS_STATE_FILE
    to_be_state_path = process_folder / PROCESS_STATE_FILE
    empty_start = bool(edits.get("empty_start"))

    with simulation_lock:
        set_stage(process_id, "state")
        if empty_start:
            write_empty_process_state(as_is_state_path, to_be_state_path)
        rerun = dict(
            bpmn_model_path=bpmn_model_path, start_time=start_time, simulation_horizon=simulation_horizon,
        )
        set_stage(process_id, "simulation")
        simulation_started = time.monotonic()
        frame = rerun_from_process_state(
            **rerun, process_state_path=to_be_state_path, bpmn_parameters_path=bpmn_parameters_path,
            short_term_simulated_log_path=log_path, seed=seed,
        )
        simulation_seconds = time.monotonic() - simulation_started

        as_is_arrivals = compute_ongoing_case_arrivals(
            ongoing_log_path, ongoing_log_ids, ongoing_case_ids(as_is_state_path),
        )
        to_be_arrivals = {} if empty_start else as_is_arrivals
        as_is = summarize_log(read_csv_log(base_folder / "short-term-simulation.csv", sim_log_ids), *window, as_is_arrivals)
        to_be = summarize_log(read_csv_log(log_path, sim_log_ids), *window, to_be_arrivals)

        set_stage(process_id, "repeats")
        repeats_folder = process_folder / "repeats"
        repeats_folder.mkdir(exist_ok=True)
        pairs = [{"asIs": as_is, "toBe": to_be}]
        repeats = min(SCENARIO_REPEATS, int(REPEAT_BUDGET_SECONDS // max(2 * simulation_seconds, 0.01)))
        for number in range(1, repeats + 1):
            pair = {}
            for side, parameters_path, state_path, arrivals in [
                ("asIs", base_folder / "json_parameters.json", as_is_state_path, as_is_arrivals),
                ("toBe", bpmn_parameters_path, to_be_state_path, to_be_arrivals),
            ]:
                repeat_log_path = repeats_folder / f"{side}-{number}.csv"
                rerun_from_process_state(
                    **rerun, process_state_path=state_path, bpmn_parameters_path=parameters_path,
                    short_term_simulated_log_path=repeat_log_path, seed=seed + number,
                )
                pair[side] = summarize_log(read_csv_log(repeat_log_path, sim_log_ids), *window, arrivals)
            pairs.append(pair)
        shutil.rmtree(repeats_folder, ignore_errors=True)

        set_stage(process_id, "playback")
        result = prepare_playback(
            process_folder, frame, ongoing_log_path, ongoing_log_ids, bpmn_model_path, start_time, log_path,
        )

    return {
        **result,
        "comparison": {
            "asIs": as_is,
            "toBe": to_be,
            "poolsAsIs": pool_summary(base_folder, *window),
            "poolsToBe": pool_summary(process_folder, *window),
            "repeats": paired_change(pairs),
        },
    }


def pool_summary(process_folder: Path, start: pd.Timestamp, end: pd.Timestamp) -> list:
    timeline = resources_of(process_folder, start, end)
    return [{key: pool[key] for key in ["id", "name", "size", "utilization"]} for pool in timeline["pools"]]


def resources_of(process_folder: Path, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    log = read_csv_log(process_folder / "short-term-simulation.csv", sim_log_ids)
    parameters = json.loads((process_folder / "json_parameters.json").read_text())
    return resource_timeline(log, parameters, start, end)


@app.post("/start")
async def start_short_term_simulation(
    process_id: str = Form(...),
    start_time: str = Form(...),
    simulation_horizon: str = Form(...),
    column_mapping: str = Form(...),
    event_log: UploadFile = File(...),
    bpmn_model: UploadFile = File(...),
    json_parameters: UploadFile = File(...),
):
    process_folder = process_folder_of(process_id)
    process_folder.mkdir(parents=True, exist_ok=True)

    ongoing_log_path = process_folder / "ongoing_event_log.csv"
    with open(ongoing_log_path, "wb") as f:
        f.write(await event_log.read())

    column_mapping = json.loads(column_mapping)
    ongoing_log_ids = log_ids_of(column_mapping)

    bpmn_model_path = process_folder / "bpmn_model.bpmn"
    with open(bpmn_model_path, "wb") as f:
        f.write(await bpmn_model.read())
    name_model_nodes(bpmn_model_path)

    bpmn_parameters_path = process_folder / "json_parameters.json"
    with open(bpmn_parameters_path, "wb") as f:
        f.write(await json_parameters.read())

    seed = secrets.randbelow(2 ** 31)
    (process_folder / RUN_SETTINGS_FILE).write_text(json.dumps({
        "seed": seed,
        "start_time": start_time,
        "simulation_horizon": simulation_horizon,
        "column_mapping": column_mapping,
        "files": {"log": event_log.filename, "bpmn": bpmn_model.filename, "parameters": json_parameters.filename},
    }))

    set_stage(process_id, "queued")
    try:
        result = await run_in_threadpool(
            simulate, process_id, process_folder, ongoing_log_path, ongoing_log_ids, bpmn_model_path,
            bpmn_parameters_path, start_time, simulation_horizon, seed,
        )
    except Exception as error:
        logger.exception("Simulation %s failed", process_id)
        stage = finish_run(process_id, error=f"{type(error).__name__}: {error}")
        raise HTTPException(status_code=422, detail={"stage": stage, "message": f"{type(error).__name__}: {error}"})
    finish_run(process_id)
    note_run_seconds(process_id, process_folder)
    return result

class ScenarioEdits(BaseModel):
    pools: dict[str, int] = {}
    arrival_percent: int = 0
    tasks: dict[str, int] = {}
    empty_start: bool = False


class ScenarioRequest(BaseModel):
    process_id: str
    base_process_id: str
    edits: ScenarioEdits


@app.post("/scenario")
def start_scenario(request: ScenarioRequest):
    process_folder_of(request.process_id)
    set_stage(request.process_id, "queued")
    try:
        result = simulate_scenario(request.process_id, request.base_process_id, request.edits.model_dump())
    except InvalidEdit as error:
        stage = finish_run(request.process_id, error=str(error))
        shutil.rmtree(process_folder_of(request.process_id), ignore_errors=True)
        raise HTTPException(status_code=422, detail={"stage": stage, "message": str(error)})
    except Exception as error:
        logger.exception("Scenario %s failed", request.process_id)
        stage = finish_run(request.process_id, error=f"{type(error).__name__}: {error}")
        shutil.rmtree(process_folder_of(request.process_id), ignore_errors=True)
        raise HTTPException(status_code=422, detail={"stage": stage, "message": f"{type(error).__name__}: {error}"})
    finish_run(request.process_id)
    return result


@app.get("/what-if/{process_id}")
def what_if_options(process_id: str):
    process_folder = process_folder_of(process_id)
    parameters_path = process_folder / "json_parameters.json"
    if not parameters_path.exists():
        raise HTTPException(status_code=404, detail="Unknown process")
    parameters = json.loads(parameters_path.read_text())
    settings = read_run_settings(process_folder)
    bpmn_model = read_bpmn_model(process_folder / "bpmn_model.bpmn")
    task_ids = {task["task_id"] for task in parameters["task_resource_distribution"]}
    return {
        "supported": settings is not None and (process_folder / PROCESS_STATE_FILE).exists(),
        "editablePools": can_edit_pools(parameters),
        "pools": [
            {"id": pool["id"], "name": pool["name"], "size": pool_size(pool)}
            for pool in parameters["resource_profiles"]
        ],
        "tasks": sorted(
            ({"id": node.id, "name": node.name} for node in bpmn_model.nodes if node.id in task_ids),
            key=lambda task: task["name"],
        ),
    }


@app.get("/run/{process_id}")
def run_details(process_id: str):
    settings = read_run_settings(process_folder_of(process_id))
    if settings is None:
        raise HTTPException(status_code=404, detail="Unknown process")
    return {
        "seed": settings["seed"],
        "files": settings.get("files"),
        "stateSeconds": settings.get("state_seconds"),
        "simulationSeconds": settings.get("simulation_seconds"),
    }


@app.get("/parameters/{process_id}")
def run_parameters(process_id: str):
    parameters_path = process_folder_of(process_id) / "json_parameters.json"
    if not parameters_path.exists():
        raise HTTPException(status_code=404, detail="Unknown process")
    return json.loads(parameters_path.read_text())


@app.get("/resources/{process_id}")
def run_resources(process_id: str, start: str, end: str):
    process_folder = process_folder_of(process_id)
    if not (process_folder / "short-term-simulation.csv").exists():
        raise HTTPException(status_code=404, detail="Unknown process")
    settings = read_run_settings(process_folder)
    simulator_time = pd.Timestamp(settings["start_time"]).utcoffset() if settings else pd.Timedelta(0)
    timezone = datetime.timezone(simulator_time)
    return resources_of(process_folder, pd.Timestamp(start).tz_convert(timezone), pd.Timestamp(end).tz_convert(timezone))


class ResumeRequest(BaseModel):
    process_id: str
    timestamp: str

@app.post("/resumption")
def resume_short_term_simulation(request: ResumeRequest, db: Session = Depends(get_db)):
    process_folder = process_folder_of(request.process_id)
    # Input params of the call
    resume_timestamp = pd.Timestamp(request.timestamp)
    # Path to files to store intermediate objects
    bpmn_model_path = process_folder / "bpmn_model.bpmn"
    short_term_simulated_log_path = process_folder / "short-term-simulation-processed.csv"
    complete_reachability_graph_path = process_folder / "complete_reachability_graph.tgf"
    reachability_graph_with_events_path = process_folder / "reachability_graph_with_events.tgf"
    n_gram_index_with_events_path = process_folder / "n_gram_index_with_events.map"

    # Compute initial frame
    frame = compute_bps_resumed_state(
        bpmn_model_path=bpmn_model_path,
        resume_timestamp=resume_timestamp,
        short_term_simulated_log_path=short_term_simulated_log_path,
        reachability_graph_path=reachability_graph_with_events_path,
        n_gram_index_path=n_gram_index_with_events_path,
        db=db,
        process_id=request.process_id,
    )

    return {
        "frames": frame,
    }

@app.delete("/delete/{process_id}")
def delete_ngrams(process_id: str, db: Session = Depends(get_db)):
    process_folder = process_folder_of(process_id)
    delete_ngrams_by_process_id(process_id, db)
    shutil.rmtree(process_folder, ignore_errors=True)
    return {"message": f"Process {process_id} deleted successfully"}


class PruneRequest(BaseModel):
    keep: list[str]
    min_age_hours: float = 48


@app.post("/prune")
def prune_processes(request: PruneRequest, db: Session = Depends(get_db)):
    keep = set(request.keep)
    cutoff = time.time() - request.min_age_hours * 3600
    removed = []
    if PROCESSES_DIR.is_dir():
        for folder in PROCESSES_DIR.iterdir():
            if folder.name in keep or not folder.is_dir() or folder.stat().st_mtime > cutoff:
                continue
            if PROCESS_ID_PATTERN.fullmatch(folder.name):
                delete_ngrams_by_process_id(folder.name, db)
            shutil.rmtree(folder, ignore_errors=True)
            removed.append(folder.name)
    return {"removed": removed}

def compute_ongoing_case_arrivals(
        ongoing_log_path: Path,
        ongoing_log_ids: EventLogIDs,
        ongoing_case_ids: list,
) -> dict:
    ongoing_case_ids = {str(case_id) for case_id in ongoing_case_ids}
    ongoing_event_log = read_csv_log(ongoing_log_path, ongoing_log_ids)
    case_ids = ongoing_event_log[ongoing_log_ids.case].astype(str)
    ongoing_event_log = ongoing_event_log[case_ids.isin(ongoing_case_ids)]
    time_columns = [
        column for column in [ongoing_log_ids.enabled_time, ongoing_log_ids.start_time]
        if column in ongoing_event_log.columns
    ]
    if ongoing_event_log.empty or not time_columns:
        return {}
    first_times = ongoing_event_log[time_columns].min(axis=1).groupby(case_ids[ongoing_event_log.index]).min()
    return {
        case_id: timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f%z")
        for case_id, timestamp in first_times.items()
        if pd.notna(timestamp)
    }

def compute_ongoing_task_times(
        simulated_log_path: Path,
        bpmn_model_path: Path,
        start_time: pd.Timestamp,
        ongoing_case_ids: list,
) -> dict:
    node_ids = {node.name: node.id for node in read_bpmn_model(bpmn_model_path).nodes}
    log = read_csv_log(simulated_log_path, sim_log_ids)
    case_ids = log[sim_log_ids.case].astype(str)
    in_flight = log[case_ids.isin({str(case_id) for case_id in ongoing_case_ids}) & (log[sim_log_ids.enabled_time] < start_time)]
    task_times = {}
    time_format = "%Y-%m-%dT%H:%M:%S.%f%z"
    for _, activity in in_flight.iterrows():
        node_id = node_ids.get(activity[sim_log_ids.activity])
        if node_id is None:
            continue
        started = activity[sim_log_ids.start_time]
        task_times.setdefault(str(activity[sim_log_ids.case]), {})[node_id] = {
            "enabled": activity[sim_log_ids.enabled_time].strftime(time_format),
            "started": started.strftime(time_format) if started < start_time else None,
        }
    return task_times


def post_process_simulated_log(
        ongoing_log_path: Path,
        ongoing_log_ids: EventLogIDs,
        start_time: pd.Timestamp,
        simulated_log_path: Path,
        post_processed_log_path: Path,
        bpmn_model_path: Path,
        ongoing_case_ids: list,
):
    # Read BPMN model
    bpmn_model = read_bpmn_model(bpmn_model_path)
    start_event = [node for node in bpmn_model.nodes if node.is_start_event()][0]  # Should be only one
    end_event = [node for node in bpmn_model.nodes if node.is_end_event()][0]  # Should be only one
    # Read simulated log
    simulated_event_log = read_csv_log(simulated_log_path, sim_log_ids)
    # Add activity instances finished prior to the [start_time] of the log, only for the cases ongoing at
    # [start_time]. New arrivals may reuse the ID of a case that already finished in the ongoing log, and
    # must not inherit its history.
    ongoing_case_ids = {str(case_id) for case_id in ongoing_case_ids}
    ongoing_event_log = read_csv_log(ongoing_log_path, ongoing_log_ids)
    previous_events = ongoing_event_log[
        (ongoing_event_log[ongoing_log_ids.case].astype(str).isin(ongoing_case_ids)) &
        (ongoing_event_log[ongoing_log_ids.end_time] < start_time)
        ]
    renamed_columns = {
        ongoing_log_ids.case: sim_log_ids.case,
        ongoing_log_ids.activity: sim_log_ids.activity,
        ongoing_log_ids.resource: sim_log_ids.resource,
        ongoing_log_ids.start_time: sim_log_ids.start_time,
        ongoing_log_ids.end_time: sim_log_ids.end_time,
        ongoing_log_ids.enabled_time: sim_log_ids.enabled_time,
    }
    previous_events = previous_events[
        [column for column in renamed_columns if column in previous_events.columns]
    ].rename(columns=renamed_columns)
    if sim_log_ids.enabled_time not in previous_events.columns:
        previous_events[sim_log_ids.enabled_time] = previous_events[sim_log_ids.start_time]
    simulated_event_log = pd.concat([simulated_event_log, previous_events], ignore_index=True)
    # Prosimos may already emit rows for the BPMN's start/end events under the same activity name.
    # Only synthesize them here if they are not already present, to avoid duplicate CASE_ARRIVAL/CASE_END.
    existing_activities = set(simulated_event_log[sim_log_ids.activity].dropna().unique())
    # Add start event
    if start_event.name not in existing_activities:
        idx_start_events = simulated_event_log.groupby(sim_log_ids.case)[sim_log_ids.start_time].idxmin()
        start_events = simulated_event_log.loc[idx_start_events].copy()
        start_events[sim_log_ids.enabled_time] = start_events[sim_log_ids.enabled_time] - pd.Timedelta(milliseconds=100)
        start_events[sim_log_ids.start_time] = start_events[sim_log_ids.enabled_time] - pd.Timedelta(milliseconds=100)
        start_events[sim_log_ids.end_time] = start_events[sim_log_ids.enabled_time]
        start_events[sim_log_ids.resource] = ""
        start_events[sim_log_ids.activity] = start_event.name
        simulated_event_log = pd.concat([start_events, simulated_event_log], ignore_index=True)
    # Add end event
    if end_event.name not in existing_activities:
        idx_end_events = simulated_event_log.groupby(sim_log_ids.case)[sim_log_ids.end_time].idxmax()
        end_events = simulated_event_log.loc[idx_end_events].copy()
        end_events[sim_log_ids.enabled_time] = end_events[sim_log_ids.end_time]
        end_events[sim_log_ids.start_time] = end_events[sim_log_ids.end_time] + pd.Timedelta(milliseconds=100)
        end_events[sim_log_ids.end_time] = end_events[sim_log_ids.end_time] + pd.Timedelta(milliseconds=100)
        end_events[sim_log_ids.resource] = ""
        end_events[sim_log_ids.activity] = end_event.name
        simulated_event_log = pd.concat([simulated_event_log, end_events], ignore_index=True)
    # Write extended event log to file
    simulated_event_log.to_csv(post_processed_log_path, date_format="%Y-%m-%dT%H:%M:%S.%f%z", index=False)
