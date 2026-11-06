import json
import logging
import multiprocessing
import re
import shutil
import threading
import time
from pathlib import Path
from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool
import pandas as pd
from ongoing_process_state.utils import read_bpmn_model
from pix_framework.io.event_log import read_csv_log, EventLogIDs
from pydantic import BaseModel

from src.compute_bps_state_and_run_simulation import compute_bps_state_and_run_simulation, compute_bps_resumed_state, \
    generate_events_with_token_movements
from src.compute_frontend_events_from_trace import sim_log_ids

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
runs: dict = {}
multiprocessing.set_start_method("forkserver", force=True)
multiprocessing.set_forkserver_preload(["pix_framework.enhancement.concurrency_oracle"])

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

def simulate(
        process_id: str,
        process_folder: Path,
        ongoing_log_path: Path,
        ongoing_log_ids: EventLogIDs,
        bpmn_model_path: Path,
        bpmn_parameters_path: Path,
        start_time: str,
        simulation_horizon: str,
):
    with simulation_lock:
        # Path to files to store intermediate objects
        short_term_simulated_log_path = process_folder / "short-term-simulation.csv"
        post_processed_log_path = process_folder / "short-term-simulation-processed.csv"
        reachability_graph_path = process_folder / "complete_reachability_graph.tgf"
        reachability_graph_with_events_path = process_folder / "reachability_graph_with_events.tgf"
        n_gram_index_with_events_path = process_folder / "n_gram_index_with_events.map"

        # Compute the initial frame and run short-term simulation
        frame, short_term_simulated_log_path = compute_bps_state_and_run_simulation(
            ongoing_log_path=ongoing_log_path,
            bpmn_model_path=bpmn_model_path,
            bpmn_parameters_path=bpmn_parameters_path,
            start_time=start_time,
            simulation_horizon=simulation_horizon,
            short_term_simulated_log_path=short_term_simulated_log_path,
            on_stage=lambda stage: set_stage(process_id, stage),
        )
        set_stage(process_id, "playback")

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
        for case in frame:
            case['arrival_time'] = arrival_times.get(str(case['case_id']))

        return {
            "frames": frame,
            "events": events,
        }


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
    ongoing_log_ids = EventLogIDs(
        case=column_mapping.get("case", "CaseId"),
        activity=column_mapping.get("activity", "Activity"),
        resource=column_mapping.get("resource", "Resource"),
        start_time=column_mapping.get("start", "StartTime"),
        end_time=column_mapping.get("end", "EndTime"),
        enabled_time=column_mapping.get("enablement", "enabled_time"),
    )

    bpmn_model_path = process_folder / "bpmn_model.bpmn"
    with open(bpmn_model_path, "wb") as f:
        f.write(await bpmn_model.read())

    bpmn_parameters_path = process_folder / "json_parameters.json"
    with open(bpmn_parameters_path, "wb") as f:
        f.write(await json_parameters.read())

    set_stage(process_id, "queued")
    try:
        result = await run_in_threadpool(
            simulate, process_id, process_folder, ongoing_log_path, ongoing_log_ids, bpmn_model_path,
            bpmn_parameters_path, start_time, simulation_horizon,
        )
    except Exception as error:
        logger.exception("Simulation %s failed", process_id)
        stage = finish_run(process_id, error=f"{type(error).__name__}: {error}")
        raise HTTPException(status_code=422, detail={"stage": stage, "message": f"{type(error).__name__}: {error}"})
    finish_run(process_id)
    return result

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
