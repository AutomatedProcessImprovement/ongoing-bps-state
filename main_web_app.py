import json
from pathlib import Path
from fastapi import FastAPI, Form, UploadFile, File
import pandas as pd
from ongoing_process_state.utils import read_bpmn_model
from pix_framework.io.event_log import read_csv_log, EventLogIDs
from pydantic import BaseModel

from src.compute_bps_state_and_run_simulation import compute_bps_state_and_run_simulation, compute_bps_resumed_state, \
    generate_events_with_token_movements
from src.compute_frontend_events_from_trace import sim_log_ids

app = FastAPI()

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
    process_folder = Path(f"./processes/{process_id}/")
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
    )

    with open(process_folder / "initial_frame.json", "w") as frame_file:
        json.dump(frame, frame_file, indent=4)

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
        bpmn_model_path=bpmn_model_path
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
    with open(process_folder / "initial_events.json", "w") as events_file:
        json.dump(events, events_file, indent=4)

    return {
        "frames": frame,
        "events": events,
    }

class ResumeRequest(BaseModel):
    process_id: str
    timestamp: str

@app.post("/resumption")
def resume_short_term_simulation(request: ResumeRequest):
    process_folder = Path(f"./processes/{request.process_id}/")
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
    )
    # Repair the tokens to match the ongoing precomputed events
    # TODO --start comment--
    #  Taleh: delete this once you retrieve the events from DB in each loop iteration
    with open(process_folder / "initial_events.json", "r") as events_file:
        events = json.load(events_file)
    # TODO --end comment--
    for element in frame:
        case_id = element['case_id']
        active_tokens = element['active_elements']
        # TODO --start comment--
        #  Taleh: you retrieve these events from DB
        previous_events = [
            event
            for event in events
            if event['case_id'] == case_id and pd.Timestamp(event['timestamp']) < resume_timestamp
        ]
        # TODO --end comment --
        if len(previous_events) > 0:
            # Only repair if there are previous events
            repaired_active_tokens = dict()
            for active_token in active_tokens:
                real_token_id, real_flow = [
                    (token_id, previous_event["paths"][token_id][-1])  # retrieve tokenID and last_flow
                    for previous_event in previous_events  # of the last previous event
                    for token_id in previous_event["paths"]  # with a token
                    if active_tokens[active_token] in previous_event["paths"][token_id]  # that passed through this path
                ][-1]
                # Reset new tokenID to new flow_name
                repaired_active_tokens[real_token_id] = real_flow
            element['active_elements'] = repaired_active_tokens

    with open(process_folder / "resume_frame.json", "w") as frame_file:
        json.dump(frame, frame_file, indent=4)

    return {
        "frames": frame,
        "events": events,
    }

def post_process_simulated_log(
        ongoing_log_path: Path,
        ongoing_log_ids: EventLogIDs,
        start_time: pd.Timestamp,
        simulated_log_path: Path,
        post_processed_log_path: Path,
        bpmn_model_path: Path,
):
    # Read BPMN model
    bpmn_model = read_bpmn_model(bpmn_model_path)
    start_event = [node for node in bpmn_model.nodes if node.is_start_event()][0]  # Should be only one
    end_event = [node for node in bpmn_model.nodes if node.is_end_event()][0]  # Should be only one
    # Read simulated log
    simulated_event_log = read_csv_log(simulated_log_path, sim_log_ids)
    simulated_case_ids = simulated_event_log[sim_log_ids.case].unique()
    # Add activity instances finished prior to the [start_time] of the log
    ongoing_event_log = read_csv_log(ongoing_log_path, ongoing_log_ids)
    previous_events = ongoing_event_log[
        (ongoing_event_log[ongoing_log_ids.case].isin(simulated_case_ids)) &
        (ongoing_event_log[ongoing_log_ids.end_time] < start_time)
        ].copy()
    previous_events.rename(columns={
        ongoing_log_ids.case: sim_log_ids.case,
        ongoing_log_ids.activity: sim_log_ids.activity,
        ongoing_log_ids.resource: sim_log_ids.resource,
        ongoing_log_ids.start_time: sim_log_ids.start_time,
        ongoing_log_ids.end_time: sim_log_ids.end_time,
        ongoing_log_ids.enabled_time: sim_log_ids.enabled_time,
    }, inplace=True)
    simulated_event_log = pd.concat([simulated_event_log, previous_events], ignore_index=True)
    # Add start event
    idx_start_events = simulated_event_log.groupby(sim_log_ids.case)[sim_log_ids.start_time].idxmin()
    start_events = simulated_event_log.loc[idx_start_events].copy()
    start_events[sim_log_ids.enabled_time] = start_events[sim_log_ids.enabled_time] - pd.Timedelta(milliseconds=100)
    start_events[sim_log_ids.start_time] = start_events[sim_log_ids.enabled_time] - pd.Timedelta(milliseconds=100)
    start_events[sim_log_ids.end_time] = start_events[sim_log_ids.enabled_time]
    start_events[sim_log_ids.resource] = ""
    start_events[sim_log_ids.activity] = start_event.name
    simulated_event_log = pd.concat([start_events, simulated_event_log], ignore_index=True)
    # Add end event
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
