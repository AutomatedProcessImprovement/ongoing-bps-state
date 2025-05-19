# runner.py

import datetime
from pathlib import Path

import pandas as pd
from ongoing_process_state.n_gram_index import NGramIndex
from ongoing_process_state.utils import read_bpmn_model
from pix_framework.io.event_log import read_csv_log
from typing import Tuple, List

from src.bpmn_handler import compute_extended_bpmn_model
from src.compute_frontend_events_from_trace import read_reachability_graph, compute_complete_reachability_graph, \
    sim_log_ids, compute_token_movements
from src.runner import run_process_state_and_simulation
from src.state_computer import _add_to_sorted_events

from sqlalchemy.orm import Session
from db.ngram_repository import save_n_gram_index_to_db, load_n_gram_index_from_db

def parse_datetime(dt_str):
    """Helper to parse an ISO date/time, removing 'Z' if present."""
    if not dt_str:
        return None
    return datetime.datetime.fromisoformat(dt_str.replace("Z", "+00:00"))


def compute_bps_state_and_run_simulation(
        ongoing_log_path: Path,
        bpmn_model_path: Path,
        bpmn_parameters_path: Path,
        start_time: str,
        simulation_horizon: str = None,
        short_term_simulated_log_path: Path = None,
        column_mapping: str = None,
) -> Tuple[List[dict], Path]:
    # Set column mapping to default if None
    if column_mapping is None:
        column_mapping = ("{\"CaseId\": \"case_id\", \"Activity\": \"activity\", \"enabled_time\": \"enable_time\", "
                          "\"StartTime\": \"start_time\", \"EndTime\": \"end_time\", \"Resource\": \"resource\"}")
    # Set default simulated log path if None
    if short_term_simulated_log_path is None:
        short_term_simulated_log_path = Path('./short_term_simulated_log.csv')

    # Call to runner to compute state and run simulation
    bps_state, simulated_log_path = run_process_state_and_simulation(
        event_log=ongoing_log_path,
        bpmn_model=bpmn_model_path,
        bpmn_parameters=bpmn_parameters_path,
        start_time=start_time,
        column_mapping=column_mapping,
        simulate=True,
        simulation_horizon=simulation_horizon,
        total_cases=20,  # default, irrelevant
        sim_stats_csv=None,
        sim_log_csv=short_term_simulated_log_path,
        produce_events_when_simulating=False
    )

    # Produce frame for front-end
    frame = []
    for case_id, case_info in bps_state.items():
        active_elements = dict()
        i = 0
        for active_flow in sorted(
                case_info["control_flow_state"]["flows"] + case_info["control_flow_state"]["activities"]
        ):
            active_elements[f"token_{i}"] = active_flow
            i += 1
        frame += [{'case_id': case_id, 'active_elements': active_elements}]

    return frame, simulated_log_path


def generate_events_with_token_movements(
        bpmn_model_path: Path,
        start_timestamp: pd.Timestamp,
        short_term_simulation_path: Path,
        reachability_graph_path: Path,
        frame: list = None
) -> List[dict]:
    # Reset frame as empty list if None
    if frame is None:
        frame = []
    # Read event log and filter out activity instances previous to cut-point
    event_log = read_csv_log(short_term_simulation_path, sim_log_ids)
    event_log = event_log[event_log[sim_log_ids.end_time] > start_timestamp]
    # Compute current state of tokens (ongoing cases)
    ongoing_token_status = {case['case_id']: case['active_elements'] for case in frame}
    # Read model
    bpmn_model = read_bpmn_model(bpmn_model_path)
    # Compute reachability graph for token movements
    if reachability_graph_path.exists():
        # Graph already precomputed, retrieve from file/DB
        reachability_graph = read_reachability_graph(reachability_graph_path, names_are_tuples=True)
    else:
        # Graph not precomputed, compute and store in file/DB
        reachability_graph = compute_complete_reachability_graph(bpmn_model)
        with open(reachability_graph_path, "w") as graph_file:
            graph_file.write(reachability_graph.to_tgf_format())
    # Go case by case generating the token movements
    events = []
    for case_id, activity_instances in event_log.groupby(sim_log_ids.case):
        # Nullify enabled/start events previous to cut-point
        activity_instances.loc[
            activity_instances[sim_log_ids.enabled_time] < start_timestamp,
            sim_log_ids.enabled_time
        ] = pd.NaT
        activity_instances.loc[
            activity_instances[sim_log_ids.start_time] < start_timestamp,
            sim_log_ids.start_time
        ] = pd.NaT
        # Compute movements
        events += compute_token_movements(
            model=bpmn_model,
            reach_graph=reachability_graph,
            trace=activity_instances,
            ongoing_token_status=ongoing_token_status.get(case_id, None)
        )
    # Return all events
    return events


def compute_bps_resumed_state(
        bpmn_model_path: Path,
        resume_timestamp: pd.Timestamp,
        short_term_simulated_log_path: Path,
        reachability_graph_path: Path,
        n_gram_index_path: Path,
        db: Session,
        process_id: str,
) -> List[dict]:
    # Read model
    bpmn_model = read_bpmn_model(bpmn_model_path)
    start_event = [node for node in bpmn_model.nodes if node.is_start_event()][0]  # Should be only one

    # If reachability graph considering events is computed, retrieve, otherwise compute and store
    if reachability_graph_path.exists():
        reachability_graph = read_reachability_graph(reachability_graph_path)
    else:
        extended_bpmn_model = compute_extended_bpmn_model(bpmn_model, treat_event_as_task=False)
        reachability_graph = extended_bpmn_model.get_reachability_graph(treat_event_as_task=False)
        with open(reachability_graph_path, "w") as graph_file:
            graph_file.write(reachability_graph.to_tgf_format())

    # Compute n-gram index considering events if it doesn't exist
    try:
        n_gram_index = load_n_gram_index_from_db(process_id=process_id, reachability_graph=reachability_graph, db=db)
    except Exception as e:
        # fallback to rebuild if not in DB
        print(f"Could not load n-gram index from DB: {e}")
        n_gram_index = NGramIndex(graph=reachability_graph, n_gram_size_limit=20)
        n_gram_index.build()
        save_n_gram_index_to_db(n_gram_index, process_id=process_id, db=db)

    # Read short-term simulated event log
    simulated_event_log = read_csv_log(short_term_simulated_log_path, sim_log_ids)
    # Retrieve ongoing cases
    ongoing_cases = simulated_event_log.groupby(sim_log_ids.case).filter(
        lambda activity_instances:
        (activity_instances[sim_log_ids.start_time].min() <= resume_timestamp or
         start_event.name not in activity_instances[sim_log_ids.activity].unique()) and
        activity_instances[sim_log_ids.end_time].max() > resume_timestamp
    )
    ongoing_cases = ongoing_cases[ongoing_cases[sim_log_ids.start_time] <= resume_timestamp]
    ongoing_cases.loc[
        ongoing_cases[sim_log_ids.end_time] > resume_timestamp,
        sim_log_ids.end_time
    ] = pd.NaT
    # Compute token state for each case
    frame = []
    for case_id, activity_instances in ongoing_cases.groupby(sim_log_ids.case):
        # Compute state
        sorted_events = list()
        activity_instances.apply(
            lambda activity_instance:
            _add_to_sorted_events(
                sorted_events=sorted_events,
                activity_label=activity_instance[sim_log_ids.activity],
                start_time=activity_instance[sim_log_ids.start_time],
                end_time=activity_instance[sim_log_ids.end_time],
            ), axis=1
        )
        n_gram = [event['label'] for event in sorted_events]
        ongoing_marking = n_gram_index.get_best_marking_state_for(n_gram)
        # Store frame for this case
        i = 0
        active_elements = dict()
        for active_element in sorted(ongoing_marking):
            active_elements[f"token_{i}"] = active_element
            i += 1
        frame += [{'case_id': case_id, 'active_elements': active_elements}]
    # Return frame with tokens of ongoing cases
    return frame
