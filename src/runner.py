# runner.py

import json
import datetime

import pandas as pd

from src.input_handler import InputHandler
from src.event_log_processor import EventLogProcessor
from src.bpmn_handler import BPMNHandler
from src.state_computer import StateComputer
from src.process_state_prosimos_run import run_short_term_simulation


def parse_datetime(dt_str):
    """Helper to parse an ISO date/time, removing 'Z' if present."""
    if not dt_str:
        return None
    return datetime.datetime.fromisoformat(dt_str.replace("Z", "+00:00"))


def run_process_state_and_simulation(
    event_log,
    bpmn_model,
    bpmn_parameters,
    start_time=None,
    column_mapping=None,
    simulate=False,
    simulation_horizon=None,
    total_cases=20,
    sim_stats_csv='simulation_stats.csv',
    sim_log_csv='simulation_log.csv',
):
    """
    1) Compute the process state from the event log + BPMN.
    2) Optionally run a Prosimos short-term simulation using the resulting partial-state.

    :param event_log: Path to event log CSV
    :param bpmn_model: Path to BPMN model file
    :param bpmn_parameters: Path to JSON simulation parameters
    :param start_time: (str) optional starting datetime
    :param column_mapping: (str) JSON string for column mapping
    :param simulate: (bool) if True, run short-term simulation after computing process state
    :param simulation_horizon: (str) time boundary for short-term
    :param total_cases: (int) how many cases if partial-state/horizon logic doesn't override
    :param sim_stats_csv: path to final stats CSV
    :param sim_log_csv: path to final event log CSV
    """
    print("=== RUNNER: Step A: Building a fake args object for InputHandler ===")

    # 1) Build a minimal "args" object for InputHandler
    class FakeArgs:
        pass

    args = FakeArgs()
    args.event_log = event_log
    args.bpmn_model = bpmn_model
    args.bpmn_parameters = bpmn_parameters
    args.start_time = start_time
    args.column_mapping = column_mapping

    print("=== RUNNER: Step B: Using InputHandler to read event log & BPMN ===")
    input_handler = InputHandler(args)
    event_log_ids = input_handler.event_log_ids
    event_log_df = input_handler.event_log_df
    bpmn_model_obj = input_handler.read_bpmn_model()
    bpmn_params = input_handler.parse_bpmn_parameters()

    print("=== RUNNER: Step C: Process event log ===")
    event_log_processor = EventLogProcessor(event_log_df, start_time, event_log_ids)
    processed_event_log = event_log_processor.process()
    concurrency_oracle = event_log_processor.concurrency_oracle

    print("=== RUNNER: Step D: Build N-Gram Index ===")
    bpmn_handler = BPMNHandler(bpmn_model_obj, bpmn_params, input_handler.bpmn_model_path)
    n_gram_index = bpmn_handler.build_n_gram_index(n_gram_size_limit=20)
    reachability_graph = bpmn_handler.get_reachability_graph()

    print("=== RUNNER: Step E: Compute process state ===")
    state_computer = StateComputer(
        n_gram_index, reachability_graph, processed_event_log,
        bpmn_handler, concurrency_oracle, event_log_ids
    )

    case_states = state_computer.compute_case_states()

    # ------------------------------------------------------------------
    # last_case_arrival  – earliest start of latest-arriving case
    # ------------------------------------------------------------------
    first_start_per_case = (
        processed_event_log
        .groupby(event_log_ids.case)[event_log_ids.start_time]
        .min()
    )
    last_case_arrival_dt = (
        first_start_per_case.max() if not first_start_per_case.empty else None
    )

    # 2) Prepare partial-state as a dict
    output_data = {
        "last_case_arrival": last_case_arrival_dt,   
        "cases": {}
    }

    for case_id, case_info in case_states.items():
        output_data['cases'][str(case_id)] = {
            "control_flow_state": {
                "flows": list(case_info["control_flow_state"]["flows"]),
                "activities": list(case_info["control_flow_state"]["activities"])
            },
            "ongoing_activities": case_info["ongoing_activities"],
            "enabled_activities": case_info["enabled_activities"],
            "enabled_gateways": case_info["enabled_gateways"],
            "enabled_events": case_info["enabled_events"]
        }

    # 3) Write partial state to output.json
    print("=== RUNNER: Step F: Writing partial-state to output.json ===")
    with open('output.json', 'w') as f:
        json.dump(output_data, f, default=str, indent=4)
    # print("Process state in 'output.json'.")

    # 4) If simulate => call short-term approach
    if simulate:
        print("=== RUNNER: Step G: Doing short-term simulation ===")
        sim_horizon_dt = parse_datetime(simulation_horizon)
        partial_state = output_data

        sim_time = run_short_term_simulation(
            start_date=start_time,
            total_cases=total_cases,
            bpmn_model=bpmn_model,
            json_sim_params=bpmn_parameters,
            out_stats_csv_path=sim_stats_csv,
            out_log_csv_path=sim_log_csv,
            process_state=partial_state,
            simulation_horizon=sim_horizon_dt
        )
        print(f"Simulation done. Duration: {sim_time} seconds. Log in '{sim_log_csv}'.")

    print("=== RUNNER: Done. ===")