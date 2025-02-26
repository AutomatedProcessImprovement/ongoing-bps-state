from pathlib import Path

import pandas as pd

from src.compute_bps_state_and_run_simulation import compute_bps_state_and_run_simulation, compute_bps_resumed_state, \
    generate_events_with_token_movements


def start_short_term_simulation(base_folder: Path):
    # Input params of the call
    ongoing_log_path = base_folder / "ongoing_event_log.csv"
    bpmn_model_path = base_folder / "bpmn_model.bpmn"
    bpmn_parameters_path = base_folder / "json_parameters.json"
    short_term_simulated_log_path = base_folder / "short-term-simulation.csv"
    start_time = "2025-03-07T11:00:00.000+02:00"
    simulation_horizon = "2025-04-07T11:00:00.000+02:00"
    # Path to files to store intermediate objects
    reachability_graph_path = base_folder / "complete_reachability_graph.tgf"

    # Compute the initial frame and run short-term simulation
    frame, short_term_simulated_log_path = compute_bps_state_and_run_simulation(
        ongoing_log_path=ongoing_log_path,
        bpmn_model_path=bpmn_model_path,
        bpmn_parameters_path=bpmn_parameters_path,
        start_time=start_time,
        simulation_horizon=simulation_horizon,
        short_term_simulated_log_path=short_term_simulated_log_path,
    )

    # Compute token events with token movements for front-end
    events = generate_events_with_token_movements(
        bpmn_model_path=bpmn_model_path,
        start_timestamp=pd.Timestamp(start_time),
        short_term_simulation_path=short_term_simulated_log_path,
        reachability_graph_path=reachability_graph_path,
        frame=frame,
    )

    # TODO - this information goes to front-end
    print(frame)
    print(events)


def resume_short_term_simulation(base_folder: Path):
    # Input params of the call
    bpmn_model_path = base_folder / "bpmn_model.bpmn"
    short_term_simulated_log_path = base_folder / "short-term-simulation.csv"
    resume_timestamp = pd.Timestamp("2025-03-25T11:00:00.000+02:00")
    # Path to files to store intermediate objects
    complete_reachability_graph_path = base_folder / "complete_reachability_graph.tgf"
    reachability_graph_with_events_path = base_folder / "reachability_graph_with_events.tgf"
    n_gram_index_with_events_path = base_folder / "n_gram_index_with_events.map"

    # Compute initial frame
    frame = compute_bps_resumed_state(
        bpmn_model_path=bpmn_model_path,
        resume_timestamp=resume_timestamp,
        short_term_simulated_log_path=short_term_simulated_log_path,
        reachability_graph_path=reachability_graph_with_events_path,
        n_gram_index_path=n_gram_index_with_events_path,
    )

    # Compute token events with token movements for front-end
    events = generate_events_with_token_movements(
        bpmn_model_path=bpmn_model_path,
        start_timestamp=resume_timestamp,
        short_term_simulation_path=short_term_simulated_log_path,
        reachability_graph_path=complete_reachability_graph_path,
        frame=frame,
    )

    # TODO - this information goes to front-end
    print(frame)
    print(events)


if __name__ == "__main__":
    # TODO - For each job, I would create a unique ID, a folder with that ID, and all paths
    #  you generate are inside that folder, in this way even if you have multiple jobs they
    #  won't override same files.
    folder_this_process = Path("./process-1/")
    # User launches the app from start
    # start_short_term_simulation(folder_this_process)
    # User clicks on a timeline point
    resume_short_term_simulation(folder_this_process)
