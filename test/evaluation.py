import sys
import os
import json
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.runner import run_process_state_and_simulation
from test import helper

# Import functions from the log-distance-measures library
from log_distance_measures.config import (
    DEFAULT_CSV_IDS,
    AbsoluteTimestampType,
    discretize_to_hour,
    EventLogIDs
)
from log_distance_measures.control_flow_log_distance import control_flow_log_distance
from log_distance_measures.n_gram_distribution import n_gram_distribution_distance
from log_distance_measures.absolute_event_distribution import absolute_event_distribution_distance
from log_distance_measures.case_arrival_distribution import case_arrival_distribution_distance
from log_distance_measures.cycle_time_distribution import cycle_time_distribution_distance

def evaluate_partial_state_simulation(
    event_log: str,
    bpmn_model: str,
    bpmn_parameters: str,
    start_time: str,
    simulation_horizon: str,
    column_mapping: str,   # JSON string, e.g. '{"case_id":"CaseId","Activity":"Activity", ...}'
    total_cases: int = 1000,
    sim_stats_csv: str = "sim_stats.csv",
    sim_log_csv: str = "sim_log.csv",
    rename_alog: dict = None,        # how to rename columns in ALog to standard columns
    rename_glog: dict = None,        # how to rename columns in GLog (if needed)
    required_columns: list = None,   # columns needed for metrics
    simulate: bool = True,
    verbose: bool = True
) -> dict:
    """
    Runs partial-state simulation and evaluates metrics comparing ALog vs GLog.

    :param event_log: Path to actual log CSV (ALog).
    :param bpmn_model: Path to BPMN model file.
    :param bpmn_parameters: Path to BPMN parameters JSON.
    :param start_time: Simulation start time (ISO-8601 string).
    :param simulation_horizon: Horizon end time (ISO-8601 string).
    :param column_mapping: JSON string with col mapping for simulation only,
                           e.g. '{"case_id":"CaseId","Activity":"Activity",...}'
    :param total_cases: Number of cases to simulate (int).
    :param sim_stats_csv: Where to write simulation stats (optional).
    :param sim_log_csv: Where to write generated log (CSV).
    :param rename_alog: Dict to rename ALog columns to standard names, e.g.:
                        {
                           "case_id": "case_id",
                           "Activity": "activity",
                           ...
                        }
    :param rename_glog: Dict to rename GLog columns to standard names (if needed).
    :param required_columns: List of columns needed for metrics,
                            e.g. ["case_id","activity","start_time","end_time","resource"]
    :param simulate: Whether to actually run simulation. If False, just read logs and compute metrics.
    :param verbose: Print progress messages if True.

    :return: A dictionary with:
        {
            "ALog_stats": {...},
            "GLog_stats": {...},
            "metrics": {...}
        }
    """
    if required_columns is None:
        required_columns = ["case_id", "activity", "start_time", "end_time", "resource"]

    # 1) Run the partial-state simulation (if requested)
    if simulate:
        if verbose:
            print("=== Step 1: Running partial-state simulation ===")
        try:
            run_process_state_and_simulation(
                event_log=event_log,
                bpmn_model=bpmn_model,
                bpmn_parameters=bpmn_parameters,
                start_time=start_time,
                column_mapping=column_mapping,
                simulate=True,
                simulation_horizon=simulation_horizon,
                total_cases=total_cases,
                sim_stats_csv=sim_stats_csv,
                sim_log_csv=sim_log_csv,
            )
        except FileNotFoundError as e:
            raise RuntimeError(f"Simulation failed - file not found: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Simulation failed with unexpected error: {e}") from e
    else:
        if verbose:
            print("Skipping simulation (simulate=False).")

    # 2) Read ALog & GLog CSV files
    if verbose:
        print("=== Step 2: Reading ALog & GLog with no special mapping ===")
    if not os.path.isfile(event_log):
        raise FileNotFoundError(f"ALog file not found: {event_log}")
    if not os.path.isfile(sim_log_csv):
        raise FileNotFoundError(f"GLog file not found: {sim_log_csv}")

    alog_df = pd.read_csv(event_log)
    glog_df = pd.read_csv(sim_log_csv)

    # 2b) Rename columns to standard names as expected by log-distance-measures
    if rename_alog:
        alog_df.rename(columns=rename_alog, inplace=True)
    if rename_glog:
        glog_df.rename(columns=rename_glog, inplace=True)

    # Convert time columns to datetime
    time_cols = ["enable_time", "start_time", "end_time"]
    for col in time_cols:
        if col in alog_df.columns:
            alog_df[col] = pd.to_datetime(alog_df[col], utc=True, errors="coerce")
        if col in glog_df.columns:
            glog_df[col] = pd.to_datetime(glog_df[col], utc=True, errors="coerce")

    # 3) Preprocess logs (remove events/cases outside the simulation window)
    if verbose:
        print("=== Step 3: Preprocessing logs ===")
    start_dt = pd.to_datetime(start_time, utc=True)
    horizon_dt = pd.to_datetime(simulation_horizon, utc=True)
    A_clean = helper.preprocess_alog(alog_df, start_time=start_dt, horizon=horizon_dt)
    G_clean = helper.preprocess_glog(glog_df, horizon=horizon_dt)

    # 4) Keep only the required columns
    A_clean = A_clean[required_columns]
    G_clean = G_clean[required_columns]

    # 5) Compute basic log stats (number of cases, events, etc.)
    if verbose:
        print("=== Step 4: Computing basic stats ===")
    statsA = helper.basic_log_stats(A_clean)
    statsG = helper.basic_log_stats(G_clean)

    custom_ids = EventLogIDs(
        case="case_id",
        activity="activity",
        start_time="start_time",
        end_time="end_time",
        resource="resource"
    )

    # 6) Compute distances using log-distance-measures
    if verbose:
        print("=== Step 5: Computing log-distance-measures distances ===")
    distances = {}
    try:
        # Control-Flow Log Distance
        distances["control_flow_log_distance"] = control_flow_log_distance(
            A_clean, custom_ids,
            G_clean, custom_ids
        )

        # N-Gram Distribution Distance
        distances["n_gram_distribution_distance"] = n_gram_distribution_distance(
            A_clean, custom_ids,
            G_clean, custom_ids,
            n=3
        )

        # Absolute Event Distribution Distance
        distances["absolute_event_distribution_distance"] = absolute_event_distribution_distance(
            A_clean, custom_ids,
            G_clean, custom_ids,
            discretize_type=AbsoluteTimestampType.START,
            discretize_event=discretize_to_hour
        )

        # Case Arrival Distribution Distance
        distances["case_arrival_distribution_distance"] = case_arrival_distribution_distance(
            A_clean, custom_ids,
            G_clean, custom_ids,
            discretize_event=discretize_to_hour
        )

        # Cycle Time Distribution Distance
        distances["cycle_time_distribution_distance"] = cycle_time_distribution_distance(
            A_clean, custom_ids,
            G_clean, custom_ids,
            bin_size=pd.Timedelta(hours=1)
        )
    except Exception as e:
        raise RuntimeError(f"Error computing log distance measures: {e}") from e

    # 7) Return the stats and computed distances as a result dictionary
    result_dict = {
        "ALog_stats": statsA,
        "GLog_stats": statsG,
        "distances": distances,
    }

    if verbose:
        print("Evaluation result:", json.dumps(result_dict, indent=4))
    return result_dict
