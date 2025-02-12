# ==============================
# test/evaluation.py
# ==============================

import sys
import os
import json
import math
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.runner import run_process_state_and_simulation
from src.process_state_prosimos_run import run_basic_simulation
from test import helper

# Log-distance-measures imports
from log_distance_measures.config import (
    AbsoluteTimestampType,
    discretize_to_hour,
    EventLogIDs
)
from log_distance_measures.control_flow_log_distance import control_flow_log_distance
from log_distance_measures.n_gram_distribution import n_gram_distribution_distance
from log_distance_measures.absolute_event_distribution import absolute_event_distribution_distance
from log_distance_measures.case_arrival_distribution import case_arrival_distribution_distance
from log_distance_measures.cycle_time_distribution import cycle_time_distribution_distance
from log_distance_measures.circadian_workforce_distribution import circadian_workforce_distribution_distance
from log_distance_measures.relative_event_distribution import relative_event_distribution_distance


def discard_partial_cases(df: pd.DataFrame, start_dt: pd.Timestamp, horizon_dt: pd.Timestamp) -> pd.DataFrame:
    """
    Discard entire cases if ANY event is partially out of [start_dt, horizon_dt].
    That is, keep a case only if all events lie fully within the time window.
    We'll assume 'start_time' and 'end_time' exist in df.
    """
    if df.empty:
        return df

    keep_case_ids = []
    grouped = df.groupby("case_id", as_index=False)
    for cid, group in grouped:
        min_start = group["start_time"].min()
        max_end   = group["end_time"].max()
        # Condition: entire case is in [start_dt, horizon_dt]
        # i.e. (min_start >= start_dt) AND (max_end <= horizon_dt)
        if (min_start >= start_dt) and (max_end <= horizon_dt):
            keep_case_ids.append(cid)

    return df[df["case_id"].isin(keep_case_ids)].copy()


###########################################################################
# PROCESS-STATE (PARTIAL-STATE) APPROACH
###########################################################################
def evaluate_partial_state_simulation(
    event_log: str,
    bpmn_model: str,
    bpmn_parameters: str,
    start_time: str,           # Evaluation start (formerly SIMULATION_CUT_DATE)
    evaluation_end: str,       # Evaluation end time (new parameter)
    simulation_horizon: str,   # Simulation horizon (may be longer than evaluation_end)
    column_mapping: str,       # JSON string for renaming columns
    total_cases: int = 1000,
    sim_stats_csv: str = "sim_stats.csv",
    sim_log_csv: str = "sim_log.csv",
    rename_alog: dict = None,
    rename_glog: dict = None,
    required_columns: list = None,
    simulate: bool = True,
    verbose: bool = True
) -> dict:
    """
    Runs the process-state simulation and then evaluates the distances between the reference event log (ALog)
    and the generated simulation log (GLog) using separate handling for cases and events:
      - Case metrics: keep entire cases if any event falls in [start_time, evaluation_end].
      - Event metrics: trim events so that events starting before start_time are set to start_time,
        and events ending after evaluation_end are set to evaluation_end.
    Saves both logs for case and event metrics.
    """
    if required_columns is None:
        required_columns = ["case_id", "activity", "start_time", "end_time", "resource"]

    # --- Step 1: Run simulation (with up to 3 retries) ---
    if simulate:
        if verbose:
            print("=== [Process-State] Step 1: Running partial-state simulation ===")
        max_attempts = 3
        attempt = 1
        while attempt <= max_attempts:
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
                break
            except Exception as e:
                if attempt == max_attempts:
                    raise RuntimeError(f"Process-state simulation failed after {max_attempts} attempts: {e}") from e
                else:
                    print(f"[Process-State] Attempt {attempt} failed: {e}. Retrying...")
                    attempt += 1
    else:
        if verbose:
            print("Skipping simulation (simulate=False).")

    # --- Step 2: Load ALog and GLog ---
    if verbose:
        print("=== [Process-State] Step 2: Reading ALog & GLog ===")
    if not os.path.isfile(event_log):
        raise FileNotFoundError(f"ALog file not found: {event_log}")
    if not os.path.isfile(sim_log_csv):
        raise FileNotFoundError(f"GLog file not found: {sim_log_csv}")

    alog_df = pd.read_csv(event_log)
    glog_df = pd.read_csv(sim_log_csv)

    if rename_alog:
        alog_df.rename(columns=rename_alog, inplace=True)
    if rename_glog:
        glog_df.rename(columns=rename_glog, inplace=True)

    # Convert time columns (do not filter out events here)
    time_cols = ["enable_time", "start_time", "end_time"]
    for col in time_cols:
        if col in alog_df.columns:
            alog_df[col] = pd.to_datetime(alog_df[col], utc=True, errors="coerce")
        if col in glog_df.columns:
            glog_df[col] = pd.to_datetime(glog_df[col], utc=True, errors="coerce")

    # --- Step 3: Define the evaluation window ---
    eval_start = pd.to_datetime(start_time, utc=True)
    eval_end_dt = pd.to_datetime(evaluation_end, utc=True)

    # --- Step 4: Preprocess logs without discarding events ---
    # Note: We call preprocess_alog and preprocess_glog WITHOUT passing start_time/horizon parameters,
    # so that we keep the full event data for each case.
    A_all = helper.preprocess_alog(alog_df)
    G_all = helper.preprocess_glog(glog_df)

    # --- Step 5: Create two versions for evaluation ---
    # (a) For case metrics: keep full events for cases that have any event in [eval_start, eval_end]
    A_case = helper.filter_cases_by_eval_window(A_all, eval_start, eval_end_dt)
    G_case = helper.filter_cases_by_eval_window(G_all, eval_start, eval_end_dt)
    # (b) For event metrics: take the same cases and trim individual events to the evaluation window.
    A_event = helper.trim_events_to_eval_window(A_all, eval_start, eval_end_dt)
    G_event = helper.trim_events_to_eval_window(G_all, eval_start, eval_end_dt)

    # --- Step 6: Save both logs for debugging ---
    if verbose:
        A_case[required_columns].to_csv("samples/output/PS_A_case.csv", index=False)
        A_event[required_columns].to_csv("samples/output/PS_A_event.csv", index=False)
        G_case[required_columns].to_csv("samples/output/PS_G_case.csv", index=False)
        G_event[required_columns].to_csv("samples/output/PS_G_event.csv", index=False)
        print("Saved PS A_case, A_event, G_case, and G_event CSV files.")

    # --- Step 7: Compute basic stats on the event metrics logs ---
    statsA = helper.basic_log_stats(A_event)
    statsG = helper.basic_log_stats(G_event)

    # --- Step 8: Compute log distances using the event metrics logs ---
    custom_ids = EventLogIDs(
        case="case_id",
        activity="activity",
        start_time="start_time",
        end_time="end_time",
        resource="resource"
    )
    if verbose:
        print("=== [Process-State] Step 8: Computing distances using event metrics logs ===")
    distances = {}
    try:
        distances["control_flow_log_distance"] = control_flow_log_distance(A_event, custom_ids, G_event, custom_ids)
        distances["n_gram_distribution_distance"] = n_gram_distribution_distance(A_event, custom_ids, G_event, custom_ids, n=3)
        distances["absolute_event_distribution_distance"] = absolute_event_distribution_distance(
            A_event, custom_ids, G_event, custom_ids,
            discretize_type=AbsoluteTimestampType.START,
            discretize_event=discretize_to_hour
        )
        distances["case_arrival_distribution_distance"] = case_arrival_distribution_distance(
            A_case, custom_ids, G_case, custom_ids,
            discretize_event=discretize_to_hour
        )
        distances["cycle_time_distribution_distance"] = cycle_time_distribution_distance(
            A_case, custom_ids, G_case, custom_ids,
            bin_size=pd.Timedelta(hours=1)
        )
        distances["circadian_workforce_distribution_distance"] = circadian_workforce_distribution_distance(
            A_case, custom_ids, G_case, custom_ids
        )
        distances["relative_event_distribution_distance"] = relative_event_distribution_distance(
            A_event, custom_ids, G_event, custom_ids,
            discretize_type=AbsoluteTimestampType.BOTH,
            discretize_event=discretize_to_hour
        )
    except Exception as e:
        raise RuntimeError(f"Error computing distances (process-state): {e}") from e

    result_dict = {
        "ALog_stats_event": statsA,
        "GLog_stats_event": statsG,
        "distances": distances,
    }
    if verbose:
        print("[Process-State] Evaluation result:", json.dumps(result_dict, indent=4))
    return result_dict


###########################################################################
# WARM-UP APPROACH
###########################################################################
def evaluate_warmup_simulation(
    event_log: str,
    bpmn_model: str,
    bpmn_parameters: str,
    warmup_start: str,       # warm-up simulation start
    simulation_cut: str,     # discard events before this date (for simulation)
    evaluation_end: str,     # evaluation end time (new parameter)
    simulation_horizon: str, # simulation horizon (can be later than evaluation_end)
    total_cases: int = 1000,
    sim_stats_csv: str = "warmup_sim_stats.csv",
    sim_log_csv: str = "warmup_sim_log.csv",
    column_mapping: str = '{"case_id":"CaseId","activity":"Activity","resource":"Resource","start_time":"StartTime","end_time":"EndTime"}',
    rename_alog: dict = None,
    required_columns: list = None,
    simulate: bool = True,
    verbose: bool = True
) -> dict:
    """
    Warm-up simulation approach:
      1) Simulate from warmup_start.
      2) Discard simulation events before simulation_cut (for simulation purposes).
      3) For evaluation, keep events (for cases and for trimming) that lie within [simulation_cut, evaluation_end].
      4) For case metrics, keep entire cases if any event falls in the evaluation window.
      5) For event metrics, trim events to the evaluation window.
      6) Save both sets of logs.
    """
    if required_columns is None:
        required_columns = ["case_id", "activity", "start_time", "end_time", "resource"]
    if rename_alog is None:
        rename_alog = {
            "CaseId": "case_id",
            "Activity": "activity",
            "Resource": "resource",
            "StartTime": "start_time",
            "EndTime": "end_time",
        }

    # --- Step 1: Possibly run the warm-up simulation ---
    if simulate:
        if verbose:
            print("=== [Warm-up] Step 1: Running basic simulation (warm-up) ===")
        max_attempts = 3
        attempt = 1
        while attempt <= max_attempts:
            try:
                run_basic_simulation(
                    bpmn_model=bpmn_model,
                    json_sim_params=bpmn_parameters,
                    total_cases=total_cases,
                    out_stats_csv_path=sim_stats_csv,
                    out_log_csv_path=sim_log_csv,
                    start_date=warmup_start
                )
                break
            except Exception as e:
                if attempt == max_attempts:
                    raise RuntimeError(f"Warm-up simulation failed after {max_attempts} attempts: {e}") from e
                else:
                    print(f"[Warm-up] Attempt {attempt} failed: {e}. Retrying...")
                    attempt += 1
    else:
        if verbose:
            print("Skipping warm-up simulation (simulate=False).")

    # --- Step 2: Load and preprocess simulation log ---
    if verbose:
        print("=== [Warm-up] Step 2: Reading simulation log ===")
    if not os.path.isfile(sim_log_csv):
        raise FileNotFoundError(f"Warm-up simulation log not found: {sim_log_csv}")
    sim_df = pd.read_csv(sim_log_csv)
    sim_df.rename(columns=rename_alog, inplace=True)
    for col in ["enable_time", "start_time", "end_time"]:
        if col in sim_df.columns:
            sim_df[col] = pd.to_datetime(sim_df[col], utc=True, errors="coerce")

    # Define the evaluation window for warm-up:
    eval_start = pd.to_datetime(simulation_cut, utc=True)
    eval_end_dt = pd.to_datetime(evaluation_end, utc=True)

    # --- Step 3: Preprocess simulation log without discarding events ---
    sim_all = helper.preprocess_alog(sim_df)
    sim_case = helper.filter_cases_by_eval_window(sim_all, eval_start, eval_end_dt)
    sim_event = helper.trim_events_to_eval_window(sim_all, eval_start, eval_end_dt)

    # --- Step 4: Save both warm-up simulation logs ---
    if verbose:
        sim_case[required_columns].to_csv("samples/output/WU_G_case.csv", index=False)
        sim_event[required_columns].to_csv("samples/output/WU_G_event.csv", index=False)
        print("Saved WU_G_case and WU_G_event CSV files.")

    # --- Step 5: Load and preprocess reference ALog similarly ---
    if verbose:
        print("=== [Warm-up] Step 5: Reading and preprocessing reference ALog ===")
    if not os.path.isfile(event_log):
        raise FileNotFoundError(f"Reference event log not found: {event_log}")
    alog_df = pd.read_csv(event_log)
    alog_df.rename(columns=rename_alog, inplace=True)
    for col in ["enable_time", "start_time", "end_time"]:
        if col in alog_df.columns:
            alog_df[col] = pd.to_datetime(alog_df[col], utc=True, errors="coerce")
    A_all = helper.preprocess_alog(alog_df)
    A_case = helper.filter_cases_by_eval_window(A_all, eval_start, eval_end_dt)
    A_event = helper.trim_events_to_eval_window(A_all, eval_start, eval_end_dt)

    # --- Step 6: Save ALog for warm-up evaluation ---
    if verbose:
        A_case[required_columns].to_csv("samples/output/WU_A_case.csv", index=False)
        A_event[required_columns].to_csv("samples/output/WU_A_event.csv", index=False)
        print("Saved WU_A_case and WU_A_event CSV files.")

    # --- Step 7: Compute basic stats on simulation event logs ---
    statsSim = helper.basic_log_stats(sim_event)

    # --- Step 8: Compute log distances using event logs for events and full-case logs for case metrics ---
    custom_ids = EventLogIDs(
        case="case_id",
        activity="activity",
        start_time="start_time",
        end_time="end_time",
        resource="resource"
    )
    if verbose:
        print("=== [Warm-up] Step 8: Computing distances using event logs ===")
    distances = {}
    try:
        distances["control_flow_log_distance"] = control_flow_log_distance(A_event, custom_ids, sim_event, custom_ids)
        distances["n_gram_distribution_distance"] = n_gram_distribution_distance(A_event, custom_ids, sim_event, custom_ids, n=3)
        distances["absolute_event_distribution_distance"] = absolute_event_distribution_distance(
            A_event, custom_ids, sim_event, custom_ids,
            discretize_type=AbsoluteTimestampType.START,
            discretize_event=discretize_to_hour
        )
        distances["case_arrival_distribution_distance"] = case_arrival_distribution_distance(
            A_case, custom_ids, sim_case, custom_ids,
            discretize_event=discretize_to_hour
        )
        distances["cycle_time_distribution_distance"] = cycle_time_distribution_distance(
            A_case, custom_ids, sim_case, custom_ids,
            bin_size=pd.Timedelta(hours=1)
        )
        distances["circadian_workforce_distribution_distance"] = circadian_workforce_distribution_distance(
            A_case, custom_ids, sim_case, custom_ids
        )
        distances["relative_event_distribution_distance"] = relative_event_distribution_distance(
            A_event, custom_ids, sim_event, custom_ids,
            discretize_type=AbsoluteTimestampType.BOTH,
            discretize_event=discretize_to_hour
        )
    except Exception as e:
        raise RuntimeError(f"Error computing distances (warm-up): {e}") from e

    result_dict = {
        "SimStats": statsSim,
        "distances": distances,
    }
    if verbose:
        print("[Warm-up] Evaluation result:", json.dumps(result_dict, indent=4))
    return result_dict
