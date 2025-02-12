import os
import sys
import json
import math
import re
import pandas as pd
from datetime import timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.process_state_prosimos_run import run_basic_simulation
from test.evaluation import evaluate_partial_state_simulation, evaluate_warmup_simulation
from test import helper


def ensure_fractional_seconds(ts: str) -> str:
    """
    Ensures the timestamp string has fractional seconds (.000) if missing.
    If a trailing 'Z' is present, it inserts the .000 before the 'Z'.
    """
    # Handle missing/NaN values:
    if pd.isnull(ts):
        return ts

    # If there's no decimal point, add '.000' (taking into account possible trailing 'Z')
    if '.' not in ts:
        # Insert .000 before any trailing 'Z'
        ts = re.sub(r'(Z)$', r'.000\1', ts)
        # If we didn't find a 'Z' to replace, and still no '.000', just append it
        if not ts.endswith('.000') and not ts.endswith('.000Z'):
            ts += '.000'
    return ts


def load_and_preprocess_alog(alog_path: str, simulation_cut_date: str, simulation_horizon: str) -> pd.DataFrame:
    """
    Load and preprocess the existing ALog.
    Converts time columns, renames columns, and filters to the evaluation window.
    """
    # Read and preprocess ALog
    alog_df = pd.read_csv(alog_path)
    
    # Rename columns to standard names
    rename_dict = {
        "CaseId": "case_id",
        "Activity": "activity",
        "Resource": "resource",
        "StartTime": "start_time",
        "EndTime": "end_time"
    }
    alog_df.rename(columns=rename_dict, inplace=True)
    
    # Ensure fractional seconds BEFORE parsing
    time_cols = ["enable_time", "start_time", "end_time"]
    for col in time_cols:
        if col in alog_df.columns:
            # First, normalize any timestamp strings
            alog_df[col] = alog_df[col].astype(str).apply(ensure_fractional_seconds)
            # Then parse them as datetimes
            alog_df[col] = pd.to_datetime(alog_df[col], utc=True, errors="coerce")
    
    # Filter to evaluation window
    cut_dt = pd.to_datetime(simulation_cut_date, utc=True)
    horizon_dt = pd.to_datetime(simulation_horizon, utc=True)
    alog_cut = alog_df[alog_df["start_time"] >= cut_dt].copy()
    if "case_id" in alog_cut.columns:
        min_st = alog_cut.groupby("case_id")["start_time"].transform("min")
        alog_cut = alog_cut[min_st <= horizon_dt]
    
    return alog_cut


def run_experiments(
    alog_path: str,
    bpmn_model: str,
    bpmn_params: str,
    simulation_cut_date: str,
    simulation_horizon: str,
    warmup_start: str,
    num_runs: int,
    proc_total_cases: int,
    warmup_total_cases: int
) -> tuple:
    """
    Run K experimental runs for both approaches.
    Returns aggregated results for process-state and warm-up approaches.
    """
    proc_run_distances = []
    warmup_run_distances = []
    colmap_str = '{"case_id":"CaseId","activity":"Activity","resource":"Resource","start_time":"StartTime","end_time":"EndTime"}'
    rename_alog_dict = {
        "CaseId": "case_id",
        "Activity": "activity",
        "Resource": "resource",
        "StartTime": "start_time",
        "EndTime": "end_time"
    }

    for run in range(num_runs):
        print(f"\n--- Run {run + 1} of {num_runs} ---")

        # Process-State Approach
        proc_result = evaluate_partial_state_simulation(
            event_log=alog_path,
            bpmn_model=bpmn_model,
            bpmn_parameters=bpmn_params,
            start_time=simulation_cut_date,
            simulation_horizon=simulation_horizon,
            column_mapping=colmap_str,
            total_cases=proc_total_cases,
            sim_stats_csv="samples/output/partial_sim_stats.csv",
            sim_log_csv="samples/output/partial_sim_log.csv",
            rename_alog=rename_alog_dict,
            required_columns=["case_id", "activity", "start_time", "end_time", "resource"],
            simulate=True,
            verbose=True
        )
        proc_run_distances.append(proc_result.get("distances", {}))

        # Warm-Up Approach
        warmup_result = evaluate_warmup_simulation(
            event_log=alog_path,
            bpmn_model=bpmn_model,
            bpmn_parameters=bpmn_params,
            warmup_start=warmup_start,
            simulation_cut=simulation_cut_date,
            simulation_horizon=simulation_horizon,
            total_cases=warmup_total_cases,
            sim_stats_csv="samples/output/warmup_sim_stats.csv",
            sim_log_csv="samples/output/warmup_sim_log.csv",
            column_mapping=colmap_str,
            rename_alog=rename_alog_dict,
            required_columns=["case_id", "activity", "start_time", "end_time", "resource"],
            simulate=True,
            verbose=True
        )
        warmup_run_distances.append(warmup_result.get("distances", {}))

    return proc_run_distances, warmup_run_distances


def aggregate_metrics(all_runs: list) -> dict:
    """Aggregate metrics across runs (mean, std, 95% CI)."""
    agg = {}
    if not all_runs:
        return agg
    
    metric_keys = all_runs[0].keys()
    for metric in metric_keys:
        values = [run_result[metric] for run_result in all_runs]
        mean_val = sum(values) / len(values)
        std_val = math.sqrt(sum((x - mean_val)**2 for x in values) / len(values))
        se = std_val / math.sqrt(len(values))
        agg[metric] = {
            "mean": mean_val,
            "std": std_val,
            "confidence_interval": [mean_val - 1.96*se, mean_val + 1.96*se],
            "individual_runs": values
        }
    return agg


def compare_results(proc_agg: dict, warmup_agg: dict) -> dict:
    """Compare aggregated results between approaches."""
    comparison = {}
    for metric in proc_agg.keys():
        proc_mean = proc_agg[metric]["mean"]
        warmup_mean = warmup_agg[metric]["mean"]
        comparison[metric] = {
            "process_state_mean": proc_mean,
            "warmup_mean": warmup_mean,
            "better_approach": (
                "process_state" if proc_mean < warmup_mean
                else "warmup" if warmup_mean < proc_mean
                else "tie"
            )
        }
    return comparison


def main():
    # ===================== CONFIGURATION =====================
    EXISTING_ALOG_PATH = "samples/real_life/AcademicCredentials_fixed.csv"
    BPMN_MODEL = "samples/real_life/AcademicCredentials.bpmn"
    BPMN_PARAMS = "samples/real_life/AcademicCredentials.json"
    
    # Simulation window parameters
    SIMULATION_CUT_DATE = "2016-04-28T10:10:00.000Z"
    SIMULATION_HORIZON = "2016-06-29T23:20:30.000Z"
    WARMUP_START_DATE = "2016-04-16T09:04:12.000Z"
    
    # Experimental parameters
    NUM_RUNS = 10
    PROC_TOTAL_CASES = 600
    WARMUP_TOTAL_CASES = 600
    
    # ===================== EXECUTION =====================
    print("=== Loading and preprocessing ALog ===")
    alog_df = load_and_preprocess_alog(
        EXISTING_ALOG_PATH, 
        SIMULATION_CUT_DATE,
        SIMULATION_HORIZON
    )
    
    print("\n=== Running experiments ===")
    proc_results, warmup_results = run_experiments(
        alog_path=EXISTING_ALOG_PATH,
        bpmn_model=BPMN_MODEL,
        bpmn_params=BPMN_PARAMS,
        simulation_cut_date=SIMULATION_CUT_DATE,
        simulation_horizon=SIMULATION_HORIZON,
        warmup_start=WARMUP_START_DATE,
        num_runs=NUM_RUNS,
        proc_total_cases=PROC_TOTAL_CASES,
        warmup_total_cases=WARMUP_TOTAL_CASES
    )
    
    print("\n=== Aggregating results ===")
    proc_agg = aggregate_metrics(proc_results)
    warmup_agg = aggregate_metrics(warmup_results)
    
    print("\n=== Comparing approaches ===")
    comparison = compare_results(proc_agg, warmup_agg)
    
    # Final output
    final_output = {
        "num_runs": NUM_RUNS,
        "process_state": {"aggregated_results": proc_agg, "all_runs": proc_results},
        "warmup": {"aggregated_results": warmup_agg, "all_runs": warmup_results},
        "comparison_summary": comparison
    }
    
    print("\n=== Final Results ===")
    print(json.dumps(final_output, indent=4))


if __name__ == "__main__":
    main()
