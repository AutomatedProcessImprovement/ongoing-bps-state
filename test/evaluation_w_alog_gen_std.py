# test/evaluation_w_alog_gen_std.py

import os
import sys
import json
import math
import pandas as pd
from datetime import timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.process_state_prosimos_run import run_basic_simulation
from test.evaluation import evaluate_partial_state_simulation, evaluate_warmup_simulation
from test import helper

def main():
    """
    1) Generate a basic simulation (ALog) that covers cases from 0 to X.
    2) Define the simulation window:
         - simulation_cut_date: the cut-off point (start of evaluation period)
         - simulation_horizon: end of evaluation period.
    3) For each of K runs, run:
         (a) The warm-up approach: run a basic simulation starting at the same simulation date,
             then cut and preprocess the log (discard events before simulation_cut_date and keep only cases finishing
             before simulation_horizon) using the same preprocessing as in the process-state approach.
         (b) The process-state approach: run the partial-state simulation starting at simulation_cut_date.
    4) For each run and each approach, compute distance metrics comparing the simulation output to the reference ALog.
    5) Aggregate results (mean, std, 95% CI) across runs for each approach.
    6) Perform a simple analysis comparing the aggregated mean distances between approaches.
    """

    # -------------------------------
    # STEP 1: Generate initial ALog (reference)
    # -------------------------------
    print("=== Step 1: Running basic simulation to produce ALog (reference) ===")
    bpmn_model = "samples/synthetic_xor_loop.bpmn"
    bpmn_params = "samples/synthetic_xor_loop.json"
    basic_stats_csv = "samples/output/basic_sim_stats.csv"
    basic_log_csv   = "samples/output/basic_sim_log.csv"

    # Increase total cases to ensure we cover the whole evaluation window
    sim_time = run_basic_simulation(
        bpmn_model=bpmn_model,
        json_sim_params=bpmn_params,
        total_cases=5000,
        out_stats_csv_path=basic_stats_csv,
        out_log_csv_path=basic_log_csv,
        start_date="2012-03-19T10:10:00.000Z"
    )
    print(f"Basic simulation (ALog) finished in {sim_time} seconds.")
    print(f"Produced ALog = '{basic_log_csv}'\n")

    # Read the ALog and convert time columns
    alog_df = pd.read_csv(basic_log_csv)
    for col in ["StartTime", "EndTime"]:
        if col in alog_df.columns:
            alog_df[col] = pd.to_datetime(alog_df[col], utc=True, errors="coerce")
    alog_df.rename(columns={
        "CaseId": "case_id",
        "Activity": "activity",
        "Resource": "resource",
        "StartTime": "start_time",
        "EndTime": "end_time"
    }, inplace=True)
    if "start_time" in alog_df.columns:
        alog_df["start_time"] = pd.to_datetime(alog_df["start_time"], utc=True, errors="coerce")
    if "end_time" in alog_df.columns:
        alog_df["end_time"] = pd.to_datetime(alog_df["end_time"], utc=True, errors="coerce")

    # Define the simulation window for evaluation:
    simulation_cut_date = "2012-04-21T10:10:00.000Z"
    simulation_horizon = "2012-09-25T23:10:30.000Z"

    # For later comparison, cut the ALog to the evaluation window.
    cut_dt = pd.to_datetime(simulation_cut_date, utc=True)
    horizon_dt = pd.to_datetime(simulation_horizon, utc=True)
    alog_cut = alog_df[alog_df["start_time"] >= cut_dt].copy()
    if "case_id" in alog_cut.columns:
        min_st = alog_cut.groupby("case_id")["start_time"].transform("min")
        alog_cut = alog_cut[min_st <= horizon_dt]

    # -------------------------------
    # STEP 2: Run experiments for both approaches (K runs)
    # -------------------------------
    num_runs = 15

    # File names for simulation outputs (they can be overwritten each run)
    proc_stats_csv = "samples/output/partial_sim_stats.csv"
    proc_log_csv   = "samples/output/partial_sim_log.csv"
    warmup_stats_csv = "samples/output/warmup_sim_stats.csv"
    warmup_log_csv   = "samples/output/warmup_sim_log.csv"

    # Column mapping and renaming (for simulation logs)
    colmap_str = '{"case_id":"CaseId","activity":"Activity","resource":"Resource","start_time":"StartTime","end_time":"EndTime"}'
    rename_alog_dict = {
        "CaseId": "case_id",
        "Activity": "activity",
        "Resource": "resource",
        "StartTime": "start_time",
        "EndTime": "end_time"
    }

    proc_total_cases = 4000

    # For the warm-up approach, the warmup simulation starts at the same simulation date as ALog.
    warmup_start = "2012-03-19T10:10:00.000Z"
    warmup_total_cases = 4000

    # Lists to collect distances for each run and each approach.
    proc_run_distances = []   # process-state approach distances
    warmup_run_distances = [] # warm-up approach distances

    print("\n=== Step 3: Running experiments over {} runs ===".format(num_runs))
    for run in range(num_runs):
        print(f"\n--- Run {run + 1} of {num_runs} ---")

        # (A) Process-State Approach
        print("\n[Process-State Approach]")
        proc_result = evaluate_partial_state_simulation(
            event_log=basic_log_csv,
            bpmn_model=bpmn_model,
            bpmn_parameters=bpmn_params,
            start_time=simulation_cut_date,
            simulation_horizon=simulation_horizon,
            column_mapping=colmap_str,
            total_cases=proc_total_cases,
            sim_stats_csv=proc_stats_csv,
            sim_log_csv=proc_log_csv,
            rename_alog=rename_alog_dict,
            rename_glog=None,
            required_columns=["case_id", "activity", "start_time", "end_time", "resource"],
            simulate=True,
            verbose=True
        )
        proc_run_distances.append(proc_result.get("distances", {}))

        # (B) Warm-Up Approach
        print("\n[Warm-up Approach]")
        warmup_result = evaluate_warmup_simulation(
            event_log=basic_log_csv,
            bpmn_model=bpmn_model,
            bpmn_parameters=bpmn_params,
            warmup_start=warmup_start,
            simulation_cut=simulation_cut_date,
            simulation_horizon=simulation_horizon,
            total_cases=warmup_total_cases,
            sim_stats_csv=warmup_stats_csv,
            sim_log_csv=warmup_log_csv,
            column_mapping=colmap_str,
            rename_alog=rename_alog_dict,
            required_columns=["case_id", "activity", "start_time", "end_time", "resource"],
            simulate=True,
            verbose=True
        )
        warmup_run_distances.append(warmup_result.get("distances", {}))

    # -------------------------------
    # STEP 3: Aggregate results for each approach
    # -------------------------------
    def aggregate_metrics(all_runs):
        agg = {}
        if not all_runs:
            return agg
        # Assume each run's distances is a dict with the same keys.
        metric_keys = all_runs[0].keys()
        for metric in metric_keys:
            values = [run_result[metric] for run_result in all_runs]
            mean_val = sum(values) / len(values)
            std_val = math.sqrt(sum((x - mean_val) ** 2 for x in values) / len(values))
            se = std_val / math.sqrt(len(values))
            lower_bound = mean_val - 1.96 * se
            upper_bound = mean_val + 1.96 * se
            agg[metric] = {
                "mean": mean_val,
                "std": std_val,
                "confidence_interval": [lower_bound, upper_bound],
                "individual_runs": values
            }
        return agg

    proc_agg = aggregate_metrics(proc_run_distances)
    warmup_agg = aggregate_metrics(warmup_run_distances)

    # -------------------------------
    # STEP 4: Comparison Analysis
    # -------------------------------
    # For each metric, compare the aggregated mean values. Lower distance is assumed to be better.
    comparison_summary = {}
    metric_keys = proc_agg.keys()
    for metric in metric_keys:
        proc_mean = proc_agg[metric]["mean"]
        warmup_mean = warmup_agg[metric]["mean"]
        if proc_mean < warmup_mean:
            better_approach = "process_state"
        elif warmup_mean < proc_mean:
            better_approach = "warmup"
        else:
            better_approach = "tie"
        comparison_summary[metric] = {
            "process_state_mean": proc_mean,
            "warmup_mean": warmup_mean,
            "better_approach": better_approach
        }

    # -------------------------------
    # STEP 5: Build final output JSON
    # -------------------------------
    final_output = {
        "num_runs": num_runs,
        "process_state": {
            "aggregated_results": proc_agg,
            "all_run_results": proc_run_distances
        },
        "warmup": {
            "aggregated_results": warmup_agg,
            "all_run_results": warmup_run_distances
        },
        "comparison_summary": comparison_summary
    }

    print("\n=== Final Aggregated Results ===")
    print(json.dumps(final_output, indent=4))

if __name__ == "__main__":
    main()
