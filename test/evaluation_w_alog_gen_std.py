# test/evaluation_w_alog_gen_std.py

import os
import sys
import json
import math
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.process_state_prosimos_run import run_basic_simulation
from test.evaluation import evaluate_partial_state_simulation

def main():
    """
    1) Run a basic Prosimos simulation (no partial-state or horizon) to produce an ALog CSV.
    2) Use that ALog as input to the partial-state simulation and distance computation.
    3) Repeat the partial-state simulation for a given number of runs (to account for stochasticity).
    4) For each metric (e.g., control-flow log distance, n-gram distribution, etc.), aggregate the results by computing:
         - Mean (μ)
         - Standard Deviation (σ)
         - 95% Confidence Interval (CI = μ ± 1.96 * (σ/√K))
    5) Output a JSON with the metrics for each run and the aggregated statistics.
    """

    # ------------------------------------------------------------------
    # STEP 1: Basic Simulation
    # ------------------------------------------------------------------
    print("=== Step 1: Running basic simulation to produce an ALog ===")
    bpmn_model = "samples/synthetic_xor_loop.bpmn"
    bpmn_params = "samples/synthetic_xor_loop.json"
    basic_stats_csv = "samples/output/basic_sim_stats.csv"
    basic_log_csv   = "samples/output/basic_sim_log.csv"

    sim_time = run_basic_simulation(
        bpmn_model=bpmn_model,
        json_sim_params=bpmn_params,
        total_cases=5000,
        out_stats_csv_path=basic_stats_csv,
        out_log_csv_path=basic_log_csv,
        start_date="2012-03-19T10:10:00.000Z"
    )
    print(f"Basic simulation finished in {sim_time} seconds.")
    print(f"Produced ALog = '{basic_log_csv}', stats = '{basic_stats_csv}'\n")

    # ------------------------------------------------------------------
    # STEP 2: Partial-State Simulation & Distance Computation (Multiple Runs)
    # ------------------------------------------------------------------
    print("=== Step 2: Running partial-state simulation & distance computation ===")
    event_log_path = basic_log_csv

    # Column mapping and renaming for ALog (used during simulation)
    colmap_str = '{"case_id":"CaseId","activity":"Activity","resource":"Resource","start_time":"StartTime","end_time":"EndTime"}'
    rename_alog_dict = {
        "CaseId":    "case_id",
        "Activity":  "activity",
        "Resource":  "resource",
        "StartTime": "start_time",
        "EndTime":   "end_time",
    }
    partial_stats_csv = "samples/output/partial_sim_stats.csv"
    partial_log_csv   = "samples/output/partial_sim_log.csv"

    # Simulation parameters for partial-state simulation
    start_time = "2012-04-21T10:10:00.000Z"
    simulation_horizon = "2012-09-25T23:10:30.000Z"
    total_cases = 4000

    # Number of runs to perform (to account for stochasticity)
    num_runs = 15

    # List to collect the distances dictionary from each run.
    all_run_distances = []

    for run in range(num_runs):
        print(f"\n--- Run {run + 1} of {num_runs} ---")
        result_dict = evaluate_partial_state_simulation(
            event_log=event_log_path,
            bpmn_model=bpmn_model,
            bpmn_parameters=bpmn_params,
            start_time=start_time,
            simulation_horizon=simulation_horizon,
            column_mapping=colmap_str,
            total_cases=total_cases,
            sim_stats_csv=partial_stats_csv,
            sim_log_csv=partial_log_csv,
            rename_alog=rename_alog_dict,
            rename_glog=None,
            required_columns=["case_id", "activity", "start_time", "end_time", "resource"],
            simulate=True,
            verbose=True
        )
        # We assume that each result_dict has a "distances" key with the computed metrics.
        all_run_distances.append(result_dict.get("distances", {}))

    # ------------------------------------------------------------------
    # STEP 3: Aggregation of Results
    # ------------------------------------------------------------------
    # For each distance metric, compute the mean, standard deviation, and 95% confidence interval.
    aggregated_results = {}
    if all_run_distances:
        # Get the list of metric names from the first run's distances
        metric_keys = all_run_distances[0].keys()
        for metric in metric_keys:
            # Collect the value for this metric from each run.
            values = [run_result[metric] for run_result in all_run_distances]
            mean_val = sum(values) / len(values)
            # Compute population standard deviation
            std_val = math.sqrt(sum((x - mean_val) ** 2 for x in values) / len(values))
            # Standard error (SE) = std / sqrt(K)
            se = std_val / math.sqrt(len(values))
            lower_bound = mean_val - 1.96 * se
            upper_bound = mean_val + 1.96 * se

            aggregated_results[metric] = {
                "mean": mean_val,
                "std": std_val,
                "confidence_interval": [lower_bound, upper_bound],
                "individual_runs": values
            }

    # Build the final output JSON.
    final_output = {
        "num_runs": num_runs,
        "aggregated_results": aggregated_results,
        "all_run_results": all_run_distances
    }

    print("\n=== Final Aggregated Results ===")
    print(json.dumps(final_output, indent=4))

if __name__ == "__main__":
    main()
