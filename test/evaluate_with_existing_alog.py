import os
import json
import pandas as pd
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from test.helper import (
    read_event_log,
    filter_cases_by_eval_window,
    trim_events_to_eval_window,
    generate_short_uuid
)
from test.evaluation import (
    evaluate_partial_state_simulation,
    evaluate_warmup_simulation,
    aggregate_metrics,
    compare_results
)


def main():
    # -----------
    #  CONFIG
    # -----------
    # Paths to reference log, BPMN model, BPMN params
    EXISTING_ALOG_PATH = "samples/real_life/AcademicCredentials_fixed.csv"
    BPMN_MODEL = "samples/real_life/AcademicCredentials.bpmn"
    BPMN_PARAMS = "samples/real_life/AcademicCredentials.json"

    # Simulation / evaluation windows
    SIMULATION_CUT_DATE = "2016-04-28T10:10:00.000Z"  # evaluation start
    EVALUATION_END_DATE = "2016-06-29T23:20:30.000Z"   # evaluation end
    SIMULATION_HORIZON  = "2016-07-29T23:20:30.000Z"   # horizon
    WARMUP_START_DATE   = "2016-04-16T09:04:12.000Z"

    # Experimental parameters
    NUM_RUNS = 5
    PROC_TOTAL_CASES = 600
    WARMUP_TOTAL_CASES = 600

    # Two mappings:
    # (1) For evaluation, our reference file already has lowercase names.
    rename_alog_dict_eval = {
        "CaseId": "case_id",
        "Activity": "activity",
        "Resource": "resource",
        "StartTime": "start_time",
        "EndTime": "end_time"
    }
    # (2) For simulation input, we need to tell the simulation the expected column names.
    # That mapping converts our evaluation names to simulation names.
    rename_alog_dict_sim = {
        "case_id": "CaseId",
        "activity": "Activity",
        "resource": "Resource",
        "start_time": "StartTime",
        "end_time": "EndTime"
    }
    # The inverse mapping (for simulation output) is:
    rename_alog_dict_sim_inv = {v: k for k, v in rename_alog_dict_sim.items()}

    # Required columns for evaluation functions (always lowercase)
    required_columns = ["case_id", "activity", "start_time", "end_time", "resource"]

    # -----------
    #  OUTPUT DIR
    # -----------
    run_id = generate_short_uuid(length=6)   # e.g. 'a1b2c3'
    out_dir = os.path.join("outputs", run_id)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # -----------
    #  STEP 1: Read & prepare reference data ONCE
    # -----------
    print("=== Loading and preprocessing ALog once ===")
    # Our reference file already has lowercase column names.
    alog_df = read_event_log(
        csv_path=EXISTING_ALOG_PATH,
        rename_map=rename_alog_dict_eval,
        required_columns=required_columns,
        verbose=True
    )

    # Convert strings to Timestamps
    eval_start_dt = pd.to_datetime(SIMULATION_CUT_DATE, utc=True)
    eval_end_dt   = pd.to_datetime(EVALUATION_END_DATE, utc=True)

    # Prepare reference subsets (using evaluation column names)
    A_case = filter_cases_by_eval_window(alog_df, eval_start_dt, eval_end_dt)
    A_event = trim_events_to_eval_window(alog_df, eval_start_dt, eval_end_dt)

    # Save these reference subsets in the top-level output folder
    A_case.to_csv(os.path.join(out_dir, "A_case.csv"), index=False)
    A_event.to_csv(os.path.join(out_dir, "A_event.csv"), index=False)

    # -----------
    #  STEP 2: Run experiments
    # -----------
    proc_run_distances = []
    warmup_run_distances = []

    horizon_dt = pd.to_datetime(SIMULATION_HORIZON, utc=True)
    warmup_start_dt = pd.to_datetime(WARMUP_START_DATE, utc=True)

    for run_index in range(1, NUM_RUNS + 1):
        print(f"\n--- Run {run_index} of {NUM_RUNS} ---")
        run_subfolder = os.path.join(out_dir, str(run_index))
        os.makedirs(run_subfolder, exist_ok=True)

        # (1) Evaluate Process-State approach
        proc_result = evaluate_partial_state_simulation(
            run_output_dir=run_subfolder,
            event_log=EXISTING_ALOG_PATH,
            bpmn_model=BPMN_MODEL,
            bpmn_parameters=BPMN_PARAMS,
            start_time=eval_start_dt,
            simulation_horizon=horizon_dt,
            total_cases=PROC_TOTAL_CASES,
            A_case_ref=A_case,
            A_event_ref=A_event,
            evaluation_end=eval_end_dt,
            column_mapping=rename_alog_dict_sim,
            required_columns=required_columns,
            simulate=True,
            verbose=True
        )
        proc_run_distances.append(proc_result.get("distances", {}))

        # (2) Evaluate Warm-Up approach
        warmup_result = evaluate_warmup_simulation(
            run_output_dir=run_subfolder,
            bpmn_model=BPMN_MODEL,
            bpmn_parameters=BPMN_PARAMS,
            warmup_start=warmup_start_dt,
            simulation_cut=eval_start_dt,
            evaluation_end=eval_end_dt,
            total_cases=WARMUP_TOTAL_CASES,
            A_case_ref=A_case,
            A_event_ref=A_event,
            rename_map=rename_alog_dict_sim_inv,
            required_columns=required_columns,
            simulate=True,
            verbose=True
        )
        warmup_run_distances.append(warmup_result.get("distances", {}))

    # -----------
    #  STEP 3: Aggregate & Compare
    # -----------
    print("\n=== Aggregating Process-State metrics ===")
    proc_agg = aggregate_metrics(proc_run_distances)

    print("\n=== Aggregating Warm-Up metrics ===")
    warmup_agg = aggregate_metrics(warmup_run_distances)

    print("\n=== Comparing Process-State vs. Warm-Up ===")
    comparison = compare_results(proc_agg, warmup_agg)

    # -----------
    #  STEP 4: Save final output
    # -----------
    final_output = {
        "num_runs": NUM_RUNS,
        "process_state": {
            "aggregated_results": proc_agg,
            "all_runs": proc_run_distances
        },
        "warmup": {
            "aggregated_results": warmup_agg,
            "all_runs": warmup_run_distances
        },
        "comparison_summary": comparison,
    }

    results_path = os.path.join(out_dir, "final_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4)

    print("\n=== Final Results saved to ===")
    print(results_path)
    print("\n=== Final Results ===")
    print(json.dumps(final_output, indent=4))

if __name__ == "__main__":
    main()
