import os
import json
import pandas as pd
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from test.helper import (
    read_event_log,
    trim_events_to_eval_window,
    filter_ongoing_cases,
    filter_complete_cases,
    generate_short_uuid
)
from test.evaluation import (
    evaluate_partial_state_simulation,
    evaluate_warmup_simulation,
    aggregate_metrics,
    compare_results
)


def main():
    # ----------- CONFIG -----------
    EXISTING_ALOG_PATH = "samples/real_life/AcademicCredentials_fixed.csv"
    BPMN_MODEL = "samples/real_life/AcademicCredentials.bpmn"
    BPMN_PARAMS = "samples/real_life/AcademicCredentials.json"

    SIMULATION_CUT_DATE = "2016-04-28T10:10:00.000Z"  
    EVALUATION_END_DATE = "2016-06-29T23:20:30.000Z"  
    SIMULATION_HORIZON  = "2016-07-29T23:20:30.000Z"  
    WARMUP_START_DATE   = "2016-04-16T09:04:12.000Z"

    NUM_RUNS = 10
    PROC_TOTAL_CASES = 600
    WARMUP_TOTAL_CASES = 600

    # EXISTING_ALOG_PATH = "samples/real_life/BPIC_2012_new.csv"
    # BPMN_MODEL = "samples/real_life/BPIC_2012.bpmn"
    # BPMN_PARAMS = "samples/real_life/BPIC_2012.json"

    # SIMULATION_CUT_DATE = "2012-01-11T10:00:00.000Z"  
    # EVALUATION_END_DATE = "2012-02-01T10:00:00.000Z"  
    # SIMULATION_HORIZON  = "2012-03-05T10:00:00.000Z"  
    # WARMUP_START_DATE   = "2011-12-24T10:00:00.000Z"

    # NUM_RUNS = 10
    # PROC_TOTAL_CASES = 3500
    # WARMUP_TOTAL_CASES = 3500


    # Column renaming
    rename_alog_dict_eval = {
        "CaseId": "case_id",
        "Activity": "activity",
        "Resource": "resource",
        "StartTime": "start_time",
        "EndTime": "end_time"
    }
    rename_alog_dict_sim = {
        "case_id": "CaseId",
        "activity": "Activity",
        "resource": "Resource",
        "start_time": "StartTime",
        "end_time": "EndTime",
        "enabled_time": "EnabledTime"
    }
    rename_alog_dict_sim_inv = {v: k for k, v in rename_alog_dict_sim.items()}

    required_columns = ["case_id", "activity", "start_time", "end_time", "resource"]

    # ----------- OUTPUT DIR -----------
    run_id = generate_short_uuid(length=6)
    out_dir = os.path.join("outputs", run_id)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # ----------- STEP 1: Read & prepare reference data -----------
    print("=== Loading and preprocessing ALog once ===")
    alog_df = read_event_log(
        csv_path=EXISTING_ALOG_PATH,
        rename_map=rename_alog_dict_eval,
        required_columns=required_columns,
        verbose=True
    )

    eval_start_dt = pd.to_datetime(SIMULATION_CUT_DATE, utc=True)
    eval_end_dt   = pd.to_datetime(EVALUATION_END_DATE, utc=True)

    # 1A) Event filter subset
    A_event_filter_ref = trim_events_to_eval_window(alog_df, eval_start_dt, eval_end_dt)

    # 1B) Ongoing subset
    #    "ongoing" if case has min(start_time) < eval_start AND max(end_time) > eval_start
    #    then we only keep events whose end_time > eval_start
    #    But the helper function filter_ongoing_cases already does that logic
    A_ongoing_ref = filter_ongoing_cases(alog_df, eval_start_dt, eval_end_dt)

    # 1C) Complete subset
    #    "complete" if min(start_time) >= eval_start
    #    keep entire case
    start = alog_df[alog_df["case_id"]==140]["start_time"].min()
    print(f"Filtering complete cases... {start} {eval_start_dt} {eval_end_dt}")
    A_complete_ref = filter_complete_cases(alog_df, eval_start_dt, eval_end_dt)

    # Optionally, save these references in top-level output folder
    A_event_filter_ref.to_csv(os.path.join(out_dir, "A_event_filter_ref.csv"), index=False)
    A_ongoing_ref.to_csv(os.path.join(out_dir, "A_ongoing_ref.csv"), index=False)
    A_complete_ref.to_csv(os.path.join(out_dir, "A_complete_ref.csv"), index=False)

    # ----------- STEP 2: Run experiments -----------
    proc_run_distances = []
    warmup_run_distances = []

    horizon_dt = pd.to_datetime(SIMULATION_HORIZON, utc=True)
    warmup_start_dt = pd.to_datetime(WARMUP_START_DATE, utc=True)

    for run_index in range(1, NUM_RUNS + 1):
        print(f"\n--- Run {run_index} of {NUM_RUNS} ---")
        run_subfolder = os.path.join(out_dir, str(run_index))
        os.makedirs(run_subfolder, exist_ok=True)

        # Evaluate Process-State approach
        proc_result = evaluate_partial_state_simulation(
            run_output_dir=run_subfolder,
            event_log=EXISTING_ALOG_PATH,
            bpmn_model=BPMN_MODEL,
            bpmn_parameters=BPMN_PARAMS,
            start_time=eval_start_dt,
            simulation_horizon=horizon_dt,
            total_cases=PROC_TOTAL_CASES,
            A_event_filter_ref=A_event_filter_ref,
            A_ongoing_ref=A_ongoing_ref,
            A_complete_ref=A_complete_ref,
            evaluation_end=eval_end_dt,
            column_mapping=rename_alog_dict_sim,
            required_columns=required_columns,
            simulate=True,
            verbose=True
        )
        proc_run_distances.append(proc_result)

        # Evaluate Warm-Up approach
        warmup_result = evaluate_warmup_simulation(
            run_output_dir=run_subfolder,
            bpmn_model=BPMN_MODEL,
            bpmn_parameters=BPMN_PARAMS,
            warmup_start=warmup_start_dt,
            simulation_cut=eval_start_dt,
            evaluation_end=eval_end_dt,
            total_cases=WARMUP_TOTAL_CASES,
            A_event_filter_ref=A_event_filter_ref,
            A_ongoing_ref=A_ongoing_ref,
            A_complete_ref=A_complete_ref,
            rename_map=rename_alog_dict_sim_inv,
            required_columns=required_columns,
            simulate=True,
            verbose=True
        )
        warmup_run_distances.append(warmup_result)

    # ----------- STEP 3: Aggregate & Compare -----------
    print("\n=== Aggregating Process-State metrics ===")
    proc_agg = aggregate_metrics(proc_run_distances)

    print("\n=== Aggregating Warm-Up metrics ===")
    warmup_agg = aggregate_metrics(warmup_run_distances)

    print("\n=== Comparing Process-State vs. Warm-Up ===")
    comparison = compare_results(proc_agg, warmup_agg)

    # ----------- STEP 4: Save final output -----------
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
