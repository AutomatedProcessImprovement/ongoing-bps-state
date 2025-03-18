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
    evaluate_warmup_simulation_variable_start,
    aggregate_metrics,
    compare_results
)


def main():
    # ----------- CONFIG -----------
    NUM_RUNS = 10

    # # # # # # #
    # BPIC 2012 #
    # # # # # # #

    # EXISTING_ALOG_PATH = "samples/bpm-2025/real-life-configuration-and-logs/BPIC_2012_W.csv"
    # BPMN_MODEL = "samples/bpm-2025/real-life-configuration-and-logs/2012_diff_extr/best_result/BPIC_2012_train.bpmn"
    # BPMN_PARAMS = "samples/bpm-2025/real-life-configuration-and-logs/2012_diff_extr/best_result/BPIC_2012_train.json"
    # PROC_TOTAL_CASES = 5000
    # WARMUP_TOTAL_CASES = 5000
    # # Split point
    # SIMULATION_CUT_DATE = pd.to_datetime("2012-01-16T13:00:00.000Z", utc=True)
    # # Horizon:
    # horizon = pd.Timedelta(days=23)  # 90% percentile of trace durations

    # # # # # # #
    # BPIC 2017 #
    # # # # # # #

    # EXISTING_ALOG_PATH = "samples/bpm-2025/real-life-configuration-and-logs/BPIC_2017_W.csv"
    # BPMN_MODEL = "samples/bpm-2025/real-life-configuration-and-logs/2017_diff_extr/best_result/BPIC_2017_train.bpmn"
    # BPMN_PARAMS = "samples/bpm-2025/real-life-configuration-and-logs/2017_diff_extr/best_result/BPIC_2017_train.json"
    # PROC_TOTAL_CASES = 10000
    # WARMUP_TOTAL_CASES = 10000
    # # Split point
    # SIMULATION_CUT_DATE = pd.to_datetime("2016-10-10T13:00:00.000Z", utc=True)
    # # Horizon:
    # horizon = pd.Timedelta(days=26)  # 90% percentile of trace durations

    # # # # # # # # # # # # #
    # Academic  credentials #
    # # # # # # # # # # # # #

    # EXISTING_ALOG_PATH = "samples/bpm-2025/real-life-configuration-and-logs/AcademicCredentials.csv"
    # BPMN_MODEL = "samples/bpm-2025/real-life-configuration-and-logs/academic_diff_extr/best_result/AcademicCredentials_train.bpmn"
    # BPMN_PARAMS = "samples/bpm-2025/real-life-configuration-and-logs/academic_diff_extr/best_result/AcademicCredentials_train.json"
    # PROC_TOTAL_CASES = 5000
    # WARMUP_TOTAL_CASES = 5000
    # # Split point
    # SIMULATION_CUT_DATE = pd.to_datetime("2016-05-02T13:00:00.000Z", utc=True)  # Monday
    # # Horizon:
    # horizon = pd.Timedelta(days=46)  # 90% percentile of trace durations

    # # # # # # # #
    # Work Orders #
    # # # # # # # #

    # EXISTING_ALOG_PATH = "samples/bpm-2025/real-life-configuration-and-logs/work_orders.csv.gz"
    # BPMN_MODEL = "samples/bpm-2025/real-life-configuration-and-logs/workorders_diff_extr/best_result/work_orders_train.bpmn"
    # BPMN_PARAMS = "samples/bpm-2025/real-life-configuration-and-logs/workorders_diff_extr/best_result/work_orders_train.json"
    # PROC_TOTAL_CASES = 20000
    # WARMUP_TOTAL_CASES = 20000
    # # Split point
    # SIMULATION_CUT_DATE = pd.to_datetime("2022-12-19T07:00:00.000Z", utc=True)  # Monday
    # # Horizon:
    # horizon = pd.Timedelta(days=24)  # 90% percentile of trace durations

    # # # # # # # # # # # # # #
    # Loan Application steady #
    # # # # # # # # # # # # # #

    # EXISTING_ALOG_PATH = "samples/bpm-2025/synthetic-configuration-and-logs/Loan_steady/Loan_Application_log.csv"
    # BPMN_MODEL = "samples/bpm-2025/synthetic-configuration-and-logs/Loan_steady/Loan_Application.bpmn"
    # BPMN_PARAMS = "samples/bpm-2025/synthetic-configuration-and-logs/Loan_steady/Loan_Application.json"
    # PROC_TOTAL_CASES = 1000
    # WARMUP_TOTAL_CASES = 1000
    # # Split point
    # SIMULATION_CUT_DATE = pd.to_datetime("2025-02-24T07:00:00.000Z", utc=True)
    # # Horizon:
    # horizon = pd.Timedelta(days=35)  # 90% percentile of trace durations

    # # # # # # # # # # # # # #
    # Loan Application wobbly #
    # # # # # # # # # # # # # #

    # EXISTING_ALOG_PATH = "samples/bpm-2025/synthetic-configuration-and-logs/Loan_wobbly/Loan_Application_log_wobbly.csv"
    # BPMN_MODEL = "samples/bpm-2025/synthetic-configuration-and-logs/Loan_wobbly/Loan_Application.bpmn"
    # BPMN_PARAMS = "samples/bpm-2025/synthetic-configuration-and-logs/Loan_wobbly/Loan_Application.json"
    # PROC_TOTAL_CASES = 3000
    # WARMUP_TOTAL_CASES = 3000
    # # Split point
    # SIMULATION_CUT_DATE = pd.to_datetime("2025-02-21T07:00:00.000Z", utc=True)  # Friday
    # # Horizon:
    # horizon = pd.Timedelta(days=60)  # 90% percentile of trace durations

    # # # # # # # #
    # P2P wobbly  #
    # # # # # # # #

    # EXISTING_ALOG_PATH = "samples/bpm-2025/synthetic-configuration-and-logs/P2P_wobbly/P2P no-steady-state.csv"
    # BPMN_MODEL = "samples/bpm-2025/synthetic-configuration-and-logs/P2P_wobbly/P2P no-steady-state.bpmn"
    # BPMN_PARAMS = "samples/bpm-2025/synthetic-configuration-and-logs/P2P_wobbly/P2P-no-steady-state.json"
    # PROC_TOTAL_CASES = 3000
    # WARMUP_TOTAL_CASES = 3000
    # # Split point
    # SIMULATION_CUT_DATE = pd.to_datetime("2020-01-14T10:00:00.000Z", utc=True)  
    # # Horizon:
    # horizon = pd.Timedelta(days=30)  # 90% percentile of trace durations

    # # # # # # # #
    # P2P steady  #
    # # # # # # # #

    EXISTING_ALOG_PATH = "samples/bpm-2025/synthetic-configuration-and-logs/P2P_wobbly/P2P no-steady-state.csv"
    BPMN_MODEL = "samples/bpm-2025/synthetic-configuration-and-logs/P2P_wobbly/P2P no-steady-state.bpmn"
    BPMN_PARAMS = "samples/bpm-2025/synthetic-configuration-and-logs/P2P_wobbly/P2P-no-steady-state.json"
    PROC_TOTAL_CASES = 3500
    WARMUP_TOTAL_CASES = 3500
    # Split point
    SIMULATION_CUT_DATE = pd.to_datetime("2020-02-16T10:00:00.000Z", utc=True)  
    # Horizon:
    horizon = pd.Timedelta(days=45)

    # # Compute rest of instants
    WARMUP_START_DATE = SIMULATION_CUT_DATE - horizon
    EVALUATION_END_DATE = SIMULATION_CUT_DATE + horizon
    SIMULATION_HORIZON = SIMULATION_CUT_DATE + (2 * horizon)

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
    # print(f"Filtering complete cases... {start} {eval_start_dt} {eval_end_dt}")
    A_complete_ref = filter_complete_cases(alog_df, eval_start_dt, eval_end_dt)

    A_event_filter_ref.to_csv(os.path.join(out_dir, "A_event_filter_ref.csv"), index=False)
    A_ongoing_ref.to_csv(os.path.join(out_dir, "A_ongoing_ref.csv"), index=False)
    A_complete_ref.to_csv(os.path.join(out_dir, "A_complete_ref.csv"), index=False)

    # ----------- STEP 2: Run experiments -----------
    proc_run_distances = []
    warmup_run_distances = []
    warmup2_run_distances = []

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

        # Copy process state file to run output directory
        if os.path.exists("output.json"):
            import shutil
            shutil.copy2("output.json", os.path.join(run_subfolder, "process_state.json"))

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

        # (C) Evaluate NEW Warm-Up Variation
        warmup2_result = evaluate_warmup_simulation_variable_start(
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
            A_full=alog_df,
            rename_map=rename_alog_dict_sim_inv,
            required_columns=required_columns,
            simulate=True,
            verbose=True
        )
        warmup2_run_distances.append(warmup2_result)

    # ----------- STEP 3: Aggregate & Compare -----------
    print("\n=== Aggregating Process-State metrics ===")
    proc_agg = aggregate_metrics(proc_run_distances)

    print("\n=== Aggregating Warm-Up metrics ===")
    warmup_agg = aggregate_metrics(warmup_run_distances)

    print("\n=== Aggregating Warm-Up v2 metrics ===")
    warmup2_agg = aggregate_metrics(warmup2_run_distances)

    print("\n=== Comparing Process-State vs. Warm-Up ===")
    comparison = compare_results(proc_agg, warmup_agg)

    comparison_proc_vs_wu2 = compare_results(proc_agg, warmup2_agg)


    # ----------- STEP 4: Save final output -----------
    final_output = {
        "num_runs": NUM_RUNS,
        "process_state": {
            "aggregated_results": proc_agg,
            # "all_runs": proc_run_distances
        },
        "warmup": {
            "aggregated_results": warmup_agg,
            # "all_runs": warmup_run_distances
        },
        "warmup2": {
            "aggregated_results": warmup2_agg,
            # "all_runs": warmup2_run_distances
        },
        # "comparison_summary": comparison,
        # "comparison_proc_vs_wu2": comparison_proc_vs_wu2
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
