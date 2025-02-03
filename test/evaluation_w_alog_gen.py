# test/evaluation_basic_then_partial.py

import os
import sys
import json
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.process_state_prosimos_run import run_basic_simulation
from test.evaluation import evaluate_partial_state_simulation

def main():
    """
    1) Run a basic Prosimos simulation (no partial-state or horizon). 
       This gives us an 'ALog' CSV.
    2) Use that generated ALog as input to the partial-state simulation code,
       then compute metrics comparing ALog vs GLog.
    """

    # ------------------------------------------------------------------
    # STEP 1: Basic simulation
    # ------------------------------------------------------------------
    print("=== Step 1: Running basic simulation to produce an ALog ===")

    bpmn_model = "samples/synthetic_xor_loop.bpmn"
    bpmn_params = "samples/synthetic_xor_loop.json"
    basic_stats_csv = "samples/output/basic_sim_stats.csv"
    basic_log_csv   = "samples/output/basic_sim_log.csv"

    # This runs a normal simulation with no partial-state, no horizon
    sim_time = run_basic_simulation(
        bpmn_model=bpmn_model,
        json_sim_params=bpmn_params,
        total_cases=1000,
        out_stats_csv_path=basic_stats_csv,
        out_log_csv_path=basic_log_csv,
        start_date="2012-03-19T10:10:00.000Z"
    )
    print(f"Basic simulation finished in {sim_time} seconds.")
    print(f"Produced ALog = '{basic_log_csv}', stats = '{basic_stats_csv}'\n")

    # ------------------------------------------------------------------
    # STEP 2: Partial-state (short-term) simulation with the new ALog
    # ------------------------------------------------------------------
    print("=== Step 2: Using the ALog for partial-state simulation & metrics ===")

    # We'll treat the CSV from Step 1 as our "ongoing event_log"
    event_log_path = basic_log_csv

    # We can choose a horizon, start_time, column_mapping, etc.
    # The 'column_mapping' is used by run_process_state_and_simulation
    # to interpret columns in the partial-state step.
    colmap_str = '{"case_id":"CaseId","activity":"Activity","resource":"Resource","start_time":"StartTime","end_time":"EndTime"}'

    # We'll rename columns in the "ALog"
    rename_alog_dict = {
        "CaseId":       "case_id",
        "Activity":     "activity",
        "Resource":     "resource",
        "StartTime":    "start_time",
        "EndTime":      "end_time",
    }

    # We'll produce new sim logs in these paths:
    partial_stats_csv = "samples/output/partial_sim_stats.csv"
    partial_log_csv   = "samples/output/partial_sim_log.csv"

    # Now call our existing function that:
    #   - runs partial-state simulation
    #   - reads the logs
    #   - computes metrics 
    #   - returns metrics JSON
    result_dict = evaluate_partial_state_simulation(
        event_log=event_log_path,
        bpmn_model=bpmn_model,
        bpmn_parameters=bpmn_params,
        start_time="2012-03-21T10:10:00.000Z",
        simulation_horizon="2012-04-25T23:10:30.000Z",
        column_mapping=colmap_str,
        total_cases=1000,
        sim_stats_csv=partial_stats_csv,
        sim_log_csv=partial_log_csv,
        rename_alog=rename_alog_dict,
        rename_glog=None,
        required_columns=["case_id","activity","start_time","end_time","resource"],
        simulate=True,
        verbose=True
    )

if __name__ == "__main__":
    main()
