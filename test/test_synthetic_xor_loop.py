# test/test_synthetic_xor_loop.py

import sys
import os
import json
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from test import evaluation

if __name__ == "__main__":
    colmap_str = '{"case_id":"CaseId","Resource":"Resource","Activity":"Activity","__start_time":"StartTime","end_time":"EndTime","enable_time":"__AssignedTime"}'

    rename_alog_dict = {
        "case_id": "case_id",
        "Activity": "activity",
        "Resource": "resource",
        "__start_time": "start_time",
        "end_time": "end_time",
    }

    result_json = evaluation.evaluate_partial_state_simulation(
        event_log="samples/synthetic_xor_loop_ongoing.csv",
        bpmn_model="samples/synthetic_xor_loop.bpmn",
        bpmn_parameters="samples/synthetic_xor_loop.json",
        start_time="2012-03-21T10:10:00.000Z",
        simulation_horizon="2012-04-25T23:10:30.000Z",
        column_mapping=colmap_str,
        total_cases=1000,
        sim_stats_csv="samples/output/sim_stats.csv",
        sim_log_csv="samples/output/sim_log.csv",
        rename_alog=rename_alog_dict,
        rename_glog=None,
        required_columns=["case_id","activity","start_time","end_time","resource"],
        simulate=True,
        verbose=True
    )

