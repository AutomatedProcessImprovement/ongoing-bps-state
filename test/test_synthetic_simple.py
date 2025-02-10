# test/test_synthetic_simple.py

import sys
import os
import multiprocessing

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.runner import run_process_state_and_simulation

def main():
    run_process_state_and_simulation(
        event_log="samples/synthetic_xor_loop_ongoing.csv",
        bpmn_model="samples/synthetic_xor_loop.bpmn",
        bpmn_parameters="samples/synthetic_xor_loop.json",
        start_time="2012-03-21T10:10:00.000Z",
        column_mapping='{"case_id":"CaseId","Resource":"Resource","Activity":"Activity","__start_time":"StartTime","end_time":"EndTime"}',
        simulate=True,
        simulation_horizon="2012-03-21T23:10:30.000Z",
        total_cases=20,
        sim_stats_csv="samples/output/sim_stats.csv",
        sim_log_csv="samples/output/sim_log.csv",
    )

if __name__ == "__main__":
    # multiprocessing.freeze_support()
    main()
