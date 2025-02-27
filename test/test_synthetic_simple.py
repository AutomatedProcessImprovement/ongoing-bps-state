# test/test_synthetic_simple.py

import os
import sys
import multiprocessing

# Ensure the project root is in the Python path so that imports work correctly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.runner import run_process_state_and_simulation

def main():
    run_process_state_and_simulation(
        event_log=os.path.join("samples", "Loan_Application_log.csv"),
        bpmn_model=os.path.join("samples", "Loan_Application.bpmn"),
        bpmn_parameters=os.path.join("samples", "Loan_Application.json"),
        start_time="2025-03-07T11:00:00.000+02:00",
        column_mapping='{"CaseId": "CaseId", "Resource": "Resource", "Activity": "Activity", "StartTime": "StartTime", "EndTime": "EndTime", "enabled_time": "EnabledTime"}',
        simulate=True,
        simulation_horizon="2025-05-07T11:00:00.000+02:00",
        total_cases=20,
        sim_stats_csv=os.path.join("samples", "output", "sim_stats.csv"),
        sim_log_csv=os.path.join("samples", "output", "sim_log.csv"),
    )

if __name__ == "__main__":
    main()
