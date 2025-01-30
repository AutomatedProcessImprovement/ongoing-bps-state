# main.py

import argparse
import sys
from src.runner import run_process_state_and_simulation

def main():
    parser = argparse.ArgumentParser(description="Process state + optional simulation (calls 'runner.py')")
    parser.add_argument("event_log", help="Path to event log CSV")
    parser.add_argument("bpmn_model", help="Path to BPMN model")
    parser.add_argument("bpmn_parameters", help="Path to BPMN JSON sim params")

    parser.add_argument("--start_time", help="Optional start time (ISO)")
    parser.add_argument("--column_mapping", help="JSON string for column mapping")
    parser.add_argument("--simulate", action="store_true", help="If set, run short-term simulation")
    parser.add_argument("--simulation_horizon", help="Time boundary for short-term simulation in ISO")
    parser.add_argument("--total_cases", type=int, default=20, help="Number of cases if partial-state/horizon not override")
    parser.add_argument("--sim_stats_csv", default="simulation_stats.csv", help="Path for simulation stats CSV")
    parser.add_argument("--sim_log_csv", default="simulation_log.csv", help="Path for simulation event log CSV")

    args = parser.parse_args()

    run_process_state_and_simulation(
        event_log=args.event_log,
        bpmn_model=args.bpmn_model,
        bpmn_parameters=args.bpmn_parameters,
        start_time=args.start_time,
        column_mapping=args.column_mapping,
        simulate=args.simulate,
        simulation_horizon=args.simulation_horizon,
        total_cases=args.total_cases,
        sim_stats_csv=args.sim_stats_csv,
        sim_log_csv=args.sim_log_csv
    )

if __name__ == "__main__":
    main()
