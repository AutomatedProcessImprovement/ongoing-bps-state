import argparse
import datetime
import json
import os
import sys

from prosimos.simulation_engine import run_simulation

def parse_datetime(dt_str):
    """
    Convert an ISO 8601 string (e.g., '2012-03-21T10:11:00.000Z') to a Python datetime.
    Removes 'Z' and replaces with '+00:00' for fromisoformat compatibility.
    """
    if not dt_str:
        return None
    return datetime.datetime.fromisoformat(dt_str.replace("Z", "+00:00"))

def parse_process_state(process_state_path):
    """
    Load the JSON partial-state file, converting relevant fields to datetime.
    """
    if not process_state_path:
        return None
    with open(process_state_path, 'r') as f:
        ps = json.load(f)

    # Convert 'enabled_time' / 'start_time' for each activity
    for case_id, case_data in ps.get('cases', {}).items():
        for act in case_data.get('enabled_activities', []):
            if 'enabled_time' in act and isinstance(act['enabled_time'], str):
                act['enabled_time'] = parse_datetime(act['enabled_time'])
        for act in case_data.get('ongoing_activities', []):
            if 'enabled_time' in act and isinstance(act['enabled_time'], str):
                act['enabled_time'] = parse_datetime(act['enabled_time'])
            if 'start_time' in act and isinstance(act['start_time'], str):
                act['start_time'] = parse_datetime(act['start_time'])
    return ps

def run_short_term_simulation(
    start_date,
    total_cases,
    bpmn_model,
    json_sim_params,
    out_stats_csv_path,
    out_log_csv_path,
    process_state=None,
    simulation_horizon=None
):
    """
    Perform a "short-term" simulation by calling Prosimos's run_simulation(...) once,
    passing partial-state & horizon. This replicates the logic of run_diff_res_simulation
    without referencing testing_scripts.
    """
    start_clock = datetime.datetime.now()

    run_simulation(
        bpmn_path=bpmn_model,
        json_path=json_sim_params,
        total_cases=total_cases,
        stat_out_path=out_stats_csv_path,
        log_out_path=out_log_csv_path,
        starting_at=start_date,
        process_state=process_state,
        simulation_horizon=simulation_horizon
    )

    sim_time = (datetime.datetime.now() - start_clock).total_seconds()
    return sim_time

def run_basic_simulation(
    bpmn_model,
    json_sim_params,
    total_cases,
    out_stats_csv_path,
    out_log_csv_path,
    start_date=None
):
    """
    Perform a standard Prosimos simulation, *without* partial-state or horizon.
    
    :param bpmn_model: Path to BPMN model file
    :param json_sim_params: Path to JSON simulation parameters
    :param total_cases: Number of cases to simulate
    :param out_stats_csv_path: CSV path for simulation stats
    :param out_log_csv_path: CSV path for simulation event log
    :param start_date: Optional simulation start time as a datetime
    
    :return: The elapsed simulation time in seconds (float)
    """
    start_clock = datetime.datetime.now()

    run_simulation(
        bpmn_path=bpmn_model,
        json_path=json_sim_params,
        total_cases=total_cases,
        stat_out_path=out_stats_csv_path,
        log_out_path=out_log_csv_path,
        starting_at=start_date,
        process_state=None,         # No partial-state
        simulation_horizon=None     # No horizon
    )

    sim_time = (datetime.datetime.now() - start_clock).total_seconds()
    return sim_time

def main():
    parser = argparse.ArgumentParser(
        description="Runs Prosimos with optional partial-state and horizon for short-term simulation."
    )
    parser.add_argument("--bpmn_model", required=True, help="Path to the BPMN model file")
    parser.add_argument("--sim_json", required=True, help="Path to the JSON simulation parameters")
    parser.add_argument("--process_state", help="Path to JSON partial-state (ongoing process) file")
    parser.add_argument("--simulation_horizon", help="Time boundary for short-term simulation in ISO format")
    parser.add_argument("--start_time", help="Simulation start datetime in ISO format")
    parser.add_argument("--total_cases", type=int, default=20, help="Number of cases if no partial-state/horizon")
    parser.add_argument("--out_stats_csv", help="CSV for stats (optional)", default="simulation_stats.csv")
    parser.add_argument("--log_csv", help="CSV for event log", default="simulation_log.csv")
    args = parser.parse_args()

    # 1) Parse horizon and start_time if given
    sim_horizon = parse_datetime(args.simulation_horizon)
    start_dt = parse_datetime(args.start_time)

    # 2) Possibly read partial-state
    ps = parse_process_state(args.process_state)

    # 3) If partial-state AND horizon => short-term approach
    if ps and sim_horizon:
        print("Running short-term approach with partial-state and horizon...")
        sim_time = run_short_term_simulation(
            start_date=start_dt or datetime.datetime.now(),
            total_cases=args.total_cases,
            bpmn_model=args.bpmn_model,
            json_sim_params=args.sim_json,
            out_stats_csv_path=args.out_stats_csv,
            out_log_csv_path=args.log_csv,
            process_state=ps,
            simulation_horizon=sim_horizon
        )
        print(f"Short-term simulation took {sim_time} seconds. Output in {args.log_csv}")
    else:
        # 4) Otherwise, standard run_simulation
        print(f"Running standard Prosimos simulation with total_cases={args.total_cases}...")
        start_clock = datetime.datetime.now()
        result = run_simulation(
            bpmn_path=args.bpmn_model,
            json_path=args.sim_json,
            total_cases=args.total_cases,
            stat_out_path=args.out_stats_csv,
            log_out_path=args.log_csv,
            starting_at=start_dt,
            process_state=ps,
            simulation_horizon=sim_horizon
        )
        sim_time = (datetime.datetime.now() - start_clock).total_seconds()
        print(f"Standard simulation took {sim_time} seconds. Log file: {args.log_csv}")
        print("result =", result)

    print("DONE. Exiting gracefully.")

if __name__ == "__main__":
    main()