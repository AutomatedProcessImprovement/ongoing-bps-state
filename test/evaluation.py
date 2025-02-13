import pandas as pd
import math
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Log-distance-measures imports
from log_distance_measures.config import (
    AbsoluteTimestampType,
    discretize_to_hour,
    EventLogIDs
)
from log_distance_measures.control_flow_log_distance import control_flow_log_distance
from log_distance_measures.n_gram_distribution import n_gram_distribution_distance
from log_distance_measures.absolute_event_distribution import absolute_event_distribution_distance
from log_distance_measures.case_arrival_distribution import case_arrival_distribution_distance
from log_distance_measures.cycle_time_distribution import cycle_time_distribution_distance
from log_distance_measures.circadian_workforce_distribution import circadian_workforce_distribution_distance
from log_distance_measures.relative_event_distribution import relative_event_distribution_distance

from test.helper import (
    basic_log_stats,
    filter_cases_by_eval_window,
    trim_events_to_eval_window,
    run_simulation_with_retries,
    read_event_log
)

# Simulation-libs (adjust path as needed)
from src.runner import run_process_state_and_simulation
from src.process_state_prosimos_run import run_basic_simulation


def compute_log_distances(
    A_event: pd.DataFrame, G_event: pd.DataFrame, A_case: pd.DataFrame, G_case: pd.DataFrame, verbose=True
) -> dict:
    """
    Computes the set of log distances between the reference (A_event, A_case)
    and the generated (G_event, G_case) logs.
    """
    custom_ids = EventLogIDs(
        case="case_id",
        activity="activity",
        start_time="start_time",
        end_time="end_time",
        resource="resource"
    )

    distances = {}
    try:
        distances["control_flow_log_distance"] = control_flow_log_distance(A_event, custom_ids, G_event, custom_ids)
        distances["n_gram_distribution_distance"] = n_gram_distribution_distance(
            A_event, custom_ids, G_event, custom_ids, n=3
        )
        distances["absolute_event_distribution_distance"] = absolute_event_distribution_distance(
            A_event,
            custom_ids,
            G_event,
            custom_ids,
            discretize_type=AbsoluteTimestampType.START,
            discretize_event=discretize_to_hour
        )
        distances["case_arrival_distribution_distance"] = case_arrival_distribution_distance(
            A_case, custom_ids, G_case, custom_ids, discretize_event=discretize_to_hour
        )
        distances["cycle_time_distribution_distance"] = cycle_time_distribution_distance(
            A_case, custom_ids, G_case, custom_ids, bin_size=pd.Timedelta(hours=1)
        )
        distances["circadian_workforce_distribution_distance"] = circadian_workforce_distribution_distance(
            A_case, custom_ids, G_case, custom_ids
        )
        distances["relative_event_distribution_distance"] = relative_event_distribution_distance(
            A_event,
            custom_ids,
            G_event,
            custom_ids,
            discretize_type=AbsoluteTimestampType.BOTH,
            discretize_event=discretize_to_hour
        )
    except Exception as e:
        raise RuntimeError(f"Error computing distances: {e}") from e

    return distances


def evaluate_partial_state_simulation(
    run_output_dir: str,
    # Simulation inputs:
    event_log: str,
    bpmn_model: str,
    bpmn_parameters: str,
    start_time: pd.Timestamp,
    simulation_horizon: pd.Timestamp,
    total_cases: int,
    # Already preprocessed reference data:
    A_case_ref: pd.DataFrame,
    A_event_ref: pd.DataFrame,
    evaluation_end: pd.Timestamp,
    # Column rename info:
    column_mapping: dict,
    required_columns: list,
    # Flags:
    simulate: bool = True,
    verbose: bool = True,
):
    """
    Runs partial-state approach simulation, loads GLog, and computes 
    distances vs. the reference subsets.
    Saves the intermediate logs in the run_output_dir.
    """
    sim_stats_csv = os.path.join(run_output_dir, "partial_sim_stats.csv")
    sim_log_csv = os.path.join(run_output_dir, "partial_sim_log.csv")

    if simulate:
        if verbose:
            print("=== [Process-State] Running simulation ===")
        sim_func_kwargs = {
            "event_log": event_log,
            "bpmn_model": bpmn_model,
            "bpmn_parameters": bpmn_parameters,
            "start_time": str(start_time.isoformat()),
            "column_mapping": json.dumps(column_mapping),
            "simulate": True,
            "simulation_horizon": str(simulation_horizon.isoformat()),
            "total_cases": total_cases,
            "sim_stats_csv": sim_stats_csv,
            "sim_log_csv": sim_log_csv,
        }
        run_simulation_with_retries(run_process_state_and_simulation, sim_func_kwargs, max_attempts=3, verbose=verbose)
    else:
        if verbose:
            print("Skipping partial-state simulation (simulate=False).")

    inv_mapping = {v: k for k, v in column_mapping.items()}

    # --- Load GLog ---
    if verbose:
        print("=== [Process-State] Loading generated simulation log ===")
    glog_df = read_event_log(
        csv_path=sim_log_csv,
        rename_map=inv_mapping,
        required_columns=None,
        verbose=verbose
    )

    # Reference subsets for partial-state approach
    A_case = A_case_ref.copy()
    A_event = A_event_ref.copy()

    # Filter/trim the generated log
    G_all = glog_df.copy()
    eval_start_dt = start_time
    eval_end_dt = evaluation_end

    G_case = filter_cases_by_eval_window(G_all, eval_start_dt, eval_end_dt)
    G_event = trim_events_to_eval_window(G_all, eval_start_dt, eval_end_dt)

    # Save logs
    # A_case.to_csv(os.path.join(run_output_dir, "PS_A_case.csv"), index=False)
    # A_event.to_csv(os.path.join(run_output_dir, "PS_A_event.csv"), index=False)
    G_case.to_csv(os.path.join(run_output_dir, "PS_G_case.csv"), index=False)
    G_event.to_csv(os.path.join(run_output_dir, "PS_G_event.csv"), index=False)

    # Compute stats
    statsA = basic_log_stats(A_event)
    statsG = basic_log_stats(G_event)

    # Compute log distances
    distances = compute_log_distances(A_event, G_event, A_case, G_case, verbose=verbose)

    result_dict = {
        "ALog_stats_event": statsA,
        "GLog_stats_event": statsG,
        "distances": distances,
    }
    if verbose:
        print("[Process-State] Evaluation result:", json.dumps(result_dict, indent=4))
    return result_dict


def evaluate_warmup_simulation(
    run_output_dir: str,
    # Simulation inputs:
    bpmn_model: str,
    bpmn_parameters: str,
    warmup_start: pd.Timestamp,
    simulation_cut: pd.Timestamp,
    evaluation_end: pd.Timestamp,
    total_cases: int,
    # Already preprocessed reference data:
    A_case_ref: pd.DataFrame,
    A_event_ref: pd.DataFrame,
    # Column rename info:
    rename_map: dict,
    required_columns: list,
    # Flags:
    simulate: bool = True,
    verbose: bool = True,
):
    """
    Runs warm-up approach from warmup_start, loads the resulting log,
    discards events before simulation_cut, then compares to reference
    subsets in [simulation_cut, evaluation_end].
    """
    sim_stats_csv = os.path.join(run_output_dir, "warmup_sim_stats.csv")
    sim_log_csv = os.path.join(run_output_dir, "warmup_sim_log.csv")

    if simulate:
        if verbose:
            print("=== [Warm-up] Running simulation ===")
        sim_func_kwargs = {
            "bpmn_model": bpmn_model,
            "json_sim_params": bpmn_parameters,
            "total_cases": total_cases,
            "out_stats_csv_path": sim_stats_csv,
            "out_log_csv_path": sim_log_csv,
            "start_date": str(warmup_start.isoformat())
        }
        run_simulation_with_retries(run_basic_simulation, sim_func_kwargs, max_attempts=3, verbose=verbose)
    else:
        if verbose:
            print("Skipping warm-up simulation (simulate=False).")

    # --- Load simulation results ---
    sim_df = read_event_log(
        csv_path=sim_log_csv,
        rename_map=rename_map,
        required_columns=None,
        verbose=verbose
    )

    eval_start_dt = simulation_cut
    eval_end_dt = evaluation_end

    # Filter/trim the simulation log
    sim_case = filter_cases_by_eval_window(sim_df, eval_start_dt, eval_end_dt)
    sim_event = trim_events_to_eval_window(sim_df, eval_start_dt, eval_end_dt)
    statsSim = basic_log_stats(sim_event)

    # Reference subsets
    A_case = A_case_ref.copy()
    A_event = A_event_ref.copy()

    # Save intermediate logs
    # A_case.to_csv(os.path.join(run_output_dir, "WU_A_case.csv"), index=False)
    # A_event.to_csv(os.path.join(run_output_dir, "WU_A_event.csv"), index=False)
    sim_case.to_csv(os.path.join(run_output_dir, "WU_G_case.csv"), index=False)
    sim_event.to_csv(os.path.join(run_output_dir, "WU_G_event.csv"), index=False)

    # Compute distances
    distances = compute_log_distances(A_event, sim_event, A_case, sim_case, verbose=verbose)

    result_dict = {
        "SimStats": statsSim,
        "distances": distances,
    }
    if verbose:
        print("[Warm-up] Evaluation result:", json.dumps(result_dict, indent=4))
    return result_dict


def aggregate_metrics(all_runs: list) -> dict:
    """
    Aggregates metrics across runs (mean, std, 95% CI).
    Each element of `all_runs` is assumed to have a dict {metricA: valA, metricB: valB, ...}.
    """
    agg = {}
    if not all_runs:
        return agg

    # Collect set of metric keys
    metric_keys = set()
    for run_result in all_runs:
        for k in run_result.keys():
            metric_keys.add(k)
    metric_keys = list(metric_keys)

    for metric in metric_keys:
        run_values = [run_result.get(metric, float("nan")) for run_result in all_runs]
        run_values = [v for v in run_values if v is not None and not pd.isnull(v)]
        if not run_values:
            agg[metric] = {
                "mean": None,
                "std": None,
                "confidence_interval": [None, None],
                "individual_runs": []
            }
            continue

        mean_val = sum(run_values) / len(run_values)
        std_val = math.sqrt(sum((x - mean_val) ** 2 for x in run_values) / len(run_values))
        se = std_val / math.sqrt(len(run_values))
        ci_lower = mean_val - 1.96 * se
        ci_upper = mean_val + 1.96 * se

        agg[metric] = {
            "mean": mean_val,
            "std": std_val,
            "confidence_interval": [ci_lower, ci_upper],
            "individual_runs": run_values
        }

    return agg


def compare_results(proc_agg: dict, warmup_agg: dict) -> dict:
    """
    Compare aggregated results between approaches.
    """
    comparison = {}
    metric_keys = set(proc_agg.keys()).union(set(warmup_agg.keys()))
    for metric in metric_keys:
        p_data = proc_agg.get(metric, {})
        w_data = warmup_agg.get(metric, {})
        proc_mean = p_data.get("mean", None)
        warmup_mean = w_data.get("mean", None)

        # Decide "better_approach" based on lower mean distance
        if proc_mean is None or warmup_mean is None:
            better = None
        else:
            if proc_mean < warmup_mean:
                better = "process_state"
            elif warmup_mean < proc_mean:
                better = "warmup"
            else:
                better = "tie"

        comparison[metric] = {
            "process_state_mean": proc_mean,
            "warmup_mean": warmup_mean,
            "better_approach": better
        }
    return comparison
