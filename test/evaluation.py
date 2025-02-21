import os
import json
import math
import pandas as pd
import numpy as np
from scipy.stats import t

# Log-distance-measures imports (only the ones we actually use)
from log_distance_measures.config import (
    AbsoluteTimestampType,
    discretize_to_hour,
    EventLogIDs
)
from log_distance_measures.n_gram_distribution import n_gram_distribution_distance
from log_distance_measures.absolute_event_distribution import absolute_event_distribution_distance
from log_distance_measures.cycle_time_distribution import cycle_time_distribution_distance
from log_distance_measures.relative_event_distribution import relative_event_distribution_distance
from log_distance_measures.circadian_event_distribution import circadian_event_distribution_distance
from log_distance_measures.circadian_workforce_distribution import circadian_workforce_distribution_distance
from log_distance_measures.case_arrival_distribution import case_arrival_distribution_distance
from log_distance_measures.remaining_time_distribution import remaining_time_distribution_distance

from test.helper import (
    run_simulation_with_retries,
    read_event_log,
    trim_events_to_eval_window,
    filter_ongoing_cases,
    filter_complete_cases
)

# Simulation-libs
from src.runner import run_process_state_and_simulation
from src.process_state_prosimos_run import run_basic_simulation

###############################################################################
# Metric Computation (Event / Ongoing / Complete)
###############################################################################
def compute_custom_metrics(
    A_event_filter_ref: pd.DataFrame,
    A_ongoing_ref: pd.DataFrame,
    A_complete_ref: pd.DataFrame,
    G_event_filter: pd.DataFrame,
    G_ongoing: pd.DataFrame,
    G_complete: pd.DataFrame,
    ongoing_reference_point: pd.Timestamp,
    verbose=True
) -> dict:
    """
    Computes metrics for the three filtering approaches.
    
    Event Filter Metrics:
      - n_gram: N-gram distribution distance (3-grams)
      - absolute_event: Absolute event distribution distance
      - circadian_event: Circadian Event Distribution
      - circadian_workflow: Circadian Workflow Distribution

    Ongoing Filter Metrics:
      - RTD: Remaining Time Distribution Distance

    Complete Filter Metrics:
      - RED: Relative Event Distribution Distance
      - cycle_time: Cycle Time Distribution Distance
      - case_arrival_rate: Case Arrival Rate Distance

    Returns a dict:
    {
      "event_filter": { "n_gram": ..., "absolute_event": ..., "circadian_event": ..., "circadian_workflow": ... },
      "ongoing_filter": { "RTD": ... },
      "complete_filter": { "RED": ..., "cycle_time": ..., "case_arrival_rate": ... }
    }
    """
    custom_ids = EventLogIDs(
        case="case_id",
        activity="activity",
        start_time="start_time",
        end_time="end_time",
        resource="resource"
    )

    results = {
        "event_filter": {},
        "ongoing_filter": {},
        "complete_filter": {}
    }

    # ----------------- EVENT FILTER -----------------
    try:
        results["event_filter"]["n_gram"] = n_gram_distribution_distance(
            A_event_filter_ref, custom_ids, G_event_filter, custom_ids, n=3
        )
    except Exception as e:
        if verbose:
            print("[compute_custom_metrics] Error computing n_gram for event_filter:", e)
        results["event_filter"]["n_gram"] = None

    try:
        results["event_filter"]["absolute_event"] = absolute_event_distribution_distance(
            A_event_filter_ref,
            custom_ids,
            G_event_filter,
            custom_ids,
            discretize_type=AbsoluteTimestampType.START,
            discretize_event=discretize_to_hour
        )
    except Exception as e:
        if verbose:
            print("[compute_custom_metrics] Error computing absolute_event for event_filter:", e)
        results["event_filter"]["absolute_event"] = None

    try:
        results["event_filter"]["circadian_event"] = circadian_event_distribution_distance(
            A_event_filter_ref, custom_ids, G_event_filter, custom_ids
        )
    except Exception as e:
        if verbose:
            print("[compute_custom_metrics] Error computing circadian_event for event_filter:", e)
        results["event_filter"]["circadian_event"] = None

    try:
        results["event_filter"]["circadian_workflow"] = circadian_workforce_distribution_distance(
            A_event_filter_ref, custom_ids, G_event_filter, custom_ids
        )
    except Exception as e:
        if verbose:
            print("[compute_custom_metrics] Error computing circadian_workflow for event_filter:", e)
        results["event_filter"]["circadian_workflow"] = None

    # ----------------- ONGOING FILTER -----------------
    try:
        results["ongoing_filter"]["RTD"] = remaining_time_distribution_distance(
            A_ongoing_ref, custom_ids,
            G_ongoing, custom_ids,
            reference_point=ongoing_reference_point,  # evaluation start/cut-off time
            bin_size=pd.Timedelta(hours=1)
        )
    except Exception as e:
        if verbose:
            print("[compute_custom_metrics] Error computing RTD for ongoing_filter:", e)
        results["ongoing_filter"]["RTD"] = None

    # ----------------- COMPLETE FILTER -----------------
    try:
        results["complete_filter"]["RED"] = relative_event_distribution_distance(
            A_complete_ref,
            custom_ids,
            G_complete,
            custom_ids,
            discretize_type=AbsoluteTimestampType.BOTH,
            discretize_event=discretize_to_hour
        )
    except Exception as e:
        if verbose:
            print("[compute_custom_metrics] Error computing RED for complete_filter:", e)
        results["complete_filter"]["RED"] = None

    try:
        results["complete_filter"]["cycle_time"] = cycle_time_distribution_distance(
            A_complete_ref, custom_ids, G_complete, custom_ids, bin_size=pd.Timedelta(hours=1)
        )
    except Exception as e:
        if verbose:
            print("[compute_custom_metrics] Error computing cycle_time for complete_filter:", e)
        results["complete_filter"]["cycle_time"] = None

    try:
        results["complete_filter"]["case_arrival_rate"] = case_arrival_distribution_distance(
            A_complete_ref, custom_ids, G_complete, custom_ids, discretize_event=discretize_to_hour
        )
    except Exception as e:
        if verbose:
            print("[compute_custom_metrics] Error computing case_arrival_rate for complete_filter:", e)
        results["complete_filter"]["case_arrival_rate"] = None

    return results



###############################################################################
# Helpers for Partial-State approach (Parsing partial_state.json)
###############################################################################
def parse_partial_state_json(json_path: str) -> set:
    """
    Reads the partial-state JSON, which has a structure like:
    {
      "cases": {
         "9": { ... },
         "10": { ... },
         ...
      }
    }
    Returns the set of case_ids that appear in partial-state (i.e. started before cutoff).
    If the file doesn't exist or is invalid, returns an empty set.
    """
    if not os.path.isfile(json_path):
        return set()  # no file -> no partial-state
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "cases" not in data:
            return set()
        return set(data["cases"].keys())  # keys are strings
    except Exception as e:
        print(f"[parse_partial_state_json] Error reading {json_path}: {e}")
        return set()


def build_partial_state_subsets(
    glog_df: pd.DataFrame,
    partial_ids: set,
    cutoff: pd.Timestamp,
    evaluation_end: pd.Timestamp
) -> tuple:
    """
    Builds the three subsets for the partial-state approach:

    1) G_event_filter: events overlapping [cutoff, evaluation_end] (using trim_events_to_eval_window).
    2) G_ongoing: events from cases in partial_ids, with event times clipped to [cutoff, evaluation_end].
       Additionally, for each case in partial_ids, an artificial event is added with start_time = end_time = cutoff.
    3) G_complete: events from cases *not* in partial_ids, where the earliest event starts 
       at or after cutoff and strictly before evaluation_end.
    """
    # (1) Event filter.
    G_event_filter = trim_events_to_eval_window(glog_df, cutoff, evaluation_end)

    # (2) Ongoing: keep only events from cases in partial_ids.
    G_ongoing = glog_df[glog_df["case_id"].astype(str).isin(partial_ids)].copy()
    # Clip event times to [cutoff, evaluation_end].
    G_ongoing.loc[G_ongoing["start_time"] < cutoff, "start_time"] = cutoff
    # G_ongoing.loc[G_ongoing["end_time"] > evaluation_end, "end_time"] = evaluation_end

    # (3) Complete: exclude partial_ids and keep only cases whose earliest start is between cutoff (inclusive)
    # and evaluation_end (exclusive).
    remainder_df = glog_df[~glog_df["case_id"].astype(str).isin(partial_ids)].copy()
    group_min_start = remainder_df.groupby("case_id")["start_time"].transform("min")
    G_complete = remainder_df[(group_min_start >= cutoff) & (group_min_start < evaluation_end)]

    return G_event_filter, G_ongoing, G_complete





###############################################################################
# Evaluate Partial-State Simulation
###############################################################################
def evaluate_partial_state_simulation(
    run_output_dir: str,
    # Simulation inputs:
    event_log: str,
    bpmn_model: str,
    bpmn_parameters: str,
    start_time: pd.Timestamp,
    simulation_horizon: pd.Timestamp,
    total_cases: int,
    # Already precomputed reference subsets (A):
    A_event_filter_ref: pd.DataFrame,
    A_ongoing_ref: pd.DataFrame,
    A_complete_ref: pd.DataFrame,
    # Extra params:
    evaluation_end: pd.Timestamp,
    column_mapping: dict,
    required_columns: list,
    simulate: bool = True,
    verbose: bool = True
) -> dict:
    """
    1) Run partial-state approach simulation
    2) Parse partial_state.json to find ongoing IDs
    3) Build G_event_filter, G_ongoing, G_complete using partial_state info
    4) compute_custom_metrics(A references, G subsets)
    5) Return the result dict
    """
    sim_stats_csv = os.path.join(run_output_dir, "partial_sim_stats.csv")
    sim_log_csv = os.path.join(run_output_dir, "partial_sim_log.csv")

    # Step 1: Run simulation
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

    # Step 2: Load GLog
    inv_mapping = {v: k for k, v in column_mapping.items()}
    if verbose:
        print("=== [Process-State] Reading partial-state simulation log ===")

    glog_df = read_event_log(
        csv_path=sim_log_csv,
        rename_map=inv_mapping,
        required_columns=None,
        verbose=verbose
    )

    # Step 2B: Parse partial_state.json to find ongoing IDs
    partial_ids = parse_partial_state_json("output.json")
    if verbose:
        print(f"[Process-State] Found {len(partial_ids)} ongoing partial-state case IDs.")

    # Step 3: Build generated subsets
    G_event_filter, G_ongoing, G_complete = build_partial_state_subsets(
        glog_df,
        partial_ids,
        cutoff=start_time,
        evaluation_end=evaluation_end
    )

    # Optionally save these G subsets
    G_event_filter.to_csv(os.path.join(run_output_dir, "PS_event_G.csv"), index=False)
    G_ongoing.to_csv(os.path.join(run_output_dir, "PS_ongoing_G.csv"), index=False)
    G_complete.to_csv(os.path.join(run_output_dir, "PS_complete_G.csv"), index=False)

    # Step 4: Compute metrics
    result_metrics = compute_custom_metrics(
        A_event_filter_ref, A_ongoing_ref, A_complete_ref,
        G_event_filter, G_ongoing, G_complete, start_time,
        verbose=verbose
    )
    return result_metrics


###############################################################################
# Evaluate Warm-Up Simulation
###############################################################################
def evaluate_warmup_simulation(
    run_output_dir: str,
    bpmn_model: str,
    bpmn_parameters: str,
    warmup_start: pd.Timestamp,
    simulation_cut: pd.Timestamp,
    evaluation_end: pd.Timestamp,
    total_cases: int,
    # Already precomputed reference subsets (A):
    A_event_filter_ref: pd.DataFrame,
    A_ongoing_ref: pd.DataFrame,
    A_complete_ref: pd.DataFrame,
    # Column rename info:
    rename_map: dict,
    required_columns: list,
    simulate: bool = True,
    verbose: bool = True
) -> dict:
    """
    Warm-up approach:
    1) Run simulation
    2) Build G_event_filter, G_ongoing, G_complete using normal helper filters
    3) compute_custom_metrics(A references, G subsets)
    """
    sim_stats_csv = os.path.join(run_output_dir, "warmup_sim_stats.csv")
    sim_log_csv = os.path.join(run_output_dir, "warmup_sim_log.csv")

    # Step 1: Simulation
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

    # Step 2: Load GLog
    if verbose:
        print("=== [Warm-up] Reading warm-up simulation log ===")
    sim_df = read_event_log(
        csv_path=sim_log_csv,
        rename_map=rename_map,
        required_columns=None,
        verbose=verbose
    )

    # Step 3: Build generated subsets (using normal filters)
    #  event_filter: [simulation_cut, evaluation_end]
    G_event_filter = trim_events_to_eval_window(sim_df, simulation_cut, evaluation_end)

    #  ongoing_filter: "ongoing" if min(start) < simulation_cut < max(end),
    #                  only keep events after simulation_cut
    G_ongoing = filter_ongoing_cases(sim_df, simulation_cut, evaluation_end)

    #  complete_filter: min(start) >= simulation_cut
    G_complete = filter_complete_cases(sim_df, simulation_cut, evaluation_end)

    # Optionally save these G subsets
    G_event_filter.to_csv(os.path.join(run_output_dir, "WU_event_G.csv"), index=False)
    G_ongoing.to_csv(os.path.join(run_output_dir, "WU_ongoing_G.csv"), index=False)
    G_complete.to_csv(os.path.join(run_output_dir, "WU_complete_G.csv"), index=False)

    # Step 4: Compute metrics
    result_metrics = compute_custom_metrics(
        A_event_filter_ref, A_ongoing_ref, A_complete_ref,
        G_event_filter, G_ongoing, G_complete, simulation_cut,
        verbose=verbose
    )
    return result_metrics


###############################################################################
# Aggregation and Comparison
###############################################################################
def compute_mean_conf_interval(data: list, confidence: float = 0.95):
    sample_mean = np.mean(data)
    sample_std = np.std(data, ddof=1)
    df = len(data) - 1
    t_value = t.ppf(1 - (1 - confidence) / 2, df)
    std_error = sample_std / np.sqrt(len(data))
    conf_interval = t_value * std_error
    return sample_mean, conf_interval

def aggregate_metrics(all_runs: list) -> dict:
    """
    Each run is a dict:
      {
        "event_filter": { ... },
        "ongoing_filter": { ... },
        "complete_filter": { ... }
      }
    We'll compute mean, std, 95% CI across runs for each sub-filter & metric.
    """
    if not all_runs:
        return {}

    aggregated = {}
    sub_filters = set()
    for run_res in all_runs:
        sub_filters.update(run_res.keys())  # gather sub-filter keys

    for sf in sub_filters:
        aggregated[sf] = {}
        metric_keys = set()
        for run_res in all_runs:
            if sf in run_res:
                metric_keys.update(run_res[sf].keys())

        for mk in metric_keys:
            vals = []
            for run_res in all_runs:
                val = run_res.get(sf, {}).get(mk, None)
                if val is not None and not pd.isnull(val):
                    vals.append(val)

            if not vals:
                aggregated[sf][mk] = {
                    "mean": None,
                    "interval": None,
                    "individual_runs": []
                }
                continue

            mean_val, conf_int = compute_mean_conf_interval(vals)

            aggregated[sf][mk] = {
                "mean": mean_val,
                "interval": conf_int,
                "individual_runs": vals
            }

    return aggregated


def compare_results(proc_agg: dict, warmup_agg: dict) -> dict:
    """
    Compare aggregated results between approaches, for each sub-filter and metric.
    Produces a structure like:
    {
      "event_filter": {
        "n_gram": {"process_state_mean": X, "warmup_mean": Y, "better_approach": ...},
        ...
      },
      ...
    }
    """
    comparison = {}
    all_sub_filters = set(proc_agg.keys()).union(warmup_agg.keys())

    for sf in all_sub_filters:
        comparison[sf] = {}
        proc_metrics = proc_agg.get(sf, {})
        warm_metrics = warmup_agg.get(sf, {})

        all_metrics = set(proc_metrics.keys()).union(warm_metrics.keys())
        for mk in all_metrics:
            p_data = proc_metrics.get(mk, {})
            w_data = warm_metrics.get(mk, {})

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

            comparison[sf][mk] = {
                "process_state_mean": proc_mean,
                "warmup_mean": warmup_mean,
                "better_approach": better
            }

    return comparison
