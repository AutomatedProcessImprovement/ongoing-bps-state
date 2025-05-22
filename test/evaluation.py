# evaluation.py
from __future__ import annotations
import json, os, shutil, pandas as pd, numpy as np
from dataclasses import dataclass
from typing import Callable
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from helper import (
    read_event_log, run_with_retries, trim_events, split_into_subsets,
    avg_remaining_time, _parse_partial_state_json, _build_ps_subsets,
    _avg_events_per_ongoing_case, _avg_events_per_case_diff
)
from log_distance_measures.config import (
    EventLogIDs, AbsoluteTimestampType, discretize_to_hour,
)
from log_distance_measures.n_gram_distribution import n_gram_distribution_distance
from log_distance_measures.absolute_event_distribution import absolute_event_distribution_distance
from log_distance_measures.relative_event_distribution import relative_event_distribution_distance
from log_distance_measures.cycle_time_distribution import cycle_time_distribution_distance
from log_distance_measures.circadian_event_distribution import circadian_event_distribution_distance
from log_distance_measures.circadian_workforce_distribution import circadian_workforce_distribution_distance
from log_distance_measures.case_arrival_distribution import case_arrival_distribution_distance
from log_distance_measures.remaining_time_distribution import remaining_time_distribution_distance

# ------------------------------------------------------------------------- #
#                             Public interface                              #
# ------------------------------------------------------------------------- #

@dataclass(slots=True)
class SimulationIO:
    """Paths produced/consumed by a single simulation run (any flavour)."""
    log_csv:  str
    stats_csv: str
    out_dir:   str

def evaluate(
    flavour: str,                        # "process_state" | "warmup" | "warmup2"
    io: SimulationIO,
    sim_runner: Callable[[SimulationIO], None],
    *,
    # evaluation window
    cut: pd.Timestamp,
    end: pd.Timestamp,
    # reference logs (already window-split)
    A_event: pd.DataFrame,
    A_ongoing: pd.DataFrame,
    A_complete: pd.DataFrame,
    # kwargs for the particular runner
    runner_kwargs: dict,
    verbose: bool = True,
) -> dict:
    """
    Generic wrapper: run simulation *flavour*, build G subsets, compute metrics.
    """
    if verbose:
        print(f"=== [{flavour}] Simulation ===")
    run_with_retries(sim_runner, {"io": io, **runner_kwargs})

    # ── NEW: copy the partial-state file for later inspection ───────────
    if flavour == "process_state" and os.path.exists("output.json"):
        shutil.copy2("output.json",
                     os.path.join(io.out_dir, "process_state.json"))

    if verbose:
        print(f"=== [{flavour}] Loading simulated log ===")
    G_raw = read_event_log(io.log_csv, rename=runner_kwargs.get("rename_map"))

    # choose the correct subset logic ----------------------------------
    if flavour == "process_state":
        partial_ids = _parse_partial_state_json(io.out_dir)
        G_event, G_ongoing, G_complete = _build_ps_subsets(
            G_raw, partial_ids, cut, end
        )
    elif flavour == "warmup2":
        # -- A) how many ongoing cases does the *reference* window have?
        ref_ongoing_count = A_ongoing["case_id"].nunique()

        # -- B) compute arrival time of every simulated case
        G_raw["arrival_time"] = (
            G_raw.groupby("case_id")["start_time"].transform("min")
        )

        # helper: active cases at time t
        def _num_active(df: pd.DataFrame, t: pd.Timestamp) -> int:
            first_last = df.groupby("case_id").agg(
                first=("start_time", "min"),
                last=("end_time", "max"),
            )
            return int(((first_last["first"] <= t) & (first_last["last"] > t)).sum())

        # check each arrival (<= cut) for a match
        arrivals = (
            G_raw.loc[G_raw["arrival_time"] <= cut, "arrival_time"]
            .drop_duplicates()
            .sort_values()
        )
        shift_by = None
        for t in arrivals:
            if _num_active(G_raw, t) == ref_ongoing_count:
                shift_by = cut - t
                break

        # -- C) shift the entire log if we found such a tine
        if shift_by is not None and shift_by != pd.Timedelta(0):
            if verbose:
                print(f"[warmup2] shifting log by {shift_by}")
            shifted_df = G_raw.copy()
            for col in ("enable_time", "start_time", "end_time"):
                if col in shifted_df.columns:
                    shifted_df[col] = shifted_df[col] + shift_by
            G_base = shifted_df
        else:
            if verbose:
                print("[warmup2] no matching active-case count found; no shift.")
            G_base = G_raw
        G_event, G_ongoing, G_complete = split_into_subsets(G_base, cut, end)
    else:
        G_event, G_ongoing, G_complete = split_into_subsets(G_raw, cut, end)

    _dump(G_event, io.out_dir, f"{flavour}_event_G.csv")
    _dump(G_ongoing, io.out_dir, f"{flavour}_ongoing_G.csv")
    _dump(G_complete, io.out_dir, f"{flavour}_complete_G.csv")

    return _metrics(
        A_event, A_ongoing, A_complete,
        G_event, G_ongoing, G_complete, cut
    )

# ------------------------------------------------------------------------- #
#                               internals                                   #
# ------------------------------------------------------------------------- #

def _dump(df: pd.DataFrame, folder: str, name: str) -> None:
    os.makedirs(folder, exist_ok=True)
    df.to_csv(os.path.join(folder, name), index=False)

def _metrics(A_event, A_ongoing, A_complete,
             G_event, G_ongoing, G_complete,
             cutoff) -> dict:
    ids = EventLogIDs("case_id", "activity", "start_time", "end_time", "resource")

    def safe(fn, *a, **k):
        try:  return fn(*a, **k)
        except Exception: return None

    out = {
        "event_filter": {
            "n_gram":            safe(n_gram_distribution_distance, A_event, ids, G_event, ids, n=3),
            "absolute_event":    safe(absolute_event_distribution_distance, A_event, ids, G_event, ids,
                                      discretize_type=AbsoluteTimestampType.START, discretize_event=discretize_to_hour),
            "circadian_event":   safe(circadian_event_distribution_distance, A_event, ids, G_event, ids),
            "circadian_workflow":safe(circadian_workforce_distribution_distance, A_event, ids, G_event, ids),
        },
        "ongoing_filter": {
            "RTD":               safe(remaining_time_distribution_distance, A_ongoing, ids, G_ongoing, ids,
                                      reference_point=cutoff, bin_size=pd.Timedelta(hours=1)),
            "avg_remaining_diff":safe(_avg_rem_diff, A_ongoing, G_ongoing, ids, cutoff),
            "ongoing_cases_count":G_ongoing["case_id"].nunique(),
            "ongoing_cases_count_diff":abs(G_ongoing["case_id"].nunique() - A_ongoing["case_id"].nunique()),
            "avg_events_per_case":    _avg_events_per_ongoing_case(G_ongoing),
            "avg_events_per_case_diff": _avg_events_per_case_diff(A_ongoing, G_ongoing),

        },
        "complete_filter": {
            "RED":               safe(relative_event_distribution_distance, A_complete, ids, G_complete, ids,
                                      discretize_type=AbsoluteTimestampType.BOTH, discretize_event=discretize_to_hour),
            "cycle_time":        safe(cycle_time_distribution_distance, A_complete, ids, G_complete, ids,
                                      bin_size=pd.Timedelta(hours=1)),
            "case_arrival_rate": safe(case_arrival_distribution_distance, A_complete, ids, G_complete, ids,
                                      discretize_event=discretize_to_hour),
        }
    }
    return out

def _avg_rem_diff(A, G, ids, cut):
    ref = avg_remaining_time(A, cut, ids)
    sim = avg_remaining_time(G, cut, ids)
    return abs(ref - sim) / ref if ref else None

# ---------- aggregation helpers ---------------------------

def _mean_ci(vals, conf=0.95):
    if not vals: return (None, None)
    mean = np.mean(vals); std = np.std(vals, ddof=1)
    ci = std * 1.96 / np.sqrt(len(vals))           # ~95 % for n>30 (t-approx)
    return float(mean), float(ci)

def aggregate(runs: list[dict]) -> dict:
    res = {}
    for sf in ("event_filter","ongoing_filter","complete_filter"):
        res[sf] = {}
        keys = {k for run in runs for k in run.get(sf, {})}
        for k in keys:
            vals = [run[sf].get(k) for run in runs if run[sf].get(k) is not None]
            mean, ci = _mean_ci(vals)
            res[sf][k] = {"mean": mean, "ci": ci}
    return res

def compare(a: dict, b: dict, names=("A","B")) -> dict:
    out = {}
    for sf in a:
        out[sf] = {}
        for k in a[sf]:
            x, y = a[sf][k]["mean"], b[sf][k]["mean"]
            if x is None or y is None:
                better = None
            else:
                better = names[0] if x < y else names[1] if y < x else "tie"
            out[sf][k] = {names[0]: x, names[1]: y, "better": better}
    return out
