# ────────────────────────────────────────────────────────────────────
# test/clustered_short_term_simulation.py
# ────────────────────────────────────────────────────────────────────
"""
Confidence-estimation & clustering pipeline on top of the existing
short-horizon simulation approach.

High-level steps
----------------
1. Take an event log (from DATASETS in evaluate_with_existing_alog.py).
2. Split it 70/30 into train/test (by complete cases, no overlaps).
   - Store these logs in an `auto/` folder next to the original CSV.
3. On TRAIN:
   - Compute 95th percentile of case durations -> simulation horizon H.
   - Take all activity end-times (or case end-times) as candidate cut timestamps.
   - Drop timestamps within the first/last H of the log.
   - For each remaining cut timestamp t:
       * Build feature vector at t (WIP, λ, resource availability,
         per-activity WIP, activity state vector).
       * Run process-state simulation with horizon H (10 replications).
       * Compute aggregated errors (RTD on ongoing, cycle-time on completes).
   - Save:
       * Raw simulation logs under `outputs_confidence/<dataset>/<run_id>/train/`.
       * Per-timestamp features + aggregated errors in `train_samples.csv`.
4. On TEST: same as 3., but using test log and storing to `test_samples.csv`.
5. Train clustering models on TRAIN samples:
   - Baseline (no grouping).
   - WIP deciles.
   - K-means on:
       * Simple features   : [WIP, λ, %availability]
       * Advanced features : simple + per-activity WIP
       * State-vector      : per-activity state scores in [0,1]
6. Apply models to TEST samples:
   - For each timestamp and each method, assign cluster / group.
   - Retrieve CI parameters from the corresponding group.
   - Check whether the actual error falls within the predicted CI.
   - Store results in `test_evaluation.csv`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Tuple

import numpy as np
import pandas as pd

import evaluation as ev
from helper import (
    generate_short_uuid,
    read_event_log,
    split_into_subsets,
)
from evaluate_with_existing_alog import (
    # DATASETS,
    # ALIASES,
    SIM_RENAME_MAP,
    ps_runner,
)

# ────────────────────────────────────────────────────────────────────
# 0b. Local dataset config (train/test logs + sim model)
# ────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class DatasetConfig:
    train_log: str           # event log for training samples
    test_log: str            # event log for test samples
    bpmn_model: str          # BPMN used by the simulator
    sim_params: str          # JSON params for the simulator
    total_cases: int         # how many cases to simulate in ProSiMoS
    cut: str                 # original cutpoint (kept for reference)
    horizon_days: int        # original horizon in days (kept for reference)


DATASETS: dict[str, DatasetConfig] = {
    # -------- Synthetic – Loan App ---------------------------------
    "LOAN_STABLE": DatasetConfig(
        train_log="samples/extension-uncertainty/synthetic-v2/Loan-stable/Loan-stable-train.csv",
        test_log="samples/extension-uncertainty/synthetic-v2/Loan-stable/Loan-stable-test.csv",
        bpmn_model="samples/extension-uncertainty/synthetic/loan-stable/Loan-stable-train.bpmn",
        sim_params="samples/extension-uncertainty/synthetic/loan-stable/Loan-stable-train.json",
        total_cases=5_000,
        cut="2025-01-20T10:00:00Z",
        horizon_days=1,
    ),
    "LOAN_CIRCADIAN": DatasetConfig(
        train_log="samples/extension-uncertainty/synthetic-v2/Loan-circadian/Loan-circadian-train.csv",
        test_log="samples/extension-uncertainty/synthetic-v2/Loan-circadian/Loan-circadian-test.csv",
        bpmn_model="samples/extension-uncertainty/synthetic/loan-circadian/Loan-circadian-train.bpmn",
        sim_params="samples/extension-uncertainty/synthetic/loan-circadian/Loan-circadian-train.json",
        total_cases=5_000,
        cut="2025-03-20T15:00:00Z",
        horizon_days=3,
    ),
    "LOAN_UNSTABLE": DatasetConfig(
        train_log="samples/extension-uncertainty/synthetic-v2/Loan-unpredictable/Loan-unpredictable-train.csv",
        test_log="samples/extension-uncertainty/synthetic-v2/Loan-unpredictable/Loan-unpredictable-test.csv",
        bpmn_model="samples/extension-uncertainty/synthetic/loan-unstable/Loan-unstable-train.bpmn",
        sim_params="samples/extension-uncertainty/synthetic/loan-unstable/Loan-unstable-train.json",
        total_cases=5_000,
        cut="2025-03-20T15:00:00Z",
        horizon_days=2,
    ),

    # -------- Synthetic – P2P --------------------------------------
    "P2P_STABLE": DatasetConfig(
        train_log="samples/extension-uncertainty/synthetic-v2/P2P-stable/P2P-stable-train.csv",
        test_log="samples/extension-uncertainty/synthetic-v2/P2P-stable/P2P-stable-test.csv",
        bpmn_model="samples/extension-uncertainty/synthetic/p2p-stable/P2P-stable-train.bpmn",
        sim_params="samples/extension-uncertainty/synthetic/p2p-stable/P2P-stable-train.json",
        total_cases=5_000,
        cut="2020-01-15T10:00:00Z",
        horizon_days=1,
    ),
    "P2P_CIRCADIAN": DatasetConfig(
        train_log="samples/extension-uncertainty/synthetic-v2/P2P-circadian/P2P-circadian-train.csv",
        test_log="samples/extension-uncertainty/synthetic-v2/P2P-circadian/P2P-circadian-test.csv",
        bpmn_model="samples/extension-uncertainty/synthetic/p2p-circadian/P2P-circadian-train.bpmn",
        sim_params="samples/extension-uncertainty/synthetic/p2p-circadian/P2P-circadian-train.json",
        total_cases=5_000,
        cut="2020-01-10T10:00:00Z",
        horizon_days=3,
    ),
    "P2P_UNSTABLE": DatasetConfig(
        train_log="samples/extension-uncertainty/synthetic/p2p-unstable/P2P-unstable-train.csv",
        test_log="samples/extension-uncertainty/synthetic/p2p-unstable/P2P-unstable-test.csv",
        bpmn_model="samples/extension-uncertainty/synthetic/p2p-unstable/P2P-unstable-train.bpmn",
        sim_params="samples/extension-uncertainty/synthetic/p2p-unstable/P2P-unstable-train.json",
        total_cases=5_000,
        cut="2020-01-10T10:00:00Z",
        horizon_days=4,
    ),

    # -------- Real-life --------------------------------------------
    "BPIC_2012": DatasetConfig(
        train_log="samples/extension-uncertainty/real-life/BPIC_2012_train_transformed.csv.gz",
        test_log="samples/extension-uncertainty/real-life/BPIC_2012_test_transformed.csv.gz",
        bpmn_model="samples/extension-uncertainty/real-life/BPIC_2012_train_transformed.bpmn",
        sim_params="samples/extension-uncertainty/real-life/BPIC_2012_train_transformed.json",
        total_cases=3_000,
        cut="2012-01-18T13:00:00Z",
        horizon_days=25,
    ),
    "BPIC_2017": DatasetConfig(
        train_log="samples/extension-uncertainty/real-life/BPIC_2017_pre-drift_transformed.csv.gz",
        test_log="samples/extension-uncertainty/real-life/BPIC_2017_post-drift_transformed.csv.gz",
        bpmn_model="samples/extension-uncertainty/real-life/BPIC_2017_pre-drift_transformed.bpmn",
        sim_params="samples/extension-uncertainty/real-life/BPIC_2017_pre-drift_transformed.json",
        total_cases=15_000,
        cut="2016-10-10T13:00:00Z",
        horizon_days=34,
    ),
    "WORK_ORDERS": DatasetConfig(
        train_log="samples/extension-uncertainty/real-life/work_orders_pre-drift_transformed.csv.gz",
        test_log="samples/extension-uncertainty/real-life/work_orders_post-drift_transformed.csv.gz",
        bpmn_model="samples/extension-uncertainty/real-life/work_orders_pre-drift_transformed.bpmn",
        sim_params="samples/extension-uncertainty/real-life/work_orders_pre-drift_transformed.json",
        total_cases=7_000,
        cut="2022-12-22T07:00:00Z",
        horizon_days=13,
    ),
}

SYNTHETIC_DATASETS = [
    "LOAN_STABLE", "LOAN_CIRCADIAN", "LOAN_UNSTABLE",
    "P2P_STABLE", "P2P_CIRCADIAN", "P2P_UNSTABLE",
]

LOAN_DATASETS = [
    "LOAN_STABLE", "LOAN_CIRCADIAN", "LOAN_UNSTABLE",
]

P2P_DATASETS = [
    "P2P_STABLE", "P2P_CIRCADIAN", "P2P_UNSTABLE",
]

REAL_LIFE_DATASETS = [
    "BPIC_2012", "BPIC_2017", "WORK_ORDERS",
]

ALIASES = {
    "ALL": list(DATASETS.keys()),
    "LOAN": LOAN_DATASETS,
    "P2P": P2P_DATASETS,
    "SYNTHETIC": SYNTHETIC_DATASETS,
    "REAL-LIFE": REAL_LIFE_DATASETS,
}


# scikit-learn is used for KMeans. If missing, you can install it or
# replace the KMeans bits with your own implementation.
from sklearn.cluster import KMeans


# ────────────────────────────────────────────────────────────────────
# 1. Data structures
# ────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class SimulationConfig:
    """Static configuration describing how to simulate a dataset."""
    dataset: str
    event_log_csv: str           # original log path (used by ProSiMoS)
    bpmn_model: str
    sim_params: str
    total_cases: int


@dataclass(slots=True)
class SplitLogs:
    """Train/test split of the canonical event log (with standard column names)."""
    train: pd.DataFrame
    test: pd.DataFrame
    train_csv_path: str
    test_csv_path: str


@dataclass(slots=True)
class FeatureEnv:
    """Precomputed helpers that speed-up feature construction."""
    case_arrival: pd.Series            # index: case_id, value: first start_time
    resources: np.ndarray              # unique resources
    activities: np.ndarray             # unique activities
    has_enable_time: bool
    activity_name_map: Dict[str, str]  # original -> safe column suffix


# ────────────────────────────────────────────────────────────────────
# 2. Utility helpers
# ────────────────────────────────────────────────────────────────────

def _safe_activity_name(name: str) -> str:
    """
    Make an activity name safe to use in column names:
    - non-alphanumeric -> '_'
    - strip leading/trailing '_'
    """
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_") or "act"


def _mean_ci(vals: np.ndarray, conf: float = 0.95) -> Tuple[float | None, float | None]:
    """
    Return (mean, approximate 95% *predictive* half-width) for a 1D array.

    NOTE: this is now meant as a prediction interval for individual
    errors, not a confidence interval for the mean. We use:

        mean ± 1.96 * std

    which is the usual ~95% interval for a normal distribution.
    """
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return None, None

    mean = float(vals.mean())
    if vals.size == 1:
        return mean, 0.0

    std = float(vals.std(ddof=1))
    # 1.96 * std  -> predictive 95% half-width (no / sqrt(n))
    ci = 1.96 * std
    return mean, ci



def _to_jsonable(obj):
    """Convert numpy scalars/arrays to plain Python types so json.dump works."""
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


# ────────────────────────────────────────────────────────────────────
# 3. Train/test split and horizon computation
# ────────────────────────────────────────────────────────────────────

# def split_log_train_test(
#     df: pd.DataFrame,
#     original_csv_path: str,
#     train_fraction: float = 0.7,
#     seed: int = 42,
#     log_fraction: float = 1.0,
# ) -> SplitLogs:
#     """
#     Split event log into train/test by *cases* so that each case is
#     entirely in either split (no overlapping leftovers).

#     New behaviour with log_fraction:
#     - Order cases by their first start_time (chronological order).
#     - Keep only the earliest `log_fraction` of cases.
#     - Within those, take the earliest `train_fraction` as TRAIN and
#       the next portion as TEST.
#     - All remaining cases are discarded.

#     Additionally writes the split logs into an `auto/` folder next to
#     the original CSV.
#     """
#     if not (0 < log_fraction <= 1.0):
#         raise ValueError(f"log_fraction must be in (0,1], got {log_fraction}")
#     if not (0 < train_fraction < 1.0):
#         raise ValueError(f"train_fraction must be in (0,1), got {train_fraction}")

#     # 1) Chronological ordering of cases by first start_time
#     case_starts = (
#         df.groupby("case_id")["start_time"]
#           .min()
#           .sort_values()  # earliest cases first
#     )
#     all_case_ids_ordered = case_starts.index.to_numpy()
#     n_total_cases = len(all_case_ids_ordered)
#     if n_total_cases == 0:
#         raise ValueError("No cases found in the log.")

#     # 2) Keep only the earliest `log_fraction` of cases
#     n_used = int(round(n_total_cases * log_fraction))
#     n_used = max(1, min(n_total_cases, n_used))
#     used_case_ids = all_case_ids_ordered[:n_used]

#     # 3) Within the used portion, split into train/test sequentially
#     n_train = int(round(n_used * train_fraction))
#     n_train = max(1, min(n_used - 1, n_train))  # ensure both splits non-empty

#     train_cases = set(used_case_ids[:n_train])
#     test_cases = set(used_case_ids[n_train:])

#     # 4) Build the train/test dataframes (remaining cases are discarded)
#     train_df = df[df["case_id"].isin(train_cases)].copy()
#     test_df = df[df["case_id"].isin(test_cases)].copy()

#     orig_path = Path(original_csv_path)
#     auto_dir = orig_path.parent / "auto"
#     auto_dir.mkdir(exist_ok=True)

#     stem = orig_path.stem  # works fine with .csv.gz too
#     train_csv_path = auto_dir / f"{stem}_train.csv.gz"
#     test_csv_path = auto_dir / f"{stem}_test.csv.gz"

#     train_df.to_csv(train_csv_path, index=False)
#     test_df.to_csv(test_csv_path, index=False)

#     return SplitLogs(
#         train=train_df,
#         test=test_df,
#         train_csv_path=str(train_csv_path),
#         test_csv_path=str(test_csv_path),
#     )

def split_log_train_test(
    df: pd.DataFrame,
    original_csv_path: str,
    train_fraction: float = 0.5,
    seed: int = 42,          # kept for API compatibility (unused)
    log_fraction: float = 1.0,
) -> SplitLogs:
    """
    Split event log into train/test based on *time span* (not by number of cases):

    - Compute log_start = min(start_time), log_end = max(end_time)
    - Optionally restrict to the first `log_fraction` of the time span:
          used_end = log_start + log_fraction * (log_end - log_start)
    - Split point in time:
          cut_time = log_start + train_fraction * (used_end - log_start)
    - Assign each case to TRAIN if its case_start <= cut_time, else TEST.
      (Whole-case split; no overlapping cases across splits.)

    Additionally writes split logs into an `auto/` folder next to original_csv_path.
    """
    if not (0 < log_fraction <= 1.0):
        raise ValueError(f"log_fraction must be in (0,1], got {log_fraction}")
    if not (0 < train_fraction < 1.0):
        raise ValueError(f"train_fraction must be in (0,1), got {train_fraction}")

    df = df.copy()
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
    df["end_time"] = pd.to_datetime(df["end_time"], utc=True)

    log_start = df["start_time"].min()
    log_end = df["end_time"].max()
    if pd.isna(log_start) or pd.isna(log_end):
        raise ValueError("Log start/end time is NaT.")
    if log_end <= log_start:
        raise ValueError(f"Invalid log span: start={log_start}, end={log_end}")

    # Restrict to first `log_fraction` of the time span (optional)
    full_span = (log_end - log_start)
    used_end = log_start + full_span * float(log_fraction)

    # Time-based split cut
    used_span = used_end - log_start
    cut_time = log_start + used_span * float(train_fraction)

    # Case start times define which half of time the case "occurred" in
    case_starts = df.groupby("case_id")["start_time"].min()

    # Only keep cases that start within the used window (if log_fraction < 1)
    allowed_cases = set(case_starts[case_starts <= used_end].index)

    train_cases = set(case_starts[(case_starts <= cut_time)].index) & allowed_cases
    test_cases = set(case_starts[(case_starts > cut_time)].index) & allowed_cases

    # Build the train/test dataframes (whole cases)
    train_df = df[df["case_id"].isin(train_cases)].copy()
    test_df = df[df["case_id"].isin(test_cases)].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Time-based split produced an empty split. "
            f"train_cases={len(train_cases)}, test_cases={len(test_cases)}. "
            "This can happen if all cases start on one side of the time cut. "
            "Try a different train_fraction (e.g. 0.4/0.6) or a different case assignment rule."
        )

    orig_path = Path(original_csv_path)
    auto_dir = orig_path.parent / "auto"
    auto_dir.mkdir(exist_ok=True)

    stem = orig_path.stem  # works with .csv.gz too
    train_csv_path = auto_dir / f"{stem}_train.csv.gz"
    test_csv_path = auto_dir / f"{stem}_test.csv.gz"

    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)

    return SplitLogs(
        train=train_df,
        test=test_df,
        train_csv_path=str(train_csv_path),
        test_csv_path=str(test_csv_path),
    )


def compute_case_duration_horizon(
    df: pd.DataFrame,
    percentile: float = 0.95,
) -> pd.Timedelta:
    """
    Compute the *percentile* of case durations (end - start) and
    return it as a Timedelta, to be used as:
    - simulation horizon (H), and
    - padding at the log start/end when picking timestamps.
    """
    grouped = df.groupby("case_id").agg(
        start=("start_time", "min"),
        end=("end_time", "max"),
    )
    durations = (grouped["end"] - grouped["start"]).dt.total_seconds().to_numpy()
    if durations.size == 0:
        raise ValueError("No case durations found in the log.")
    h_seconds = np.quantile(durations, percentile)
    # Ensure at least 1 second horizon
    h_seconds = max(float(h_seconds), 1.0)
    return pd.Timedelta(seconds=h_seconds)


def choose_candidate_cut_times(
    df: pd.DataFrame,
    horizon: pd.Timedelta,
    source: Literal["activity_end", "case_end"] = "activity_end",
) -> List[pd.Timestamp]:
    """
    Return a list of candidate cut timestamps from the log:

    - If source == "activity_end":
        Use all activity end times.
    - If source == "case_end":
        Use case completion times (max end_time per case).

    In both cases we:
    - sort unique times; and
    - discard those inside the first/last *horizon* of the log.
    """
    first_ts = df["start_time"].min()
    last_ts = df["end_time"].max()
    safe_start = first_ts + horizon
    safe_end = last_ts - horizon
    if safe_start >= safe_end:
        raise ValueError(f"Log is shorter than twice the horizon – no safe region, horizon is {horizon}. First timestamp: {first_ts}, last timestamp: {last_ts}.")

    if source == "activity_end":
        all_ts = df["end_time"].dropna().sort_values().unique()
    elif source == "case_end":
        all_ts = (
            df.groupby("case_id")["end_time"]
            .max()
            .dropna()
            .sort_values()
            .unique()
        )
    else:
        raise ValueError(f"Unknown timestamp source: {source}")

    out = [
        pd.Timestamp(t).tz_convert("UTC") if pd.Timestamp(t).tzinfo else pd.Timestamp(t).tz_localize("UTC")
        for t in all_ts
        if safe_start <= pd.Timestamp(t) <= safe_end
    ]
    return out


# ────────────────────────────────────────────────────────────────────
# 3b. Subsampling of cut timestamps
# ────────────────────────────────────────────────────────────────────

def subsample_cuts_equally(
    cut_times: List[pd.Timestamp],
    max_points: int,
) -> List[pd.Timestamp]:
    """
    Subsample *cut_times* to at most *max_points* timestamps, keeping them
    roughly equally spaced in their sorted order.

    If there are fewer than *max_points* timestamps, the list is returned
    unchanged.
    """
    if max_points is None or max_points <= 0:
        return cut_times

    n = len(cut_times)
    if n <= max_points:
        return cut_times

    # indices spaced between 0 and n-1 (inclusive)
    idx = np.linspace(0, n - 1, num=max_points, dtype=int)
    idx = np.unique(idx)  # guard against duplicates from rounding

    return [cut_times[i] for i in idx]


# ────────────────────────────────────────────────────────────────────
# 4. Feature extraction
# ────────────────────────────────────────────────────────────────────

def prepare_feature_env(df: pd.DataFrame) -> FeatureEnv:
    case_arrival = (
        df.groupby("case_id")["start_time"]
        .min()
        .astype("datetime64[ns, UTC]")
    )
    resources = df["resource"].dropna().unique()
    activities = df["activity"].dropna().unique()
    has_enable_time = "enable_time" in df.columns

    name_map: Dict[str, str] = {}
    for a in activities:
        name_map[a] = _safe_activity_name(a)

    return FeatureEnv(
        case_arrival=case_arrival,
        resources=resources,
        activities=activities,
        has_enable_time=has_enable_time,
        activity_name_map=name_map,
    )


def compute_features_at_cut(
    df: pd.DataFrame,
    cut: pd.Timestamp,
    horizon: pd.Timedelta,
    env: FeatureEnv,
) -> Dict[str, float]:
    """
    Compute all feature families at a given cut timestamp:

    - WIP (overall)
    - λ (arrival rate per hour over the previous horizon)
    - % resource availability
    - WIP advanced: per-activity enabled/ongoing counts
    - Activity state vector: per-activity [0,1] values
    """
    out: Dict[str, float] = {}
    cut = pd.to_datetime(cut, utc=True)

    # ---- WIP (overall) ----------------------------------------------------
    ongoing_mask = (df["start_time"] <= cut) & (df["end_time"] > cut)
    ongoing = df.loc[ongoing_mask]
    wip = int(ongoing["case_id"].nunique())
    out["wip"] = float(wip)

    # ---- Arrival rate over [cut - horizon, cut] ---------------------------
    window_start = cut - horizon
    ca = env.case_arrival
    num_arrivals = int(((ca > window_start) & (ca <= cut)).sum())
    horizon_hours = horizon.total_seconds() / 3600.0
    arrival_rate = num_arrivals / horizon_hours if horizon_hours > 0 else 0.0
    out["arrival_rate_per_hour"] = float(arrival_rate)

    # ---- Resource availability -------------------------------------------
    total_resources = len(env.resources)
    if total_resources > 0:
        busy_resources = ongoing["resource"].dropna().unique()
        available = total_resources - len(busy_resources)
        out["resource_availability"] = float(available / total_resources)
    else:
        out["resource_availability"] = float("nan")

    # ---- Advanced per-activity WIP ---------------------------------------
    # ongoing per activity (cases overlapping cut)
    ongoing_by_act = (
        ongoing.groupby("activity")["case_id"]
        .nunique()
        .to_dict()
    )

    # enabled-but-not-started per activity, if we have enable_time
    if env.has_enable_time:
        enabled_mask = (
            (df["enable_time"] <= cut)
            & (df["start_time"] > cut)
        )
        enabled = df.loc[enabled_mask]
        enabled_by_act = (
            enabled.groupby("activity")["case_id"]
            .nunique()
            .to_dict()
        )
    else:
        enabled_by_act = {}

    # ---- Activity state vector [0,1] -------------------------------------
    # For each activity:
    #   0.0       -> no enabled/ongoing
    #   (0,0.5]   -> enabled, scaled by max enabled age
    #   (0.5,1.0] -> ongoing, scaled by max ongoing age
    # We scale by the global max age among all activities so that
    # the *longest* enabled gets 0.5 and the *longest* ongoing gets 1.0.

    if env.has_enable_time:
        enabled_full = df.loc[
            (df["enable_time"] <= cut) & (df["start_time"] > cut),
            ["activity", "enable_time"],
        ].copy()
        enabled_full["age_sec"] = (cut - enabled_full["enable_time"]).dt.total_seconds()
    else:
        enabled_full = df.iloc[0:0][["activity"]].copy()
        enabled_full["age_sec"] = []

    ongoing_full = df.loc[
        (df["start_time"] <= cut) & (df["end_time"] > cut),
        ["activity", "start_time"],
    ].copy()
    ongoing_full["age_sec"] = (cut - ongoing_full["start_time"]).dt.total_seconds()

    max_enabled_age = (
        float(enabled_full["age_sec"].max()) if len(enabled_full) > 0 else 0.0
    )
    max_ongoing_age = (
        float(ongoing_full["age_sec"].max()) if len(ongoing_full) > 0 else 0.0
    )
    global_max_age = max(max_enabled_age, max_ongoing_age, 0.0)

    for act in env.activities:
        safe_name = env.activity_name_map[act]

        # per-activity WIP (advanced)
        enabled_count = int(enabled_by_act.get(act, 0))
        ongoing_count = int(ongoing_by_act.get(act, 0))
        out[f"wip_enabled_{safe_name}"] = float(enabled_count)
        out[f"wip_ongoing_{safe_name}"] = float(ongoing_count)

        # state vector component
        # default: 0.0 (no presence at all)
        val = 0.0
        if global_max_age > 0.0:
            # Enabled age
            e_age = enabled_full.loc[
                enabled_full["activity"] == act, "age_sec"
            ]
            if not e_age.empty:
                # Map [0, global_max] -> (0,0.5]
                val = max(
                    val,
                    0.0 + 0.5 * float(e_age.max()) / global_max_age,
                )
            # Ongoing age
            o_age = ongoing_full.loc[
                ongoing_full["activity"] == act, "age_sec"
            ]
            if not o_age.empty:
                # Map [0, global_max] -> (0.5,1.0]
                val = max(
                    val,
                    0.5 + 0.5 * float(o_age.max()) / global_max_age,
                )
        out[f"statevec_{safe_name}"] = float(val)

    return out


# ────────────────────────────────────────────────────────────────────
# 5. Simulation for a single cut timestamp
# ────────────────────────────────────────────────────────────────────

def simulate_for_cut(
    cfg: SimulationConfig,
    reference_df: pd.DataFrame,
    cut: pd.Timestamp,
    horizon: pd.Timedelta,
    n_runs: int,
    out_root: Path,
    verbose: bool = True,
) -> Dict:
    """
    Run process-state simulations for a single cut timestamp.

    - reference_df: train or test subset (canonical columns).
    - horizon: 95th percentile of case durations.
    - For each of n_runs, we:
        * build reference subsets A_event, A_ongoing, A_complete;
        * call ev.evaluate("process_state", ps_runner, ...).
    - Returns the aggregated metrics (across the Monte-Carlo runs).
    - Writes all simulation logs under out_root / <cut_iso> / <run>.
    """

    cut = pd.to_datetime(cut, utc=True)
    cut_folder_name = cut.isoformat().replace(":", "-")
    cut_dir = out_root / cut_folder_name
    cut_dir.mkdir(parents=True, exist_ok=True)

    end_ts = cut + horizon
    sim_horizon = cut + 2 * horizon  # run simulation a bit beyond the evaluation window

    # Reference subsets for metrics
    A_event, A_ongoing, A_complete = split_into_subsets(reference_df, cut, end_ts)

    # Keep reference subsets if we want to inspect later
    ev._dump(A_event, cut_dir, "A_event_filter.csv")
    ev._dump(A_ongoing, cut_dir, "A_ongoing.csv")
    ev._dump(A_complete, cut_dir, "A_complete.csv")

    runs: List[Dict] = []

    for run_no in range(1, n_runs + 1):
        run_dir = cut_dir / str(run_no)
        run_dir.mkdir()

        io_obj = ev.SimulationIO(
            log_csv=run_dir / "sim_log.csv",
            stats_csv=run_dir / "sim_stats.csv",
            out_dir=run_dir,
        )

        if verbose:
            print(f"[{cfg.dataset}] cut={cut} run={run_no}/{n_runs}")

        # process-state flavour
        result = ev.evaluate(
            "process_state",
            io_obj,
            ps_runner,
            cut=cut,
            end=end_ts,
            A_event=A_event,
            A_ongoing=A_ongoing,
            A_complete=A_complete,
            runner_kwargs=dict(
                event_log=cfg.event_log_csv,
                bpmn_model=cfg.bpmn_model,
                bpmn_parameters=cfg.sim_params,
                start_time=cut,
                simulation_horizon=sim_horizon,
                total_cases=cfg.total_cases,
                column_mapping={v: k for k, v in SIM_RENAME_MAP.items()},
                rename_map=SIM_RENAME_MAP,
            ),
        )
        runs.append(result)

        warnings_path = run_dir / "simulation_warnings.txt"
        try:
            warnings_path.unlink(missing_ok=True)  # Python 3.8+: remove if exists
        except TypeError:
            # For older Python: missing_ok not available
            if warnings_path.exists():
                warnings_path.unlink()
        except OSError:
            # Ignore e.g. file locked / permission issues
            pass

    agg = ev.aggregate(runs)
    return agg


def extract_error_targets(
    agg_metrics: Dict,
) -> Dict[str, float | None]:
    """
    Extract error targets from the aggregated metric dictionary.

    We focus on:
    - ongoing_filter / RTD      -> err_RTD_mean, err_RTD_ci
    - complete_filter / cycle_time -> err_cycle_mean, err_cycle_ci
    (if present; otherwise None).
    """
    out: Dict[str, float | None] = {
        "err_RTD_mean": None,
        "err_RTD_ci": None,
        "err_cycle_mean": None,
        "err_cycle_ci": None,
    }

    try:
        rtd_info = agg_metrics["ongoing_filter"]["RTD"]
        out["err_RTD_mean"] = (
            float(rtd_info["mean"]) if rtd_info["mean"] is not None else None
        )
        out["err_RTD_ci"] = (
            float(rtd_info["ci"]) if rtd_info["ci"] is not None else None
        )
    except Exception:
        pass

    try:
        ctd_info = agg_metrics["complete_filter"]["cycle_time"]
        out["err_cycle_mean"] = (
            float(ctd_info["mean"]) if ctd_info["mean"] is not None else None
        )
        out["err_cycle_ci"] = (
            float(ctd_info["ci"]) if ctd_info["ci"] is not None else None
        )
    except Exception:
        pass

    return out


# ────────────────────────────────────────────────────────────────────
# 6. Build per-timestamp samples for train/test
# ────────────────────────────────────────────────────────────────────

def build_samples_for_split(
    split_name: Literal["train", "test"],
    cfg: SimulationConfig,
    log_df: pd.DataFrame,
    horizon: pd.Timedelta,
    feature_env: FeatureEnv,
    cut_times: List[pd.Timestamp],
    n_runs: int,
    out_root: Path,
    timestamp_source: str,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    For every cut in cut_times:
        - compute feature vector at the cut;
        - run n_runs simulations and aggregate metrics;
        - collect feature + error data.

    Writes raw simulation outputs under:
        out_root / <split_name> / <cut_iso> / <run_no> / ...

    Returns a DataFrame with one row per (split, cut) sample.
    """
    split_root = out_root / split_name
    split_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []

    for i, cut in enumerate(cut_times, start=1):
        if verbose:
            print(f"\n[{cfg.dataset}/{split_name}] cut {i}/{len(cut_times)} at {cut}")

        # features from reference log (no simulation)
        feat = compute_features_at_cut(log_df, cut, horizon, feature_env)

        # simulation & metrics
        agg = simulate_for_cut(
            cfg=cfg,
            reference_df=log_df,
            cut=cut,
            horizon=horizon,
            n_runs=n_runs,
            out_root=split_root,
            verbose=verbose,
        )
        err = extract_error_targets(agg)

        row = {
            "split": split_name,
            "cut_time": pd.to_datetime(cut, utc=True),
            "timestamp_source": timestamp_source,
        }
        row.update(feat)
        row.update(err)
        rows.append(row)

    df_samples = pd.DataFrame(rows)
    # make sure cut_time is ISO string on disk to ease later reuse
    df_samples["cut_time_iso"] = df_samples["cut_time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return df_samples


# ────────────────────────────────────────────────────────────────────
# 7. Clustering models (on training samples)
# ────────────────────────────────────────────────────────────────────

def train_baseline_model(
    train_df: pd.DataFrame,
    target_col: str,
) -> Dict:
    """Baseline: single global mean+CI, plus some quantiles."""
    errs = train_df[target_col].to_numpy(dtype=float)
    errs = errs[~np.isnan(errs)]
    mean, ci = _mean_ci(errs)

    quantiles = {}
    for q in (0.05, 0.25, 0.5, 0.75, 0.95):
        quantiles[str(q)] = float(np.quantile(errs, q)) if errs.size > 0 else None

    return {
        "target_col": target_col,
        "mean": mean,
        "ci": ci,
        "quantiles": quantiles,
    }


def train_wip_deciles_model(
    train_df: pd.DataFrame,
    target_col: str,
    n_bins: int = 3,
) -> Dict:
    """
    Group by quantiles of WIP; compute mean+CI per bucket.

    n_bins controls how many WIP groups we create (should normally
    match the number of clusters used by K-means).
    """
    df = train_df[["wip", target_col]].dropna()
    if df.empty:
        return {
            "target_col": target_col,
            "bin_edges": [],
            "groups": {},
        }

    wip = df["wip"].to_numpy(dtype=float)
    errs = df[target_col].to_numpy(dtype=float)

    # quantile edges; handle potential duplicates (constant wip)
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(wip, qs)
    edges[0] = float(edges[0] - 1e-9)  # ensure left-open on the very left

    groups: Dict[int, Dict] = {}
    for i in range(n_bins):
        left, right = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (wip > left) & (wip <= right + 1e-9)
        else:
            mask = (wip > left) & (wip <= right)

        bin_errs = errs[mask]
        mean, ci = _mean_ci(bin_errs)
        groups[i] = {
            "mean": mean,
            "ci": ci,
            "count": int(bin_errs.size),
            "wip_range": [float(left), float(right)],
        }

    return {
        "target_col": target_col,
        "bin_edges": [float(x) for x in edges],
        "groups": groups,
    }


def train_random_groups_model(
    train_df: pd.DataFrame,
    target_col: str,
    n_clusters: int = 3,
) -> Dict:
    """
    Random control technique.

    Split the *training samples* into n_clusters groups of (roughly)
    equal size based only on their index, without using any features.
    For each group we compute mean+CI of the target errors.

    This lets us see how much we gain/lose from using smaller subsets
    compared to the global baseline, when the subsets are *not*
    informed by process state.
    """
    errs = train_df[target_col].to_numpy(dtype=float)
    mask = ~np.isnan(errs)
    vals = errs[mask]
    if vals.size == 0:
        return {
            "target_col": target_col,
            "n_clusters": n_clusters,
            "groups": {},
        }

    n_clusters = max(1, min(int(n_clusters), vals.size))

    groups: Dict[int, Dict] = {}
    idx = np.arange(vals.size)
    cluster_ids = idx % n_clusters  # first 33 → cluster 0, next 33 → 1, etc.

    for cid in range(n_clusters):
        cluster_errs = vals[cluster_ids == cid]
        mean, ci = _mean_ci(cluster_errs)
        groups[cid] = {
            "mean": mean,
            "ci": ci,
            "count": int(cluster_errs.size),
        }

    return {
        "target_col": target_col,
        "n_clusters": n_clusters,
        "groups": groups,
    }


def _train_kmeans_on_features(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    n_clusters: int,
    random_state: int = 0,
) -> Dict:
    """Train a KMeans model on the given feature columns and attach CIs per cluster."""
    available = [c for c in feature_cols if c in train_df.columns]
    if not available:
        return {
            "target_col": target_col,
            "feature_cols": [],
            "centers": [],
            "clusters": {},
        }

    X = train_df[available].to_numpy(dtype=float)
    y = train_df[target_col].to_numpy(dtype=float)

    mask = (~np.isnan(X).any(axis=1)) & (~np.isnan(y))
    X_valid = X[mask]
    y_valid = y[mask]

    if X_valid.shape[0] == 0:
        return {
            "target_col": target_col,
            "feature_cols": available,
            "centers": [],
            "clusters": {},
        }

    n_clusters = min(n_clusters, max(1, X_valid.shape[0]))
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init="auto",
    )
    labels = kmeans.fit_predict(X_valid)

    clusters: Dict[int, Dict] = {}
    for cid in range(n_clusters):
        errs = y_valid[labels == cid]
        mean, ci = _mean_ci(errs)
        clusters[cid] = {
            "mean": mean,
            "ci": ci,
            "count": int(errs.size),
        }

    return {
        "target_col": target_col,
        "feature_cols": available,
        "centers": kmeans.cluster_centers_.tolist(),
        "clusters": clusters,
    }


def train_clustering_models(
    train_df: pd.DataFrame,
    target_col: str,
    n_clusters_basic: int = 3,
    n_clusters_advanced: int = 3,
    n_clusters_state: int = 3,
) -> Dict[str, Dict]:
    """
    Train all requested clustering strategies on the training samples.
    """
    models: Dict[str, Dict] = {}

    # 1) Global baseline (no grouping)
    models["baseline"] = train_baseline_model(train_df, target_col)

    # 2) Random control technique (index-based equal-size groups)
    models["random_groups"] = train_random_groups_model(
        train_df, target_col, n_clusters=n_clusters_basic
    )

    # 3) WIP quantile groups – same number as basic K-means clusters
    models["wip_deciles"] = train_wip_deciles_model(
        train_df, target_col, n_bins=n_clusters_basic
    )

    # 4) K-means on simple features
    simple_features = ["wip", "arrival_rate_per_hour", "resource_availability"]
    models["kmeans_simple"] = _train_kmeans_on_features(
        train_df, simple_features, target_col, n_clusters_basic
    )

    # 5) K-means on advanced WIP features (simple + per-activity WIP)
    advanced_features = simple_features + [
        c
        for c in train_df.columns
        if c.startswith("wip_enabled_") or c.startswith("wip_ongoing_")
    ]
    models["kmeans_advanced_wip"] = _train_kmeans_on_features(
        train_df, advanced_features, target_col, n_clusters_advanced
    )

    # 6) K-means on activity state vector features
    state_features = [c for c in train_df.columns if c.startswith("statevec_")]
    models["kmeans_statevec"] = _train_kmeans_on_features(
        train_df, state_features, target_col, n_clusters_state
    )

    return models



# ────────────────────────────────────────────────────────────────────
# 8. Apply clustering models to test samples
# ────────────────────────────────────────────────────────────────────

def _assign_wip_decile(
    wip_value: float,
    bin_edges: List[float],
) -> int | None:
    if not bin_edges:
        return None
    edges = np.asarray(bin_edges, dtype=float)
    idx = int(np.searchsorted(edges, wip_value, side="right") - 1)
    idx = max(0, min(idx, len(edges) - 2))
    return idx


def _assign_kmeans(
    row: pd.Series,
    model: Dict,
) -> Tuple[int | None, float | None, float | None, bool | None]:
    """
    Assign a single test sample to a KMeans cluster and check whether the
    error is inside the cluster CI.

    Returns (cluster_id, pred_mean, pred_ci, in_ci)
    """
    centers = np.asarray(model.get("centers", []), dtype=float)
    feature_cols = model.get("feature_cols", [])
    if centers.size == 0 or not feature_cols:
        return None, None, None, None

    x = row[feature_cols].to_numpy(dtype=float)
    if np.isnan(x).any():
        return None, None, None, None

    dists = np.linalg.norm(centers - x, axis=1)
    cid = int(np.argmin(dists))
    info = model["clusters"].get(cid)
    if not info:
        return cid, None, None, None
    mean = info["mean"]
    ci = info["ci"]
    err = row.get(model["target_col"])
    if err is None or (mean is None) or (ci is None):
        return cid, mean, ci, None
    in_ci = abs(float(err) - float(mean)) <= float(ci)
    return cid, mean, ci, bool(in_ci)


def apply_models_to_test(
    models: Dict[str, Dict],
    test_df: pd.DataFrame,
    target_col: str,
) -> pd.DataFrame:
    """
    For each sample, apply all clustering models and record:

    - predicted mean/CI from the assigned group;
    - whether the true error falls inside that CI.

    We reuse this both for TRAIN (in-sample) and TEST (out-of-sample).
    """
    records: List[Dict] = []

    baseline = models.get("baseline")
    wip_model = models.get("wip_deciles")
    km_simple = models.get("kmeans_simple")
    km_adv = models.get("kmeans_advanced_wip")
    km_state = models.get("kmeans_statevec")
    rand_model = models.get("random_groups")

    n_rand_clusters = int(rand_model.get("n_clusters", 0)) if rand_model else 0

    for idx, (_, row) in enumerate(test_df.iterrows()):
        rec: Dict[str, object] = {
            "cut_time": row.get("cut_time_iso"),
            "split": row.get("split"),
            "true_error": float(row[target_col]) if not pd.isna(row[target_col]) else None,
        }

        # Baseline (global)
        if baseline:
            m = baseline.get("mean")
            ci = baseline.get("ci")
            err = rec["true_error"]
            in_ci = (
                (err is not None and m is not None and ci is not None)
                and (abs(float(err) - float(m)) <= float(ci))
            )
            rec.update(
                baseline_mean=m,
                baseline_ci=ci,
                baseline_in_ci=bool(in_ci)
                if err is not None and m is not None and ci is not None
                else None,
            )

        # Random control technique – assign groups purely by index
        if rand_model and n_rand_clusters > 0:
            cid = idx % n_rand_clusters
            group = rand_model["groups"].get(cid, {})
            m = group.get("mean")
            ci = group.get("ci")
            err = rec["true_error"]
            in_ci = (
                (err is not None and m is not None and ci is not None)
                and (abs(float(err) - float(m)) <= float(ci))
            )
            rec.update(
                random_groups_id=cid,
                random_groups_mean=m,
                random_groups_ci=ci,
                random_groups_in_ci=bool(in_ci)
                if err is not None and m is not None and ci is not None
                else None,
            )

        # WIP quantile groups
        if wip_model and "wip" in row:
            wip_val = float(row["wip"]) if not pd.isna(row["wip"]) else None
            edges = wip_model.get("bin_edges", [])
            cid = _assign_wip_decile(wip_val, edges) if wip_val is not None else None
            if cid is not None:
                group = wip_model["groups"].get(cid, {})
                m = group.get("mean")
                ci = group.get("ci")
                err = rec["true_error"]
                in_ci = (
                    (err is not None and m is not None and ci is not None)
                    and (abs(float(err) - float(m)) <= float(ci))
                )
                rec.update(
                    wip_decile_id=cid,
                    wip_decile_mean=m,
                    wip_decile_ci=ci,
                    wip_decile_in_ci=bool(in_ci)
                    if err is not None and m is not None and ci is not None
                    else None,
                )
            else:
                rec.update(
                    wip_decile_id=None,
                    wip_decile_mean=None,
                    wip_decile_ci=None,
                    wip_decile_in_ci=None,
                )

        # K-means (simple features)
        if km_simple:
            cid, m, ci, in_ci = _assign_kmeans(row, km_simple)
            rec.update(
                kmeans_simple_id=cid,
                kmeans_simple_mean=m,
                kmeans_simple_ci=ci,
                kmeans_simple_in_ci=in_ci,
            )

        # K-means (advanced WIP)
        if km_adv:
            cid, m, ci, in_ci = _assign_kmeans(row, km_adv)
            rec.update(
                kmeans_advanced_wip_id=cid,
                kmeans_advanced_wip_mean=m,
                kmeans_advanced_wip_ci=ci,
                kmeans_advanced_wip_in_ci=in_ci,
            )

        # K-means (activity state vector)
        if km_state:
            cid, m, ci, in_ci = _assign_kmeans(row, km_state)
            rec.update(
                kmeans_statevec_id=cid,
                kmeans_statevec_mean=m,
                kmeans_statevec_ci=ci,
                kmeans_statevec_in_ci=in_ci,
            )

        records.append(rec)

    return pd.DataFrame(records)


# ────────────────────────────────────────────────────────────────────
# 8b. Summary statistics for evaluation
# ────────────────────────────────────────────────────────────────────

def summarise_evaluation(eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-method summary:

    method_key, method_label, n_points, coverage, avg_ci_width, avg_abs_error
    """
    methods = [
        ("baseline", "baseline_in_ci", "baseline_ci", "Global baseline"),
        ("wip_deciles", "wip_decile_in_ci", "wip_decile_ci", "WIP percentile groups"),
        ("kmeans_simple", "kmeans_simple_in_ci", "kmeans_simple_ci",
         "K-means (simple features)"),
        ("kmeans_advanced_wip", "kmeans_advanced_wip_in_ci", "kmeans_advanced_wip_ci",
         "K-means (advanced WIP)"),
        ("kmeans_statevec", "kmeans_statevec_in_ci", "kmeans_statevec_ci",
         "K-means (activity state vector)"),
        ("random_groups", "random_groups_in_ci", "random_groups_ci",
         "Random control (equal-size groups)"),
    ]

    rows: List[Dict[str, object]] = []

    for key, in_col, ci_col, label in methods:
        if in_col not in eval_df.columns or ci_col not in eval_df.columns:
            continue

        mask = eval_df[in_col].notna()
        if not mask.any():
            continue

        coverage = float(eval_df.loc[mask, in_col].mean())
        avg_ci = float(eval_df.loc[mask, ci_col].mean())
        avg_abs_err = float(eval_df.loc[mask, "true_error"].abs().mean())

        rows.append(
            {
                "method_key": key,
                "method_label": label,
                "n_points": int(mask.sum()),
                "coverage": coverage,
                "avg_ci_width": avg_ci,
                "avg_abs_error": avg_abs_err,
            }
        )

    return pd.DataFrame(rows)



# ────────────────────────────────────────────────────────────────────
# 9. Orchestration per dataset
# ────────────────────────────────────────────────────────────────────

# def _process_single_dataset(
#     dataset_name: str,
#     runs_per_cut: int,
#     timestamp_source: str,
#     error_metric: str,
#     output_root: str,
#     train_fraction: float = 0.7,
#     seed: int = 42,
#     n_clusters: int = 3,
#     log_fraction: float = 1.0,
# ) -> None:
#     cfg_ds = DATASETS[dataset_name]
#     cfg = SimulationConfig(
#         dataset=dataset_name,
#         event_log_csv=cfg_ds.alog,
#         bpmn_model=cfg_ds.model,
#         sim_params=cfg_ds.params,
#         total_cases=cfg_ds.total_cases,
#     )

#     out_root = Path(output_root) / dataset_name / generate_short_uuid()
#     out_root.mkdir(parents=True, exist_ok=True)
#     print(f"\n=== [{dataset_name}] Output root: {out_root} ===")

#     # 1) Load canonical event log (for features / metrics)
#     alog_df = read_event_log(
#         cfg_ds.alog,
#         rename={
#             "CaseId": "case_id",
#             "Activity": "activity",
#             "Resource": "resource",
#             "StartTime": "start_time",
#             "EndTime": "end_time",
#         },
#         required=["case_id", "activity", "start_time", "end_time", "resource"],
#     )

#     # 2) Train/test split
#     splits = split_log_train_test(
#         alog_df,
#         original_csv_path=cfg_ds.alog,
#         train_fraction=train_fraction,   # e.g. 0.5, if log_fraction is 0.4 and train_fraction=0.5, then train uses first 20% of cases
#         seed=seed,
#         log_fraction=log_fraction,                # use only first 40% of cases
#     )

#     print(
#         f"[{dataset_name}] Train cases: {splits.train['case_id'].nunique()}, "
#         f"Test cases: {splits.test['case_id'].nunique()}"
#     )

#     # 3) Horizon from train (95th percentile of case duration)
#     horizon = compute_case_duration_horizon(splits.train, percentile=0.95)
#     print(f"[{dataset_name}] Horizon (95th percentile case duration): {horizon}")

#     # 4) Candidate cut times
#     train_cuts_all = choose_candidate_cut_times(
#         splits.train, horizon=horizon, source=timestamp_source
#     )
#     test_cuts_all = choose_candidate_cut_times(
#         splits.test, horizon=horizon, source=timestamp_source
#     )

#     # Subsample to at most N cuts per split
#     MAX_TRAIN_CUTS = 150
#     MAX_TEST_CUTS = 75
#     train_cuts = subsample_cuts_equally(train_cuts_all, MAX_TRAIN_CUTS)
#     test_cuts = subsample_cuts_equally(test_cuts_all, MAX_TEST_CUTS)

#     print(f"[{dataset_name}] #train cut timestamps (after subsampling): {len(train_cuts)}")
#     print(f"[{dataset_name}] #test  cut timestamps (after subsampling): {len(test_cuts)}")

#     # 5) Feature environments
#     train_env = prepare_feature_env(splits.train)
#     test_env = prepare_feature_env(splits.test)

#     # 6) Build samples for train and test
#     train_samples = build_samples_for_split(
#         split_name="train",
#         cfg=cfg,
#         log_df=splits.train,
#         horizon=horizon,
#         feature_env=train_env,
#         cut_times=train_cuts,
#         n_runs=runs_per_cut,
#         out_root=out_root,
#         timestamp_source=timestamp_source,
#         verbose=True,
#     )
#     test_samples = build_samples_for_split(
#         split_name="test",
#         cfg=cfg,
#         log_df=splits.test,
#         horizon=horizon,
#         feature_env=test_env,
#         cut_times=test_cuts,
#         n_runs=runs_per_cut,
#         out_root=out_root,
#         timestamp_source=timestamp_source,
#         verbose=True,
#     )

#     # Save samples
#     train_samples.to_csv(out_root / "train_samples.csv", index=False)
#     test_samples.to_csv(out_root / "test_samples.csv", index=False)

#     # 7) Train clustering models (all based on TRAIN only)
#     target_col = "err_RTD_mean" if error_metric.upper() == "RTD" else "err_cycle_mean"
#     models = train_clustering_models(
#         train_samples,
#         target_col=target_col,
#         n_clusters_basic=n_clusters,
#         n_clusters_advanced=n_clusters,
#         n_clusters_state=n_clusters,
#     )
#     with open(out_root / "cluster_models.json", "w", encoding="utf-8") as fh:
#         json.dump(_to_jsonable(models), fh, indent=2)

#     # 8) Coverage on TRAIN (in-sample)
#     train_eval = apply_models_to_test(models, train_samples, target_col=target_col)
#     train_eval.to_csv(out_root / "train_evaluation.csv", index=False)
#     train_summary = summarise_evaluation(train_eval)

#     # 9) Coverage on TEST (out-of-sample)
#     test_eval = apply_models_to_test(models, test_samples, target_col=target_col)
#     test_eval.to_csv(out_root / "test_evaluation.csv", index=False)
#     test_summary = summarise_evaluation(test_eval)

#     # 10) Write TXT summary (train + test in the same file)
#     summary_txt = out_root / "method_summary.txt"
#     with open(summary_txt, "w", encoding="utf-8") as fh:
#         fh.write("=== TRAIN coverage (in-sample, using training samples) ===\n")
#         fh.write(train_summary.to_string(index=False))
#         fh.write("\n\n=== TEST coverage (out-of-sample) ===\n")
#         fh.write(test_summary.to_string(index=False))
#         fh.write("\n")

#     print("\n=== TRAIN coverage (in-sample, using training samples) ===")
#     print(train_summary.to_string(index=False))
#     print("\n=== TEST coverage (out-of-sample) ===")
#     print(test_summary.to_string(index=False))

#     # 11) Store global metadata
#     meta = {
#         "dataset": dataset_name,
#         "event_log": cfg_ds.alog,
#         "bpmn_model": cfg_ds.model,
#         "sim_params": cfg_ds.params,
#         "total_cases": cfg_ds.total_cases,
#         "horizon_seconds": float(horizon.total_seconds()),
#         "train_fraction": train_fraction,
#         "timestamp_source": timestamp_source,
#         "runs_per_cut": runs_per_cut,
#         "error_metric": error_metric,
#         "n_clusters": n_clusters,
#         "train_num_cuts": len(train_cuts),
#         "test_num_cuts": len(test_cuts),
#     }
#     with open(out_root / "metadata.json", "w", encoding="utf-8") as fh:
#         json.dump(_to_jsonable(meta), fh, indent=2)

#     print(f"\n[{dataset_name}] Finished. Output written to: {out_root}")


def _process_single_dataset(
    dataset_name: str,
    runs_per_cut: int,
    timestamp_source: str,
    error_metric: str,
    output_root: str,
    n_clusters: int = 3,
) -> None:
    cfg_ds = DATASETS[dataset_name]

    out_root = Path(output_root) / dataset_name / generate_short_uuid()
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"\n=== [{dataset_name}] Output root: {out_root} ===")

    # 1) Load logs
    is_real_life = dataset_name in REAL_LIFE_DATASETS

    rename_map = {
        "CaseId": "case_id",
        "Activity": "activity",
        "Resource": "resource",
        "StartTime": "start_time",
        "EndTime": "end_time",
    }

    if is_real_life:
        # REAL-LIFE: read ONE log (you said: use only the "test" file),
        # then split it 50/50 time-wise by cases (no case overlap).
        full_df = read_event_log(
            cfg_ds.test_log,
            rename=rename_map,
            required=["case_id", "activity", "start_time", "end_time", "resource"],
        )

        splits = split_log_train_test(
            full_df,
            original_csv_path=cfg_ds.test_log,
            train_fraction=0.5,
            seed=42,
            log_fraction=1.0,
        )

        train_df = splits.train
        test_df = splits.test

        # Use the auto-written split paths as runner inputs (important!)
        train_log_for_runner = splits.train_csv_path
        test_log_for_runner = splits.test_csv_path

        print(f"earliest timestamp (full): {full_df['start_time'].min()} and latest: {full_df['end_time'].max()}")
        print(
            f"[{dataset_name}] (REAL-LIFE) Split from single log: {cfg_ds.test_log}\n"
            f"  -> train: {train_df['case_id'].nunique()} cases\n"
            f"  -> test : {test_df['case_id'].nunique()} cases"
        )
    else:
        # SYNTHETIC: keep your existing behaviour (separate train/test files).
        train_df = read_event_log(
            cfg_ds.train_log,
            rename=rename_map,
            required=["case_id", "activity", "start_time", "end_time", "resource"],
        )
        test_df = read_event_log(
            cfg_ds.test_log,
            rename=rename_map,
            required=["case_id", "activity", "start_time", "end_time", "resource"],
        )

        train_log_for_runner = cfg_ds.train_log
        test_log_for_runner = cfg_ds.test_log

        print(
            f"[{dataset_name}] Train cases: {train_df['case_id'].nunique()}, "
            f"Test cases: {test_df['case_id'].nunique()}"
        )

    # SimulationConfig still describes how ProSiMoS should run.
    # We use the TRAIN log as the reference event log for simulation.
    cfg_train = SimulationConfig(
        dataset=dataset_name,
        event_log_csv=train_log_for_runner,
        bpmn_model=cfg_ds.bpmn_model,
        sim_params=cfg_ds.sim_params,
        total_cases=cfg_ds.total_cases,
    )

    cfg_test = SimulationConfig(
        dataset=dataset_name,
        event_log_csv=test_log_for_runner,
        bpmn_model=cfg_ds.bpmn_model,
        sim_params=cfg_ds.sim_params,
        total_cases=cfg_ds.total_cases,
    )

    print(f"train log earliest timestamp: {train_df['start_time'].min()} and latest: {train_df['end_time'].max()}")
    print(f"test log earliest timestamp: {test_df['start_time'].min()} and latest: {test_df['end_time'].max()}")

    # 2) Horizon from TRAIN (95th percentile of case duration)
    horizon = compute_case_duration_horizon(train_df, percentile=0.95)
    print(f"[{dataset_name}] Horizon (95th percentile case duration): {horizon}")

    # 3) Candidate cut times (each log has its own safe region, drop first/last H)
    train_cuts_all = choose_candidate_cut_times(
        train_df, horizon=horizon, source=timestamp_source
    )
    test_cuts_all = choose_candidate_cut_times(
        test_df, horizon=horizon, source=timestamp_source
    )

    # Subsample to at most N cuts per split (equally spaced)
    MAX_TRAIN_CUTS = 150
    MAX_TEST_CUTS = 150
    train_cuts = subsample_cuts_equally(train_cuts_all, MAX_TRAIN_CUTS)
    test_cuts = subsample_cuts_equally(test_cuts_all, MAX_TEST_CUTS)

    print(f"[{dataset_name}] #train cut timestamps (after subsampling): {len(train_cuts)}")
    print(f"[{dataset_name}] #test  cut timestamps (after subsampling): {len(test_cuts)}")

    # 4) Feature environments
    train_env = prepare_feature_env(train_df)
    test_env = prepare_feature_env(test_df)

    # 5) Build samples for train and test
    train_samples = build_samples_for_split(
        split_name="train",
        cfg=cfg_train,
        log_df=train_df,
        horizon=horizon,
        feature_env=train_env,
        cut_times=train_cuts,
        n_runs=runs_per_cut,
        out_root=out_root,
        timestamp_source=timestamp_source,
        verbose=True,
    )

    test_samples = build_samples_for_split(
        split_name="test",
        cfg=cfg_test,
        log_df=test_df,
        horizon=horizon,
        feature_env=test_env,
        cut_times=test_cuts,
        n_runs=runs_per_cut,
        out_root=out_root,
        timestamp_source=timestamp_source,
        verbose=True,
    )


    # Save samples
    train_samples.to_csv(out_root / "train_samples.csv", index=False)
    test_samples.to_csv(out_root / "test_samples.csv", index=False)

    # 6) Train clustering models (on TRAIN only)
    target_col = "err_RTD_mean" if error_metric.upper() == "RTD" else "err_cycle_mean"
    models = train_clustering_models(
        train_samples,
        target_col=target_col,
        n_clusters_basic=n_clusters,
        n_clusters_advanced=n_clusters,
        n_clusters_state=n_clusters,
    )
    with open(out_root / "cluster_models.json", "w", encoding="utf-8") as fh:
        json.dump(_to_jsonable(models), fh, indent=2)

    # 7) Coverage on TRAIN (in-sample)
    train_eval = apply_models_to_test(models, train_samples, target_col=target_col)
    train_eval.to_csv(out_root / "train_evaluation.csv", index=False)
    train_summary = summarise_evaluation(train_eval)

    # 8) Coverage on TEST (out-of-sample)
    test_eval = apply_models_to_test(models, test_samples, target_col=target_col)
    test_eval.to_csv(out_root / "test_evaluation.csv", index=False)
    test_summary = summarise_evaluation(test_eval)

    # 9) Write TXT summary
    summary_txt = out_root / "method_summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as fh:
        fh.write("=== TRAIN coverage (in-sample, using training samples) ===\n")
        fh.write(train_summary.to_string(index=False))
        fh.write("\n\n=== TEST coverage (out-of-sample) ===\n")
        fh.write(test_summary.to_string(index=False))
        fh.write("\n")

    print("\n=== TRAIN coverage (in-sample, using training samples) ===")
    print(train_summary.to_string(index=False))
    print("\n=== TEST coverage (out-of-sample) ===")
    print(test_summary.to_string(index=False))

    # 10) Store global metadata
    meta = {
        "dataset": dataset_name,
        "is_real_life": is_real_life,
        "source_single_log": cfg_ds.test_log if is_real_life else None,
        "train_log_used_by_runner": train_log_for_runner,
        "test_log_used_by_runner": test_log_for_runner,
        "bpmn_model": cfg_ds.bpmn_model,
        "sim_params": cfg_ds.sim_params,
        "total_cases": cfg_ds.total_cases,
        "cut": cfg_ds.cut,
        "horizon_days": cfg_ds.horizon_days,
        "horizon_seconds": float(horizon.total_seconds()),
        "timestamp_source": timestamp_source,
        "runs_per_cut": runs_per_cut,
        "error_metric": error_metric,
        "n_clusters": n_clusters,
        "train_num_cuts": len(train_cuts),
        "test_num_cuts": len(test_cuts),
    }
    with open(out_root / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(_to_jsonable(meta), fh, indent=2)

    print(f"\n[{dataset_name}] Finished. Output written to: {out_root}")




# ────────────────────────────────────────────────────────────────────
# 10.  entry-point 
# ────────────────────────────────────────────────────────────────────

# def main() -> None:
#     # -----------------------------
#     # USER-CONFIGURABLE SETTINGS
#     # -----------------------------

#     # One of: a single dataset key from DATASETS (e.g. "BPIC_2012", "P2P_STABLE")
#     # or a group key from ALIASES ("ALL", "SYNTHETIC", "REAL-LIFE").
#     dataset = "BPIC_2017"

#     # Number of simulation replications per cut timestamp
#     runs_per_cut = 5

#     # How to derive cut timestamps: "activity_end" or "case_end"
#     timestamp_source = "case_end"

#     # Which error metric to use as the clustering target: "RTD" or "cycle_time"
#     error_metric = "RTD"

#     # Root folder for all outputs
#     output_root = "outputs_confidence"

#     # Fraction of cases that go into the training split
#     train_fraction = 0.6

#     # Fraction of the log to use (e.g. 0.4 means first 40% of cases only)
#     log_fraction = 1  

#     # Random seed for train/test split
#     seed = 42

#     # Number of clusters for each KMeans / WIP / random model
#     n_clusters = 3

#     # -----------------------------
#     # DRIVER LOGIC (no CLI)
#     # -----------------------------
#     if dataset in ALIASES:
#         for name in ALIASES[dataset]:
#             print(f"\n\n===== Running dataset: {name} =====")
#             _process_single_dataset(
#                 dataset_name=name,
#                 runs_per_cut=runs_per_cut,
#                 timestamp_source=timestamp_source,
#                 error_metric=error_metric,
#                 output_root=output_root,
#                 train_fraction=train_fraction,
#                 seed=seed,
#                 n_clusters=n_clusters,
#                 log_fraction=log_fraction,
#             )
#     else:
#         _process_single_dataset(
#             dataset_name=dataset,
#             runs_per_cut=runs_per_cut,
#             timestamp_source=timestamp_source,
#             error_metric=error_metric,
#             output_root=output_root,
#             train_fraction=train_fraction,
#             seed=seed,
#             n_clusters=n_clusters,
#             log_fraction=log_fraction,
#         )


# if __name__ == "__main__":
#     main()


def main() -> None:
    # -----------------------------
    # USER-CONFIGURABLE SETTINGS
    # -----------------------------

    # Either:
    #  - a single dataset key (e.g. "LOAN_STABLE", "BPIC_2017"), or
    #  - an alias key from ALIASES ("ALL", "SYNTHETIC", "REAL-LIFE").
    dataset = "P2P_UNSTABLE"

    # Number of simulation replications per cut timestamp
    runs_per_cut = 5

    # How to derive cut timestamps: "activity_end" or "case_end"
    timestamp_source = "case_end"

    # Which error metric to use as the clustering target: "RTD" or "cycle_time"
    error_metric = "RTD"

    # Root folder for all outputs
    output_root = "outputs_confidence_v3"

    # Number of clusters for each KMeans / WIP / random model
    n_clusters = 3

    # -----------------------------
    # DRIVER LOGIC
    # -----------------------------
    if dataset in ALIASES:
        for name in ALIASES[dataset]:
            print(f"\n\n===== Running dataset: {name} =====")
            _process_single_dataset(
                dataset_name=name,
                runs_per_cut=runs_per_cut,
                timestamp_source=timestamp_source,
                error_metric=error_metric,
                output_root=output_root,
                n_clusters=n_clusters,
            )
    else:
        _process_single_dataset(
            dataset_name=dataset,
            runs_per_cut=runs_per_cut,
            timestamp_source=timestamp_source,
            error_metric=error_metric,
            output_root=output_root,
            n_clusters=n_clusters,
        )

if __name__ == "__main__":
    main()