# evaluation/features.py
"""Feature engineering utilities for computing process-state features at cut timestamps."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass(slots=True)
class FeatureEnv:
    """Precomputed helpers that speed-up feature construction."""
    case_arrival: pd.Series            # index: case_id, value: first start_time
    resources: np.ndarray              # unique resources
    activities: np.ndarray             # unique activities
    has_enable_time: bool
    activity_name_map: Dict[str, str]  # original -> safe column suffix


def _safe_activity_name(name: str) -> str:
    """
    Make an activity name safe to use in column names:
    - non-alphanumeric -> '_'
    - strip leading/trailing '_'
    """
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_") or "act"


def prepare_feature_env(df: pd.DataFrame) -> FeatureEnv:
    """Construct a FeatureEnv from an event log DataFrame."""
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
    - arrival rate per hour over the previous horizon
    - resource availability percentage
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
