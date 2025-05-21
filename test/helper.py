# helper.py
from __future__ import annotations
import os, random, string, pandas as pd
from typing import Callable
from log_distance_measures.config import EventLogIDs

# ---------- General utils -------------------------------------------------

def generate_short_uuid(k: int = 6) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=k))

def run_with_retries(fn: Callable, kwargs: dict, attempts: int = 3, /, *, verbose=True):
    """Call *fn(**kwargs)*, retrying up to *attempts* times."""
    for i in range(1, attempts + 1):
        try:
            return fn(**kwargs)
        except Exception as e:
            if i == attempts:
                raise
            if verbose:
                print(f"[retry] attempt {i}/{attempts} failed: {e}. Retrying…")

# ---------- I/O -----------------------------------------------------------

def read_event_log(
    csv_path: str,
    rename: dict[str, str] | None = None,
    required: list[str] | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if rename:
        df.rename(columns=rename, inplace=True)

    for col in ("enable_time", "start_time", "end_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)

    if required and (missing := [c for c in required if c not in df.columns]):
        raise ValueError(f"Missing required cols in {csv_path}: {missing}")
    return df

# ---------- Window helpers ------------------------------------------------

def trim_events(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Return events that overlap [start,end] and clip their boundaries."""
    mask = (df["end_time"] >= start) & (df["start_time"] <= end)
    out = df.loc[mask].copy()
    out.loc[out["start_time"] < start, "start_time"] = start
    out.loc[out["end_time"]   > end,   "end_time"]   = end
    return out


def _cat(frags: list[pd.DataFrame], proto: pd.DataFrame) -> pd.DataFrame:
    """
    Safe concatenation:
      • if *frags* is empty → return an empty DataFrame
        that preserves the original column order & dtypes;
      • else → normal concat.
    """
    if frags:
        return pd.concat(frags, ignore_index=True)
    else:
        # Empty but with the same columns + dtypes as *proto*
        return proto.iloc[0:0].copy()


def _split_cases(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    ongoing, complete = [], []
    for _, g in df.groupby("case_id"):
        first, last = g["start_time"].min(), g["end_time"].max()
        if first < start < last:                 # “ongoing” at the cutoff
            ongoing.append(g[g["end_time"] > start])
        elif start <= first <= end:              # “complete” in window
            complete.append(g)

    return _cat(ongoing, df), _cat(complete, df)

def split_into_subsets(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp):
    """A_event_filter, A_ongoing, A_complete (same semantics as before)."""
    event_filter = trim_events(df, start, end)
    ongoing, complete = _split_cases(df, start, end)
    return event_filter, ongoing, complete

# ---------- Misc ----------------------------------------------------------

def avg_remaining_time(log: pd.DataFrame, cutoff: pd.Timestamp, ids: EventLogIDs) -> float:
    """Mean remaining seconds per case after *cutoff*."""
    secs = [
        (g[ids.end_time].max() - cutoff).total_seconds()
        for _, g in log.groupby(ids.case)
    ]
    return float(sum(secs) / len(secs)) if secs else 0.0


# ── helpers specific to the process-state flavour ──────────────────
from pathlib import Path
import json

def _parse_partial_state_json(sim_dir: Path) -> set[str]:
    """
    Return the set of case-ids that were already running at the cut-off.
    First try <sim_dir>/process_state.json (what the wrapper copies);
    fall back to the global 'output.json' for older runs.
    """
    for p in (sim_dir / "process_state.json", Path("output.json")):
        if p.is_file():
            try:
                with p.open(encoding="utf-8") as fh:
                    data = json.load(fh)
                return set(data.get("cases", {}).keys())
            except Exception:
                pass
    return set()


def _build_ps_subsets(df: pd.DataFrame,
                      partial_ids: set[str],
                      cut: pd.Timestamp,
                      end: pd.Timestamp):
    """
    Replicates the original logic from *build_partial_state_subsets()*.
    """
    # 1) events that overlap the window
    event_filter = trim_events(df, cut, end)

    # 2) ongoing: only cases in *partial_ids*, clip to the window
    ongoing = df[df["case_id"].astype(str).isin(partial_ids)].copy()
    ongoing.loc[ongoing["start_time"] < cut, "start_time"] = cut

    # 3) complete: cases NOT in *partial_ids* that start inside the window
    rest = df[~df["case_id"].astype(str).isin(partial_ids)].copy()
    first_start = rest.groupby("case_id")["start_time"].transform("min")
    complete = rest[(first_start >= cut) & (first_start < end)]

    return event_filter, ongoing, complete


def _avg_events_per_ongoing_case(df: pd.DataFrame) -> float | None:
    """
    Mean number of events per ongoing case in *df*.

    Returns
    -------
    float | None
        • mean value as float;
        • None if *df* is empty or has no case rows.
    """
    if df.empty:
        return None
    cases = df["case_id"].nunique()
    return float(len(df) / cases) if cases else None


def _avg_events_per_case(df: pd.DataFrame) -> float | None:
    """
    Average number of events per case in *df*.
    Returns None if *df* is empty.
    """
    if df.empty:
        return None
    cases = df["case_id"].nunique()
    return float(len(df) / cases) if cases else None


def _avg_events_per_case_diff(A: pd.DataFrame, G: pd.DataFrame) -> float | None:
    """
    Absolute difference between the reference and simulated averages.
    """
    a = _avg_events_per_case(A)
    g = _avg_events_per_case(G)
    return abs(a - g) if (a is not None and g is not None) else None