# helper.py
from __future__ import annotations
import os, random, string, pandas as pd, numpy as np
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


# def _build_ps_subsets(df: pd.DataFrame,
#                       partial_ids: set[str],
#                       cut: pd.Timestamp,
#                       end: pd.Timestamp):
#     """
#     Replicates the original logic from *build_partial_state_subsets()*.
#     """
#     # 1) events that overlap the window
#     event_filter = trim_events(df, cut, end)

#     # 2) ongoing: only cases in *partial_ids*, clip to the window
#     # ongoing = df[df["case_id"].astype(str).isin(partial_ids)].copy()
#     # ongoing.loc[ongoing["start_time"] < cut, "start_time"] = cut
#     ongoing_candidates = df[df["case_id"].astype(str).isin(partial_ids)]
#     still_open_ids = (
#         ongoing_candidates.groupby("case_id")["end_time"].max()
#         .loc[lambda s: s > cut]                 #  ← key line
#         .index
#     )
#     ongoing = ongoing_candidates[
#         ongoing_candidates["case_id"].isin(still_open_ids)
#     ].copy()
#     ongoing.loc[ongoing["start_time"] < cut, "start_time"] = cut

#     # 3) complete: cases NOT in *partial_ids* that start inside the window
#     rest = df[~df["case_id"].astype(str).isin(partial_ids)].copy()
#     first_start = rest.groupby("case_id")["start_time"].transform("min")
#     complete = rest[(first_start >= cut) & (first_start < end)]

#     return event_filter, ongoing, complete


def _build_ps_subsets(df: pd.DataFrame,
                      partial_ids: set[str],
                      cut: pd.Timestamp,
                      end: pd.Timestamp):
    """
    Builds event_filter / ongoing / complete for the *process-state* flavour.

    ▸ We keep the original event_filter logic unchanged.  
    ▸ For **ongoing**:
        1.  start with the ids in *process_state.json*;
        2.  ADD any other cases that log evidence shows are still open
            (start < cut < end);
        3.  clip their start times to the window.
    ▸ For **complete**, we take the remaining cases that begin inside
      [cut, end).
    """
    # 1) events that overlap the window
    event_filter = trim_events(df, cut, end)

    # 2) assemble the definitive set of “still running” case-ids
    snapshot_ids = {str(x) for x in partial_ids}        # simulator view
    truly_open = (
        df.groupby("case_id")
          .agg(first=("start_time", "min"), last=("end_time", "max"))
          .loc[lambda x: (x["first"] < cut) & (x["last"] > cut)]
          .index.astype(str)                            # log-based view
    )
    ongoing_ids = snapshot_ids | set(truly_open)

    # 3) slice the DataFrame
    ongoing = df[df["case_id"].astype(str).isin(ongoing_ids)].copy()
    ongoing.loc[ongoing["start_time"] < cut, "start_time"] = cut

    #commented idle activity in cases
    # have = set(ongoing["case_id"].astype(str))
    # idle = ongoing_ids - have          
    # if idle:
    #     stub = pd.DataFrame({
    #         "case_id":   list(idle),
    #         "activity":  "__idle__",   
    #         "resource":  None,
    #         "start_time": [cut] * len(idle),
    #         "end_time":   [cut] * len(idle),
    #     })
    #     ongoing = pd.concat([ongoing, stub], ignore_index=True)

    # 4) complete = everybody else that starts inside the window
    rest = df[~df["case_id"].astype(str).isin(ongoing_ids)].copy()
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


def compute_cut_points(
    log_df: pd.DataFrame,
    horizon_days: int,
    *,
    strategy: str = "fixed",
    fixed_cut: str | None = None,
    rng: np.random.Generator | None = None,
) -> list[pd.Timestamp]:
    """
    Return a list of cut-off timestamps according to *strategy*.

    Strategies
    ----------
    fixed
        Exactly one timestamp, taken from *fixed_cut*.
    wip3
        Three timestamps where the Work-in-Process equals 10 %, 50 %, and
        90 % of the maximum observed WiP.
    segment10
        Ten timestamps: drop the first and last *horizon* and divide the
        remaining interval into ten equal segments; pick one random moment
        from each segment.
    """
    if strategy == "fixed":
        if fixed_cut is None:
            raise ValueError("strategy 'fixed' needs a cut-off timestamp")
        return [pd.to_datetime(fixed_cut, utc=True)]

    rng = rng or np.random.default_rng()

    first_ts = log_df["start_time"].min()
    last_ts  = log_df["end_time"].max()

    safe_start = first_ts + pd.Timedelta(days=horizon_days)
    safe_end   = last_ts  - pd.Timedelta(days=horizon_days)
    if safe_start >= safe_end:
        raise ValueError("the event log is shorter than twice the horizon")

    # helper: active cases at a given time
    case_bounds = log_df.groupby("case_id").agg(
        start=("start_time", "min"),
        end=("end_time",   "max"),
    )
    def active_cases_at(ts: pd.Timestamp) -> int:
        mask = (case_bounds["start"] <= ts) & (case_bounds["end"] > ts)
        return int(mask.sum())

    if strategy == "wip3":
        # evaluate WiP only at case arrival moments
        arrivals = case_bounds["start"].sort_values()
        wip_series = pd.Series(
            {ts: active_cases_at(ts) for ts in arrivals}
        )
        max_wip = wip_series.max()
        targets = [int(round(max_wip * q)) for q in (0.10, 0.50, 0.90)]

        cuts: list[pd.Timestamp] = []
        for tgt in targets:
            exact = wip_series[wip_series == tgt]
            if not exact.empty:
                cuts.append(exact.index[0])
                continue
            greater = wip_series[wip_series > tgt]
            if not greater.empty:
                cuts.append(greater.index[0])
                continue
            cuts.append(wip_series.index[0]) 
        return cuts

    if strategy == "segment10":
        span = safe_end - safe_start
        # span = safe_end - first_ts
        segment_length = span / 10
        cuts: list[pd.Timestamp] = []
        for i in range(10):
            seg_start = safe_start + i * segment_length
            # seg_start = first_ts + i * segment_length
            jitter = rng.uniform(0, segment_length.total_seconds())
            cuts.append(seg_start + pd.Timedelta(seconds=float(jitter)))
        return cuts

    raise ValueError(f"unknown cut strategy: {strategy}")


def build_aggregated_from_cuts(metric_dicts: list[dict]) -> dict:
    """
    Combine several `aggregated[<subfamily>]` dictionaries – one from each
    cut-off – into a single dictionary that has the same structure.

    Input
    -----
    metric_dicts : list of dict
        Each element looks like
        {
          'n_gram':           {'mean': 0.37, 'ci': 0.04},
          'absolute_event':   {'mean': 0.12, 'ci': 0.01},
          ...
        }

    Output
    ------
    dict
        {
          'n_gram':         {'mean': <average of the means>},
          'absolute_event': {'mean': <average of the means>},
          ...
        }
        (the *ci* values are ignored because they are not meaningful after
        averaging averages).
    """
    result: dict = {}
    all_metrics = {m for d in metric_dicts for m in d}
    for m in all_metrics:
        means = [
            d[m]["mean"]
            for d in metric_dicts
            if m in d and d[m]["mean"] is not None
        ]
        result[m] = {"mean": float(np.mean(means))} if means else {"mean": None}
    return result
