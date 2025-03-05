import os
import pandas as pd
import random
import string
from log_distance_measures.config import EventLogIDs


def read_event_log(csv_path, rename_map=None, required_columns=None, verbose=True):
    """
    Reads an event log CSV and optionally renames columns.
    Converts time columns to datetime.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Event log CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    # Convert time columns
    for tcol in ["enable_time", "start_time", "end_time"]:
        if tcol in df.columns:
            df[tcol] = pd.to_datetime(df[tcol], utc=True, format="ISO8601")

    # Check required columns
    if required_columns:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    return df


def trim_events_to_eval_window(df: pd.DataFrame, eval_start: pd.Timestamp, eval_end: pd.Timestamp) -> pd.DataFrame:
    """
    For each event, ensure times are within [eval_start, eval_end].
    Only keep events that overlap the evaluation window.
    This additionally eliminates events that start and finish completely
    outside the evaluation window (and thus, cases having only such events).
    """
    out = df.copy()
    
    # Keep only events that have any overlap with the evaluation window:
    # - Events that end after eval_start and start before eval_end.
    mask = (out["end_time"] >= eval_start) & (out["start_time"] <= eval_end)
    out = out[mask].copy()
    
    # Trim events that extend beyond the evaluation window:
    out.loc[out["start_time"] < eval_start, "start_time"] = eval_start
    out.loc[out["end_time"] > eval_end, "end_time"] = eval_end
    
    return out


def basic_log_stats(df):
    """
    Simple stats about number of cases, number of events, earliest start, latest end.
    """
    n_cases = df["case_id"].nunique() if "case_id" in df.columns else 0
    n_events = len(df)
    earliest = df["start_time"].min() if not df.empty else None
    latest = df["end_time"].max() if not df.empty else None

    return {
        "cases": n_cases,
        "events": n_events,
        "earliest_start": str(earliest) if earliest is not None else None,
        "latest_end": str(latest) if latest is not None else None,
    }


def generate_short_uuid(length=6):
    """
    Generate a short random ID. If you have shortuuid, you can replace this 
    with the shortuuid usage directly.
    """
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))


def run_simulation_with_retries(sim_func, sim_func_kwargs, max_attempts=3, verbose=True):
    """
    Helper function to run a simulation with multiple retries.
    Raises an exception if all attempts fail.
    """
    attempt = 1
    while attempt <= max_attempts:
        try:
            sim_func(**sim_func_kwargs)
            return
        except Exception as e:
            if attempt == max_attempts:
                raise RuntimeError(f"Simulation failed after {max_attempts} attempts: {e}") from e
            else:
                if verbose:
                    print(f"[Simulation] Attempt {attempt} failed: {e}. Retrying...")
                attempt += 1

def filter_ongoing_cases(df: pd.DataFrame, eval_start: pd.Timestamp, eval_end: pd.Timestamp) -> pd.DataFrame:
    """
    Return only the part of cases that were 'ongoing' at eval_start:
      - A case is 'ongoing' if it has at least one event starting before eval_start
        AND at least one event ending after eval_start.
      - Then keep only events with end_time > eval_start.
      - Finally, adjust (clip) each event's start and end times so that they lie within [eval_start, eval_end].
    
    Additionally, for each kept case, an artificial event is added with start_time = end_time = eval_start.
    """
    out = df.copy()
    
    # Identify cases that are ongoing at eval_start.
    keep_case_ids = []
    for cid, group in out.groupby("case_id"):
        min_start = group["start_time"].min()
        max_end = group["end_time"].max()
        if min_start < eval_start and max_end > eval_start:
            keep_case_ids.append(cid)
    
    # Filter to only ongoing cases.
    out = out[out["case_id"].isin(keep_case_ids)].copy()
    
    # Keep only events that end after eval_start.
    out = out[out["end_time"] > eval_start].copy()
    
    # Clip event times to the evaluation window.
    out.loc[out["start_time"] < eval_start, "start_time"] = eval_start

    is_anomaly = out["start_time"] > out["end_time"]
    if is_anomaly.any():
        print(f"[filter_ongoing_cases] Found {is_anomaly.sum()} row(s) where start_time > end_time.")
        for idx, row in out[is_anomaly].iterrows():
            print(f"[filter_ongoing_cases]  Anomaly idx={idx}: case_id={row['case_id']}, start_time={row['start_time']}, end_time={row['end_time']}")
    
    return out




def filter_complete_cases(df: pd.DataFrame, eval_start: pd.Timestamp, eval_end: pd.Timestamp) -> pd.DataFrame:
    """
    Return only the full cases whose min(start_time) >= eval_start and min(start_time) <= eval_end.
    Keep ALL events for those cases (even if they extend beyond eval_end).
    """
    out = df.copy()
    keep_case_ids = []
    for cid, group in out.groupby("case_id"):
        min_st = group["start_time"].min()
        if min_st >= eval_start and min_st <= eval_end:
            keep_case_ids.append(cid)

    # print(f"Keeping cases: {keep_case_ids}")
    return out[out["case_id"].isin(keep_case_ids)].copy()

def compute_avg_remaining_time(log: pd.DataFrame, cutoff: pd.Timestamp, ids: EventLogIDs) -> float:
    """
    Computes the average remaining time (in seconds) for a log.
    The remaining time for a case is the difference between its last event end and the cutoff.
    """
    durations = []
    for case, group in log.groupby(ids.case):
        # Compute remaining time in seconds
        remaining = (group[ids.end_time].max() - cutoff).total_seconds()
        durations.append(remaining)
    if durations:
        return sum(durations) / len(durations)
    else:
        return 0.0
