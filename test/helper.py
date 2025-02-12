# helper.py

import pandas as pd
import numpy as np
import os

def read_event_log(csv_path, column_mapping=None):
    """
    Reads an event log CSV and applies the column mapping so that
    the returned DataFrame has these standard columns:
      - case_id
      - activity
      - resource
      - enable_time
      - start_time
      - end_time

    Example column_mapping:
      {
        "case_id": "CaseId",
        "activity": "Activity",
        "resource": "Resource",
        "enable_time": "EnabledTime",
        "start_time": "StartTime",
        "end_time": "EndTime"
      }

    If some keys are missing, we fall back to the default (same as keys).
    """

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Event log CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Default standard columns = themselves
    default_map = {
        "case_id": "case_id",
        "activity": "activity",
        "resource": "resource",
        "enable_time": "enable_time",
        "start_time": "start_time",
        "end_time": "end_time",
    }
    if column_mapping is None:
        column_mapping = {}

    # Build the rename dict for df.rename(...)
    rename_dict = {}
    for std_col, user_col in default_map.items():
        # If user gave a mapping for std_col, we use that. Otherwise fallback
        if std_col in column_mapping:
            rename_dict[column_mapping[std_col]] = std_col
        else:
            # if the user didn't provide a mapping, we assume the DF is already correct
            # meaning user_col == std_col
            # but if user_col != std_col in DF, we do an additional check
            if user_col in df.columns:
                # then we do nothing, columns match
                pass
            else:
                # There's no column for it. Possibly missing?
                # We'll rely on code below to handle missing columns gracefully.
                pass

    # Actually rename
    df = df.rename(columns=rename_dict)

    # Convert time columns to datetime if present
    time_cols = ["enable_time", "start_time", "end_time"]
    for tcol in time_cols:
        if tcol in df.columns:
            df[tcol] = pd.to_datetime(df[tcol], utc=True, errors="coerce")

    return df



def preprocess_alog(df, start_time=None, horizon=None):
    """
    1) remove events that ended before 'start_time' => means they finished too early
    2) remove entire cases that only start after horizon
    """
    out = df.copy()

    if start_time is not None and "end_time" in out.columns:
        out = out[~(out["end_time"] < start_time)]

    if horizon is not None and "start_time" in out.columns:
        min_st = out.groupby("case_id")["start_time"].transform("min")
        out = out[~(min_st > horizon)]

    return out


def preprocess_glog(df, horizon=None):
    """
    remove cases that only start after horizon
    """
    out = df.copy()
    if horizon is not None and "start_time" in out.columns:
        min_st = out.groupby("case_id")["start_time"].transform("min")
        out = out[~(min_st > horizon)]
    return out


def basic_log_stats(df):
    n_cases = df["case_id"].nunique() if "case_id" in df.columns else 0
    n_events = len(df)
    earliest = df["start_time"].min() if "start_time" in df.columns and not df.empty else None
    latest = df["end_time"].max() if "end_time" in df.columns and not df.empty else None

    return {
        "cases": n_cases,
        "events": n_events,
        "earliest_start": str(earliest) if earliest is not None else None,
        "latest_end": str(latest) if latest is not None else None,
    }

def filter_cases_by_eval_window(df: pd.DataFrame, eval_start: pd.Timestamp, eval_end: pd.Timestamp) -> pd.DataFrame:
    """
    Keep entire cases if at least one event has its start or end time within the evaluation window.
    (A case is kept if any event in the group satisfies:
       eval_start <= start_time <= eval_end  OR  eval_start <= end_time <= eval_end)
    """
    keep_case_ids = []
    for cid, group in df.groupby("case_id"):
        if (((group["start_time"] >= eval_start) & (group["start_time"] <= eval_end)).any() or 
            ((group["end_time"] >= eval_start) & (group["end_time"] <= eval_end)).any()):
            keep_case_ids.append(cid)
    return df[df["case_id"].isin(keep_case_ids)].copy()


def trim_events_to_eval_window(df: pd.DataFrame, eval_start: pd.Timestamp, eval_end: pd.Timestamp) -> pd.DataFrame:
    """
    For each event in df, adjust the event’s times so that:
      - If an event starts before eval_start, set its start_time = eval_start.
      - If an event ends after eval_end, set its end_time = eval_end.
    Also keep only events that overlap the evaluation window.
    """
    df = df.copy()
    # Keep only events that actually overlap the evaluation window.
    df = df[(df["end_time"] > eval_start) & (df["start_time"] < eval_end)]
    # Trim start and end times
    df.loc[df["start_time"] < eval_start, "start_time"] = eval_start
    df.loc[df["end_time"] > eval_end, "end_time"] = eval_end
    return df