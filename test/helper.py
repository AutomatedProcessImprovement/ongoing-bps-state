import os
import pandas as pd
import random
import string


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
            df[tcol] = pd.to_datetime(df[tcol], utc=True, errors="coerce")

    # Check required columns
    if required_columns:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    return df


def filter_cases_by_eval_window(df: pd.DataFrame, eval_start: pd.Timestamp, eval_end: pd.Timestamp) -> pd.DataFrame:
    """
    Keep entire cases if at least one event has its start or end time 
    within [eval_start, eval_end].
    """
    keep_case_ids = []
    for cid, group in df.groupby("case_id"):
        if (
            ((group["start_time"] >= eval_start) & (group["start_time"] <= eval_end)).any()
            or ((group["end_time"] >= eval_start) & (group["end_time"] <= eval_end)).any()
        ):
            keep_case_ids.append(cid)
    return df[df["case_id"].isin(keep_case_ids)].copy()


def trim_events_to_eval_window(df: pd.DataFrame, eval_start: pd.Timestamp, eval_end: pd.Timestamp) -> pd.DataFrame:
    """
    For each event, ensure times are within [eval_start, eval_end].
    Only keep events that overlap with the window in any way.
    """
    out = df.copy()
    out = out[(out["end_time"] > eval_start) & (out["start_time"] < eval_end)]
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
