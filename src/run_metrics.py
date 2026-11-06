import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from prosimos.simulation_properties_parser import parse_resource_calendars

from src.compute_frontend_events_from_trace import sim_log_ids
from src.scenario import pool_members

HOUR_MS = 3600000
MAX_SAMPLES = 1500
MAX_COMPARISON_SAMPLES = 300
OCCUPANCY_POINTS = 20000
MINUTES_PER_WEEK = 7 * 24 * 60
MONDAY = datetime.datetime(2024, 1, 1)


def sample_step_ms(start: pd.Timestamp, end: pd.Timestamp, max_samples: int = MAX_SAMPLES) -> int:
    total_ms = max((end - start) / pd.Timedelta(milliseconds=1), 0)
    return max(HOUR_MS, int(np.ceil(total_ms / max_samples)))


def _sample_times(start: pd.Timestamp, end: pd.Timestamp, step_ms: int) -> np.ndarray:
    count = int((end - start) / pd.Timedelta(milliseconds=step_ms)) + 1
    return _to_ms(start) + np.arange(count, dtype=np.int64) * step_ms


def _to_ms(timestamp: pd.Timestamp) -> int:
    return int(timestamp.timestamp() * 1000)


def _column_ms(column: pd.Series) -> np.ndarray:
    return (pd.to_datetime(column, utc=True) - pd.Timestamp(0, tz="UTC")).to_numpy() / np.timedelta64(1, "ms")


def _queue_series(enabled_ms: np.ndarray, started_ms: np.ndarray, times: np.ndarray) -> np.ndarray:
    return (
        np.searchsorted(np.sort(enabled_ms), times, side="right")
        - np.searchsorted(np.sort(started_ms), times, side="right")
    )


def summarize_log(log: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, ongoing_arrivals: Dict[str, str]) -> dict:
    start_ms, end_ms = _to_ms(start), _to_ms(end)
    window_ms = max(end_ms - start_ms, 1)
    step_ms = sample_step_ms(start, end, MAX_COMPARISON_SAMPLES)
    times = _sample_times(start, end, step_ms)

    enabled = _column_ms(log[sim_log_ids.enabled_time])
    started = _column_ms(log[sim_log_ids.start_time])
    ended = _column_ms(log[sim_log_ids.end_time])
    activities = log[sim_log_ids.activity].to_numpy()
    case_ids = log[sim_log_ids.case].astype(str).to_numpy()

    counted = (started >= start_ms) & (enabled <= end_ms)
    waits = np.maximum(started - enabled, 0)
    queued_ms = np.clip(np.minimum(started, end_ms) - np.maximum(enabled, start_ms), 0, None)

    tasks = {}
    total_series = np.zeros(len(times), dtype=np.int64)
    for activity in pd.unique(activities):
        here = activities == activity
        series = _queue_series(enabled[here], started[here], times)
        total_series += series
        waits_here = waits[here & counted]
        tasks[str(activity)] = {
            "count": int(len(waits_here)),
            "waitMs": float(waits_here.mean()) if len(waits_here) else None,
            "queueAverage": float(queued_ms[here].sum() / window_ms),
            "queuePeak": int(series.max()) if len(series) else 0,
            "queue": series.tolist(),
        }

    arrivals = pd.Series(enabled).groupby(case_ids).min()
    case_ends = pd.Series(ended).groupby(case_ids).max()
    for case_id, arrival in ongoing_arrivals.items():
        if case_id in arrivals.index:
            arrivals[case_id] = min(arrivals[case_id], _to_ms(pd.Timestamp(arrival)))
    ongoing = arrivals.index.isin(list(ongoing_arrivals))
    waits_counted = waits[counted]

    return {
        "startDate": start.isoformat(),
        "endDate": end.isoformat(),
        "stepMs": step_ms,
        "arrived": int((~ongoing).sum()),
        "finished": int((case_ends <= end_ms).sum()),
        "openAtEnd": int((case_ends > end_ms).sum()),
        "cycleMs": float((case_ends - arrivals).mean()) if len(arrivals) else None,
        "waitMs": float(waits_counted.mean()) if len(waits_counted) else None,
        "queueAverage": float(queued_ms.sum() / window_ms),
        "queuePeak": int(total_series.max()) if len(total_series) else 0,
        "waitingAtEnd": int(((enabled <= end_ms) & (started > end_ms)).sum()),
        "queue": total_series.tolist(),
        "tasks": tasks,
    }


def _weekly_masks(parameters: dict) -> Dict[str, np.ndarray]:
    if parameters.get("model_type") == "FUZZY":
        return {}
    masks = {}
    for calendar_id, calendar in parse_resource_calendars(parameters["resource_calendars"]).items():
        moments = [MONDAY + datetime.timedelta(minutes=minute) for minute in range(MINUTES_PER_WEEK)]
        masks[calendar_id] = np.array([calendar.is_working_datetime(moment)[0] for moment in moments])
    return masks


def _minute_of_week(times_ms: np.ndarray, offset: datetime.timedelta) -> np.ndarray:
    local_minutes = (times_ms + offset.total_seconds() * 1000) // 60000
    monday_minutes = _to_ms(pd.Timestamp(MONDAY, tz="UTC")) // 60000
    return ((local_minutes - monday_minutes) % MINUTES_PER_WEEK).astype(np.int64)


def _merged_intervals(starts: np.ndarray, ends: np.ndarray):
    order = np.argsort(starts)
    merged_starts, merged_ends = [], []
    for begin, finish in zip(starts[order], ends[order]):
        if merged_ends and begin <= merged_ends[-1]:
            merged_ends[-1] = max(merged_ends[-1], finish)
        else:
            merged_starts.append(begin)
            merged_ends.append(finish)
    return np.array(merged_starts), np.array(merged_ends)


def resource_timeline(log: pd.DataFrame, parameters: dict, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    step_ms = sample_step_ms(start, end)
    grid_ms = max(60000, int(np.ceil(max(_to_ms(end) - _to_ms(start), 1) / OCCUPANCY_POINTS)))
    grid = np.arange(_to_ms(start), _to_ms(end) + 1, grid_ms, dtype=np.int64)
    stride = max(1, step_ms // grid_ms)

    masks = _weekly_masks(parameters)
    minute_of_week = _minute_of_week(grid, start.utcoffset() or datetime.timedelta(0))
    always_on = np.ones(len(grid), dtype=bool)

    log_started = _column_ms(log[sim_log_ids.start_time])
    log_ended = _column_ms(log[sim_log_ids.end_time])
    log_resources = log[sim_log_ids.resource].astype(str).to_numpy()

    pools = []
    for pool in parameters["resource_profiles"]:
        members = pool_members(pool)
        busy_total = np.zeros(len(grid), dtype=np.int64)
        idle_total = np.zeros(len(grid), dtype=np.int64)
        resources = []
        for member in members:
            name = member["name"]
            weekly = masks.get(member["calendar"])
            working = weekly[minute_of_week] if weekly is not None else always_on
            own = log_resources == name
            busy = np.zeros(len(grid), dtype=bool)
            if own.any():
                starts, ends = _merged_intervals(log_started[own], log_ended[own])
                slots = np.searchsorted(starts, grid, side="right") - 1
                busy = (slots >= 0) & (grid < ends[np.maximum(slots, 0)])
            busy_at_work = busy & working
            busy_total += busy_at_work
            idle_total += working & ~busy
            available = int(working.sum())
            resources.append({
                "name": name,
                "utilization": float(busy_at_work.sum() / available) if available else None,
            })
        available_points = int(busy_total.sum() + idle_total.sum())
        pools.append({
            "id": pool["id"],
            "name": pool["name"],
            "size": len(members),
            "utilization": float(busy_total.sum() / available_points) if available_points else None,
            "busy": busy_total[::stride].tolist(),
            "idle": idle_total[::stride].tolist(),
            "resources": resources,
        })

    return {
        "startDate": start.isoformat(),
        "endDate": end.isoformat(),
        "stepMs": int(stride * grid_ms),
        "calendarsKnown": bool(masks),
        "pools": pools,
    }


def paired_change(pairs: List[dict]) -> dict:
    changes = {}
    for metric in ["waitMs", "queueAverage", "cycleMs", "finished"]:
        values = [
            pair["toBe"][metric] - pair["asIs"][metric] for pair in pairs
            if pair["toBe"][metric] is not None and pair["asIs"][metric] is not None
        ]
        changes[metric] = {
            "mean": float(np.mean(values)) if values else None,
            "low": float(min(values)) if values else None,
            "high": float(max(values)) if values else None,
        }
    return {"seeds": len(pairs), "changes": changes}
