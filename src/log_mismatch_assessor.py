from __future__ import annotations

"""
Assess mismatches between ground-truth and simulated interval event logs.

Usage:
  assessment = assess_event_logs(ground_log, sim_log, options)

Options:
  - views: list of view names to compute.
  - top_k_contributors: int for per-interval top contributors.
  - normalize_time: True or {"start": ..., "end": ...} to normalize timestamps.
  - field_map: mapping for case/activity/start/end/resource/enable keys.
  - role_mapping: dict mapping resource -> role.
  - case_active_mode: "interval" (default) or "running".
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None


DEFAULT_FIELD_MAP = {
    "case": "case_id",
    "activity": "activity",
    "start": "start_time",
    "end": "end_time",
    "resource": "resource",
    "enable": "enable_time",
}

VIEW_ACTIVITY = "activity"
VIEW_ACTIVITY_CASE = "activity_case"
VIEW_ACTIVITY_RESOURCE = "activity_resource"
VIEW_ACTIVITY_CASE_RESOURCE = "activity_case_resource"
VIEW_RESOURCE = "resource"
VIEW_CASE = "case"
VIEW_ACTIVITY_ROLE = "activity_role"
VIEW_ACTIVITY_PHASE = "activity_phase"

DEFAULT_VIEWS = [
    VIEW_ACTIVITY,
    VIEW_ACTIVITY_CASE,
    VIEW_ACTIVITY_RESOURCE,
    VIEW_ACTIVITY_CASE_RESOURCE,
    VIEW_RESOURCE,
    VIEW_CASE,
    VIEW_ACTIVITY_ROLE,
    VIEW_ACTIVITY_PHASE,
]


@dataclass
class NormalizedEvent:
    case_id: Any
    activity: Any
    start: float
    end: float
    resource: Any
    role: Any
    enable: Optional[float]


def assess_event_logs(ground_log: Any, sim_log: Any, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Compare two interval-event logs using multiple multiset views.

    Options keys:
      - views: list of view names to compute.
      - top_k_contributors: int for per-interval top contributors.
      - normalize_time: True or {"start": ..., "end": ...} to normalize timestamps.
      - field_map: mapping for case/activity/start/end/resource/enable keys.
      - role_mapping: dict mapping resource -> role.
      - case_active_mode: "interval" (default) or "running".
    """
    options = options or {}
    views = _normalize_view_list(options.get("views"))
    field_map = {**DEFAULT_FIELD_MAP, **(options.get("field_map") or {})}
    role_mapping = options.get("role_mapping") or {}
    top_k = options.get("top_k_contributors")
    case_active_mode = options.get("case_active_mode", "interval")

    ground_events, ground_meta = _normalize_log(
        ground_log, field_map=field_map, role_mapping=role_mapping
    )
    sim_events, sim_meta = _normalize_log(
        sim_log, field_map=field_map, role_mapping=role_mapping
    )

    _apply_enable_fallback(ground_events)
    _apply_enable_fallback(sim_events)

    time_norm = _build_time_normalizer(options.get("normalize_time"), ground_events, sim_events)
    if time_norm:
        _normalize_event_times(ground_events, time_norm)
        _normalize_event_times(sim_events, time_norm)

    assessment = {
        "meta": {
            "invalid_intervals": {
                "ground": ground_meta["invalid_intervals"],
                "sim": sim_meta["invalid_intervals"],
            },
            "time_normalization": time_norm or None,
        },
        "views": {},
    }

    if not views:
        views = _default_views_for_data(ground_events, sim_events, role_mapping)

    for view in views:
        view_name = _canonical_view_name(view)
        if view_name == VIEW_ACTIVITY_ROLE and not role_mapping:
            continue
        if view_name == VIEW_ACTIVITY_PHASE and not _has_enable_data(ground_events, sim_events):
            continue

        ground_intervals = _build_view_intervals(
            view_name, ground_events, role_mapping, case_active_mode
        )
        sim_intervals = _build_view_intervals(
            view_name, sim_events, role_mapping, case_active_mode
        )
        change_points = _collect_change_points(ground_intervals, sim_intervals)
        timeline, overall_score, contributors = _sweep_distance(
            change_points,
            ground_intervals,
            sim_intervals,
            top_k=top_k,
        )
        assessment["views"][view_name] = {
            "overall_score": overall_score,
            "timeline": timeline,
            "contributors": contributors,
        }

    return assessment


def interpret_assessment(
    assessment: Dict[str, Any],
    top_n_views: int = 3,
    top_n_intervals: int = 3,
    top_n_contributors: int = 3,
) -> str:
    """Return a short diagnostic summary without changing assessment values."""
    views = assessment.get("views", {})
    if not views:
        return "No views computed."

    view_scores = [
        (name, data.get("overall_score", 0.0)) for name, data in views.items()
    ]
    view_scores.sort(key=lambda item: item[1], reverse=True)
    top_views = view_scores[:top_n_views]

    lines = []
    for name, score in top_views:
        reason = _view_interpretation_hint(name)
        lines.append(f"{name}: overall {score:.4f}. {reason}")

        timeline = views[name].get("timeline", [])
        if timeline:
            top_intervals = sorted(
                timeline, key=lambda rec: rec.get("distance", 0.0), reverse=True
            )[:top_n_intervals]
            for interval in top_intervals:
                contributors = interval.get("contributors") or []
                contributor_text = _format_contributors(contributors, top_n_contributors)
                lines.append(
                    "  window "
                    f"[{interval['start']}, {interval['end']}): "
                    f"distance {interval['distance']:.4f}. {contributor_text}"
                )
        else:
            ranked = views[name].get("contributors") or []
            contributor_text = _format_contributors(ranked, top_n_contributors)
            if contributor_text:
                lines.append(f"  top contributors: {contributor_text}")

    return "\n".join(lines)


def _normalize_view_list(views: Optional[Iterable[str]]) -> List[str]:
    if not views:
        return []
    return [_canonical_view_name(view) for view in views]


def _canonical_view_name(view: str) -> str:
    return view.strip().lower().replace("-", "_")


def _default_views_for_data(
    ground_events: List[NormalizedEvent],
    sim_events: List[NormalizedEvent],
    role_mapping: Dict[Any, Any],
) -> List[str]:
    views = list(DEFAULT_VIEWS)
    if not role_mapping and VIEW_ACTIVITY_ROLE in views:
        views.remove(VIEW_ACTIVITY_ROLE)
    if not _has_enable_data(ground_events, sim_events) and VIEW_ACTIVITY_PHASE in views:
        views.remove(VIEW_ACTIVITY_PHASE)
    return views


def _has_enable_data(
    ground_events: List[NormalizedEvent],
    sim_events: List[NormalizedEvent],
) -> bool:
    for event in ground_events + sim_events:
        if event.enable is not None and event.enable < event.start:
            return True
    return False


def _normalize_log(
    log: Any,
    field_map: Dict[str, str],
    role_mapping: Dict[Any, Any],
) -> Tuple[List[NormalizedEvent], Dict[str, Any]]:
    records = _records_from_log(log)
    events: List[NormalizedEvent] = []
    invalid = 0

    for record in records:
        start = _coerce_timestamp(record.get(field_map["start"]))
        end = _coerce_timestamp(record.get(field_map["end"]))
        if start is None or end is None or end <= start:
            invalid += 1
            continue

        case_id = record.get(field_map["case"])
        activity = record.get(field_map["activity"])
        resource = record.get(field_map.get("resource"))
        role = role_mapping.get(resource) if role_mapping else None
        enable = _coerce_timestamp(record.get(field_map.get("enable")))

        events.append(
            NormalizedEvent(
                case_id=case_id,
                activity=activity,
                start=start,
                end=end,
                resource=resource,
                role=role,
                enable=enable,
            )
        )

    return events, {"invalid_intervals": invalid}


def _records_from_log(log: Any) -> List[Dict[str, Any]]:
    if pd is not None and isinstance(log, pd.DataFrame):
        return log.to_dict(orient="records")
    if isinstance(log, list):
        return log
    raise TypeError("Event logs must be a list of dicts or a pandas DataFrame.")


def _coerce_timestamp(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if math.isnan(value):
            return None
        return float(value)
    if isinstance(value, datetime):
        return value.timestamp()
    if pd is None:
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.value / 1e9


def _apply_enable_fallback(events: List[NormalizedEvent]) -> None:
    by_case: Dict[Any, List[NormalizedEvent]] = defaultdict(list)
    for event in events:
        by_case[event.case_id].append(event)
    for case_events in by_case.values():
        case_events.sort(key=lambda ev: (ev.start, ev.end))
        prev_end = None
        for event in case_events:
            if event.enable is None and prev_end is not None:
                event.enable = prev_end
            prev_end = event.end


def _build_time_normalizer(
    normalize_time: Any,
    ground_events: List[NormalizedEvent],
    sim_events: List[NormalizedEvent],
) -> Optional[Dict[str, Any]]:
    if not normalize_time:
        return None

    if isinstance(normalize_time, dict):
        start = _coerce_timestamp(normalize_time.get("start"))
        end = _coerce_timestamp(normalize_time.get("end"))
    else:
        start = min((ev.start for ev in ground_events + sim_events), default=None)
        end = max((ev.end for ev in ground_events + sim_events), default=None)

    if start is None or end is None or end <= start:
        return None

    scale = end - start
    return {"start": start, "end": end, "scale": scale}


def _normalize_event_times(events: List[NormalizedEvent], normalizer: Dict[str, Any]) -> None:
    start = normalizer["start"]
    scale = normalizer["scale"] or 1.0
    for event in events:
        event.start = (event.start - start) / scale
        event.end = (event.end - start) / scale
        if event.enable is not None:
            event.enable = (event.enable - start) / scale


def _build_view_intervals(
    view_name: str,
    events: List[NormalizedEvent],
    role_mapping: Dict[Any, Any],
    case_active_mode: str,
) -> List[Dict[str, Any]]:
    intervals: List[Dict[str, Any]] = []

    if view_name == VIEW_CASE:
        if case_active_mode == "running":
            for event in events:
                intervals.append(
                    {"start": event.start, "end": event.end, "key": event.case_id}
                )
            return intervals

        case_bounds: Dict[Any, Tuple[float, float]] = {}
        for event in events:
            if event.case_id not in case_bounds:
                case_bounds[event.case_id] = (event.start, event.end)
            else:
                current_start, current_end = case_bounds[event.case_id]
                case_bounds[event.case_id] = (
                    min(current_start, event.start),
                    max(current_end, event.end),
                )
        for case_id, (start, end) in case_bounds.items():
            intervals.append({"start": start, "end": end, "key": case_id})
        return intervals

    for event in events:
        key = None
        if view_name == VIEW_ACTIVITY:
            key = event.activity
        elif view_name == VIEW_ACTIVITY_CASE:
            key = (event.activity, event.case_id)
        elif view_name == VIEW_ACTIVITY_RESOURCE:
            key = (event.activity, event.resource)
        elif view_name == VIEW_ACTIVITY_CASE_RESOURCE:
            key = (event.activity, event.case_id, event.resource)
        elif view_name == VIEW_RESOURCE:
            key = event.role if role_mapping else event.resource
        elif view_name == VIEW_ACTIVITY_ROLE:
            key = (event.activity, event.role)
        elif view_name == VIEW_ACTIVITY_PHASE:
            if event.enable is not None and event.enable < event.start:
                intervals.append(
                    {
                        "start": event.enable,
                        "end": event.start,
                        "key": (event.activity, "waiting"),
                    }
                )
            key = (event.activity, "processing")
        else:
            continue

        intervals.append({"start": event.start, "end": event.end, "key": key})

    return intervals


def _collect_change_points(
    ground_intervals: List[Dict[str, Any]],
    sim_intervals: List[Dict[str, Any]],
) -> List[float]:
    points = set()
    for interval in ground_intervals + sim_intervals:
        points.add(interval["start"])
        points.add(interval["end"])
    return sorted(points)


def _sweep_distance(
    change_points: List[float],
    ground_intervals: List[Dict[str, Any]],
    sim_intervals: List[Dict[str, Any]],
    top_k: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], float, List[Dict[str, Any]]]:
    starts_ground, ends_ground = _build_update_maps(ground_intervals)
    starts_sim, ends_sim = _build_update_maps(sim_intervals)

    active_ground = Counter()
    active_sim = Counter()
    timeline: List[Dict[str, Any]] = []
    contributor_totals: Dict[Any, float] = defaultdict(float)
    weighted_sum = 0.0
    total_duration = 0.0

    for idx in range(len(change_points) - 1):
        t = change_points[idx]
        t_next = change_points[idx + 1]
        _apply_updates(active_ground, ends_ground.get(t), -1)
        _apply_updates(active_sim, ends_sim.get(t), -1)
        _apply_updates(active_ground, starts_ground.get(t), 1)
        _apply_updates(active_sim, starts_sim.get(t), 1)

        duration = t_next - t
        if duration <= 0:
            continue

        denom = sum(active_ground.values()) + sum(active_sim.values())
        if denom == 0:
            distance = 0.0
            contributions = {}
        else:
            distance, contributions = _bray_curtis_contributions(
                active_ground, active_sim, denom, duration
            )
        weighted_sum += distance * duration
        total_duration += duration

        for key, contribution in contributions.items():
            contributor_totals[key] += contribution

        record = {
            "start": t,
            "end": t_next,
            "duration": duration,
            "distance": distance,
        }
        if top_k:
            record["contributors"] = _top_k_contributors(contributions, duration, top_k)
        timeline.append(record)

    overall_score = (weighted_sum / total_duration) if total_duration > 0 else 0.0
    contributor_rankings = _rank_contributors(contributor_totals)
    return timeline, overall_score, contributor_rankings


def _build_update_maps(
    intervals: List[Dict[str, Any]],
) -> Tuple[Dict[float, List[Any]], Dict[float, List[Any]]]:
    starts: Dict[float, List[Any]] = defaultdict(list)
    ends: Dict[float, List[Any]] = defaultdict(list)
    for interval in intervals:
        starts[interval["start"]].append(interval["key"])
        ends[interval["end"]].append(interval["key"])
    return starts, ends


def _apply_updates(counter: Counter, keys: Optional[List[Any]], delta: int) -> None:
    if not keys:
        return
    for key in keys:
        counter[key] += delta
        if counter[key] <= 0:
            del counter[key]


def _bray_curtis_contributions(
    active_ground: Counter,
    active_sim: Counter,
    denom: float,
    duration: float,
) -> Tuple[float, Dict[Any, float]]:
    diff_sum = 0.0
    contributions: Dict[Any, float] = {}
    all_keys = set(active_ground.keys()) | set(active_sim.keys())
    for key in all_keys:
        diff = abs(active_ground.get(key, 0) - active_sim.get(key, 0))
        if diff > 0:
            contributions[key] = (diff / denom) * duration
        diff_sum += diff
    return diff_sum / denom, contributions


def _top_k_contributors(
    contributions: Dict[Any, float],
    duration: float,
    top_k: int,
) -> List[Dict[str, Any]]:
    items = sorted(contributions.items(), key=lambda item: item[1], reverse=True)[:top_k]
    result = []
    for key, contribution in items:
        share = contribution / duration if duration > 0 else 0.0
        result.append({"key": key, "contribution": contribution, "share": share})
    return result


def _rank_contributors(contributor_totals: Dict[Any, float]) -> List[Dict[str, Any]]:
    total = sum(contributor_totals.values())
    ranked = sorted(contributor_totals.items(), key=lambda item: item[1], reverse=True)
    return [
        {
            "key": key,
            "contribution": contribution,
            "share": (contribution / total) if total > 0 else 0.0,
        }
        for key, contribution in ranked
    ]


def _view_interpretation_hint(view_name: str) -> str:
    hints = {
        VIEW_ACTIVITY: "High mismatch suggests global process behavior differences.",
        VIEW_ACTIVITY_CASE: "High mismatch suggests incorrect case progression or queueing dynamics.",
        VIEW_ACTIVITY_RESOURCE: "High mismatch suggests mis-modeled staffing, calendars, or dispatching.",
        VIEW_ACTIVITY_CASE_RESOURCE: (
            "Mismatch concentrated here implies micro-level pairing differences."
        ),
        VIEW_RESOURCE: "Mismatch suggests staffing volume or allocation differences.",
        VIEW_CASE: "Mismatch suggests case arrival/completion timing differences.",
        VIEW_ACTIVITY_ROLE: "High mismatch suggests role-level staffing or assignment issues.",
        VIEW_ACTIVITY_PHASE: "Mismatch suggests waiting vs processing timing differences.",
    }
    return hints.get(view_name, "")


def _format_contributors(contributors: List[Dict[str, Any]], top_n: int) -> str:
    if not contributors:
        return ""
    formatted = []
    for item in contributors[:top_n]:
        formatted.append(f"{_format_key(item['key'])} ({item['contribution']:.4f})")
    return "top contributors: " + ", ".join(formatted)


def _format_key(key: Any) -> str:
    if isinstance(key, tuple):
        return " | ".join(str(part) for part in key)
    return str(key)


def _load_csv_log(path: str) -> List[Dict[str, Any]]:
    if pd is None:
        raise RuntimeError("pandas is required for CSV usage.")
    return pd.read_csv(path).to_dict(orient="records")


if __name__ == "__main__":
    # Edit these to run the module directly without CLI args.
    GROUND_LOG_PATH = "samples/output/partial_sim_log.csv"
    SIM_LOG_PATH = "samples/output/sim_log.csv"
    FIELD_MAP = {
        "case": "case_id",
        "activity": "activity",
        "start": "start_time",
        "end": "end_time",
        "resource": "resource",
        "enable": "enable_time",
    }
    ROLE_MAPPING: Dict[Any, Any] = {}
    OPTIONS = {
        "views": DEFAULT_VIEWS,
        "top_k_contributors": 3,
        "normalize_time": True,
        "field_map": FIELD_MAP,
        "role_mapping": ROLE_MAPPING,
        "case_active_mode": "interval",
    }

    ground_log = _load_csv_log(GROUND_LOG_PATH)
    sim_log = _load_csv_log(SIM_LOG_PATH)

    assessment = assess_event_logs(ground_log, sim_log, OPTIONS)
    print("Overall scores:")
    for view_name, view_data in assessment["views"].items():
        print(f"  {view_name}: {view_data['overall_score']:.4f}")

    print("\nDiagnostic:")
    print(interpret_assessment(assessment))
