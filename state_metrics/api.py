"""Public API for state-based metrics.

Typical use::

    from state_metrics import compute_state_distance, compute_all_state_distances

    result = compute_state_distance(gt, sim, view="activity_case", distance="jaccard")
    print(result.summary_distance)

Summary-distance semantics
--------------------------
By default, the summary distance is a time-weighted average over
*observed event boundaries*: the denominator is
``last_event_ts - first_event_ts``. Intervals outside that span are not
represented in either numerator or denominator. Since both bags are
empty in those intervals and all distance functions return 0 for two
empty bags, the unrepresented intervals are zero-distance — they only
affect the *denominator*, making the reported value an average over
"active" time rather than wall-clock time.

This default is well-behaved for comparing relative responses across runs
that share the same cutoff, horizon, and dataset. It is **not** the right
choice when the active span itself depends on the perturbation (e.g.,
Scope A continuation quality, where heavy perturbation stalls cases and
extends the active span). For that case, pass an explicit ``window``:

    result = compute_state_distance(
        gt, sim, view="activity", distance="jaccard",
        window=(cutoff_ts, horizon_end_ts),
    )

When ``window`` is given, the denominator is fixed at
``(window[1] - window[0]).total_seconds()`` regardless of where events
actually fall, making summaries comparable across runs whose active
spans differ. Both logs must be filtered to lie inside the window
beforehand; the library does not re-clip them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from .core import sweep_active_instances
from .distances import DISTANCE_FUNCTIONS
from .views import VIEWS, project


View = Literal[
    "activity_case", "case", "activity", "activity_role",
    "case_type", "activity_type",
]
Distance = Literal["jaccard", "cardinality"]


@dataclass
class StateDistanceResult:
    summary_distance: float
    timeline: list[tuple[pd.Timestamp, float]]
    total_duration_seconds: float
    view: str
    distance_function: str


Window = tuple[pd.Timestamp, pd.Timestamp]


def _validate_window(window: Window | None) -> float | None:
    if window is None:
        return None
    start, end = window
    if end <= start:
        raise ValueError(f"window end must be > start, got {window!r}")
    return (end - start).total_seconds()


def compute_state_distance(
    gt_log: pd.DataFrame,
    sim_log: pd.DataFrame,
    *,
    view: View,
    distance: Distance,
    role_map: dict[str, str] | None = None,
    window: Window | None = None,
) -> StateDistanceResult:
    if view not in VIEWS:
        raise ValueError(f"unknown view {view!r}")
    if distance not in DISTANCE_FUNCTIONS:
        raise ValueError(f"unknown distance {distance!r}")

    all_results = _sweep_and_compute(
        gt_log, sim_log, views=(view,), distances=(distance,),
        role_map=role_map, window=window,
    )
    return all_results[(view, distance)]


def compute_all_state_distances(
    gt_log: pd.DataFrame,
    sim_log: pd.DataFrame,
    *,
    role_map: dict[str, str] | None = None,
    window: Window | None = None,
) -> dict[tuple[str, str], StateDistanceResult]:
    """Compute all (view × distance) combinations in one sweep.

    If ``role_map`` is None the ``activity_role`` view is skipped. The
    ``case_type`` and ``activity_type`` views are included only when at
    least one of the input logs carries a ``case_type`` column (otherwise
    every active instance has an empty case_type and both projections would
    be trivially zero). See the module docstring for ``window`` semantics.
    """
    has_case_type = (
        "case_type" in gt_log.columns or "case_type" in sim_log.columns
    )
    views = tuple(
        v for v in VIEWS
        if (v != "activity_role" or role_map is not None)
        and (v not in ("case_type", "activity_type") or has_case_type)
    )
    distances = tuple(DISTANCE_FUNCTIONS.keys())
    return _sweep_and_compute(
        gt_log, sim_log, views=views, distances=distances,
        role_map=role_map, window=window,
    )


def _sweep_and_compute(
    gt_log: pd.DataFrame,
    sim_log: pd.DataFrame,
    *,
    views: tuple[str, ...],
    distances: tuple[str, ...],
    role_map: dict[str, str] | None,
    window: Window | None = None,
) -> dict[tuple[str, str], StateDistanceResult]:
    # Per (view, distance) we accumulate the timeline of (t_i, d_i) pairs.
    timelines: dict[tuple[str, str], list[tuple[pd.Timestamp, float]]] = {
        (v, d): [] for v in views for d in distances
    }
    timestamps: list[pd.Timestamp] = []

    # Project inside the sweep: the bags are mutated in place across yields,
    # so we cannot materialize them to a list first.
    for t, gt_bag, sim_bag in sweep_active_instances(gt_log, sim_log):
        timestamps.append(t)
        projected_gt = {v: project(gt_bag, v, role_map=role_map) for v in views}
        projected_sim = {v: project(sim_bag, v, role_map=role_map) for v in views}
        for v in views:
            for d in distances:
                dist_fn = DISTANCE_FUNCTIONS[d]
                timelines[(v, d)].append((t, dist_fn(projected_gt[v], projected_sim[v])))

    window_seconds = _validate_window(window)

    if not timestamps:
        # No events at all. If a window was given, the entire window is
        # zero-distance (both bags empty everywhere); report its length so
        # callers can sanity-check the denominator they asked for.
        denom = window_seconds if window_seconds is not None else 0.0
        return {
            key: StateDistanceResult(
                summary_distance=0.0,
                timeline=[],
                total_duration_seconds=denom,
                view=key[0],
                distance_function=key[1],
            )
            for key in timelines
        }

    # Time-weight using (t_{i+1} - t_i); the final sample covers no interval.
    weights = [
        (timestamps[i + 1] - timestamps[i]).total_seconds()
        for i in range(len(timestamps) - 1)
    ]
    active_duration = sum(weights)
    # Denominator: explicit window if given, else observed-events span.
    total_duration = window_seconds if window_seconds is not None else active_duration

    out: dict[tuple[str, str], StateDistanceResult] = {}
    for key, timeline in timelines.items():
        if total_duration <= 0:
            summary = 0.0
        else:
            # Use d_i at t_i as the distance on [t_i, t_{i+1}); drop last.
            weighted_sum = sum(
                timeline[i][1] * weights[i] for i in range(len(weights))
            )
            summary = weighted_sum / total_duration
        out[key] = StateDistanceResult(
            summary_distance=summary,
            timeline=timeline,
            total_duration_seconds=total_duration,
            view=key[0],
            distance_function=key[1],
        )
    return out
