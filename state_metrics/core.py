"""Timestamp sweep over two event logs tracking active activity instances.

The public entry point is :func:`sweep_active_instances`, which yields
`(t_i, active_gt, active_sim)` at every distinct event boundary. The API layer
turns those samples into per-interval distances.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterator

import pandas as pd

from .views import ActiveInstance


REQUIRED_COLUMNS = ("case_id", "activity", "start_time", "end_time", "resource")


def _validate(df: pd.DataFrame, label: str) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{label} log missing columns: {missing}")
    if len(df) == 0:
        return
    if df["start_time"].isna().any() or df["end_time"].isna().any():
        raise ValueError(f"{label} log contains NaT in start_time/end_time")
    bad = df["end_time"] < df["start_time"]
    if bad.any():
        n = int(bad.sum())
        raise ValueError(
            f"{label} log contains {n} row(s) with end_time < start_time"
        )


def _build_indices(
    df: pd.DataFrame,
) -> tuple[dict[pd.Timestamp, list[ActiveInstance]], dict[pd.Timestamp, list[ActiveInstance]]]:
    """Group activity instances by start and end timestamp.

    Zero-duration rows (``start_time == end_time``) are skipped: under the
    half-open ``[start, end)`` semantics they have no active interval, and
    including them would leak the instance into the bag forever (nothing
    removes it at a later timestamp).
    """
    starts: dict[pd.Timestamp, list[ActiveInstance]] = defaultdict(list)
    ends: dict[pd.Timestamp, list[ActiveInstance]] = defaultdict(list)
    has_case_type = "case_type" in df.columns
    # Use .itertuples for speed; coerce types defensively.
    for row in df.itertuples(index=False):
        if row.start_time == row.end_time:
            continue
        case_type = ""
        if has_case_type:
            v = getattr(row, "case_type", None)
            if v is not None and pd.notna(v):
                case_type = str(v)
        inst = ActiveInstance(
            case_id=str(row.case_id),
            activity=str(row.activity),
            resource=str(row.resource) if row.resource is not None else "",
            case_type=case_type,
        )
        starts[row.start_time].append(inst)
        ends[row.end_time].append(inst)
    return starts, ends


class _ActiveBag:
    """Ordered multiset of active instances with O(1) add/remove.

    Entries are keyed by ``id()`` of the ActiveInstance tuple so duplicate
    tuples remain distinct in the bag — this matches the "multiset of
    instances" semantics.
    """

    __slots__ = ("_items",)

    def __init__(self) -> None:
        self._items: dict[int, ActiveInstance] = {}

    def add(self, inst: ActiveInstance) -> None:
        self._items[id(inst)] = inst

    def remove(self, inst: ActiveInstance) -> None:
        self._items.pop(id(inst), None)

    def __iter__(self):
        return iter(self._items.values())

    def __len__(self) -> int:
        return len(self._items)


def sweep_active_instances(
    gt_log: pd.DataFrame,
    sim_log: pd.DataFrame,
) -> Iterator[tuple[pd.Timestamp, _ActiveBag, _ActiveBag]]:
    """Yield `(t_i, active_gt, active_sim)` at every distinct event boundary.

    The bags are mutated in place across iterations — callers must project
    them before advancing (or copy the snapshot they need).
    """
    _validate(gt_log, "ground-truth")
    _validate(sim_log, "simulated")

    gt_starts, gt_ends = _build_indices(gt_log)
    sim_starts, sim_ends = _build_indices(sim_log)

    all_ts = sorted(
        set(gt_starts) | set(gt_ends) | set(sim_starts) | set(sim_ends)
    )

    gt_bag = _ActiveBag()
    sim_bag = _ActiveBag()

    for t in all_ts:
        # Half-open: an instance ending at t is NOT active at t.
        for inst in gt_ends.get(t, ()):
            gt_bag.remove(inst)
        for inst in sim_ends.get(t, ()):
            sim_bag.remove(inst)
        for inst in gt_starts.get(t, ()):
            gt_bag.add(inst)
        for inst in sim_starts.get(t, ()):
            sim_bag.add(inst)
        yield t, gt_bag, sim_bag
