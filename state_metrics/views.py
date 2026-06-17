"""Projections from a set of active activity instances to a multiset."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, NamedTuple


class ActiveInstance(NamedTuple):
    case_id: str
    activity: str
    resource: str
    # Optional per-case attribute (e.g. customer segment "green"/"red"). Empty
    # string is the default for logs without a case_type column, which keeps
    # all existing callers unaffected. Only the case_type / activity_type
    # projections read this field.
    case_type: str = ""


VIEWS = (
    "activity_case", "case", "activity", "activity_role",
    "case_type", "activity_type",
)


def project(
    instances: Iterable[ActiveInstance],
    view: str,
    role_map: dict[str, str] | None = None,
) -> Counter:
    if view == "activity_case":
        return Counter((i.activity, i.case_id) for i in instances)
    if view == "case":
        return Counter(i.case_id for i in instances)
    if view == "activity":
        return Counter(i.activity for i in instances)
    if view == "activity_role":
        if role_map is None:
            raise ValueError("activity_role view requires a role_map")
        out: Counter = Counter()
        for i in instances:
            if i.resource not in role_map:
                raise KeyError(f"resource '{i.resource}' not found in role_map")
            out[(i.activity, role_map[i.resource])] += 1
        return out
    if view == "case_type":
        # Multiset of active per-case attributes. Detects when the population
        # composition shifts (more "green" than "red" active at time t).
        return Counter(i.case_type for i in instances)
    if view == "activity_type":
        # Multiset of (activity, case_type) pairs. Distinguishes "same
        # activity executed by a different case type" — the projection that
        # the label_swap oracle relies on, because activity labels and
        # timestamps stay identical while only case_type tags change.
        return Counter((i.activity, i.case_type) for i in instances)
    raise ValueError(f"unknown view: {view!r} (expected one of {VIEWS})")
