"""Multiset distance functions used by the state-based metrics."""

from __future__ import annotations

from collections import Counter
from typing import Hashable


def jaccard_multiset(a: Counter[Hashable], b: Counter[Hashable]) -> float:
    """Weighted Jaccard distance over multisets.

    `1 - sum(min(a[s], b[s])) / sum(max(a[s], b[s]))`. Both empty → 0.
    """
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    num = 0
    den = 0
    for k in keys:
        ak = a.get(k, 0)
        bk = b.get(k, 0)
        num += min(ak, bk)
        den += max(ak, bk)
    if den == 0:
        return 0.0
    return 1.0 - num / den


def cardinality(a: Counter[Hashable], b: Counter[Hashable]) -> float:
    """Normalised absolute difference of multiset sizes.

    `|total(a) - total(b)| / max(total(a), total(b))`. Both empty → 0.
    """
    ta = sum(a.values())
    tb = sum(b.values())
    m = max(ta, tb)
    if m == 0:
        return 0.0
    return abs(ta - tb) / m


DISTANCE_FUNCTIONS = {
    "jaccard": jaccard_multiset,
    "cardinality": cardinality,
}
