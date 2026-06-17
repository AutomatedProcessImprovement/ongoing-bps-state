# src/attributes.py
"""Extract Prosimos attribute values from a (pre-cut) event log.

The updated Prosimos short-term engine
(``AutomatedProcessImprovement/Prosimos@short-term-simulation``) restores the
attribute values captured in the process-state snapshot instead of re-sampling
them on resume:

* top-level ``global_attributes`` — current global attribute values,
* per-case ``case_attributes`` — the values the case already had (so gateway
  conditions / prioritisation rules branch the same way),
* per-case ``event_attributes`` — baseline per-case event-attribute values,
* per-ongoing-activity ``event_attributes`` — the point-in-time values the
  activity carried in the historical log.

These fields are all OPTIONAL: when absent the engine falls back to sampled
values (emitting a warning only when the model actually relies on case
attributes). This module reads the real values out of the event log so the
snapshot can carry them.

Every helper is a no-op when the simulation parameters declare no attributes or
the log carries no matching columns, so datasets without attributes produce a
snapshot byte-identical to the pre-attribute behaviour.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def _coerce(value: Any):
    """Return a JSON-/engine-friendly native scalar, or ``None`` if missing.

    numpy scalars are unwrapped via ``.item()`` so the snapshot carries plain
    Python ints/floats/bools/strings.
    """
    if value is None:
        return None
    if not isinstance(value, (list, dict)):
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
    item = getattr(value, "item", None)
    return item() if callable(item) else value


def _uniq(names: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def declared_attribute_names(bpmn_params: dict) -> dict[str, list[str]]:
    """Collect attribute names declared in Prosimos parameters, grouped by family.

    Returns ``{"case": [...], "event": [...], "global": [...]}`` (each possibly
    empty). ``event`` aggregates the attribute names from every per-event entry.
    """
    case_names = [
        a["name"] for a in (bpmn_params.get("case_attributes") or []) if "name" in a
    ]
    global_names = [
        a["name"] for a in (bpmn_params.get("global_attributes") or []) if "name" in a
    ]
    event_names: list[str] = []
    for entry in bpmn_params.get("event_attributes") or []:
        for a in entry.get("attributes", []):
            if "name" in a:
                event_names.append(a["name"])
    return {
        "case": _uniq(case_names),
        "event": _uniq(event_names),
        "global": _uniq(global_names),
    }


def present_columns(names: list[str], df: pd.DataFrame) -> list[str]:
    """Subset of ``names`` that are actual columns in ``df``."""
    return [n for n in names if n in df.columns]


def row_attributes(row: pd.Series, names: list[str]) -> dict:
    """Attribute values carried by a single log row (skips missing values)."""
    out: dict = {}
    for name in names:
        value = _coerce(row.get(name))
        if value is not None:
            out[name] = value
    return out


def case_attribute_values(group: pd.DataFrame, names: list[str]) -> dict:
    """Per-case (constant) attribute values — first non-null value in the case."""
    out: dict = {}
    for name in names:
        if name not in group.columns:
            continue
        non_null = group[name].dropna()
        if not non_null.empty:
            value = _coerce(non_null.iloc[0])
            if value is not None:
                out[name] = value
    return out


def latest_attribute_values(
    df: pd.DataFrame, start_time_col: str, names: list[str]
) -> dict:
    """Most-recent (by ``start_time_col``) attribute values across ``df``.

    Used for global attributes (process-wide current value) and for the per-case
    baseline event-attribute values.
    """
    if df.empty or not names:
        return {}
    last_row = df.sort_values(start_time_col).iloc[-1]
    return row_attributes(last_row, names)
