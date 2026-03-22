# src/utils.py
"""Shared utility functions for the src package."""
from __future__ import annotations

import datetime as _dt
from typing import Optional


def parse_datetime(dt_str: Optional[str], has_date: bool | None = None) -> Optional[_dt.datetime]:
    """Convert an ISO-8601 string into a timezone-aware datetime.

    Handles strings ending with 'Z' by replacing it with '+00:00'.
    The *has_date* parameter is ignored; it exists only for signature
    compatibility with ``prosimos.utils.parse_datetime()``.
    """
    if not dt_str:
        return None
    return _dt.datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
