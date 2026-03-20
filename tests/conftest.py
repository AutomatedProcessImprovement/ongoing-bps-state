"""Shared fixtures for the test suite."""
from __future__ import annotations

import pandas as pd
import numpy as np
import pytest


@pytest.fixture()
def simple_event_log():
    """A small synthetic event log with 3 cases spanning a known time range.

    Timeline (UTC):
        Case 1: 2025-01-01 08:00 → 2025-01-01 12:00  (2 events)
        Case 2: 2025-01-01 09:00 → 2025-01-01 14:00  (2 events)
        Case 3: 2025-01-01 11:00 → 2025-01-01 15:00  (2 events)
    """
    data = {
        "case_id":    [1, 1, 2, 2, 3, 3],
        "activity":   ["A", "B", "A", "C", "A", "B"],
        "resource":   ["R1", "R2", "R1", "R2", "R1", "R2"],
        "start_time": pd.to_datetime([
            "2025-01-01 08:00", "2025-01-01 10:00",
            "2025-01-01 09:00", "2025-01-01 12:00",
            "2025-01-01 11:00", "2025-01-01 13:00",
        ], utc=True),
        "end_time": pd.to_datetime([
            "2025-01-01 09:30", "2025-01-01 12:00",
            "2025-01-01 11:00", "2025-01-01 14:00",
            "2025-01-01 12:30", "2025-01-01 15:00",
        ], utc=True),
    }
    return pd.DataFrame(data)


@pytest.fixture()
def event_log_ids():
    """Standard EventLogIDs used across the evaluation code."""
    from log_distance_measures.config import EventLogIDs
    return EventLogIDs("case_id", "activity", "start_time", "end_time", "resource")
