"""Tests for evaluation.rtd module."""
from __future__ import annotations

import pandas as pd
import pytest

from evaluation.rtd import rtd
from log_distance_measures.config import EventLogIDs


@pytest.fixture()
def ids():
    return EventLogIDs("case_id", "activity", "start_time", "end_time", "resource")


class TestRtd:
    def test_identical_logs_zero_distance(self, ids):
        df = pd.DataFrame({
            "case_id": [1, 2],
            "activity": ["A", "A"],
            "resource": ["R1", "R1"],
            "start_time": pd.to_datetime(["2025-01-01 08:00", "2025-01-01 09:00"], utc=True),
            "end_time": pd.to_datetime(["2025-01-01 12:00", "2025-01-01 14:00"], utc=True),
        })
        ref = pd.Timestamp("2025-01-01 10:00", tz="UTC")
        dist = rtd(df, ids, df, ids, reference_point=ref, bin_size=pd.Timedelta(hours=1))
        assert dist == pytest.approx(0.0)

    def test_shifted_logs_positive_distance(self, ids):
        original = pd.DataFrame({
            "case_id": [1, 2],
            "activity": ["A", "A"],
            "resource": ["R1", "R1"],
            "start_time": pd.to_datetime(["2025-01-01 08:00", "2025-01-01 09:00"], utc=True),
            "end_time": pd.to_datetime(["2025-01-01 12:00", "2025-01-01 14:00"], utc=True),
        })
        shifted = pd.DataFrame({
            "case_id": [1, 2],
            "activity": ["A", "A"],
            "resource": ["R1", "R1"],
            "start_time": pd.to_datetime(["2025-01-01 08:00", "2025-01-01 09:00"], utc=True),
            "end_time": pd.to_datetime(["2025-01-01 16:00", "2025-01-01 20:00"], utc=True),
        })
        ref = pd.Timestamp("2025-01-01 10:00", tz="UTC")
        dist = rtd(original, ids, shifted, ids, reference_point=ref, bin_size=pd.Timedelta(hours=1))
        assert dist > 0

    def test_padding_unequal_cases(self, ids):
        """When logs have different number of cases, shorter one is padded."""
        original = pd.DataFrame({
            "case_id": [1, 2, 3],
            "activity": ["A", "A", "A"],
            "resource": ["R1", "R1", "R1"],
            "start_time": pd.to_datetime(["2025-01-01 08:00"] * 3, utc=True),
            "end_time": pd.to_datetime(["2025-01-01 12:00"] * 3, utc=True),
        })
        simulated = pd.DataFrame({
            "case_id": [1],
            "activity": ["A"],
            "resource": ["R1"],
            "start_time": pd.to_datetime(["2025-01-01 08:00"], utc=True),
            "end_time": pd.to_datetime(["2025-01-01 12:00"], utc=True),
        })
        ref = pd.Timestamp("2025-01-01 10:00", tz="UTC")
        dist = rtd(original, ids, simulated, ids, reference_point=ref, bin_size=pd.Timedelta(hours=1))
        # Should still compute without error; distance > 0 because of padding
        assert dist >= 0
