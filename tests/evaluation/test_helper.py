"""Tests for evaluation.helper module."""
from __future__ import annotations

import string
import pandas as pd
import numpy as np
import pytest

from evaluation.helper import (
    generate_short_uuid,
    run_with_retries,
    read_event_log,
    trim_events,
    _split_cases,
    split_into_subsets,
    avg_remaining_time,
    _build_ps_subsets,
    _avg_events_per_ongoing_case,
    _avg_events_per_case_diff,
    compute_cut_points,
    build_aggregated_from_cuts,
)


# ── generate_short_uuid ──────────────────────────────────────────────

class TestGenerateShortUuid:
    def test_default_length(self):
        result = generate_short_uuid()
        assert len(result) == 6

    def test_custom_length(self):
        result = generate_short_uuid(12)
        assert len(result) == 12

    def test_character_set(self):
        allowed = set(string.ascii_lowercase + string.digits)
        result = generate_short_uuid(100)
        assert set(result).issubset(allowed)


# ── run_with_retries ─────────────────────────────────────────────────

class TestRunWithRetries:
    def test_succeeds_first_try(self):
        result = run_with_retries(lambda x: x * 2, {"x": 5}, 3, verbose=False)
        assert result == 10

    def test_succeeds_after_failures(self):
        call_count = {"n": 0}

        def flaky(**kw):
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise RuntimeError("boom")
            return "ok"

        result = run_with_retries(flaky, {}, 3, verbose=False)
        assert result == "ok"
        assert call_count["n"] == 3

    def test_raises_after_exhaustion(self):
        def always_fail(**kw):
            raise ValueError("nope")

        with pytest.raises(ValueError, match="nope"):
            run_with_retries(always_fail, {}, 2, verbose=False)


# ── read_event_log ───────────────────────────────────────────────────

class TestReadEventLog:
    def test_basic_read(self, tmp_path):
        csv = tmp_path / "log.csv"
        csv.write_text(
            "case_id,activity,resource,start_time,end_time\n"
            "1,A,R1,2025-01-01T08:00:00Z,2025-01-01T09:00:00Z\n"
        )
        df = read_event_log(str(csv))
        assert len(df) == 1
        assert pd.api.types.is_datetime64_any_dtype(df["start_time"])
        assert pd.api.types.is_datetime64_any_dtype(df["end_time"])

    def test_rename_columns(self, tmp_path):
        csv = tmp_path / "log.csv"
        csv.write_text("CaseId,Activity,Resource,StartTime,EndTime\n1,A,R1,2025-01-01T08:00:00Z,2025-01-01T09:00:00Z\n")
        df = read_event_log(
            str(csv),
            rename={"CaseId": "case_id", "StartTime": "start_time", "EndTime": "end_time"},
        )
        assert "case_id" in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df["start_time"])

    def test_missing_required_raises(self, tmp_path):
        csv = tmp_path / "log.csv"
        csv.write_text("case_id,activity\n1,A\n")
        with pytest.raises(ValueError, match="Missing required"):
            read_event_log(str(csv), required=["case_id", "start_time"])


# ── trim_events ──────────────────────────────────────────────────────

class TestTrimEvents:
    def test_clips_boundaries(self, simple_event_log):
        start = pd.Timestamp("2025-01-01 10:00", tz="UTC")
        end = pd.Timestamp("2025-01-01 13:00", tz="UTC")
        result = trim_events(simple_event_log, start, end)
        # Events overlapping [10:00, 13:00]: case1-B, case2-A, case2-C, case3-A, case3-B
        assert not result.empty
        # All start_times should be >= start
        assert (result["start_time"] >= start).all()
        # All end_times should be <= end
        assert (result["end_time"] <= end).all()

    def test_empty_window(self, simple_event_log):
        start = pd.Timestamp("2025-01-02 00:00", tz="UTC")
        end = pd.Timestamp("2025-01-02 01:00", tz="UTC")
        result = trim_events(simple_event_log, start, end)
        assert result.empty


# ── _split_cases ─────────────────────────────────────────────────────

class TestSplitCases:
    def test_ongoing_and_complete(self, simple_event_log):
        # Cut at 10:30 — Case 1 started at 08:00, ends at 12:00 → ongoing
        # Case 2 started at 09:00, ends at 14:00 → ongoing
        # Case 3 started at 11:00 → complete (starts after cut)
        start = pd.Timestamp("2025-01-01 10:30", tz="UTC")
        end = pd.Timestamp("2025-01-01 15:00", tz="UTC")
        ongoing, complete = _split_cases(simple_event_log, start, end)
        ongoing_ids = set(ongoing["case_id"].unique())
        complete_ids = set(complete["case_id"].unique())
        assert 1 in ongoing_ids
        assert 2 in ongoing_ids
        assert 3 in complete_ids


# ── split_into_subsets ───────────────────────────────────────────────

class TestSplitIntoSubsets:
    def test_returns_three_dataframes(self, simple_event_log):
        start = pd.Timestamp("2025-01-01 10:00", tz="UTC")
        end = pd.Timestamp("2025-01-01 14:00", tz="UTC")
        event_filter, ongoing, complete = split_into_subsets(simple_event_log, start, end)
        assert isinstance(event_filter, pd.DataFrame)
        assert isinstance(ongoing, pd.DataFrame)
        assert isinstance(complete, pd.DataFrame)


# ── avg_remaining_time ───────────────────────────────────────────────

class TestAvgRemainingTime:
    def test_known_values(self, event_log_ids):
        df = pd.DataFrame({
            "case_id": [1, 2],
            "activity": ["A", "A"],
            "resource": ["R1", "R1"],
            "start_time": pd.to_datetime(["2025-01-01 08:00", "2025-01-01 09:00"], utc=True),
            "end_time": pd.to_datetime(["2025-01-01 12:00", "2025-01-01 14:00"], utc=True),
        })
        cutoff = pd.Timestamp("2025-01-01 10:00", tz="UTC")
        # Case 1: 12:00 - 10:00 = 2h = 7200s
        # Case 2: 14:00 - 10:00 = 4h = 14400s
        # Mean = 10800s
        result = avg_remaining_time(df, cutoff, event_log_ids)
        assert result == pytest.approx(10800.0)

    def test_empty_log(self, event_log_ids):
        df = pd.DataFrame(columns=["case_id", "activity", "resource", "start_time", "end_time"])
        cutoff = pd.Timestamp("2025-01-01 10:00", tz="UTC")
        assert avg_remaining_time(df, cutoff, event_log_ids) == 0.0


# ── _build_ps_subsets ────────────────────────────────────────────────

class TestBuildPsSubsets:
    def test_basic_partitioning(self):
        df = pd.DataFrame({
            "case_id": ["c1", "c1", "c2", "c2", "c3", "c3", "c4", "c4"],
            "activity": ["A", "B"] * 4,
            "resource": ["R1"] * 8,
            "start_time": pd.to_datetime([
                "2025-01-01 08:00", "2025-01-01 09:00",
                "2025-01-01 08:30", "2025-01-01 10:00",
                "2025-01-01 10:30", "2025-01-01 11:00",
                "2025-01-01 11:30", "2025-01-01 12:00",
            ], utc=True),
            "end_time": pd.to_datetime([
                "2025-01-01 09:00", "2025-01-01 11:00",
                "2025-01-01 10:00", "2025-01-01 12:00",
                "2025-01-01 11:00", "2025-01-01 13:00",
                "2025-01-01 12:00", "2025-01-01 14:00",
            ], utc=True),
        })
        cut = pd.Timestamp("2025-01-01 10:00", tz="UTC")
        end = pd.Timestamp("2025-01-01 14:00", tz="UTC")
        # c1 and c2 are ongoing at cut (started before, end after cut)
        # partial_ids from simulator: only c1
        partial_ids = {"c1"}
        event_filter, ongoing, complete = _build_ps_subsets(df, partial_ids, cut, end)
        # c1 and c2 should both be in ongoing (c2 via log evidence)
        ongoing_ids = set(ongoing["case_id"].unique())
        assert "c1" in ongoing_ids
        assert "c2" in ongoing_ids
        # c3 and c4 start after cut → complete
        complete_ids = set(complete["case_id"].unique())
        assert "c3" in complete_ids or "c4" in complete_ids


# ── _avg_events_per_ongoing_case ─────────────────────────────────────

class TestAvgEventsPerOngoingCase:
    def test_simple_counts(self):
        df = pd.DataFrame({
            "case_id": [1, 1, 1, 2, 2],
            "activity": ["A", "B", "C", "A", "B"],
        })
        # 5 events / 2 cases = 2.5
        assert _avg_events_per_ongoing_case(df) == pytest.approx(2.5)

    def test_empty_df(self):
        df = pd.DataFrame(columns=["case_id", "activity"])
        assert _avg_events_per_ongoing_case(df) is None


# ── _avg_events_per_case_diff ────────────────────────────────────────

class TestAvgEventsPerCaseDiff:
    def test_known_diff(self):
        A = pd.DataFrame({"case_id": [1, 1, 2], "activity": ["A", "B", "A"]})
        G = pd.DataFrame({"case_id": [1, 2, 2, 2], "activity": ["A", "A", "B", "C"]})
        # A: 3/2=1.5, G: 4/2=2.0, diff = 0.5
        assert _avg_events_per_case_diff(A, G) == pytest.approx(0.5)

    def test_empty_returns_none(self):
        A = pd.DataFrame(columns=["case_id", "activity"])
        G = pd.DataFrame({"case_id": [1], "activity": ["A"]})
        assert _avg_events_per_case_diff(A, G) is None


# ── compute_cut_points ───────────────────────────────────────────────

class TestComputeCutPoints:
    @pytest.fixture()
    def wide_log(self):
        """Log spanning 60 days with multiple cases."""
        np.random.seed(42)
        records = []
        for i in range(100):
            start = pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(hours=i * 12)
            records.append({
                "case_id": i,
                "activity": "A",
                "resource": "R1",
                "start_time": start,
                "end_time": start + pd.Timedelta(hours=6),
            })
        return pd.DataFrame(records)

    def test_fixed_strategy(self, wide_log):
        cuts = compute_cut_points(wide_log, 5, strategy="fixed", fixed_cut="2025-01-10T10:00:00Z")
        assert len(cuts) == 1
        assert cuts[0] == pd.Timestamp("2025-01-10T10:00:00Z")

    def test_fixed_requires_cut(self, wide_log):
        with pytest.raises(ValueError, match="needs a cut-off"):
            compute_cut_points(wide_log, 5, strategy="fixed")

    def test_segment10_returns_ten(self, wide_log):
        cuts = compute_cut_points(wide_log, 5, strategy="segment10", rng=np.random.default_rng(0))
        assert len(cuts) == 10
        # All should be sorted chronologically within the safe window
        for i in range(len(cuts) - 1):
            assert cuts[i] < cuts[i + 1]

    def test_wip3_returns_three(self, wide_log):
        cuts = compute_cut_points(wide_log, 5, strategy="wip3")
        assert len(cuts) == 3

    def test_unknown_strategy_raises(self, wide_log):
        with pytest.raises(ValueError, match="unknown"):
            compute_cut_points(wide_log, 5, strategy="banana")


# ── build_aggregated_from_cuts ───────────────────────────────────────

class TestBuildAggregatedFromCuts:
    def test_averages_means(self):
        dicts = [
            {"n_gram": {"mean": 0.3, "ci": 0.01}, "absolute_event": {"mean": 0.1, "ci": 0.02}},
            {"n_gram": {"mean": 0.5, "ci": 0.03}, "absolute_event": {"mean": 0.3, "ci": 0.04}},
            {"n_gram": {"mean": 0.4, "ci": 0.02}, "absolute_event": {"mean": 0.2, "ci": 0.01}},
        ]
        result = build_aggregated_from_cuts(dicts)
        assert result["n_gram"]["mean"] == pytest.approx(0.4)
        assert result["absolute_event"]["mean"] == pytest.approx(0.2)

    def test_handles_none_means(self):
        dicts = [
            {"metric_a": {"mean": None, "ci": None}},
            {"metric_a": {"mean": 0.5, "ci": 0.1}},
        ]
        result = build_aggregated_from_cuts(dicts)
        assert result["metric_a"]["mean"] == pytest.approx(0.5)

    def test_all_none(self):
        dicts = [
            {"metric_a": {"mean": None, "ci": None}},
            {"metric_a": {"mean": None, "ci": None}},
        ]
        result = build_aggregated_from_cuts(dicts)
        assert result["metric_a"]["mean"] is None
