"""Tests for evaluation.evaluation module."""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import pytest

from evaluation.evaluation import _mean_ci, aggregate, compare, _dump


# ── _mean_ci ─────────────────────────────────────────────────────────

class TestMeanCi:
    def test_known_values(self):
        vals = [10.0, 20.0, 30.0]
        mean, ci = _mean_ci(vals)
        assert mean == pytest.approx(20.0)
        expected_std = np.std(vals, ddof=1)
        expected_ci = expected_std * 1.96 / np.sqrt(3)
        assert ci == pytest.approx(expected_ci)

    def test_single_value(self):
        mean, ci = _mean_ci([5.0])
        assert mean == pytest.approx(5.0)
        # std with ddof=1 of single value is nan → ci is nan
        # Implementation uses np.std(ddof=1) which gives nan for n=1

    def test_empty_list(self):
        mean, ci = _mean_ci([])
        assert mean is None
        assert ci is None


# ── aggregate ────────────────────────────────────────────────────────

class TestAggregate:
    def test_three_runs(self):
        runs = [
            {
                "event_filter": {"n_gram": 0.3, "absolute_event": 0.1},
                "ongoing_filter": {"RTD": 0.5},
                "complete_filter": {"cycle_time": 100.0},
            },
            {
                "event_filter": {"n_gram": 0.5, "absolute_event": 0.3},
                "ongoing_filter": {"RTD": 0.7},
                "complete_filter": {"cycle_time": 200.0},
            },
            {
                "event_filter": {"n_gram": 0.4, "absolute_event": 0.2},
                "ongoing_filter": {"RTD": 0.6},
                "complete_filter": {"cycle_time": 150.0},
            },
        ]
        result = aggregate(runs)
        assert result["event_filter"]["n_gram"]["mean"] == pytest.approx(0.4)
        assert result["ongoing_filter"]["RTD"]["mean"] == pytest.approx(0.6)
        assert result["complete_filter"]["cycle_time"]["mean"] == pytest.approx(150.0)
        # CI should be present and non-negative
        assert result["event_filter"]["n_gram"]["ci"] >= 0

    def test_handles_none_values(self):
        runs = [
            {"event_filter": {"metric": None}, "ongoing_filter": {}, "complete_filter": {}},
            {"event_filter": {"metric": 0.5}, "ongoing_filter": {}, "complete_filter": {}},
        ]
        result = aggregate(runs)
        assert result["event_filter"]["metric"]["mean"] == pytest.approx(0.5)


# ── compare ──────────────────────────────────────────────────────────

class TestCompare:
    def test_basic_comparison(self):
        a = {
            "event_filter": {
                "n_gram": {"mean": 0.3, "ci": 0.01},
            },
        }
        b = {
            "event_filter": {
                "n_gram": {"mean": 0.5, "ci": 0.02},
            },
        }
        result = compare(a, b, ("PS", "WU"))
        assert result["event_filter"]["n_gram"]["better"] == "PS"
        assert result["event_filter"]["n_gram"]["PS"] == 0.3
        assert result["event_filter"]["n_gram"]["WU"] == 0.5

    def test_tie(self):
        a = {"event_filter": {"m": {"mean": 0.5, "ci": 0.01}}}
        b = {"event_filter": {"m": {"mean": 0.5, "ci": 0.01}}}
        result = compare(a, b, ("A", "B"))
        assert result["event_filter"]["m"]["better"] == "tie"

    def test_none_handling(self):
        a = {"event_filter": {"m": {"mean": None, "ci": None}}}
        b = {"event_filter": {"m": {"mean": 0.5, "ci": 0.01}}}
        result = compare(a, b, ("A", "B"))
        assert result["event_filter"]["m"]["better"] is None


# ── _dump ────────────────────────────────────────────────────────────

class TestDump:
    def test_writes_csv(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        _dump(df, str(tmp_path), "test_output.csv")
        out_file = tmp_path / "test_output.csv"
        assert out_file.exists()
        loaded = pd.read_csv(out_file)
        assert list(loaded.columns) == ["a", "b"]
        assert len(loaded) == 2

    def test_creates_directory(self, tmp_path):
        subdir = str(tmp_path / "nested" / "dir")
        df = pd.DataFrame({"x": [1]})
        _dump(df, subdir, "out.csv")
        assert os.path.isfile(os.path.join(subdir, "out.csv"))
