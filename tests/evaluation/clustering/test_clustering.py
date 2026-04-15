"""Tests for evaluation.clustering.models module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.clustering.models import (
    _mean_ci,
    _to_jsonable,
    train_baseline_model,
    train_clustering_models,
    apply_models_to_test,
    summarise_evaluation,
)


# ── _mean_ci (prediction interval, NOT confidence interval) ──────────

class TestMeanCi:
    def test_prediction_interval_no_sqrt_n(self):
        """Verify this is mean ± 1.96*std, NOT mean ± 1.96*std/sqrt(n)."""
        vals = np.array([10.0, 20.0, 30.0])
        mean, ci = _mean_ci(vals)
        std = float(np.std(vals, ddof=1))
        # Prediction interval: 1.96 * std (no /sqrt(n))
        assert ci == pytest.approx(1.96 * std)
        # Ensure it's NOT the confidence interval version
        ci_for_mean = 1.96 * std / np.sqrt(len(vals))
        assert ci != pytest.approx(ci_for_mean)

    def test_empty_returns_none(self):
        mean, ci = _mean_ci(np.array([]))
        assert mean is None
        assert ci is None

    def test_single_value(self):
        mean, ci = _mean_ci(np.array([5.0]))
        assert mean == pytest.approx(5.0)
        assert ci == pytest.approx(0.0)

    def test_nan_filtered(self):
        vals = np.array([1.0, np.nan, 3.0])
        mean, ci = _mean_ci(vals)
        assert mean == pytest.approx(2.0)


# ── _to_jsonable ─────────────────────────────────────────────────────

class TestToJsonable:
    def test_numpy_int(self):
        assert _to_jsonable(np.int64(42)) == 42
        assert isinstance(_to_jsonable(np.int64(42)), int)

    def test_numpy_float(self):
        result = _to_jsonable(np.float64(3.14))
        assert isinstance(result, float)

    def test_numpy_array(self):
        result = _to_jsonable(np.array([1, 2, 3]))
        assert result == [1, 2, 3]

    def test_nested_dict(self):
        result = _to_jsonable({"a": np.int64(1), "b": [np.float64(2.0)]})
        assert result == {"a": 1, "b": [2.0]}

    def test_plain_types_passthrough(self):
        assert _to_jsonable("hello") == "hello"
        assert _to_jsonable(42) == 42


# ── train_baseline_model ─────────────────────────────────────────────

class TestTrainBaselineModel:
    def test_basic_structure(self):
        df = pd.DataFrame({"err_RTD_mean": [0.1, 0.2, 0.3, 0.4, 0.5]})
        model = train_baseline_model(df, "err_RTD_mean")
        assert "target_col" in model
        assert "mean" in model
        assert "ci" in model
        assert "quantiles" in model
        assert model["mean"] == pytest.approx(0.3)


# ── train_clustering_models ──────────────────────────────────────────

class TestTrainClusteringModels:
    @pytest.fixture()
    def sample_df(self):
        np.random.seed(42)
        n = 30
        return pd.DataFrame({
            "wip": np.random.randint(1, 10, n).astype(float),
            "arrival_rate_per_hour": np.random.uniform(0.5, 5.0, n),
            "resource_availability": np.random.uniform(0.1, 1.0, n),
            "wip_enabled_A": np.random.randint(0, 3, n).astype(float),
            "wip_ongoing_A": np.random.randint(0, 3, n).astype(float),
            "statevec_A": np.random.uniform(0, 1, n),
            "err_RTD_mean": np.random.uniform(0.0, 2.0, n),
        })

    def test_returns_all_six_methods(self, sample_df):
        models = train_clustering_models(sample_df, "err_RTD_mean")
        expected_keys = {
            "baseline", "random_groups", "wip_deciles",
            "kmeans_simple", "kmeans_advanced_wip", "kmeans_statevec",
        }
        assert set(models.keys()) == expected_keys

    def test_baseline_has_mean(self, sample_df):
        models = train_clustering_models(sample_df, "err_RTD_mean")
        assert models["baseline"]["mean"] is not None


# ── apply_models_to_test ─────────────────────────────────────────────

class TestApplyModelsToTest:
    def test_returns_dataframe_with_coverage(self):
        np.random.seed(42)
        n = 20
        train_df = pd.DataFrame({
            "wip": np.random.randint(1, 10, n).astype(float),
            "arrival_rate_per_hour": np.random.uniform(0.5, 5.0, n),
            "resource_availability": np.random.uniform(0.1, 1.0, n),
            "statevec_A": np.random.uniform(0, 1, n),
            "err_RTD_mean": np.random.uniform(0.0, 2.0, n),
            "cut_time_iso": ["2025-01-01T10:00:00Z"] * n,
            "split": ["test"] * n,
        })
        models = train_clustering_models(train_df, "err_RTD_mean")
        result = apply_models_to_test(models, train_df, "err_RTD_mean")
        assert isinstance(result, pd.DataFrame)
        assert "true_error" in result.columns
        assert "baseline_in_ci" in result.columns
        assert len(result) == n


# ── summarise_evaluation ─────────────────────────────────────────────

class TestSummariseEvaluation:
    def test_summary_columns(self):
        eval_df = pd.DataFrame({
            "true_error": [0.1, 0.2, 0.3],
            "baseline_in_ci": [True, True, False],
            "baseline_ci": [0.5, 0.5, 0.5],
        })
        summary = summarise_evaluation(eval_df)
        assert "method_key" in summary.columns
        assert "coverage" in summary.columns
        assert "avg_ci_width" in summary.columns
        assert len(summary) >= 1
