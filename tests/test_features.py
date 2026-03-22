"""Tests for evaluation.features module."""
from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from evaluation.features import _safe_activity_name, prepare_feature_env, compute_features_at_cut


class TestSafeActivityName:
    def test_simple_name(self):
        assert _safe_activity_name("CheckForm") == "CheckForm"

    def test_spaces_and_special_chars(self):
        result = _safe_activity_name("Check application form (v2)")
        assert " " not in result
        assert "(" not in result
        assert result == "Check_application_form_v2_"  or "_" in result

    def test_empty_string_fallback(self):
        assert _safe_activity_name("!!!") == "act"

    def test_numeric_prefix(self):
        assert _safe_activity_name("123-start") == "123_start"


class TestPrepareFeatureEnv:
    def test_returns_feature_env(self, simple_event_log):
        env = prepare_feature_env(simple_event_log)
        assert hasattr(env, "case_arrival")
        assert hasattr(env, "resources")
        assert hasattr(env, "activities")
        assert hasattr(env, "has_enable_time")
        assert hasattr(env, "activity_name_map")
        assert len(env.activities) > 0
        assert len(env.resources) > 0

    def test_activity_name_map_populated(self, simple_event_log):
        env = prepare_feature_env(simple_event_log)
        for act in env.activities:
            assert act in env.activity_name_map


class TestComputeFeaturesAtCut:
    def test_basic_features_present(self, simple_event_log):
        env = prepare_feature_env(simple_event_log)
        cut = pd.Timestamp("2025-01-01 10:30", tz="UTC")
        horizon = pd.Timedelta(hours=4)
        features = compute_features_at_cut(simple_event_log, cut, horizon, env)
        assert "wip" in features
        assert "arrival_rate_per_hour" in features
        assert "resource_availability" in features
        assert features["wip"] >= 0

    def test_wip_count(self, simple_event_log):
        env = prepare_feature_env(simple_event_log)
        # At 10:30: case1 (08:00-12:00) ongoing, case2 (09:00-14:00) ongoing
        # case3 starts at 11:00, not yet
        cut = pd.Timestamp("2025-01-01 10:30", tz="UTC")
        horizon = pd.Timedelta(hours=4)
        features = compute_features_at_cut(simple_event_log, cut, horizon, env)
        assert features["wip"] == 2.0

    def test_per_activity_features(self, simple_event_log):
        env = prepare_feature_env(simple_event_log)
        cut = pd.Timestamp("2025-01-01 10:30", tz="UTC")
        horizon = pd.Timedelta(hours=4)
        features = compute_features_at_cut(simple_event_log, cut, horizon, env)
        # Should have wip_ongoing_* and statevec_* for each activity
        for act in env.activities:
            safe = env.activity_name_map[act]
            assert f"wip_ongoing_{safe}" in features
            assert f"statevec_{safe}" in features
