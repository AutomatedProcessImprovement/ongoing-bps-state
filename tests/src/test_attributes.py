"""Tests for src.attributes — extraction of Prosimos attribute values from a log."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.attributes import (
    _coerce,
    case_attribute_values,
    declared_attribute_names,
    latest_attribute_values,
    present_columns,
    row_attributes,
)


class TestCoerce:
    def test_none_and_nan_become_none(self):
        assert _coerce(None) is None
        assert _coerce(np.nan) is None
        assert _coerce(pd.NA) is None

    def test_numpy_scalars_unwrapped_to_python(self):
        v = _coerce(np.int64(5))
        assert v == 5 and isinstance(v, int)
        v = _coerce(np.float64(2.5))
        assert v == 2.5 and isinstance(v, float)
        assert _coerce(np.bool_(True)) is True

    def test_plain_values_passthrough(self):
        assert _coerce("BUSINESS") == "BUSINESS"
        assert _coerce(7) == 7


class TestDeclaredAttributeNames:
    def test_empty_params(self):
        assert declared_attribute_names({}) == {"case": [], "event": [], "global": []}
        # Falsy/empty declarations are treated as absent.
        assert declared_attribute_names({"case_attributes": [], "prioritisation_rules": []}) == {
            "case": [], "event": [], "global": []
        }

    def test_collects_all_families(self):
        params = {
            "case_attributes": [{"name": "client_type"}, {"name": "amount"}],
            "global_attributes": [{"name": "season"}],
            "event_attributes": [
                {"event_id": "A", "attributes": [{"name": "stage"}, {"name": "score"}]},
                {"event_id": "B", "attributes": [{"name": "stage"}]},  # duplicate name
            ],
        }
        out = declared_attribute_names(params)
        assert out["case"] == ["client_type", "amount"]
        assert out["global"] == ["season"]
        assert out["event"] == ["stage", "score"]  # de-duplicated, order preserved


class TestColumnHelpers:
    def test_present_columns(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        assert present_columns(["a", "b", "missing"], df) == ["a", "b"]

    def test_row_attributes_skips_missing(self):
        row = pd.Series({"stage": np.int64(20), "score": np.nan, "other": 1})
        assert row_attributes(row, ["stage", "score", "absent"]) == {"stage": 20}

    def test_case_attribute_values_first_non_null(self):
        group = pd.DataFrame({"client_type": [np.nan, "BUSINESS", "BUSINESS"]})
        assert case_attribute_values(group, ["client_type"]) == {"client_type": "BUSINESS"}

    def test_case_attribute_values_all_null(self):
        group = pd.DataFrame({"client_type": [np.nan, np.nan]})
        assert case_attribute_values(group, ["client_type"]) == {}

    def test_latest_attribute_values_by_start_time(self):
        df = pd.DataFrame({
            "start_time": pd.to_datetime(
                ["2024-01-01T10:00:00Z", "2024-01-01T12:00:00Z", "2024-01-01T11:00:00Z"]
            ),
            "season": [1, 3, 2],
        })
        # most recent start_time is the 12:00 row -> season 3
        assert latest_attribute_values(df, "start_time", ["season"]) == {"season": 3}

    def test_latest_attribute_values_empty(self):
        df = pd.DataFrame({"start_time": pd.to_datetime([])})
        assert latest_attribute_values(df, "start_time", ["season"]) == {}
        assert latest_attribute_values(pd.DataFrame({"start_time": [1]}), "start_time", []) == {}
