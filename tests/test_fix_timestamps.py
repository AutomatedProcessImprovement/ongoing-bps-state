"""Tests for tools.fix_timestamps module."""
from __future__ import annotations

import pandas as pd
import pytest

from tools.fix_timestamps import _format_to_ms, fix_canonical_log


class TestFormatToMs:
    def test_known_datetime(self):
        series = pd.to_datetime(pd.Series(["2025-01-01 08:30:45.123456"]))
        result = _format_to_ms(series)
        assert result.iloc[0] == "2025-01-01T08:30:45.123"

    def test_preserves_three_decimals(self):
        series = pd.to_datetime(pd.Series(["2025-06-15 23:59:59.999999"]))
        result = _format_to_ms(series)
        assert result.iloc[0] == "2025-06-15T23:59:59.999"

    def test_zero_microseconds(self):
        series = pd.to_datetime(pd.Series(["2025-01-01 00:00:00.000000"]))
        result = _format_to_ms(series)
        assert result.iloc[0] == "2025-01-01T00:00:00.000"


class TestFixCanonicalLog:
    def test_basic_conversion(self, tmp_path):
        input_csv = tmp_path / "input.csv"
        output_csv = tmp_path / "output.csv"
        input_csv.write_text(
            "case_id,activity,resource,start_time,end_time,enabled_time\n"
            "1,A,R1,2025-01-01T08:00:00.762000Z,2025-01-01T09:23:43.902000Z,2025-01-01 08:00:00.762000+0000\n"
        )
        fix_canonical_log(input_csv, output_csv)
        df = pd.read_csv(output_csv)
        # enabled_time should be renamed to enable_time
        assert "enable_time" in df.columns
        assert "enabled_time" not in df.columns
        # Timestamps should be in ms format
        assert df["start_time"].iloc[0] == "2025-01-01T08:00:00.762"
        assert df["end_time"].iloc[0] == "2025-01-01T09:23:43.902"
        assert df["enable_time"].iloc[0] == "2025-01-01T08:00:00.762"

    def test_missing_column_raises(self, tmp_path):
        input_csv = tmp_path / "bad.csv"
        output_csv = tmp_path / "out.csv"
        input_csv.write_text("case_id,activity\n1,A\n")
        with pytest.raises(ValueError, match="Missing"):
            fix_canonical_log(input_csv, output_csv)

    def test_inplace_same_path(self, tmp_path):
        csv = tmp_path / "log.csv"
        csv.write_text(
            "case_id,start_time,end_time,enabled_time\n"
            "1,2025-01-01T10:00:00.000Z,2025-01-01T11:00:00.000Z,2025-01-01 10:00:00.000+0000\n"
        )
        fix_canonical_log(csv, csv)
        df = pd.read_csv(csv)
        assert "enable_time" in df.columns
