"""Tests for src.utils module."""
from __future__ import annotations

import datetime
import pytest

from src.utils import parse_datetime


class TestParseDatetime:
    def test_iso_with_z(self):
        result = parse_datetime("2025-01-01T10:00:00Z")
        assert result == datetime.datetime(2025, 1, 1, 10, 0, 0,
                                           tzinfo=datetime.timezone.utc)

    def test_iso_with_offset(self):
        result = parse_datetime("2025-01-01T10:00:00+02:00")
        assert result.tzinfo is not None
        assert result.hour == 10

    def test_iso_without_tz(self):
        result = parse_datetime("2025-01-01T10:00:00")
        assert result is not None
        assert result.hour == 10

    def test_none_input(self):
        assert parse_datetime(None) is None

    def test_empty_string(self):
        assert parse_datetime("") is None

    def test_has_date_ignored(self):
        """The has_date parameter exists for compatibility but is ignored."""
        result = parse_datetime("2025-01-01T10:00:00Z", has_date=True)
        assert result is not None

    def test_with_fractional_seconds(self):
        result = parse_datetime("2025-01-01T10:00:00.123Z")
        assert result.microsecond == 123000
