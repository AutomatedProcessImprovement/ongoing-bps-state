"""Tests for src.runner module."""
from __future__ import annotations

from src.runner import RunnerArgs


class TestRunnerArgs:
    def test_construction_with_required_fields(self):
        args = RunnerArgs(
            event_log="path/to/log.csv",
            bpmn_model="path/to/model.bpmn",
            bpmn_parameters="path/to/params.json",
        )
        assert args.event_log == "path/to/log.csv"
        assert args.bpmn_model == "path/to/model.bpmn"
        assert args.bpmn_parameters == "path/to/params.json"
        assert args.start_time is None
        assert args.column_mapping is None

    def test_construction_with_all_fields(self):
        args = RunnerArgs(
            event_log="log.csv",
            bpmn_model="model.bpmn",
            bpmn_parameters="params.json",
            start_time="2025-01-01T10:00:00Z",
            column_mapping='{"case_id": "CaseId"}',
        )
        assert args.start_time == "2025-01-01T10:00:00Z"
        assert args.column_mapping == '{"case_id": "CaseId"}'

    def test_has_expected_attributes_for_input_handler(self):
        """InputHandler accesses .event_log, .bpmn_model, .bpmn_parameters, .start_time, .column_mapping."""
        args = RunnerArgs(
            event_log="a", bpmn_model="b", bpmn_parameters="c",
        )
        # These are the 5 attributes InputHandler.__init__ reads
        assert hasattr(args, "event_log")
        assert hasattr(args, "bpmn_model")
        assert hasattr(args, "bpmn_parameters")
        assert hasattr(args, "start_time")
        assert hasattr(args, "column_mapping")
