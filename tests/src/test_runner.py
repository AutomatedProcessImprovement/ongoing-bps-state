"""Tests for src.runner module."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.runner import RunnerArgs, run_process_state_and_simulation

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTH_BPMN = REPO_ROOT / "samples" / "dev-samples" / "synthetic_xor_loop.bpmn"
SYNTH_PARAMS = REPO_ROOT / "samples" / "dev-samples" / "synthetic_xor_loop.json"


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


@pytest.mark.skipif(
    not (SYNTH_BPMN.exists() and SYNTH_PARAMS.exists()),
    reason="synthetic_xor_loop assets not available",
)
class TestProcessStateAttributeEmission:
    """The partial-state snapshot must carry the real attribute values so the
    updated Prosimos short-term engine restores (not re-samples) them on resume.
    """

    CUTOFF = "2026-06-16T15:54:00.000+00:00"

    def _write_inputs(self, tmp_path: Path, *, with_attributes: bool):
        # A is complete before the cutoff; B is ongoing at the cutoff.
        rows = [
            # CaseId, Activity, Resource, StartTime, EndTime
            (0, "A", "Pucci", "2026-06-16T15:46:14.000+00:00", "2026-06-16T15:51:14.000+00:00"),
            (0, "B", "DIO", "2026-06-16T15:51:14.000+00:00", "2026-06-16T15:56:14.000+00:00"),
        ]
        df = pd.DataFrame(rows, columns=["CaseId", "Activity", "Resource", "StartTime", "EndTime"])
        params = json.loads(SYNTH_PARAMS.read_text())
        if with_attributes:
            df["client_type"] = "BUSINESS"          # case attribute (constant per case)
            df["stage"] = [10, 20]                   # event attribute (per row)
            df["season"] = 3                          # global attribute
            params["case_attributes"] = [{"name": "client_type"}]
            params["event_attributes"] = [
                {"event_id": "Activity_02r7xaq", "attributes": [{"name": "stage"}]}
            ]
            params["global_attributes"] = [{"name": "season"}]
        log_csv = tmp_path / "log.csv"
        df.to_csv(log_csv, index=False)
        params_json = tmp_path / "params.json"
        params_json.write_text(json.dumps(params))
        return log_csv, params_json

    def _run_and_load_snapshot(self, tmp_path, monkeypatch, *, with_attributes):
        log_csv, params_json = self._write_inputs(tmp_path, with_attributes=with_attributes)
        # The runner writes output.json relative to cwd; isolate it in tmp_path.
        monkeypatch.chdir(tmp_path)
        run_process_state_and_simulation(
            event_log=str(log_csv),
            bpmn_model=str(SYNTH_BPMN),
            bpmn_parameters=str(params_json),
            start_time=self.CUTOFF,
            simulate=False,
        )
        return json.loads((tmp_path / "output.json").read_text())

    def test_attributes_emitted_when_declared(self, tmp_path, monkeypatch):
        snapshot = self._run_and_load_snapshot(tmp_path, monkeypatch, with_attributes=True)

        assert snapshot["global_attributes"] == {"season": 3}
        assert "0" in snapshot["cases"], "case 0 should be ongoing at the cutoff"
        case = snapshot["cases"]["0"]
        assert case["case_attributes"] == {"client_type": "BUSINESS"}
        # baseline per-case event attribute = most recent value before the cut (B -> 20)
        assert case["event_attributes"] == {"stage": 20}
        # B is the ongoing activity; it carries its point-in-time historical value
        ongoing = case["ongoing_activities"]
        assert ongoing, "B should be ongoing"
        assert any(a.get("event_attributes") == {"stage": 20} for a in ongoing)

    def test_no_attribute_fields_when_not_declared(self, tmp_path, monkeypatch):
        snapshot = self._run_and_load_snapshot(tmp_path, monkeypatch, with_attributes=False)

        assert "global_attributes" not in snapshot
        for case in snapshot["cases"].values():
            assert "case_attributes" not in case
            assert "event_attributes" not in case
            for act in case["ongoing_activities"]:
                assert "event_attributes" not in act
