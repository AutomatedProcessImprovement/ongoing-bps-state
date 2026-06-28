"""End-to-end flows for the generic (attribute-less) short-term simulation.

Each test runs a real Prosimos GT sim -> prefix cut -> partial-state snapshot ->
resumed short-term sim, and asserts one property of that pipeline. They are
intentionally varied: snapshot schema, the no-op (attribute-less) guarantee,
case continuation, cutoff boundary semantics, new-arrival generation, the
horizon bound, and horizon monotonicity.
"""
from __future__ import annotations

import pandas as pd
import pytest

from tests.e2e.conftest import (
    XOR_BPMN,
    XOR_PARAMS,
    fraction_cutoff,
    finished_case_ids,
    ongoing_case_ids,
    run_short_term,
    write_prefix,
)

pytestmark = pytest.mark.e2e

# The full snapshot contract a resumed engine relies on.
_PER_CASE_KEYS = {
    "control_flow_state", "ongoing_activities",
    "enabled_activities", "enabled_gateways", "enabled_events",
}


def _prefix_and_run(xor_gt, tmp_path, *, fraction=0.5, horizon_fraction=0.95):
    cutoff = fraction_cutoff(xor_gt, fraction)
    horizon_end = fraction_cutoff(xor_gt, horizon_fraction)
    prefix_csv = tmp_path / "prefix.csv"
    write_prefix(xor_gt, cutoff, prefix_csv)
    snapshot, sim = run_short_term(
        prefix_csv=prefix_csv, bpmn=XOR_BPMN, params=XOR_PARAMS,
        cutoff=cutoff, horizon_end=horizon_end, work_dir=tmp_path,
    )
    return cutoff, horizon_end, snapshot, sim


def test_snapshot_schema_is_well_formed(xor_gt, tmp_path):
    """Every case in the snapshot exposes the full partial-state contract."""
    _, _, snapshot, _ = _prefix_and_run(xor_gt, tmp_path)
    assert "last_case_arrival" in snapshot
    assert snapshot["cases"], "expected at least one ongoing case at the cutoff"
    for case in snapshot["cases"].values():
        assert _PER_CASE_KEYS <= set(case), f"missing keys: {_PER_CASE_KEYS - set(case)}"
        cfs = case["control_flow_state"]
        assert set(cfs) == {"flows", "activities"}
        assert isinstance(cfs["flows"], list) and isinstance(cfs["activities"], list)


def test_attributeless_dataset_emits_no_attribute_fields(xor_gt, tmp_path):
    """No declared attributes -> snapshot carries no attribute keys anywhere."""
    _, _, snapshot, _ = _prefix_and_run(xor_gt, tmp_path)
    assert "global_attributes" not in snapshot
    for case in snapshot["cases"].values():
        assert "case_attributes" not in case
        assert "event_attributes" not in case
        for act in case["ongoing_activities"]:
            assert "event_attributes" not in act


def test_ongoing_cases_continue_after_resume(xor_gt, tmp_path):
    """Cases ongoing at the cutoff acquire new events at/after the cutoff."""
    cutoff, _, snapshot, sim = _prefix_and_run(xor_gt, tmp_path)
    snap_ids = {int(c) for c in snapshot["cases"]}
    post = sim[sim["start_time"] >= cutoff]
    continued = set(post["case_id"].unique()) & snap_ids
    assert continued, "no ongoing case produced a post-cutoff event"


def test_finished_cases_excluded_ongoing_included(xor_gt, tmp_path):
    """Snapshot = exactly the cases ongoing at the cutoff (finished ones dropped)."""
    cutoff = fraction_cutoff(xor_gt, 0.5)
    prefix_csv = tmp_path / "prefix.csv"
    write_prefix(xor_gt, cutoff, prefix_csv)
    snapshot, _ = run_short_term(
        prefix_csv=prefix_csv, bpmn=XOR_BPMN, params=XOR_PARAMS,
        cutoff=cutoff, horizon_end=fraction_cutoff(xor_gt, 0.95), work_dir=tmp_path,
    )
    snap_ids = {int(c) for c in snapshot["cases"]}
    finished = {int(c) for c in finished_case_ids(xor_gt, cutoff)}
    ongoing = {int(c) for c in ongoing_case_ids(xor_gt, cutoff)}
    assert snap_ids.isdisjoint(finished), "a case finished before the cutoff leaked in"
    assert ongoing <= snap_ids, "an ongoing case is missing from the snapshot"


def test_new_cases_arrive_after_cutoff(xor_gt, tmp_path):
    """The resumed sim generates fresh cases not present in the prefix.

    Arrivals are resampled from the snapshot's ``last_case_arrival`` anchor (which
    precedes the cutoff), so a new case may begin slightly before the cutoff; the
    invariant we assert is that genuinely future cases appear and none predate the
    resume anchor.
    """
    cutoff, _, snapshot, sim = _prefix_and_run(xor_gt, tmp_path)
    anchor = pd.Timestamp(snapshot["last_case_arrival"])
    snap_ids = {int(c) for c in snapshot["cases"]}
    new_ids = set(sim["case_id"].unique()) - snap_ids
    assert new_ids, "expected at least one newly-arrived case"
    arrivals = sim[sim["case_id"].isin(new_ids)].groupby("case_id")["start_time"].min()
    assert (arrivals > cutoff).any(), "no genuinely future case was generated"
    assert (arrivals >= anchor).all(), "a new case predates the resume anchor"


def test_output_respects_horizon(xor_gt, tmp_path):
    """The horizon bounds NEW-case arrivals (first activity), per the engine.

    Started cases run to completion past the horizon; only fresh arrivals are
    gated (simulation_engine: "horizon only filters out NEW cases whose first
    activity starts at/after the horizon"). So we assert on per-new-case arrival
    times, not on every event.
    """
    cutoff, horizon_end, snapshot, sim = _prefix_and_run(
        xor_gt, tmp_path, horizon_fraction=0.6
    )
    snap_ids = {int(c) for c in snapshot["cases"]}
    arrivals = sim.groupby("case_id")["start_time"].min()
    new_arrivals = arrivals[~arrivals.index.isin(snap_ids)]
    assert len(new_arrivals) > 0
    assert (new_arrivals < horizon_end).all()


def test_longer_horizon_yields_more_events(xor_gt, tmp_path):
    """Extending the horizon strictly increases the number of simulated events."""
    cutoff = fraction_cutoff(xor_gt, 0.4)
    prefix_csv = tmp_path / "prefix.csv"
    write_prefix(xor_gt, cutoff, prefix_csv)

    short_end = fraction_cutoff(xor_gt, 0.45)
    long_end = fraction_cutoff(xor_gt, 0.95)
    _, sim_short = run_short_term(
        prefix_csv=prefix_csv, bpmn=XOR_BPMN, params=XOR_PARAMS,
        cutoff=cutoff, horizon_end=short_end, work_dir=tmp_path / "short",
    )
    _, sim_long = run_short_term(
        prefix_csv=prefix_csv, bpmn=XOR_BPMN, params=XOR_PARAMS,
        cutoff=cutoff, horizon_end=long_end, work_dir=tmp_path / "long",
    )
    assert len(sim_long) > len(sim_short)
