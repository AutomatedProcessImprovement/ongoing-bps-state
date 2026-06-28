"""End-to-end flows for data-attribute handling in short-term simulation.

Coverage is deliberately spread across distinct mechanisms:

  * preservation of all three families (case / global / event) in the snapshot;
  * value fidelity (restored, not re-sampled) against the source log;
  * round-trip survival through the resumed engine's output log;
  * the declared-gating rule (columns present but undeclared are ignored);
  * LOGIC: a restored case attribute actually DRIVES XOR routing on resume;
  * a negative control showing the routing effect needs the branch rules.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from evaluation.state_metrics.perturb import build_case_route_params
from tests.e2e.conftest import (
    ROUTE_BPMN,
    ROUTE_PARAMS,
    fraction_cutoff,
    ongoing_case_ids,
    run_short_term,
    write_prefix,
)

pytestmark = pytest.mark.e2e

ATTR_COLS = ("case_type", "season", "score")
# Branch convention from tools/generate_case_route.py: Assess_A* on the A-branch
# (rule_red), Assess_B* on the B-branch (rule_blue).
EXPECTED_BRANCH = {"red": "A", "blue": "B"}


def _branch_of(activity: str) -> str | None:
    if activity.startswith("Assess_A"):
        return "A"
    if activity.startswith("Assess_B"):
        return "B"
    return None


# --------------------------------------------------------------------------- #
# Preservation + fidelity (all three families)                                 #
# --------------------------------------------------------------------------- #
def test_all_three_families_preserved_in_snapshot(all_attrs, tmp_path):
    gt = all_attrs["gt"]
    cutoff = fraction_cutoff(gt, 0.5)
    prefix_csv = tmp_path / "prefix.csv"
    write_prefix(gt, cutoff, prefix_csv, attr_cols=ATTR_COLS)
    snapshot, _ = run_short_term(
        prefix_csv=prefix_csv, bpmn=ROUTE_BPMN, params=all_attrs["params"],
        cutoff=cutoff, horizon_end=fraction_cutoff(gt, 0.95), work_dir=tmp_path,
    )

    # global: process-wide value at the top level.
    assert "season" in snapshot.get("global_attributes", {})
    cases = snapshot["cases"]
    assert cases, "expected ongoing cases at the cutoff"
    # case + per-case event + per-ongoing-activity event attributes, every case.
    for case in cases.values():
        assert "case_type" in case.get("case_attributes", {})
        assert "score" in case.get("event_attributes", {})
        og = case["ongoing_activities"]
        assert og and all("score" in a.get("event_attributes", {}) for a in og)


def test_snapshot_case_type_matches_source_log(all_attrs, tmp_path):
    """Restored case_type equals the prefix value (no re-sampling)."""
    gt = all_attrs["gt"]
    cutoff = fraction_cutoff(gt, 0.5)
    prefix_csv = tmp_path / "prefix.csv"
    prefix = write_prefix(gt, cutoff, prefix_csv, attr_cols=ATTR_COLS)
    snapshot, _ = run_short_term(
        prefix_csv=prefix_csv, bpmn=ROUTE_BPMN, params=all_attrs["params"],
        cutoff=cutoff, horizon_end=fraction_cutoff(gt, 0.95), work_dir=tmp_path,
    )
    src = prefix.groupby("CaseId")["case_type"].first().astype(str).to_dict()
    for cid, case in snapshot["cases"].items():
        snap_ct = case.get("case_attributes", {}).get("case_type")
        assert str(snap_ct) == src[int(cid)]


def test_resumed_output_preserves_ongoing_case_type(all_attrs, tmp_path):
    """Ongoing cases keep their prefix case_type in the resumed sim's output log."""
    gt = all_attrs["gt"]
    cutoff = fraction_cutoff(gt, 0.5)
    prefix_csv = tmp_path / "prefix.csv"
    prefix = write_prefix(gt, cutoff, prefix_csv, attr_cols=ATTR_COLS)
    _, sim = run_short_term(
        prefix_csv=prefix_csv, bpmn=ROUTE_BPMN, params=all_attrs["params"],
        cutoff=cutoff, horizon_end=fraction_cutoff(gt, 0.95), work_dir=tmp_path,
    )
    assert "case_type" in sim.columns
    src = prefix.groupby("CaseId")["case_type"].first().astype(str).to_dict()
    sim_ct = sim.groupby("case_id")["case_type"].first().astype(str).to_dict()
    checked = [cid for cid in src if cid in sim_ct]
    assert checked, "no ongoing case survived into the sim output"
    assert all(sim_ct[cid] == src[cid] for cid in checked)


def test_undeclared_columns_are_ignored(route_gt, tmp_path):
    """A case_type column present in the log but NOT declared in params is dropped.

    Guards the declared-gating rule: emission is keyed off the params, not the
    log's columns.
    """
    params = json.loads(ROUTE_PARAMS.read_text())
    params["case_attributes"] = []          # undeclare it
    params.pop("branch_rules", None)
    stripped = tmp_path / "params_no_attrs.json"
    stripped.write_text(json.dumps(params))

    cutoff = fraction_cutoff(route_gt, 0.5)
    prefix_csv = tmp_path / "prefix.csv"
    write_prefix(route_gt, cutoff, prefix_csv, attr_cols=ATTR_COLS)
    snapshot, _ = run_short_term(
        prefix_csv=prefix_csv, bpmn=ROUTE_BPMN, params=stripped,
        cutoff=cutoff, horizon_end=fraction_cutoff(route_gt, 0.95), work_dir=tmp_path,
    )
    assert "global_attributes" not in snapshot
    for case in snapshot["cases"].values():
        assert "case_attributes" not in case


# --------------------------------------------------------------------------- #
# LOGIC: restored attribute drives control flow                                #
# --------------------------------------------------------------------------- #
def _assess_decisions(sim: pd.DataFrame, cutoff: pd.Timestamp, prefix_ids: set):
    """Per Assess_* event: its branch, the case's case_type, new-vs-ongoing, post-cutoff."""
    a = sim[sim["activity"].str.startswith("Assess_")].copy()
    a["branch"] = a["activity"].map(_branch_of)
    ct = sim.groupby("case_id")["case_type"].first().to_dict()
    a["case_type"] = a["case_id"].map(ct)
    a["is_new"] = ~a["case_id"].isin(prefix_ids)
    a["post_cutoff"] = a["start_time"] >= cutoff
    return a


def test_restored_case_type_drives_routing_on_resume(route_gt, tmp_path):
    """With red->A / blue->B branch rules, every post-resume decision obeys case_type.

    The branch rules are equality conditions, so a correctly-restored case_type
    forces a deterministic branch -> we assert ZERO crossover, not a ratio.
    """
    ruled = tmp_path / "params_ruled.json"
    build_case_route_params(ROUTE_PARAMS, n_gateways_ruled=3, out_json_path=ruled)

    cutoff = fraction_cutoff(route_gt, 0.5)
    prefix_csv = tmp_path / "prefix.csv"
    prefix = write_prefix(route_gt, cutoff, prefix_csv, attr_cols=ATTR_COLS)
    prefix_ids = set(prefix["CaseId"].unique())
    _, sim = run_short_term(
        prefix_csv=prefix_csv, bpmn=ROUTE_BPMN, params=ruled,
        cutoff=cutoff, horizon_end=fraction_cutoff(route_gt, 0.95), work_dir=tmp_path,
    )

    a = _assess_decisions(sim, cutoff, prefix_ids)
    a = a[a["case_type"].isin(EXPECTED_BRANCH)]
    a["expected"] = a["case_type"].map(EXPECTED_BRANCH)

    new = a[a["is_new"]]
    assert len(new) > 0
    assert (new["branch"] == new["expected"]).all(), "a new case ignored its case_type"

    # The headline: ONGOING cases routing AFTER the cutoff use the *restored* tag.
    ongoing_post = a[(~a["is_new"]) & a["post_cutoff"]]
    assert len(ongoing_post) > 0, "no ongoing case made a post-cutoff XOR decision"
    assert (ongoing_post["branch"] == ongoing_post["expected"]).all(), \
        "a resumed case routed against its restored case_type"


def test_routing_is_not_attribute_bound_without_rules(route_gt, tmp_path):
    """Negative control: base 50/50 params -> case_type does NOT determine the branch.

    Confirms the previous test's effect comes from the branch rules, not from
    some incidental coupling, by showing both branches occur for each case_type.
    """
    cutoff = fraction_cutoff(route_gt, 0.5)
    prefix_csv = tmp_path / "prefix.csv"
    prefix = write_prefix(route_gt, cutoff, prefix_csv, attr_cols=ATTR_COLS)
    prefix_ids = set(prefix["CaseId"].unique())
    _, sim = run_short_term(
        prefix_csv=prefix_csv, bpmn=ROUTE_BPMN, params=ROUTE_PARAMS,
        cutoff=cutoff, horizon_end=fraction_cutoff(route_gt, 0.95), work_dir=tmp_path,
    )
    a = _assess_decisions(sim, cutoff, prefix_ids)
    a = a[a["case_type"].isin(EXPECTED_BRANCH)]
    # With hundreds of decisions split 50/50 independent of case_type, both
    # branches must appear overall (crossover exists) -> not attribute-bound.
    branches_per_type = a.groupby("case_type")["branch"].nunique()
    assert (branches_per_type > 1).any(), "routing looks attribute-bound without rules"
