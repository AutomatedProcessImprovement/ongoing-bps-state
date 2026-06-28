"""Tests for tools.generate_case_route — the case-route synthetic generator."""
from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from tools.generate_case_route import build_model

BPMN_NS = "http://www.omg.org/spec/BPMN/20100524/MODEL"


def _tasks(bpmn_xml):
    root = ET.fromstring(bpmn_xml)
    return {t.get("id"): t.get("name") for t in root.iter(f"{{{BPMN_NS}}}task")}


def test_build_model_is_well_formed_xml():
    bpmn, _ = build_model(3)
    # Must parse — both Prosimos and ongoing_process_state read this.
    ET.fromstring(bpmn)


def test_branch_activities_distinct_and_duration_symmetric():
    bpmn, params = build_model(3, workers=2, arrival_mean=100, service=240)
    names = _tasks(bpmn)
    # Each gateway k has distinct A/B branch labels...
    for k in (1, 2, 3):
        assert names[f"t_a{k}"] == f"Assess_A{k}"
        assert names[f"t_b{k}"] == f"Assess_B{k}"
    # ...but every task shares one duration distribution (cycle-time neutral).
    durations = {
        tuple(p["value"] for p in r["distribution_params"])
        for t in params["task_resource_distribution"]
        for r in t["resources"]
    }
    assert durations == {(240,)}


def test_case_type_declared_and_rules_present():
    _, params = build_model(2)
    attrs = params["case_attributes"]
    assert len(attrs) == 1 and attrs[0]["name"] == "case_type"
    keys = {v["key"] for v in attrs[0]["values"]}
    assert keys == {"red", "blue"}
    assert {r["id"] for r in params["branch_rules"]} == {"rule_red", "rule_blue"}


def test_n_gateways_controls_split_count():
    _, params = build_model(4)
    splits = [
        e for e in params["gateway_branching_probabilities"]
        if len(e["probabilities"]) == 2
    ]
    assert len(splits) == 4


def test_rejects_zero_gateways():
    with pytest.raises(ValueError):
        build_model(0)
