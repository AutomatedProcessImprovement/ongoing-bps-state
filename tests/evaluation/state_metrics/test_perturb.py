import json
from pathlib import Path

import pytest

from evaluation.state_metrics.perturb import (
    build_arrival_burstier_params,
    build_calendar_shifted_params,
    build_case_route_params,
    build_gateway_biased_params,
    build_perturbed_params,
    build_role_swap_params,
)


XOR_JSON = Path(__file__).resolve().parents[3] / "samples" / "dev-samples" / "synthetic_xor_loop.json"
LOAN_JSON = Path(__file__).resolve().parents[3] / "samples" / "icpm-2025" / "synthetic" / "Loan-stable.json"
CASE_ROUTE_JSON = (
    Path(__file__).resolve().parents[3] / "samples" / "dev-samples" / "synthetic_case_route.json"
)


@pytest.mark.skipif(not CASE_ROUTE_JSON.exists(), reason="synthetic_case_route.json missing")
def test_case_route_zero_is_noop(tmp_path):
    out = tmp_path / "p.json"
    manifest = build_case_route_params(CASE_ROUTE_JSON, n_gateways_ruled=0, out_json_path=out)
    assert manifest["n_gateways_ruled"] == 0
    assert manifest["ruled_gateways"] == []
    data = json.loads(out.read_text())
    # No path anywhere is bound to a condition; all keep static probabilities.
    for e in data["gateway_branching_probabilities"]:
        for p in e["probabilities"]:
            assert "condition_id" not in p
            assert "value" in p


@pytest.mark.skipif(not CASE_ROUTE_JSON.exists(), reason="synthetic_case_route.json missing")
def test_case_route_rules_first_k_splits(tmp_path):
    out = tmp_path / "p.json"
    manifest = build_case_route_params(CASE_ROUTE_JSON, n_gateways_ruled=2, out_json_path=out)
    assert manifest["n_gateways_ruled"] == 2
    assert manifest["ruled_gateways"] == ["g1s", "g2s"]   # ordered, first two splits
    assert manifest["n_split_gateways"] == 3

    data = json.loads(out.read_text())
    by_id = {e["gateway_id"]: e for e in data["gateway_branching_probabilities"]}
    # Ruled gateways: _a -> rule_red, _b -> rule_blue, no static value left.
    for gid in ("g1s", "g2s"):
        for p in by_id[gid]["probabilities"]:
            assert "value" not in p
            assert p["condition_id"] == ("rule_red" if p["path_id"].endswith("_a") else "rule_blue")
    # Untouched split keeps static probabilities.
    for p in by_id["g3s"]["probabilities"]:
        assert "condition_id" not in p and "value" in p
    # Join gateways (single outgoing) are never ruled.
    assert all("condition_id" not in p
               for e in data["gateway_branching_probabilities"]
               for p in e["probabilities"] if len(e["probabilities"]) == 1)


@pytest.mark.skipif(not CASE_ROUTE_JSON.exists(), reason="synthetic_case_route.json missing")
def test_case_route_too_many_raises(tmp_path):
    with pytest.raises(ValueError, match="only 3 split"):
        build_case_route_params(CASE_ROUTE_JSON, n_gateways_ruled=99, out_json_path=tmp_path / "p.json")


def test_case_route_missing_branch_rules_raises(tmp_path):
    base = tmp_path / "base.json"
    base.write_text(json.dumps({
        "gateway_branching_probabilities": [
            {"gateway_id": "g1s", "probabilities": [
                {"path_id": "f_g1s_a", "value": "0.5"}, {"path_id": "f_g1s_b", "value": "0.5"}]},
        ],
        "branch_rules": [],   # no rule_red / rule_blue
    }))
    with pytest.raises(ValueError, match="missing branch rule"):
        build_case_route_params(base, n_gateways_ruled=1, out_json_path=tmp_path / "p.json")


@pytest.mark.skipif(not XOR_JSON.exists(), reason="synthetic_xor_loop.json missing")
def test_perturb_drops_one_resource(tmp_path):
    out = tmp_path / "perturbed.json"
    manifest = build_perturbed_params(
        XOR_JSON, remove_from_profile="Good", n_to_remove=1, out_json_path=out
    )

    assert manifest["profile"] == "Good"
    assert len(manifest["removed"]) == 1
    assert manifest["remaining"] == 2

    with open(out) as f:
        data = json.load(f)
    good = next(p for p in data["resource_profiles"] if p["name"] == "Good")
    evil = next(p for p in data["resource_profiles"] if p["name"] == "Evil")
    assert len(good["resource_list"]) == 2
    assert len(evil["resource_list"]) == 2  # untouched


@pytest.mark.skipif(not XOR_JSON.exists(), reason="synthetic_xor_loop.json missing")
def test_perturb_prunes_task_resource_distribution(tmp_path):
    out = tmp_path / "perturbed.json"
    # Record the ids of the resources that will be dropped.
    with open(XOR_JSON) as f:
        src = json.load(f)
    good = next(p for p in src["resource_profiles"] if p["name"] == "Good")
    dropped_ids = {r["id"] for r in good["resource_list"][:2]}

    build_perturbed_params(
        XOR_JSON, remove_from_profile="Good", n_to_remove=2, out_json_path=out
    )

    with open(out) as f:
        data = json.load(f)
    for task in data["task_resource_distribution"]:
        for r in task["resources"]:
            assert r["resource_id"] not in dropped_ids


@pytest.mark.skipif(not XOR_JSON.exists(), reason="synthetic_xor_loop.json missing")
def test_perturb_refuses_to_empty_profile(tmp_path):
    with pytest.raises(ValueError):
        build_perturbed_params(
            XOR_JSON, remove_from_profile="Good", n_to_remove=3,
            out_json_path=tmp_path / "p.json",
        )


@pytest.mark.skipif(not XOR_JSON.exists(), reason="synthetic_xor_loop.json missing")
def test_perturb_unknown_profile(tmp_path):
    with pytest.raises(ValueError):
        build_perturbed_params(
            XOR_JSON, remove_from_profile="Nope", n_to_remove=1,
            out_json_path=tmp_path / "p.json",
        )


@pytest.mark.skipif(not XOR_JSON.exists(), reason="synthetic_xor_loop.json missing")
def test_perturb_zero_is_noop(tmp_path):
    out = tmp_path / "perturbed.json"
    manifest = build_perturbed_params(
        XOR_JSON, remove_from_profile="Good", n_to_remove=0, out_json_path=out
    )
    assert manifest["removed"] == []
    assert manifest["added"] == []
    with open(XOR_JSON) as f:
        src = json.load(f)
    with open(out) as f:
        data = json.load(f)
    # Resource counts and task assignments must be unchanged.
    src_good = next(p for p in src["resource_profiles"] if p["name"] == "Good")
    out_good = next(p for p in data["resource_profiles"] if p["name"] == "Good")
    assert {r["id"] for r in src_good["resource_list"]} == {
        r["id"] for r in out_good["resource_list"]
    }


@pytest.mark.skipif(not XOR_JSON.exists(), reason="synthetic_xor_loop.json missing")
def test_perturb_adds_resources_when_negative(tmp_path):
    out = tmp_path / "perturbed.json"
    manifest = build_perturbed_params(
        XOR_JSON, remove_from_profile="Good", n_to_remove=-2, out_json_path=out
    )
    assert manifest["removed"] == []
    assert len(manifest["added"]) == 2
    assert manifest["remaining"] == 5  # 3 original + 2 added

    with open(out) as f:
        data = json.load(f)
    good = next(p for p in data["resource_profiles"] if p["name"] == "Good")
    assert len(good["resource_list"]) == 5
    new_ids = set(manifest["added"])
    existing_ids = {r["id"] for r in good["resource_list"]}
    assert new_ids.issubset(existing_ids)

    # Every task that referenced the template resource must now also list
    # the new clones with the same distribution.
    with open(XOR_JSON) as f:
        src = json.load(f)
    src_good = next(p for p in src["resource_profiles"] if p["name"] == "Good")
    template_id = src_good["resource_list"][-1]["id"]
    for task in data["task_resource_distribution"]:
        template_entry = next(
            (r for r in task["resources"] if r["resource_id"] == template_id),
            None,
        )
        if template_entry is None:
            continue
        for new_id in new_ids:
            clone = next(
                (r for r in task["resources"] if r["resource_id"] == new_id),
                None,
            )
            assert clone is not None
            assert clone["distribution_name"] == template_entry["distribution_name"]
            assert clone["distribution_params"] == template_entry["distribution_params"]


@pytest.mark.skipif(not XOR_JSON.exists(), reason="synthetic_xor_loop.json missing")
def test_perturb_added_resources_have_unique_ids(tmp_path):
    out1 = tmp_path / "p1.json"
    out2 = tmp_path / "p2.json"
    m1 = build_perturbed_params(
        XOR_JSON, remove_from_profile="Good", n_to_remove=-1, out_json_path=out1
    )
    m2 = build_perturbed_params(
        XOR_JSON, remove_from_profile="Good", n_to_remove=-3, out_json_path=out2
    )
    # All new IDs in both runs should be unique relative to the source resources.
    with open(XOR_JSON) as f:
        src = json.load(f)
    src_ids = {r["id"]
               for p in src["resource_profiles"]
               for r in p["resource_list"]}
    for new in m1["added"] + m2["added"]:
        assert new not in src_ids


# ---------------------------------------------------------------------------
# role_swap
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not LOAN_JSON.exists(), reason="Loan-stable.json missing")
def test_role_swap_zero_is_noop(tmp_path):
    out = tmp_path / "p.json"
    m = build_role_swap_params(
        LOAN_JSON, from_profile="Role 2_profile", to_profile="Role 3_profile",
        n_activities=0, out_json_path=out,
    )
    assert m["swapped_tasks"] == []
    with open(LOAN_JSON) as f:
        src = json.load(f)
    with open(out) as f:
        dst = json.load(f)
    assert src["task_resource_distribution"] == dst["task_resource_distribution"]


@pytest.mark.skipif(not LOAN_JSON.exists(), reason="Loan-stable.json missing")
def test_role_swap_moves_resources(tmp_path):
    out = tmp_path / "p.json"
    m = build_role_swap_params(
        LOAN_JSON, from_profile="Role 2_profile", to_profile="Role 3_profile",
        n_activities=1, out_json_path=out,
    )
    assert len(m["swapped_tasks"]) == 1

    with open(LOAN_JSON) as f:
        src = json.load(f)
    with open(out) as f:
        dst = json.load(f)
    role2_ids = {r["id"] for p in src["resource_profiles"]
                 if p["name"] == "Role 2_profile"
                 for r in p["resource_list"]}
    role3_ids = {r["id"] for p in src["resource_profiles"]
                 if p["name"] == "Role 3_profile"
                 for r in p["resource_list"]}
    swapped = m["swapped_tasks"][0]
    task = next(t for t in dst["task_resource_distribution"]
                if t["task_id"] == swapped)
    resource_ids = {r["resource_id"] for r in task["resources"]}
    # Role 2 resources removed, every Role 3 resource added.
    assert resource_ids.isdisjoint(role2_ids)
    assert role3_ids.issubset(resource_ids)


@pytest.mark.skipif(not LOAN_JSON.exists(), reason="Loan-stable.json missing")
def test_role_swap_rejects_same_profile(tmp_path):
    with pytest.raises(ValueError):
        build_role_swap_params(
            LOAN_JSON, from_profile="Role 2_profile",
            to_profile="Role 2_profile", n_activities=1,
            out_json_path=tmp_path / "p.json",
        )


@pytest.mark.skipif(not LOAN_JSON.exists(), reason="Loan-stable.json missing")
def test_role_swap_unknown_profile(tmp_path):
    with pytest.raises(ValueError):
        build_role_swap_params(
            LOAN_JSON, from_profile="Nope", to_profile="Role 3_profile",
            n_activities=1, out_json_path=tmp_path / "p.json",
        )


# ---------------------------------------------------------------------------
# calendar_shift
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not LOAN_JSON.exists(), reason="Loan-stable.json missing")
def test_calendar_shift_zero_is_noop(tmp_path):
    out = tmp_path / "p.json"
    m = build_calendar_shifted_params(
        LOAN_JSON, profile_name="Role 2_profile",
        shift_hours=0, out_json_path=out,
    )
    assert m["original_periods"] == m["shifted_periods"]


@pytest.mark.skipif(not LOAN_JSON.exists(), reason="Loan-stable.json missing")
def test_calendar_shift_rejects_24h_calendar(tmp_path):
    # Loan-stable's calendars are 24/7 (00:00:00 - 23:59:59.999 every day).
    # A shift is information-less on them — refuse with a clear error.
    with pytest.raises(ValueError, match="sub-24h periods"):
        build_calendar_shifted_params(
            LOAN_JSON, profile_name="Role 2_profile",
            shift_hours=4, out_json_path=tmp_path / "p.json",
        )


def test_calendar_shift_moves_business_hours(tmp_path):
    # A realistic business-hours calendar: shift within the same day works.
    params = {
        "resource_calendars": [{
            "id": "X_calendar", "name": "X_calendar",
            "time_periods": [{
                "from": "MONDAY", "to": "MONDAY",
                "beginTime": "09:00:00", "endTime": "17:00:00",
            }],
        }],
    }
    src = tmp_path / "src.json"
    src.write_text(json.dumps(params))
    out = tmp_path / "dst.json"
    m = build_calendar_shifted_params(
        src, profile_name="X", shift_hours=2, out_json_path=out,
    )
    # 09:00 + 2h = 11:00; 17:00 + 2h = 19:00. Stays within the day.
    assert m["shifted_periods"][0]["beginTime"] == "11:00:00"
    assert m["shifted_periods"][0]["endTime"] == "19:00:00"


def test_calendar_shift_rejects_wraparound(tmp_path):
    # 09:00-17:00 shifted by +18h would cross midnight; not implemented yet.
    params = {
        "resource_calendars": [{
            "id": "X_calendar", "name": "X_calendar",
            "time_periods": [{
                "from": "MONDAY", "to": "MONDAY",
                "beginTime": "09:00:00", "endTime": "17:00:00",
            }],
        }],
    }
    src = tmp_path / "src.json"
    src.write_text(json.dumps(params))
    with pytest.raises(NotImplementedError, match="wrap midnight"):
        build_calendar_shifted_params(
            src, profile_name="X", shift_hours=18,
            out_json_path=tmp_path / "dst.json",
        )


# ---------------------------------------------------------------------------
# gateway_bias
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not LOAN_JSON.exists(), reason="Loan-stable.json missing")
def test_gateway_zero_is_noop(tmp_path):
    out = tmp_path / "p.json"
    m = build_gateway_biased_params(
        LOAN_JSON, gateway_id=None, bias_level=0, out_json_path=out,
    )
    assert m["original_probs"] == m["new_probs"]


@pytest.mark.skipif(not LOAN_JSON.exists(), reason="Loan-stable.json missing")
def test_gateway_picks_most_balanced(tmp_path):
    out = tmp_path / "p.json"
    m = build_gateway_biased_params(
        LOAN_JSON, gateway_id=None, bias_level=1, out_json_path=out,
    )
    # The 0.5064/0.4936 or 0.5045/0.4955 gateways are most balanced in
    # Loan-stable; the picked one should have a majority close to 0.5.
    majority_before = max(p["value"] for p in m["original_probs"])
    assert majority_before <= 0.55
    # After +0.1 bias, majority increases by ~0.1.
    majority_after = max(p["value"] for p in m["new_probs"])
    assert majority_after == pytest.approx(majority_before + 0.1, abs=1e-6)
    # Probabilities still sum to 1.
    assert sum(p["value"] for p in m["new_probs"]) == pytest.approx(1.0)


@pytest.mark.skipif(not LOAN_JSON.exists(), reason="Loan-stable.json missing")
def test_gateway_explicit_id(tmp_path):
    with open(LOAN_JSON) as f:
        src = json.load(f)
    target_gw = src["gateway_branching_probabilities"][0]["gateway_id"]
    out = tmp_path / "p.json"
    m = build_gateway_biased_params(
        LOAN_JSON, gateway_id=target_gw, bias_level=2, out_json_path=out,
    )
    assert m["gateway_id"] == target_gw


@pytest.mark.skipif(not LOAN_JSON.exists(), reason="Loan-stable.json missing")
def test_gateway_unknown_id_raises(tmp_path):
    with pytest.raises(ValueError):
        build_gateway_biased_params(
            LOAN_JSON, gateway_id="not-a-gateway", bias_level=1,
            out_json_path=tmp_path / "p.json",
        )


# ---------------------------------------------------------------------------
# arrival_burstiness
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not LOAN_JSON.exists(), reason="Loan-stable.json missing")
def test_arrival_burst_preserves_mean(tmp_path):
    out = tmp_path / "p.json"
    m = build_arrival_burstier_params(
        LOAN_JSON, cv2_multiplier=2.0, out_json_path=out,
    )
    assert m["mean_before"] == m["mean_after"]
    assert m["variance_after"] == pytest.approx(m["variance_before"] * 2.0)
    with open(out) as f:
        dst = json.load(f)
    assert dst["arrival_time_distribution"]["distribution_params"][1]["value"] == pytest.approx(
        m["variance_after"]
    )


@pytest.mark.skipif(not LOAN_JSON.exists(), reason="Loan-stable.json missing")
def test_arrival_burst_unity_is_noop(tmp_path):
    out = tmp_path / "p.json"
    m = build_arrival_burstier_params(
        LOAN_JSON, cv2_multiplier=1.0, out_json_path=out,
    )
    assert m["variance_before"] == m["variance_after"]


def test_arrival_burst_rejects_non_gamma(tmp_path):
    params = {"arrival_time_distribution": {
        "distribution_name": "expon",
        "distribution_params": [{"value": 100.0}, {"value": 50.0}],
    }}
    src = tmp_path / "src.json"
    src.write_text(json.dumps(params))
    with pytest.raises(NotImplementedError):
        build_arrival_burstier_params(
            src, cv2_multiplier=2.0, out_json_path=tmp_path / "dst.json",
        )
