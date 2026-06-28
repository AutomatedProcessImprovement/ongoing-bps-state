"""Generate the synthetic *case-route* dataset (BPMN + Prosimos params).

The process is a chain of ``n_gateways`` exclusive split/join gateways, each
separated by a common ("triage") activity so that no two branch activities are
ever adjacent in the activity stream:

    Start -> Register
          -> [G1 split -> {Assess_A1 | Assess_B1} -> G1 join] -> Triage_1
          -> [G2 split -> {Assess_A2 | Assess_B2} -> G2 join] -> Triage_2
          -> ...
          -> Decide -> End

Both branches of every gateway have IDENTICAL duration distributions and run on
the SAME resource pool, so a case's cycle time does not depend on which branch
it takes. A ``case_type`` ∈ {red, blue} case attribute is declared (50/50). In
the *base* params produced here the gateways route 50/50 by static probability,
so case_type is independent of the path taken. The state-metrics ``case_route``
perturbation (see ``evaluation/state_metrics/perturb.build_case_route_params``)
turns the first K gateways into branch-rule gateways that send red -> A-branch
and blue -> B-branch, correlating case_type with path while keeping every
gateway's path marginal at 50/50.

Because the branch activities are duration-symmetric and never adjacent (the
common triage activity breaks the bigram chain), this correlation is invisible
to cycle_time and to the 2-gram distribution (ngd_n2): only the joint
(activity, case_type) projection — ``activity_type`` — detects it.

Run::

    python -m tools.generate_case_route          # writes the default 3-gateway asset
    python -m tools.generate_case_route --gateways 3 --out-dir samples/dev-samples
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

BPMN_NS = "http://www.omg.org/spec/BPMN/20100524/MODEL"


def _task(tid: str, name: str, incoming: str, outgoing: str) -> str:
    return (
        f'<bpmn:task id="{tid}" name="{escape(name)}">'
        f"<bpmn:incoming>{incoming}</bpmn:incoming>"
        f"<bpmn:outgoing>{outgoing}</bpmn:outgoing>"
        f"</bpmn:task>"
    )


def _xor(gid: str, name: str, incoming: list[str], outgoing: list[str]) -> str:
    inc = "".join(f"<bpmn:incoming>{f}</bpmn:incoming>" for f in incoming)
    out = "".join(f"<bpmn:outgoing>{f}</bpmn:outgoing>" for f in outgoing)
    return f'<bpmn:exclusiveGateway id="{gid}" name="{escape(name)}">{inc}{out}</bpmn:exclusiveGateway>'


def _flow(fid: str, src: str, tgt: str) -> str:
    return f'<bpmn:sequenceFlow id="{fid}" sourceRef="{src}" targetRef="{tgt}" />'


def build_model(
    n_gateways: int,
    *,
    workers: int = 48,
    arrival_mean: float = 60.0,
    service: float = 300.0,
) -> tuple[str, dict]:
    """Return (bpmn_xml, prosimos_params) for an ``n_gateways``-gateway chain.

    ``workers`` / ``arrival_mean`` / ``service`` set the load. The defaults put
    ~35 cases in flight at ~75% utilization (work per case ≈ (2*n_gateways+1) *
    service, offered load = work / arrival_mean), so the evaluation window has a
    dense Scope-A active-instance set rather than a handful of cases.
    """
    if n_gateways < 1:
        raise ValueError("n_gateways must be >= 1")

    nodes: list[str] = []
    flows: list[str] = []
    # Activity ids -> names, and the gateway branching spec for the JSON.
    task_ids: list[str] = []
    gateway_specs: list[dict] = []  # one entry per gateway needing probabilities

    nodes.append('<bpmn:startEvent id="StartEvent_1" name="Start"><bpmn:outgoing>f_start</bpmn:outgoing></bpmn:startEvent>')

    # Register (common entry activity).
    nodes.append(_task("t_reg", "Register", "f_start", "f_reg_g1s"))
    task_ids.append("t_reg")
    flows.append(_flow("f_start", "StartEvent_1", "t_reg"))

    prev_out = "f_reg_g1s"          # flow feeding the next split gateway
    prev_src = "t_reg"
    for k in range(1, n_gateways + 1):
        g_split = f"g{k}s"
        g_join = f"g{k}j"
        t_a, t_b = f"t_a{k}", f"t_b{k}"
        f_in = prev_out
        f_sa, f_sb = f"f_g{k}s_a", f"f_g{k}s_b"
        f_aj, f_bj = f"f_a{k}_g{k}j", f"f_b{k}_g{k}j"
        # The flow leaving the join: to a common triage activity, except the
        # last gateway whose join feeds the final Decide activity.
        is_last = k == n_gateways
        common_id = "t_dec" if is_last else f"t_tri{k}"
        common_name = "Decide" if is_last else f"Triage_{k}"
        f_join_out = f"f_g{k}j_{common_id}"

        flows.append(_flow(f_in, prev_src, g_split))
        nodes.append(_xor(g_split, f"XOR-split-{k}", [f_in], [f_sa, f_sb]))
        nodes.append(_task(t_a, f"Assess_A{k}", f_sa, f_aj))
        nodes.append(_task(t_b, f"Assess_B{k}", f_sb, f_bj))
        flows.append(_flow(f_sa, g_split, t_a))
        flows.append(_flow(f_sb, g_split, t_b))
        flows.append(_flow(f_aj, t_a, g_join))
        flows.append(_flow(f_bj, t_b, g_join))
        nodes.append(_xor(g_join, f"XOR-join-{k}", [f_aj, f_bj], [f_join_out]))
        task_ids.extend([t_a, t_b])

        # Common activity after the join.
        next_out = "f_dec_end" if is_last else f"f_{common_id}_g{k + 1}s"
        nodes.append(_task(common_id, common_name, f_join_out, next_out))
        flows.append(_flow(f_join_out, g_join, common_id))
        task_ids.append(common_id)

        gateway_specs.append({
            "gateway_id": g_split,
            "split": True,
            "paths": [f_sa, f_sb],     # [A-branch, B-branch]
        })
        gateway_specs.append({
            "gateway_id": g_join,
            "split": False,
            "paths": [f_join_out],
        })

        prev_out = next_out
        prev_src = common_id

    # prev_src is t_dec, prev_out is f_dec_end.
    nodes.append('<bpmn:endEvent id="EndEvent_1" name="End"><bpmn:incoming>f_dec_end</bpmn:incoming></bpmn:endEvent>')
    flows.append(_flow("f_dec_end", "t_dec", "EndEvent_1"))

    process_body = "".join(nodes) + "".join(flows)
    bpmn = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<bpmn:definitions xmlns:bpmn="{BPMN_NS}" id="Definitions_case_route" '
        'targetNamespace="http://bpmn.io/schema/bpmn">'
        '<bpmn:process id="Process_case_route" isExecutable="false">'
        f"{process_body}"
        "</bpmn:process></bpmn:definitions>"
    )

    params = _build_params(task_ids, gateway_specs,
                           workers=workers, arrival_mean=arrival_mean, service=service)
    return bpmn, params


def _build_params(
    task_ids: list[str],
    gateway_specs: list[dict],
    *,
    workers: int,
    arrival_mean: float,
    service: float,
) -> dict:
    cal_id = "cal_247"
    periods = [{"from": "MONDAY", "to": "SUNDAY",
                "beginTime": "00:00:00", "endTime": "23:59:59.999"}]
    resource_ids = [f"Worker_{i}" for i in range(1, workers + 1)]

    def fix_service(rid: str) -> dict:
        return {
            "distribution_name": "fix",
            "distribution_params": [{"value": service}],
            "resource_id": rid,
        }

    task_resource_distribution = [
        {"task_id": tid, "resources": [fix_service(r) for r in resource_ids]}
        for tid in task_ids
    ]

    resource_profiles = [{
        "id": "Worker_profile",
        "name": "Worker",
        "resource_list": [
            {"id": rid, "name": rid, "cost_per_hour": 0, "amount": 1,
             "calendar": cal_id, "assignedTasks": list(task_ids)}
            for rid in resource_ids
        ],
    }]

    gateway_branching_probabilities = []
    for spec in gateway_specs:
        if spec["split"]:
            a, b = spec["paths"]
            probs = [{"path_id": a, "value": "0.5"}, {"path_id": b, "value": "0.5"}]
        else:
            probs = [{"path_id": spec["paths"][0], "value": "1"}]
        gateway_branching_probabilities.append(
            {"gateway_id": spec["gateway_id"], "probabilities": probs}
        )

    return {
        "task_resource_distribution": task_resource_distribution,
        "resource_calendars": [{"id": cal_id, "name": "24/7", "time_periods": periods}],
        "gateway_branching_probabilities": gateway_branching_probabilities,
        # Poisson-ish arrivals so WIP varies enough for the p90 cutoff to bite.
        # expon params are [mean, min, max] in Prosimos/pix-framework.
        "arrival_time_distribution": {
            "distribution_name": "expon",
            "distribution_params": [{"value": arrival_mean}, {"value": 0}, {"value": 3600}],
        },
        "arrival_time_calendar": periods,
        "resource_profiles": resource_profiles,
        "event_distribution": [],
        "batch_processing": [],
        # case_type is DECLARED here (so it is sampled + logged) but the base
        # gateways above route by static probability, so case_type is
        # independent of the path. The case_route perturbation adds branch_rules.
        "case_attributes": [{
            "name": "case_type",
            "type": "discrete",
            "values": [{"key": "red", "value": 0.5}, {"key": "blue", "value": 0.5}],
        }],
        # Reusable equality rules the perturbation references per gateway.
        "branch_rules": [
            {"id": "rule_red", "rules": [[{"attribute": "case_type", "comparison": "=", "value": "red"}]]},
            {"id": "rule_blue", "rules": [[{"attribute": "case_type", "comparison": "=", "value": "blue"}]]},
        ],
        "prioritisation_rules": [],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the case-route synthetic dataset")
    ap.add_argument("--gateways", type=int, default=3)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--arrival-mean", type=float, default=60.0)
    ap.add_argument("--service", type=float, default=300.0)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parents[1] / "samples" / "dev-samples")
    ap.add_argument("--name", default="synthetic_case_route")
    args = ap.parse_args()

    bpmn, params = build_model(args.gateways, workers=args.workers,
                               arrival_mean=args.arrival_mean, service=args.service)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    bpmn_path = args.out_dir / f"{args.name}.bpmn"
    json_path = args.out_dir / f"{args.name}.json"
    bpmn_path.write_text(bpmn, encoding="utf-8")
    json_path.write_text(json.dumps(params, indent=2), encoding="utf-8")
    print(f"wrote {bpmn_path}")
    print(f"wrote {json_path}")
    print(f"  tasks={sum(1 for _ in params['task_resource_distribution'])} "
          f"gateways={args.gateways} "
          f"(split branch activities: "
          f"{[t for t in [r['task_id'] for r in params['task_resource_distribution']] if t.startswith(('t_a','t_b'))]})")


if __name__ == "__main__":
    main()
