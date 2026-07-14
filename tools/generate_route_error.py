"""Generate the synthetic *route-error* dataset (BPMN + Prosimos params).

Scenario #3 (the "model-error routing" experiment). A chain of ``n_gateways``
exclusive split/join gateways, each separated by a common ("triage") activity so
that no two branch activities are ever adjacent in the activity stream::

    Start -> Register
          -> [G1 split -> {A-branch (long) | B-branch (short)} -> G1 join] -> Triage_1
          -> [G2 split -> {A-branch (long) | B-branch (short)} -> G2 join] -> Triage_2
          -> ...
          -> Decide -> End

Unlike the ``case_route`` synthetic, each branch is a **multi-activity chain**
(A-branch: ``a_len`` activities, B-branch: ``b_len`` activities), so a wrong
routing sends a case down a whole wrong sub-path. The branches are
**structurally distinct but timing-symmetric** — same per-activity duration on
the same pool and, by default, the same length — so a case's total duration does
not depend on which branch it takes. The *base* params emitted here route by a
genuine ``case_type`` ∈ {red, blue} case attribute (50/50) via Prosimos
``branch_rules``: **red -> A-branch, blue -> B-branch** at every gateway. That is
the CORRECT model — the ground truth.

The state-metrics ``route_error`` perturbation
(``evaluation.state_metrics.perturb.build_route_error_params``) then *inverts*
the routing on the first ``level`` gateways (red -> B, blue -> A) and lets
Prosimos-short-term continue the ongoing cases with that WRONG model. Because the
population stays 50/50 red/blue, at every gateway 50 % of cases still take the
A-branch and 50 % the B-branch, so:

* aggregate cycle time is unchanged (the long/short mix is symmetric),
* the 2-gram distribution is unchanged (the common triage activity breaks the
  bigram between a branch choice and the next), so ``ngd_n2`` is blind,
* the aggregate relative-event-distribution is unchanged, so ``red`` is blind.

Only pairing each ongoing case to its own ground-truth continuation reveals that
a red case that *should* be on the long A-branch is now on the short B-branch:
the ``activity_case`` (case_id, activity) projection detects it, as does the
attribute-aware ``activity_type`` (activity, case_type) projection.

Run::

    python -m tools.generate_route_error          # default 3-gateway asset
    python -m tools.generate_route_error --gateways 3 --a-len 4 --out-dir samples/dev-samples
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


def _norm(mean: float) -> dict:
    """A *narrow* normal duration distribution (Prosimos ``norm`` order:
    ``[mean, stddev, min, max]``). Std = 7 % of the mean, clamped to a
    non-negative floor — deliberately tight ("not very wide") so two same-model
    sims diverge little: this keeps the per-instant state metric's level-0 noise
    floor low, so the (structurally large) per-case path change from a routing
    inversion stands out against it."""
    std = 0.07 * mean
    return {
        "distribution_name": "norm",
        "distribution_params": [
            {"value": mean},
            {"value": std},
            {"value": max(1.0, mean - 2 * std)},
            {"value": mean + 3 * std},
        ],
    }


def build_model(
    n_gateways: int,
    *,
    a_len: int = 3,
    b_len: int | None = None,
    workers: int = 64,
    arrival_mean: float = 60.0,
    branch_service: float = 300.0,
    common_service: float = 180.0,
) -> tuple[str, dict]:
    """Return (bpmn_xml, prosimos_params) for an ``n_gateways``-gateway chain.

    Each gateway has two structurally-distinct branches: an A-branch chain of
    ``a_len`` activities and a B-branch chain of ``b_len`` activities (default
    ``b_len == a_len``). Both use the SAME per-activity duration ``branch_service``
    on the SAME pool, so with equal lengths a case's total duration is
    branch-independent — inverting the routing is cycle-time-/rtd-/red-neutral by
    construction and only the per-case PATH changes. ``workers`` / ``arrival_mean``
    set the offered load; the defaults keep a dense Scope-A window at ~75 %
    utilization.
    """
    if b_len is None:
        b_len = a_len
    if n_gateways < 1:
        raise ValueError("n_gateways must be >= 1")
    if a_len < 1 or b_len < 1:
        raise ValueError("a_len and b_len must be >= 1")

    nodes: list[str] = []
    flows: list[str] = []
    # Activity id -> mean service, and the gateway branching spec for the JSON.
    task_means: dict[str, float] = {}
    gateway_specs: list[dict] = []

    nodes.append('<bpmn:startEvent id="StartEvent_1" name="Start"><bpmn:outgoing>f_start</bpmn:outgoing></bpmn:startEvent>')

    nodes.append(_task("t_reg", "Register", "f_start", "f_reg_g1s"))
    task_means["t_reg"] = common_service
    flows.append(_flow("f_start", "StartEvent_1", "t_reg"))

    prev_out = "f_reg_g1s"          # flow feeding the next split gateway
    prev_src = "t_reg"
    for k in range(1, n_gateways + 1):
        g_split = f"g{k}s"
        g_join = f"g{k}j"
        f_in = prev_out
        # Flow ids leaving the split keep the _a / _b suffix convention the
        # route builders rely on to identify the A / B branch.
        f_sa, f_sb = f"f_g{k}s_a", f"f_g{k}s_b"
        is_last = k == n_gateways
        common_id = "t_dec" if is_last else f"t_tri{k}"
        common_name = "Decide" if is_last else f"Triage_{k}"
        f_join_out = f"f_g{k}j_{common_id}"

        flows.append(_flow(f_in, prev_src, g_split))
        nodes.append(_xor(g_split, f"XOR-split-{k}", [f_in], [f_sa, f_sb]))

        def _branch(side: str, length: int, split_flow: str) -> str:
            """Emit a ``length``-activity chain for one branch; return the flow
            id that feeds the join. ``side`` is 'a' or 'b' (distinct labels)."""
            label = side.upper()
            ids = [f"t_{side}{k}_{i}" for i in range(1, length + 1)]
            f_join_in = f"f_{side}{k}_g{k}j"
            in_flows = [split_flow] + [f"f_{side}{k}_link{i}" for i in range(1, length)]
            out_flows = [f"f_{side}{k}_link{i}" for i in range(1, length)] + [f_join_in]
            for i, tid in enumerate(ids):
                nodes.append(_task(tid, f"Assess_{label}{k}_{i + 1}", in_flows[i], out_flows[i]))
                src = g_split if i == 0 else ids[i - 1]
                flows.append(_flow(in_flows[i], src, tid))
                task_means[tid] = branch_service
            flows.append(_flow(f_join_in, ids[-1], g_join))
            return f_join_in

        # Two structurally-distinct but TIMING-SYMMETRIC branches: same number
        # of activities and same duration distribution, only different labels.
        # A case's total duration is therefore branch-independent, so inverting
        # the routing is cycle-time-/rtd-/red-neutral by construction — only the
        # per-case PATH identity changes (which pairing by case_id detects).
        f_aj = _branch("a", a_len, f_sa)
        f_bj = _branch("b", b_len, f_sb)

        nodes.append(_xor(g_join, f"XOR-join-{k}", [f_aj, f_bj], [f_join_out]))

        # Common activity after the join.
        next_out = "f_dec_end" if is_last else f"f_{common_id}_g{k + 1}s"
        nodes.append(_task(common_id, common_name, f_join_out, next_out))
        flows.append(_flow(f_join_out, g_join, common_id))
        task_means[common_id] = common_service

        gateway_specs.append({
            "gateway_id": g_split,
            "split": True,
            "paths": [f_sa, f_sb],     # [A-branch (red), B-branch (blue)]
        })
        gateway_specs.append({
            "gateway_id": g_join,
            "split": False,
            "paths": [f_join_out],
        })

        prev_out = next_out
        prev_src = common_id

    nodes.append('<bpmn:endEvent id="EndEvent_1" name="End"><bpmn:incoming>f_dec_end</bpmn:incoming></bpmn:endEvent>')
    flows.append(_flow("f_dec_end", "t_dec", "EndEvent_1"))

    process_body = "".join(nodes) + "".join(flows)
    bpmn = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<bpmn:definitions xmlns:bpmn="{BPMN_NS}" id="Definitions_route_error" '
        'targetNamespace="http://bpmn.io/schema/bpmn">'
        '<bpmn:process id="Process_route_error" isExecutable="false">'
        f"{process_body}"
        "</bpmn:process></bpmn:definitions>"
    )

    params = _build_params(task_means, gateway_specs,
                           workers=workers, arrival_mean=arrival_mean)
    return bpmn, params


def _build_params(
    task_means: dict[str, float],
    gateway_specs: list[dict],
    *,
    workers: int,
    arrival_mean: float,
) -> dict:
    cal_id = "cal_247"
    periods = [{"from": "MONDAY", "to": "SUNDAY",
                "beginTime": "00:00:00", "endTime": "23:59:59.999"}]
    resource_ids = [f"Worker_{i}" for i in range(1, workers + 1)]
    task_ids = list(task_means.keys())

    task_resource_distribution = [
        {
            "task_id": tid,
            "resources": [
                {**_norm(task_means[tid]), "resource_id": rid}
                for rid in resource_ids
            ],
        }
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

    # CORRECT routing baked in: split gateways route by case_type via condition.
    #   _a branch (long)  -> rule_red   (red cases)
    #   _b branch (short) -> rule_blue  (blue cases)
    gateway_branching_probabilities = []
    for spec in gateway_specs:
        if spec["split"]:
            a, b = spec["paths"]
            probs = [
                {"path_id": a, "condition_id": "rule_red"},
                {"path_id": b, "condition_id": "rule_blue"},
            ]
        else:
            probs = [{"path_id": spec["paths"][0], "value": "1"}]
        gateway_branching_probabilities.append(
            {"gateway_id": spec["gateway_id"], "probabilities": probs}
        )

    return {
        "task_resource_distribution": task_resource_distribution,
        "resource_calendars": [{"id": cal_id, "name": "24/7", "time_periods": periods}],
        "gateway_branching_probabilities": gateway_branching_probabilities,
        "arrival_time_distribution": {
            "distribution_name": "expon",
            "distribution_params": [{"value": arrival_mean}, {"value": 0}, {"value": 3600}],
        },
        "arrival_time_calendar": periods,
        "resource_profiles": resource_profiles,
        "event_distribution": [],
        "batch_processing": [],
        "case_attributes": [{
            "name": "case_type",
            "type": "discrete",
            "values": [{"key": "red", "value": 0.5}, {"key": "blue", "value": 0.5}],
        }],
        "branch_rules": [
            {"id": "rule_red", "rules": [[{"attribute": "case_type", "comparison": "=", "value": "red"}]]},
            {"id": "rule_blue", "rules": [[{"attribute": "case_type", "comparison": "=", "value": "blue"}]]},
        ],
        "prioritisation_rules": [],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the route-error synthetic dataset")
    ap.add_argument("--gateways", type=int, default=3)
    ap.add_argument("--a-len", type=int, default=3,
                    help="number of activities in the A-branch")
    ap.add_argument("--b-len", type=int, default=None,
                    help="number of activities in the B-branch (default: = a-len, "
                         "so the branches are timing-symmetric)")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--arrival-mean", type=float, default=60.0)
    ap.add_argument("--branch-service", type=float, default=300.0)
    ap.add_argument("--common-service", type=float, default=180.0)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parents[1] / "samples" / "dev-samples")
    ap.add_argument("--name", default="synthetic_route_error")
    args = ap.parse_args()

    bpmn, params = build_model(
        args.gateways, a_len=args.a_len, b_len=args.b_len, workers=args.workers,
        arrival_mean=args.arrival_mean, branch_service=args.branch_service,
        common_service=args.common_service,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    bpmn_path = args.out_dir / f"{args.name}.bpmn"
    json_path = args.out_dir / f"{args.name}.json"
    bpmn_path.write_text(bpmn, encoding="utf-8")
    json_path.write_text(json.dumps(params, indent=2), encoding="utf-8")
    tasks = [r["task_id"] for r in params["task_resource_distribution"]]
    print(f"wrote {bpmn_path}")
    print(f"wrote {json_path}")
    print(f"  tasks={len(tasks)} gateways={args.gateways} "
          f"a_len={args.a_len} b_len={args.b_len if args.b_len is not None else args.a_len}")
    print(f"  A-branch tasks: {[t for t in tasks if t.startswith('t_a')]}")
    print(f"  B-branch tasks: {[t for t in tasks if t.startswith('t_b')]}")


if __name__ == "__main__":
    main()
