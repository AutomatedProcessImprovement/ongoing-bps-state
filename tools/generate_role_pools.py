"""Generate the synthetic *role-migration* dataset (BPMN + Prosimos params).

Composition robustness instance C3 — the one that exercises the *role*
projection. A simple sequential chain::

    Start -> Register -> S1 -> S2 -> ... -> SN -> Decide -> End

Every step activity ``S1..SN`` is served, in the base (correct) model, by a
single role pool ``RoleA``. A second pool ``RoleB`` of the SAME size, calendar
and per-task duration distributions exists and is idle. Both pools are sized
generously so contention is negligible.

The state-metrics ``role_swap`` perturbation
(``evaluation.state_metrics.perturb.build_role_swap_params``) migrates the first
``level`` step activities from ``RoleA`` to ``RoleB`` — the same work, same
durations, a different (equally capable) pool. Consequently:

* cycle time is unchanged (identical duration distribution on an equal pool),
* the activity labels, per-case activity sequence and 2-grams are unchanged
  (``ngd_n2`` / ``red`` / ``rtd`` / ``cycle_time`` all blind),
* and the case- and activity-only state projections are unchanged too.

Only the ``activity_role`` projection — which maps each active instance's
resource to its role — sees that an activity is now carried by the wrong role.
This isolates the role dimension: it is the single state projection that fires.

Run::

    python -m tools.generate_role_pools --steps 5
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


def _flow(fid: str, src: str, tgt: str) -> str:
    return f'<bpmn:sequenceFlow id="{fid}" sourceRef="{src}" targetRef="{tgt}" />'


def _norm(mean: float) -> dict:
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
    *,
    n_steps: int = 5,
    workers_per_role: int = 48,
    arrival_mean: float = 60.0,
    step_service: float = 300.0,
    common_service: float = 180.0,
) -> tuple[str, dict, list[str]]:
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    nodes: list[str] = []
    flows: list[str] = []

    nodes.append('<bpmn:startEvent id="StartEvent_1" name="Start">'
                 '<bpmn:outgoing>f_start</bpmn:outgoing></bpmn:startEvent>')
    nodes.append(_task("t_reg", "Register", "f_start", "f_reg_s1"))
    flows.append(_flow("f_start", "StartEvent_1", "t_reg"))

    step_ids = [f"t_s{i}" for i in range(1, n_steps + 1)]
    prev_src = "t_reg"
    prev_out = "f_reg_s1"
    for i, tid in enumerate(step_ids, start=1):
        is_last = i == n_steps
        out_flow = "f_sN_dec" if is_last else f"f_s{i}_s{i + 1}"
        nodes.append(_task(tid, f"Step_{i}", prev_out, out_flow))
        flows.append(_flow(prev_out, prev_src, tid))
        prev_src = tid
        prev_out = out_flow

    nodes.append(_task("t_dec", "Decide", "f_sN_dec", "f_dec_end"))
    flows.append(_flow("f_sN_dec", prev_src, "t_dec"))
    nodes.append('<bpmn:endEvent id="EndEvent_1" name="End">'
                 '<bpmn:incoming>f_dec_end</bpmn:incoming></bpmn:endEvent>')
    flows.append(_flow("f_dec_end", "t_dec", "EndEvent_1"))

    process_body = "".join(nodes) + "".join(flows)
    bpmn = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<bpmn:definitions xmlns:bpmn="{BPMN_NS}" id="Definitions_role_pools" '
        'targetNamespace="http://bpmn.io/schema/bpmn">'
        '<bpmn:process id="Process_role_pools" isExecutable="false">'
        f"{process_body}"
        "</bpmn:process></bpmn:definitions>"
    )

    params = _build_params(
        step_ids, workers_per_role=workers_per_role, arrival_mean=arrival_mean,
        step_service=step_service, common_service=common_service,
    )
    return bpmn, params, step_ids


def _pool(name: str, prefix: str, n: int, cal_id: str, tasks: list[str]) -> dict:
    return {
        "id": f"{name}_profile",
        "name": name,
        "resource_list": [
            {"id": f"{prefix}_{i}", "name": f"{prefix}_{i}", "cost_per_hour": 0,
             "amount": 1, "calendar": cal_id, "assignedTasks": list(tasks)}
            for i in range(1, n + 1)
        ],
    }


def _build_params(
    step_ids: list[str],
    *,
    workers_per_role: int,
    arrival_mean: float,
    step_service: float,
    common_service: float,
) -> dict:
    cal_id = "cal_247"
    periods = [{"from": "MONDAY", "to": "SUNDAY",
                "beginTime": "00:00:00", "endTime": "23:59:59.999"}]
    common_tasks = ["t_reg", "t_dec"]

    # RoleA serves the step activities in the base model; RoleB is an equal,
    # idle pool the role_swap perturbation migrates work onto. A shared Common
    # pool handles Register/Decide so the step activities are the only ones
    # whose role can move.
    common_pool = _pool("Common", "Common", workers_per_role, cal_id, common_tasks)
    role_a = _pool("RoleA", "RoleA", workers_per_role, cal_id, list(step_ids))
    role_b = _pool("RoleB", "RoleB", workers_per_role, cal_id, list(step_ids))

    trd: list[dict] = []
    for tid in common_tasks:
        trd.append({"task_id": tid,
                    "resources": [{**_norm(common_service), "resource_id": r["id"]}
                                  for r in common_pool["resource_list"]]})
    for tid in step_ids:
        # Base model: step activities served ONLY by RoleA.
        trd.append({"task_id": tid,
                    "resources": [{**_norm(step_service), "resource_id": r["id"]}
                                  for r in role_a["resource_list"]]})

    return {
        "task_resource_distribution": trd,
        "resource_calendars": [{"id": cal_id, "name": "24/7", "time_periods": periods}],
        "gateway_branching_probabilities": [],
        "arrival_time_distribution": {
            "distribution_name": "expon",
            "distribution_params": [{"value": arrival_mean}, {"value": 0}, {"value": 3600}],
        },
        "arrival_time_calendar": periods,
        "resource_profiles": [common_pool, role_a, role_b],
        "event_distribution": [],
        "batch_processing": [],
        "case_attributes": [],
        "prioritisation_rules": [],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the role-migration synthetic dataset")
    ap.add_argument("--steps", type=int, default=5, help="number of step activities S1..SN")
    ap.add_argument("--workers-per-role", type=int, default=48)
    ap.add_argument("--arrival-mean", type=float, default=60.0)
    ap.add_argument("--step-service", type=float, default=300.0)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parents[1] / "samples" / "dev-samples")
    ap.add_argument("--name", default="synthetic_role_pools")
    args = ap.parse_args()

    bpmn, params, step_ids = build_model(
        n_steps=args.steps, workers_per_role=args.workers_per_role,
        arrival_mean=args.arrival_mean, step_service=args.step_service,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    bpmn_path = args.out_dir / f"{args.name}.bpmn"
    json_path = args.out_dir / f"{args.name}.json"
    bpmn_path.write_text(bpmn, encoding="utf-8")
    json_path.write_text(json.dumps(params, indent=2), encoding="utf-8")
    print(f"wrote {bpmn_path}")
    print(f"wrote {json_path}")
    print(f"  step activities (role_swap targets RoleA->RoleB): {step_ids}")


if __name__ == "__main__":
    main()
