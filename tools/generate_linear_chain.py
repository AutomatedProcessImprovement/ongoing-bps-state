"""Generate the synthetic *linear-chain* dataset (BPMN + Prosimos params).

A purely sequential process of ``n_tasks`` comparable-duration activities on a
single resource pool:

    Start -> Step_1 -> Step_2 -> ... -> Step_N -> End

No gateways, no branching: every case runs the same activity sequence with the
same per-task service distribution, so the baseline is symmetric and the only
thing a perturbation can move is *where in the case* duration mass sits.

The state-metrics ``front_back_load`` perturbation
(``evaluation.state_metrics.perturb.build_front_back_load_params``) reweights
the per-task means by chain position while holding the per-case **total**
duration constant: front-loading makes early steps longer and late steps
shorter (back-loading is the mirror). Because the per-case total (cycle time),
the activity set, the bigrams, and the aggregate utilisation are all unchanged,
``cycle_time`` and ``ngd_n2`` are blind; only the time-weighted state metric
localises the moved duration mass — the timeline-localisation story.

Run::

    python -m tools.generate_linear_chain
    python -m tools.generate_linear_chain --tasks 5 --out-dir samples/dev-samples
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


def build_model(
    n_tasks: int,
    *,
    workers: int = 40,
    arrival_mean: float = 90.0,
    service: float = 180.0,
) -> tuple[str, dict, list[str]]:
    """Return (bpmn_xml, prosimos_params, chain_task_ids).

    All ``n_tasks`` steps share the same ``service`` mean, so the unperturbed
    chain is duration-symmetric — front/back-loading is a pure redistribution.
    The pool is sized so contention is minimal (the win is timeline, not
    throughput).
    """
    if n_tasks < 2:
        raise ValueError("n_tasks must be >= 2")

    nodes: list[str] = []
    flows: list[str] = []
    task_ids: list[str] = []

    nodes.append('<bpmn:startEvent id="StartEvent_1" name="Start">'
                 '<bpmn:outgoing>f_start</bpmn:outgoing></bpmn:startEvent>')
    prev_out = "f_start"
    prev_src = "StartEvent_1"
    for i in range(1, n_tasks + 1):
        tid = f"t_step{i}"
        task_ids.append(tid)
        out_flow = "f_end" if i == n_tasks else f"f_step{i}"
        nodes.append(_task(tid, f"Step_{i}", prev_out, out_flow))
        flows.append(_flow(prev_out, prev_src, tid))
        prev_out = out_flow
        prev_src = tid
    nodes.append('<bpmn:endEvent id="EndEvent_1" name="End">'
                 '<bpmn:incoming>f_end</bpmn:incoming></bpmn:endEvent>')
    flows.append(_flow("f_end", prev_src, "EndEvent_1"))

    process_body = "".join(nodes) + "".join(flows)
    bpmn = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<bpmn:definitions xmlns:bpmn="{BPMN_NS}" id="Definitions_linear_chain" '
        'targetNamespace="http://bpmn.io/schema/bpmn">'
        '<bpmn:process id="Process_linear_chain" isExecutable="false">'
        f"{process_body}"
        "</bpmn:process></bpmn:definitions>"
    )

    params = _build_params(task_ids, workers=workers,
                           arrival_mean=arrival_mean, service=service)
    return bpmn, params, task_ids


def _build_params(
    task_ids: list[str], *, workers: int, arrival_mean: float, service: float,
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

    trd = [
        {"task_id": tid, "resources": [fix_service(r) for r in resource_ids]}
        for tid in task_ids
    ]
    resource_profiles = [{
        "id": "Worker_profile", "name": "Worker",
        "resource_list": [
            {"id": rid, "name": rid, "cost_per_hour": 0, "amount": 1,
             "calendar": cal_id, "assignedTasks": list(task_ids)}
            for rid in resource_ids
        ],
    }]

    return {
        "task_resource_distribution": trd,
        "resource_calendars": [{"id": cal_id, "name": "24/7", "time_periods": periods}],
        "gateway_branching_probabilities": [],
        "arrival_time_distribution": {
            "distribution_name": "expon",
            "distribution_params": [{"value": arrival_mean}, {"value": 0}, {"value": 3600}],
        },
        "arrival_time_calendar": periods,
        "resource_profiles": resource_profiles,
        "event_distribution": [],
        "batch_processing": [],
        "case_attributes": [],
        "prioritisation_rules": [],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the linear-chain synthetic dataset")
    ap.add_argument("--tasks", type=int, default=5)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--arrival-mean", type=float, default=90.0)
    ap.add_argument("--service", type=float, default=180.0)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parents[1] / "samples" / "dev-samples")
    ap.add_argument("--name", default="synthetic_linear_chain")
    args = ap.parse_args()

    bpmn, params, ids = build_model(args.tasks, workers=args.workers,
                                    arrival_mean=args.arrival_mean, service=args.service)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    bpmn_path = args.out_dir / f"{args.name}.bpmn"
    json_path = args.out_dir / f"{args.name}.json"
    bpmn_path.write_text(bpmn, encoding="utf-8")
    json_path.write_text(json.dumps(params, indent=2), encoding="utf-8")
    print(f"wrote {bpmn_path}")
    print(f"wrote {json_path}")
    print(f"  chain tasks (ordered): {ids}")


if __name__ == "__main__":
    main()
