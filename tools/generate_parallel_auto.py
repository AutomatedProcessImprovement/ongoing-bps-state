"""Generate the synthetic *parallel-automation* dataset (BPMN + Prosimos params).

The process has one AND (parallel) block with a *critical* and a *non-critical*
branch running concurrently:

    Start -> Register
          -> AND-split
                -> Critical                     (long; sets the cycle time)
                -> NonCritical_1 -> ... -> NonCritical_M   (short chain)
          -> AND-join
          -> Decide -> End

The critical branch's duration strictly dominates the non-critical chain
(``D_crit`` >> ``Σ D_nc``) and the two branches run on SEPARATE resource pools.
Consequently:

* the per-case cycle time is set by the critical branch, so shrinking the
  non-critical work leaves cycle time unchanged;
* automating the non-critical branch does not touch the critical pool, so the
  critical resource utilisation is unchanged.

The state-metrics ``parallel_auto`` perturbation
(``evaluation.state_metrics.perturb.build_branch_automation_params``) scales the
non-critical task durations toward a small floor. Because the non-critical
activities still appear once per case (the floor keeps them in the trace) but
occupy the active-instance set for progressively less time, the per-instant
active multiset shifts while:

* ``cycle_time`` stays blind (critical path unchanged), and
* ``ngd_n2`` stays blind (same activities, same bigrams, same per-case counts).

Only the time-weighted state metric sees the non-critical branch vanishing from
the concurrent active set. This is Marlon's parallel-branch-automation example.

Run::

    python -m tools.generate_parallel_auto
    python -m tools.generate_parallel_auto --nc-tasks 3 --out-dir samples/dev-samples
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


def _parallel(gid: str, name: str, incoming: list[str], outgoing: list[str]) -> str:
    inc = "".join(f"<bpmn:incoming>{f}</bpmn:incoming>" for f in incoming)
    out = "".join(f"<bpmn:outgoing>{f}</bpmn:outgoing>" for f in outgoing)
    return f'<bpmn:parallelGateway id="{gid}" name="{escape(name)}">{inc}{out}</bpmn:parallelGateway>'


def _flow(fid: str, src: str, tgt: str) -> str:
    return f'<bpmn:sequenceFlow id="{fid}" sourceRef="{src}" targetRef="{tgt}" />'


def build_model(
    n_nc_tasks: int,
    *,
    common_workers: int = 12,
    critical_workers: int = 40,
    nc_workers: int = 40,
    arrival_mean: float = 30.0,
    common_service: float = 60.0,
    critical_service: float = 600.0,
    nc_service: float = 120.0,
) -> tuple[str, dict, list[str]]:
    """Return (bpmn_xml, prosimos_params, nc_task_ids).

    ``critical_service`` must dominate ``n_nc_tasks * nc_service`` so the
    critical branch sets the cycle time; the defaults give 600 vs 360.
    Pools are sized generously so contention is minimal and the win is a pure
    composition/timeline effect rather than a throughput effect.
    """
    if n_nc_tasks < 1:
        raise ValueError("n_nc_tasks must be >= 1")
    nc_total = n_nc_tasks * nc_service
    if critical_service <= nc_total:
        raise ValueError(
            f"critical_service ({critical_service}) must exceed the non-critical "
            f"chain total ({nc_total}) so the critical branch sets cycle time"
        )

    nodes: list[str] = []
    flows: list[str] = []

    nodes.append('<bpmn:startEvent id="StartEvent_1" name="Start">'
                 '<bpmn:outgoing>f_start</bpmn:outgoing></bpmn:startEvent>')
    nodes.append(_task("t_reg", "Register", "f_start", "f_reg_split"))
    flows.append(_flow("f_start", "StartEvent_1", "t_reg"))

    # AND-split: one incoming (from Register), two outgoing (critical / nc).
    f_split_crit = "f_split_crit"
    f_split_nc = "f_split_nc0"
    flows.append(_flow("f_reg_split", "t_reg", "g_split"))
    nodes.append(_parallel("g_split", "AND-split", ["f_reg_split"],
                           [f_split_crit, f_split_nc]))

    # Critical branch: single long task into the join.
    f_crit_join = "f_crit_join"
    nodes.append(_task("t_crit", "Critical", f_split_crit, f_crit_join))
    flows.append(_flow(f_split_crit, "g_split", "t_crit"))

    # Non-critical branch: a chain of short tasks into the join.
    nc_task_ids: list[str] = []
    prev_out = f_split_nc
    prev_src = "g_split"
    for i in range(1, n_nc_tasks + 1):
        tid = f"t_nc{i}"
        nc_task_ids.append(tid)
        is_last = i == n_nc_tasks
        out_flow = "f_nc_join" if is_last else f"f_nc{i}"
        nodes.append(_task(tid, f"NonCritical_{i}", prev_out, out_flow))
        flows.append(_flow(prev_out, prev_src, tid))
        prev_out = out_flow
        prev_src = tid

    # AND-join: two incoming (critical / nc), one outgoing (to Decide).
    nodes.append(_parallel("g_join", "AND-join", [f_crit_join, "f_nc_join"],
                           ["f_join_dec"]))
    flows.append(_flow(f_crit_join, "t_crit", "g_join"))
    flows.append(_flow("f_nc_join", prev_src, "g_join"))

    nodes.append(_task("t_dec", "Decide", "f_join_dec", "f_dec_end"))
    flows.append(_flow("f_join_dec", "g_join", "t_dec"))
    nodes.append('<bpmn:endEvent id="EndEvent_1" name="End">'
                 '<bpmn:incoming>f_dec_end</bpmn:incoming></bpmn:endEvent>')
    flows.append(_flow("f_dec_end", "t_dec", "EndEvent_1"))

    process_body = "".join(nodes) + "".join(flows)
    bpmn = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<bpmn:definitions xmlns:bpmn="{BPMN_NS}" id="Definitions_parallel_auto" '
        'targetNamespace="http://bpmn.io/schema/bpmn">'
        '<bpmn:process id="Process_parallel_auto" isExecutable="false">'
        f"{process_body}"
        "</bpmn:process></bpmn:definitions>"
    )

    params = _build_params(
        nc_task_ids,
        common_workers=common_workers, critical_workers=critical_workers,
        nc_workers=nc_workers, arrival_mean=arrival_mean,
        common_service=common_service, critical_service=critical_service,
        nc_service=nc_service,
    )
    return bpmn, params, nc_task_ids


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
    nc_task_ids: list[str],
    *,
    common_workers: int,
    critical_workers: int,
    nc_workers: int,
    arrival_mean: float,
    common_service: float,
    critical_service: float,
    nc_service: float,
) -> dict:
    cal_id = "cal_247"
    periods = [{"from": "MONDAY", "to": "SUNDAY",
                "beginTime": "00:00:00", "endTime": "23:59:59.999"}]

    common_tasks = ["t_reg", "t_dec"]
    critical_tasks = ["t_crit"]

    common_pool = _pool("Common", "Common", common_workers, cal_id, common_tasks)
    critical_pool = _pool("Critical", "Critical", critical_workers, cal_id, critical_tasks)
    nc_pool = _pool("NonCritical", "NonCritical", nc_workers, cal_id, nc_task_ids)

    def fix(rid: str, service: float) -> dict:
        return {
            "distribution_name": "fix",
            "distribution_params": [{"value": service}],
            "resource_id": rid,
        }

    trd: list[dict] = []
    for tid in common_tasks:
        trd.append({"task_id": tid,
                    "resources": [fix(r["id"], common_service)
                                  for r in common_pool["resource_list"]]})
    for tid in critical_tasks:
        trd.append({"task_id": tid,
                    "resources": [fix(r["id"], critical_service)
                                  for r in critical_pool["resource_list"]]})
    for tid in nc_task_ids:
        trd.append({"task_id": tid,
                    "resources": [fix(r["id"], nc_service)
                                  for r in nc_pool["resource_list"]]})

    return {
        "task_resource_distribution": trd,
        "resource_calendars": [{"id": cal_id, "name": "24/7", "time_periods": periods}],
        # Parallel (AND) gateways take all branches -> no branching probabilities.
        "gateway_branching_probabilities": [],
        "arrival_time_distribution": {
            "distribution_name": "expon",
            "distribution_params": [{"value": arrival_mean}, {"value": 0}, {"value": 3600}],
        },
        "arrival_time_calendar": periods,
        "resource_profiles": [common_pool, critical_pool, nc_pool],
        "event_distribution": [],
        "batch_processing": [],
        "case_attributes": [],
        "prioritisation_rules": [],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the parallel-automation synthetic dataset")
    ap.add_argument("--nc-tasks", type=int, default=3,
                    help="number of tasks in the non-critical chain")
    ap.add_argument("--arrival-mean", type=float, default=30.0)
    ap.add_argument("--critical-service", type=float, default=600.0)
    ap.add_argument("--nc-service", type=float, default=120.0)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parents[1] / "samples" / "dev-samples")
    ap.add_argument("--name", default="synthetic_parallel_auto")
    args = ap.parse_args()

    bpmn, params, nc_ids = build_model(
        args.nc_tasks, arrival_mean=args.arrival_mean,
        critical_service=args.critical_service, nc_service=args.nc_service,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    bpmn_path = args.out_dir / f"{args.name}.bpmn"
    json_path = args.out_dir / f"{args.name}.json"
    bpmn_path.write_text(bpmn, encoding="utf-8")
    json_path.write_text(json.dumps(params, indent=2), encoding="utf-8")
    print(f"wrote {bpmn_path}")
    print(f"wrote {json_path}")
    print(f"  non-critical tasks (automation targets): {nc_ids}")


if __name__ == "__main__":
    main()
