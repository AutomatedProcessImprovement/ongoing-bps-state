"""Generate the synthetic *front/back-XOR* dataset (BPMN + Prosimos params).

Scenario #4 (the "compensating front/back-load" experiment — the one designed to
be invisible even to the relative-event-distribution baseline ``red``).

A single XOR gateway routes cases 50/50 into two branches of the SAME number of
activities and the SAME total duration, but with opposite *time profiles*::

    Start -> Register
          -> XOR (50/50)
               -> Branch P: P1 (long) -> P2 (med) -> P3 (short)   [FRONT-loaded]
               -> Branch Q: Q1 (short) -> Q2 (med) -> Q3 (long)   [BACK-loaded]
          -> XOR-join -> Decide -> End

Both branches run on the same pool and total the same wall-clock time, so a
case's cycle time does not depend on which branch it takes, and (population 50/50)
the *aggregate* relative-event-distribution is the average of a front- and a
back-loaded profile either way.

The state-metrics ``front_back_swap`` perturbation
(``evaluation.state_metrics.perturb.build_front_back_swap_params``) then *swaps*
the two time profiles in the BPS model — Branch P becomes back-loaded and Branch
Q front-loaded — while holding each branch's total duration exactly fixed. So:

* cycle time is unchanged (per-branch total invariant),
* the activity sequence is unchanged (``ngd_n2`` blind),
* and because the front/back profiles compensate across the 50/50 population, the
  aggregate relative-event-distribution is unchanged too (``red`` blind).

Only pairing each ongoing case to its own ground-truth continuation reveals that
*where in its timeline the work sits* has moved — a case on Branch P that used to
front-load its remaining work now back-loads it. The per-instant state view
(``activity_case``) localises it. This is the "where is the error" scenario.

Run::

    python -m tools.generate_front_back_xor          # default asset
    python -m tools.generate_front_back_xor --branch-len 3 --out-dir samples/dev-samples
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


def _norm(mean: float, cv: float = 0.07) -> dict:
    """Narrow normal duration (Prosimos ``norm`` order ``[mean, std, min, max]``).

    ``cv`` is the coefficient of variation. The default 7 % keeps two same-model
    sims close, but that is a *relative* bound and the gate activity is an order
    of magnitude longer than the branch tasks: at 7 % its absolute jitter is
    ~660 s against the branch tasks' ~40 s, which desynchronises when each case
    enters its branch and dominates the level-0 floor of any case-keyed view.
    ``build_model(gate_cv=...)`` therefore sets the gate's CV independently.
    The gate is common to both logs and carries no injected error, so its
    variance is pure measurement noise.
    """
    std = cv * mean
    return {
        "distribution_name": "norm",
        "distribution_params": [
            {"value": mean},
            {"value": std},
            {"value": max(1.0, mean - 2 * std)},
            {"value": mean + 3 * std},
        ],
    }


def _profile(branch_len: int, total: float, *, front: bool) -> list[float]:
    """A monotone front- or back-loaded duration profile over ``branch_len``
    activities summing to ``total``. Front-loaded ramps DOWN (early long),
    back-loaded ramps UP (late long); both are mirror images so their sums match.
    """
    # Linear ramp of positive weights, then normalise to `total`.
    weights = [float(i) for i in range(1, branch_len + 1)]        # [1,2,...,n]
    if front:
        weights = weights[::-1]                                    # [n,...,2,1]
    s = sum(weights)
    return [total * w / s for w in weights]


def build_model(
    *,
    branch_len: int = 3,
    branch_total: float = 1200.0,
    common_service: float = 180.0,
    gate_service: float = 0.0,
    gate_cv: float = 0.07,
    workers: int = 64,
    arrival_mean: float = 60.0,
) -> tuple[str, dict]:
    """Return (bpmn_xml, prosimos_params) for the front/back-XOR model.

    Branch P (front-loaded) and Branch Q (back-loaded) each have ``branch_len``
    activities summing to ``branch_total`` seconds. ``workers`` / ``arrival_mean``
    set the offered load.

    ``gate_service > 0`` inserts a long common "Intake" activity between Register
    and the XOR split. Its purpose is to move the high-WIP cutoff *before* the
    branch: with a gate this long, most cases in flight at any instant are still
    inside Intake, so their whole branch runs after the cutoff under a single
    time profile. Without it the cutoff lands mid-branch and the swap splices the
    head of one profile onto the tail of the other, which does NOT preserve the
    case total — the leak that makes cycle_time and rtd detect the perturbation.
    """
    if branch_len < 2:
        raise ValueError("branch_len must be >= 2 to carry a front/back profile")

    nodes: list[str] = []
    flows: list[str] = []
    task_means: dict[str, float] = {}

    nodes.append('<bpmn:startEvent id="StartEvent_1" name="Start"><bpmn:outgoing>f_start</bpmn:outgoing></bpmn:startEvent>')
    nodes.append(_task("t_reg", "Register", "f_start", "f_reg_gs"))
    task_means["t_reg"] = common_service
    flows.append(_flow("f_start", "StartEvent_1", "t_reg"))

    # Optional long common activity ahead of the split (see build_model docstring).
    if gate_service > 0:
        nodes.append(_task("t_gate", "Intake", "f_reg_gs", "f_gate_gs"))
        task_means["t_gate"] = gate_service
        flows.append(_flow("f_reg_gs", "t_reg", "t_gate"))
        flows.append(_flow("f_gate_gs", "t_gate", "gs"))
        split_in = "f_gate_gs"
    else:
        flows.append(_flow("f_reg_gs", "t_reg", "gs"))
        split_in = "f_reg_gs"

    # Split gateway. Flow ids leaving keep the _p / _q suffix so the builder can
    # find the two branches.
    f_sp, f_sq = "f_gs_p", "f_gs_q"
    nodes.append(_xor("gs", "XOR-split", [split_in], [f_sp, f_sq]))

    p_front = _profile(branch_len, branch_total, front=True)   # Branch P front-loaded
    q_back = _profile(branch_len, branch_total, front=False)   # Branch Q back-loaded

    def _branch(side: str, split_flow: str, means: list[float]) -> str:
        label = side.upper()
        ids = [f"t_{side}{i}" for i in range(1, branch_len + 1)]
        f_join_in = f"f_{side}_gj"
        in_flows = [split_flow] + [f"f_{side}_link{i}" for i in range(1, branch_len)]
        out_flows = [f"f_{side}_link{i}" for i in range(1, branch_len)] + [f_join_in]
        for i, tid in enumerate(ids):
            nodes.append(_task(tid, f"{label}{i + 1}", in_flows[i], out_flows[i]))
            src = "gs" if i == 0 else ids[i - 1]
            flows.append(_flow(in_flows[i], src, tid))
            task_means[tid] = means[i]
        flows.append(_flow(f_join_in, ids[-1], "gj"))
        return f_join_in

    f_pj = _branch("p", f_sp, p_front)
    f_qj = _branch("q", f_sq, q_back)

    nodes.append(_xor("gj", "XOR-join", [f_pj, f_qj], ["f_gj_dec"]))
    nodes.append(_task("t_dec", "Decide", "f_gj_dec", "f_dec_end"))
    task_means["t_dec"] = common_service
    flows.append(_flow("f_gj_dec", "gj", "t_dec"))
    nodes.append('<bpmn:endEvent id="EndEvent_1" name="End"><bpmn:incoming>f_dec_end</bpmn:incoming></bpmn:endEvent>')
    flows.append(_flow("f_dec_end", "t_dec", "EndEvent_1"))

    process_body = "".join(nodes) + "".join(flows)
    bpmn = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<bpmn:definitions xmlns:bpmn="{BPMN_NS}" id="Definitions_front_back_xor" '
        'targetNamespace="http://bpmn.io/schema/bpmn">'
        '<bpmn:process id="Process_front_back_xor" isExecutable="false">'
        f"{process_body}"
        "</bpmn:process></bpmn:definitions>"
    )

    params = _build_params(task_means, workers=workers, arrival_mean=arrival_mean,
                           task_cv={"t_gate": gate_cv})
    return bpmn, params


def _build_params(task_means: dict[str, float], *, workers: int, arrival_mean: float,
                  task_cv: dict[str, float] | None = None) -> dict:
    cal_id = "cal_247"
    periods = [{"from": "MONDAY", "to": "SUNDAY",
                "beginTime": "00:00:00", "endTime": "23:59:59.999"}]
    resource_ids = [f"Worker_{i}" for i in range(1, workers + 1)]
    task_ids = list(task_means.keys())

    task_resource_distribution = [
        {"task_id": tid,
         "resources": [{**_norm(task_means[tid], (task_cv or {}).get(tid, 0.07)),
                        "resource_id": rid} for rid in resource_ids]}
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
    gateway_branching_probabilities = [{
        "gateway_id": "gs",
        "probabilities": [{"path_id": "f_gs_p", "value": "0.5"},
                          {"path_id": "f_gs_q", "value": "0.5"}],
    }, {
        "gateway_id": "gj",
        "probabilities": [{"path_id": "f_gj_dec", "value": "1"}],
    }]
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
        "case_attributes": [],
        "prioritisation_rules": [],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the front/back-XOR synthetic dataset")
    ap.add_argument("--branch-len", type=int, default=3,
                    help="activities per branch (>= 2)")
    ap.add_argument("--branch-total", type=float, default=1200.0,
                    help="total seconds of work per branch (equal for P and Q)")
    ap.add_argument("--common-service", type=float, default=180.0)
    ap.add_argument("--gate-cv", type=float, default=0.07,
                    help="coefficient of variation for the gate activity only "
                         "(default 0.07, same as every other task)")
    ap.add_argument("--gate-service", type=float, default=0.0,
                    help="seconds of a common 'Intake' activity before the XOR "
                         "(0 = none). Long values push the high-WIP cutoff "
                         "ahead of the branch.")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--arrival-mean", type=float, default=60.0)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parents[1] / "samples" / "dev-samples")
    ap.add_argument("--name", default="synthetic_front_back_xor")
    args = ap.parse_args()

    bpmn, params = build_model(
        branch_len=args.branch_len, branch_total=args.branch_total,
        common_service=args.common_service, gate_service=args.gate_service,
        gate_cv=args.gate_cv,
        workers=args.workers,
        arrival_mean=args.arrival_mean,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    bpmn_path = args.out_dir / f"{args.name}.bpmn"
    json_path = args.out_dir / f"{args.name}.json"
    bpmn_path.write_text(bpmn, encoding="utf-8")
    json_path.write_text(json.dumps(params, indent=2), encoding="utf-8")

    def _mean(tid: str) -> float:
        t = next(t for t in params["task_resource_distribution"] if t["task_id"] == tid)
        return t["resources"][0]["distribution_params"][0]["value"]

    p_tasks = [f"t_p{i}" for i in range(1, args.branch_len + 1)]
    q_tasks = [f"t_q{i}" for i in range(1, args.branch_len + 1)]
    print(f"wrote {bpmn_path}")
    print(f"wrote {json_path}")
    print(f"  Branch P (front) means: {[round(_mean(t)) for t in p_tasks]} total={round(sum(_mean(t) for t in p_tasks))}")
    print(f"  Branch Q (back)  means: {[round(_mean(t)) for t in q_tasks]} total={round(sum(_mean(t) for t in q_tasks))}")


if __name__ == "__main__":
    main()
