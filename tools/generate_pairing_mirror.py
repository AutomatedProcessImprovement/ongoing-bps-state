"""Generate the *mirrored-branch* pairing synthetic (BPMN + Prosimos params).

The pairing family exists to show an error that every population-level summary
is blind to while every individual case is wrong. Earlier designs got there by
luck -- the two branches carried *different* activity labels, so a routing error
changed the population's activity mix and the plain ``activity`` projection saw
it. This design makes the blindness provable instead.

Both branches carry the SAME six activity labels in MIRRORED order, with one
shared duration distribution::

    Start -> Register -> Assess (long, common)
          -> [pos 1: R_1 | R_6] -> [pos 2: R_2 | R_5] -> [pos 3: R_3 | R_4]
          -> [pos 4: R_4 | R_3] -> [pos 5: R_5 | R_2] -> [pos 6: R_6 | R_1]
          -> Settle -> End

Routing is by a ``case_type`` case attribute, so a case takes the same side at
every position: a P-case runs R_1, R_2, ..., R_6 and a Q-case runs R_6, R_5,
..., R_1. Nine activities per case -- Register, Assess, six perturbable
positions, Settle -- matching the composition and timeline geometries.

Why six position gateways rather than two literal branches: the injected error
reverses the order over a *centred window* of positions, and a partial reversal
is a path that two whole-chain branches cannot express. Splitting the choice
per position makes every even-width centred window representable while leaving
the two realised paths exactly the mirrored chains above -- the routing rule is
one attribute test, so all six positions move together unless the error says
otherwise.

Why the window is centred and even: position ``i`` offers ``{R_i, R_(7-i)}``,
so flipping position ``i`` maps label ``R_i`` to ``R_(7-i)``. A centred window
of even width has ``a + b = 7`` for its first and last position, so flipping it
IS the reversal ``i -> 7-i`` on that window and the identity outside it. That
is what makes the population invariants below hold exactly rather than on
average.

What is preserved, and how exactly (see the audit for the measured residuals):

* **per-case label set** -- ``{Register, Assess, R_1..R_6, Settle}`` on both
  sides and at every extent. Exact.
* **per-position population multiset** -- at position ``i`` the two sides
  contribute ``{R_i, R_(7-i)}`` before the error and ``{R_(7-i), R_i}`` after
  it. The same multiset. Exact at every extent.
* **case duration and within-case event positions** -- every position task
  draws from one distribution, so which label a case gets at position ``i``
  does not change how long it takes. Exact in distribution.
* **bigram histogram** -- exact only at the full extent, where the two branches
  simply exchange sequences. A partial reversal cuts the chain at the window's
  two edges, and the four adjacencies across those edges are new: of the ten
  interior bigram types, six survive and four are exchanged. ``ngd`` therefore
  responds at extents 2 and 4 and goes blind at extent 6, which is worth saying
  out loud rather than hiding.

The residual on the instantaneous ``activity`` projection is the honest part.
Per-position cancellation is exact, but at a given instant the ongoing cases
are spread across positions in finite numbers, so the cancellation is only as
good as the phase mix in the window -- the same mechanism as the ngd
upstream-imbalance result. Expect ``activity`` at the residual level and
``activity_case`` at full strength.

Run::

    python -m tools.generate_pairing_mirror --positions 6 --name synthetic_pairing_mirror6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

from tools.generate_route_error import (
    _build_params_common,
)
from tools.genprov import write_sidecar

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
    return (f'<bpmn:exclusiveGateway id="{gid}" name="{escape(name)}">'
            f"{inc}{out}</bpmn:exclusiveGateway>")


def _flow(fid: str, src: str, tgt: str) -> str:
    return f'<bpmn:sequenceFlow id="{fid}" sourceRef="{src}" targetRef="{tgt}" />'


def mirrored_labels(n: int) -> tuple[list[str], list[str]]:
    """(P-side labels, Q-side labels) by position, 1..n.

    P runs ``R_1 .. R_n``; Q runs the same labels in reverse. Position ``i``
    therefore offers ``R_i`` on the P side and ``R_(n+1-i)`` on the Q side.
    """
    p = [f"R_{i}" for i in range(1, n + 1)]
    q = [f"R_{n + 1 - i}" for i in range(1, n + 1)]
    return p, q


def centred_window(n: int, extent: int) -> list[int]:
    """The ``extent`` positions of a 1..n chain centred on the chain's middle.

    ``extent`` must be even and no larger than ``n``; the window's first and
    last positions then sum to ``n + 1``, which is what makes flipping it a
    reversal (see the module docstring).
    """
    if extent % 2 or not 0 < extent <= n:
        raise ValueError(f"extent must be even and in (0, {n}], got {extent}")
    lo = (n - extent) // 2 + 1
    positions = list(range(lo, lo + extent))
    assert positions[0] + positions[-1] == n + 1
    return positions


def build_model(
    n_positions: int = 6,
    *,
    workers: int = 340,
    arrival_mean: float = 29.0,
    position_service: float = 300.0,
    register_service: float = 180.0,
    assess_service: float = 1200.0,
    settle_service: float = 180.0,
    cv: float = 0.07,
    threshold: float = 50.0,
) -> tuple[str, dict, dict]:
    """Return (bpmn_xml, prosimos_params, structure) for the mirrored geometry.

    ``assess_service`` is the long common activity the cut-off is designed to
    land inside, so the ongoing set is dominated by cases that have not yet
    reached the branch; ``position_service`` is shared by all ``2 * n`` position
    tasks, which is what makes the error timing-neutral.
    """
    if n_positions < 2 or n_positions % 2:
        raise ValueError("n_positions must be even and >= 2")

    p_labels, q_labels = mirrored_labels(n_positions)
    nodes: list[str] = []
    flows: list[str] = []
    task_means: dict[str, float] = {}
    gateway_specs: list[dict] = []

    nodes.append('<bpmn:startEvent id="StartEvent_1" name="Start">'
                 "<bpmn:outgoing>f_start</bpmn:outgoing></bpmn:startEvent>")
    nodes.append(_task("t_reg", "Register", "f_start", "f_reg_assess"))
    task_means["t_reg"] = register_service
    flows.append(_flow("f_start", "StartEvent_1", "t_reg"))

    nodes.append(_task("t_assess", "Assess", "f_reg_assess", "f_assess_g1s"))
    task_means["t_assess"] = assess_service
    flows.append(_flow("f_reg_assess", "t_reg", "t_assess"))

    prev_out, prev_src = "f_assess_g1s", "t_assess"
    for k in range(1, n_positions + 1):
        g_split, g_join = f"g{k}s", f"g{k}j"
        # The _a / _b suffixes are load-bearing: the route-error builders read
        # them to tell the two sides apart.
        f_sa, f_sb = f"f_g{k}s_a", f"f_g{k}s_b"
        f_aj, f_bj = f"f_a{k}_g{k}j", f"f_b{k}_g{k}j"
        is_last = k == n_positions
        f_join_out = f"f_g{k}j_next"

        flows.append(_flow(prev_out, prev_src, g_split))
        nodes.append(_xor(g_split, f"XOR-split-{k}", [prev_out], [f_sa, f_sb]))

        # Same position, two labels: R_k on the P side, R_(n+1-k) on the Q side.
        for side, flow_in, flow_out, label in (
            ("a", f_sa, f_aj, p_labels[k - 1]),
            ("b", f_sb, f_bj, q_labels[k - 1]),
        ):
            tid = f"t_{side}{k}"
            nodes.append(_task(tid, label, flow_in, flow_out))
            flows.append(_flow(flow_in, g_split, tid))
            flows.append(_flow(flow_out, tid, g_join))
            task_means[tid] = position_service

        nodes.append(_xor(g_join, f"XOR-join-{k}", [f_aj, f_bj], [f_join_out]))
        gateway_specs.append({"gateway_id": g_split, "split": True,
                              "paths": [f_sa, f_sb]})
        gateway_specs.append({"gateway_id": g_join, "split": False,
                              "paths": [f_join_out]})
        prev_out, prev_src = f_join_out, g_join

        if is_last:
            nodes.append(_task("t_settle", "Settle", f_join_out, "f_settle_end"))
            task_means["t_settle"] = settle_service
            flows.append(_flow(f_join_out, g_join, "t_settle"))

    nodes.append('<bpmn:endEvent id="EndEvent_1" name="End">'
                 "<bpmn:incoming>f_settle_end</bpmn:incoming></bpmn:endEvent>")
    flows.append(_flow("f_settle_end", "t_settle", "EndEvent_1"))

    process_body = "".join(nodes) + "".join(flows)
    bpmn = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<bpmn:definitions xmlns:bpmn="{BPMN_NS}" id="Definitions_pairing_mirror" '
        'targetNamespace="http://bpmn.io/schema/bpmn">'
        '<bpmn:process id="Process_pairing_mirror" isExecutable="false">'
        f"{process_body}"
        "</bpmn:process></bpmn:definitions>"
    )

    params = _build_params_common(
        task_means, gateway_specs, workers=workers, arrival_mean=arrival_mean,
        route_attribute="continuous", threshold=threshold, cv=cv,
    )
    structure = {
        "n_positions": n_positions,
        "p_sequence": p_labels,
        "q_sequence": q_labels,
        "distinct_labels": sorted({*p_labels, *q_labels}),
        "activities_per_case": n_positions + 3,
        "task_nodes": len(task_means),
        "windows": {e: centred_window(n_positions, e)
                    for e in range(2, n_positions + 1, 2)},
    }
    return bpmn, params, structure


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate the mirrored-branch pairing synthetic")
    ap.add_argument("--positions", type=int, default=6)
    ap.add_argument("--workers", type=int, default=340)
    ap.add_argument("--arrival-mean", type=float, default=29.0)
    ap.add_argument("--position-service", type=float, default=300.0)
    ap.add_argument("--register-service", type=float, default=180.0)
    ap.add_argument("--assess-service", type=float, default=1200.0)
    ap.add_argument("--settle-service", type=float, default=180.0)
    ap.add_argument("--cv", type=float, default=0.07)
    ap.add_argument("--threshold", type=float, default=50.0)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parents[1] / "samples" / "dev-samples")
    ap.add_argument("--name", default="synthetic_pairing_mirror6")
    args = ap.parse_args()

    bpmn, params, structure = build_model(
        args.positions, workers=args.workers, arrival_mean=args.arrival_mean,
        position_service=args.position_service,
        register_service=args.register_service,
        assess_service=args.assess_service,
        settle_service=args.settle_service,
        cv=args.cv, threshold=args.threshold,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{args.name}.bpmn").write_text(bpmn, encoding="utf-8")
    (args.out_dir / f"{args.name}.json").write_text(
        json.dumps(params, indent=2), encoding="utf-8")
    print(f"wrote {args.out_dir / (args.name + '.bpmn')}")
    print(f"wrote {args.out_dir / (args.name + '.json')}")
    print(f"  P runs {' -> '.join(structure['p_sequence'])}")
    print(f"  Q runs {' -> '.join(structure['q_sequence'])}")
    print(f"  {structure['activities_per_case']} activities per case, "
          f"{structure['task_nodes']} task nodes, "
          f"{args.positions} split gateways")
    for e, w in structure["windows"].items():
        print(f"  extent {e}: positions {w}")
    sidecar = write_sidecar(
        args.out_dir, args.name, generator="tools/generate_pairing_mirror.py",
        params={"positions": args.positions, "workers": args.workers,
                "arrival_mean": args.arrival_mean,
                "position_service": args.position_service,
                "register_service": args.register_service,
                "assess_service": args.assess_service,
                "settle_service": args.settle_service,
                "cv": args.cv, "threshold": args.threshold},
    )
    print(f"wrote {sidecar}")


if __name__ == "__main__":
    main()
