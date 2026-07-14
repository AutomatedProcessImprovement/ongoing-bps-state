"""CLI entry: run the state-metrics pipeline on a registered dataset.

Usage::

    # Full pipeline (sims + per-pair metrics + ranking stats):
    python -m evaluation.state_metrics.run_pipeline Loan-stable --runs 5 --levels 0,1,2,3

    # Recompute rankings only from an existing results.csv:
    python -m evaluation.state_metrics.run_pipeline --rankings-only path/to/results.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from evaluation.state_metrics.datasets import DATASETS, REPO_ROOT
from evaluation.state_metrics.pipeline import (
    PipelineConfig,
    run_pipeline,
    write_rankings_csv,
)


def _parse_levels(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(",") if x.strip())


def _recompute_rankings(results_csv: Path, out_path: Path | None,
                        n_bootstrap: int) -> Path:
    if not results_csv.exists():
        sys.exit(f"results.csv not found: {results_csv}")
    df = pd.read_csv(results_csv)
    target = out_path if out_path is not None else results_csv.with_name(
        "rankings.csv"
    )
    write_rankings_csv(df, target, n_bootstrap=n_bootstrap)
    print(f"[rankings-only] wrote {target}")
    return target


def main() -> None:
    p = argparse.ArgumentParser(description="Run state-metrics pipeline")
    p.add_argument("dataset", nargs="?", choices=sorted(DATASETS.keys()),
                   help="dataset to run (omit when using --rankings-only)")
    p.add_argument("--runs", type=int, default=None)
    p.add_argument("--levels", type=_parse_levels, default=None)
    p.add_argument("--perturbation",
                   choices=["resources", "duration", "role_swap",
                            "calendar_shift", "calendar_shift_all", "gateway",
                            "arrival_burst", "relabel", "rephase",
                            "mix_ratio", "label_swap", "case_route",
                            "parallel_auto", "front_back_load",
                            "case_type_drift", "route_error"],
                   default="resources",
                   help="which perturbation family to apply")
    p.add_argument("--remove-from-profile", default=None,
                   help="override the dataset's default perturbation profile "
                        "(also used as the `from` profile for role_swap and "
                        "default calendar for calendar_shift)")
    p.add_argument("--role-swap-to", default=None,
                   help="role_swap: destination profile (resources from this "
                        "profile take over the swapped tasks)")
    p.add_argument("--calendar-shift-profile", default=None,
                   help="calendar_shift: profile whose calendar to shift; "
                        "defaults to --remove-from-profile")
    p.add_argument("--gateway-id", default=None,
                   help="gateway: bias this gateway_id (defaults to the "
                        "most balanced gateway)")
    p.add_argument("--relabel-activity", default=None,
                   help="relabel: which activity to split (defaults to the "
                        "most frequent activity in the window)")
    p.add_argument("--relabel-to", default=None,
                   help="relabel: new label for the relabeled instances "
                        "(defaults to '<activity>__alt')")
    p.add_argument("--mix-green-params", type=Path, default=None,
                   help="mix_ratio: path to the green-population params JSON")
    p.add_argument("--mix-red-params", type=Path, default=None,
                   help="mix_ratio: path to the red-population params JSON")
    p.add_argument("--mix-baseline-green", type=float, default=0.5,
                   help="mix_ratio: baseline fraction of green cases (the "
                        "reference mix). Default 0.5.")
    p.add_argument("--case-route-ruled", type=int, default=None,
                   help="case_route / case_type_drift: number of XOR splits the "
                        "reference sim routes by case_type (default: all splits)")
    p.add_argument("--load-direction", choices=["front", "back"], default="front",
                   help="front_back_load: which end of the chain gets the "
                        "duration mass (default: front)")
    p.add_argument("--gt-total-cases", type=int, default=2000)
    p.add_argument("--sim-total-cases", type=int, default=2000)
    p.add_argument("--outputs-root", type=Path,
                   default=REPO_ROOT / "outputs" / "state_metrics")
    p.add_argument("--cutoff-strategy",
                   choices=["p90_wip", "fraction", "n_ongoing"],
                   default="p90_wip",
                   help="how to pick the cutoff timestamp (default: p90_wip)")
    p.add_argument("--cutoff-fraction", type=float, default=0.5,
                   help="fraction of the log span when --cutoff-strategy=fraction")
    p.add_argument("--target-ongoing", type=int, default=30,
                   help="target ongoing-case count when --cutoff-strategy=n_ongoing")
    p.add_argument("--horizon-hours", type=float, default=None,
                   help="fixed horizon length in hours. If omitted, falls back "
                        "to legacy 2× mean case duration (NOT comparable across "
                        "utilization regimes — set this for cross-dataset runs).")
    p.add_argument("--rankings-only", type=Path, default=None, metavar="RESULTS_CSV",
                   help="skip simulation; recompute rankings.csv from an "
                        "existing results.csv and exit")
    p.add_argument("--rankings-out", type=Path, default=None,
                   help="output path for rankings.csv (defaults to "
                        "<results_csv_dir>/rankings.csv)")
    p.add_argument("--bootstrap-iters", type=int, default=1000,
                   help="number of bootstrap iterations for ranking CIs")
    args = p.parse_args()

    if args.rankings_only is not None:
        _recompute_rankings(args.rankings_only, args.rankings_out,
                            args.bootstrap_iters)
        return

    if args.dataset is None:
        p.error("dataset is required unless --rankings-only is given")

    spec = DATASETS[args.dataset]
    if args.levels is not None:
        levels = args.levels
    elif args.perturbation == "duration":
        levels = spec.default_duration_levels
    elif args.perturbation == "role_swap":
        levels = (0, 1, 2, 3)
    elif args.perturbation == "calendar_shift":
        levels = (-8, -4, 0, 4, 8)
    elif args.perturbation == "calendar_shift_all":
        # 09:00-17:00 window: feasible (non-wrapping) shifts are 0..+7h.
        levels = (0, 2, 4, 6)
    elif args.perturbation == "gateway":
        levels = (0, 1, 2, 3)
    elif args.perturbation == "arrival_burst":
        # CV² multipliers 1.0, 1.5, 2.0, 3.0, 5.0 (encoded as 0/50/100/200/400).
        levels = (0, 50, 100, 200, 400)
    elif args.perturbation == "relabel":
        # Percentage of the chosen activity's instances relabeled.
        levels = (0, 10, 20, 30, 40)
    elif args.perturbation == "rephase":
        # Jitter magnitude in hours (per-case uniform [-level, +level]).
        levels = (0, 1, 2, 4, 8)
    elif args.perturbation == "mix_ratio":
        # Percentage-point shifts from `mix_baseline_green` toward more green.
        levels = (0, 5, 10, 20, 30)
    elif args.perturbation == "label_swap":
        # Percentage of cases whose green/red label is flipped (balanced).
        levels = (0, 10, 20, 30, 40)
    elif args.perturbation == "case_route":
        # Percentage of case_type tags swapped on a real attribute-routed sim.
        levels = (0, 10, 20, 30, 40)
    elif args.perturbation == "case_type_drift":
        # Percentage drift strength of case_type tags along the timeline.
        levels = (0, 25, 50, 75, 100)
    elif args.perturbation == "parallel_auto":
        # Percentage automation (duration shrink) of the non-critical branch.
        levels = spec.default_levels
    elif args.perturbation == "front_back_load":
        # Percentage of duration mass moved toward the chosen chain end.
        levels = spec.default_levels
    elif args.perturbation == "route_error":
        # Number of leading XOR splits whose case_type routing is inverted.
        levels = spec.default_levels
    else:
        levels = spec.default_levels

    cfg = PipelineConfig(
        dataset_name=spec.name,
        bpmn_path=spec.bpmn,
        params_path=spec.params,
        remove_from_profile=args.remove_from_profile or spec.remove_from_profile,
        outputs_root=args.outputs_root,
        runs=args.runs if args.runs is not None else spec.default_runs,
        levels=levels,
        perturbation=args.perturbation,
        gt_total_cases=args.gt_total_cases,
        sim_total_cases=args.sim_total_cases,
        cutoff_strategy=args.cutoff_strategy,
        cutoff_fraction=args.cutoff_fraction,
        target_ongoing=args.target_ongoing,
        horizon_hours=args.horizon_hours,
        role_swap_to=args.role_swap_to,
        calendar_shift_profile=args.calendar_shift_profile,
        gateway_id=args.gateway_id,
        relabel_activity=args.relabel_activity,
        relabel_to=args.relabel_to,
        mix_green_params=args.mix_green_params,
        mix_red_params=args.mix_red_params,
        mix_baseline_green=args.mix_baseline_green,
        case_route_ruled=args.case_route_ruled,
        automate_task_ids=spec.automate_task_ids or None,
        chain_task_ids=spec.chain_task_ids or None,
        load_direction=args.load_direction,
    )
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
