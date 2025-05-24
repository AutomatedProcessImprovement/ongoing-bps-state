# ────────────────────────────────────────────────────────────────────
#  test/evaluate_with_existing_alog.py
# ────────────────────────────────────────────────────────────────────
"""
Run the three-flavour evaluation pipeline.

Examples
--------
...
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
import numpy as np 

import evaluation as ev
from helper import generate_short_uuid, read_event_log, compute_cut_points, build_aggregated_from_cuts

# ────────────────────────────────────────────────────────────────────
# 1.  Dataset catalogue
# ────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class Dataset:
    alog: str
    model: str
    params: str
    total_cases: int
    cut: str            # ISO-8601 (UTC)
    horizon_days: int


DATASETS: dict[str, Dataset] = {
    # -------- Real-life --------------------------------------------
    "BPIC_2012": Dataset(
        alog="samples/bpm-2025/real-life-configuration-and-logs/BPIC_2012_W.csv",
        model="samples/bpm-2025/real-life-configuration-and-logs/2012_diff_extr/best_result/BPIC_2012_train.bpmn",
        params="samples/bpm-2025/real-life-configuration-and-logs/2012_diff_extr/best_result/BPIC_2012_train.json",
        total_cases=5_000,
        cut="2012-01-18T13:00:00Z",
        horizon_days=10,
    ),
    "BPIC_2017": Dataset(
        alog="samples/bpm-2025/real-life-configuration-and-logs/BPIC_2017_W.csv",
        model="samples/bpm-2025/real-life-configuration-and-logs/2017_diff_extr/best_result/BPIC_2017_train.bpmn",
        params="samples/bpm-2025/real-life-configuration-and-logs/2017_diff_extr/best_result/BPIC_2017_train.json",
        total_cases=10_000,
        cut="2016-10-10T13:00:00Z",
        horizon_days=26,
    ),
    "ACADEMIC_CREDENTIALS": Dataset(
        alog="samples/bpm-2025/real-life-configuration-and-logs/AcademicCredentials.csv",
        model="samples/bpm-2025/real-life-configuration-and-logs/academic_diff_extr/best_result/AcademicCredentials_train.bpmn",
        params="samples/bpm-2025/real-life-configuration-and-logs/academic_diff_extr/best_result/AcademicCredentials_train.json",
        total_cases=5_000,
        cut="2016-05-02T13:00:00Z",
        horizon_days=46,
    ),
    "WORK_ORDERS": Dataset(
        alog="samples/bpm-2025/real-life-configuration-and-logs/work_orders.csv.gz",
        model="samples/bpm-2025/real-life-configuration-and-logs/workorders_diff_extr/best_result/work_orders_train.bpmn",
        params="samples/bpm-2025/real-life-configuration-and-logs/workorders_diff_extr/best_result/work_orders_train.json",
        total_cases=20_000,
        cut="2022-12-22T07:00:00Z",
        horizon_days=15,
    ),
    # -------- Synthetic – Loan App ---------------------------------
    "LOAN_STABLE": Dataset(
        alog="samples/icpm-2025/synthetic/Loan-stable.csv",
        model="samples/icpm-2025/synthetic/Loan-stable.bpmn",
        params="samples/icpm-2025/synthetic/Loan-stable.json",
        total_cases=5_000,
        cut="2025-01-20T10:00:00Z",
        horizon_days=1,
    ),
    "LOAN_CIRCADIAN": Dataset(
        alog="samples/icpm-2025/synthetic/Loan-circadian.csv",
        model="samples/icpm-2025/synthetic/Loan-circadian.bpmn",
        params="samples/icpm-2025/synthetic/Loan-circadian.json",
        total_cases=5_000,
        cut="2025-03-20T15:00:00Z",
        horizon_days=3, #4 for 95
    ),
    "LOAN_UNSTABLE": Dataset(
        alog="samples/icpm-2025/synthetic/Loan-unpredictable.csv",
        model="samples/icpm-2025/synthetic/Loan-unpredictable.bpmn",
        params="samples/icpm-2025/synthetic/Loan-unpredictable.json",
        total_cases=5_000,
        cut="2025-03-20T15:00:00Z",
        horizon_days=2, #3 for 95
    ),
    # -------- Synthetic – P2P --------------------------------------
    "P2P_STABLE": Dataset(
        alog="samples/icpm-2025/synthetic/P2P-stable.csv",
        model="samples/icpm-2025/synthetic/P2P-stable.bpmn",
        params="samples/icpm-2025/synthetic/P2P-stable.json",
        total_cases=5_000,
        cut="2020-01-15T10:00:00Z",
        horizon_days=1,
    ),
    "P2P_CIRCADIAN": Dataset(
        alog="samples/icpm-2025/synthetic/P2P-circadian.csv",
        model="samples/icpm-2025/synthetic/P2P-circadian.bpmn",
        params="samples/icpm-2025/synthetic/P2P-circadian.json",
        total_cases=5_000,
        cut="2020-01-10T10:00:00Z",
        horizon_days=3, #4 for 95
    ),
    "P2P_UNSTABLE": Dataset(
        alog="samples/icpm-2025/synthetic/P2P-unstable.csv",
        model="samples/icpm-2025/synthetic/P2P-unstable.bpmn",
        params="samples/icpm-2025/synthetic/P2P-unstable.json",
        total_cases=5_000,
        cut="2020-01-10T10:00:00Z",
        horizon_days=4, #5 for 95
    ),
}

SYNTHETIC_DATASETS = [
    "LOAN_STABLE", "LOAN_CIRCADIAN", "LOAN_UNSTABLE",
    "P2P_STABLE", "P2P_CIRCADIAN", "P2P_UNSTABLE"
]

REAL_LIFE_DATASETS = [
    "BPIC_2012", "BPIC_2017", "ACADEMIC_CREDENTIALS", "WORK_ORDERS"
]

ALIASES = {
    "ALL": list(DATASETS.keys()),
    "SYNTHETIC": SYNTHETIC_DATASETS,
    "REAL-LIFE": REAL_LIFE_DATASETS,
}

# ────────────────────────────────────────────────────────────────────
# 2.  Column rename map that Prosimos always emits
# ────────────────────────────────────────────────────────────────────
SIM_RENAME_MAP = {
    "CaseId":       "case_id",
    "Activity":     "activity",
    "Resource":     "resource",
    "StartTime":    "start_time",
    "EndTime":      "end_time",
    "EnabledTime":  "enable_time",
}


# ── wrapper helpers that execute the engines ──────────────────────
from src.runner import run_process_state_and_simulation as _run_ps
from src.process_state_prosimos_run import run_basic_simulation as _run_basic
import json, pandas as pd


def ps_runner(*, io: ev.SimulationIO, **kwargs):
    """
    Wrapper for the “process-state” flavour.
    - keeps the original kwargs intact for the caller
    - converts column_mapping → JSON string
    - strips helper-only keys before delegating to ProSiMoS
    """
    kw = kwargs.copy()                      # do NOT mutate caller’s dict
    kw.pop("rename_map", None)              # helper-only
    if isinstance(kw.get("column_mapping"), dict):
        kw["column_mapping"] = json.dumps(kw["column_mapping"])

    # ProSiMoS expects ISO-8601 strings
    for key in ("start_time", "simulation_horizon"):
        if isinstance(kw.get(key), pd.Timestamp):
            kw[key] = kw[key].isoformat()

    kw.setdefault("simulate", True)         # required by _run_ps

    _run_ps(
        sim_stats_csv=str(io.stats_csv),
        sim_log_csv=str(io.log_csv),
        **kw,
    )


def wu_runner(*, io: ev.SimulationIO, **kwargs):
    """
    Wrapper for both warm-up flavours.
    Converts helper keys and passes through to basic simulation.
    """
    kw = kwargs.copy()
    kw.pop("rename_map", None)              # helper-only

    if isinstance(kw.get("start_date"), pd.Timestamp):
        kw["start_date"] = kw["start_date"].isoformat()

    _run_basic(
        out_stats_csv_path=str(io.stats_csv),
        out_log_csv_path=str(io.log_csv),
        **kw,
    )


# ────────────────────────────────────────────────────────────────────
# 4.  Main driver
# ────────────────────────────────────────────────────────────────────


def _run_dataset(dataset: str, runs: int, *, cut_strategy: str) -> None:
    cfg = DATASETS[dataset]

    # root output folder
    out_root = Path("outputs") / dataset / generate_short_uuid()
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_root}")

    # load reference event log
    alog_df = read_event_log(
        cfg.alog,
        rename={
            "CaseId":   "case_id",
            "Activity": "activity",
            "Resource": "resource",
            "StartTime":"start_time",
            "EndTime":  "end_time",
        },
        required=["case_id", "activity", "start_time", "end_time", "resource"],
    )

    # cut-off list according to the selected strategy
    cut_points = compute_cut_points(
        alog_df,
        cfg.horizon_days,
        strategy=cut_strategy,
        fixed_cut=cfg.cut,
    )

    # container for per-cut summaries
    per_cut_results: dict[str, dict] = {}

    for cut_ts in cut_points:
        horizon = pd.Timedelta(days=cfg.horizon_days)
        end_ts = cut_ts + horizon

        # sub-folder name safe for most filesystems
        cut_folder_name = cut_ts.isoformat().replace(":", "-")
        cut_dir = out_root / cut_folder_name
        cut_dir.mkdir()

        # split reference log into A_event, A_ongoing, A_complete
        A_event, A_ongoing, A_complete = ev.split_into_subsets(
            alog_df, cut_ts, end_ts
        )

        # keep the reference subsets for inspection
        for df, fname in (
            (A_event,   "A_event_filter.csv"),
            (A_ongoing, "A_ongoing.csv"),
            (A_complete,"A_complete.csv"),
        ):
            ev._dump(df, cut_dir, fname)

        # run the three flavours “runs” times
        runs_PS, runs_WU, runs_WU2 = [], [], []
        for run_no in range(1, runs + 1):
            run_dir = cut_dir / str(run_no)
            run_dir.mkdir()

            io_obj = ev.SimulationIO(
                log_csv=run_dir / "sim_log.csv",
                stats_csv=run_dir / "sim_stats.csv",
                out_dir=run_dir,
            )

            # process-state flavour
            runs_PS.append(
                ev.evaluate(
                    "process_state",
                    io_obj,
                    ps_runner,
                    cut=cut_ts,
                    end=end_ts,
                    A_event=A_event,
                    A_ongoing=A_ongoing,
                    A_complete=A_complete,
                    runner_kwargs=dict(
                        event_log=cfg.alog,
                        bpmn_model=cfg.model,
                        bpmn_parameters=cfg.params,
                        start_time=cut_ts,
                        simulation_horizon=end_ts + horizon,
                        total_cases=cfg.total_cases,
                        column_mapping={v: k for k, v in SIM_RENAME_MAP.items()},
                        rename_map=SIM_RENAME_MAP,
                    ),
                )
            )

            # warm-up flavour
            # runs_WU.append(
            #     ev.evaluate(
            #         "warmup",
            #         io_obj,
            #         wu_runner,
            #         cut=cut_ts,
            #         end=end_ts,
            #         A_event=A_event,
            #         A_ongoing=A_ongoing,
            #         A_complete=A_complete,
            #         runner_kwargs=dict(
            #             bpmn_model=cfg.model,
            #             json_sim_params=cfg.params,
            #             total_cases=cfg.total_cases,
            #             start_date=(cut_ts - horizon).isoformat(),
            #             rename_map=SIM_RENAME_MAP,
            #         ),
            #     )
            # )

            # warm-up v2 flavour
            runs_WU2.append(
                ev.evaluate(
                    "warmup2",
                    io_obj,
                    wu_runner,
                    cut=cut_ts,
                    end=end_ts,
                    A_event=A_event,
                    A_ongoing=A_ongoing,
                    A_complete=A_complete,
                    runner_kwargs=dict(
                        bpmn_model=cfg.model,
                        json_sim_params=cfg.params,
                        total_cases=cfg.total_cases,
                        start_date=(cut_ts - horizon).isoformat(),
                        rename_map=SIM_RENAME_MAP,
                    ),
                )
            )

        # aggregate the Monte-Carlo repetitions for this cut-off
        agg_PS  = ev.aggregate(runs_PS)
        agg_WU  = ev.aggregate(runs_WU)
        agg_WU2 = ev.aggregate(runs_WU2)

        per_cut_results[cut_ts.isoformat()] = {
            "num_runs": runs,
            "process_state": {"aggregated": agg_PS},
            # "warmup":        {"aggregated": agg_WU},
            "warmup2":       {"aggregated": agg_WU2},
            # "PS_vs_WU":  ev.compare(agg_PS,  agg_WU,  ("process_state", "warmup")),
            "PS_vs_WU2": ev.compare(agg_PS,  agg_WU2, ("process_state", "warmup2")),
        }

    # helper to average the “mean” fields across cut-offs
    def average_dicts(list_of_dicts: list[dict]) -> dict:
        result: dict = {}
        keys = {k for d in list_of_dicts for k in d}
        for k in keys:
            values = [d[k]["mean"] for d in list_of_dicts if d[k]["mean"] is not None]
            result[k] = float(np.mean(values)) if values else None
        return result

    # --- overall averages across all cut-offs --------------------------
    overall_average: dict = {}
    #warmup here in for loop
    for flavour in ("process_state", "warmup2"):
        sample_any_cut = next(iter(per_cut_results.values()))
        subfamilies = sample_any_cut[flavour]["aggregated"].keys()
        overall_average[flavour] = {"aggregated": {}}
        for sf in subfamilies:
            aggregates_for_sf = [
                per_cut_results[c][flavour]["aggregated"][sf]
                for c in per_cut_results
            ]
            overall_average[flavour]["aggregated"][sf] = build_aggregated_from_cuts(
                aggregates_for_sf
            )

    # --- comparisons of those overall averages -------------------------
    overall_comparison = {
        # "PS_vs_WU": ev.compare(
        #     overall_average["process_state"]["aggregated"],
        #     overall_average["warmup"]["aggregated"],
        #     ("process_state", "warmup"),
        # ),
        "PS_vs_WU2": ev.compare(
            overall_average["process_state"]["aggregated"],
            overall_average["warmup2"]["aggregated"],
            ("process_state", "warmup2"),
        ),
    }


    final_report = {
        "num_runs": runs,
        "cut_strategy": cut_strategy,
        "cut_offs": list(per_cut_results.keys()),
        "per_cut": per_cut_results,
        "overall_average": overall_average,
        "overall_comparison": overall_comparison,
    }


    with open(out_root / "final_results.json", "w", encoding="utf-8") as fh:
        json.dump(final_report, fh, indent=2)

    print(json.dumps(final_report, indent=2))



def main(dataset: str, runs: int = 10, *, cut_strategy: str = "fixed") -> None:
    if dataset in ALIASES:
        for name in ALIASES[dataset]:
            print(f"\n\n===== Running dataset: {name} =====")
            _run_dataset(name, runs, cut_strategy=cut_strategy)

    else:
        _run_dataset(dataset, runs, cut_strategy=cut_strategy)



# ────────────────────────────────────────────────────────────────────
# 5.  CLI entry-point
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "dataset",
        choices=list(DATASETS.keys()) + list(ALIASES.keys()),
        help="Name of a dataset or group (ALL, SYNTHETIC, REAL-LIFE)"
    )
    p.add_argument("--runs", type=int, default=10,
                   help="Number of repetitions (default: 10)")
    p.add_argument(
        "--cut-strategy",
        default="fixed",
        choices=["fixed", "wip3", "segment10"],
        help=(
            "How to choose cut-off timestamps.\n"
            "  fixed       – use the cut stored in DATASETS;\n"
            "  wip3        – 3 points at 10 / 50 / 90 % of the maximum WiP;\n"
            "  segment10   – 10 random points in 10 equal time segments."
        )
    )
    cli = p.parse_args()
    main(dataset=cli.dataset, runs=cli.runs, cut_strategy=cli.cut_strategy)

