# ────────────────────────────────────────────────────────────────────
#  test/evaluate_with_existing_alog.py
# ────────────────────────────────────────────────────────────────────
"""
Run the three-flavour evaluation pipeline.

Examples
--------
# 10 runs on the BPIC 2017 log
python test/evaluate_with_existing_alog.py BPIC_2017 --runs 10
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from dataclasses import dataclass

import pandas as pd

import evaluation as ev
from helper import generate_short_uuid, read_event_log


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
        horizon_days=25,
    ),
    "LOAN_CIRCADIAN": Dataset(
        alog="samples/icpm-2025/synthetic/Loan-circadian.csv",
        model="samples/icpm-2025/synthetic/Loan-circadian.bpmn",
        params="samples/icpm-2025/synthetic/Loan-circadian.json",
        total_cases=15_000, #15000
        cut="2025-03-21T15:00:00Z", #20
        horizon_days=30, #200
    ),
    # -------- Synthetic – P2P --------------------------------------
    "P2P_STABLE": Dataset(
        alog="samples/icpm-2025/synthetic/P2P-stable.csv",
        model="samples/icpm-2025/synthetic/P2P-stable.bpmn",
        params="samples/icpm-2025/synthetic/P2P-stable.json",
        total_cases=5_000,
        cut="2020-01-15T10:00:00Z",
        horizon_days=20,
    ),
    "P2P_CIRCADIAN": Dataset(
        alog="samples/icpm-2025/synthetic/P2P-circadian.csv",
        model="samples/icpm-2025/synthetic/P2P-circadian.bpmn",
        params="samples/icpm-2025/synthetic/P2P-circadian.json",
        total_cases=5_000,
        cut="2020-01-10T10:00:00Z",
        horizon_days=40,
    ),
    "P2P_UNSTABLE": Dataset(
        alog="samples/icpm-2025/synthetic/P2P-unstable.csv",
        model="samples/icpm-2025/synthetic/P2P-unstable.bpmn",
        params="samples/icpm-2025/synthetic/P2P-unstable.json",
        total_cases=5_000,
        cut="2020-01-10T10:00:00Z",
        horizon_days=40,
    ),
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


# ── wrapper helpers that REALLY execute the engines ──────────────────────
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
def main(dataset: str, runs: int = 10) -> None:
    cfg = DATASETS[dataset]

    # 4-A  output folder
    out_base = Path("outputs") / generate_short_uuid()
    out_base.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_base}")

    # 4-B  read reference log & window-split
    alog_df = read_event_log(
        cfg.alog,
        rename={
            "CaseId": "case_id",
            "Activity": "activity",
            "Resource": "resource",
            "StartTime": "start_time",
            "EndTime": "end_time",
        },
        required=["case_id", "activity", "start_time", "end_time", "resource"],
    )

    cut = pd.to_datetime(cfg.cut, utc=True)
    end = cut + pd.Timedelta(days=cfg.horizon_days)
    A_event, A_ongoing, A_complete = ev.split_into_subsets(alog_df, cut, end)

    for df, fname in [
        (A_event,    "A_event_filter.csv"),
        (A_ongoing,  "A_ongoing.csv"),
        (A_complete, "A_complete.csv"),
    ]:
        ev._dump(df, out_base, fname)

    # 4-C  Monte-Carlo runs
    runs_PS, runs_WU, runs_WU2 = [], [], []
    for i in range(1, runs + 1):
        run_dir = out_base / str(i)
        run_dir.mkdir()

        io_obj = ev.SimulationIO(
            log_csv=run_dir / "sim_log.csv",
            stats_csv=run_dir / "sim_stats.csv",
            out_dir=run_dir,
        )

        # ---- 1. Process-state ----------------------------------------
        runs_PS.append(
            ev.evaluate(
                "process_state",
                io_obj,
                ps_runner,
                cut=cut,
                end=end,
                A_event=A_event,
                A_ongoing=A_ongoing,
                A_complete=A_complete,
                runner_kwargs=dict(
                    event_log=cfg.alog,
                    bpmn_model=cfg.model,
                    bpmn_parameters=cfg.params,
                    start_time=cut,
                    simulation_horizon=end + (end - cut),
                    total_cases=cfg.total_cases,
                    column_mapping={v: k for k, v in SIM_RENAME_MAP.items()},
                    rename_map=SIM_RENAME_MAP,
                ),
            )
        )

        # ---- 2. Warm-up --------------------------------------------
        runs_WU.append(
            ev.evaluate(
                "warmup",
                io_obj,
                wu_runner,
                cut=cut,
                end=end,
                A_event=A_event,
                A_ongoing=A_ongoing,
                A_complete=A_complete,
                runner_kwargs=dict(
                    bpmn_model=cfg.model,
                    json_sim_params=cfg.params,
                    total_cases=cfg.total_cases,
                    start_date=str((cut - pd.Timedelta(days=cfg.horizon_days)).isoformat()),
                    rename_map=SIM_RENAME_MAP,
                ),
            )
        )


        # ---- 3. Warm-up v2 ------------------------------------------
        runs_WU2.append(
            ev.evaluate(
                "warmup2",
                io_obj,
                wu_runner,
                cut=cut,
                end=end,
                A_event=A_event,
                A_ongoing=A_ongoing,
                A_complete=A_complete,
                runner_kwargs=dict(
                    bpmn_model=cfg.model,
                    json_sim_params=cfg.params,
                    total_cases=cfg.total_cases,
                    start_date=str((cut - pd.Timedelta(days=cfg.horizon_days)).isoformat()),
                    rename_map=SIM_RENAME_MAP,
                ),
            )
        )


    # 4-D  aggregate & save
    agg_PS = ev.aggregate(runs_PS)
    agg_WU = ev.aggregate(runs_WU)
    agg_WU2 = ev.aggregate(runs_WU2)

    summary = {
        "num_runs": runs,
        "process_state": {"aggregated": agg_PS},
        "warmup": {"aggregated": agg_WU},
        "warmup2": {"aggregated": agg_WU2},
        "PS_vs_WU": ev.compare(agg_PS, agg_WU, ("process_state", "warmup")),
        "PS_vs_WU2": ev.compare(agg_PS, agg_WU2, ("process_state", "warmup2")),
    }

    with open(out_base / "final_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


# ────────────────────────────────────────────────────────────────────
# 5.  CLI entry-point
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=DATASETS.keys(),
                   help="Key from the DATASETS catalogue")
    p.add_argument("--runs", type=int, default=10,
                   help="Number of repetitions (default: 10)")
    cli = p.parse_args()
    main(dataset=cli.dataset, runs=cli.runs)
