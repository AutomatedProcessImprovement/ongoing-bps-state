# ────────────────────────────────────────────────────────────────
#  src/process_state_prosimos_run.py
# ────────────────────────────────────────────────────────────────
"""
Utility wrapper around Prosimos' `run_simulation`:
• run_short_term_simulation → with partial-state + horizon
• run_basic_simulation      → plain Prosimos run
The module also exposes a CLI (`python -m …`).
"""

from __future__ import annotations
import argparse
import datetime as _dt
import json
from pathlib import Path
import sys
from typing import Optional

from prosimos.simulation_engine import run_simulation

# -----------------------------------------------------------------
# 1) a tolerant parse_datetime and a global monkey-patch            #
# -----------------------------------------------------------------
def parse_datetime(dt_str: Optional[str], has_date: bool | None = None):
    """
    Convert an ISO-8601 string (optionally ending with 'Z') into a
    timezone-aware `datetime`.  The *has_date* parameter is ignored; it
    exists only so the signature matches Prosimos 1.4.x.
    """
    if not dt_str:
        return None
    return _dt.datetime.fromisoformat(dt_str.replace("Z", "+00:00"))


import sys
for _name, _mod in sys.modules.items():
    if _name.startswith("prosimos") and hasattr(_mod, "parse_datetime"):
        _mod.parse_datetime = parse_datetime

# -----------------------------------------------------------------
# 2) helpers                                                       #
# -----------------------------------------------------------------
def _dt_now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def _iso_or_none(ts: Optional[str]) -> Optional[_dt.datetime]:
    return parse_datetime(ts) if ts else None


def _load_process_state(path: Optional[str]):
    if not path:
        return None
    with open(path, "r") as fh:
        ps = json.load(fh)

    # Convert enabled_time / start_time strings to datetime
    for case in ps.get("cases", {}).values():
        for col in ("enabled_activities", "ongoing_activities"):
            for act in case.get(col, []):
                for key in ("enabled_time", "start_time"):
                    if isinstance(act.get(key), str):
                        act[key] = parse_datetime(act[key])
    return ps


# -----------------------------------------------------------------
# 3) two public functions                                           #
# -----------------------------------------------------------------
def run_short_term_simulation(
    *,
    start_date: str | _dt.datetime | None,
    total_cases: int,
    bpmn_model: str | Path,
    json_sim_params: str | Path,
    out_stats_csv_path: str | Path,
    out_log_csv_path: str | Path,
    process_state: dict | None,
    simulation_horizon: str | _dt.datetime | None,
) -> float:
    """Prosimos run with *partial-state* + finite horizon."""
    t0 = _dt_now()

    run_simulation(
        bpmn_path=str(bpmn_model),
        json_path=str(json_sim_params),
        total_cases=total_cases,
        stat_out_path=str(out_stats_csv_path),
        log_out_path=str(out_log_csv_path),
        starting_at=start_date,
        process_state=process_state,
        simulation_horizon=simulation_horizon,
    )

    return (_dt_now() - t0).total_seconds()


def run_basic_simulation(
    *,
    bpmn_model: str | Path,
    json_sim_params: str | Path,
    total_cases: int,
    out_stats_csv_path: str | Path,
    out_log_csv_path: str | Path,
    start_date: str | _dt.datetime | None = None,
) -> float:
    """Plain Prosimos run (no partial-state, no horizon)."""
    t0 = _dt_now()

    run_simulation(
        bpmn_path=str(bpmn_model),
        json_path=str(json_sim_params),
        total_cases=total_cases,
        stat_out_path=str(out_stats_csv_path),
        log_out_path=str(out_log_csv_path),
        starting_at=start_date,
        process_state=None,
        simulation_horizon=None,
    )

    return (_dt_now() - t0).total_seconds()


# -----------------------------------------------------------------
# 4) CLI entry-point (unchanged interface)                          #
# -----------------------------------------------------------------
def _cli():
    ap = argparse.ArgumentParser(
        description="Run Prosimos, optionally with partial-state + horizon."
    )
    ap.add_argument("--bpmn_model", required=True)
    ap.add_argument("--sim_json", required=True)
    ap.add_argument("--process_state")
    ap.add_argument("--simulation_horizon")
    ap.add_argument("--start_time")
    ap.add_argument("--total_cases", type=int, default=20)
    ap.add_argument("--out_stats_csv", default="simulation_stats.csv")
    ap.add_argument("--log_csv", default="simulation_log.csv")
    args = ap.parse_args()

    start_dt = _iso_or_none(args.start_time)
    horizon_dt = _iso_or_none(args.simulation_horizon)
    ps = _load_process_state(args.process_state)

    if ps and horizon_dt:
        print("→ short-term simulation (partial-state + horizon)")
        secs = run_short_term_simulation(
            start_date=start_dt or _dt_now(),
            total_cases=args.total_cases,
            bpmn_model=args.bpmn_model,
            json_sim_params=args.sim_json,
            out_stats_csv_path=args.out_stats_csv,
            out_log_csv_path=args.log_csv,
            process_state=ps,
            simulation_horizon=horizon_dt,
        )
    else:
        print(f"→ standard Prosimos run (total_cases={args.total_cases})")
        secs = run_basic_simulation(
            bpmn_model=args.bpmn_model,
            json_sim_params=args.sim_json,
            total_cases=args.total_cases,
            out_stats_csv_path=args.out_stats_csv,
            out_log_csv_path=args.log_csv,
            start_date=start_dt,
        )

    print(f"Simulation finished in {secs:.2f} s → {args.log_csv}")


if __name__ == "__main__":
    _cli()
