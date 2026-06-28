"""Shared infrastructure for end-to-end Prosimos flows.

Unlike the rest of the suite (pure-function unit tests), every test under
``tests/e2e/`` exercises the REAL pipeline:

    full basic Prosimos sim  ->  ground-truth log
        ->  cut into a prefix (Prosimos/InputHandler schema, ongoing acts left open)
        ->  run_process_state_and_simulation(simulate=True)
                = compute partial-state snapshot + resume a short-term Prosimos run

These are slower than unit tests (each generation + resume is ~0.2-0.5s) so they
are marked ``e2e`` (see ``pytest.ini``). Run only them with ``pytest -m e2e``;
skip them with ``pytest -m "not e2e"``.

The helpers below are deliberately generic so each test can vary one knob
(dataset, cutoff, horizon, attribute declaration, branch rules) while reusing
the same plumbing.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from src.process_state_prosimos_run import run_basic_simulation
from src.runner import run_process_state_and_simulation

REPO_ROOT = Path(__file__).resolve().parents[2]
DEV = REPO_ROOT / "samples" / "dev-samples"

XOR_BPMN = DEV / "synthetic_xor_loop.bpmn"
XOR_PARAMS = DEV / "synthetic_xor_loop.json"
ROUTE_BPMN = DEV / "synthetic_case_route.bpmn"
ROUTE_PARAMS = DEV / "synthetic_case_route.json"

# Prosimos output column -> InputHandler "standard" column name.
PROSIMOS_TO_INPUT = {
    "case_id": "CaseId",
    "activity": "Activity",
    "resource": "Resource",
    "start_time": "StartTime",
    "end_time": "EndTime",
}

# All assets must exist or the whole e2e module is meaningless; skip cleanly.
_assets_present = all(
    p.exists() for p in (XOR_BPMN, XOR_PARAMS, ROUTE_BPMN, ROUTE_PARAMS)
)
pytestmark = pytest.mark.skipif(
    not _assets_present, reason="dev-sample BPMN/params not available"
)


# --------------------------------------------------------------------------- #
# Plain helpers (importable from tests)                                        #
# --------------------------------------------------------------------------- #
def basic_sim(bpmn: Path, params: Path, n: int, out_dir: Path, name: str = "gt") -> pd.DataFrame:
    """Run one full Prosimos sim; return the GT log as a tz-aware DataFrame."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_csv = out_dir / f"{name}.csv"
    run_basic_simulation(
        bpmn_model=str(bpmn),
        json_sim_params=str(params),
        total_cases=n,
        out_stats_csv_path=str(out_dir / f"{name}_stats.csv"),
        out_log_csv_path=str(log_csv),
    )
    df = pd.read_csv(log_csv)
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
    df["end_time"] = pd.to_datetime(df["end_time"], utc=True)
    return df


def fraction_cutoff(gt: pd.DataFrame, fraction: float = 0.5) -> pd.Timestamp:
    """A cutoff timestamp at ``fraction`` of the log's [min start, max end] span."""
    lo, hi = gt["start_time"].min(), gt["end_time"].max()
    return lo + (hi - lo) * fraction


def ongoing_case_ids(gt: pd.DataFrame, cutoff: pd.Timestamp) -> set:
    """Cases with an activity straddling the cutoff (started before, ends after)."""
    m = (gt["start_time"] < cutoff) & ((gt["end_time"] >= cutoff) | gt["end_time"].isna())
    return set(gt.loc[m, "case_id"].unique())


def finished_case_ids(gt: pd.DataFrame, cutoff: pd.Timestamp) -> set:
    """Cases whose every event ends strictly before the cutoff."""
    last_end = gt.groupby("case_id")["end_time"].max()
    return set(last_end[last_end < cutoff].index)


def write_prefix(gt: pd.DataFrame, cutoff: pd.Timestamp, out_path: Path,
                 attr_cols: tuple[str, ...] = ()) -> pd.DataFrame:
    """Write events with start_time < cutoff in InputHandler schema.

    Extra attribute columns are carried through un-renamed so the runner can
    restore them into the snapshot. EndTime is left as-is; the runner clips
    end-times beyond the cutoff to "ongoing" internally.
    """
    cols = list(PROSIMOS_TO_INPUT) + [c for c in attr_cols if c in gt.columns]
    prefix = gt.loc[gt["start_time"] < cutoff, cols].copy()
    prefix = prefix.rename(columns=PROSIMOS_TO_INPUT)
    for c in ("StartTime", "EndTime"):
        prefix[c] = prefix[c].dt.strftime("%Y-%m-%dT%H:%M:%S.%f%z")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    prefix.to_csv(out_path, index=False)
    return prefix


def compute_snapshot(*, prefix_csv: Path, bpmn: Path, params: Path,
                     cutoff: pd.Timestamp, work_dir: Path) -> dict:
    """Run only the process-state computation (simulate=False); return the snapshot."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    prev = os.getcwd()
    os.chdir(work_dir)  # runner writes output.json to cwd
    try:
        run_process_state_and_simulation(
            event_log=str(prefix_csv), bpmn_model=str(bpmn), bpmn_parameters=str(params),
            start_time=cutoff.isoformat(), simulate=False,
        )
        return json.loads((work_dir / "output.json").read_text())
    finally:
        os.chdir(prev)


def run_short_term(*, prefix_csv: Path, bpmn: Path, params: Path, cutoff: pd.Timestamp,
                   horizon_end: pd.Timestamp, work_dir: Path,
                   total_cases: int = 400) -> tuple[dict, pd.DataFrame]:
    """Resume a short-term sim from the prefix; return (snapshot dict, sim log df)."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    sim_log = work_dir / "sim_log.csv"
    prev = os.getcwd()
    os.chdir(work_dir)
    try:
        run_process_state_and_simulation(
            event_log=str(prefix_csv), bpmn_model=str(bpmn), bpmn_parameters=str(params),
            start_time=cutoff.isoformat(), simulate=True,
            simulation_horizon=horizon_end.isoformat(), total_cases=total_cases,
            sim_stats_csv=str(work_dir / "sim_stats.csv"), sim_log_csv=str(sim_log),
        )
        snapshot = json.loads((work_dir / "output.json").read_text())
    finally:
        os.chdir(prev)
    sim = pd.read_csv(sim_log)
    sim["start_time"] = pd.to_datetime(sim["start_time"], utc=True)
    sim["end_time"] = pd.to_datetime(sim["end_time"], utc=True)
    return snapshot, sim


def augment_all_attribute_families(base_params_path: Path, out_path: Path) -> dict:
    """Write a copy of ``base_params_path`` declaring all three attribute families.

    case_type (case attr) already exists in the case-route params; this adds a
    discrete global ``season`` and a discrete event ``score`` on every task, so a
    single GT sim emits columns for all three families.
    """
    params = json.loads(Path(base_params_path).read_text())
    params["global_attributes"] = [{
        "name": "season", "type": "discrete",
        "values": [{"key": "winter", "value": 0.5}, {"key": "summer", "value": 0.5}],
    }]
    task_ids = [t["task_id"] for t in params["task_resource_distribution"]]
    params["event_attributes"] = [{
        "event_id": tid,
        "attributes": [{
            "name": "score", "type": "discrete",
            "values": [{"key": "low", "value": 0.5}, {"key": "high", "value": 0.5}],
        }],
    } for tid in task_ids]
    Path(out_path).write_text(json.dumps(params, indent=2))
    return params


# --------------------------------------------------------------------------- #
# Session-scoped ground-truth logs (generated once, reused across tests)       #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def xor_gt(tmp_path_factory) -> pd.DataFrame:
    d = tmp_path_factory.mktemp("xor_gt")
    return basic_sim(XOR_BPMN, XOR_PARAMS, n=200, out_dir=d)


@pytest.fixture(scope="session")
def route_gt(tmp_path_factory) -> pd.DataFrame:
    """Case-route GT (carries a real case_type column), base 50/50 routing."""
    d = tmp_path_factory.mktemp("route_gt")
    return basic_sim(ROUTE_BPMN, ROUTE_PARAMS, n=250, out_dir=d)


@pytest.fixture(scope="session")
def all_attrs(tmp_path_factory) -> dict:
    """Params declaring all 3 families + a GT log carrying their columns.

    Returns {"params": Path, "gt": DataFrame}.
    """
    d = tmp_path_factory.mktemp("all_attrs")
    params = d / "params_all_attrs.json"
    augment_all_attribute_families(ROUTE_PARAMS, params)
    gt = basic_sim(ROUTE_BPMN, params, n=250, out_dir=d, name="gt_all_attrs")
    return {"params": params, "gt": gt}
