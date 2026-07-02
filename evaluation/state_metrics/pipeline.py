"""Evaluation pipeline for the state-based metrics.

Design (revised 2026-05-18): one basic Prosimos run produces the *prefix*
only. Every downstream simulation — both the **baseline** (zero-perturbation
reference) and the **system under test** at level N — is a short-term sim
initialized from that shared prefix.

Stochastic L=0 baseline: at every level (including 0) we run ``cfg.runs``
independent short-term sims and compare each to one of the ``cfg.runs``
baseline draws. At level 0 both sides use the unperturbed params, so the
resulting distance reflects sim-to-sim noise — the real noise floor of
each metric under the current windowing. Prior versions (2026-05 lock)
reused the baseline log as the sim side at L=0, forcing distance=0 by
construction; that gave every level > 0 a free signal-vs-zero comparison
and masked the noise floor.

Window selection is configurable (added 2026-05-18 to compare across
utilization regimes):
    * ``cutoff_strategy``:
        - "p90_wip" — middle of top-decile WIP band (legacy).
        - "fraction" — timestamp at ``cutoff_fraction`` of the log span.
    * ``horizon_hours``:
        - None — adaptive ``2 × mean case duration`` (legacy).
        - float — fixed hour count, clipped to log end.
Use the fixed/fraction combo for cross-dataset comparisons; the legacy
adaptive setup makes windows incomparable across utilization regimes
because mean case duration moves with queue depth.

For each perturbation level N we collect ``cfg.runs`` baseline/sim draws
and compute one metric per (baseline_k, sim_k) pair. Pairing by index
``k`` is *not* seed-matched — Prosimos does not expose deterministic RNG
control, so each run is an independent draw. The downstream ranking
analysis (see ``ranking.py``) is built around this: ``concordance_index``
works directly over cross-replicate pairs and does not assume paired
observations.

Metrics are reported under two scopes:
    * Scope A — ongoing cases only (IDs match semantically).
    * Scope B — ongoing + new arrivals (new-arrival IDs match by Prosimos'
      sequential numbering; distance interpretation is by arrival order).

After per-pair metrics are written to ``results.csv``, the pipeline computes
ranking-quality statistics (c-index, Spearman ρ, Kendall τ, each with
bootstrap CIs) per (scope, family, metric) and writes ``rankings.csv``
alongside.
"""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.helper import generate_short_uuid, read_event_log
from evaluation.state_metrics.perturb import (
    build_all_calendars_shifted_params,
    build_arrival_burstier_params,
    build_branch_automation_params,
    build_calendar_shifted_params,
    build_case_route_params,
    build_duration_scaled_params,
    build_front_back_load_params,
    build_gateway_biased_params,
    build_perturbed_params,
    build_role_swap_params,
)
from src.process_state_prosimos_run import run_basic_simulation
from src.runner import run_process_state_and_simulation
from state_metrics import compute_all_state_distances


# Prosimos emits camel-case columns; our canonical schema is snake-case.
PROSIMOS_TO_CANONICAL = {
    "CaseId": "case_id",
    "Activity": "activity",
    "Resource": "resource",
    "StartTime": "start_time",
    "EndTime": "end_time",
    "EnabledTime": "enable_time",
}
CANONICAL_TO_PROSIMOS = {v: k for k, v in PROSIMOS_TO_CANONICAL.items()}


@dataclass
class PipelineConfig:
    dataset_name: str
    bpmn_path: Path
    params_path: Path
    remove_from_profile: str
    outputs_root: Path
    runs: int
    levels: tuple[int, ...]
    # Perturbation family — controls how `level` is interpreted by
    # `_prepare_params_for_level`. Magnitude families:
    #   "resources" — add/remove N resources from `remove_from_profile`.
    #   "duration"  — scale every duration distribution by (1 + level/100).
    # Structural families (added 2026-05 to expose state metrics' WHAT/WHEN
    # advantage where cycle_time is blind):
    #   "role_swap"      — route N tasks from `role_swap_from` (defaults to
    #                      `remove_from_profile`) to `role_swap_to`.
    #   "calendar_shift" — shift the calendar of `calendar_shift_profile`
    #                      (defaults to `remove_from_profile`) by `level`
    #                      hours.
    #   "gateway"        — bias the most-balanced (or `gateway_id`) gateway
    #                      by `level` * 0.1 toward its majority path.
    #   "arrival_burst"  — multiply arrival variance by (1 + level/100),
    #                      keeping mean (i.e. throughput) constant.
    # Oracle family (added 2026-05 as a zero-confound existence proof):
    #   "relabel"        — reuse ONE sim per replicate and relabel `level`%
    #                      of a chosen activity's instances to a new label.
    #                      Timestamps/cases/resources are untouched, so every
    #                      timing-based baseline (cycle_time, rtd, red) is
    #                      invariant by construction and the L=0 noise floor
    #                      is exactly 0. `level` is read as a percentage.
    #   "rephase"        — reuse ONE sim per replicate and shift each case's
    #                      entire timeline by a per-case uniform offset on
    #                      [-`level`h, +`level`h], clamped to keep events in
    #                      [cutoff, horizon_end). Activity/case/resource
    #                      labels are untouched and per-case duration, order,
    #                      bigrams and relative event positions are preserved
    #                      exactly. cycle_time/ngd/red are 0 at every level;
    #                      cross-case state metrics carry the full signal.
    #                      `level` is the jitter magnitude in hours.
    perturbation: str = "resources"
    gt_total_cases: int = 500
    sim_total_cases: int = 500
    # Structural-perturbation kwargs. Each is consulted only by the matching
    # `perturbation` family; ignored otherwise.
    role_swap_to: str | None = None         # required for role_swap
    calendar_shift_profile: str | None = None  # defaults to remove_from_profile
    gateway_id: str | None = None           # None → auto-pick most-balanced
    # relabel kwargs.
    relabel_activity: str | None = None     # None → most frequent in the window
    relabel_to: str | None = None           # None → f"{activity}__alt"
    # mix_ratio kwargs. Two param files for the green/red populations and the
    # baseline mix to compare against. `level` is interpreted as a percentage-
    # point shift FROM `mix_baseline_green` toward more green.
    mix_green_params: Path | None = None    # required for mix_ratio
    mix_red_params: Path | None = None      # required for mix_ratio
    mix_baseline_green: float = 0.5         # baseline fraction of green cases
    # case_route kwargs. The reference is a real short-term sim whose first
    # `case_route_ruled` XOR splits route by case_type (None -> all splits);
    # `level` is then the percentage of case_type tags swapped on that
    # reference (the pairing-unique attack on a genuinely attribute-routed log).
    case_route_ruled: int | None = None
    # parallel_auto kwargs (scenario #2). The non-critical AND-branch tasks to
    # automate; `level` is read as the percent shrink toward `automation_floor`.
    automate_task_ids: tuple[str, ...] | None = None
    automation_floor_seconds: float = 1.0
    # front_back_load kwargs (scenario #4). The ordered chain tasks whose mean
    # durations are reweighted total-invariantly; `load_direction` decides the
    # sign so the (non-negative) `level` ladder stays monotone for ranking.
    chain_task_ids: tuple[str, ...] | None = None
    load_direction: str = "front"           # "front" or "back"
    # Windowing controls (added 2026-05 to enable cross-utilization comparison).
    # cutoff_strategy:
    #   "p90_wip"   — middle of top-decile WIP band (legacy default).
    #   "fraction"  — timestamp at cutoff_fraction of the log's [min,max] span.
    #   "n_ongoing" — timestamp where the number of cases active at the cutoff
    #                 matches ``target_ongoing``. Matches multiset size across
    #                 utilization regimes, making cross-dataset Scope A
    #                 comparisons fair.
    # horizon_hours:
    #   None  — adaptive (2 × mean case duration, clipped to log end). Legacy.
    #   float — fixed-hour horizon, clipped to log end. Use this for
    #           apples-to-apples comparison across utilization regimes.
    cutoff_strategy: str = "p90_wip"
    cutoff_fraction: float = 0.5
    target_ongoing: int = 30
    horizon_hours: float | None = None


def _load_prosimos_log(csv_path: Path) -> pd.DataFrame:
    """Read a Prosimos output CSV, returning canonical snake-case columns.

    Keeps a ``case_type`` column when the simulation logged one (i.e. the params
    declared a ``case_type`` case attribute). Carrying it through lets the
    state-metrics ``case_type`` / ``activity_type`` projections fire on real
    attribute-driven runs (the case_route oracle) instead of only on the
    post-hoc merge oracles that inject case_type by hand.
    """
    df = read_event_log(str(csv_path), rename=PROSIMOS_TO_CANONICAL)
    keep = ["case_id", "activity", "start_time", "end_time", "resource"]
    if "case_type" in df.columns:
        keep.append("case_type")
    return df[keep].copy()


def build_basic_sim_for_prefix(cfg: PipelineConfig) -> Path:
    """Run a full Prosimos simulation once; return the log CSV path.

    This log is only used to derive a realistic prefix (everything before
    the cutoff). It is *not* the ground truth any more — the short-term sim
    at level 0 takes that role.
    """
    gt_dir = cfg.outputs_root / cfg.dataset_name
    gt_dir.mkdir(parents=True, exist_ok=True)
    prefix_source_csv = gt_dir / "prefix_source_log.csv"
    prefix_source_stats = gt_dir / "prefix_source_stats.csv"

    if prefix_source_csv.exists():
        return prefix_source_csv
    # Backward compat: reuse an older run's gt_log.csv if present.
    legacy = gt_dir / "gt_log.csv"
    if legacy.exists():
        return legacy

    run_basic_simulation(
        bpmn_model=cfg.bpmn_path,
        json_sim_params=cfg.params_path,
        total_cases=cfg.gt_total_cases,
        out_stats_csv_path=prefix_source_stats,
        out_log_csv_path=prefix_source_csv,
    )
    return prefix_source_csv


def _select_high_wip_cutoff(log: pd.DataFrame, quantile: float = 0.90) -> pd.Timestamp:
    """Pick the middle timestamp in the top-quantile WIP band.

    Targeting high WIP (default: 90th percentile) rather than median gives
    the perturbed simulations something to queue on; otherwise a resource
    reduction has no visible effect because there was no contention.
    """
    case_bounds = log.groupby("case_id").agg(
        first=("start_time", "min"),
        last=("end_time", "max"),
    )
    probe_times = sorted(set(case_bounds["first"]) | set(case_bounds["last"]))

    def wip(ts: pd.Timestamp) -> int:
        return int(((case_bounds["first"] <= ts) & (case_bounds["last"] > ts)).sum())

    wip_series = pd.Series({t: wip(t) for t in probe_times})
    target = wip_series.quantile(quantile)
    candidates = wip_series[wip_series >= target]
    return candidates.index[len(candidates) // 2]


def _compute_horizon(log: pd.DataFrame, cutoff: pd.Timestamp) -> pd.Timedelta:
    case_durations = (
        log.groupby("case_id")
        .agg(first=("start_time", "min"), last=("end_time", "max"))
        .assign(dur=lambda x: x["last"] - x["first"])["dur"]
    )
    h = 2 * case_durations.mean()
    # Clip horizon so at least a few ongoing cases can finish naturally.
    log_end = log["end_time"].max()
    room = log_end - cutoff
    if room <= pd.Timedelta(0):
        raise ValueError("cutoff at or past the end of the log")
    return min(h, room)


def _select_n_ongoing_cutoff(log: pd.DataFrame, target: int) -> pd.Timestamp:
    """Pick the timestamp where the number of ongoing cases ~= ``target``.

    Searches event-boundary timestamps for those whose WIP is within a small
    band of the target (``±max(2, 0.2*target)``); among matches, returns the
    median by time. Falls back to the single closest match if no timestamp
    sits in the band.

    Designed for cross-utilization comparison: by anchoring on the number
    of ongoing cases rather than wall-clock fraction, the active-instance
    multiset has comparable cardinality across regimes, so Scope A state
    metrics are not dominated by multiset-size effects.
    """
    if target <= 0:
        raise ValueError("target must be > 0")
    case_bounds = log.groupby("case_id").agg(
        first=("start_time", "min"),
        last=("end_time", "max"),
    )
    probe_times = sorted(set(case_bounds["first"]) | set(case_bounds["last"]))
    wips = pd.Series(
        {t: int(((case_bounds["first"] <= t) & (case_bounds["last"] > t)).sum())
         for t in probe_times}
    )
    tol = max(2, int(0.2 * target))
    band = wips[(wips >= target - tol) & (wips <= target + tol)]
    if not band.empty:
        # Median-by-time within the band — stable interior point.
        sorted_band = band.sort_index()
        return sorted_band.index[len(sorted_band) // 2]
    diff = (wips - target).abs()
    return diff.idxmin()


def _select_fraction_of_log_cutoff(
    log: pd.DataFrame, fraction: float = 0.5
) -> pd.Timestamp:
    """Pick the timestamp at ``fraction`` of the log's [min, max] span.

    Cross-utilization comparisons need a cutoff that does not bias toward
    high-load regions of the log. p90_wip lands in the middle of saturation
    for a 95% util log but at a brief peak in a 70% util log, so windows are
    not comparable. Anchoring on a fraction of wall-clock time gives a
    stable reference point that is monotone in the log.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be in (0, 1)")
    start = log["start_time"].min()
    end = log["end_time"].max()
    return start + (end - start) * fraction


def get_ongoing_case_ids(log: pd.DataFrame, cutoff: pd.Timestamp) -> set[str]:
    """Case IDs whose first event started before cutoff and last event ends after."""
    case_bounds = log.groupby("case_id").agg(
        first=("start_time", "min"),
        last=("end_time", "max"),
    )
    ids = case_bounds.loc[
        (case_bounds["first"] < cutoff) & (case_bounds["last"] > cutoff)
    ].index
    return {str(x) for x in ids}


def filter_to_cases_in_window(
    log: pd.DataFrame,
    case_ids: set[str] | None,
    cutoff: pd.Timestamp,
    horizon: pd.Timedelta,
) -> pd.DataFrame:
    """Events clipped to [cutoff, cutoff+horizon).

    If ``case_ids`` is None, include all cases (Scope B — ongoing + new
    arrivals). If a set is given, restrict to those IDs (Scope A — ongoing
    cases only; IDs are prefix-derived so they match across GT and sim by
    construction under short-term simulation).
    """
    end = cutoff + horizon
    mask = (log["end_time"] > cutoff) & (log["start_time"] < end)
    if case_ids is not None:
        mask &= log["case_id"].astype(str).isin(case_ids)
    sub = log.loc[mask].copy()
    sub.loc[sub["start_time"] < cutoff, "start_time"] = cutoff
    sub.loc[sub["end_time"] > end, "end_time"] = end
    return sub


def merge_logs_at_ratio(
    green_log: pd.DataFrame,
    red_log: pd.DataFrame,
    *,
    fraction_green: float,
    target_cases: int,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a merged log: ``fraction_green`` of ``target_cases`` are taken
    from ``green_log``, the rest from ``red_log``. Each case is tagged with
    ``case_type`` ∈ {"green", "red"}; case IDs are namespaced with "g_"/"r_"
    prefixes so collisions are impossible. Events are sorted by start_time.

    Selection: case IDs in each source are randomly permuted (seeded by
    ``seed``) and the first ``round(fraction_green * target_cases)`` of the
    green permutation plus the first ``target_cases − n_green`` of the red
    permutation are kept. Two calls with the same inputs and same ``seed``
    are identical, which is what makes the level-0 reference vs level-0
    perturbed comparison in the mix_ratio oracle have an exactly-zero noise
    floor.

    A randomised permutation (rather than picking the temporally earliest N)
    is required because cases ongoing at the cutoff are concentrated in the
    early part of the log: a "first-N-by-start-time" rule picks the same
    early cases at every ratio, so the windowed ongoing-case set never
    changes. Random permutation spreads the swapped-in / swapped-out cases
    across the timeline.

    Used by the mix_ratio oracle to vary the green/red population mix
    without re-simulating: green and red are two independent basic Prosimos
    runs (typically same BPMN, different params).
    """
    if not 0.0 <= fraction_green <= 1.0:
        raise ValueError("fraction_green must be in [0, 1]")
    if target_cases <= 0:
        raise ValueError("target_cases must be > 0")
    n_green = int(round(fraction_green * target_cases))
    n_red = target_cases - n_green

    green = green_log.copy()
    red = red_log.copy()
    green["case_id"] = "g_" + green["case_id"].astype(str)
    red["case_id"] = "r_" + red["case_id"].astype(str)

    g_cases = sorted(green["case_id"].unique())
    r_cases = sorted(red["case_id"].unique())
    if n_green > len(g_cases) or n_red > len(r_cases):
        raise ValueError(
            f"not enough source cases: requested green={n_green} (have "
            f"{len(g_cases)}), red={n_red} (have {len(r_cases)})"
        )
    rng = np.random.default_rng(seed)
    # Use disjoint RNG branches for green and red so the green permutation
    # doesn't depend on the red case count (lets you sweep target_cases
    # without resampling green).
    g_perm = rng.permutation(len(g_cases))
    r_perm = np.random.default_rng(seed + 1).permutation(len(r_cases))
    g_pick = {g_cases[i] for i in g_perm[:n_green]}
    r_pick = {r_cases[i] for i in r_perm[:n_red]}

    g_sel = green[green["case_id"].isin(g_pick)].copy()
    r_sel = red[red["case_id"].isin(r_pick)].copy()
    g_sel["case_type"] = "green"
    r_sel["case_type"] = "red"

    merged = pd.concat([g_sel, r_sel], ignore_index=True)
    merged = merged.sort_values("start_time").reset_index(drop=True)
    return merged


def rephase_cases(
    log: pd.DataFrame,
    *,
    jitter_hours: float,
    rng: np.random.Generator,
    cutoff: pd.Timestamp,
    horizon_end: pd.Timestamp,
) -> pd.DataFrame:
    """Return a copy of ``log`` with each case's events shifted by a per-case
    random offset Δ_c.

    Every event in case ``c`` is shifted by the same Δ_c, so per-case
    duration, ordering, bigrams and relative event positions are preserved
    exactly. Cycle-time, ngd, and red distances are therefore identically
    zero at every level by construction; only the cross-case timing pattern
    (active-instance multisets at each timestamp) moves, which is what the
    state metrics are designed to detect.

    Offsets are drawn uniformly per case from ``[-max_back, +max_fwd]`` where
    ``max_back = min(jitter_hours, case_start − cutoff)`` and ``max_fwd =
    min(jitter_hours, horizon_end − case_end)``. Clamping to feasible per-case
    bounds keeps the entire trajectory inside the window so that
    window-clipping never breaks the per-case invariants. Cases for which
    neither direction has room are passed through unchanged.

    At ``jitter_hours == 0`` the output is identical to the input.
    """
    out = log.copy()
    if jitter_hours <= 0 or out.empty:
        return out
    bounds = out.groupby("case_id").agg(
        first=("start_time", "min"), last=("end_time", "max"),
    )
    jitter_s = float(jitter_hours) * 3600.0
    case_to_offset: dict[str, float] = {}
    for cid, row in bounds.iterrows():
        max_back_s = min(jitter_s, max(0.0, (row["first"] - cutoff).total_seconds()))
        max_fwd_s = min(jitter_s, max(0.0, (horizon_end - row["last"]).total_seconds()))
        if max_back_s + max_fwd_s <= 0.0:
            continue
        case_to_offset[cid] = float(rng.uniform(-max_back_s, max_fwd_s))
    if not case_to_offset:
        return out
    deltas = out["case_id"].map(case_to_offset).fillna(0.0).to_numpy()
    offsets = pd.to_timedelta(deltas, unit="s")
    out["start_time"] = out["start_time"] + offsets
    out["end_time"] = out["end_time"] + offsets
    return out


def relabel_activity_fraction(
    log: pd.DataFrame,
    *,
    activity: str,
    to_label: str,
    fraction: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Return a copy of ``log`` with ``fraction`` of ``activity`` rows relabeled.

    Only the ``activity`` column changes; case_id, timestamps, and resource are
    left untouched. This makes the transform a pure activity-composition shift:
    every timing-based metric (cycle_time, rtd, red) and the ``case`` /
    ``cardinality`` state projections are invariant by construction, while the
    ``activity`` / ``activity_case`` / ``activity_role`` projections move with
    ``fraction``. At ``fraction == 0`` the output is identical to the input, so
    a baseline-vs-relabeled comparison has an exactly-zero noise floor.

    Relabeling is per-instance (each matching event drawn independently) so the
    induced proportion shift is the clean ``fraction`` with no case correlation.
    """
    out = log.copy()
    if fraction <= 0:
        return out
    pos = np.flatnonzero((out["activity"] == activity).to_numpy())
    if pos.size == 0:
        return out
    n = int(round(fraction * pos.size))
    if n <= 0:
        return out
    chosen = rng.choice(pos, size=n, replace=False)
    out.iloc[chosen, out.columns.get_loc("activity")] = to_label
    return out


def _write_prefix_csv(log: pd.DataFrame, cutoff: pd.Timestamp, out_path: Path) -> None:
    """Write all events with start_time < cutoff, renamed to Prosimos columns.

    A ``case_type`` column, when present, is carried through un-renamed so the
    short-term runner can restore each ongoing case's historical attribute value
    into the partial-state snapshot (and thus keep attribute-driven routing
    consistent on resume) instead of re-sampling it.
    """
    cols = [c for c in CANONICAL_TO_PROSIMOS if c in log.columns]
    extra = ["case_type"] if "case_type" in log.columns else []
    prefix = log.loc[log["start_time"] < cutoff, cols + extra].copy()
    prefix = prefix.rename(columns={c: CANONICAL_TO_PROSIMOS[c] for c in cols})
    # Prosimos/InputHandler wants ISO-8601 strings with fractional seconds.
    for c in ("StartTime", "EndTime"):
        prefix[c] = prefix[c].dt.strftime("%Y-%m-%dT%H:%M:%S.%f%z")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prefix.to_csv(out_path, index=False)


def _run_one_sim(
    *,
    prefix_csv: Path,
    bpmn: Path,
    params: Path,
    cutoff: pd.Timestamp,
    horizon_end: pd.Timestamp,
    total_cases: int,
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    sim_log_csv = out_dir / "sim_log.csv"
    sim_stats_csv = out_dir / "sim_stats.csv"
    run_process_state_and_simulation(
        event_log=str(prefix_csv),
        bpmn_model=str(bpmn),
        bpmn_parameters=str(params),
        start_time=cutoff.isoformat(),
        simulate=True,
        simulation_horizon=horizon_end.isoformat(),
        total_cases=total_cases,
        sim_stats_csv=str(sim_stats_csv),
        sim_log_csv=str(sim_log_csv),
    )
    return sim_log_csv


def _compute_metrics_row(
    *,
    baseline_continuation: pd.DataFrame,
    sim_continuation: pd.DataFrame,
    level: int,
    k_baseline: int,
    k_sim: int,
    scope: str,
    window: tuple[pd.Timestamp, pd.Timestamp],
    role_map: dict[str, str] | None = None,
) -> list[dict]:
    from log_distance_measures.config import (
        AbsoluteTimestampType,
        EventLogIDs,
        discretize_to_hour,
    )
    from log_distance_measures.cycle_time_distribution import (
        cycle_time_distribution_distance,
    )
    from log_distance_measures.n_gram_distribution import n_gram_distribution_distance
    from log_distance_measures.relative_event_distribution import (
        relative_event_distribution_distance,
    )

    ids = EventLogIDs("case_id", "activity", "start_time", "end_time", "resource")
    rows: list[dict] = []

    # State-based metrics. *.cardinality is view-invariant by construction
    # (cardinality ignores the projection key), so emit it once.
    # Pass window=(cutoff, horizon_end) so the denominator is the full
    # evaluation window rather than the observed-events span. Both logs
    # are pre-clipped to that range upstream.
    # Pass role_map so the activity_role projection is actually computed.
    # Without a role_map, the state_metrics API silently drops activity_role
    # from the result set — and that's the projection designed to detect
    # role-swap perturbations (where activity X moves from Role A to Role B).
    # Resources in the sim log that aren't in the map (e.g. unknown sim-
    # internal IDs) would raise; cap by restricting to logged resources via
    # an inclusive default.
    sim_role_map = role_map
    if sim_role_map is not None:
        present = set(baseline_continuation["resource"].astype(str)) | set(
            sim_continuation["resource"].astype(str)
        )
        # If any resource appears in the logs but not in the map, fall back
        # to a synthetic role "unknown" rather than crashing.
        if any(r not in sim_role_map for r in present):
            sim_role_map = {
                **{r: "unknown" for r in present},
                **sim_role_map,
            }
    state_results = compute_all_state_distances(
        baseline_continuation, sim_continuation,
        role_map=sim_role_map, window=window,
    )
    emitted_cardinality = False
    for (view, dist), r in state_results.items():
        if dist == "cardinality":
            if emitted_cardinality:
                continue
            metric = "cardinality"
            emitted_cardinality = True
        else:
            metric = f"{view}.{dist}"
        rows.append({
            "level": level, "k_baseline": k_baseline, "k_sim": k_sim, "scope": scope,
            "family": "state", "metric": metric, "value": r.summary_distance,
        })

    def _safe(name, fn, *a, **kw):
        try:
            v = fn(*a, **kw)
        except Exception as e:
            print(f"[baseline {name}] failed: {e}")
            v = None
        rows.append({
            "level": level, "k_baseline": k_baseline, "k_sim": k_sim, "scope": scope,
            "family": "baseline", "metric": name, "value": v,
        })

    _safe(
        "ngd_n2",
        n_gram_distribution_distance,
        baseline_continuation, ids, sim_continuation, ids, n=2,
    )
    _safe(
        "red",
        relative_event_distribution_distance,
        baseline_continuation, ids, sim_continuation, ids,
        discretize_type=AbsoluteTimestampType.BOTH, discretize_event=discretize_to_hour,
    )
    _safe(
        "cycle_time",
        cycle_time_distribution_distance,
        baseline_continuation, ids, sim_continuation, ids,
        bin_size=pd.Timedelta(hours=1),
    )
    # RTD: remaining-time-distribution distance from the cutoff. Most
    # meaningful for Scope A (ongoing cases) but computable on any log.
    from evaluation.rtd import rtd as rtd_distance
    _safe(
        "rtd",
        rtd_distance,
        baseline_continuation, ids, sim_continuation, ids,
        reference_point=window[0],
        bin_size=pd.Timedelta(hours=1),
    )
    return rows


def _resource_to_profile(params: dict) -> dict[str, str]:
    """Map every resource_id to its profile name."""
    out: dict[str, str] = {}
    for prof in params.get("resource_profiles", []):
        name = prof.get("name")
        for r in prof.get("resource_list", []):
            out[r["id"]] = name
    return out


def _compute_utilization_rows(
    *,
    sim_log: pd.DataFrame,
    params: dict,
    cutoff: pd.Timestamp,
    horizon_end: pd.Timestamp,
    level: int,
    k_sim: int,
) -> list[dict]:
    """Wall-clock utilization per profile within [cutoff, horizon_end].

    Util = sum(clipped activity duration) over resources in the profile
    divided by (n_resources_in_profile × window_seconds). Calendar-off time
    is ignored — same convention used for the ad-hoc measurements during
    dataset calibration, so numbers are directly comparable.
    """
    window_secs = (horizon_end - cutoff).total_seconds()
    if window_secs <= 0:
        return []
    res_to_prof = _resource_to_profile(params)
    profile_counts: dict[str, int] = {}
    for prof in params.get("resource_profiles", []):
        profile_counts[prof.get("name")] = len(prof.get("resource_list", []))

    log = sim_log.copy()
    log = log[(log["end_time"] > cutoff) & (log["start_time"] < horizon_end)]
    if log.empty:
        return [
            {
                "level": level, "k_sim": k_sim, "role": prof_name,
                "n_resources": profile_counts.get(prof_name, 0),
                "busy_seconds": 0.0, "window_seconds": window_secs,
                "utilization": 0.0,
            }
            for prof_name in profile_counts
        ]
    log = log.assign(
        clipped_start=log["start_time"].clip(lower=cutoff),
        clipped_end=log["end_time"].clip(upper=horizon_end),
    )
    log["dur"] = (log["clipped_end"] - log["clipped_start"]).dt.total_seconds()
    log["profile"] = log["resource"].map(res_to_prof)

    busy_by_profile = (
        log.dropna(subset=["profile"]).groupby("profile")["dur"].sum().to_dict()
    )
    rows: list[dict] = []
    for prof_name, n_res in profile_counts.items():
        busy = float(busy_by_profile.get(prof_name, 0.0))
        denom = n_res * window_secs if n_res > 0 else float("nan")
        util = busy / denom if denom else float("nan")
        rows.append({
            "level": level, "k_sim": k_sim, "role": prof_name,
            "n_resources": n_res, "busy_seconds": busy,
            "window_seconds": window_secs, "utilization": util,
        })
    return rows


def _seed_world(seed: int) -> None:
    """Seed stdlib + numpy RNGs as a best-effort gesture.

    Prosimos does not expose deterministic RNG control, so two runs with the
    same ``seed`` are NOT guaranteed to produce identical outputs. We keep
    this for the parts of the pipeline that *are* deterministic (resource
    pruning, file naming) and to make the per-run RNG state at least
    namespaced. The ranking analysis downstream does not rely on pairing.
    """
    random.seed(seed)
    np.random.seed(seed)


def _prepare_params_for_level(
    cfg: PipelineConfig, level: int, run_dir: Path
) -> tuple[Path, dict]:
    if level == 0:
        base_manifest = {
            "perturbation": cfg.perturbation,
            "level": 0,
            "note": "baseline (no perturbation)",
        }
        return Path(cfg.params_path), base_manifest

    level_dir = run_dir / f"level_{level}"
    level_dir.mkdir(exist_ok=True)
    out_params = level_dir / "params.json"

    if cfg.perturbation == "resources":
        manifest = build_perturbed_params(
            cfg.params_path,
            remove_from_profile=cfg.remove_from_profile,
            n_to_remove=level,
            out_json_path=out_params,
        )
    elif cfg.perturbation == "duration":
        scale = 1.0 + level / 100.0
        manifest = build_duration_scaled_params(
            cfg.params_path,
            scale_factor=scale,
            out_json_path=out_params,
        )
    elif cfg.perturbation == "role_swap":
        if cfg.role_swap_to is None:
            raise ValueError("role_swap requires cfg.role_swap_to")
        if level < 0:
            raise ValueError("role_swap levels must be >= 0")
        manifest = build_role_swap_params(
            cfg.params_path,
            from_profile=cfg.remove_from_profile,
            to_profile=cfg.role_swap_to,
            n_activities=level,
            out_json_path=out_params,
        )
    elif cfg.perturbation == "calendar_shift":
        cal_profile = cfg.calendar_shift_profile or cfg.remove_from_profile
        manifest = build_calendar_shifted_params(
            cfg.params_path,
            profile_name=cal_profile,
            shift_hours=float(level),
            out_json_path=out_params,
        )
    elif cfg.perturbation == "calendar_shift_all":
        # Phase translation: shift every resource calendar + arrival calendar
        # together, so relative role availability (and cycle time) is
        # preserved while wall-clock alignment moves.
        manifest = build_all_calendars_shifted_params(
            cfg.params_path,
            shift_hours=float(level),
            out_json_path=out_params,
        )
    elif cfg.perturbation == "gateway":
        if level < 0:
            raise ValueError("gateway perturbation levels must be >= 0")
        manifest = build_gateway_biased_params(
            cfg.params_path,
            gateway_id=cfg.gateway_id,
            bias_level=level,
            out_json_path=out_params,
        )
    elif cfg.perturbation == "arrival_burst":
        if level < 0:
            raise ValueError(
                "arrival_burst levels must be >= 0 (encoded as CV² × (1+level/100))"
            )
        cv2_mult = 1.0 + level / 100.0
        manifest = build_arrival_burstier_params(
            cfg.params_path,
            cv2_multiplier=cv2_mult,
            out_json_path=out_params,
        )
    elif cfg.perturbation == "parallel_auto":
        if cfg.automate_task_ids is None:
            raise ValueError("parallel_auto requires cfg.automate_task_ids")
        if level < 0:
            raise ValueError("parallel_auto levels must be >= 0 (percent automated)")
        manifest = build_branch_automation_params(
            cfg.params_path,
            automate_task_ids=list(cfg.automate_task_ids),
            level=level,
            floor_seconds=cfg.automation_floor_seconds,
            out_json_path=out_params,
        )
    elif cfg.perturbation == "front_back_load":
        if cfg.chain_task_ids is None:
            raise ValueError("front_back_load requires cfg.chain_task_ids")
        if level < 0:
            raise ValueError(
                "front_back_load levels must be >= 0; direction is set by "
                "cfg.load_direction"
            )
        if cfg.load_direction not in ("front", "back"):
            raise ValueError("load_direction must be 'front' or 'back'")
        # Non-negative ladder stays monotone for the c-index; the sign of the
        # shift is taken from the configured direction.
        signed = level if cfg.load_direction == "front" else -level
        manifest = build_front_back_load_params(
            cfg.params_path,
            chain_task_ids=list(cfg.chain_task_ids),
            shift=signed,
            out_json_path=out_params,
        )
    else:
        raise ValueError(f"unknown perturbation type {cfg.perturbation!r}")
    manifest = {"perturbation": cfg.perturbation, "level": level, **manifest}
    return out_params, manifest


def _run_short_term(
    *, prefix_csv: Path, bpmn: Path, params: Path, cutoff: pd.Timestamp,
    horizon_end: pd.Timestamp, total_cases: int, out_dir: Path, seed: int,
) -> pd.DataFrame:
    """Run a short-term sim and return its continuation log as a DataFrame."""
    _seed_world(seed)
    sim_log_csv = _run_one_sim(
        prefix_csv=prefix_csv, bpmn=bpmn, params=params,
        cutoff=cutoff, horizon_end=horizon_end,
        total_cases=total_cases, out_dir=out_dir,
    )
    return _load_prosimos_log(sim_log_csv)


def _run_relabel_levels(
    cfg: PipelineConfig,
    run_dir: Path,
    *,
    prefix_csv: Path,
    cutoff: pd.Timestamp,
    horizon: pd.Timedelta,
    horizon_end: pd.Timestamp,
    ongoing_ids: set[str],
    results: list[dict],
    util_rows: list[dict],
) -> None:
    """Relabel-oracle flow: one short-term sim per replicate, compared to a
    relabeled copy of itself at each level.

    Unlike the param-perturbation families, this does NOT re-simulate per
    level. Each of ``cfg.runs`` reference draws serves as BOTH the baseline
    and the source for relabeling, so baseline-vs-sim share identical
    timestamps. ``level`` is interpreted as the percentage of the chosen
    activity's instances relabeled to a new label. Appends rows to
    ``results`` / ``util_rows`` in place.
    """
    level0_params, level0_manifest = _prepare_params_for_level(cfg, 0, run_dir)
    (run_dir / "level_0_manifest.json").write_text(json.dumps(level0_manifest, indent=2))
    with open(level0_params, encoding="utf-8") as f:
        level_params = json.load(f)
    role_map = _resource_to_profile(level_params)

    ref_logs: list[pd.DataFrame] = []
    for k in range(1, cfg.runs + 1):
        print(f"[relabel] reference run k={k}/{cfg.runs}")
        ref = _run_short_term(
            prefix_csv=prefix_csv, bpmn=cfg.bpmn_path, params=level0_params,
            cutoff=cutoff, horizon_end=horizon_end,
            total_cases=cfg.sim_total_cases,
            out_dir=run_dir / "reference" / f"k_{k}",
            seed=10_000 + k,
        )
        ref_logs.append(ref)

    if cfg.relabel_activity is not None:
        target = cfg.relabel_activity
    else:
        # Pick an activity well-represented in BOTH scopes so neither the
        # ongoing-only (A) nor the all-cases (B) signal is degenerate. The
        # naive "most frequent overall" pick lands on an early activity that
        # new arrivals dominate but mid-flight ongoing cases have passed,
        # leaving Scope A near-empty. Choose the activity present in both
        # windows with the highest Scope-A count; fall back to Scope-B max.
        win_a = filter_to_cases_in_window(ref_logs[0], ongoing_ids, cutoff, horizon)
        win_b = filter_to_cases_in_window(ref_logs[0], None, cutoff, horizon)
        cnt_a = win_a["activity"].value_counts()
        cnt_b = win_b["activity"].value_counts()
        in_both = [a for a in cnt_a.index if a in cnt_b.index]
        if in_both:
            target = str(max(in_both, key=lambda a: cnt_a[a]))
        else:
            target = str(cnt_b.idxmax())
    to_label = cfg.relabel_to or f"{target}__alt"
    (run_dir / "relabel_manifest.json").write_text(json.dumps({
        "relabel_activity": target,
        "relabel_to": to_label,
        "levels_are_percent_relabeled": True,
    }, indent=2))
    print(f"[relabel] splitting activity {target!r} -> {to_label!r}")

    for level in cfg.levels:
        frac = level / 100.0
        for k in range(1, cfg.runs + 1):
            print(f"[relabel] level={level}% k={k}/{cfg.runs}")
            ref = ref_logs[k - 1]
            rng = np.random.default_rng(20_000 + 1_000 * level + k)
            relabeled = relabel_activity_fraction(
                ref, activity=target, to_label=to_label, fraction=frac, rng=rng,
            )
            base_A = filter_to_cases_in_window(ref, ongoing_ids, cutoff, horizon)
            base_B = filter_to_cases_in_window(ref, None, cutoff, horizon)
            sim_A = filter_to_cases_in_window(relabeled, ongoing_ids, cutoff, horizon)
            sim_B = filter_to_cases_in_window(relabeled, None, cutoff, horizon)
            # Utilization from the reference (unmodified) log: identical across
            # levels by construction, a sanity artifact that ρ is invariant.
            util_rows.extend(_compute_utilization_rows(
                sim_log=ref, params=level_params,
                cutoff=cutoff, horizon_end=horizon_end, level=level, k_sim=k,
            ))
            results.extend(_compute_metrics_row(
                baseline_continuation=base_A, sim_continuation=sim_A,
                level=level, k_baseline=k, k_sim=k, scope="A_ongoing",
                window=(cutoff, horizon_end), role_map=role_map,
            ))
            results.extend(_compute_metrics_row(
                baseline_continuation=base_B, sim_continuation=sim_B,
                level=level, k_baseline=k, k_sim=k, scope="B_all",
                window=(cutoff, horizon_end), role_map=role_map,
            ))


def _run_mix_levels(
    cfg: PipelineConfig,
    run_dir: Path,
    *,
    results: list[dict],
    util_rows: list[dict],
) -> None:
    """Mix-ratio oracle: blend two independent basic-sim populations (green
    and red) at varying ratios and compare each perturbed merge to the
    baseline merge.

    Unlike the param-perturbation flow this does NOT use the short-term sim
    at all — both the reference and the "perturbed" log are deterministic
    merges of the same per-replicate green and red basic sims, just at
    different ratios. ``level`` is interpreted as a percentage-point shift
    from ``cfg.mix_baseline_green`` toward more green cases. At ``level==0``
    the same merge is produced on both sides, so every distance is exactly
    zero by construction (oracle pattern).

    Cycle_time is approximately blind by construction provided green and red
    share the same case-duration distribution (only the XOR routing differs,
    not the activity durations). ngd detects because the bigram histogram
    shifts with the green/red proportion. State metrics detect because the
    cross-case active-instance pattern shifts with which cases are in flight.
    """
    if cfg.mix_green_params is None or cfg.mix_red_params is None:
        raise ValueError(
            "mix_ratio requires cfg.mix_green_params and cfg.mix_red_params"
        )
    if not 0.0 < cfg.mix_baseline_green < 1.0:
        raise ValueError("mix_baseline_green must be in (0, 1)")

    # K independent basic sims per population. Cached under run_dir parent
    # so repeat runs of the same dataset skip the simulation.
    sims_root = cfg.outputs_root / cfg.dataset_name / "mix_basic_sims"
    sims_root.mkdir(parents=True, exist_ok=True)
    green_logs: list[pd.DataFrame] = []
    red_logs: list[pd.DataFrame] = []
    for k in range(1, cfg.runs + 1):
        for tag, params, bucket in (
            ("green", cfg.mix_green_params, green_logs),
            ("red", cfg.mix_red_params, red_logs),
        ):
            csv = sims_root / tag / f"k_{k}.csv"
            stats = sims_root / tag / f"k_{k}_stats.csv"
            if not csv.exists():
                csv.parent.mkdir(parents=True, exist_ok=True)
                print(f"[mix_ratio] basic sim {tag} k={k}/{cfg.runs}")
                run_basic_simulation(
                    bpmn_model=cfg.bpmn_path, json_sim_params=params,
                    total_cases=cfg.gt_total_cases,
                    out_stats_csv_path=stats, out_log_csv_path=csv,
                )
            bucket.append(_load_prosimos_log(csv))

    # Target merged-log size = the user-set sim_total_cases, capped by the
    # smaller of the per-population case counts (so even fraction=1.0 has
    # enough green cases).
    min_green = min(g["case_id"].nunique() for g in green_logs)
    min_red = min(r["case_id"].nunique() for r in red_logs)
    target_cases = min(cfg.sim_total_cases, min_green, min_red)
    if target_cases <= 0:
        raise ValueError("no green/red cases available")

    # Build reference (baseline-ratio) merges per replicate; pick the cutoff
    # and horizon from replicate 1's reference so windowing is shared across
    # replicates and across levels. Per-replicate seed = k ensures different
    # replicates draw different permutations (so they're independent draws),
    # while same-seed+same-fraction at level 0 guarantees ref == perturbed.
    baseline_merges = [
        merge_logs_at_ratio(
            green_logs[k], red_logs[k],
            fraction_green=cfg.mix_baseline_green,
            target_cases=target_cases,
            seed=40_000 + (k + 1),
        )
        for k in range(cfg.runs)
    ]
    ref0 = baseline_merges[0]
    if cfg.cutoff_strategy == "p90_wip":
        cutoff = _select_high_wip_cutoff(ref0)
    elif cfg.cutoff_strategy == "fraction":
        cutoff = _select_fraction_of_log_cutoff(ref0, cfg.cutoff_fraction)
    elif cfg.cutoff_strategy == "n_ongoing":
        cutoff = _select_n_ongoing_cutoff(ref0, cfg.target_ongoing)
    else:
        raise ValueError(f"unknown cutoff_strategy {cfg.cutoff_strategy!r}")
    log_end = ref0["end_time"].max()
    if cfg.horizon_hours is not None:
        horizon = min(pd.Timedelta(hours=cfg.horizon_hours), log_end - cutoff)
    else:
        horizon = _compute_horizon(ref0, cutoff)
    if horizon <= pd.Timedelta(0):
        raise ValueError("cutoff at or past end of merged log")
    horizon_end = cutoff + horizon

    (run_dir / "cutoff.json").write_text(json.dumps({
        "cutoff": cutoff.isoformat(),
        "horizon_end": horizon_end.isoformat(),
        "horizon_seconds": horizon.total_seconds(),
        "target_cases": target_cases,
        "mix_baseline_green": cfg.mix_baseline_green,
    }, indent=2))

    # Single role_map for utilization: derived from the green params. We
    # assume green and red use the same resource pool (the spec only varies
    # XOR routing / role assignment, not the resource list).
    with open(cfg.mix_green_params, encoding="utf-8") as f:
        level_params = json.load(f)
    role_map = _resource_to_profile(level_params)

    (run_dir / "mix_manifest.json").write_text(json.dumps({
        "mix_green_params": str(cfg.mix_green_params),
        "mix_red_params": str(cfg.mix_red_params),
        "mix_baseline_green": cfg.mix_baseline_green,
        "levels_are_percentage_point_shifts": True,
        "target_cases": target_cases,
        "n_green_basic_sims": len(green_logs),
        "n_red_basic_sims": len(red_logs),
    }, indent=2))

    for level in cfg.levels:
        frac_green = cfg.mix_baseline_green + level / 100.0
        if not 0.0 <= frac_green <= 1.0:
            print(f"[mix_ratio] skipping level={level}: implied "
                  f"fraction_green={frac_green:.2f} out of [0, 1]")
            continue
        for k in range(1, cfg.runs + 1):
            print(f"[mix_ratio] level={level}pp green={frac_green:.2f} "
                  f"k={k}/{cfg.runs}")
            ref = baseline_merges[k - 1]
            perturbed = merge_logs_at_ratio(
                green_logs[k - 1], red_logs[k - 1],
                fraction_green=frac_green, target_cases=target_cases,
                seed=40_000 + k,
            )
            # Per-log ongoing case set: case IDs differ across (baseline,
            # perturbed) by construction, so each log defines its own Scope A.
            ref_ongoing = get_ongoing_case_ids(ref, cutoff)
            pert_ongoing = get_ongoing_case_ids(perturbed, cutoff)
            base_A = filter_to_cases_in_window(ref, ref_ongoing, cutoff, horizon)
            sim_A = filter_to_cases_in_window(perturbed, pert_ongoing, cutoff, horizon)
            util_rows.extend(_compute_utilization_rows(
                sim_log=ref, params=level_params,
                cutoff=cutoff, horizon_end=horizon_end, level=level, k_sim=k,
            ))
            results.extend(_compute_metrics_row(
                baseline_continuation=base_A, sim_continuation=sim_A,
                level=level, k_baseline=k, k_sim=k, scope="A_ongoing",
                window=(cutoff, horizon_end), role_map=role_map,
            ))


def label_swap_case_types(
    log: pd.DataFrame,
    *,
    fraction: float,
    rng: np.random.Generator,
    values: tuple[str, str] | None = None,
) -> pd.DataFrame:
    """Return a copy of ``log`` with ``fraction`` of the cases' ``case_type``
    tags flipped between the two case-type values.

    The two values default to ("green", "red") for the merge-based label_swap
    oracle, but any binary tagging works — pass ``values`` or leave it None to
    auto-detect the two distinct case_type values present (used by the
    case_route oracle, whose tags are red/blue).

    The activity rows, timestamps, case IDs, and resources are left
    untouched — only the per-case ``case_type`` column changes. We flip an
    equal number from each side (``floor(fraction × min(n_a, n_b))``) so the
    marginal is preserved exactly. That is the pairing-unique attack: every
    per-case marginal (including ngd's bigram histogram and the activity-only
    state projections) is invariant, and only case_type-aware projections
    detect the swap.

    At ``fraction == 0`` the output is identical to the input.
    """
    out = log.copy()
    if "case_type" not in out.columns:
        raise ValueError("log has no 'case_type' column to swap")
    if fraction <= 0:
        return out
    if fraction > 1.0:
        raise ValueError("fraction must be in [0, 1]")
    by_case = out.groupby("case_id")["case_type"].first()
    if values is None:
        present = sorted(by_case.dropna().unique().tolist())
        if len(present) != 2:
            raise ValueError(
                f"label swap needs exactly two case_type values, found {present!r}"
            )
        a, b = present
    else:
        a, b = values
    a_cases = by_case[by_case == a].index.to_numpy()
    b_cases = by_case[by_case == b].index.to_numpy()
    n_flip = int(fraction * min(len(a_cases), len(b_cases)))
    if n_flip <= 0:
        return out
    swap_a = rng.choice(a_cases, size=n_flip, replace=False)
    swap_b = rng.choice(b_cases, size=n_flip, replace=False)
    swap_set = set(swap_a.tolist()) | set(swap_b.tolist())
    # Vectorised flip: build a new column from the case_id mask.
    def flip(row_case_id: str, row_type: str) -> str:
        if row_case_id in swap_set:
            return b if row_type == a else a
        return row_type
    out["case_type"] = [
        flip(c, t)
        for c, t in zip(out["case_id"].to_numpy(), out["case_type"].to_numpy())
    ]
    return out


def drift_case_types(
    log: pd.DataFrame,
    *,
    strength: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Return a copy of ``log`` whose ``case_type`` tags are made to *drift*
    along the arrival timeline, with the global marginal preserved exactly.

    A fraction ``strength`` of the cases (chosen at random, by arrival
    position) have their tags re-sorted so that — among the chosen positions —
    the lexicographically-smaller value lands on the earlier arrivals and the
    larger value on the later ones. Because the re-sort only permutes the tags
    *already present* on the chosen positions, the per-value counts (and hence
    the global red/blue marginal) are unchanged; only the temporal arrangement
    shifts. At ``strength == 1`` the chosen set is the whole log and the tags
    are fully time-sorted; at ``strength == 0`` the output is identical.

    This is the controlled, synthetic analogue of a real log whose case-type
    mix drifts over time (scenario #6). Unlike ``label_swap_case_types`` (random
    marginal-preserving swaps → only the *paired* ``activity_type`` projection
    moves), a temporal drift skews the active case_type composition inside any
    sub-window, so the marginal-blind baselines stay flat while BOTH the
    ``case_type`` and ``activity_type`` state projections detect it.

    The event stream (activities, timestamps, resources, case ids) is untouched,
    so cycle_time / ngd / red / rtd and the type-agnostic state projections are
    invariant by construction.
    """
    out = log.copy()
    if "case_type" not in out.columns:
        raise ValueError("log has no 'case_type' column to drift")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    if strength <= 0:
        return out

    first_start = out.groupby("case_id")["start_time"].min().sort_values()
    case_order = list(first_start.index)
    tag_by_case = out.groupby("case_id")["case_type"].first().to_dict()
    n = len(case_order)
    k = int(round(strength * n))
    if k <= 1:
        return out

    # Choose k arrival positions to re-sort; keep them in arrival order.
    positions = np.sort(rng.choice(n, size=k, replace=False))
    chosen_cases = [case_order[p] for p in positions]
    chosen_tags = [tag_by_case[c] for c in chosen_cases]
    distinct = sorted(set(t for t in chosen_tags if t is not None and t == t))
    if len(distinct) < 2:
        return out  # only one value among the chosen — nothing to drift
    early_val = distinct[0]
    n_early = sum(t == early_val for t in chosen_tags)
    # early_val first (earliest arrivals), everything else after.
    resorted = [early_val] * n_early + [
        t for t in chosen_tags if t != early_val
    ]
    new_tag = dict(tag_by_case)
    for case, nt in zip(chosen_cases, resorted):
        new_tag[case] = nt
    out["case_type"] = out["case_id"].map(new_tag)
    return out


def _run_label_swap_levels(
    cfg: PipelineConfig,
    run_dir: Path,
    *,
    results: list[dict],
    util_rows: list[dict],
) -> None:
    """Label-swap oracle: build a fixed 50/50 reference merge, then per level
    flip ``level%`` of the green↔red case_type tags. The event log itself
    (activities, timestamps, resources) stays identical; only the case_type
    column changes.

    By construction this leaves cycle_time / ngd / red / rtd at exactly 0
    (the event stream is unchanged) AND leaves the activity/case-only state
    projections at 0 (same case IDs, same activities). Only case_type-aware
    projections (case_type, activity_type) detect the swap — the genuinely
    pairing-unique pattern.

    Uses the same per-population basic-sim infrastructure as mix_ratio.
    """
    if cfg.mix_green_params is None or cfg.mix_red_params is None:
        raise ValueError(
            "label_swap requires cfg.mix_green_params and cfg.mix_red_params"
        )
    sims_root = cfg.outputs_root / cfg.dataset_name / "mix_basic_sims"
    sims_root.mkdir(parents=True, exist_ok=True)
    green_logs: list[pd.DataFrame] = []
    red_logs: list[pd.DataFrame] = []
    for k in range(1, cfg.runs + 1):
        for tag, params, bucket in (
            ("green", cfg.mix_green_params, green_logs),
            ("red", cfg.mix_red_params, red_logs),
        ):
            csv = sims_root / tag / f"k_{k}.csv"
            stats = sims_root / tag / f"k_{k}_stats.csv"
            if not csv.exists():
                csv.parent.mkdir(parents=True, exist_ok=True)
                print(f"[label_swap] basic sim {tag} k={k}/{cfg.runs}")
                run_basic_simulation(
                    bpmn_model=cfg.bpmn_path, json_sim_params=params,
                    total_cases=cfg.gt_total_cases,
                    out_stats_csv_path=stats, out_log_csv_path=csv,
                )
            bucket.append(_load_prosimos_log(csv))

    min_green = min(g["case_id"].nunique() for g in green_logs)
    min_red = min(r["case_id"].nunique() for r in red_logs)
    target_cases = min(cfg.sim_total_cases, min_green, min_red)
    if target_cases <= 0:
        raise ValueError("no green/red cases available")

    reference_merges = [
        merge_logs_at_ratio(
            green_logs[k], red_logs[k],
            fraction_green=cfg.mix_baseline_green,
            target_cases=target_cases,
            seed=40_000 + (k + 1),
        )
        for k in range(cfg.runs)
    ]
    ref0 = reference_merges[0]
    if cfg.cutoff_strategy == "p90_wip":
        cutoff = _select_high_wip_cutoff(ref0)
    elif cfg.cutoff_strategy == "fraction":
        cutoff = _select_fraction_of_log_cutoff(ref0, cfg.cutoff_fraction)
    elif cfg.cutoff_strategy == "n_ongoing":
        cutoff = _select_n_ongoing_cutoff(ref0, cfg.target_ongoing)
    else:
        raise ValueError(f"unknown cutoff_strategy {cfg.cutoff_strategy!r}")
    log_end = ref0["end_time"].max()
    horizon = (
        min(pd.Timedelta(hours=cfg.horizon_hours), log_end - cutoff)
        if cfg.horizon_hours is not None
        else _compute_horizon(ref0, cutoff)
    )
    if horizon <= pd.Timedelta(0):
        raise ValueError("cutoff at or past end of merged log")
    horizon_end = cutoff + horizon

    (run_dir / "cutoff.json").write_text(json.dumps({
        "cutoff": cutoff.isoformat(),
        "horizon_end": horizon_end.isoformat(),
        "horizon_seconds": horizon.total_seconds(),
        "target_cases": target_cases,
        "mix_baseline_green": cfg.mix_baseline_green,
    }, indent=2))

    with open(cfg.mix_green_params, encoding="utf-8") as f:
        level_params = json.load(f)
    role_map = _resource_to_profile(level_params)

    (run_dir / "label_swap_manifest.json").write_text(json.dumps({
        "mix_green_params": str(cfg.mix_green_params),
        "mix_red_params": str(cfg.mix_red_params),
        "mix_baseline_green": cfg.mix_baseline_green,
        "levels_are_percent_pairs_swapped": True,
        "target_cases": target_cases,
    }, indent=2))

    for level in cfg.levels:
        frac = level / 100.0
        for k in range(1, cfg.runs + 1):
            print(f"[label_swap] level={level}% k={k}/{cfg.runs}")
            ref = reference_merges[k - 1]
            rng = np.random.default_rng(50_000 + 1_000 * level + k)
            perturbed = label_swap_case_types(ref, fraction=frac, rng=rng)
            ongoing = get_ongoing_case_ids(ref, cutoff)
            base_A = filter_to_cases_in_window(ref, ongoing, cutoff, horizon)
            sim_A = filter_to_cases_in_window(perturbed, ongoing, cutoff, horizon)
            util_rows.extend(_compute_utilization_rows(
                sim_log=ref, params=level_params,
                cutoff=cutoff, horizon_end=horizon_end, level=level, k_sim=k,
            ))
            results.extend(_compute_metrics_row(
                baseline_continuation=base_A, sim_continuation=sim_A,
                level=level, k_baseline=k, k_sim=k, scope="A_ongoing",
                window=(cutoff, horizon_end), role_map=role_map,
            ))


def _run_rephase_levels(
    cfg: PipelineConfig,
    run_dir: Path,
    *,
    prefix_csv: Path,
    cutoff: pd.Timestamp,
    horizon: pd.Timedelta,
    horizon_end: pd.Timestamp,
    ongoing_ids: set[str],
    results: list[dict],
    util_rows: list[dict],
) -> None:
    """Re-phase oracle: shift each case's timeline by a per-case random offset,
    preserving every per-case marginal exactly.

    Like the relabel oracle this does NOT re-simulate per level. Each of
    ``cfg.runs`` reference draws serves as BOTH the baseline and the source
    for re-phasing, so baseline-vs-sim share identical event sequences
    case-by-case — only absolute placement on the timeline differs. ``level``
    is interpreted as the jitter magnitude in hours: a per-case uniform draw
    on ``[-level, +level]`` clamped to keep every event inside
    ``[cutoff, horizon_end)``. By construction every per-case marginal
    metric (cycle_time, ngd, red) is zero at every level; the cross-case
    state metrics carry the entire signal.
    """
    level0_params, level0_manifest = _prepare_params_for_level(cfg, 0, run_dir)
    (run_dir / "level_0_manifest.json").write_text(json.dumps(level0_manifest, indent=2))
    with open(level0_params, encoding="utf-8") as f:
        level_params = json.load(f)
    role_map = _resource_to_profile(level_params)

    ref_logs: list[pd.DataFrame] = []
    for k in range(1, cfg.runs + 1):
        print(f"[rephase] reference run k={k}/{cfg.runs}")
        ref = _run_short_term(
            prefix_csv=prefix_csv, bpmn=cfg.bpmn_path, params=level0_params,
            cutoff=cutoff, horizon_end=horizon_end,
            total_cases=cfg.sim_total_cases,
            out_dir=run_dir / "reference" / f"k_{k}",
            seed=10_000 + k,
        )
        ref_logs.append(ref)

    (run_dir / "rephase_manifest.json").write_text(json.dumps({
        "levels_are_jitter_hours": True,
        "scheme": (
            "per-case uniform [-jitter_h, +jitter_h] clamped to "
            "[cutoff, horizon_end); shift applied AFTER window filtering"
        ),
    }, indent=2))

    for level in cfg.levels:
        jitter_h = float(level)
        for k in range(1, cfg.runs + 1):
            print(f"[rephase] level={level}h k={k}/{cfg.runs}")
            ref = ref_logs[k - 1]
            base_A = filter_to_cases_in_window(ref, ongoing_ids, cutoff, horizon)
            # Apply rephasing AFTER window filtering: clipped per-case bounds
            # define the feasible offset range, and shifts inside that range
            # never trigger further clipping, so per-case invariants hold.
            rng_a = np.random.default_rng(30_000 + 1_000 * level + k)
            sim_A = rephase_cases(
                base_A, jitter_hours=jitter_h, rng=rng_a,
                cutoff=cutoff, horizon_end=horizon_end,
            )
            # Utilization from the unmodified reference: ρ is invariant by
            # construction (shifting cases preserves total busy time across
            # the window, modulo boundary clipping which is excluded by the
            # per-case feasible-range clamp).
            util_rows.extend(_compute_utilization_rows(
                sim_log=ref, params=level_params,
                cutoff=cutoff, horizon_end=horizon_end, level=level, k_sim=k,
            ))
            results.extend(_compute_metrics_row(
                baseline_continuation=base_A, sim_continuation=sim_A,
                level=level, k_baseline=k, k_sim=k, scope="A_ongoing",
                window=(cutoff, horizon_end), role_map=role_map,
            ))
            # Scope B (ongoing + new arrivals) intentionally disabled — we
            # focus exclusively on ongoing cases. To re-enable, restore
            # base_B = filter_to_cases_in_window(ref, None, cutoff, horizon),
            # sim_B = rephase_cases(base_B, ..., rng=np.random.default_rng(
            # 30_500 + 1_000 * level + k)), then add a _compute_metrics_row
            # call with scope="B_all".


def _count_split_gateways(params_path: Path) -> int:
    """Number of two-way XOR splits in a case-route params file (paths _a/_b)."""
    with open(params_path, encoding="utf-8") as f:
        params = json.load(f)
    n = 0
    for e in params.get("gateway_branching_probabilities", []):
        ids = [p["path_id"] for p in e.get("probabilities", [])]
        if len(ids) == 2 and any(i.endswith("_a") for i in ids) and any(
            i.endswith("_b") for i in ids
        ):
            n += 1
    return n


def _run_case_route_levels(
    cfg: PipelineConfig,
    run_dir: Path,
    *,
    prefix_csv: Path,
    cutoff: pd.Timestamp,
    horizon: pd.Timedelta,
    horizon_end: pd.Timestamp,
    ongoing_ids: set[str],
    results: list[dict],
    util_rows: list[dict],
) -> None:
    """Case-route oracle: the reference is a REAL short-term sim in which
    ``case_type`` genuinely drives XOR routing (red -> A-branch, blue ->
    B-branch) via Prosimos branch_rules, with each ongoing case's historical
    ``case_type`` restored from the partial-state snapshot on resume. Each
    ``level`` then swaps ``level%`` of the case_type tags on that reference.

    Because every branch is duration-symmetric and the branches are separated
    by common activities, the swap leaves cycle_time / ngd_n2 / rtd and the
    type-agnostic state projections (activity, case, cardinality) exactly
    unchanged, and preserves the red/blue marginal — so the ``case_type``
    projection is blind too. Only the joint ``activity_type`` projection, which
    sees that (B-branch, red) and (A-branch, blue) instances appear after the
    swap, detects it. This is the pairing-unique result of the label_swap
    oracle, but on a genuinely attribute-routed simulation rather than a
    post-hoc merge of two basic sims.
    """
    n_splits = _count_split_gateways(Path(cfg.params_path))
    n_ruled = cfg.case_route_ruled if cfg.case_route_ruled is not None else n_splits
    ruled_params = run_dir / "case_route_ruled.json"
    route_manifest = build_case_route_params(
        cfg.params_path, n_gateways_ruled=n_ruled, out_json_path=ruled_params,
    )
    with open(ruled_params, encoding="utf-8") as f:
        level_params = json.load(f)
    role_map = _resource_to_profile(level_params)

    ref_logs: list[pd.DataFrame] = []
    for k in range(1, cfg.runs + 1):
        print(f"[case_route] reference (real attribute-routed) sim k={k}/{cfg.runs}")
        ref = _run_short_term(
            prefix_csv=prefix_csv, bpmn=cfg.bpmn_path, params=ruled_params,
            cutoff=cutoff, horizon_end=horizon_end,
            total_cases=cfg.sim_total_cases,
            out_dir=run_dir / "reference" / f"k_{k}",
            seed=10_000 + k,
        )
        if "case_type" not in ref.columns:
            raise RuntimeError(
                "reference sim has no case_type column; the params must declare "
                "a case_type case attribute"
            )
        ref_logs.append(ref)

    (run_dir / "case_route_manifest.json").write_text(json.dumps({
        **route_manifest,
        "levels_are_percent_tags_swapped": True,
        "reference": "real short-term sim with case_type-driven branch_rules",
    }, indent=2))

    for level in cfg.levels:
        frac = level / 100.0
        for k in range(1, cfg.runs + 1):
            print(f"[case_route] swap level={level}% k={k}/{cfg.runs}")
            ref = ref_logs[k - 1]
            rng = np.random.default_rng(60_000 + 1_000 * level + k)
            perturbed = label_swap_case_types(ref, fraction=frac, rng=rng)
            base_A = filter_to_cases_in_window(ref, ongoing_ids, cutoff, horizon)
            sim_A = filter_to_cases_in_window(perturbed, ongoing_ids, cutoff, horizon)
            util_rows.extend(_compute_utilization_rows(
                sim_log=ref, params=level_params,
                cutoff=cutoff, horizon_end=horizon_end, level=level, k_sim=k,
            ))
            results.extend(_compute_metrics_row(
                baseline_continuation=base_A, sim_continuation=sim_A,
                level=level, k_baseline=k, k_sim=k, scope="A_ongoing",
                window=(cutoff, horizon_end), role_map=role_map,
            ))


def _run_case_type_drift_levels(
    cfg: PipelineConfig,
    run_dir: Path,
    *,
    prefix_csv: Path,
    cutoff: pd.Timestamp,
    horizon: pd.Timedelta,
    horizon_end: pd.Timestamp,
    ongoing_ids: set[str],
    results: list[dict],
    util_rows: list[dict],
) -> None:
    """Case-type-drift oracle (scenario #6, controlled synthetic form).

    Like ``_run_case_route_levels`` the reference is a REAL short-term sim in
    which ``case_type`` genuinely drives XOR routing (red -> A-branch, blue ->
    B-branch) via Prosimos branch_rules. Each ``level`` then makes the
    ``case_type`` tags *drift* along the arrival timeline at strength
    ``level%`` (``drift_case_types``), holding the global red/blue marginal
    fixed.

    Because the event stream is untouched and the marginal is preserved,
    cycle_time / ngd_n2 / red / rtd and the type-agnostic state projections are
    blind. Unlike the random label_swap, a *temporal* drift skews the active
    case_type composition inside the evaluation window, so BOTH the
    ``case_type`` and the joint ``activity_type`` state projections detect it —
    the data-attribute, drift-over-time win.
    """
    n_splits = _count_split_gateways(Path(cfg.params_path))
    n_ruled = cfg.case_route_ruled if cfg.case_route_ruled is not None else n_splits
    ruled_params = run_dir / "case_route_ruled.json"
    route_manifest = build_case_route_params(
        cfg.params_path, n_gateways_ruled=n_ruled, out_json_path=ruled_params,
    )
    with open(ruled_params, encoding="utf-8") as f:
        level_params = json.load(f)
    role_map = _resource_to_profile(level_params)

    ref_logs: list[pd.DataFrame] = []
    for k in range(1, cfg.runs + 1):
        print(f"[case_type_drift] reference (real attribute-routed) sim k={k}/{cfg.runs}")
        ref = _run_short_term(
            prefix_csv=prefix_csv, bpmn=cfg.bpmn_path, params=ruled_params,
            cutoff=cutoff, horizon_end=horizon_end,
            total_cases=cfg.sim_total_cases,
            out_dir=run_dir / "reference" / f"k_{k}",
            seed=10_000 + k,
        )
        if "case_type" not in ref.columns:
            raise RuntimeError(
                "reference sim has no case_type column; the params must declare "
                "a case_type case attribute"
            )
        ref_logs.append(ref)

    (run_dir / "case_type_drift_manifest.json").write_text(json.dumps({
        **route_manifest,
        "levels_are_percent_drift_strength": True,
        "reference": "real short-term sim with case_type-driven branch_rules",
    }, indent=2))

    for level in cfg.levels:
        strength = level / 100.0
        for k in range(1, cfg.runs + 1):
            print(f"[case_type_drift] drift level={level}% k={k}/{cfg.runs}")
            ref = ref_logs[k - 1]
            rng = np.random.default_rng(70_000 + 1_000 * level + k)
            perturbed = drift_case_types(ref, strength=strength, rng=rng)
            base_A = filter_to_cases_in_window(ref, ongoing_ids, cutoff, horizon)
            sim_A = filter_to_cases_in_window(perturbed, ongoing_ids, cutoff, horizon)
            util_rows.extend(_compute_utilization_rows(
                sim_log=ref, params=level_params,
                cutoff=cutoff, horizon_end=horizon_end, level=level, k_sim=k,
            ))
            results.extend(_compute_metrics_row(
                baseline_continuation=base_A, sim_continuation=sim_A,
                level=level, k_baseline=k, k_sim=k, scope="A_ongoing",
                window=(cutoff, horizon_end), role_map=role_map,
            ))


def run_pipeline(cfg: PipelineConfig) -> Path:
    """Run the end-to-end pipeline; return path to the results.csv."""
    run_id = generate_short_uuid()
    run_dir = cfg.outputs_root / cfg.dataset_name / cfg.perturbation / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # mix_ratio bypasses the prefix/short-term-sim machinery entirely. It
    # runs its own per-population basic sims and derives cutoff/horizon
    # internally from the merged reference. Branch early so we don't
    # gratuitously run a `Loan-stable` basic sim we never use.
    if cfg.perturbation == "mix_ratio":
        results: list[dict] = []
        util_rows: list[dict] = []
        _run_mix_levels(cfg, run_dir, results=results, util_rows=util_rows)
        return _finalize_pipeline_outputs(run_dir, results, util_rows)
    if cfg.perturbation == "label_swap":
        results: list[dict] = []
        util_rows: list[dict] = []
        _run_label_swap_levels(cfg, run_dir, results=results, util_rows=util_rows)
        return _finalize_pipeline_outputs(run_dir, results, util_rows)

    # Step 1: one basic sim to source the prefix.
    source_path = build_basic_sim_for_prefix(cfg)
    source_log = _load_prosimos_log(source_path)

    # Step 2: cutoff + horizon on the source log.
    if cfg.cutoff_strategy == "p90_wip":
        cutoff = _select_high_wip_cutoff(source_log)
    elif cfg.cutoff_strategy == "fraction":
        cutoff = _select_fraction_of_log_cutoff(source_log, cfg.cutoff_fraction)
    elif cfg.cutoff_strategy == "n_ongoing":
        cutoff = _select_n_ongoing_cutoff(source_log, cfg.target_ongoing)
    else:
        raise ValueError(f"unknown cutoff_strategy {cfg.cutoff_strategy!r}")

    log_end = source_log["end_time"].max()
    if cfg.horizon_hours is not None:
        horizon = min(pd.Timedelta(hours=cfg.horizon_hours), log_end - cutoff)
        if horizon <= pd.Timedelta(0):
            raise ValueError(
                "cutoff at or past the end of the log; cannot fit any horizon"
            )
    else:
        horizon = _compute_horizon(source_log, cutoff)
    horizon_end = cutoff + horizon
    ongoing_ids = get_ongoing_case_ids(source_log, cutoff)

    prefix_csv = run_dir / "prefix.csv"
    _write_prefix_csv(source_log, cutoff, prefix_csv)
    (run_dir / "cutoff.json").write_text(json.dumps({
        "cutoff": cutoff.isoformat(),
        "horizon_end": horizon_end.isoformat(),
        "horizon_seconds": horizon.total_seconds(),
        "cutoff_strategy": cfg.cutoff_strategy,
        "cutoff_fraction": cfg.cutoff_fraction if cfg.cutoff_strategy == "fraction" else None,
        "target_ongoing": cfg.target_ongoing if cfg.cutoff_strategy == "n_ongoing" else None,
        "horizon_hours_config": cfg.horizon_hours,
        "ongoing_cases_in_source": len(ongoing_ids),
    }, indent=2))

    # Relabel oracle: dedicated flow (one sim per replicate, relabeled in
    # place). Shares prefix/cutoff/horizon setup above and the results/
    # rankings tail below, but skips the param-perturbation re-sim loop.
    results: list[dict] = []
    util_rows: list[dict] = []
    if cfg.perturbation == "relabel":
        _run_relabel_levels(
            cfg, run_dir, prefix_csv=prefix_csv, cutoff=cutoff,
            horizon=horizon, horizon_end=horizon_end, ongoing_ids=ongoing_ids,
            results=results, util_rows=util_rows,
        )
        return _finalize_pipeline_outputs(run_dir, results, util_rows)
    if cfg.perturbation == "rephase":
        _run_rephase_levels(
            cfg, run_dir, prefix_csv=prefix_csv, cutoff=cutoff,
            horizon=horizon, horizon_end=horizon_end, ongoing_ids=ongoing_ids,
            results=results, util_rows=util_rows,
        )
        return _finalize_pipeline_outputs(run_dir, results, util_rows)
    if cfg.perturbation == "case_route":
        _run_case_route_levels(
            cfg, run_dir, prefix_csv=prefix_csv, cutoff=cutoff,
            horizon=horizon, horizon_end=horizon_end, ongoing_ids=ongoing_ids,
            results=results, util_rows=util_rows,
        )
        return _finalize_pipeline_outputs(run_dir, results, util_rows)
    if cfg.perturbation == "case_type_drift":
        _run_case_type_drift_levels(
            cfg, run_dir, prefix_csv=prefix_csv, cutoff=cutoff,
            horizon=horizon, horizon_end=horizon_end, ongoing_ids=ongoing_ids,
            results=results, util_rows=util_rows,
        )
        return _finalize_pipeline_outputs(run_dir, results, util_rows)

    # Step 3: K short-term BASELINE runs at level 0 (shared across all levels).
    # The level-0 short-term sim is the locked baseline: zero-perturbation
    # reference. It is NOT the long-term simulation any more.
    level0_params, level0_manifest = _prepare_params_for_level(cfg, 0, run_dir)
    (run_dir / "level_0_manifest.json").write_text(json.dumps(level0_manifest, indent=2))

    baseline_conts_A: dict[int, pd.DataFrame] = {}
    baseline_conts_B: dict[int, pd.DataFrame] = {}
    for k in range(1, cfg.runs + 1):
        print(f"[pipeline] baseline run k={k}/{cfg.runs}")
        baseline_log = _run_short_term(
            prefix_csv=prefix_csv, bpmn=cfg.bpmn_path, params=level0_params,
            cutoff=cutoff, horizon_end=horizon_end,
            total_cases=cfg.sim_total_cases,
            out_dir=run_dir / "baseline" / f"k_{k}",
            seed=10_000 + k,   # disjoint seed range from sim seeds
        )
        baseline_conts_A[k] = filter_to_cases_in_window(
            baseline_log, ongoing_ids, cutoff, horizon
        )
        baseline_conts_B[k] = filter_to_cases_in_window(
            baseline_log, None, cutoff, horizon
        )

    # Step 4: per level × k, one sim run vs baseline[k].
    # Level 0 is a stochastic baseline: run an independent same-params sim
    # so the L=0 distances reflect sim-to-sim noise instead of 0. This
    # exposes the metric's noise floor at the current windowing — without
    # it, every level > 0 gets a free signal-vs-zero discrimination that
    # masks the real ranking ability.
    for level in cfg.levels:
        params_path, manifest = _prepare_params_for_level(cfg, level, run_dir)
        (run_dir / f"level_{level}_manifest.json").write_text(json.dumps(manifest, indent=2))
        with open(params_path, encoding="utf-8") as f:
            level_params = json.load(f)
        # Recompute the role_map per level: role_swap and resource add/remove
        # both change which resource_ids exist and which profile they belong to.
        level_role_map = _resource_to_profile(level_params)

        for k in range(1, cfg.runs + 1):
            print(f"[pipeline] level={level} sim k={k}/{cfg.runs}")
            # Level 0 uses cfg.params_path (returned by _prepare_params_for_level
            # for level==0). Distinct seed so it does NOT collide with the
            # baseline draw at the same k.
            sim_log = _run_short_term(
                prefix_csv=prefix_csv, bpmn=cfg.bpmn_path, params=Path(params_path),
                cutoff=cutoff, horizon_end=horizon_end,
                total_cases=cfg.sim_total_cases,
                out_dir=run_dir / f"level_{level}" / f"k_{k}",
                # Negative levels (resource-add) still get a unique seed band.
                seed=20_000 + 1_000 * (level + 100) + k,
            )
            sim_A = filter_to_cases_in_window(sim_log, ongoing_ids, cutoff, horizon)
            sim_B = filter_to_cases_in_window(sim_log, None, cutoff, horizon)
            util_rows.extend(_compute_utilization_rows(
                sim_log=sim_log, params=level_params,
                cutoff=cutoff, horizon_end=horizon_end,
                level=level, k_sim=k,
            ))

            results.extend(_compute_metrics_row(
                baseline_continuation=baseline_conts_A[k], sim_continuation=sim_A,
                level=level, k_baseline=k, k_sim=k, scope="A_ongoing",
                window=(cutoff, horizon_end), role_map=level_role_map,
            ))
            results.extend(_compute_metrics_row(
                baseline_continuation=baseline_conts_B[k], sim_continuation=sim_B,
                level=level, k_baseline=k, k_sim=k, scope="B_all",
                window=(cutoff, horizon_end), role_map=level_role_map,
            ))

    return _finalize_pipeline_outputs(run_dir, results, util_rows)


def _finalize_pipeline_outputs(
    run_dir: Path, results: list[dict], util_rows: list[dict],
) -> Path:
    """Write results.csv / utilization.csv / rankings.csv; return results path."""
    results_path = run_dir / "results.csv"
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_path, index=False)
    print(f"[pipeline] results written to {results_path}")

    util_path = run_dir / "utilization.csv"
    pd.DataFrame(util_rows).to_csv(util_path, index=False)
    print(f"[pipeline] utilization written to {util_path}")

    # Ranking-quality statistics (c-index, Spearman, Kendall + CIs).
    rankings_path = write_rankings_csv(results_df, run_dir / "rankings.csv")
    print(f"[pipeline] rankings written to {rankings_path}")

    # Leave intermediate output.json from the runner wherever it landed.
    stray = Path("output.json")
    if stray.exists():
        shutil.copy(stray, run_dir / "output.json")

    return results_path


def write_rankings_csv(
    results_df: pd.DataFrame, out_path: Path, *, n_bootstrap: int = 1000,
) -> Path:
    """Compute rankings from a results DataFrame and write to ``out_path``."""
    from evaluation.state_metrics.ranking import compute_rankings

    rankings = compute_rankings(results_df, n_bootstrap=n_bootstrap)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rankings.to_csv(out_path, index=False)
    return out_path
