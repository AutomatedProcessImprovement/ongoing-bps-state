"""Ranking-quality statistics for the state-metrics pipeline.

The headline question shifted from "how big is the distance?" to "does the
metric rank perturbation levels correctly?". For each (scope, family, metric)
cell in ``results.csv`` we have ``runs`` independent distance draws per level.
This module turns that into three ranking-quality numbers:

* **Concordance index (c-index / AUC).** Probability that, across all
  cross-replicate pairs from two different levels, the metric assigns a
  larger value to the higher-perturbation replicate. ``0.5`` is random;
  ``1.0`` is perfect monotone ranking. The right statistic when seed-matched
  pairing is unavailable — it works directly over cross-pairs.
* **Spearman ρ on per-level means.** Pearson correlation between the level
  index (treated as the ground-truth ordinal) and the mean metric value at
  that level.
* **Kendall τ on per-level means.** Same idea, but counts concordant vs
  discordant pairs over level-means rather than ranks.

All three assume that the *perturbation level number itself* is the ground
truth ordering (true for both the "resources" ladder, integers = #resources
removed, and the "duration" ladder, integers = scale percentage above 1.0×).

All three are reported with percentile bootstrap CIs: replicates are
resampled with replacement *within each level*, the statistic is recomputed,
and the [α/2, 1-α/2] percentiles of the resampled values give the interval.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TypeAlias

import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats


# A mapping {level_int -> list of replicate metric values at that level}.
ValuesByLevel: TypeAlias = Mapping[int, list[float]]


def concordance_index(values_by_level: ValuesByLevel) -> float:
    """Probability that a higher-perturbation replicate beats a lower one.

    For every ordered pair of levels (Li, Lj) with Li < Lj and every
    cross-replicate pair (rep_i ∈ Li, rep_j ∈ Lj) count:
    ``+1`` if ``metric(rep_j) > metric(rep_i)``, ``+0.5`` on ties, ``0``
    otherwise. Divide by the total number of cross-pairs.
    """
    levels = sorted(values_by_level)
    if len(levels) < 2:
        return float("nan")

    score = 0.0
    total = 0
    for i, lo in enumerate(levels):
        vi = np.asarray(values_by_level[lo], dtype=float)
        if vi.size == 0:
            continue
        for hi in levels[i + 1:]:
            vj = np.asarray(values_by_level[hi], dtype=float)
            if vj.size == 0:
                continue
            diff = vj[:, None] - vi[None, :]
            score += float((diff > 0).sum()) + 0.5 * float((diff == 0).sum())
            total += int(diff.size)
    return score / total if total else float("nan")


def _correlation_on_means(values_by_level: ValuesByLevel, fn) -> float:
    levels = sorted(values_by_level)
    if len(levels) < 2:
        return float("nan")
    means = [float(np.mean(values_by_level[L])) for L in levels]
    # scipy emits a RuntimeWarning + nan when one side is constant; pre-check.
    if len(set(means)) < 2:
        return float("nan")
    stat, _ = fn(levels, means)
    return float(stat)


def spearman_on_means(values_by_level: ValuesByLevel) -> float:
    """Spearman ρ between the level index and the per-level mean metric."""
    return _correlation_on_means(values_by_level, _scipy_stats.spearmanr)


def kendall_on_means(values_by_level: ValuesByLevel) -> float:
    """Kendall τ between the level index and the per-level mean metric."""
    return _correlation_on_means(values_by_level, _scipy_stats.kendalltau)


def bootstrap_ci(
    stat_fn: Callable[[ValuesByLevel], float],
    values_by_level: ValuesByLevel,
    *,
    n_iter: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap CI for a ranking statistic.

    Resamples replicates *within each level* with replacement, recomputes
    ``stat_fn`` on the resampled dict, and returns ``(low, high)`` at the
    [α/2, 1-α/2] percentiles. NaN samples (e.g. when all per-level means
    collide on a resample) are dropped before percentile.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    arrays = {L: np.asarray(v, dtype=float) for L, v in values_by_level.items()}
    samples: list[float] = []
    for _ in range(n_iter):
        resampled = {
            L: rng.choice(arr, size=arr.size, replace=True).tolist()
            for L, arr in arrays.items() if arr.size > 0
        }
        s = stat_fn(resampled)
        if isinstance(s, float) and not np.isnan(s):
            samples.append(s)
    if not samples:
        return float("nan"), float("nan")
    low = float(np.percentile(samples, 100 * alpha / 2))
    high = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return low, high


def compute_rankings(
    results_df: pd.DataFrame,
    *,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """Compute c-index, Spearman, Kendall (with bootstrap CIs) per metric.

    ``results_df`` must have columns ``level``, ``scope``, ``family``,
    ``metric``, ``value``. Rows with NaN ``value`` are dropped. Returns one
    row per (scope, family, metric).
    """
    needed = {"level", "scope", "family", "metric", "value"}
    missing = needed - set(results_df.columns)
    if missing:
        raise ValueError(f"results_df is missing columns: {sorted(missing)}")

    rng = np.random.default_rng(seed)
    clean = results_df.dropna(subset=["value"]).copy()
    clean["level"] = clean["level"].astype(int)

    out_rows: list[dict] = []
    for (scope, family, metric), grp in clean.groupby(
        ["scope", "family", "metric"], sort=False
    ):
        vbl: dict[int, list[float]] = {
            int(L): grp.loc[grp["level"] == L, "value"].astype(float).tolist()
            for L in sorted(grp["level"].unique())
        }
        if len(vbl) < 2:
            continue

        c_idx = concordance_index(vbl)
        rho = spearman_on_means(vbl)
        tau = kendall_on_means(vbl)
        c_lo, c_hi = bootstrap_ci(
            concordance_index, vbl, n_iter=n_bootstrap, alpha=alpha, rng=rng,
        )
        rho_lo, rho_hi = bootstrap_ci(
            spearman_on_means, vbl, n_iter=n_bootstrap, alpha=alpha, rng=rng,
        )
        tau_lo, tau_hi = bootstrap_ci(
            kendall_on_means, vbl, n_iter=n_bootstrap, alpha=alpha, rng=rng,
        )

        out_rows.append({
            "scope": scope,
            "family": family,
            "metric": metric,
            "c_index": c_idx,
            "c_index_ci_low": c_lo,
            "c_index_ci_high": c_hi,
            "spearman": rho,
            "spearman_ci_low": rho_lo,
            "spearman_ci_high": rho_hi,
            "kendall": tau,
            "kendall_ci_low": tau_lo,
            "kendall_ci_high": tau_hi,
            "n_levels": len(vbl),
            "min_replicates_per_level": min(len(v) for v in vbl.values()),
        })

    return pd.DataFrame(out_rows)
