"""Unit tests for ranking statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.state_metrics.ranking import (
    bootstrap_ci,
    compute_rankings,
    concordance_index,
    kendall_on_means,
    spearman_on_means,
)


# ---- concordance_index ----------------------------------------------------


def test_concordance_index_perfect_monotonic():
    # Every replicate at a higher level beats every replicate at a lower level.
    vbl = {0: [0.0, 0.1], 1: [0.5, 0.6], 2: [1.0, 1.1]}
    assert concordance_index(vbl) == pytest.approx(1.0)


def test_concordance_index_perfect_inverted():
    vbl = {0: [1.0, 1.1], 1: [0.5, 0.6], 2: [0.0, 0.1]}
    assert concordance_index(vbl) == pytest.approx(0.0)


def test_concordance_index_random_constant_is_half():
    # All identical values → every cross-pair is a tie → 0.5.
    vbl = {0: [0.5, 0.5], 1: [0.5, 0.5], 2: [0.5, 0.5]}
    assert concordance_index(vbl) == pytest.approx(0.5)


def test_concordance_index_non_monotonic():
    # 0 < 2 < 1: middle level is artificially highest → mixed pairs.
    vbl = {0: [0.0], 1: [1.0], 2: [0.5]}
    # Pairs: (0,1) → 1.0 vs 0.0 → +1
    #        (0,2) → 0.5 vs 0.0 → +1
    #        (1,2) → 0.5 vs 1.0 → 0
    # 2/3 ≈ 0.667
    assert concordance_index(vbl) == pytest.approx(2 / 3)


def test_concordance_index_single_level_is_nan():
    assert np.isnan(concordance_index({0: [0.1, 0.2]}))


# ---- spearman / kendall on means ------------------------------------------


def test_spearman_perfect_monotonic():
    vbl = {0: [0.0, 0.1], 1: [0.5, 0.6], 2: [1.0, 1.1]}
    assert spearman_on_means(vbl) == pytest.approx(1.0)
    assert kendall_on_means(vbl) == pytest.approx(1.0)


def test_spearman_inverted():
    vbl = {0: [1.0], 1: [0.5], 2: [0.0]}
    assert spearman_on_means(vbl) == pytest.approx(-1.0)
    assert kendall_on_means(vbl) == pytest.approx(-1.0)


def test_correlation_on_flat_means_is_nan():
    vbl = {0: [0.5], 1: [0.5], 2: [0.5]}
    assert np.isnan(spearman_on_means(vbl))
    assert np.isnan(kendall_on_means(vbl))


def test_correlation_handles_non_monotonic():
    # Means 0.0, 1.0, 0.5 → Kendall τ = (1 concordant - 1 discordant) / 3.
    vbl = {0: [0.0], 1: [1.0], 2: [0.5]}
    assert kendall_on_means(vbl) == pytest.approx(1 / 3)


# ---- bootstrap CI ---------------------------------------------------------


def test_bootstrap_ci_bounds_point_estimate():
    rng = np.random.default_rng(42)
    vbl = {0: list(np.linspace(0, 0.2, 10)),
           1: list(np.linspace(0.4, 0.6, 10)),
           2: list(np.linspace(0.8, 1.0, 10))}
    point = concordance_index(vbl)
    low, high = bootstrap_ci(concordance_index, vbl, n_iter=300, rng=rng)
    assert low <= point + 1e-9
    assert high >= point - 1e-9
    assert low >= 0.5  # signal is strong; CI should not drop to chance.


def test_bootstrap_ci_degenerate_returns_nan_pair():
    # Flat distribution; spearman is undefined on every resample.
    vbl = {0: [0.5, 0.5], 1: [0.5, 0.5], 2: [0.5, 0.5]}
    low, high = bootstrap_ci(spearman_on_means, vbl, n_iter=50)
    assert np.isnan(low) and np.isnan(high)


# ---- compute_rankings ------------------------------------------------------


def _make_results_df():
    rows = []
    # cardinality grows monotonically with level; cycle_time is non-monotonic.
    for L, card_vals, ct_vals in [
        (0, [0.10, 0.12], [2.3, 2.4]),
        (1, [0.20, 0.22], [5.0, 5.1]),
        (2, [0.35, 0.37], [12.0, 12.1]),
        (3, [0.55, 0.57], [16.0, 15.5]),  # peaks here
        (4, [0.70, 0.72], [10.0, 9.5]),   # drops
    ]:
        for v in card_vals:
            rows.append({"level": L, "scope": "A_ongoing", "family": "state",
                         "metric": "cardinality", "value": v,
                         "k_baseline": 1, "k_sim": 1})
        for v in ct_vals:
            rows.append({"level": L, "scope": "A_ongoing", "family": "baseline",
                         "metric": "cycle_time", "value": v,
                         "k_baseline": 1, "k_sim": 1})
    return pd.DataFrame(rows)


def test_compute_rankings_separates_monotonic_from_blip():
    df = _make_results_df()
    out = compute_rankings(df, n_bootstrap=200, seed=1)
    by_metric = {(r["family"], r["metric"]): r for _, r in out.iterrows()}

    card = by_metric[("state", "cardinality")]
    cyc = by_metric[("baseline", "cycle_time")]

    assert card["c_index"] == pytest.approx(1.0)
    assert card["spearman"] == pytest.approx(1.0)
    assert card["kendall"] == pytest.approx(1.0)

    # cycle_time is monotonic for early levels then dips — both Spearman
    # and c-index should noticeably trail cardinality.
    assert cyc["c_index"] < card["c_index"]
    assert cyc["spearman"] < card["spearman"]

    # All CIs sane.
    for _, row in out.iterrows():
        for stat in ("c_index", "spearman", "kendall"):
            lo, hi = row[f"{stat}_ci_low"], row[f"{stat}_ci_high"]
            if not (np.isnan(lo) or np.isnan(hi)):
                assert lo <= hi


def test_compute_rankings_drops_nan_values():
    df = _make_results_df()
    # Inject NaNs; ensure no crash and counts reflect drop.
    df.loc[df["metric"] == "cardinality", "value"] = df.loc[
        df["metric"] == "cardinality", "value"
    ].mask(df["level"].isin([4]))
    out = compute_rankings(df, n_bootstrap=100, seed=1)
    card = out[(out["family"] == "state") & (out["metric"] == "cardinality")].iloc[0]
    assert card["n_levels"] == 4


def test_compute_rankings_requires_columns():
    df = pd.DataFrame({"level": [0], "scope": ["A"]})
    with pytest.raises(ValueError):
        compute_rankings(df)
