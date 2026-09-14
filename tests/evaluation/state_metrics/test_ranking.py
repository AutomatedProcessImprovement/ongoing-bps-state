"""Unit tests for ranking statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.state_metrics.ranking import (
    bootstrap_ci,
    compute_rankings,
    concordance_index,
    concordance_index_on_means,
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


# ---- concordance_index_on_means -------------------------------------------


def test_cindex_on_means_perfect_monotonic():
    vbl = {0: [0.1, 0.3], 1: [0.4, 0.6], 2: [0.7, 0.9]}
    assert concordance_index_on_means(vbl) == pytest.approx(1.0)


def test_cindex_on_means_averages_out_replicate_noise():
    # Replicate-level ranking is imperfect (the level-0 and level-1 draws
    # overlap) but the level MEANS are ordered, so the mean-first variant
    # scores 1.0 where the pairwise c-index does not.
    vbl = {0: [0.0, 0.4], 1: [0.1, 0.5], 2: [0.2, 0.6]}
    assert concordance_index(vbl) < 1.0
    assert concordance_index_on_means(vbl) == pytest.approx(1.0)


def test_cindex_on_means_constant_is_half():
    vbl = {0: [0.2, 0.2], 1: [0.2, 0.2], 2: [0.2, 0.2]}
    assert concordance_index_on_means(vbl) == pytest.approx(0.5)


def test_cindex_on_means_matches_kendall_transform():
    vbl = {0: [0.5], 1: [0.1], 2: [0.9], 3: [0.3]}
    assert concordance_index_on_means(vbl) == pytest.approx(
        (kendall_on_means(vbl) + 1) / 2
    )


def test_cindex_on_means_single_level_is_nan():
    assert np.isnan(concordance_index_on_means({0: [0.1, 0.2]}))


# ---- seed stability across processes ---------------------------------------


def test_stable_seed_is_deterministic_across_processes():
    """The bootstrap seed must not depend on PYTHONHASHSEED.

    ``abs(hash((...)))`` was used here originally. CPython randomises string
    hashing per process, so every interpreter produced different seeds and the
    reported confidence intervals moved between runs on identical data.
    """
    import subprocess
    import sys

    code = (
        "import sys; sys.path.insert(0, '.');"
        "from evaluation.state_metrics.ranking import _stable_seed;"
        "print(_stable_seed(0, 'A_ongoing', 'baseline', 'ngd_n2'))"
    )
    seeds = {
        subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True,
        ).stdout.strip()
        for _ in range(3)
    }
    assert len(seeds) == 1, f"seed differs between interpreters: {seeds}"


def test_compute_rankings_ci_reproduces_across_processes(tmp_path):
    """Two independent interpreters must report identical intervals."""
    import subprocess
    import sys

    csv = tmp_path / "results.csv"
    rows = []
    for level in range(4):
        for k in range(5):
            rows.append(
                {"level": level, "scope": "A_ongoing", "family": "state",
                 "metric": "activity.jaccard", "value": 0.1 * level + 0.01 * k}
            )
    pd.DataFrame(rows).to_csv(csv, index=False)

    code = (
        "import sys; sys.path.insert(0, '.');"
        "import pandas as pd;"
        "from evaluation.state_metrics.ranking import compute_rankings;"
        f"df = compute_rankings(pd.read_csv(r'{csv}'), n_bootstrap=200);"
        "print(round(float(df['c_index_means_ci_low'].iloc[0]), 12),"
        "      round(float(df['c_index_means_ci_high'].iloc[0]), 12))"
    )
    out = {
        subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True,
        ).stdout.strip()
        for _ in range(3)
    }
    assert len(out) == 1, f"CI differs between interpreters: {out}"
