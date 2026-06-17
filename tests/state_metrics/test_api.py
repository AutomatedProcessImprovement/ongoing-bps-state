import pandas as pd
import pytest

from state_metrics import (
    StateDistanceResult,
    compute_all_state_distances,
    compute_state_distance,
)


def _log(rows):
    return pd.DataFrame(
        rows,
        columns=["case_id", "activity", "start_time", "end_time", "resource"],
    )


@pytest.fixture
def logs_activity_diverge():
    """Two logs identical on [10:00, 12:00) and fully divergent on [12:00, 13:00)."""
    t10 = pd.Timestamp("2025-01-01 10:00", tz="UTC")
    t12 = pd.Timestamp("2025-01-01 12:00", tz="UTC")
    t13 = pd.Timestamp("2025-01-01 13:00", tz="UTC")
    gt = _log([
        ("c1", "A", t10, t12, "R1"),
        ("c1", "B", t12, t13, "R2"),
    ])
    sim = _log([
        ("c1", "A", t10, t12, "R1"),
        ("c1", "C", t12, t13, "R2"),
    ])
    return gt, sim


def test_identical_logs_yield_zero_distance(logs_activity_diverge):
    gt, _ = logs_activity_diverge
    res = compute_all_state_distances(gt, gt)
    assert len(res) == 6  # 3 views * 2 distances (no role_map → no activity_role)
    for key, r in res.items():
        assert r.summary_distance == 0.0, key
        assert all(d == 0.0 for _, d in r.timeline), key


def test_expected_summary_by_hand(logs_activity_diverge):
    """
    Intervals:
      [10:00, 12:00) = 7200s, activity view: {A:1} vs {A:1} → jaccard 0
      [12:00, 13:00) = 3600s, activity view: {B:1} vs {C:1} → jaccard 1
    Weighted:  (0*7200 + 1*3600) / 10800 = 1/3
    """
    gt, sim = logs_activity_diverge
    r = compute_state_distance(gt, sim, view="activity", distance="jaccard")
    assert r.summary_distance == pytest.approx(1 / 3)
    assert r.total_duration_seconds == 10800.0
    assert r.view == "activity"
    assert r.distance_function == "jaccard"


def test_cardinality_zero_when_sizes_equal(logs_activity_diverge):
    gt, sim = logs_activity_diverge
    r = compute_state_distance(gt, sim, view="activity", distance="cardinality")
    # Both logs have exactly 1 active instance at every interval.
    assert r.summary_distance == 0.0


def test_compute_all_includes_activity_role_when_role_map_given(logs_activity_diverge):
    gt, sim = logs_activity_diverge
    role_map = {"R1": "junior", "R2": "senior"}
    res = compute_all_state_distances(gt, sim, role_map=role_map)
    assert len(res) == 8
    assert ("activity_role", "jaccard") in res
    assert ("activity_role", "cardinality") in res


def test_result_is_dataclass(logs_activity_diverge):
    gt, sim = logs_activity_diverge
    r = compute_state_distance(gt, sim, view="case", distance="jaccard")
    assert isinstance(r, StateDistanceResult)
    assert isinstance(r.timeline, list)
    assert len(r.timeline) >= 1


# ---- new: invariants and broader correctness --------------------------


def test_disjoint_logs_yield_distance_one():
    """Two logs that are never simultaneously non-empty → jaccard 1
    during active intervals."""
    t10, t11, t12, t13 = (pd.Timestamp(f"2025-01-01 {h}:00", tz="UTC")
                          for h in (10, 11, 12, 13))
    gt = _log([("c1", "A", t10, t11, "R1")])
    sim = _log([("c2", "B", t12, t13, "R2")])
    r = compute_state_distance(gt, sim, view="activity", distance="jaccard")
    # Timeline: 10→(A vs ∅, d=1), 11→(∅ vs ∅, d=0), 12→(∅ vs B, d=1), 13→(∅ vs ∅)
    # Weights: 1h, 1h, 1h.  Weighted sum: 1*1 + 0*1 + 1*1 = 2 / 3.
    assert r.summary_distance == pytest.approx(2 / 3)


def test_cardinality_is_view_invariant(logs_activity_diverge):
    """cardinality ignores the projection key and only sees total bag
    size, so it must return the same value across all views (given a
    complete role_map)."""
    gt, sim = logs_activity_diverge
    role_map = {"R1": "junior", "R2": "senior"}
    res = compute_all_state_distances(gt, sim, role_map=role_map)
    card_values = {
        view: res[(view, "cardinality")].summary_distance
        for view, _ in res.keys() if _ == "cardinality"
    }
    # All four views produce identical cardinality summaries.
    assert len(set(card_values.values())) == 1


def test_summary_three_intervals_by_hand():
    """Full worked example: non-trivial 3-interval case.

    GT:  A on c1 for [10, 12)
         B on c1 for [12, 13)
    Sim: A on c1 for [10, 11)
         X on c2 for [11, 13)

    Sweep timestamps (boundaries): 10, 11, 12, 13.
    Bags at each t (after ends-then-starts at t):
      10: gt={(A,c1)}               sim={(A,c1)}               d_activity=0
      11: gt={(A,c1)}               sim={(X,c2)}               d_activity=1
      12: gt={(B,c1)}               sim={(X,c2)}               d_activity=1
      13: gt={}                     sim={}                     d_activity=0

    Weights (gap to next t): [1h, 1h, 1h] = 3600*3 = 10800s.
    Weighted sum: 0*3600 + 1*3600 + 1*3600 = 7200s.
    Summary distance: 7200/10800 = 2/3.
    """
    t10 = pd.Timestamp("2025-01-01 10:00", tz="UTC")
    t11 = pd.Timestamp("2025-01-01 11:00", tz="UTC")
    t12 = pd.Timestamp("2025-01-01 12:00", tz="UTC")
    t13 = pd.Timestamp("2025-01-01 13:00", tz="UTC")
    gt = _log([
        ("c1", "A", t10, t12, "R1"),
        ("c1", "B", t12, t13, "R2"),
    ])
    sim = _log([
        ("c1", "A", t10, t11, "R1"),
        ("c2", "X", t11, t13, "R3"),
    ])
    r = compute_state_distance(gt, sim, view="activity", distance="jaccard")
    assert r.summary_distance == pytest.approx(2 / 3)
    assert r.total_duration_seconds == 10800.0
    # Timeline returns all 4 samples; the last one has weight 0 in the sum.
    ts_in_timeline = [t for t, _ in r.timeline]
    assert ts_in_timeline == [t10, t11, t12, t13]


def test_empty_logs_yield_zero_distance():
    """Both logs empty → every metric is 0 (nothing to compare)."""
    empty = _log([])
    res = compute_all_state_distances(empty, empty)
    for key, r in res.items():
        assert r.summary_distance == 0.0, key
        assert r.total_duration_seconds == 0.0, key
        assert r.timeline == [], key


def test_single_side_empty_yields_distance_one_for_jaccard():
    t10 = pd.Timestamp("2025-01-01 10:00", tz="UTC")
    t11 = pd.Timestamp("2025-01-01 11:00", tz="UTC")
    gt = _log([("c1", "A", t10, t11, "R1")])
    sim = _log([])
    # GT has activity from 10 to 11. Sim is empty.
    # Sweep touches 10 (gt={A}, sim=∅, d=1) and 11 (both empty, d=0).
    # Weight on t10 = 1h = 3600s; total = 3600s. Summary = 1.0.
    r = compute_state_distance(gt, sim, view="activity", distance="jaccard")
    assert r.summary_distance == pytest.approx(1.0)
    assert r.total_duration_seconds == 3600.0


def test_unknown_view_or_distance_raises():
    empty = _log([])
    with pytest.raises(ValueError, match="unknown view"):
        compute_state_distance(empty, empty, view="bogus", distance="jaccard")
    with pytest.raises(ValueError, match="unknown distance"):
        compute_state_distance(empty, empty, view="activity", distance="bogus")


def test_window_matches_active_span_equals_default(logs_activity_diverge):
    """A window covering exactly the active span must reproduce the
    default (no-window) summary, since the denominator is identical."""
    gt, sim = logs_activity_diverge
    t10 = pd.Timestamp("2025-01-01 10:00", tz="UTC")
    t13 = pd.Timestamp("2025-01-01 13:00", tz="UTC")

    default = compute_state_distance(gt, sim, view="activity", distance="jaccard")
    windowed = compute_state_distance(
        gt, sim, view="activity", distance="jaccard", window=(t10, t13)
    )
    assert windowed.summary_distance == pytest.approx(default.summary_distance)
    assert windowed.total_duration_seconds == pytest.approx(
        default.total_duration_seconds
    )


def test_window_wider_than_active_rescales_summary(logs_activity_diverge):
    """A window twice the active span must halve the summary distance —
    the empty extra time contributes 0 to the numerator but adds to the
    denominator."""
    gt, sim = logs_activity_diverge
    # Active span is [10:00, 13:00) = 3h. Window is [09:00, 15:00) = 6h.
    t9 = pd.Timestamp("2025-01-01 09:00", tz="UTC")
    t15 = pd.Timestamp("2025-01-01 15:00", tz="UTC")

    default = compute_state_distance(gt, sim, view="activity", distance="jaccard")
    windowed = compute_state_distance(
        gt, sim, view="activity", distance="jaccard", window=(t9, t15)
    )
    # default = 1/3, with 2× window denominator → 1/6.
    assert windowed.summary_distance == pytest.approx(default.summary_distance / 2)
    assert windowed.total_duration_seconds == pytest.approx(6 * 3600)


def test_window_applies_to_all_view_distance_pairs(logs_activity_diverge):
    """compute_all_state_distances must also honor the window."""
    gt, sim = logs_activity_diverge
    t9 = pd.Timestamp("2025-01-01 09:00", tz="UTC")
    t15 = pd.Timestamp("2025-01-01 15:00", tz="UTC")

    res = compute_all_state_distances(gt, sim, window=(t9, t15))
    for r in res.values():
        assert r.total_duration_seconds == pytest.approx(6 * 3600)


def test_window_with_empty_logs_returns_zero_over_full_window():
    """Empty logs + explicit window: summary 0, duration = window length."""
    empty = _log([])
    t10 = pd.Timestamp("2025-01-01 10:00", tz="UTC")
    t13 = pd.Timestamp("2025-01-01 13:00", tz="UTC")
    r = compute_state_distance(
        empty, empty, view="activity", distance="jaccard", window=(t10, t13)
    )
    assert r.summary_distance == 0.0
    assert r.total_duration_seconds == pytest.approx(3 * 3600)


def test_window_invalid_raises():
    empty = _log([])
    t10 = pd.Timestamp("2025-01-01 10:00", tz="UTC")
    t9 = pd.Timestamp("2025-01-01 09:00", tz="UTC")
    with pytest.raises(ValueError, match="window end must be > start"):
        compute_state_distance(
            empty, empty, view="activity", distance="jaccard", window=(t10, t9)
        )
    with pytest.raises(ValueError, match="window end must be > start"):
        compute_state_distance(
            empty, empty, view="activity", distance="jaccard", window=(t10, t10)
        )


def test_role_map_opt_out_excludes_activity_role():
    """Without a role_map, activity_role is silently skipped by
    compute_all_state_distances (it requires one)."""
    t10 = pd.Timestamp("2025-01-01 10:00", tz="UTC")
    t11 = pd.Timestamp("2025-01-01 11:00", tz="UTC")
    gt = _log([("c1", "A", t10, t11, "R1")])
    sim = _log([("c1", "A", t10, t11, "R1")])
    res = compute_all_state_distances(gt, sim)    # no role_map
    assert not any(v == "activity_role" for v, _ in res.keys())
    # With role_map, it's included.
    res = compute_all_state_distances(gt, sim, role_map={"R1": "j"})
    assert any(v == "activity_role" for v, _ in res.keys())
