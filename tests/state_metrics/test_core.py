import pandas as pd
import pytest

from state_metrics.core import sweep_active_instances


def _log(rows):
    return pd.DataFrame(
        rows,
        columns=["case_id", "activity", "start_time", "end_time", "resource"],
    )


def _ts(h, m=0):
    return pd.Timestamp(f"2025-01-01 {h:02d}:{m:02d}", tz="UTC")


def test_sweep_yields_boundary_timestamps():
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

    seen = []
    for t, gt_bag, sim_bag in sweep_active_instances(gt, sim):
        seen.append((t, sorted((i.activity for i in gt_bag)), sorted((i.activity for i in sim_bag))))
    assert seen == [
        (t10, ["A"], ["A"]),
        (t12, ["B"], ["C"]),
        (t13, [], []),
    ]


def test_sweep_half_open_endpoint_not_active():
    """If an instance ends at t, it must not appear in the bag at t."""
    t10 = pd.Timestamp("2025-01-01 10:00", tz="UTC")
    t11 = pd.Timestamp("2025-01-01 11:00", tz="UTC")

    gt = _log([("c1", "A", t10, t11, "R1")])
    sim = _log([("c1", "A", t10, t11, "R1")])

    states = [(t, len(gb), len(sb)) for t, gb, sb in sweep_active_instances(gt, sim)]
    # At t10 the instance starts → 1 active. At t11 it ends → 0 active.
    assert states == [(t10, 1, 1), (t11, 0, 0)]


def test_sweep_validates_columns():
    good = _log([])
    bad = pd.DataFrame({"case_id": [], "activity": []})
    with pytest.raises(ValueError):
        list(sweep_active_instances(bad, good))


# ---- new: correctness invariants ---------------------------------------


def test_sweep_empty_logs_yields_nothing():
    """Sweep over two empty logs must produce no samples (not error)."""
    empty = _log([])
    assert list(sweep_active_instances(empty, empty)) == []


def test_sweep_rejects_nat_timestamps():
    t10 = _ts(10)
    bad = _log([("c1", "A", t10, pd.NaT, "R1")])
    good = _log([])
    with pytest.raises(ValueError, match="NaT"):
        list(sweep_active_instances(bad, good))


def test_sweep_rejects_negative_duration():
    t10, t11 = _ts(10), _ts(11)
    bad = _log([("c1", "A", t11, t10, "R1")])   # end < start
    good = _log([])
    with pytest.raises(ValueError, match="end_time"):
        list(sweep_active_instances(bad, good))


def test_sweep_skips_zero_duration_rows():
    """A zero-duration activity must not contaminate later samples.

    Regression: previously, start==end caused the instance to be added to
    the bag at t and never removed (its end is also t), so the bag would
    retain it for the rest of the sweep.
    """
    t10, t11, t12 = _ts(10), _ts(11), _ts(12)
    gt = _log([
        ("c1", "Z", t10, t10, "R1"),   # zero-duration — must be skipped
        ("c1", "A", t11, t12, "R1"),
    ])
    sim = _log([("c1", "A", t11, t12, "R1")])
    states = [(t, [i.activity for i in gb], [i.activity for i in sb])
              for t, gb, sb in sweep_active_instances(gt, sim)]
    # t10 must not appear at all — the zero-duration row contributes no
    # boundary. Sweep boundaries come only from the non-degenerate rows.
    assert states == [(t11, ["A"], ["A"]), (t12, [], [])]


def test_sweep_disjoint_time_ranges():
    """GT active 10–11, sim active 12–13 — one side is empty during the
    other's activity."""
    gt = _log([("c1", "A", _ts(10), _ts(11), "R1")])
    sim = _log([("c2", "B", _ts(12), _ts(13), "R2")])

    states = [(t, len(gb), len(sb)) for t, gb, sb in sweep_active_instances(gt, sim)]
    assert states == [
        (_ts(10), 1, 0),   # GT active, sim empty
        (_ts(11), 0, 0),   # GT just ended
        (_ts(12), 0, 1),   # sim starts
        (_ts(13), 0, 0),   # sim ends
    ]


def test_sweep_duplicate_tuple_instances_counted_separately():
    """Two rows with identical (case_id, activity, resource) but disjoint
    time windows create two *distinct* ActiveInstance objects; each is
    tracked by object identity so they never collide.
    """
    # Case 1 does activity A twice (loop), no overlap.
    gt = _log([
        ("c1", "A", _ts(10), _ts(11), "R1"),
        ("c1", "A", _ts(11), _ts(12), "R1"),
    ])
    sim = _log([])
    counts = [len(gb) for _, gb, _ in sweep_active_instances(gt, sim)]
    # At 10:00 one active, at 11:00 one active (first ended, second started),
    # at 12:00 zero. Key: they don't leak.
    assert counts == [1, 1, 0]


def test_sweep_overlapping_same_activity_same_case():
    """Two concurrent instances of (A, c1) — the bag must hold both."""
    gt = _log([
        ("c1", "A", _ts(10), _ts(12), "R1"),
        ("c1", "A", _ts(11), _ts(13), "R2"),
    ])
    sim = _log([])
    counts = [len(gb) for _, gb, _ in sweep_active_instances(gt, sim)]
    # 10→1, 11→2 (both active), 12→1, 13→0
    assert counts == [1, 2, 1, 0]


def test_sweep_bag_mutation_in_place():
    """Bags are mutated in place across yields — documented contract.
    Snapshotting the bag object (not its contents) should show it
    shrinking to empty by the final yield."""
    gt = _log([("c1", "A", _ts(10), _ts(11), "R1")])
    sim = _log([])
    snapshots = []
    for _, gb, _ in sweep_active_instances(gt, sim):
        snapshots.append(gb)     # same object reference each time
    assert snapshots[0] is snapshots[1]   # confirm identity
    # And after the sweep, it's empty.
    assert len(snapshots[-1]) == 0
