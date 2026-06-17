import json

import numpy as np
import pandas as pd
import pytest

from evaluation.state_metrics.pipeline import (
    _compute_horizon,
    _compute_utilization_rows,
    _count_split_gateways,
    _load_prosimos_log,
    _select_fraction_of_log_cutoff,
    _select_high_wip_cutoff,
    _select_n_ongoing_cutoff,
    _write_prefix_csv,
    filter_to_cases_in_window,
    get_ongoing_case_ids,
    label_swap_case_types,
    merge_logs_at_ratio,
    relabel_activity_fraction,
    rephase_cases,
)


def _mklog(rows):
    df = pd.DataFrame(
        rows,
        columns=["case_id", "activity", "start_time", "end_time", "resource"],
    )
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
    df["end_time"] = pd.to_datetime(df["end_time"], utc=True)
    return df


def test_select_high_wip_cutoff_is_inside_log():
    log = _mklog([
        ("c1", "A", "2025-01-01 10:00", "2025-01-01 15:00", "R1"),
        ("c2", "A", "2025-01-01 11:00", "2025-01-01 16:00", "R1"),
        ("c3", "A", "2025-01-01 12:00", "2025-01-01 17:00", "R1"),
        ("c4", "A", "2025-01-01 13:00", "2025-01-01 14:00", "R1"),
    ])
    cut = _select_high_wip_cutoff(log)
    assert log["start_time"].min() <= cut <= log["end_time"].max()


def test_compute_horizon_positive():
    log = _mklog([
        ("c1", "A", "2025-01-01 10:00", "2025-01-01 12:00", "R1"),
        ("c2", "A", "2025-01-01 11:00", "2025-01-01 14:00", "R1"),
    ])
    cut = pd.Timestamp("2025-01-01 11:30", tz="UTC")
    h = _compute_horizon(log, cut)
    assert h > pd.Timedelta(0)


def test_get_ongoing_case_ids():
    log = _mklog([
        # c1 ongoing at cutoff.
        ("c1", "A", "2025-01-01 09:00", "2025-01-01 11:00", "R1"),
        # c2 starts after cutoff.
        ("c2", "A", "2025-01-01 11:00", "2025-01-01 12:00", "R1"),
        # c3 finishes before cutoff.
        ("c3", "A", "2025-01-01 08:00", "2025-01-01 09:00", "R1"),
    ])
    cut = pd.Timestamp("2025-01-01 10:00", tz="UTC")
    assert get_ongoing_case_ids(log, cut) == {"c1"}


def test_filter_to_cases_in_window_clips_boundaries():
    log = _mklog([
        ("c1", "A", "2025-01-01 09:00", "2025-01-01 11:00", "R1"),
        ("c1", "B", "2025-01-01 11:00", "2025-01-01 13:00", "R2"),
        ("c2", "A", "2025-01-01 11:00", "2025-01-01 12:00", "R1"),
    ])
    cut = pd.Timestamp("2025-01-01 10:00", tz="UTC")
    horizon = pd.Timedelta(hours=4)
    out = filter_to_cases_in_window(log, {"c1"}, cut, horizon)
    assert set(out["case_id"]) == {"c1"}
    assert (out["start_time"] >= cut).all()
    assert (out["end_time"] <= cut + horizon).all()


def test_filter_to_cases_in_window_empty_if_no_overlap():
    log = _mklog([
        ("c1", "A", "2025-01-01 08:00", "2025-01-01 09:00", "R1"),
    ])
    cut = pd.Timestamp("2025-01-01 12:00", tz="UTC")
    out = filter_to_cases_in_window(log, {"c1"}, cut, pd.Timedelta(hours=1))
    assert out.empty


def test_select_fraction_of_log_cutoff_midpoint():
    log = _mklog([
        ("c1", "A", "2025-01-01 00:00", "2025-01-01 02:00", "R1"),
        ("c2", "A", "2025-01-01 04:00", "2025-01-01 06:00", "R1"),
    ])
    cut = _select_fraction_of_log_cutoff(log, fraction=0.5)
    assert cut == pd.Timestamp("2025-01-01 03:00", tz="UTC")


def test_select_n_ongoing_cutoff_in_band():
    # Build a log where WIP rises from 1 to 3 then falls back.
    log = _mklog([
        ("c1", "A", "2025-01-01 00:00", "2025-01-01 02:00", "R1"),  # ongoing 00:00-02:00
        ("c2", "A", "2025-01-01 01:00", "2025-01-01 03:00", "R1"),  # +1 from 01:00
        ("c3", "A", "2025-01-01 01:30", "2025-01-01 03:30", "R1"),  # +1 from 01:30, WIP=3
    ])
    # At target=3, only the 01:30 - 02:00 interval has WIP==3. Closest match.
    cut = _select_n_ongoing_cutoff(log, target=3)
    assert pd.Timestamp("2025-01-01 01:30", tz="UTC") <= cut < pd.Timestamp("2025-01-01 02:00", tz="UTC")


def test_select_n_ongoing_cutoff_falls_back_to_closest():
    # WIP never exceeds 2, target=10.
    log = _mklog([
        ("c1", "A", "2025-01-01 00:00", "2025-01-01 02:00", "R1"),
        ("c2", "A", "2025-01-01 01:00", "2025-01-01 03:00", "R1"),
    ])
    cut = _select_n_ongoing_cutoff(log, target=10)
    # Should pick the timestamp where WIP is maximised (2).
    assert pd.Timestamp("2025-01-01 01:00", tz="UTC") <= cut <= pd.Timestamp("2025-01-01 02:00", tz="UTC")


def test_select_fraction_of_log_cutoff_rejects_invalid_fraction():
    log = _mklog([
        ("c1", "A", "2025-01-01 00:00", "2025-01-01 01:00", "R1"),
    ])
    with pytest.raises(ValueError):
        _select_fraction_of_log_cutoff(log, fraction=0.0)
    with pytest.raises(ValueError):
        _select_fraction_of_log_cutoff(log, fraction=1.0)


def test_relabel_activity_fraction_zero_is_identity():
    log = _mklog([
        ("c1", "X", "2025-01-01 10:00", "2025-01-01 11:00", "R1"),
        ("c2", "Y", "2025-01-01 10:00", "2025-01-01 11:00", "R1"),
    ])
    rng = np.random.default_rng(0)
    out = relabel_activity_fraction(
        log, activity="X", to_label="X__alt", fraction=0.0, rng=rng,
    )
    assert out.equals(log)


def test_relabel_activity_fraction_count_and_invariants():
    log = _mklog([
        (str(i), "X", "2025-01-01 10:00", "2025-01-01 11:00", f"R{i}")
        for i in range(10)
    ] + [
        (str(100 + i), "Y", "2025-01-01 10:00", "2025-01-01 11:00", "R0")
        for i in range(5)
    ])
    rng = np.random.default_rng(1)
    out = relabel_activity_fraction(
        log, activity="X", to_label="X__alt", fraction=0.3, rng=rng,
    )
    # 30% of 10 X-instances relabeled; Y untouched.
    assert (out["activity"] == "X__alt").sum() == 3
    assert (out["activity"] == "X").sum() == 7
    assert (out["activity"] == "Y").sum() == 5
    # Timestamps, cases, and resources are invariant (only the label changes),
    # which is what makes every timing-based metric blind by construction.
    for col in ("case_id", "start_time", "end_time", "resource"):
        assert (out[col].values == log[col].values).all()


def test_relabel_activity_fraction_absent_activity_is_noop():
    log = _mklog([
        ("c1", "X", "2025-01-01 10:00", "2025-01-01 11:00", "R1"),
    ])
    rng = np.random.default_rng(0)
    out = relabel_activity_fraction(
        log, activity="ZZZ", to_label="ZZZ__alt", fraction=0.5, rng=rng,
    )
    assert out.equals(log)


def test_rephase_cases_zero_jitter_is_identity():
    log = _mklog([
        ("c1", "A", "2025-01-01 10:00", "2025-01-01 11:00", "R1"),
        ("c1", "B", "2025-01-01 11:00", "2025-01-01 13:00", "R2"),
        ("c2", "A", "2025-01-01 11:00", "2025-01-01 12:00", "R1"),
    ])
    cut = pd.Timestamp("2025-01-01 10:00", tz="UTC")
    horizon_end = cut + pd.Timedelta(hours=24)
    out = rephase_cases(
        log, jitter_hours=0.0, rng=np.random.default_rng(0),
        cutoff=cut, horizon_end=horizon_end,
    )
    assert out.equals(log)


def test_rephase_cases_preserves_per_case_invariants():
    log = _mklog([
        ("c1", "A", "2025-01-01 02:00", "2025-01-01 03:00", "R1"),
        ("c1", "B", "2025-01-01 04:00", "2025-01-01 05:30", "R2"),
        ("c2", "X", "2025-01-01 02:30", "2025-01-01 03:30", "R1"),
        ("c2", "Y", "2025-01-01 06:00", "2025-01-01 07:00", "R3"),
    ])
    cut = pd.Timestamp("2025-01-01 00:00", tz="UTC")
    horizon_end = pd.Timestamp("2025-01-01 12:00", tz="UTC")
    out = rephase_cases(
        log, jitter_hours=2.0, rng=np.random.default_rng(7),
        cutoff=cut, horizon_end=horizon_end,
    )
    # Identity-preserving columns (case_id, activity, resource).
    for col in ("case_id", "activity", "resource"):
        assert (out[col].values == log[col].values).all()
    # Per-case duration and per-case gaps (which encode bigram timing and
    # relative-event-distribution position) are invariant: a uniform Δ_c
    # leaves all within-case time differences unchanged.
    for cid, sub in log.groupby("case_id"):
        sub_out = out[out["case_id"] == cid]
        diffs_in = sub["end_time"].to_numpy() - sub["start_time"].to_numpy()
        diffs_out = sub_out["end_time"].to_numpy() - sub_out["start_time"].to_numpy()
        assert (diffs_in == diffs_out).all()
        case_dur_in = sub["end_time"].max() - sub["start_time"].min()
        case_dur_out = sub_out["end_time"].max() - sub_out["start_time"].min()
        assert case_dur_in == case_dur_out
        # Within-case event spacing (start_i+1 - start_i) is the same too.
        spacing_in = np.diff(sub["start_time"].astype("int64").to_numpy())
        spacing_out = np.diff(sub_out["start_time"].astype("int64").to_numpy())
        assert (spacing_in == spacing_out).all()


def test_rephase_cases_stays_in_window():
    log = _mklog([
        ("c1", "A", "2025-01-01 02:00", "2025-01-01 03:00", "R1"),
        ("c1", "B", "2025-01-01 04:00", "2025-01-01 05:30", "R2"),
        ("c2", "X", "2025-01-01 02:30", "2025-01-01 11:30", "R1"),
    ])
    cut = pd.Timestamp("2025-01-01 00:00", tz="UTC")
    horizon_end = pd.Timestamp("2025-01-01 12:00", tz="UTC")
    out = rephase_cases(
        log, jitter_hours=100.0, rng=np.random.default_rng(42),
        cutoff=cut, horizon_end=horizon_end,
    )
    assert (out["start_time"] >= cut).all()
    assert (out["end_time"] <= horizon_end).all()


def test_rephase_cases_offsets_vary_across_cases():
    # Multiple cases with plenty of headroom → uniform draws yield distinct
    # per-case offsets; therefore at least one case's start_time changes.
    log = _mklog([
        ("c1", "A", "2025-01-01 04:00", "2025-01-01 05:00", "R1"),
        ("c2", "A", "2025-01-01 04:00", "2025-01-01 05:00", "R1"),
        ("c3", "A", "2025-01-01 04:00", "2025-01-01 05:00", "R1"),
        ("c4", "A", "2025-01-01 04:00", "2025-01-01 05:00", "R1"),
    ])
    cut = pd.Timestamp("2025-01-01 00:00", tz="UTC")
    horizon_end = pd.Timestamp("2025-01-01 12:00", tz="UTC")
    out = rephase_cases(
        log, jitter_hours=1.0, rng=np.random.default_rng(123),
        cutoff=cut, horizon_end=horizon_end,
    )
    # At least one case got shifted (probability ~1 with 4 cases × Uniform).
    assert not (out["start_time"].values == log["start_time"].values).all()
    # Each per-case offset is consistent: start_time delta equals end_time delta.
    deltas_start = (out["start_time"] - log["start_time"]).dt.total_seconds()
    deltas_end = (out["end_time"] - log["end_time"]).dt.total_seconds()
    assert (deltas_start.values == deltas_end.values).all()


def test_merge_logs_at_ratio_case_counts_and_tags():
    green = _mklog([
        ("g1", "A", "2025-01-01 00:00", "2025-01-01 01:00", "R1"),
        ("g2", "A", "2025-01-01 02:00", "2025-01-01 03:00", "R1"),
        ("g3", "A", "2025-01-01 04:00", "2025-01-01 05:00", "R1"),
        ("g4", "A", "2025-01-01 06:00", "2025-01-01 07:00", "R1"),
    ])
    red = _mklog([
        ("r1", "B", "2025-01-01 00:30", "2025-01-01 01:30", "R2"),
        ("r2", "B", "2025-01-01 02:30", "2025-01-01 03:30", "R2"),
        ("r3", "B", "2025-01-01 04:30", "2025-01-01 05:30", "R2"),
        ("r4", "B", "2025-01-01 06:30", "2025-01-01 07:30", "R2"),
    ])
    out = merge_logs_at_ratio(green, red, fraction_green=0.5, target_cases=4)
    # 2 green + 2 red.
    assert (out["case_type"] == "green").sum() == 2
    assert (out["case_type"] == "red").sum() == 2
    # IDs are namespaced — no collision possible.
    assert all(cid.startswith(("g_", "r_")) for cid in out["case_id"])
    # Events sorted by start_time.
    assert out["start_time"].is_monotonic_increasing


def test_merge_logs_at_ratio_deterministic():
    green = _mklog([
        ("g1", "A", "2025-01-01 00:00", "2025-01-01 01:00", "R1"),
        ("g2", "A", "2025-01-01 02:00", "2025-01-01 03:00", "R1"),
    ])
    red = _mklog([
        ("r1", "B", "2025-01-01 00:30", "2025-01-01 01:30", "R2"),
        ("r2", "B", "2025-01-01 02:30", "2025-01-01 03:30", "R2"),
    ])
    a = merge_logs_at_ratio(green, red, fraction_green=0.5, target_cases=2)
    b = merge_logs_at_ratio(green, red, fraction_green=0.5, target_cases=2)
    # Same inputs → identical outputs; this is what makes level=0 in the
    # mix_ratio oracle give an exactly-zero noise floor.
    assert a.equals(b)


def test_merge_logs_at_ratio_extreme_ratios():
    green = _mklog([
        ("g1", "A", "2025-01-01 00:00", "2025-01-01 01:00", "R1"),
        ("g2", "A", "2025-01-01 02:00", "2025-01-01 03:00", "R1"),
    ])
    red = _mklog([
        ("r1", "B", "2025-01-01 00:30", "2025-01-01 01:30", "R2"),
        ("r2", "B", "2025-01-01 02:30", "2025-01-01 03:30", "R2"),
    ])
    all_green = merge_logs_at_ratio(green, red, fraction_green=1.0, target_cases=2)
    assert (all_green["case_type"] == "green").all()
    all_red = merge_logs_at_ratio(green, red, fraction_green=0.0, target_cases=2)
    assert (all_red["case_type"] == "red").all()


def test_merge_logs_at_ratio_rejects_invalid_inputs():
    green = _mklog([("g1", "A", "2025-01-01 00:00", "2025-01-01 01:00", "R1")])
    red = _mklog([("r1", "B", "2025-01-01 00:30", "2025-01-01 01:30", "R2")])
    with pytest.raises(ValueError):
        merge_logs_at_ratio(green, red, fraction_green=-0.1, target_cases=2)
    with pytest.raises(ValueError):
        merge_logs_at_ratio(green, red, fraction_green=1.5, target_cases=2)
    with pytest.raises(ValueError):
        merge_logs_at_ratio(green, red, fraction_green=0.5, target_cases=0)
    with pytest.raises(ValueError):
        # Requesting more green than exist in source.
        merge_logs_at_ratio(green, red, fraction_green=1.0, target_cases=10)


def _mklog_typed(rows):
    df = pd.DataFrame(
        rows,
        columns=["case_id", "activity", "start_time", "end_time", "resource", "case_type"],
    )
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
    df["end_time"] = pd.to_datetime(df["end_time"], utc=True)
    return df


def test_label_swap_zero_is_identity():
    log = _mklog_typed([
        ("c1", "A", "2025-01-01 10:00", "2025-01-01 11:00", "R1", "green"),
        ("c2", "A", "2025-01-01 10:00", "2025-01-01 11:00", "R1", "red"),
    ])
    out = label_swap_case_types(log, fraction=0.0, rng=np.random.default_rng(0))
    assert out.equals(log)


def test_label_swap_preserves_marginal_proportion():
    log = _mklog_typed([
        ("c1", "A", "2025-01-01 10:00", "2025-01-01 11:00", "R1", "green"),
        ("c2", "A", "2025-01-01 10:00", "2025-01-01 11:00", "R1", "green"),
        ("c3", "A", "2025-01-01 10:00", "2025-01-01 11:00", "R1", "green"),
        ("c4", "A", "2025-01-01 10:00", "2025-01-01 11:00", "R1", "green"),
        ("c5", "A", "2025-01-01 10:00", "2025-01-01 11:00", "R1", "red"),
        ("c6", "A", "2025-01-01 10:00", "2025-01-01 11:00", "R1", "red"),
        ("c7", "A", "2025-01-01 10:00", "2025-01-01 11:00", "R1", "red"),
        ("c8", "A", "2025-01-01 10:00", "2025-01-01 11:00", "R1", "red"),
    ])
    out = label_swap_case_types(log, fraction=0.5, rng=np.random.default_rng(1))
    # 50% of min(4,4)=4 → 2 swaps each side → marginal exactly preserved.
    assert (out["case_type"] == "green").sum() == 4
    assert (out["case_type"] == "red").sum() == 4


def test_label_swap_leaves_events_untouched():
    log = _mklog_typed([
        ("c1", "A", "2025-01-01 10:00", "2025-01-01 11:00", "R1", "green"),
        ("c2", "B", "2025-01-01 10:30", "2025-01-01 11:30", "R2", "red"),
    ])
    out = label_swap_case_types(log, fraction=1.0, rng=np.random.default_rng(2))
    # Everything except case_type identical.
    for col in ("case_id", "activity", "start_time", "end_time", "resource"):
        assert (out[col].values == log[col].values).all()
    # Both got flipped.
    assert out.loc[out["case_id"] == "c1", "case_type"].iloc[0] == "red"
    assert out.loc[out["case_id"] == "c2", "case_type"].iloc[0] == "green"


def test_label_swap_requires_case_type_column():
    log = _mklog([
        ("c1", "A", "2025-01-01 10:00", "2025-01-01 11:00", "R1"),
    ])
    with pytest.raises(ValueError):
        label_swap_case_types(log, fraction=0.5, rng=np.random.default_rng(0))


def test_label_swap_auto_detects_red_blue():
    # The case_route oracle tags cases red/blue, not green/red; auto-detection
    # must handle any binary tagging without an explicit `values` argument.
    log = _mklog_typed([
        ("c1", "A", "2025-01-01 10:00", "2025-01-01 11:00", "R1", "red"),
        ("c2", "A", "2025-01-01 10:00", "2025-01-01 11:00", "R1", "red"),
        ("c3", "A", "2025-01-01 10:00", "2025-01-01 11:00", "R1", "blue"),
        ("c4", "A", "2025-01-01 10:00", "2025-01-01 11:00", "R1", "blue"),
    ])
    out = label_swap_case_types(log, fraction=1.0, rng=np.random.default_rng(3))
    # Balanced flip preserves the marginal exactly.
    assert (out["case_type"] == "red").sum() == 2
    assert (out["case_type"] == "blue").sum() == 2
    # Only the case_type column changed.
    for col in ("case_id", "activity", "start_time", "end_time", "resource"):
        assert (out[col].values == log[col].values).all()


def test_label_swap_three_values_raises():
    log = _mklog_typed([
        ("c1", "A", "2025-01-01 10:00", "2025-01-01 11:00", "R1", "red"),
        ("c2", "A", "2025-01-01 10:00", "2025-01-01 11:00", "R1", "blue"),
        ("c3", "A", "2025-01-01 10:00", "2025-01-01 11:00", "R1", "green"),
    ])
    with pytest.raises(ValueError, match="exactly two case_type"):
        label_swap_case_types(log, fraction=0.5, rng=np.random.default_rng(0))


def test_count_split_gateways(tmp_path):
    params = {"gateway_branching_probabilities": [
        {"gateway_id": "g1s", "probabilities": [
            {"path_id": "f_g1s_a", "value": "0.5"}, {"path_id": "f_g1s_b", "value": "0.5"}]},
        {"gateway_id": "g1j", "probabilities": [{"path_id": "f_g1j_x", "value": "1"}]},
        {"gateway_id": "g2s", "probabilities": [
            {"path_id": "f_g2s_a", "value": "0.5"}, {"path_id": "f_g2s_b", "value": "0.5"}]},
        # A 2-way gateway that is NOT a case-route split (no _a/_b convention).
        {"gateway_id": "other", "probabilities": [
            {"path_id": "Flow_1", "value": "0.5"}, {"path_id": "Flow_2", "value": "0.5"}]},
    ]}
    p = tmp_path / "p.json"
    p.write_text(json.dumps(params))
    assert _count_split_gateways(p) == 2


def test_load_prosimos_log_keeps_case_type(tmp_path):
    csv = tmp_path / "sim.csv"
    pd.DataFrame({
        "case_id": [1, 1], "activity": ["A", "B"],
        "enable_time": ["2025-01-01T10:00:00.000+00:00"] * 2,
        "start_time": ["2025-01-01T10:00:00.000+00:00", "2025-01-01T10:05:00.000+00:00"],
        "end_time": ["2025-01-01T10:05:00.000+00:00", "2025-01-01T10:10:00.000+00:00"],
        "resource": ["R1", "R1"], "case_type": ["red", "red"],
    }).to_csv(csv, index=False)
    df = _load_prosimos_log(csv)
    assert "case_type" in df.columns
    assert (df["case_type"] == "red").all()


def test_write_prefix_csv_carries_case_type(tmp_path):
    log = _mklog_typed([
        ("c1", "A", "2025-01-01 10:00", "2025-01-01 10:30", "R1", "red"),
        ("c1", "B", "2025-01-01 11:30", "2025-01-01 12:00", "R1", "red"),  # after cutoff
    ])
    out = tmp_path / "prefix.csv"
    _write_prefix_csv(log, pd.Timestamp("2025-01-01 11:00", tz="UTC"), out)
    prefix = pd.read_csv(out)
    # Only the pre-cutoff event is written, and it carries case_type un-renamed.
    assert list(prefix["CaseId"]) == ["c1"]
    assert "case_type" in prefix.columns
    assert prefix["case_type"].iloc[0] == "red"


def test_compute_utilization_rows_partial_busy():
    log = _mklog([
        # R1 busy 1h within a 4h window.
        ("c1", "A", "2025-01-01 10:00", "2025-01-01 11:00", "R1"),
        # R2 busy 2h (one event spanning the cutoff, clipped to window start).
        ("c2", "A", "2025-01-01 11:30", "2025-01-01 13:00", "R2"),
        # Outside the window, ignored.
        ("c3", "A", "2025-01-02 08:00", "2025-01-02 09:00", "R1"),
    ])
    params = {
        "resource_profiles": [
            {"name": "Team A", "resource_list": [{"id": "R1"}, {"id": "R2"}]},
            {"name": "Team B", "resource_list": [{"id": "R3"}]},
        ]
    }
    cut = pd.Timestamp("2025-01-01 10:00", tz="UTC")
    horizon_end = cut + pd.Timedelta(hours=4)
    rows = _compute_utilization_rows(
        sim_log=log, params=params,
        cutoff=cut, horizon_end=horizon_end, level=0, k_sim=1,
    )
    by_role = {r["role"]: r for r in rows}
    # 4h window = 14400s. R1+R2 busy = 3600s + 5400s = 9000s. 2 resources.
    assert by_role["Team A"]["utilization"] == pytest.approx(9000 / (2 * 14400))
    # Team B never used → 0.
    assert by_role["Team B"]["utilization"] == 0.0
