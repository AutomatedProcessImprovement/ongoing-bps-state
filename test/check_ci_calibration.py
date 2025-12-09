# test/check_ci_calibration.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


METHOD_LABELS = {
    "baseline": "Global baseline",
    "wip_deciles": "WIP decile groups",
    "kmeans_simple": "K-means (simple features)",
    "kmeans_advanced_wip": "K-means (advanced WIP)",
    "kmeans_statevec": "K-means (activity state vector)",
}

# Order in which rows will be shown
METHOD_ORDER = [
    "kmeans_advanced_wip",
    "baseline",
    "kmeans_statevec",
    "wip_deciles",
    "kmeans_simple",
]


# ---------- helpers reused from the main pipeline ------------------


def _assign_wip_decile(wip_value: float, bin_edges: List[float]) -> int | None:
    """Assign a WIP value to a decile/bin index based on bin_edges."""
    if not bin_edges or wip_value is None or np.isnan(wip_value):
        return None
    edges = np.asarray(bin_edges, dtype=float)
    idx = int(np.searchsorted(edges, wip_value, side="right") - 1)
    idx = max(0, min(idx, len(edges) - 2))
    return idx


def _get_from_dict(d: Dict, key: int) -> Dict | None:
    """Helper: JSON will turn int keys into strings, so try both."""
    if key in d:
        return d[key]
    s = str(key)
    return d.get(s)


def _assign_kmeans_row(row: pd.Series, model: Dict) -> Tuple[int | None, float | None, float | None, bool | None]:
    """
    Assign a single *row* to the nearest K-means center and check whether
    the true error is inside the cluster CI.

    Returns (cluster_id, pred_mean, pred_ci, in_ci)
    """
    centers = np.asarray(model.get("centers", []), dtype=float)
    feature_cols = model.get("feature_cols", [])
    target_col = model.get("target_col")

    if centers.size == 0 or not feature_cols or not target_col:
        return None, None, None, None

    # extract feature vector
    x = row[feature_cols].to_numpy(dtype=float)
    if np.isnan(x).any():
        return None, None, None, None

    # nearest center
    dists = np.linalg.norm(centers - x, axis=1)
    cid = int(np.argmin(dists))

    clusters = model.get("clusters", {})
    info = _get_from_dict(clusters, cid)
    if not info:
        return cid, None, None, None

    m = info.get("mean")
    ci = info.get("ci")

    err_val = row.get(target_col)
    if pd.isna(err_val) or m is None or ci is None:
        return cid, m, ci, None

    in_ci = abs(float(err_val) - float(m)) <= float(ci)
    return cid, float(m), float(ci), bool(in_ci)


# ---------- TRAIN coverage, using train_samples + cluster_models ---


def compute_train_coverage(exp_root: Path) -> pd.DataFrame:
    """
    Compute coverage on the *training* samples, using the cluster_models.json
    learned on train. This checks whether our nominal ~95% predictive
    intervals are actually around 0.95 on the data they were fitted to.
    """
    train_path = exp_root / "train_samples.csv"
    models_path = exp_root / "cluster_models.json"

    if not train_path.is_file():
        raise FileNotFoundError(f"Missing {train_path}")
    if not models_path.is_file():
        raise FileNotFoundError(f"Missing {models_path}")

    train_df = pd.read_csv(train_path)
    with models_path.open(encoding="utf-8") as fh:
        models = json.load(fh)

    # Use the target_col stored in the baseline model (same for all models)
    baseline_model = models.get("baseline", {})
    target_col = baseline_model.get("target_col")
    if not target_col:
        # fallback: guess from available columns
        if "err_RTD_mean" in train_df.columns:
            target_col = "err_RTD_mean"
        elif "err_cycle_mean" in train_df.columns:
            target_col = "err_cycle_mean"
        else:
            raise RuntimeError("Cannot infer target_col (err_RTD_mean / err_cycle_mean).")

    # For scale reference: absolute error from zero (same for all methods)
    errs_all = train_df[target_col].to_numpy(dtype=float)
    errs_all = errs_all[~np.isnan(errs_all)]
    avg_abs_error = float(np.mean(np.abs(errs_all))) if errs_all.size > 0 else np.nan

    rows = []

    for key in METHOD_ORDER:
        model = models.get(key)
        if model is None:
            continue

        label = METHOD_LABELS.get(key, key)
        in_ci_flags: list[bool] = []
        ci_widths: list[float] = []

        if key == "baseline":
            m = model.get("mean")
            ci = model.get("ci")
            if m is None or ci is None:
                continue
            for _, r in train_df.iterrows():
                err_val = r.get(target_col)
                if pd.isna(err_val):
                    continue
                in_ci_flags.append(abs(float(err_val) - float(m)) <= float(ci))
                ci_widths.append(float(ci))

        elif key == "wip_deciles":
            edges = model.get("bin_edges", [])
            groups = model.get("groups", {})
            for _, r in train_df.iterrows():
                wip_val = r.get("wip")
                err_val = r.get(target_col)
                if pd.isna(wip_val) or pd.isna(err_val):
                    continue
                cid = _assign_wip_decile(float(wip_val), edges)
                if cid is None:
                    continue
                group = _get_from_dict(groups, cid)
                if not group:
                    continue
                m = group.get("mean")
                ci = group.get("ci")
                if m is None or ci is None:
                    continue
                in_ci_flags.append(abs(float(err_val) - float(m)) <= float(ci))
                ci_widths.append(float(ci))

        else:
            # Any of the K-means models
            for _, r in train_df.iterrows():
                cid, m, ci, in_ci = _assign_kmeans_row(r, model)
                if m is None or ci is None or in_ci is None:
                    continue
                in_ci_flags.append(bool(in_ci))
                ci_widths.append(float(ci))

        if not in_ci_flags:
            continue

        coverage = float(np.mean(in_ci_flags))
        avg_ci_width = float(np.mean(ci_widths))

        rows.append(
            {
                "method_key": key,
                "method_label": label,
                "n_points": len(in_ci_flags),
                "coverage": coverage,
                "avg_ci_width": avg_ci_width,
                "avg_abs_error": avg_abs_error,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["method_key", "method_label", "n_points", "coverage", "avg_ci_width", "avg_abs_error"])

    df = pd.DataFrame(rows)
    # keep desired method order
    df["order"] = df["method_key"].apply(lambda k: METHOD_ORDER.index(k) if k in METHOD_ORDER else 999)
    df = df.sort_values("order").drop(columns=["order"])
    return df


# ---------- TEST coverage (sanity check, recomputed) ---------------


def compute_test_coverage(exp_root: Path) -> pd.DataFrame:
    """
    Recompute coverage on the *test* set from test_evaluation.csv,
    just to double-check that the numbers you see there are consistent.
    """
    eval_path = exp_root / "test_evaluation.csv"
    if not eval_path.is_file():
        raise FileNotFoundError(f"Missing {eval_path}")

    df = pd.read_csv(eval_path)

    methods = [
        ("baseline", "baseline_in_ci", "baseline_ci"),
        ("wip_deciles", "wip_decile_in_ci", "wip_decile_ci"),
        ("kmeans_simple", "kmeans_simple_in_ci", "kmeans_simple_ci"),
        ("kmeans_advanced_wip", "kmeans_advanced_wip_in_ci", "kmeans_advanced_wip_ci"),
        ("kmeans_statevec", "kmeans_statevec_in_ci", "kmeans_statevec_ci"),
    ]

    # scale of errors on test (same for all methods)
    if "true_error" in df.columns:
        errs_all = df["true_error"].to_numpy(dtype=float)
        errs_all = errs_all[~np.isnan(errs_all)]
        avg_abs_error = float(np.mean(np.abs(errs_all))) if errs_all.size > 0 else np.nan
    else:
        avg_abs_error = np.nan

    rows = []
    for key, in_col, ci_col in methods:
        if in_col not in df.columns or ci_col not in df.columns:
            continue

        mask = df[in_col].notna() & df[ci_col].notna()
        if not mask.any():
            continue

        coverage = float(df.loc[mask, in_col].mean())
        avg_ci = float(df.loc[mask, ci_col].mean())

        rows.append(
            {
                "method_key": key,
                "method_label": METHOD_LABELS.get(key, key),
                "n_points": int(mask.sum()),
                "coverage": coverage,
                "avg_ci_width": avg_ci,
                "avg_abs_error": avg_abs_error,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["method_key", "method_label", "n_points", "coverage", "avg_ci_width", "avg_abs_error"])

    out = pd.DataFrame(rows)
    out["order"] = out["method_key"].apply(lambda k: METHOD_ORDER.index(k) if k in METHOD_ORDER else 999)
    out = out.sort_values("order").drop(columns=["order"])
    return out


# ---------- main ---------------------------------------------------


def main() -> None:
    # CHANGE THIS to point to an existing experiment folder
    # e.g. Path("outputs_confidence/LOAN_STABLE/init")
    EXP_ROOT = Path("outputs_confidence/LOAN_STABLE/init")

    if not EXP_ROOT.is_dir():
        raise FileNotFoundError(f"Experiment folder not found: {EXP_ROOT}")

    print(f"Using experiment root: {EXP_ROOT}\n")

    # TRAIN
    train_summary = compute_train_coverage(EXP_ROOT)
    print("=== TRAIN coverage (in-sample, using train_samples + cluster_models) ===")
    if train_summary.empty:
        print("No data / could not compute train coverage.")
    else:
        print(train_summary.to_string(index=False))
        train_summary.to_csv(EXP_ROOT / "train_coverage_summary.csv", index=False)

    print("\n")

    # TEST (sanity check)
    test_summary = compute_test_coverage(EXP_ROOT)
    print("=== TEST coverage (recomputed from test_evaluation.csv) ===")
    if test_summary.empty:
        print("No data / could not compute test coverage.")
    else:
        print(test_summary.to_string(index=False))
        test_summary.to_csv(EXP_ROOT / "test_coverage_summary_recomputed.csv", index=False)


if __name__ == "__main__":
    main()
