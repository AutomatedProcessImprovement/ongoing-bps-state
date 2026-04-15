# evaluation/clustering.py
"""Clustering model training, application, and evaluation for confidence estimation."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


# ────────────────────────────────────────────────────────────────────
# Utility helpers
# ────────────────────────────────────────────────────────────────────

def _mean_ci(vals: np.ndarray, conf: float = 0.95) -> Tuple[float | None, float | None]:
    """
    Return (mean, approximate 95% *predictive* half-width) for a 1D array.

    NOTE: this is a prediction interval for individual errors, not a
    confidence interval for the mean. We use:

        mean ± 1.96 * std

    which is the usual ~95% interval for a normal distribution.
    """
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return None, None

    mean = float(vals.mean())
    if vals.size == 1:
        return mean, 0.0

    std = float(vals.std(ddof=1))
    # 1.96 * std  -> predictive 95% half-width (no / sqrt(n))
    ci = 1.96 * std
    return mean, ci


def _to_jsonable(obj):
    """Convert numpy scalars/arrays to plain Python types so json.dump works."""
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


# ────────────────────────────────────────────────────────────────────
# Model training
# ────────────────────────────────────────────────────────────────────

def train_baseline_model(
    train_df: pd.DataFrame,
    target_col: str,
) -> Dict:
    """Baseline: single global mean+CI, plus some quantiles."""
    errs = train_df[target_col].to_numpy(dtype=float)
    errs = errs[~np.isnan(errs)]
    mean, ci = _mean_ci(errs)

    quantiles = {}
    for q in (0.05, 0.25, 0.5, 0.75, 0.95):
        quantiles[str(q)] = float(np.quantile(errs, q)) if errs.size > 0 else None

    return {
        "target_col": target_col,
        "mean": mean,
        "ci": ci,
        "quantiles": quantiles,
    }


def train_wip_deciles_model(
    train_df: pd.DataFrame,
    target_col: str,
    n_bins: int = 3,
) -> Dict:
    """
    Group by quantiles of WIP; compute mean+CI per bucket.

    n_bins controls how many WIP groups we create (should normally
    match the number of clusters used by K-means).
    """
    df = train_df[["wip", target_col]].dropna()
    if df.empty:
        return {
            "target_col": target_col,
            "bin_edges": [],
            "groups": {},
        }

    wip = df["wip"].to_numpy(dtype=float)
    errs = df[target_col].to_numpy(dtype=float)

    # quantile edges; handle potential duplicates (constant wip)
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(wip, qs)
    edges[0] = float(edges[0] - 1e-9)  # ensure left-open on the very left

    groups: Dict[int, Dict] = {}
    for i in range(n_bins):
        left, right = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (wip > left) & (wip <= right + 1e-9)
        else:
            mask = (wip > left) & (wip <= right)

        bin_errs = errs[mask]
        mean, ci = _mean_ci(bin_errs)
        groups[i] = {
            "mean": mean,
            "ci": ci,
            "count": int(bin_errs.size),
            "wip_range": [float(left), float(right)],
        }

    return {
        "target_col": target_col,
        "bin_edges": [float(x) for x in edges],
        "groups": groups,
    }


def train_random_groups_model(
    train_df: pd.DataFrame,
    target_col: str,
    n_clusters: int = 3,
) -> Dict:
    """
    Random control technique.

    Split the *training samples* into n_clusters groups of (roughly)
    equal size based only on their index, without using any features.
    For each group we compute mean+CI of the target errors.

    This lets us see how much we gain/lose from using smaller subsets
    compared to the global baseline, when the subsets are *not*
    informed by process state.
    """
    errs = train_df[target_col].to_numpy(dtype=float)
    mask = ~np.isnan(errs)
    vals = errs[mask]
    if vals.size == 0:
        return {
            "target_col": target_col,
            "n_clusters": n_clusters,
            "groups": {},
        }

    n_clusters = max(1, min(int(n_clusters), vals.size))

    groups: Dict[int, Dict] = {}
    idx = np.arange(vals.size)
    cluster_ids = idx % n_clusters  # first 33 → cluster 0, next 33 → 1, etc.

    for cid in range(n_clusters):
        cluster_errs = vals[cluster_ids == cid]
        mean, ci = _mean_ci(cluster_errs)
        groups[cid] = {
            "mean": mean,
            "ci": ci,
            "count": int(cluster_errs.size),
        }

    return {
        "target_col": target_col,
        "n_clusters": n_clusters,
        "groups": groups,
    }


def _train_kmeans_on_features(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    n_clusters: int,
    random_state: int = 0,
) -> Dict:
    """Train a KMeans model on the given feature columns and attach CIs per cluster."""
    available = [c for c in feature_cols if c in train_df.columns]
    if not available:
        return {
            "target_col": target_col,
            "feature_cols": [],
            "centers": [],
            "clusters": {},
        }

    X = train_df[available].to_numpy(dtype=float)
    y = train_df[target_col].to_numpy(dtype=float)

    mask = (~np.isnan(X).any(axis=1)) & (~np.isnan(y))
    X_valid = X[mask]
    y_valid = y[mask]

    if X_valid.shape[0] == 0:
        return {
            "target_col": target_col,
            "feature_cols": available,
            "centers": [],
            "clusters": {},
        }

    n_clusters = min(n_clusters, max(1, X_valid.shape[0]))
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init="auto",
    )
    labels = kmeans.fit_predict(X_valid)

    clusters: Dict[int, Dict] = {}
    for cid in range(n_clusters):
        errs = y_valid[labels == cid]
        mean, ci = _mean_ci(errs)
        clusters[cid] = {
            "mean": mean,
            "ci": ci,
            "count": int(errs.size),
        }

    return {
        "target_col": target_col,
        "feature_cols": available,
        "centers": kmeans.cluster_centers_.tolist(),
        "clusters": clusters,
    }


def train_clustering_models(
    train_df: pd.DataFrame,
    target_col: str,
    n_clusters_basic: int = 3,
    n_clusters_advanced: int = 3,
    n_clusters_state: int = 3,
) -> Dict[str, Dict]:
    """Train all six clustering strategies on the training samples.

    Returns a dict keyed by method name (``baseline``, ``random_groups``,
    ``wip_deciles``, ``kmeans_basic``, ``kmeans_advanced``, ``kmeans_state``),
    each mapping to a model dict containing group means, CIs, and any fitted
    scaler/centroids.
    """
    models: Dict[str, Dict] = {}

    # 1) Global baseline (no grouping)
    models["baseline"] = train_baseline_model(train_df, target_col)

    # 2) Random control technique (index-based equal-size groups)
    models["random_groups"] = train_random_groups_model(
        train_df, target_col, n_clusters=n_clusters_basic
    )

    # 3) WIP quantile groups – same number as basic K-means clusters
    models["wip_deciles"] = train_wip_deciles_model(
        train_df, target_col, n_bins=n_clusters_basic
    )

    # 4) K-means on simple features
    simple_features = ["wip", "arrival_rate_per_hour", "resource_availability"]
    models["kmeans_simple"] = _train_kmeans_on_features(
        train_df, simple_features, target_col, n_clusters_basic
    )

    # 5) K-means on advanced WIP features (simple + per-activity WIP)
    advanced_features = simple_features + [
        c
        for c in train_df.columns
        if c.startswith("wip_enabled_") or c.startswith("wip_ongoing_")
    ]
    models["kmeans_advanced_wip"] = _train_kmeans_on_features(
        train_df, advanced_features, target_col, n_clusters_advanced
    )

    # 6) K-means on activity state vector features
    state_features = [c for c in train_df.columns if c.startswith("statevec_")]
    models["kmeans_statevec"] = _train_kmeans_on_features(
        train_df, state_features, target_col, n_clusters_state
    )

    return models


# ────────────────────────────────────────────────────────────────────
# Model application
# ────────────────────────────────────────────────────────────────────

def _assign_wip_decile(
    wip_value: float,
    bin_edges: List[float],
) -> int | None:
    if not bin_edges:
        return None
    edges = np.asarray(bin_edges, dtype=float)
    idx = int(np.searchsorted(edges, wip_value, side="right") - 1)
    idx = max(0, min(idx, len(edges) - 2))
    return idx


def _assign_kmeans(
    row: pd.Series,
    model: Dict,
) -> Tuple[int | None, float | None, float | None, bool | None]:
    """
    Assign a single test sample to a KMeans cluster and check whether the
    error is inside the cluster CI.

    Returns (cluster_id, pred_mean, pred_ci, in_ci)
    """
    centers = np.asarray(model.get("centers", []), dtype=float)
    feature_cols = model.get("feature_cols", [])
    if centers.size == 0 or not feature_cols:
        return None, None, None, None

    x = row[feature_cols].to_numpy(dtype=float)
    if np.isnan(x).any():
        return None, None, None, None

    dists = np.linalg.norm(centers - x, axis=1)
    cid = int(np.argmin(dists))
    info = model["clusters"].get(cid)
    if not info:
        return cid, None, None, None
    mean = info["mean"]
    ci = info["ci"]
    err = row.get(model["target_col"])
    if err is None or (mean is None) or (ci is None):
        return cid, mean, ci, None
    in_ci = abs(float(err) - float(mean)) <= float(ci)
    return cid, mean, ci, bool(in_ci)


def apply_models_to_test(
    models: Dict[str, Dict],
    test_df: pd.DataFrame,
    target_col: str,
) -> pd.DataFrame:
    """
    For each sample, apply all clustering models and record:

    - predicted mean/CI from the assigned group;
    - whether the true error falls inside that CI.

    We reuse this both for TRAIN (in-sample) and TEST (out-of-sample).
    """
    records: List[Dict] = []

    baseline = models.get("baseline")
    wip_model = models.get("wip_deciles")
    km_simple = models.get("kmeans_simple")
    km_adv = models.get("kmeans_advanced_wip")
    km_state = models.get("kmeans_statevec")
    rand_model = models.get("random_groups")

    n_rand_clusters = int(rand_model.get("n_clusters", 0)) if rand_model else 0

    for idx, (_, row) in enumerate(test_df.iterrows()):
        rec: Dict[str, object] = {
            "cut_time": row.get("cut_time_iso"),
            "split": row.get("split"),
            "true_error": float(row[target_col]) if not pd.isna(row[target_col]) else None,
        }

        # Baseline (global)
        if baseline:
            m = baseline.get("mean")
            ci = baseline.get("ci")
            err = rec["true_error"]
            in_ci = (
                (err is not None and m is not None and ci is not None)
                and (abs(float(err) - float(m)) <= float(ci))
            )
            rec.update(
                baseline_mean=m,
                baseline_ci=ci,
                baseline_in_ci=bool(in_ci)
                if err is not None and m is not None and ci is not None
                else None,
            )

        # Random control technique – assign groups purely by index
        if rand_model and n_rand_clusters > 0:
            cid = idx % n_rand_clusters
            group = rand_model["groups"].get(cid, {})
            m = group.get("mean")
            ci = group.get("ci")
            err = rec["true_error"]
            in_ci = (
                (err is not None and m is not None and ci is not None)
                and (abs(float(err) - float(m)) <= float(ci))
            )
            rec.update(
                random_groups_id=cid,
                random_groups_mean=m,
                random_groups_ci=ci,
                random_groups_in_ci=bool(in_ci)
                if err is not None and m is not None and ci is not None
                else None,
            )

        # WIP quantile groups
        if wip_model and "wip" in row:
            wip_val = float(row["wip"]) if not pd.isna(row["wip"]) else None
            edges = wip_model.get("bin_edges", [])
            cid = _assign_wip_decile(wip_val, edges) if wip_val is not None else None
            if cid is not None:
                group = wip_model["groups"].get(cid, {})
                m = group.get("mean")
                ci = group.get("ci")
                err = rec["true_error"]
                in_ci = (
                    (err is not None and m is not None and ci is not None)
                    and (abs(float(err) - float(m)) <= float(ci))
                )
                rec.update(
                    wip_decile_id=cid,
                    wip_decile_mean=m,
                    wip_decile_ci=ci,
                    wip_decile_in_ci=bool(in_ci)
                    if err is not None and m is not None and ci is not None
                    else None,
                )
            else:
                rec.update(
                    wip_decile_id=None,
                    wip_decile_mean=None,
                    wip_decile_ci=None,
                    wip_decile_in_ci=None,
                )

        # K-means (simple features)
        if km_simple:
            cid, m, ci, in_ci = _assign_kmeans(row, km_simple)
            rec.update(
                kmeans_simple_id=cid,
                kmeans_simple_mean=m,
                kmeans_simple_ci=ci,
                kmeans_simple_in_ci=in_ci,
            )

        # K-means (advanced WIP)
        if km_adv:
            cid, m, ci, in_ci = _assign_kmeans(row, km_adv)
            rec.update(
                kmeans_advanced_wip_id=cid,
                kmeans_advanced_wip_mean=m,
                kmeans_advanced_wip_ci=ci,
                kmeans_advanced_wip_in_ci=in_ci,
            )

        # K-means (activity state vector)
        if km_state:
            cid, m, ci, in_ci = _assign_kmeans(row, km_state)
            rec.update(
                kmeans_statevec_id=cid,
                kmeans_statevec_mean=m,
                kmeans_statevec_ci=ci,
                kmeans_statevec_in_ci=in_ci,
            )

        records.append(rec)

    return pd.DataFrame(records)


# ────────────────────────────────────────────────────────────────────
# Evaluation summary
# ────────────────────────────────────────────────────────────────────

def summarise_evaluation(eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-method summary:

    method_key, method_label, n_points, coverage, avg_ci_width, avg_abs_error
    """
    methods = [
        ("baseline", "baseline_in_ci", "baseline_ci", "Global baseline"),
        ("wip_deciles", "wip_decile_in_ci", "wip_decile_ci", "WIP percentile groups"),
        ("kmeans_simple", "kmeans_simple_in_ci", "kmeans_simple_ci",
         "K-means (simple features)"),
        ("kmeans_advanced_wip", "kmeans_advanced_wip_in_ci", "kmeans_advanced_wip_ci",
         "K-means (advanced WIP)"),
        ("kmeans_statevec", "kmeans_statevec_in_ci", "kmeans_statevec_ci",
         "K-means (activity state vector)"),
        ("random_groups", "random_groups_in_ci", "random_groups_ci",
         "Random control (equal-size groups)"),
    ]

    rows: List[Dict[str, object]] = []

    for key, in_col, ci_col, label in methods:
        if in_col not in eval_df.columns or ci_col not in eval_df.columns:
            continue

        mask = eval_df[in_col].notna()
        if not mask.any():
            continue

        coverage = float(eval_df.loc[mask, in_col].mean())
        avg_ci = float(eval_df.loc[mask, ci_col].mean())
        avg_abs_err = float(eval_df.loc[mask, "true_error"].abs().mean())

        rows.append(
            {
                "method_key": key,
                "method_label": label,
                "n_points": int(mask.sum()),
                "coverage": coverage,
                "avg_ci_width": avg_ci,
                "avg_abs_error": avg_abs_err,
            }
        )

    return pd.DataFrame(rows)
