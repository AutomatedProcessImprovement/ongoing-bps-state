"""Empirical counterpart to the analytic invariant check.

The analytic script reasons about complete cases run end-to-end under one
model. The evaluation does not see those: it sees the CONTINUATION of a case
that already executed part of its path under the ground truth, spliced onto a
suffix produced by the perturbed model. The invariants survive that splice only
over the population -- a P-case's suffix and a Q-case's suffix EXCHANGE, so
cancellation needs equal P and Q counts at each branch position -- which is a
finite-sample property, not an exact one.

This measures each claimed invariant directly on the stored logs and separates
two things that a raw distance confuses:

  * level 0 vs level 0   -- two runs of the SAME model: replication noise.
  * level 0 vs level 100 -- the top of the ladder: noise + whatever the error
                            actually changed.

An invariant "holds" when the second is within the spread of the first. The
residual reported is the excess of the second over the first, in units of the
first's standard deviation.

Usage:  venv/bin/python <this> <run_dir> [<run_dir> ...]
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def _read(p: Path) -> pd.DataFrame:
    d = pd.read_csv(p)
    for c in ("start_time", "end_time"):
        d[c] = pd.to_datetime(d[c], format="mixed", utc=True)
    return d


def _clip(d: pd.DataFrame, lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DataFrame:
    return d[(d["start_time"] < hi) & (d["end_time"] > lo)]


def tv(a: Counter, b: Counter) -> float:
    """Total-variation distance between two histograms, normalised to [0, 1]."""
    ta, tb = sum(a.values()), sum(b.values())
    if not ta or not tb:
        return float(ta != tb)
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a[k] / ta - b[k] / tb) for k in keys)


def activity_hist(d: pd.DataFrame) -> Counter:
    return Counter(d["activity"])


def bigram_hist(d: pd.DataFrame) -> Counter:
    c: Counter = Counter()
    for _, g in d.sort_values("start_time").groupby("case_id"):
        s = list(g["activity"])
        c.update(zip(s, s[1:]))
    return c


def label_sets(d: pd.DataFrame) -> dict[str, frozenset]:
    return {k: frozenset(g["activity"])
            for k, g in d.groupby("case_id")}


def case_spans(d: pd.DataFrame) -> dict[str, float]:
    g = d.groupby("case_id").agg(s=("start_time", "min"), e=("end_time", "max"))
    return ((g["e"] - g["s"]).dt.total_seconds()).to_dict()


def event_offsets(d: pd.DataFrame) -> list[float]:
    """Each event's start, as a fraction of its case's continuation span."""
    out: list[float] = []
    for _, g in d.groupby("case_id"):
        t0 = g["start_time"].min()
        span = (g["end_time"].max() - t0).total_seconds()
        if span <= 0:
            continue
        out += [(t - t0).total_seconds() / span for t in g["start_time"]]
    return out


MEASURES = {
    "population activity multiset": lambda a, b: tv(activity_hist(a), activity_hist(b)),
    "bigram histogram":             lambda a, b: tv(bigram_hist(a), bigram_hist(b)),
    "per-case label set":           lambda a, b: _labelset_disagreement(a, b),
    "case duration (mean, rel.)":   lambda a, b: _mean_rel(case_spans(a), case_spans(b)),
    "within-case event position":   lambda a, b: _offset_shift(a, b),
}


def _labelset_disagreement(a: pd.DataFrame, b: pd.DataFrame) -> float:
    la, lb = label_sets(a), label_sets(b)
    both = set(la) & set(lb)
    if not both:
        return float("nan")
    return sum(la[k] != lb[k] for k in both) / len(both)


def _mean_rel(a: dict, b: dict) -> float:
    both = set(a) & set(b)
    ma = np.mean([a[k] for k in both]) or 1.0
    return abs(np.mean([b[k] for k in both]) - ma) / ma


def _offset_shift(a: pd.DataFrame, b: pd.DataFrame) -> float:
    oa, ob = np.array(event_offsets(a)), np.array(event_offsets(b))
    qs = np.linspace(0.05, 0.95, 19)
    return float(np.mean(np.abs(np.quantile(ob, qs) - np.quantile(oa, qs))))


def position_balance(run: Path) -> tuple[int, int, float]:
    """(ongoing cases, summed |P-Q| across positions, imbalance as a fraction)."""
    cut = pd.Timestamp(json.loads((run / "cutoff.json").read_text())["cutoff"])
    pre = pd.read_csv(run / "prefix.csv")
    gt = _read(run / "baseline" / "k_1" / "sim_log.csv")
    ong = set(gt["case_id"].unique())
    sub = pre[pre["CaseId"].isin(ong)]
    ct = sub.groupby("CaseId")["case_type"].first().astype(float)
    done = (sub[sub["Activity"].str.startswith("R_")]
            .groupby("CaseId").size().reindex(ct.index).fillna(0).astype(int))
    df = pd.DataFrame({"side": np.where(ct < 50, "P", "Q"), "done": done})
    tab = df.groupby(["done", "side"]).size().unstack(fill_value=0)
    for s in ("P", "Q"):
        if s not in tab:
            tab[s] = 0
    imb = int((tab["P"] - tab["Q"]).abs().sum())
    return len(df), imb, imb / max(len(df), 1)


def main() -> None:
    for arg in sys.argv[1:]:
        run = Path(arg)
        cut = pd.Timestamp(json.loads((run / "cutoff.json").read_text())["cutoff"])
        hz = pd.Timestamp(json.loads((run / "cutoff.json").read_text())["horizon_end"])
        ks = sorted(int(p.name.split("_")[1])
                    for p in (run / "baseline").glob("k_*"))
        top = max(int(p.name.split("_")[1])
                  for p in run.glob("level_*") if p.is_dir())
        base = {k: _clip(_read(run / "baseline" / f"k_{k}" / "sim_log.csv"), cut, hz)
                for k in ks}
        lvl0 = {k: _clip(_read(run / "level_0" / f"k_{k}" / "sim_log.csv"), cut, hz)
                for k in ks}
        lvlT = {k: _clip(_read(run / f"level_{top}" / f"k_{k}" / "sim_log.csv"), cut, hz)
                for k in ks}

        n, imb, frac = position_balance(run)
        print(f"\n=== {run.name}  ({n} ongoing cases, top level {top}) ===")
        print(f"per-position |P-Q| imbalance: {imb}  ({100 * frac:.1f}% of the "
              f"ongoing set) -- this is what bounds every cancellation below\n")
        print(f"{'invariant':30s} {'noise (L0)':>22s} {'top level':>12s} "
              f"{'excess':>9s}  verdict")
        for name, fn in MEASURES.items():
            noise = np.array([fn(base[k], lvl0[k]) for k in ks], float)
            sig = np.array([fn(base[k], lvlT[k]) for k in ks], float)
            mu, sd = float(np.nanmean(noise)), float(np.nanstd(noise, ddof=1))
            sm = float(np.nanmean(sig))
            z = (sm - mu) / sd if sd > 0 else float("inf") if sm > mu else 0.0
            verdict = ("exact" if sm == 0 and mu == 0 else
                       "preserved" if z < 2 else
                       f"residual")
            print(f"{name:30s} {mu:>12.5f} +-{sd:<8.5f} {sm:>12.5f} "
                  f"{z:>8.1f}s  {verdict}")


if __name__ == "__main__":
    main()
