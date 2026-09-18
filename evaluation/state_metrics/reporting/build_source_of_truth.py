"""Build results_source_of_truth.csv + the scratch report from recomputed cells."""
import csv, glob, json, os, pickle, re, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).parent))
import runs as RUNS
from evaluation.state_metrics.ranking import (
    concordance_index_on_means, bootstrap_ci, _stable_seed)

ARGV = sys.argv[1:]
DATA = RUNS.data_root(ARGV)
SP = Path(ARGV[0])
MAN = json.load(open(SP / "run_manifest.json"))
R = pickle.load(open(SP / "recomputed.pkl", "rb")) if (SP / "recomputed.pkl").exists() else {}
ONLY = ARGV[1] if len(ARGV) > 1 else None

# Canonical order for the merged table: family first, then extent, then the
# measure order the paper's table uses. Alphabetical would interleave the
# scratch ladder variants with the reported models.
MODEL_ORDER = ["C1", "C2", "C3", "T1", "T2", "T3", "P1", "P2", "P3", "X1", "X2"]
MEASURE_ORDER = ["(a,c)", "act", "(a,r)", "case", "card", "cf", "ct",
                 "ngd", "red", "cyc", "rtd"]

if ONLY == "merge":
    # Merge the per-model parts written by earlier `... <work-dir> <model>`
    # invocations. Split out so the slow part (100 bootstrap seeds x 1000
    # iterations per cell) can run one process per model.
    parts = sorted(glob.glob(str(SP / "part_*.csv")))
    if not parts:
        sys.exit(f"no part_*.csv under {SP}; run this per model first")
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    df["__m"] = df.model.map(lambda m: order.get(m, len(order) + 1))
    df["__o"] = df.measure.map({m: i for i, m in enumerate(MEASURE_ORDER)})
    df = df.sort_values(["reported", "__m", "model", "__o"],
                        ascending=[False, True, True, True]).drop(
                            columns=["__m", "__o"])
    df.to_csv(SP / "results_source_of_truth_FULL.csv", index=False)
    rep = df[df.reported == 1].drop(columns=["reported"])
    rep.to_csv(SP / "results_source_of_truth.csv", index=False)
    lm = {}
    for p in sorted(glob.glob(str(SP / "part_*.json"))):
        lm.update(json.load(open(p)))
    json.dump(lm, open(SP / "level_means.json", "w"), indent=1)
    print(f"merged {len(parts)} parts -> {SP / 'results_source_of_truth.csv'}: "
          f"{len(rep)} reported rows, {len(df)} total")
    sys.exit(0)

if ONLY:
    R = {ONLY: R[ONLY]}

SHORT = {"activity_case.jaccard":"(a,c)","activity.jaccard":"act",
         "activity_role.jaccard":"(a,r)","case.jaccard":"case","cardinality":"card",
         "paired_cf":"cf","paired_ct":"ct","ngd_n2":"ngd","cycle_time":"cyc",
         "red":"red","rtd":"rtd"}
ORDER = MEASURE_ORDER

def model_geometry(bpmn_path, params_path):
    x = open(RUNS.in_repo(bpmn_path)).read()
    # Loan-stable's BPMN has no namespace prefix on its elements; the generated
    # synthetics use bpmn:. Match either, or the controls silently read 0.
    tasks = re.findall(r'<(?:\w+:)?task\s[^>]*id="([^"]+)"[^>]*?(?:name="([^"]*)")?', x)
    tasks = [(t, n or "") for t, n in tasks]
    splits = len(re.findall(
        r'<(?:\w+:)?exclusiveGateway\b(?:(?!</(?:\w+:)?exclusiveGateway>).)*?'
        r'<(?:\w+:)?outgoing>.*?</(?:\w+:)?outgoing>\s*<(?:\w+:)?outgoing>', x, re.S))
    p = json.load(open(RUNS.in_repo(params_path)))
    workers = sum(len(pr.get("resource_list", [])) for pr in p.get("resource_profiles", []))
    return len(tasks), workers, splits, [t for t, _ in tasks], p

def task_mean(p, tid):
    for t in p["task_resource_distribution"]:
        if t["task_id"] == tid:
            r = t["resources"][0]
            dp = [v["value"] for v in r["distribution_params"]]
            return dp[0], (dp[1] / dp[0] if len(dp) > 1 and r["distribution_name"] == "norm" and dp[0] else 0.0)
    return None, None

rows, report = [], {}
ALL_LEVELS = sorted({l for e in R.values() for l in e["levels"]})
for mdl, e in R.items():
    m = MAN[mdl]
    run = RUNS.run_dir(m, DATA)
    n_tasks, n_workers, n_splits, tids, params = model_geometry(m["bpmn"], m["params"])
    # timeline descriptors
    gate_mean = gate_cv = pre_pct = None; design = None; bla = blb = None
    pmean = {t: task_mean(params, t)[0] for t in tids}
    if any(t.startswith("t_p") for t in tids):
        bla = sum(1 for t in tids if re.fullmatch(r"t_p\d+", t))
        blb = sum(1 for t in tids if re.fullmatch(r"t_q\d+", t))
        gm, gcv = task_mean(params, "t_gate")
        gate_mean, gate_cv = gm, (gcv * 100 if gcv is not None else None)
        pre = (pmean.get("t_reg") or 0) + (gm or 0)
        branch = sum(v for t, v in pmean.items() if re.fullmatch(r"t_p\d+", t))
        total = pre + branch + (pmean.get("t_dec") or 0)
        pre_pct = 100 * pre / total if total else None
        design = f"gated CV {gate_cv:.1f}%" if gate_cv else "ungated"
    elif any(t.startswith("t_a1_") for t in tids):
        bla = sum(1 for t in tids if re.fullmatch(r"t_a1_\d+", t))
        blb = sum(1 for t in tids if re.fullmatch(r"t_b1_\d+", t))
    # the run's own rankings.csv = what the pipeline reported at run time
    own = {}
    rk = run / "rankings.csv"
    if rk.exists() or rk.with_suffix(".csv.gz").exists():
        rd = pd.read_csv(RUNS.resolve(rk))
        rd = rd[rd.scope == "A_ongoing"]
        own = {r.metric: r.c_index_means for r in rd.itertuples()}
    # The run records when it started; an archived copy has fresh mtimes, so
    # falling back to those would date the archive, not the experiment.
    started = m.get("run_started_utc") or pd.Timestamp(
        min(p.stat().st_mtime for p in run.iterdir()), unit="s",
        tz="UTC").isoformat(timespec="seconds")
    for full, short in SHORT.items():
        vbl = {l: [v for v in e["vals"][full][l]] for l in e["levels"]}
        fam = e["fam"][full]
        rng = np.random.default_rng(_stable_seed(0, "A_ongoing", fam, full))
        c = concordance_index_on_means(vbl)
        lo, hi = bootstrap_ci(concordance_index_on_means, vbl, n_iter=1000, alpha=0.05, rng=rng)
        star = not (lo <= 0.5 <= hi)
        nstar = 0
        for s in range(100):
            r2 = np.random.default_rng(_stable_seed(s, "A_ongoing", fam, full))
            l2, h2 = bootstrap_ci(concordance_index_on_means, vbl, n_iter=1000, alpha=0.05, rng=r2)
            nstar += int(not (l2 <= 0.5 <= h2))
        means = {l: float(np.mean(v)) for l, v in vbl.items()}
        floor, top = means[min(means)], means[max(means)]
        rows.append({
            "family": m["family"], "model": mdl, "measure": short, "measure_full": full,
            "c_index_means": round(c, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4),
            "star_ci_excludes_half": int(star),
            "frac_of_100_seeds_that_star": round(nstar / 100, 2),
            "star_is_seed_dependent": int(0 < nstar < 100),
            "level_0_mean_noise_floor": f"{floor:.10g}", "top_level_mean": f"{top:.10g}",
            "rise_over_floor": (f"{top/floor:.4f}" if floor > 0 else ""),
            "reported_point_before_audit": own.get(full, ""),
            "point_matches_report": int(abs(own.get(full, c) - c) < 5e-4) if full in own else "",
            "run_id": os.path.basename(m["run_dir"]), "dataset": m["dataset"],
            "run_started_utc": started,
            "git_commit": m["git_commit"], "n_levels": len(e["levels"]),
            "levels": ",".join(str(l) for l in e["levels"]),
            "n_replications": len(e["ks"]), "ongoing_cases": e["ongoing"],
            "resource_utilisation": round(e["utilisation"], 3),
            "horizon_min": round(e["horizon_min"], 2), "bin_min": round(e["bin_min"], 4),
            "baseline_bins": 50, "scope": "A_ongoing",
            "n_tasks": n_tasks, "n_workers": n_workers, "n_split_gateways": n_splits,
            "branch_len_a": bla, "branch_len_b": blb, "timeline_design": design,
            "gate_mean_s": gate_mean, "gate_cv_pct": (round(gate_cv, 2) if gate_cv else None),
            "pre_branch_pct": (round(pre_pct, 1) if pre_pct else None),
            "perturb_extent": m["extent"] or "", "ladder": m["ladder"] or "",
            "reported": int(m["reported"]),
            **{f"mean_L{l}": (f"{means[l]:.10g}" if l in means else "") for l in ALL_LEVELS},
        })
    report[mdl] = {"means": {SHORT[f]: {l: float(np.mean(v)) for l, v in e["vals"][f].items()}
                            for f in SHORT}, "levels": e["levels"]}

df = pd.DataFrame(rows)
if ONLY:
    df.to_csv(SP / f"part_{ONLY}.csv", index=False)
    json.dump(report, open(SP / f"part_{ONLY}.json", "w"), indent=1)
    print("wrote part", ONLY, len(df), "rows")
else:
    df["__o"] = df.measure.map({m: i for i, m in enumerate(ORDER)})
    df = df.sort_values(["reported", "family", "model", "__o"],
                        ascending=[False, True, True, True]).drop(columns="__o")
    df.to_csv(SP / "results_source_of_truth_FULL.csv", index=False)
    df[df.reported == 1].drop(columns=["reported"]).to_csv(
        SP / "results_source_of_truth.csv", index=False)
    json.dump(report, open(SP / "level_means.json", "w"), indent=1)
    print("wrote", SP / "results_source_of_truth.csv",
          len(df[df.reported == 1]), "reported rows;", len(df), "rows total")
