"""Recompute every grid cell from stored sim logs. Reads nothing from results.csv."""
import glob, json, os, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from evaluation.state_metrics.pipeline import (
    filter_to_cases_in_window, get_ongoing_case_ids, _baseline_bin_size,
    _true_case_starts, _relative_event_distance_from_starts,
    _cycle_time_distance_from_starts, _resource_to_profile)
from evaluation.rtd import rtd as rtd_distance
from state_metrics.api import compute_all_state_distances
from log_distance_measures.config import EventLogIDs
from log_distance_measures.n_gram_distribution import n_gram_distribution_distance

IDS = EventLogIDs("case_id","activity","start_time","end_time","resource")
REN = {"CaseId":"case_id","Activity":"activity","Resource":"resource",
       "StartTime":"start_time","EndTime":"end_time"}

def load(p):
    d = pd.read_csv(p).rename(columns=REN)
    for c in ("start_time","end_time"):
        d[c] = pd.to_datetime(d[c], utc=True, format="mixed")
    d["case_id"] = d["case_id"].astype(str)
    return d[["case_id","activity","resource","start_time","end_time"]]

def paired_cf(b, s):
    ba = {c: frozenset(g["activity"].astype(str)) for c,g in b.groupby("case_id")}
    sa = {c: frozenset(g["activity"].astype(str)) for c,g in s.groupby("case_id")}
    cases = set(ba) | set(sa)
    def jac(x,y):
        if not x and not y: return 0.0
        return 1.0 - len(x & y)/len(x | y)
    return (sum(jac(ba.get(c,frozenset()), sa.get(c,frozenset())) for c in cases)/len(cases)) if cases else None

def paired_ct(b, s):
    def dur(df):
        g = df.groupby("case_id")
        return ((g["end_time"].max()-g["start_time"].min()).dt.total_seconds()).to_dict()
    bd, sd = dur(b), dur(s)
    common = set(bd) & set(sd)
    if not common: return None
    mae = sum(abs(bd[c]-sd[c]) for c in common)/len(common)
    return mae / ((sum(bd[c] for c in common)/len(common)) or 1.0)

MAN = json.load(open(sys.argv[1]))
only = sys.argv[2] if len(sys.argv) > 2 else None
out = {}
for mdl, d in MAN.items():
    if only and mdl != only: continue
    R = d["run_dir"]; ds = d["dataset"]
    cut = json.load(open(f"{R}/cutoff.json")); c0 = pd.Timestamp(cut["cutoff"])
    hz = pd.Timedelta(seconds=cut["horizon_seconds"]); win = (c0, c0+hz)
    src = load(ROOT / "outputs" / "state_metrics" / ds / "prefix_source_log.csv")
    ong = get_ongoing_case_ids(src, c0)
    starts = _true_case_starts(src, ong)
    bin_td = _baseline_bin_size(win); bin_sec = bin_td.total_seconds()
    role_map = _resource_to_profile(json.load(open(d["params"])))
    ks = sorted(int(os.path.basename(os.path.dirname(p)).split("_")[1])
                for p in glob.glob(f"{R}/baseline/k_*/sim_log.csv"))
    levels = sorted(int(os.path.basename(p).split("_")[1])
                    for p in glob.glob(f"{R}/level_*") if os.path.isdir(p))
    cache = {}
    def W(p):
        if p not in cache: cache[p] = filter_to_cases_in_window(load(p), ong, c0, hz)
        return cache[p]
    vals, fam = {}, {}
    def put(m, l, v, f):
        vals.setdefault(m, {}).setdefault(l, []).append(v); fam[m] = f
    for l in levels:
        for k in ks:
            b = W(f"{R}/baseline/k_{k}/sim_log.csv")
            s = W(f"{R}/level_{l}/k_{k}/sim_log.csv")
            rm = role_map
            present = set(b["resource"].astype(str)) | set(s["resource"].astype(str))
            if any(r not in rm for r in present):
                rm = {**{r:"unknown" for r in present}, **rm}
            done_card = False
            for (view,dist), r in compute_all_state_distances(b, s, role_map=rm, window=win).items():
                if dist == "cardinality":
                    if done_card: continue
                    m = "cardinality"; done_card = True
                else:
                    m = f"{view}.{dist}"
                put(m, l, r.summary_distance, "state")
            put("paired_cf", l, paired_cf(b,s), "baseline")
            put("paired_ct", l, paired_ct(b,s), "baseline")
            put("ngd_n2", l, n_gram_distribution_distance(b, IDS, s, IDS, n=2), "baseline")
            put("red", l, _relative_event_distance_from_starts(b, s, starts, bin_sec), "baseline")
            put("cycle_time", l, _cycle_time_distance_from_starts(b, s, starts, bin_sec), "baseline")
            put("rtd", l, rtd_distance(b, IDS, s, IDS, reference_point=c0, bin_size=bin_td), "baseline")
    u = pd.read_csv(f"{R}/utilization.csv")
    out[mdl] = {"vals": vals, "fam": fam, "levels": levels, "ks": ks,
                "ongoing": len(ong), "bin_min": bin_sec/60,
                "horizon_min": hz.total_seconds()/60,
                "utilisation": float(u[u.level == 0].utilization.mean()),
                "util_by_level": u.groupby("level").utilization.mean().round(4).to_dict(),
                "util_by_level_role": u.groupby(["level","role"]).utilization.mean().round(4).to_dict(),
                **{k: d[k] for k in ("run_dir","dataset","extent","ladder","reported","perturbation")}}
    print(f"{mdl:14} done  metrics={len(vals)} reps={len(ks)} levels={levels}", flush=True)
tag = f"_{only}" if only else ""
pickle.dump(out, open(f"{os.path.dirname(sys.argv[1])}/recomputed{tag}.pkl","wb"))
print("RECOMPUTE DONE")
