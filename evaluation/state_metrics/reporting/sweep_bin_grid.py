"""Four-bin baseline sweep over the reported grid, recomputed from stored logs."""
import glob, json, os, pickle, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from evaluation.state_metrics.pipeline import (
    filter_to_cases_in_window, get_ongoing_case_ids, _baseline_bin_size,
    _true_case_starts, _relative_event_distance_from_starts,
    _cycle_time_distance_from_starts)
from evaluation.rtd import rtd as rtd_distance
from log_distance_measures.config import EventLogIDs

IDS = EventLogIDs("case_id","activity","start_time","end_time","resource")
REN = {"CaseId":"case_id","Activity":"activity","Resource":"resource",
       "StartTime":"start_time","EndTime":"end_time"}
def load(p):
    d = pd.read_csv(p).rename(columns=REN)
    for c in ("start_time","end_time"):
        d[c] = pd.to_datetime(d[c], utc=True, format="mixed")
    d["case_id"] = d["case_id"].astype(str)
    return d[["case_id","activity","resource","start_time","end_time"]]

SP = os.path.dirname(os.path.abspath(sys.argv[1]))
MAN = json.load(open(sys.argv[1])); mdl = sys.argv[2]
BINS = {"fixed 1h": None, "win/20": 20, "win/50": 50, "win/100": 100}
m = MAN[mdl]; R = m["run_dir"]
cut = json.load(open(f"{R}/cutoff.json")); c0 = pd.Timestamp(cut["cutoff"])
hz = pd.Timedelta(seconds=cut["horizon_seconds"]); win = (c0, c0 + hz)
src = load(ROOT / "outputs" / "state_metrics" / m["dataset"]
          / "prefix_source_log.csv")
ong = get_ongoing_case_ids(src, c0); starts = _true_case_starts(src, ong)
levels = sorted(int(os.path.basename(p).split("_")[1])
                for p in glob.glob(f"{R}/level_*") if os.path.isdir(p))
ks = sorted(int(os.path.basename(os.path.dirname(p)).split("_")[1])
            for p in glob.glob(f"{R}/baseline/k_*/sim_log.csv"))
cache = {}
def W(p):
    if p not in cache: cache[p] = filter_to_cases_in_window(load(p), ong, c0, hz)
    return cache[p]
sweep = {t: {x: {l: [] for l in levels} for x in ("cycle_time","red","rtd")} for t in BINS}
for tag, n in BINS.items():
    bs = pd.Timedelta(hours=1) if n is None else (win[1] - win[0]) / n
    sec = bs.total_seconds()
    for l in levels:
        for k in ks:
            b = W(f"{R}/baseline/k_{k}/sim_log.csv"); s = W(f"{R}/level_{l}/k_{k}/sim_log.csv")
            sweep[tag]["red"][l].append(_relative_event_distance_from_starts(b, s, starts, sec))
            sweep[tag]["cycle_time"][l].append(_cycle_time_distance_from_starts(b, s, starts, sec))
            sweep[tag]["rtd"][l].append(rtd_distance(b, IDS, s, IDS, reference_point=c0, bin_size=bs))
pickle.dump({mdl: {"sweep": sweep, "levels": levels, "ks": ks, "run": R,
                   "horizon_min": hz.total_seconds()/60,
                   "bin_min": _baseline_bin_size(win).total_seconds()/60,
                   "nreps": len(ks)}}, open(f"{SP}/sweep_{mdl}.pkl", "wb"))
print(f"{mdl} sweep done")
