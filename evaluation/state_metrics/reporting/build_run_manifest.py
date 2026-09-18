"""Rebuild run_manifest.json by IDENTIFYING each run, not by launch order.

Nine composition runs live in one output directory, so a run cannot be picked
by "the newest one under this dataset" or by the order the jobs were launched
-- an earlier version of this script did the latter and mis-assigned nine of
seventeen runs. Every run instead declares what it is in its own
level_<top>_manifest.json (extent, ladder), and this reads that declaration and
matches it to the job that asked for it. A job with no unique match is an error,
not a guess.
"""
import json, subprocess, sys
from pathlib import Path

SP = Path(sys.argv[1])
SINCE = float(sys.argv[2])
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs" / "state_metrics"
sys.path.insert(0, str(ROOT))
from evaluation.state_metrics.datasets import DATASETS as DS   # noqa: E402

FAM = {"C": "Composition", "T": "Timeline", "P": "Pairing", "X": "Control"}
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True, cwd=ROOT).stdout.strip()


def describe(run: Path) -> dict:
    """(extent, ladder) as the run itself recorded them."""
    tops = sorted(int(p.name.split("_")[1]) for p in run.glob("level_*")
                  if p.is_dir())
    m = json.loads((run / f"level_{tops[-1]}_manifest.json").read_text())
    ext = m.get("extent")
    if ext is None and "automate_task_ids" in m:
        ext = len(m["automate_task_ids"])
    return {"extent": ext or 0, "ladder": m.get("ladder")}


jobs = [l.strip().split("|", 1) for l in open(SP / "jobs.txt") if l.strip()]
man, used, errs = {}, set(), []
for tag, argv in jobs:
    key = argv.split()[0]
    spec = DS.get(key)
    name = spec.name if spec else key
    pert = argv.split("--perturbation ")[1].split()[0]
    want_ext = spec.perturb_extent if spec else 0
    want_lad = ("stretch" if "--automation-ladder stretch" in argv else
                "shrink_bounded" if "shrink_bounded" in argv else
                "shrink" if "--automation-ladder shrink" in argv else None)
    hits = []
    for p in (OUT / name / pert).glob("run_*"):
        if not p.is_dir() or p.stat().st_mtime < SINCE or str(p) in used:
            continue
        if not (p / "results.csv").exists():
            continue
        got = describe(p)
        if got["extent"] == want_ext and got["ladder"] == want_lad:
            hits.append(p)
    if len(hits) != 1:
        errs.append(f"{tag}: {len(hits)} candidate run dirs (want extent="
                    f"{want_ext} ladder={want_lad} under {name}/{pert})")
        continue
    d = hits[0]
    used.add(str(d))
    man[tag] = {
        "job": tag, "family": FAM[tag[0]] + ("-alt" if "_" in tag else ""),
        "dataset_key": key, "dataset": name, "run_dir": str(d),
        "params": str(spec.params) if spec else "",
        "bpmn": str(spec.bpmn) if spec else "",
        "extent": want_ext, "ladder": want_lad,
        "reported": "_" not in tag,
        "argv": argv, "perturbation": pert, "git_commit": commit,
    }
if errs:
    print("UNRESOLVED:", *errs, sep="\n  ", file=sys.stderr)
    sys.exit(1)
json.dump(man, open(SP / "run_manifest.json", "w"), indent=1)
print(f"pinned {len(man)} runs, {sum(1 for v in man.values() if v['reported'])} reported")
for k, v in man.items():
    print(f"  {k:12s} {v['run_dir'].split('/')[-1]}  extent={v['extent']} "
          f"ladder={v['ladder']}")
