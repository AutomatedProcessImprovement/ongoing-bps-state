"""Copy the runs behind the reported grid into the repository, gzipped.

A run directory as Prosimos leaves it is mostly noise: on the pairing family,
5.2 GB of a 5.3 GB run is `simulation_warnings.txt`. What the scoring pass
actually reads is the simulation logs and a handful of small sidecars -- 909 MB
across the eleven reported runs, 158 MB gzipped, which is small enough to ship.

Shipping it is what makes the reported table checkable without a simulator.
Prosimos is not reproducible from a seed alone (see the README), so a fresh run
gives statistically equivalent numbers, not the reported ones; only the stored
logs can reproduce the published cells exactly.

The archive mirrors `outputs/state_metrics/` one for one -- same dataset and
run-id directories, same file names with `.gz` appended -- so every reporting
script reads it by pointing `--data-root` at it instead.

Usage:  venv/bin/python <this> <work-dir-holding-run_manifest.json>
"""
from __future__ import annotations

import gzip
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import runs as RUNS

# Per run. Anything not named here is either regenerable from these or noise.
PER_RUN = ["cutoff.json", "utilization.csv", "results.csv", "rankings.csv",
           "prefix.csv"]
# Per dataset: the ground-truth log the ongoing-case set and the true case
# starts come from. Shared by every run of that dataset.
PER_DATASET = ["prefix_source_log.csv", "prefix_source_stats.csv"]
# Never worth keeping: Prosimos writes one warning line per simulated event.
GZIP_EXEMPT = {".json"}


def _copy(src: Path, dst: Path) -> int:
    """Copy `src` to `dst`, gzipping unless it is already small and structured."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix in GZIP_EXEMPT:
        shutil.copy2(src, dst)
        return dst.stat().st_size
    with open(src, "rb") as i, gzip.open(str(dst) + ".gz", "wb", 6) as o:
        shutil.copyfileobj(i, o)
    return Path(str(dst) + ".gz").stat().st_size


def main() -> None:
    work = Path(sys.argv[1])
    src_root = RUNS.data_root(sys.argv[1:])
    man = json.load(open(work / "run_manifest.json"))
    out = RUNS.ARCHIVE
    if out.exists():
        shutil.rmtree(out)

    total, n = 0, 0
    datasets = set()
    for tag, e in sorted(man.items()):
        if not e["reported"]:
            continue
        run = RUNS.run_dir(e, src_root)
        dst = out / e["run_dir"]
        datasets.add(e["dataset"])
        for name in PER_RUN:
            p = run / name
            if p.exists():
                total += _copy(p, dst / name); n += 1
        for lvl in run.glob("level_*"):
            if lvl.is_dir():
                for k in RUNS.replications(run):
                    total += _copy(lvl / f"k_{k}" / "sim_log.csv",
                                   dst / lvl.name / f"k_{k}" / "sim_log.csv")
                    n += 1
            elif lvl.suffix == ".json":      # level_<top>_manifest.json
                total += _copy(lvl, dst / lvl.name); n += 1
        for k in RUNS.replications(run):
            total += _copy(run / "baseline" / f"k_{k}" / "sim_log.csv",
                           dst / "baseline" / f"k_{k}" / "sim_log.csv")
            n += 1
        print(f"  {tag:3s} {e['run_dir']}")

    for ds in sorted(datasets):
        for name in PER_DATASET:
            p = src_root / ds / name
            if p.exists():
                total += _copy(p, out / ds / name); n += 1

    json.dump({k: v for k, v in man.items() if v["reported"]},
              open(out / "run_manifest.json", "w"), indent=1)
    print(f"\n{n} files, {total / 1e6:.0f} MB -> {out.relative_to(RUNS.ROOT)}")


if __name__ == "__main__":
    main()
