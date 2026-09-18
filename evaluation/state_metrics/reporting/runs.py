"""Where a finished run's files live, and how to open them.

Two things vary between a run that was just simulated and one read back from
the archive this repository ships, and every reporting script has to cope with
both:

* **Where.** A fresh run lands under ``outputs/state_metrics/``; the archived
  grid lives under ``evaluation/state_metrics/reference_runs/``. The two have
  the same shape, so a manifest stores each run's path *relative* to a data
  root, and the caller picks the root with ``--data-root`` or the
  ``STATE_METRICS_DATA_ROOT`` environment variable.

* **How.** The archive is gzipped, because a simulation log is text and
  compresses about sixfold. ``pandas.read_csv`` reads either form from the
  filename alone, so the only question is which name is on disk.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = ROOT / "outputs" / "state_metrics"
ARCHIVE = ROOT / "evaluation" / "state_metrics" / "reference_runs"


def data_root(argv: list[str] | None = None) -> Path:
    """The directory run paths are relative to, and remove --data-root from argv.

    Precedence: the flag, then the environment, then a fresh run's outputs.
    """
    if argv and "--data-root" in argv:
        i = argv.index("--data-root")
        del argv[i]
        return Path(argv.pop(i)).resolve()
    env = os.environ.get("STATE_METRICS_DATA_ROOT")
    return Path(env).resolve() if env else DEFAULT_ROOT


def run_dir(entry: dict, root: Path) -> Path:
    """A manifest entry's run directory.

    Manifests written before the archive existed stored an absolute path; those
    are honoured as they are, so an old manifest still scores.
    """
    p = Path(entry["run_dir"])
    return p if p.is_absolute() else root / p


def in_repo(path) -> Path:
    """A manifest's BPMN / params path, which is stored relative to the repo."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def resolve(path) -> Path:
    """`path` if it exists, else `path.gz`. Raises naming both if neither does."""
    p = Path(path)
    if p.exists():
        return p
    gz = p.with_suffix(p.suffix + ".gz")
    if gz.exists():
        return gz
    raise FileNotFoundError(f"neither {p} nor {gz}")


def replications(run: Path) -> list[int]:
    """The k of every replication the run stored, in order."""
    return sorted(int(p.name.split("_")[1]) for p in (run / "baseline").glob("k_*")
                  if p.is_dir())


def levels(run: Path) -> list[int]:
    """The perturbation levels the run stored, in order."""
    return sorted(int(p.name.split("_")[1]) for p in run.glob("level_*")
                  if p.is_dir())


def sim_log(run: Path, level, k: int) -> Path:
    """One replication's simulation log. `level` is an int, or "baseline"."""
    sub = "baseline" if level == "baseline" else f"level_{level}"
    return resolve(run / sub / f"k_{k}" / "sim_log.csv")
