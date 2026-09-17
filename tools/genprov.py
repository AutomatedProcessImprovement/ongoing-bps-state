"""Record how a generated BPS model was produced.

The provenance audit's one unclosed item: *no generator recorded its
arguments*, so every structural claim about a model ("branches are symmetric",
"the gate CV is 0.2%") had to be reconstructed from the artefact instead of
read off the command that made it. Reconstruction agreed with the claims, but
it is weaker evidence and it cannot distinguish a model that was regenerated
from one that was hand-edited.

The record goes in a sidecar ``<name>.gen.json`` rather than inside the params
JSON, so the params file stays exactly what Prosimos consumes and a byte-diff
of two models is not polluted by metadata.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
            cwd=Path(__file__).resolve().parents[1],
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def write_sidecar(out_dir: Path, name: str, *, generator: str,
                  params: dict | None = None) -> Path:
    """Write ``<out_dir>/<name>.gen.json`` describing this generation run."""
    record = {
        "name": name,
        "generator": generator,
        "argv": sys.argv,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "parameters": params or {},
    }
    path = Path(out_dir) / f"{name}.gen.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path
