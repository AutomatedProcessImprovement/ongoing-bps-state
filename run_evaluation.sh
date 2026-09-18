#!/usr/bin/env bash
#
# Reproduce the state-metrics evaluation grid end to end, from a fresh clone.
#
#   ./run_evaluation.sh                # install if needed, then run everything
#   ./run_evaluation.sh --setup-only   # just build the venv
#   ./run_evaluation.sh --smoke        # 2 replications, 2 levels -- ~5 minutes
#   ./run_evaluation.sh --jobs 4       # cap concurrent simulations
#
# The grid is eleven models x eleven measures x twenty replications x six
# perturbation levels. Every model is independent, so they run concurrently;
# on a 16-core machine expect roughly an hour, most of it in the pairing
# family, which simulates 3000 cases per replication.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

VENV="$ROOT/venv"
PY="$VENV/bin/python"
WORK="${WORK_DIR:-$ROOT/outputs/evaluation_run}"
RUNS=20
LEVELS=""                 # empty = each dataset's own six-step ladder
JOBS="$( (command -v nproc >/dev/null && nproc) || echo 4 )"
SETUP_ONLY=0
SKIP_SETUP=0
# The reference table is only comparable at the reported protocol.
FULL_PROTOCOL=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --setup-only) SETUP_ONLY=1; shift ;;
    --skip-setup) SKIP_SETUP=1; shift ;;
    --smoke)      RUNS=2; LEVELS="0,100"; FULL_PROTOCOL=0; shift ;;
    --runs)       RUNS="$2"; [[ "$2" == 20 ]] || FULL_PROTOCOL=0; shift 2 ;;
    --jobs)       JOBS="$2"; shift 2 ;;
    --work-dir)   WORK="$2"; shift 2 ;;
    -h|--help)    sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- 1. install
if [[ $SKIP_SETUP -eq 0 ]]; then
  say "Setting up the virtualenv at $VENV"
  if [[ ! -x "$PY" ]]; then
    python3 -m venv "$VENV"
  fi
  "$PY" -m pip install --quiet --upgrade pip
  # requirements.txt pins the Prosimos fork by branch. pip will not notice new
  # commits on a branch it has already installed, because the upstream version
  # string is not bumped -- so force that one line and let the rest resolve
  # normally.
  "$PY" -m pip install --quiet -r requirements.txt
  "$PY" -m pip install --quiet --force-reinstall --no-deps \
      "git+https://github.com/AutomatedProcessImprovement/Prosimos@short-term-simulation"
  "$PY" - <<'EOF'
import prosimos, inspect
from prosimos.simulation_engine import run_simulation
ok = "process_state" in inspect.signature(run_simulation).parameters
print(f"  prosimos {prosimos.__file__}")
print(f"  short-term-simulation fork: {'yes' if ok else 'NO'}")
raise SystemExit(0 if ok else
                 "installed Prosimos does not accept process_state=; "
                 "the short-term-simulation branch did not install")
EOF
fi
[[ $SETUP_ONLY -eq 1 ]] && { say "Setup done."; exit 0; }

# ---------------------------------------------------------------- 2. models
# The generators are unseeded, so regenerating would produce DIFFERENT models
# and move every number. The models under samples/dev-samples/ are committed
# for exactly that reason and are used as they are; each carries a .gen.json
# recording the generator, argv and commit that produced it.
say "Checking the committed models"
for m in synthetic_composition6 synthetic_timeline6 synthetic_pairing_mirror6; do
  for ext in bpmn json gen.json; do
    f="samples/dev-samples/$m.$ext"
    [[ -f "$f" ]] || { echo "missing $f" >&2; exit 1; }
  done
  echo "  $m  ($("$PY" -c "import json;d=json.load(open('samples/dev-samples/$m.gen.json'));print(d['generator'],'@',d['git_commit'])"))"
done

# ---------------------------------------------------------------- 3. the grid
mkdir -p "$WORK"
LVL_ARG=""; [[ -n "$LEVELS" ]] && LVL_ARG="--levels $LEVELS"
# X1 runs a bespoke ladder (resource counts, not percentages).
X1_LEVELS="0,2,4,6,7,8"; [[ -n "$LEVELS" ]] && X1_LEVELS="0,8"

# tag|pipeline arguments. Composition carries three extra ladder variants that
# are not reported but justify the reported one; drop them for a quicker run.
cat > "$WORK/jobs.txt" <<EOF
C1|composition6_e2 --runs $RUNS --perturbation parallel_auto --automation-ladder stretch --automation-stretch-max 1.8 $LVL_ARG
C2|composition6_e4 --runs $RUNS --perturbation parallel_auto --automation-ladder stretch --automation-stretch-max 1.8 $LVL_ARG
C3|composition6_e6 --runs $RUNS --perturbation parallel_auto --automation-ladder stretch --automation-stretch-max 1.8 $LVL_ARG
T1|timeline6_e2 --runs $RUNS --perturbation front_back_swap $LVL_ARG
T2|timeline6_e4 --runs $RUNS --perturbation front_back_swap $LVL_ARG
T3|timeline6_e6 --runs $RUNS --perturbation front_back_swap $LVL_ARG
P1|pairing_mirror6_e2 --runs $RUNS --perturbation route_error_graded --gt-total-cases 3000 --sim-total-cases 3000 $LVL_ARG
P2|pairing_mirror6_e4 --runs $RUNS --perturbation route_error_graded --gt-total-cases 3000 --sim-total-cases 3000 $LVL_ARG
P3|pairing_mirror6_e6 --runs $RUNS --perturbation route_error_graded --gt-total-cases 3000 --sim-total-cases 3000 $LVL_ARG
X1|Loan-stable --runs $RUNS --perturbation resources --levels ${X1_LEVELS}
X2|Loan-stable --runs $RUNS --perturbation arrival_burst $LVL_ARG
EOF
MODELS=$(cut -d'|' -f1 "$WORK/jobs.txt" | tr '\n' ' ')

# The ground-truth log of each dataset is cached on first use and reused by
# every model of that family, so the three models of a family must not race to
# build it. Seed each family's cache with a one-replication run first.
say "Seeding the ground-truth caches (3 datasets)"
for seed in "composition6_e2 --perturbation parallel_auto --automation-ladder stretch --automation-stretch-max 1.8" \
            "timeline6_e2 --perturbation front_back_swap" \
            "pairing_mirror6_e2 --perturbation route_error_graded --gt-total-cases 3000 --sim-total-cases 3000"; do
  ds=${seed%% *}
  name=$("$PY" -c "from evaluation.state_metrics.datasets import DATASETS;print(DATASETS['$ds'].name)")
  if [[ -f "outputs/state_metrics/$name/prefix_source_log.csv" ]]; then
    echo "  $name: cached"
  else
    echo "  $name: building"
    # shellcheck disable=SC2086
    "$PY" -m evaluation.state_metrics.run_pipeline $seed --runs 1 --levels 0 \
        > "$WORK/gtseed_$name.log" 2>&1
  fi
done

SINCE=$(date +%s)
say "Running the grid: $(wc -l < "$WORK/jobs.txt") models, $RUNS replications, up to $JOBS at a time"
run_one() {
  tag="${1%%|*}"; argv="${1#*|}"
  # shellcheck disable=SC2086
  if "$PY" -m evaluation.state_metrics.run_pipeline $argv > "$WORK/$tag.log" 2>&1; then
    echo "  $tag done"
  else
    echo "  $tag FAILED -- see $WORK/$tag.log" >&2; return 1
  fi
}
export -f run_one; export PY WORK
if command -v xargs >/dev/null; then
  tr -d '\r' < "$WORK/jobs.txt" | xargs -P "$JOBS" -I{} bash -c 'run_one "$@"' _ {}
else
  while IFS= read -r line; do run_one "$line"; done < "$WORK/jobs.txt"
fi

# ------------------------------------------------------- 4. score and report
REP="evaluation/state_metrics/reporting"

say "Pinning each run to its job"
"$PY" "$REP/build_run_manifest.py" "$WORK" "$SINCE"

say "Recomputing every cell from the stored simulation logs"
# Reads only each run's sim_log.csv files, never its own results.csv, so the
# table cannot inherit a stale baseline bin or a superseded scoring pass.
printf '%s\n' $MODELS | xargs -P "$JOBS" -I{} \
    "$PY" "$REP/recompute_from_logs.py" "$WORK/run_manifest.json" {}
"$PY" - "$WORK" <<'EOF'
import glob, pickle, sys
work = sys.argv[1]
R = {}
for f in glob.glob(f"{work}/recomputed_*.pkl"):
    R.update(pickle.load(open(f, "rb")))
pickle.dump(R, open(f"{work}/recomputed.pkl", "wb"))
print(f"  merged {len(R)} models")
EOF

say "Scoring: c-index, bootstrap intervals, noise floors"
printf '%s\n' $MODELS | xargs -P "$JOBS" -I{} \
    "$PY" "$REP/build_source_of_truth.py" "$WORK" {}
"$PY" "$REP/build_source_of_truth.py" "$WORK" merge

say "Bin sensitivity (fixed 1h / window-20 / window-50 / window-100)"
printf '%s\n' $MODELS | xargs -P "$JOBS" -I{} \
    "$PY" "$REP/sweep_bin_grid.py" "$WORK/run_manifest.json" {}

# ---------------------------------------------------------------- 5. compare
say "Result"
"$PY" - "$WORK" "$FULL_PROTOCOL" <<'EOF'
import sys
import pandas as pd
work, full = sys.argv[1], sys.argv[2] == "1"
new = pd.read_csv(f"{work}/results_source_of_truth.csv")
ORDER = ["(a,c)", "act", "(a,r)", "case", "card", "cf", "ct", "ngd", "red",
         "cyc", "rtd"]
MODELS = ["C1", "C2", "C3", "T1", "T2", "T3", "P1", "P2", "P3", "X1", "X2"]
d = {(r.model, r.measure): r for r in new.itertuples()}
print("\nc-index on per-level means, Scope A.  * = bootstrap CI excludes 0.50\n")
print(f"{'model':6s}" + "".join(f"{m:>8s}" for m in ORDER))
for mo in MODELS:
    if (mo, "(a,c)") not in d:
        continue
    print(f"{mo:6s}" + "".join(
        f"{d[(mo, me)].c_index_means:>7.3f}"
        f"{'*' if d[(mo, me)].star_ci_excludes_half else ' '}" for me in ORDER))

if not full:
    print("""
Reference comparison skipped: it only means anything under the reported
protocol. The mean-first c-index over L levels has C(L,2) pairs, so this run's
shortened ladder gives one pair per cell -- every c-index is 0.000 or 1.000 and
every bootstrap interval is degenerate. Re-run without --smoke to compare.""")
    print(f"full table: {work}/results_source_of_truth.csv")
    raise SystemExit(0)

ref = pd.read_csv("evaluation/state_metrics/reference_results.csv")
r = {(x.model, x.measure): x for x in ref.itertuples()}
common = [k for k in d if k in r]
flips = [(k, r[k].star_ci_excludes_half, d[k].star_ci_excludes_half)
         for k in common
         if r[k].star_ci_excludes_half != d[k].star_ci_excludes_half]
print(f"\nvs evaluation/state_metrics/reference_results.csv: "
      f"{len(common)} cells compared, {len(flips)} changed significance")
for (mo, me), a, b in flips:
    print(f"  {mo:3s} {me:5s} {'lost' if a else 'gained'} its star "
          f"({r[(mo, me)].c_index_means:.3f} -> {d[(mo, me)].c_index_means:.3f})")
print("""
Prosimos is seeded, but a cell reproduces EXACTLY only under the same cached
ground-truth log, the same _seed_world seed and the same PYTHONHASHSEED. A
fresh run gives statistically equivalent numbers, not identical ones, so a few
cells near the star threshold moving is expected; a whole column flipping is
not.""")
print(f"full table: {work}/results_source_of_truth.csv")
EOF
