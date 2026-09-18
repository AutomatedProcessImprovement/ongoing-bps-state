# Process State Evaluation

Research codebase for **process-state-based short-term simulation**: taking a
running process's partial state at a cut-off and continuing the simulation from
it, rather than from an empty marking. It supports three lines of work:

1. **ICPM-2025** — three-flavour comparison (process state vs warm-up approaches).
2. **Uncertainty / clustering extension** — confidence-interval estimation over
   simulation error, using clustering over pre-cut feature vectors.
3. **State-based metrics** — per-timestamp multiset distances over active
   activity instances, benchmarked against baseline log-distance measures under
   controlled perturbations. This is the current line.

---

## Quick start

One script installs everything and reproduces the whole state-metrics grid:

```bash
git clone https://github.com/AutomatedProcessImprovement/ongoing-bps-state.git
cd ongoing-bps-state
./run_evaluation.sh
```

It builds a virtualenv, installs the pinned dependencies **including the
Prosimos fork** and checks the fork actually took, runs all eleven models
concurrently, recomputes every cell from the stored simulation logs, scores the
grid, and prints it next to the committed reference table so you can see which
cells moved.

```bash
./run_evaluation.sh --smoke        # 2 replications, 2 levels -- a few minutes
./run_evaluation.sh --setup-only   # just build the venv
./run_evaluation.sh --jobs 4       # cap concurrency (default: all cores)
./run_evaluation.sh --skip-setup   # venv already built
./run_evaluation.sh --help
```

Requires **Python 3.11** and **git** (pip needs it to fetch the Prosimos fork).
Everything else the run reads — BPMN models, Prosimos parameter JSONs, event
logs — is in the clone; nothing is downloaded at run time.

The full grid is eleven models x twenty replications x six perturbation levels.
On a 16-core machine expect roughly an hour, most of it in the pairing family,
which simulates 3000 cases per replication. Results land in
`outputs/evaluation_run/` (override with `--work-dir`); the scored table is
`results_source_of_truth.csv` there.

`--smoke` shortens the ladder to two levels, which is enough to prove the
install works but not to reproduce anything: the mean-first c-index over L
levels has C(L,2) pairs, so at two levels every cell is 0.000 or 1.000 and the
bootstrap intervals are degenerate. The script says so and skips the reference
comparison rather than printing a meaningless diff.

---

## Setup by hand

`run_evaluation.sh` does all of this; these are the same steps if you would
rather drive them yourself, or you only want the environment.

```bash
python3.11 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **The Prosimos dependency is a fork, not the PyPI package.** `requirements.txt`
> pins `AutomatedProcessImprovement/Prosimos@short-term-simulation`; a vanilla
> Prosimos install will not accept the `process_state=` kwarg and every
> simulation call will fail. The fork's version string is not bumped on new
> commits, so to pick them up:
>
> ```bash
> pip install --force-reinstall --no-deps \
>   "git+https://github.com/AutomatedProcessImprovement/Prosimos@short-term-simulation"
> ```

### Check the install

```bash
pytest tests/ -m "not e2e" -q     # 258 unit tests, ~8 s, no simulator
pytest tests/e2e -m e2e           # 13 end-to-end flows, runs real Prosimos
```

If the unit tests pass but the e2e ones fail on `process_state=`, the fork did
not install — re-run the force-reinstall above.

Everything the pipelines read is in the clone: BPMN models, Prosimos parameter
JSONs, and gzipped event logs under `samples/`. Nothing has to be downloaded.

---

## State-metrics benchmark

Runs a ground-truth simulation, picks a high-WIP cut-off, then re-simulates from
the partial state under a perturbation applied at increasing levels, and scores
every metric by how monotonically its distance rises with the injected error.

```bash
python -m evaluation.state_metrics.run_pipeline synthetic_parallel_auto \
    --perturbation parallel_auto --levels 0,25,50,75,100 --runs 20
```

Results land in `outputs/state_metrics/<dataset>/<perturbation>/run_<id>/`:

| path | contents |
| --- | --- |
| `results.csv` | long-form distances: one row per (level, replication, scope, metric) |
| `rankings.csv` | c-index and bootstrap interval per metric |
| `cutoff.json` | the chosen cut-off and the horizon derived from it |
| `utilization.csv` | per-level resource utilisation, for checking the run is not saturated |
| `prefix.csv` | the pre-cut-off log slice every short-term run resumes from |
| `level_<L>/k_<K>/sim_log.csv` | the simulated log behind every cell |
| `baseline/k_<K>/sim_log.csv` | the unperturbed comparison run |

`python -m evaluation.state_metrics.run_pipeline --help` lists every
perturbation family and the cut-off strategies.

### Reproducing the reported grid

`./run_evaluation.sh` runs everything below. This section is what it runs and
why, for when you want one model rather than the grid.

Each family has **one** BPMN geometry, and its three models differ only in
how much of that geometry the injected error touches. The extent is matched
across families — two, four, six — and every family runs the same six-step
ladder, so a cell is comparable down a column as well as across one.

| Paper | Dataset | `--perturbation` | Geometry | Extent | `--levels` |
| --- | --- | --- | --- | --- | --- |
| C1 | `composition6_e2` | `parallel_auto` | non-critical chain of 6 | 2 lengthened | `0,20,40,60,80,100` |
| C2 | `composition6_e4` | `parallel_auto` | non-critical chain of 6 | 4 lengthened | `0,20,40,60,80,100` |
| C3 | `composition6_e6` | `parallel_auto` | non-critical chain of 6 | 6 lengthened | `0,20,40,60,80,100` |
| T1 | `timeline6_e2` | `front_back_swap` | two branches of 6 | 2 swapped | `0,20,40,60,80,100` |
| T2 | `timeline6_e4` | `front_back_swap` | two branches of 6 | 4 swapped | `0,20,40,60,80,100` |
| T3 | `timeline6_e6` | `front_back_swap` | two branches of 6 | 6 swapped | `0,20,40,60,80,100` |
| P1 | `pairing_mirror6_e2` | `route_error_graded` | two mirrored branches of 6 | 2 reversed | `0,20,40,60,80,100` |
| P2 | `pairing_mirror6_e4` | `route_error_graded` | two mirrored branches of 6 | 4 reversed | `0,20,40,60,80,100` |
| P3 | `pairing_mirror6_e6` | `route_error_graded` | two mirrored branches of 6 | 6 reversed | `0,20,40,60,80,100` |
| X1 | `Loan-stable` | `resources` | — | — | `0,2,4,6,7,8` |
| X2 | `Loan-stable` | `arrival_burst` | — | — | `0,50,100,200,300,400` |

The three geometries are `samples/dev-samples/synthetic_composition6`,
`synthetic_timeline6` and `synthetic_pairing_mirror6`; the `*_e2` / `*_e4` / `*_e6`
dataset keys bind the same BPMN and params to a different perturbation extent.
Each model carries a `.gen.json` sidecar (written by `tools/genprov.py`)
recording the generator, its argv and the commit it ran at.

The extents are **even** because the timeline family cannot do odd ones: a
cross-branch position swap moves one task each way, so preserving the branch
totals takes an even number of positions — a single-position swap is either a
no-op or it changes branch length as well as content, which is the confound the
one-geometry redesign exists to remove.

```bash
# one model at a time -- this is exactly what run_evaluation.sh issues
python -m evaluation.state_metrics.run_pipeline composition6_e2 --runs 20 --perturbation parallel_auto --automation-ladder stretch --automation-stretch-max 1.8
python -m evaluation.state_metrics.run_pipeline composition6_e4 --runs 20 --perturbation parallel_auto --automation-ladder stretch --automation-stretch-max 1.8
python -m evaluation.state_metrics.run_pipeline composition6_e6 --runs 20 --perturbation parallel_auto --automation-ladder stretch --automation-stretch-max 1.8
python -m evaluation.state_metrics.run_pipeline timeline6_e2    --runs 20 --perturbation front_back_swap
python -m evaluation.state_metrics.run_pipeline timeline6_e4    --runs 20 --perturbation front_back_swap
python -m evaluation.state_metrics.run_pipeline timeline6_e6    --runs 20 --perturbation front_back_swap
python -m evaluation.state_metrics.run_pipeline pairing_mirror6_e2 --runs 20 --perturbation route_error_graded --gt-total-cases 3000 --sim-total-cases 3000
python -m evaluation.state_metrics.run_pipeline pairing_mirror6_e4 --runs 20 --perturbation route_error_graded --gt-total-cases 3000 --sim-total-cases 3000
python -m evaluation.state_metrics.run_pipeline pairing_mirror6_e6 --runs 20 --perturbation route_error_graded --gt-total-cases 3000 --sim-total-cases 3000
python -m evaluation.state_metrics.run_pipeline Loan-stable     --runs 20 --perturbation resources     --levels 0,2,4,6,7,8
python -m evaluation.state_metrics.run_pipeline Loan-stable     --runs 20 --perturbation arrival_burst
```

The three models of a family share one ground-truth simulation, which is cached
on first use — so run one model of each family to completion before starting
its siblings, or they will race to build the same cache.

### Scoring a finished grid

The reported table is **not** read out of the runs' own `results.csv` files. It
is recomputed from the stored `sim_log.csv` files, so it cannot inherit a stale
baseline bin or a superseded scoring pass. Four scripts under
`evaluation/state_metrics/reporting/`, in order:

```bash
REP=evaluation/state_metrics/reporting
W=outputs/evaluation_run          # holds jobs.txt, written by run_evaluation.sh

# 1. pin each reported cell to a run directory
python $REP/build_run_manifest.py $W <unix-time-the-batch-started>
# 2. re-derive all eleven measures from the logs (one process per model)
python $REP/recompute_from_logs.py $W/run_manifest.json C1
# 3. c-index, bootstrap intervals, noise floors (one process per model)
python $REP/build_source_of_truth.py $W C1
python $REP/build_source_of_truth.py $W merge
# 4. bin sensitivity: fixed 1 h vs window/20, /50, /100
python $REP/sweep_bin_grid.py $W/run_manifest.json C1
```

Step 1 matters more than it looks. Nine composition runs land in one output
directory, so a run cannot be identified by "the newest one under this
dataset", and matching by launch order silently mis-assigned nine of seventeen
runs when it was tried. Each run instead declares its own extent and ladder in
its `level_<top>_manifest.json`, and the manifest builder matches on that and
refuses rather than guesses.

Step 3 is the slow one — 100 bootstrap seeds x 1000 iterations per cell — which
is why it takes one model at a time and merges at the end.

`sweep_bin_grid.py`'s `window/50` column must reproduce the scored table
exactly; that is the check that the grid and the sensitivity table came from
the same scoring code.

`evaluation/state_metrics/reference_results.csv` is the committed result of all
of this: 121 rows, eleven models x eleven measures, with point estimates,
bootstrap intervals, noise floors, utilisation and the run id and commit behind
each cell. `run_evaluation.sh` diffs a fresh run against it.

`evaluation/state_metrics/reporting/check_pairing_invariants.py <run-dir>`
measures, on the logs, each population quantity the mirrored pairing design
claims to preserve, separating a real residual from replication noise.

The composition ladder **lengthens** the perturbed activities rather than
shortening them, staying strictly inside the critical branch's slack. Shrinking
saturates: past a scale factor of `(k-1)/k` the k-th activity of the chain no
longer overlaps the interval it occupies in the unperturbed model, its pairs
stop changing, and the distance flattens — the closed form is
`perturb.overlap_bound`, and `--automation-ladder shrink_bounded` is the
alternative fix that clamps the ladder at that bound instead.

All three geometries execute **nine activities per case**: an opening activity,
one long common activity, six perturbable elements, and a closing activity.
Composition runs the six in parallel with the long one; timeline and pairing
run them on one of two branches.

The pairing branches carry the **same six labels in mirrored order** -- a case
routed one way runs `R_1 ... R_6`, a case routed the other way runs
`R_6 ... R_1` -- with one shared duration distribution. Position `i` offers
`{R_i, R_(7-i)}`, so reversing a centred even window leaves the per-position
population multiset, the per-case label set, case duration and within-case
event positions all unchanged, and only which case is where differs. That makes
every aggregate blind by construction rather than by luck; the residual is
finite-sample (P and Q must balance at each position) and is measured and
reported rather than assumed. The earlier 3-vs-3 design carried different
labels on the two branches, so the plain `activity` view saw the error too.

Composition activities run at **CV 7 %**, like the other two families. With the
deterministic `fix` durations it used to have, two level-0 runs produced the
same log, the measured noise floor was exactly zero, the replication budget did
nothing, every bootstrap interval collapsed to a point and `rise_over_floor`
was undefined.

The `activity_role` view is still implemented and still described in the paper,
but it is not in the reported grid: with no role perturbation left in the
family set it is identical to `activity` on every model, which is checked
rather than assumed (see the provenance audit).

**Reproducibility boundary.** A cell reproduces *exactly* only under all three
of: the same cached ground-truth log, the same `_seed_world` seed, and the same
`PYTHONHASHSEED`. Prosimos is seeded through `_seed_world`, but parts of the
engine iterate over hash-ordered containers, so the interpreter's hash seed
still changes the draw order — on `Loan-stable`, hash seed 0 gives a 2737-row
log where seeds 1, 2 and 12345 give a 2621-row one from identical params.
Without pinning all three you get statistically equivalent numbers, not
identical ones. "What will and will not reproduce" below has the detail.

### Re-scoring without re-simulating

Rankings are a pure function of a `results.csv`, so they can be recomputed from
a finished run — for a different bootstrap budget, or after a change to the
scoring code — without touching the simulator:

```bash
python -m evaluation.state_metrics.run_pipeline \
    --rankings-only outputs/state_metrics/<dataset>/<perturbation>/run_<id>/results.csv \
    --bootstrap-iters 2000
```

### What will and will not reproduce

**Prosimos is seeded, but not sufficient on its own.** `pipeline._seed_world`
seeds the engine, and three composition runs launched with the same seed came
back byte-identical — but two `Loan-stable` runs with the same params and the
same seed did not, because parts of the engine iterate over hash-ordered
containers. A cell therefore reproduces exactly only under all three of: the
same **cached ground-truth log** (`build_basic_sim_for_prefix` is not seeded,
so a fresh GT is a different GT), the same **`_seed_world` seed**, and the same
**`PYTHONHASHSEED`**. On `Loan-stable`, hash seed 0 produces a 2737-row log
where seeds 1, 2 and 12345 all produce a 2621-row one. Change any of the three
and you get statistically equivalent numbers, not identical ones.

The generator scripts in `tools/generate_*.py` are unseeded too — re-running
those produces different models, which is why the models themselves are
committed under `samples/dev-samples/` rather than built on demand, each with a
`.gen.json` sidecar recording the argv and commit that produced it.

What *is* deterministic regardless is the scoring: the bootstrap seeds from
SHA-256 of `(seed, scope, family, metric)` rather than `hash()`, so intervals
are stable across processes and machines.

Every reported cell, with its provenance, is in
[`evaluation/state_metrics/reference_results.csv`](evaluation/state_metrics/reference_results.csv):
the point estimate, its bootstrap interval, the fraction of 100 bootstrap seeds
that would star it, the noise floor, the model geometry, and the run id and
commit the cell came from. `run_evaluation.sh` diffs a fresh run against it and
reports which cells changed significance.

---

## ICPM-2025 pipeline

```bash
python -m evaluation.short_term_simulation.evaluate_with_existing_alog DATASET_NAME \
    --runs 10 --cut-strategy fixed
```

- `DATASET_NAME`: a dataset (`BPIC_2012`, `BPIC_2017`, `WORK_ORDERS`, `P2PFIN`,
  `LOAN_STABLE`, `LOAN_CIRCADIAN`, `LOAN_UNSTABLE`, `P2P_STABLE`,
  `P2P_CIRCADIAN`, `P2P_UNSTABLE`) or a group (`ALL`, `SYNTHETIC`, `REAL-LIFE`).
- `--runs`: Monte-Carlo repetitions per cut-off (default 10).
- `--cut-strategy`: `fixed` (one timestamp from the dataset config), `wip3`
  (WiP percentiles 10/50/90), or `segment10` (ten random points in equal
  segments).

Per-dataset paths, cut-offs and horizons live in the `DATASETS` dict in
`evaluation/short_term_simulation/evaluate_with_existing_alog.py`; add new
datasets there. Output goes to `outputs/<DATASET>/<run_id>/`: the reference
subsets (`A_event_filter.csv`, `A_ongoing.csv`, `A_complete.csv`), the
simulation logs per repetition, and `final_results.json`.

## Clustering / uncertainty extension

```bash
python -m evaluation.clustering.clustered_short_term_simulation
python -m evaluation.clustering.check_ci_calibration
```

> Neither entry point takes command-line arguments — running one starts work
> immediately, on whatever is hardcoded. Configure them by editing the
> constants in the source: dataset, replication count, error metric, cluster
> count and output root at the top of `main()` in
> `clustered_short_term_simulation.py`, and `EXP_ROOT` in `main()` in
> `check_ci_calibration.py`.

Outputs land under `outputs_confidence_v3/<dataset>/<run_id>/` as
`train_samples.csv` / `test_samples.csv`, which `models.py` and
`check_ci_calibration.py` then read.

---

## Single run

```bash
python main.py EVENT_LOG.csv MODEL.bpmn PARAMS.json \
    --simulate --start_time 2024-01-01T00:00:00Z --simulation_horizon 2024-01-08T00:00:00Z
```

Computes the partial process state at the cut-off, writes it to `output.json`,
and — with `--simulate` — continues a Prosimos short-term simulation from it.

---

## Repository layout

```
src/                     core library: event log + BPMN -> partial process state -> simulation
  runner.py              the single orchestration entry point (stages A-G)
  state_computer.py      per-case control-flow state, ongoing/enabled activities
  attributes.py          case and event attributes carried across the cut-off
state_metrics/           standalone multiset-distance library over active instances
  api.py                 compute_state_distance / compute_all_state_distances
  core.py                the joint sweep over both logs' event boundaries
  views.py distances.py  projections, and the Jaccard / cardinality distances
evaluation/              benchmarks built on the two libraries above
  short_term_simulation/ ICPM-2025 three-flavour pipeline
  clustering/            uncertainty / CI extension
  state_metrics/         perturbation benchmark: pipeline, perturbations, datasets, ranking
    reporting/           recompute, score and audit a finished grid
    reference_results.csv  every reported cell, with its provenance
run_evaluation.sh        install + run + score + compare, in one command
tools/                   generate_*.py, the synthetic BPS model generators
samples/                 BPMN models, Prosimos params, gzipped event logs
tests/                   pytest suite; tests/e2e/ runs a real simulator
outputs*/                generated, gitignored; every run under its own short uuid
```

`CLAUDE.md` (gitignored) carries the longer architecture notes for anyone
working on the code.
