# Process State Evaluation

Research codebase for **process-state-based short-term simulation**: taking a
running process's partial state at a cut-off and continuing the simulation from
it, rather than from an empty marking. It supports three lines of work:

1. **ICPM-2025** — three-flavour comparison (process state vs warm-up approaches).
2. **Uncertainty / clustering extension** — confidence-interval estimation over
   simulation error, using clustering over pre-cut feature vectors.
3. **State-based metrics** — per-timestamp multiset distances over active
   activity instances, benchmarked against baseline log-distance measures under
   controlled perturbations. This is the current line; see
   [`docs/state-metrics/`](docs/state-metrics/) for the recorded results.

---

## Setup

Requires **Python 3.11** (what this is developed and tested on) and **git**, which
pip needs to fetch the Prosimos fork.

```bash
git clone https://github.com/AutomatedProcessImprovement/ongoing-bps-state.git
cd ongoing-bps-state
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

Eleven models, twenty replications each. Wall time is roughly 3–25 minutes per
model on one machine, about two hours for the grid.

Each family now has **one** BPMN geometry, and its three models differ only in
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
| P1 | `pairing6_e2` | `route_error_graded` | chain of 6 gateways | 2 inverted | `0,20,40,60,80,100` |
| P2 | `pairing6_e4` | `route_error_graded` | chain of 6 gateways | 4 inverted | `0,20,40,60,80,100` |
| P3 | `pairing6_e6` | `route_error_graded` | chain of 6 gateways | 6 inverted | `0,20,40,60,80,100` |
| X1 | `Loan-stable` | `resources` | — | — | `0,2,4,6,7,8` |
| X2 | `Loan-stable` | `arrival_burst` | — | — | `0,50,100,200,400` |

The three geometries are `samples/dev-samples/synthetic_composition6`,
`synthetic_timeline6` and `synthetic_pairing6`; the `*_e2` / `*_e4` / `*_e6`
dataset keys bind the same BPMN and params to a different perturbation extent.
Each model carries a `.gen.json` sidecar (written by `tools/genprov.py`)
recording the generator, its argv and the commit it ran at.

The extents are **even** because the timeline family cannot do odd ones: a
cross-branch position swap moves one task each way, so preserving the branch
totals takes an even number of positions — a single-position swap is either a
no-op or it changes branch length as well as content, which is the confound the
one-geometry redesign exists to remove.

```bash
# the whole grid
python -m evaluation.state_metrics.run_pipeline composition6_e2 --runs 20 --perturbation parallel_auto      --automation-ladder stretch --automation-stretch-max 1.8
python -m evaluation.state_metrics.run_pipeline composition6_e4 --runs 20 --perturbation parallel_auto      --automation-ladder stretch --automation-stretch-max 1.8
python -m evaluation.state_metrics.run_pipeline composition6_e6 --runs 20 --perturbation parallel_auto      --automation-ladder stretch --automation-stretch-max 1.8
python -m evaluation.state_metrics.run_pipeline timeline6_e2    --runs 20 --perturbation front_back_swap
python -m evaluation.state_metrics.run_pipeline timeline6_e4    --runs 20 --perturbation front_back_swap
python -m evaluation.state_metrics.run_pipeline timeline6_e6    --runs 20 --perturbation front_back_swap
python -m evaluation.state_metrics.run_pipeline pairing6_e2     --runs 20 --perturbation route_error_graded
python -m evaluation.state_metrics.run_pipeline pairing6_e4     --runs 20 --perturbation route_error_graded
python -m evaluation.state_metrics.run_pipeline pairing6_e6     --runs 20 --perturbation route_error_graded
python -m evaluation.state_metrics.run_pipeline Loan-stable     --runs 20 --perturbation resources     --levels 0,2,4,6,7,8
python -m evaluation.state_metrics.run_pipeline Loan-stable     --runs 20 --perturbation arrival_burst --levels 0,50,100,200,400
```

The composition ladder **lengthens** the perturbed activities rather than
shortening them, staying strictly inside the critical branch's slack. Shrinking
saturates: past a scale factor of `(k-1)/k` the k-th activity of the chain no
longer overlaps the interval it occupies in the unperturbed model, its pairs
stop changing, and the distance flattens — the closed form is
`perturb.overlap_bound`, and `--automation-ladder shrink_bounded` is the
alternative fix that clamps the ladder at that bound instead.

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
identical ones. See
[`docs/state-metrics/reproducibility.md`](docs/state-metrics/reproducibility.md).

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

The numbers actually reported, with their provenance, are in
[`docs/state-metrics/`](docs/state-metrics/) — `results_source_of_truth.csv`
holds every reported cell, and `provenance_audit.md` records the pass that
recomputed each one from its run's stored logs.

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
docs/state-metrics/      recorded results, provenance audit, design decisions
tools/                   generate_*.py, the synthetic BPS model generators
samples/                 BPMN models, Prosimos params, gzipped event logs
tests/                   pytest suite; tests/e2e/ runs a real simulator
outputs*/                generated, gitignored; every run under its own short uuid
```

`CLAUDE.md` (gitignored) carries the longer architecture notes for anyone
working on the code.
