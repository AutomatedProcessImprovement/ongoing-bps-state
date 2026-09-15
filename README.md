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

| Paper | Dataset | `--perturbation` | `--levels` |
| --- | --- | --- | --- |
| C1 | `synthetic_parallel_auto` | `parallel_auto` | `0,25,50,75,100` |
| C2 | `synthetic_parallel_auto5` | `parallel_auto` | `0,25,50,75,100` |
| C3 | `synthetic_role_pools` | `role_swap` | `0,1,2,3,4,5` |
| T1 | `synthetic_front_back_gated_lv` | `front_back_swap` | `0,25,50,75,100` |
| T2 | `synthetic_front_back_gated4_lv` | `front_back_swap` | `0,25,50,75,100` |
| T3 | `synthetic_front_back_gated5_lv` | `front_back_swap` | `0,25,50,75,100` |
| P1 | `synthetic_route_error_wide` | `route_error` | `0,1,2,3,4,5,6` |
| P2 | `synthetic_route_error_asym` | `route_error` | `0,1,2,3,4` |
| P3 | `synthetic_route_error_deep` | `route_error` | `0,1,2,3,4,5` |
| X1 | `Loan-stable` | `resources` | `0,2,4,6,7,8` |
| X2 | `Loan-stable` | `arrival_burst` | `0,50,100,200,400` |

The timeline family reports the **low-variance gated** design (`*_gated*_lv`,
gate CV 0.2%). The ungated `synthetic_front_back_xor*` and the CV 7%
`synthetic_front_back_gated*` datasets are the two designs it was compared
against; [`docs/state-metrics/gate_variance_decision.md`](docs/state-metrics/gate_variance_decision.md)
records the comparison and the rule for picking one. In
`results_source_of_truth.csv` those three designs appear as `T1`–`T3`,
`T1'`–`T3'` and `T1lv`–`T3lv` respectively.

```bash
# the whole grid
python -m evaluation.state_metrics.run_pipeline synthetic_parallel_auto        --perturbation parallel_auto   --levels 0,25,50,75,100 --runs 20
python -m evaluation.state_metrics.run_pipeline synthetic_parallel_auto5       --perturbation parallel_auto   --levels 0,25,50,75,100 --runs 20
python -m evaluation.state_metrics.run_pipeline synthetic_role_pools           --perturbation role_swap       --levels 0,1,2,3,4,5    --runs 20
python -m evaluation.state_metrics.run_pipeline synthetic_front_back_gated_lv  --perturbation front_back_swap --levels 0,25,50,75,100 --runs 20
python -m evaluation.state_metrics.run_pipeline synthetic_front_back_gated4_lv --perturbation front_back_swap --levels 0,25,50,75,100 --runs 20
python -m evaluation.state_metrics.run_pipeline synthetic_front_back_gated5_lv --perturbation front_back_swap --levels 0,25,50,75,100 --runs 20
python -m evaluation.state_metrics.run_pipeline synthetic_route_error_wide     --perturbation route_error     --levels 0,1,2,3,4,5,6  --runs 20
python -m evaluation.state_metrics.run_pipeline synthetic_route_error_asym     --perturbation route_error     --levels 0,1,2,3,4      --runs 20
python -m evaluation.state_metrics.run_pipeline synthetic_route_error_deep     --perturbation route_error     --levels 0,1,2,3,4,5    --runs 20
python -m evaluation.state_metrics.run_pipeline Loan-stable                    --perturbation resources       --levels 0,2,4,6,7,8    --runs 20
python -m evaluation.state_metrics.run_pipeline Loan-stable                    --perturbation arrival_burst   --levels 0,50,100,200,400 --runs 20
```

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

**Prosimos is unseeded.** Re-running the grid gives statistically equivalent
numbers, not identical ones, and the generator scripts in `tools/generate_*.py`
are unseeded too — re-running those produces different models, which is why the
models themselves are committed under `samples/dev-samples/` rather than built
on demand. What *is* deterministic is the scoring: the bootstrap seeds from
SHA-256 of the key rather than `hash()`, so intervals are stable across
processes and machines.

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
