# Perturbation catalogue for state-metrics evaluation

Catalogue of perturbations to apply to a Prosimos GT simulation in order to
benchmark the state-metrics library against histogram-binning baselines
(`rtd`, `cycle_time`, `red`, `ngd_n2`).

The thesis: state metrics integrate disagreement **over time**, while baselines
collapse the time axis into histograms. Perturbations whose disagreement is
*temporal but aggregate-invariant* should be the strongest demonstrations of
state-metric superiority.

## Legend

- 🟢 — state metrics should expose; baselines often miss
- 🟡 — baseline territory (binning's home turf); state metrics may catch indirectly
- 🔴 — both should agree (sanity check / negative control)

## Catalogue

### 1. Time-localised perturbations
🟢 **Time-shifted bottleneck**
- GT: 1-hour resource outage at `cutoff + 0.5h`. Sim: same outage at `cutoff + horizon - 0.5h`.
- Aggregate: total queueing, throughput, RTD/cycle_time histograms ≈ identical.
- State metrics: see two trajectories diverging at different times.
- Implementation: pure JSON, calendar exception in `resource_calendars`.

🟢 **Stall-and-recover**
- Sim freezes all activity for 30 min mid-horizon, then catches up.
- Aggregate stats unaffected once recovered.
- Implementation: pure JSON, brief calendar zero-out.

🟢 **Mid-horizon regime change vs steady-state**
- GT keeps full pool first half, drops 5 resources second half; sim runs with permanently reduced pool sized for same average throughput.
- End-state and aggregate match; pile-up timing differs.
- Implementation: pure JSON, calendar with two periods.

### 2. Compositional / activity-mix
🟢 **Phantom rework loop**
- Sim re-routes 20% of cases through an extra activity, compensated by faster service so cycle-time matches.
- RTD/cycle_time invariant; `activity.jaccard`, `case.jaccard` change.
- Implementation: requires BPMN edit (add loop) + JSON service-time tweak.

🟢 **Branch-probability shift**
- Change a gateway split from 70/30 to 30/70.
- `ngd_n2` natural home; RTD/cycle_time barely move if branches have similar durations.
- Implementation: pure JSON, edit `gateway_branching_probabilities`.

🟢 **Activity merging / splitting**
- Fuse two consecutive activities into one (or vice-versa).
- Cycle_time invariant; active-bag cardinality profile differs.
- Implementation: BPMN edit + JSON.

### 3. Resource-mix (same capacity, different identity)
🟢 **Resource swap**
- Pools A and B have identical service-time distributions but distinct IDs; sim assigns the wrong pool to an activity.
- Cycle_time, RTD, red blind. `activity_role` view (with role map) catches.
- **Cleanest single-perturbation win for state metrics.**
- Implementation: pure JSON, rename resource IDs in `resource_profiles` and `task_resource_distribution`.

🟢 **Cross-training leak**
- Sim lets resources execute activities they shouldn't.
- Throughput same; (activity, resource) pairs differ.
- Implementation: pure JSON, extend `task_resource_distribution`.

🟢 **Resource-pool consolidation**
- Sim merges two specialised pools into one generalist.
- Capacity preserved; who-does-what differs over time.
- Implementation: pure JSON.

### 4. Calendar / working-hours
🟢 **Shifted working calendar**
- GT 09–17, sim 13–21. All cases delayed uniformly.
- `red` sees a scalar shift; state metrics see anti-correlated active-bags at any wall-clock `t`.
- Implementation: pure JSON, offset `time_periods` in `resource_calendars`.

🟢 **Hidden holiday in sim**
- Sim observes a 1-day holiday GT doesn't.
- Histograms compress; state metrics see a flatline period.
- Implementation: pure JSON, calendar exception.

### 5. Concurrency / WIP-policy
🟢 **WIP cap**
- Sim refuses new arrivals when active count > N.
- If cap rarely hit, throughput matches; active-bag upper tail truncated.
- Implementation: **not stock Prosimos** — needs a fork or post-hoc filter on the sim log.

🟢 **Batching**
- Sim batches activity X (waits until 5 cases queued, then releases).
- Throughput preserved on average; active-bag oscillates.
- Implementation: not stock Prosimos.

🟢 **Priority inversion**
- Sim FIFO, GT priority queue (or vice-versa).
- Throughput same on average; case-level timing reorders.
- Implementation: not stock Prosimos.

### 6. Stochastic-shape (binning's home turf)
🟡 **Same mean, different variance**
- Replace exponential with deterministic durations or vice versa.
- Baselines designed for this. State metrics catch indirectly via active-bag fluctuation.
- Implementation: pure JSON, swap `distribution_name`.

🟡 **Bimodal vs unimodal durations** with matched mean. Baselines win.

🟡 **Heavy-tail injection**
- 5% of cases get 10× duration.
- Cycle-time tail moves; state metrics see occasional long-active cases.
- Implementation: pure JSON, optionally extend Prosimos with case-attribute-conditional service times.

### 7. Arrival-process
🟢 **Phase-shifted bursts**
- Same arrival-rate marginal, different timing.
- Implementation: pure JSON, custom `arrival_time_calendar`.

🔴 **Constant rate change**
- Sim λ = 1.2× GT λ.
- Should move every metric. Sanity baseline.
- Implementation: pure JSON.

🟡 **Inter-arrival distribution shape** — Poisson vs deterministic with same mean.

### 8. Inter-case interactions / unmodelled effects
🟢 **Hidden contention**
- Secondary process consumes a fraction of pool's time during business hours only.
- Cycle-time histogram drifts; state metrics see time-of-day pattern.
- Implementation: pure JSON, complex calendar.

🟢 **Priority inversion** (see §5).

### 9. Labelling / observation-noise (negative controls)
🔴 **Activity rename**
- Rename one activity in sim only.
- State metrics will explode. Bad demo — trivial schema diff.
- Useful as robustness test.
- Implementation: pure JSON + BPMN.

🔴 **Timestamp jitter**
- ±5min noise on event times.
- Should barely move any metric. Stability test.
- Implementation: post-hoc on sim log.

## Recommended demonstration set

Three rows that pin **all baselines within their level-0 noise floor** but state metrics move:

| Demo | Perturbation | Expected state-metric signal | Expected baseline signal |
|---|---|---|---|
| Time-shifted bottleneck | resource outage at start vs end of horizon | strong (case.jaccard, cardinality) | within noise |
| Resource swap | pool A↔B, same service-time | strong (activity_role) | within noise |
| Calendar shift | GT 09–17, sim 13–21 | strong (all four state metrics) | partial (red shifts) |

Plus 1–2 honest rows where baselines win (heavy-tail injection) so the paper isn't a strawman.

## Implementation feasibility

| Class | Tooling needed |
|---|---|
| Pure JSON edits | extend `evaluation/state_metrics/perturb.py` with sibling functions next to `build_perturbed_params` |
| BPMN edits (small, surgical) | `lxml` — Prosimos BPMNs are small, only constraint is task-id ↔ JSON-id linkage |
| BPMN authoring from scratch | bpmn.io free web editor (no Apromore needed) |
| Prosimos-semantics changes (WIP cap, batching) | Prosimos fork or post-hoc sim-log filter |

## Workflow to prototype any perturbation

1. Locate base BPMN + JSON in `samples/.../<dataset>/`.
2. Write `build_<name>_params(params, level)` in `perturb.py` returning a perturbed JSON dict.
3. Wire perturbation name into `pipeline.py`'s level loop (matches existing `resources` / `duration` pattern).
4. Add `DatasetSpec.default_<name>_levels` in `datasets.py`.
5. Run `python -m evaluation.state_metrics.run_pipeline <DATASET> --perturbation <name> --runs 10`.
6. Analyse with `/tmp/full_analysis.py` (all four state metrics + baselines, mean ± std per level).

## Cross-references in this repo

- `evaluation/state_metrics/perturb.py` — existing pattern (`build_perturbed_params`, resource removal).
- `evaluation/state_metrics/pipeline.py` — perturbation dispatch, level loop, metrics computation.
- `evaluation/state_metrics/datasets.py` — `DatasetSpec`, default levels per perturbation.
- `evaluation/rtd.py` — RTD baseline.
- `state_metrics/api.py` — `compute_state_distance`, `compute_all_state_distances` with `window=` kwarg.
- `samples/icpm-2025/synthetic/Loan-stable/` — base BPMN + Prosimos JSON for the synthetic dataset used in current experiments.
