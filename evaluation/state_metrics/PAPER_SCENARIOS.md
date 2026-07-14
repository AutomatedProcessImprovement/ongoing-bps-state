# Paper experiment scenarios — where state metrics win

Final experiment set for the short-term-simulation state-metrics paper, after
David's review of the scenario list. The goal is to show, on controlled
synthetic data, that the proposed state-based metric **catches per-case
continuation errors that the SOTA (long-term) metrics miss because those errors
*even out* in the aggregate**.

Baselines referenced: cycle-time MAE (`cycle_time`), N-gram distance (`ngd_n2`),
relative event distribution (`red`), remaining-time distribution (`rtd`) — the
measures from the Arik et al. log-distance paper, designed for **long-term**
simulation where aggregate fidelity is what matters.

## Methodology (David's core instruction)

> **Apply the modification in the BPS model, not the log.** Generate the ground
> truth with the *original* model; then modify the model so that
> Prosimos-short-term continues the ongoing cases with the *wrong* one.
> Evaluation question: **can the metric detect the deviation from the original
> model?**

Story for the paper: *we have an ongoing process; we want to simulate its
continuation, but the BPS model has an error (the modification); is the metric
able to recognise it?* This is cleaner and more realistic than mutating the
event log post-hoc.

**Why our metric wins — the hypothesis.** We *pair* the ongoing cases and
evaluate the continuation of each case with tuples `(case_id, …)`. When
per-case errors **even out** — a *desirable* property for long-term simulation —
the SOTA metrics, which only look at aggregations, are blind by construction.
The per-instant, per-case state view is not. Every scenario below is built to
create exactly that situation: hold every aggregate/marginal summary fixed
(cycle time, utilisation ρ, WIP/λ, overall ratios, activity counts) and change
only *what is active*, *where in time*, or *which case is on which path*.

## The three "win" mechanisms

1. **Composition** — *what* is active changes while aggregate timing is
   untouched.
2. **Timeline localization** — *where / when* the error occurs, even when the
   aggregate cycle-time distribution is unchanged.
3. **Pairing** — a shift visible only when comparing each case's continuation to
   its own ground-truth path, invisible to any marginal distribution.

## Final scenario set

| # | Scenario | Family | Mechanism | Detected by state | Blind baselines | Status |
|---|----------|--------|-----------|-------------------|-----------------|--------|
| 2 | **Parallel-branch automation** | `parallel_auto` | composition / timeline | activity, case, cardinality, role (c-index 1.0) | cycle_time, ngd, red, rtd | ✅ built & validated — *"very good"* |
| 3 | **Model-error routing** (was "XOR redistribution") | `route_error` *(new)* | pairing | activity_case, activity_type | cycle_time, ngd_n2, (red if branches balanced) | 🔨 **to build** — replaces old #3 |
| 4 | **Compensating front/back XOR** | `front_back_swap` *(new)* | **timeline** | activity/case/activity_case (0.94–0.96) | cycle_time, rtd, ngd, **and red** (all ~0.5) | ✅ **built & validated** — beats RED |
| 1 | **Single-activity rename** | `relabel` | composition | activity, activity_case | cycle_time, ngd, red, rtd | ✅ built — *keep, needs justification* |
| 7 | **Per-case timeline jitter** | `rephase` | timeline | all projections | ctd, ngd, red (exactly 0) | ✅ built |

**Removed / de-featured** (see bottom of file): old **#3 XOR redistribution**
(`mix_ratio`/`gateway`) — ngd catches it, so replaced by model-error routing;
**#6 temporal case-type drift** — Prosimos has no time-dependent case attribute
and it adds little for the ongoing-case scope; **#5 asymmetric case-type as a
log tag-swap** — folded into #3's model-error framing (invert routing in the
model instead of swapping tags in the log).

## Scenarios explained

### 2 · Parallel-branch automation — `parallel_auto`  *(David: "very good")*
- **Process:** an AND-split runs two things at once — a long **Critical** task
  (10 min) and a short **Non-critical** chain (3×2 min) on a *separate* team —
  then they join. The critical task alone sets how long the case takes.
- **Change (in the model):** "automate" the non-critical branch — shrink its
  durations toward ~0 (a robot now does that paperwork instantly).
- **Kept identical:** total cycle time (critical path unmoved); the non-critical
  activity still *appears* once per case (sequence/bigrams unchanged).
- **Why baselines miss it:** cycle_time unchanged; ngd sees the same sequence.
- **Why state catches it:** that non-critical task used to *occupy* the active
  set for 6 min/case and now occupies ~0 → at any instant less is genuinely
  happening in parallel.
- **Status:** built (`tools/generate_parallel_auto.py`,
  `build_branch_automation_params`); state c-index 1.0, baselines 0.5.

### 3 · Model-error routing — `route_error` *(new — replaces XOR redistribution)*
The meeting example. Cases carry a real `case_type` ∈ {red, blue} attribute that
**drives routing** through a chain of 2–3 XOR splits separated by common
activities. The **branches are length-asymmetric** (one path several activities
/ longer, the other 1–2 / shorter) so a wrong routing has a large per-case
impact — but the population is 50/50, so the *aggregate* cycle time compensates.
- **Ground truth:** the *correct* model (red → its branch, blue → its branch).
- **Change (in the model):** **invert** the routing rules (red → the other
  branch, blue → the other branch) for a graded fraction of the gateways, and
  run Prosimos-short-term to continue the ongoing cases with this wrong model.
  `level=0` is the correct model.
- **Kept identical:** red/blue overall ratio (50/50), each gateway's path
  marginal (50/50), activity counts, and — because branch totals are balanced —
  aggregate cycle time.
- **Why baselines miss it:** no marginal moved; the common separators keep the
  2-gram histogram unchanged (`ngd_n2` blind); it is a long-term dependency the
  n-gram cannot see; 50/50 compensation keeps `cycle_time` flat.
- **Why state catches it:** pairing by `case_id` sees that each ongoing case is
  no longer following its ground-truth path — the plain `(case_id, activity)`
  (`activity_case`) projection already fires; `(activity, case_type)`
  (`activity_type`) fires too and shows the attribute pickup.
- **Note (David):** `(case_id, activity)` alone suffices — we do *not* need the
  `case_type` tuple to detect it, because the case demonstrably leaves its
  ground-truth path. Modifying the **model** (not the log) is the clean framing.

### 4 · Compensating front/back XOR — `front_back_swap` *(built & validated)*
- **Process:** an XOR with **two branches of equal total duration** — one
  **front-loaded** (early steps long, late short) and one **back-loaded** (the
  mirror). 50/50 population. (`tools/generate_front_back_xor.py`.)
- **Change (in the model):** **swap** the two loadings (the front-loaded branch
  becomes back-loaded and vice-versa) via `build_front_back_swap_params`; `level`
  = percent swapped (0 = GT, 100 = full swap). Each branch's total duration is
  held exactly invariant at every level.
- **Kept identical:** per-case total duration → cycle time; the activity
  sequence; **and the aggregate relative-event-distribution** — because the two
  branches compensate each other across the population.
- **Why baselines miss it — including RED:** cycle_time and ngd see "same total,
  same steps"; and because the front/back branches compensate in aggregate,
  **`red` is blind too**. (This is the key upgrade over the old single-chain
  front/back-load, which RED *would* detect.)
- **Why state catches it:** *where in time the work sits* changed per case — the
  per-instant active-set view localises it. The "where is the error" story.
- **Validated result** (Scope A, 5 runs, levels 0/25/50/75/100): every baseline
  blind — `cycle_time` / `red` / `rtd` c-index **0.500** (distances exactly 0 at
  every level), `ngd_n2` 0.55; state detects — `activity` 0.956, `case` 0.952,
  `activity_case` 0.936. The cleanest shutout in the set: **all four baselines,
  RED included, are blind while state localises the per-case timeline shift.**

### 1 · Single-activity rename — `relabel`  *(David: keep, needs justification)*
- **Change:** rename a fraction of one activity's occurrences to a new label;
  nothing else moves — every timestamp, case, resource, duration byte-identical.
- **Why state catches it:** the set of *what's active* holds the new label; a
  pure composition shift. Simplest composition proof.
- **David's note:** just a composition change — include only if we can justify a
  realistic "process change without affecting timing." Keep as a warm-up, not a
  headline.

### 7 · Per-case timeline jitter — `rephase`
- **Change:** slide each case's *entire* timeline earlier/later by a small random
  offset — the case is internally unchanged (same durations, gaps, order).
- **Kept identical (exactly 0 by construction):** per-case cycle time, activity
  sequence, relative event positions → ctd / ngd / red are exactly 0.
- **Why state catches it:** across cases the overlap pattern changes — at each
  instant a different set of cases is active. state c-index 0.94–1.0.

## Removed / de-featured scenarios

- **Old #3 · XOR redistribution** (`mix_ratio` / `gateway`) — shifting branch
  proportions (50→70%) changes the activity-sequence histogram, so **`ngd`
  detects it** (~0.97 vs state ~0.84). Replaced by the model-error routing
  scenario above. Code retained only as an honest "baselines also do well
  sometimes" data point; not featured.
- **#5 · Asymmetric case-type as a post-hoc tag swap** (`label_swap` /
  log-mutating `case_route`) — David: modify the **model** instead (invert the
  routing). Folded into #3 (`route_error`). The old log-swap oracles stay in the
  code as provenance but are not the featured form.
- **#6 · Temporal case-type drift** (`case_type_drift`) — Prosimos has **no
  time-dependent case attribute** (a probability is fixed for the whole run), so
  the drift can only be faked as a post-hoc log transform, which contradicts the
  model-error methodology; and for the ongoing-case scope it adds little over
  #3. Removed from the featured set.
- **#7 · Activity time-shift** (`calendar_shift`) — not cycle-time-neutral
  (desynchronises hand-offs), so `cycle_time` detects it; no state advantage.
  Already flagged cautionary; kept only to illustrate a baseline's strength.

## Suggested demonstration set

Tight, non-redundant, one per mechanism:

- **#2** parallel-branch automation — structural **composition**.
- **#3** model-error routing — **pairing** (the hypothesis, right on point).
- **#4** compensating front/back XOR — **timeline localization** (beats RED).
- **#1** / **#7** as supporting composition / timeline warm-ups.

---

## Build plan — remaining work

### P1 · `route_error` (scenario #3) — the headline, to build

- **Dataset:** new `tools/generate_route_error.py` — initial `case_type`-driven
  XOR chain (2–3 splits) separated by common activities; **length-asymmetric**
  branches with **balanced totals**; distribution-based (not `fix`) durations.
- **Perturbation:** `build_route_error_params(base, *, level, out)` in
  `perturb.py` — inverts the branch rules (red↔blue) for a graded fraction of
  gateways. `level=0` = correct model.
- **Oracle:** `_run_route_error_levels` in `pipeline.py` — GT = short-term sim on
  the correct model; sim = short-term sim on the inverted model from the **same
  prefix/snapshot** (ongoing `case_type` restored via `src/attributes.py`, proven
  by the case_route e2e). **Both sides real sims — no log mutation.**
- **Expected:** `activity_case` + `activity_type` detect; `cycle_time` / `ngd_n2`
  ≈ 0.5; `red` blind if branch totals balanced.
- **Open question:** per-color 90/10 probabilistic routing (David's realism
  enhancement) may exceed Prosimos `branch_rules` (deterministic conditions).
  Ship deterministic red→A / blue→B first; verify 90/10 capability separately.

### P2 · `front_back_swap` (scenario #4) — ✅ DELIVERED

- `tools/generate_front_back_xor.py` → `synthetic_front_back_xor.{bpmn,json}`:
  a 50/50 XOR into an equal-total front-loaded (P) and back-loaded (Q) branch.
- `perturb.build_front_back_swap_params`: morphs each branch toward the other's
  profile (`level` = percent swapped), holding each branch total exactly fixed.
- Wired as a simulator-driven param family (`--perturbation front_back_swap`),
  `DatasetSpec synthetic_front_back_xor`, unit tests.
- **Result:** all baselines (cycle_time/red/rtd/ngd) ~0.5; state 0.94–0.96.
  RED confirmed blind — the compensating design defeats it.

### Not doing

- **JSD distance** — decision stands: Jaccard + cardinality separate cleanly on
  these synthetics.
- **Real-life logs** — not required if the proofs + synthetics land.
- **#6 temporal drift** — removed (Prosimos limitation; low value for ongoing
  scope).
