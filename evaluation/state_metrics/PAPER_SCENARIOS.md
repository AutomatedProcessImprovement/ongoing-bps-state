# Paper experiment scenarios — where state metrics win

Candidate experiments for the short-term-simulation state-metrics paper, drawn
from the project meeting notes and cross-referenced with what is currently
implemented in this repo. The goal is to show, on controlled synthetic data,
that the proposed state-based metric **replicates what baselines detect and also
covers cases they miss** (complementary, not adversarial).

Baselines referenced: cycle-time MAE (`cycle_time`), N-gram distance (`ngd_n2`),
relative event distribution (`red`), remaining-time distribution (`rtd`).

## Design principle (from the meetings)

> Hold **cycle time**, **resource utilization (ρ)**, and **WIP/λ** constant, so
> the baselines see "no difference" while the state metric does.

Every scenario below is constructed to keep one or more of these invariants
fixed, isolating the effect that only the state metric is meant to catch.

## The three "win" mechanisms

The metric's advantage reduces to one of three mechanisms; a good paper set
covers all three with non-redundant examples:

1. **Composition** — *what* is active changes (activity mix / labels) while
   aggregate timing is untouched.
2. **Timeline localization** — *where / when* the error occurs, even when the
   aggregate cycle-time distribution is unchanged.
3. **Pairing** — case-type-aware: a shift visible only when comparing *paired*
   sub-populations, invisible to any marginal distribution.

## Candidate experiments

`Attr?` = does the scenario use case attributes? **native** = real simulator
case attribute; **labels** = artificial post-hoc red/green tags; **no** = none.

| # | Scenario | Attr? | Invariants held | Mechanism | Expected winner / blind baselines | Implemented? | Evidence so far |
|---|----------|-------|-----------------|-----------|-----------------------------------|--------------|-----------------|
| 1 | **Single-activity rename** (pure composition) | no | cycle time, ρ, WIP, *timeline identical* | composition | **State** (`activity`, `activity_case`) wins; `cycle_time`/`ngd`/`red`/`rtd` blind | ✅ `relabel` | **Strong** — cycle_time exactly 0; activity c-index up to 1.0 |
| 2 | **Parallel-branch automation** (automate non-critical path; critical path sets cycle time) | no | cycle time, ρ, WIP | composition / timeline | **State** wins; `cycle_time` blind | ✅ `parallel_auto` (sim-driven) | **Strong** — state (activity/case/cardinality/role) c-index 1.0; ngd/red/cycle_time/rtd all 0.5 |
| 3 | **XOR redistribution**, equal-duration branches (shift gateway probs / mix ratio) | labels | mean cycle time, ρ, WIP | composition | mixed — **ngd competitive / wins** | ⚠️ `mix_ratio`, `gateway` | **Cautionary** — ngd ~0.97 most monotonic, state ~0.84 (state *loses*) |
| 4 | **Front-load vs back-load** durations (+ swap resource capacity to keep ρ symmetric) | no | aggregate cycle time, ρ, WIP | **timeline / "where"** | **State (time-weighted)** wins; `cycle_time`, `ngd` blind | ✅ `front_back_load` (sim-driven) | **Strong** — state c-index 1.0; cycle_time/ngd/red/rtd all 0.5 (per-case total held exactly invariant) |
| 5 | **Asymmetric case-type** (red/blue), distribution unchanged, pairing reveals it | **native** | cycle time **distribution**, ρ, WIP | **pairing** | **State** (`case_type`, `activity_type`) wins; **all** baselines ~0.5 (blind) | ✅ `case_route` (sim-driven) + `label_swap` (post-hoc) | **Strongest** — activity_type c-index ≈ 0.92–0.94; every baseline 0.5 |
| 6 | **Temporal case-type drift** (case-type mix drifts along the arrival timeline) | **native** | cycle time, ρ, WIP, **global marginal** | **pairing + timeline** | **State** (`case_type` AND `activity_type`); all baselines blind | ✅ `case_type_drift` (sim-driven, synthetic controlled form) | **Strong** — `case_type`/`activity_type` c-index 1.0; every baseline AND type-agnostic state projection 0.5 |
| 6r | *(open)* **Real-life** temporal case-type shift (same mechanism, real log) | **native** | — (observational) | **pairing** | **State** (`case_type`/`activity_type`); risk: Jaccard ~0.9 floor on real logs | ❌ not built | data-discovery task; `case_type_drift` is its controlled synthetic stand-in. Optional "icing" |
| 7 | **Activity time-shift** (~30 min start/end shift) | no | composition, ρ, WIP | timeline | **`cycle_time` wins**; `ngd`/`red` blind — *state has no edge* | ✅ `calendar_shift` | **Cautionary** — cycle_time detects; dead-end for a state-advantage claim |
| — | *(bonus)* **Per-case timeline jitter** | no | per-case duration, gaps, composition | timeline | **State** wins; `ctd`/`ngd`/`red` exactly 0 | ✅ `rephase` | **Strong** — state c-index 0.94–1.0 |

## Scenarios explained (plain English)

Each scenario is built the same way: hold every *aggregate / marginal* summary
fixed (cycle time, utilization, overall ratios, activity counts) so the baselines
are blind **by construction**, and change only *what is active*, *where in time*,
or *which type is paired with which path* — which is exactly what the per-instant
state metric measures.

### 1 · Single-activity rename — `relabel`
- **Process:** any synthetic log.
- **Change:** rename a fraction of one activity's occurrences to a new label
  (e.g. "Review" → "Review_alt"); nothing else moves.
- **Kept identical:** every timestamp, case, resource, duration — the *timing*
  is byte-for-byte the same, only some event *names* change.
- **Why baselines miss it:** cycle_time / red / rtd only look at timing → zero
  difference.
- **Why state catches it:** the set of *what's active* at each moment now holds
  "Review_alt" instead of "Review" — a pure composition shift. *(Simplest
  composition proof.)*

### 2 · Parallel-branch automation — `parallel_auto`
- **Process:** an AND-split runs two things at once — a long **Critical** task
  (10 min) and a short **Non-critical** chain (3×2 min) on a *separate* team —
  then they join. The critical task alone sets how long the case takes.
- **Change:** "automate" the non-critical branch — shrink its durations toward
  ~0 (a robot now does that paperwork instantly).
- **Kept identical:** total cycle time (critical path unmoved), and the
  non-critical activity still *appears* once per case (sequence/bigrams
  unchanged).
- **Why baselines miss it:** cycle_time unchanged; ngd sees the same activity
  sequence.
- **Why state catches it:** that non-critical task used to *occupy* the active
  set for 6 min/case and now occupies ~0 → at any instant less is genuinely
  happening in parallel. *(Classic BPR "automate the side-process" example.)*

### 3 · XOR redistribution — `mix_ratio` / `gateway`  ⚠️
- **Process:** a case takes one of two equally-long branches at a fork (or a
  population mixes two case types).
- **Change:** shift the proportions — e.g. 70% take branch A instead of 50%.
- **Kept identical:** mean cycle time (branches equal-duration), ρ, WIP.
- **Result (cautionary):** the baseline **`ngd` actually wins** (~0.97 vs state
  ~0.84), because changing branch proportions changes the activity-sequence
  (bigram) histogram — exactly what ngd detects. Keep only as an honest
  "baselines also do well sometimes" point; **not** a state win.

### 4 · Front-load vs back-load — `front_back_load`
- **Process:** a simple 5-step sequence (Step 1→…→Step 5), all steps equal
  length.
- **Change:** redistribute the *time* across steps — front-loading makes early
  steps long and late steps short (back-loading is the mirror) — **while
  keeping each case's total duration exactly the same.**
- **Kept identical:** total cycle time (held to the microsecond by
  renormalization), ρ, the activity sequence.
- **Why baselines miss it:** cycle_time and ngd see "same total, same steps."
- **Why state catches it:** *where in time the work sits* changed — early there's
  now more concurrent work, later less. The per-instant state view is the only
  metric that localizes it. *(The "where is the error" / timeline story — most
  novel.)*

### 5 · Asymmetric case-type — `case_route` / `label_swap`  *(strongest)*
- **Process:** cases carry a real red/blue attribute that **drives routing**
  (red → branch A, blue → branch B), but both branches are equally long.
- **Change:** swap the red/blue tags on a fraction of cases (so some "red" cases
  now sit on the B-branch, etc.).
- **Kept identical:** the red/blue *overall ratio* (50/50), all timing, activity
  counts — every single-variable summary is unchanged.
- **Why baselines (and even simple state views) miss it:** no marginal moved —
  red still 50%, branch-A still 50%, durations identical. The change is *only in
  the pairing*: which type is on which branch.
- **Why state catches it:** the joint `(activity, case_type)` view sees
  "branch-A is now run by blue cases" — the only metric comparing *paired*
  sub-populations. *(All baselines 0.5; state ≈0.93. The unique contribution.)*

### 6 · Temporal case-type drift — `case_type_drift`  *(the data-attribute one)*
- **Process:** same red/blue attribute-routed process as #5.
- **Change:** instead of random swaps, make the red/blue mix **drift over time**
  (early arrivals skew blue, later ones skew red) — **while keeping the overall
  50/50 ratio exactly fixed.**
- **Kept identical:** global red/blue ratio, all timing, activity counts.
- **Why baselines miss it:** overall composition is still 50/50 and the event
  stream is untouched.
- **Why state catches it — and how it differs from #5:** because the mix varies
  *along the timeline*, at any window more of one color is active than baseline →
  **both** the simple `case_type` view *and* the paired `activity_type` view fire
  (in #5 only `activity_type` fired). That extra `case_type` signal is the
  fingerprint of *temporal* drift.

### 6r · Real-life temporal case-type shift  *(open / optional)*
- Same mechanism as #6, but on a **real** log whose case-type mix genuinely
  drifts over months. Observational (no controlled invariant), gated on finding
  such a log and on Jaccard not hitting its ~0.9 real-log floor. `case_type_drift`
  is its controlled synthetic stand-in; left as optional data-discovery.

### 7 · Activity time-shift — `calendar_shift`  ⚠️
- **Process:** a business-hours process.
- **Change:** shift one team's working hours by ~30 min–N hours.
- **Result (cautionary):** this is *not* cycle-time-neutral — shifting hours
  desynchronizes hand-offs and **changes cycle time**, so `cycle_time` detects it
  and **state has no special edge**. Kept only to illustrate a baseline's
  strength.

### — · Per-case timeline jitter *(bonus)* — `rephase`
- **Process:** any synthetic.
- **Change:** slide each case's *entire* timeline earlier/later by a small random
  offset — the case is internally unchanged (same durations, gaps, order).
- **Kept identical (exactly zero by construction):** per-case cycle time, the
  activity sequence, relative event positions → ctd / ngd / red are *exactly* 0.
- **Why state catches it:** even with each case internally untouched, *across*
  cases the overlap pattern changes — at each instant a different set of cases is
  active. Only the per-instant state view sees it. (state c-index 0.94–1.0)

## Attribute-related subset

- **#5 Asymmetric case-type** — the core, using **native** simulator case
  attributes. Pairing pillar; strongest evidence.
- **#6 Temporal case-type drift** (`case_type_drift`) — **built**. A controlled
  synthetic drift of the native `case_type` over the arrival timeline (global
  marginal held fixed) on a genuinely attribute-routed short-term sim. Detected
  by **both** `case_type` and `activity_type` (c-index 1.0) — the extra
  `case_type` hit over `label_swap` is the temporal-drift signature. The
  real-log form (#6r) remains optional data-discovery; this is its stand-in.
- **#3 XOR redistribution (`mix_ratio`)** — only *attribute-adjacent*: it tags
  cases red/green with **artificial post-hoc labels**, not native attributes.

Everything else (#1, #2, #4, #7, bonus) is attribute-free.

## Cautionary / negative cases (baselines win — keep for the "complementary" framing only)

- **#3** — `ngd` is most monotonic; do not feature it as a state win.
- **#7** — `cycle_time` detects the shift; state has no edge. Useful only to
  show a baseline's strength, not ours.

## Reading the table

**Two cleanest pillars — already implemented and validated:**

- **Composition pillar** → #1 `relabel` (baselines literally blind).
- **Pairing pillar** → #5 `case_route` / `label_swap` (the headline: all
  baselines at 0.5, state ~0.94). The unique contribution to protect.

**Avoid leading with #3** — our own runs show `ngd` *beats* state there, so it is
a weak/negative case to feature. Keep it only as an honest "baselines also do
well here" point supporting the complementary framing.

**Two highest-value gaps — now built and validated** — each cycle-time-neutral,
demonstrating a *different* mechanism than the pillars:

- **#4 front/back-loading** (`front_back_load`) — the candidate showcasing
  **timeline localization** ("where the error is"). No baseline captures it
  (all 0.5); state c-index 1.0. Per-case total duration held exactly invariant.
- **#2 parallel-branch automation** (`parallel_auto`) — a structural composition
  change baselines can't see; reviewer-friendly (maps to BPR literature, e.g.
  the Ford A/P redesign cited in the notes). State c-index 1.0, baselines 0.5.

## Suggested demonstration set

A tight, non-redundant set covering all three mechanisms:

- **#1** (composition) + **#4** (timeline) + **#5** (pairing) — the core.
- **#2** as a second, *structural* (not just labeling) composition case.
- **#3** only as an honest "baselines also do well here" data point.

## Implementation status note

A constraint discovered while implementing #5: Prosimos can attribute-condition
**routing** but **not durations**. So the realizable form of "asymmetric
case-type" is duration-symmetric attribute-driven *routing* (`case_route`), not
the red-slower / blue-faster duration design from the notes — but the scenario's
*purpose* (pairing reveals what aggregates hide) is unchanged.

See `evaluation/state_metrics/PERTURBATIONS.md` for the full perturbation
catalogue and `tests/e2e/` for the end-to-end attribute/routing verification.

---

# Build plans — scenarios #2, #4, #6  ✅ DELIVERED

> **Status (2026-06-28): all three are implemented, unit-tested, and validated
> end-to-end with real Prosimos.** Smoke runs (2 replicates, p90_wip cutoff)
> reproduce the intended result in every case — state metrics rank the
> perturbation monotonically (c-index 1.0) while the baselines stay at 0.5:
>
> | Scenario | Family | Dataset | State (c-index 1.0) | Baselines |
> |---|---|---|---|---|
> | #2 | `parallel_auto` | `synthetic_parallel_auto` | activity, case, cardinality, activity_role | ngd/red/cycle_time/rtd = 0.5 |
> | #4 | `front_back_load` | `synthetic_linear_chain` | all projections | ngd/red/cycle_time/rtd = 0.5 |
> | #6 | `case_type_drift` | `synthetic_case_route` | case_type, activity_type | ngd/red/cycle_time/rtd + type-agnostic state = 0.5 |
>
> New assets: generators `tools/generate_parallel_auto.py` and
> `tools/generate_linear_chain.py`; builders `build_branch_automation_params` /
> `build_front_back_load_params` in `perturb.py`; the `drift_case_types`
> transform + `_run_case_type_drift_levels` oracle in `pipeline.py`; dataset
> specs + CLI choices (`--perturbation parallel_auto|front_back_load|case_type_drift`,
> `--load-direction`). Unit tests in `tests/evaluation/state_metrics/`.
>
> **JSD/Jaccard:** explicitly *not* pursued (per decision 2026-06-28 — we keep
> Jaccard). The scenarios above are designed so the existing Jaccard +
> cardinality distances already separate cleanly on synthetic data.

The original build notes are kept below for design provenance.

How the machinery works (so the plans below slot in):

- **Simulator-driven param families** (`resources`, `duration`, `gateway`, …):
  add a `build_<x>_params(base, *, level…, out)` to `perturb.py`, then a
  dispatch branch in `pipeline._prepare_params_for_level` (the
  `if cfg.perturbation == …` chain). The standard prefix → K short-term re-sims
  → distances loop then runs automatically.
- **Oracle families** (`relabel`, `rephase`, `case_route`, `mix_ratio`,
  `label_swap`): a dedicated `_run_<x>_levels(...)` in `pipeline.py`, branched
  early in `run_pipeline`, for post-hoc log transforms or custom sim loops.
- Both need: a `--perturbation` choice in `run_pipeline.py`, any new `cfg`
  fields on `PipelineConfig`, a `DatasetSpec` in `datasets.py`, and tests under
  `tests/evaluation/state_metrics/`.

Recommended order: **#2 → #4 → #6** (increasing effort and decreasing
certainty). The planned **JSD distance** (see PERTURBATIONS / strategy notes)
helps #4 and #6 by avoiding Jaccard's ~0.9 real-log floor; consider it a
prerequisite for those two.

## Plan — #2 Parallel-branch automation

**Claim:** automating the *non-critical* branch of a parallel (AND) block leaves
cycle time (set by the critical path) and ρ on the critical resource unchanged,
but the non-critical activity stops occupying the active-instance set →
**state wins, `cycle_time` and `ngd_n2` blind.**

- **Dataset (new):** `tools/generate_parallel_auto.py` (mirror
  `generate_case_route.py`) emitting `synthetic_parallel_auto.{bpmn,json}`:
  `Start → AND-split → { Critical (mean D_c) | NonCritical chain (Σ means = D_nc, with D_nc < D_c) } → AND-join → End`.
  Critical path strictly dominates so cycle time = D_c regardless of the
  non-critical branch. Same resource pool size on both branches, sized so
  removing non-critical work does not change critical-resource ρ.
  - Alternative quick start: reuse `samples/dev-samples/synthetic_and_k5.bpmn`
    (1 parallel gateway) if a clean critical/non-critical split can be
    identified; a purpose-built generator is cleaner for the paper.
- **Perturbation (new):** `build_branch_automation_params(base, *, automate_task_ids, level, out)`
  scaling the non-critical task means by `(1 - level/100)` toward a small floor
  (not exact 0 — keep the event in the trace so `ngd_n2` stays blind). `level=0`
  no-op; `level=100` ≈ instantaneous.
- **Integration:** param family `parallel_auto`; dispatch branch in
  `_prepare_params_for_level`; `cfg.automate_task_ids`; choice + `DatasetSpec`.
- **Invariants:** cycle time = D_c (constant while D_nc·(1-level/100) < D_c);
  arrival/λ unchanged; ρ on the critical resource unchanged.
- **Expected result / assertion:** `state` (`activity`, `activity_case`, and the
  time-weighted summary) c-index high and monotonic in `level`; `cycle_time`
  and `ngd_n2` ≈ 0.5. Ranking via existing `ranking.py`.
- **Risks:** (a) Prosimos may drop zero-duration tasks from the log → keep a
  duration floor so the label still appears (otherwise `ngd` would also move and
  the "baseline blind" claim breaks). (b) If automation frees a resource, ρ can
  shift — keep the non-critical pool separate / small so the critical ρ is
  untouched. Verify ρ with the existing `_compute_utilization_rows`.
- **Effort:** medium (one generator + one builder + dispatch + dataset + test).

## Plan — #4 Front-load vs back-load durations

**Claim:** redistributing duration mass along a sequential chain (early-heavy vs
late-heavy) while holding the per-case total constant keeps aggregate cycle time
and ρ fixed, but moves *where in time* work sits → **state (time-weighted)
localizes it; `cycle_time` and `ngd_n2` blind.** (Honest caveat: `red`/`rtd`
likely co-detect, since event positions move — the differentiator is the
per-instant active-set view, not a marginal.)

- **Dataset (new or reuse):** a linear chain of N comparable-duration activities
  on one pool — generate `synthetic_linear_chain.{bpmn,json}` (N≈5) for a clean,
  symmetric baseline; or reuse an existing sequential synthetic.
- **Perturbation (new):** `build_front_back_load_params(base, *, chain_task_ids, shift, out)`
  reweighting task means by position so `Σ means` per case is invariant. Signed
  `level`: `+` front-loads (early tasks longer), `−` back-loads. E.g. mean_i ·=
  `1 + (level/100)·w_i` with `w_i` antisymmetric about the chain midpoint and
  Σ(w_i·mean_i)=0 (renormalize to hold the total exactly).
- **Optional v2 (the notes' "swap resource capacity"):** also shift pool
  capacity / calendars toward the heavy end so *instantaneous* ρ stays symmetric,
  not just aggregate. Adds complexity; ship v1 (single pool, aggregate ρ held)
  first.
- **Integration:** param family `front_back_load`; signed levels; dispatch
  branch; `cfg.chain_task_ids`; choice + `DatasetSpec` (signed default levels).
- **Invariants:** per-case total duration (→ cycle time) constant; total work
  (→ aggregate ρ) constant; λ unchanged.
- **Expected result / assertion:** `state` time-weighted distance monotonic in
  `|level|` and roughly symmetric for front vs back; `cycle_time`, `ngd_n2` ≈
  0.5; report `red`/`rtd` as partial co-detectors (supports the "complementary"
  framing rather than a clean shutout). Prefer **JSD/cardinality** over Jaccard
  for the summary here.
- **Risks:** deterministic durations may make the short-term re-sims too rigid;
  keep distributional shapes. The active-overlap profile must actually change —
  validate on a dry run before committing levels.
- **Effort:** medium-high (builder reweighting + total-invariance math + dataset;
  v2 capacity-swap is extra).

## Plan — #6 Real-life temporal case-type shift (optional / "icing")

**Claim:** on a real log whose case-type mix drifts over time, the
`case_type`/`activity_type` projections track the composition shift while
`cycle_time`/`ngd_n2` stay flat. Observational, not a controlled perturbation.

- **Not a builder** — this is data discovery + a measurement script.
- **Candidate logs/attributes:** BPIC-2017 (`ApplicationType`, `LoanGoal`),
  BPIC-2012 (`AMOUNT_REQ` bucketed into red/blue), or `P2PFIN` (a categorical
  case field). Need a *binary-izable* case attribute.
- **Step 1 — drift profiling:** bin cases by arrival time into windows; compute
  the case-type marginal per window; quantify drift (total-variation distance
  between windows). Select a log/attribute where drift is real and material.
- **Step 2 — experiment:** pick cutoffs in different drift regimes (or compare a
  high-red vs high-blue window); run the existing short-term pipeline carrying
  the real attribute as the `case_type` column (already supported by
  `_load_prosimos_log` / `_write_prefix_csv`); measure projections vs baselines.
- **Integration:** new `evaluation/state_metrics/realworld_case_type.py` +
  `DatasetSpec` entries mapping the real attribute column → `case_type`.
- **Risks:** (a) Jaccard ~0.9 floor on high-variability real logs → use the
  planned JSD or `cardinality`. (b) A log with genuine temporal case-type drift
  may not exist in our set — this is explicitly "icing on the cake," not
  required for the core hypothesis. (c) No ground-truth control, so framing is
  weaker than the synthetics.
- **Effort:** high uncertainty (data-dependent); do last, only if #2/#4 land.

## Cross-cutting prerequisite

`distances.py` currently has only `jaccard_multiset` + `cardinality`. Adding a
**JSD / distribution distance** (strategy note: prefer JSD over Jaccard; Maksym's
Jaccard-sensitivity concern) should precede #4 and #6, since both risk the
Jaccard real-log floor. Small, self-contained: `distances.py` + wire into
`api.py` summary + `ranking.py` + a test.
