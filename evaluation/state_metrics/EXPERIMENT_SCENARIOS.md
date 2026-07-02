# State-metrics experiment scenarios — table + descriptions

Standalone reference for the short-term-simulation state-metrics paper: the
candidate experiment table and a plain-English description of every scenario.
For design rationale, build plans, and implementation status see the companion
`PAPER_SCENARIOS.md`.

**Goal:** show that the proposed per-instant state metric **replicates what the
baselines detect and also covers cases they miss** (complementary, not
adversarial).

**Baselines referenced:** cycle-time MAE (`cycle_time`), N-gram distance
(`ngd_n2`), relative event distribution (`red`), remaining-time distribution
(`rtd`).

**Design principle:** hold every *aggregate / marginal* summary fixed (cycle
time, resource utilization ρ, WIP/λ, overall ratios, activity counts) so the
baselines are blind **by construction**, and change only *what is active*,
*where in time*, or *which type is paired with which path* — which is exactly
what the per-instant state metric measures.

**Three "win" mechanisms:**
1. **Composition** — *what* is active changes (activity mix / labels) while
   aggregate timing is untouched.
2. **Timeline localization** — *where / when* the error occurs, even when the
   aggregate cycle-time distribution is unchanged.
3. **Pairing** — case-type-aware: a shift visible only when comparing *paired*
   sub-populations, invisible to any marginal distribution.

## Table

`Attr?`: **native** = real simulator case attribute · **labels** = post-hoc
red/green tags · **no** = none. c-index: 1.0 = perfect monotone detection,
0.5 = blind.

| # | Scenario | Attr? | Invariants held | Mechanism | Winner / blind baselines | Family | Evidence |
|---|----------|-------|-----------------|-----------|--------------------------|--------|----------|
| 1 | **Single-activity rename** (pure composition) | no | cycle time, ρ, WIP, *timeline identical* | composition | **State** (`activity`, `activity_case`) wins; `cycle_time`/`ngd`/`red`/`rtd` blind | ✅ `relabel` | **Strong** — cycle_time exactly 0; activity c-index up to 1.0 |
| 2 | **Parallel-branch automation** (automate non-critical path; critical path sets cycle time) | no | cycle time, ρ, WIP | composition / timeline | **State** wins; `cycle_time` blind | ✅ `parallel_auto` (sim-driven) | **Strong** — state (activity/case/cardinality/role) c-index 1.0; ngd/red/cycle_time/rtd all 0.5 |
| 3 | **XOR redistribution**, equal-duration branches (shift gateway probs / mix ratio) | labels | mean cycle time, ρ, WIP | composition | mixed — **ngd competitive / wins** | ⚠️ `mix_ratio`, `gateway` | **Cautionary** — ngd ~0.97 most monotonic, state ~0.84 (state *loses*) |
| 4 | **Front-load vs back-load** durations (+ swap resource capacity to keep ρ symmetric) | no | aggregate cycle time, ρ, WIP | **timeline / "where"** | **State (time-weighted)** wins; `cycle_time`, `ngd` blind | ✅ `front_back_load` (sim-driven) | **Strong** — state c-index 1.0; cycle_time/ngd/red/rtd all 0.5 (per-case total held exactly invariant) |
| 5 | **Asymmetric case-type** (red/blue), distribution unchanged, pairing reveals it | **native** | cycle time **distribution**, ρ, WIP | **pairing** | **State** (`case_type`, `activity_type`) wins; **all** baselines ~0.5 (blind) | ✅ `case_route` (sim-driven) + `label_swap` (post-hoc) | **Strongest** — activity_type c-index ≈ 0.92–0.94; every baseline 0.5 |
| 6 | **Temporal case-type drift** (case-type mix drifts along the arrival timeline) | **native** | cycle time, ρ, WIP, **global marginal** | **pairing + timeline** | **State** (`case_type` AND `activity_type`); all baselines blind | ✅ `case_type_drift` (sim-driven, synthetic controlled form) | **Strong** — `case_type`/`activity_type` c-index 1.0; every baseline AND type-agnostic state projection 0.5 |
| 6r | *(open)* **Real-life** temporal case-type shift (same mechanism, real log) | **native** | — (observational) | **pairing** | **State** (`case_type`/`activity_type`); risk: Jaccard ~0.9 floor on real logs | ❌ not built | data-discovery task; `case_type_drift` is its controlled synthetic stand-in. Optional "icing" |
| 7 | **Activity time-shift** (~30 min start/end shift) | no | composition, ρ, WIP | timeline | **`cycle_time` wins**; `ngd`/`red` blind — *state has no edge* | ✅ `calendar_shift` | **Cautionary** — cycle_time detects; dead-end for a state-advantage claim |
| — | *(bonus)* **Per-case timeline jitter** | no | per-case duration, gaps, composition | timeline | **State** wins; `ctd`/`ngd`/`red` exactly 0 | ✅ `rephase` | **Strong** — state c-index 0.94–1.0 |

**Status summary:**
- ✅ **Ready state-wins:** #1, #2, #4, #5, #6, bonus
- ⚠️ **Cautionary** (baseline wins / no edge — keep only for the "complementary" framing): #3, #7
- ❌ **Not built** (optional): #6r (real-log)

## Scenario descriptions

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
