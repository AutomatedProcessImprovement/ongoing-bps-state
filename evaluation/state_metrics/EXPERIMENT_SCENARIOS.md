# State-metrics experiment scenarios — table + descriptions

Standalone reference for the short-term-simulation state-metrics paper: the
final experiment table and a plain-English description of every scenario, after
David's review. For design rationale, build plans, and implementation status see
the companion `PAPER_SCENARIOS.md`.

**Goal:** show that the proposed per-instant, per-case state metric **catches
continuation errors the SOTA long-term metrics miss because those errors even
out in the aggregate.**

**Baselines referenced:** cycle-time MAE (`cycle_time`), N-gram distance
(`ngd_n2`), relative event distribution (`red`), remaining-time distribution
(`rtd`) — the Arik et al. log-distance measures, designed for long-term fidelity.

**Methodology:** apply the modification **in the BPS model** (not the log).
Generate the ground truth with the *original* model; continue the ongoing cases
with the *modified (wrong)* model via Prosimos-short-term; ask whether the metric
detects the deviation. We **pair** the ongoing cases and evaluate each case's
continuation with tuples `(case_id, …)`, so that when per-case errors even out
the aggregation-only baselines are blind by construction.

**Three "win" mechanisms:**
1. **Composition** — *what* is active changes while aggregate timing is untouched.
2. **Timeline localization** — *where / when* the error occurs.
3. **Pairing** — a shift visible only when comparing each case's continuation to
   its own ground-truth path.

## Table

c-index: 1.0 = perfect monotone detection, 0.5 = blind.

| # | Scenario | Mechanism | Winner / blind baselines | Family | Status |
|---|----------|-----------|--------------------------|--------|--------|
| 2 | **Parallel-branch automation** (automate non-critical AND-branch) | composition / timeline | **State** (activity/case/cardinality/role c-index 1.0); cycle_time/ngd/red/rtd 0.5 | ✅ `parallel_auto` | validated — *"very good"* |
| 3 | **Model-error routing** (invert `case_type` routing in the model) | **pairing** | **State** (`activity_case`, `activity_type`); cycle_time/ngd_n2 0.5, red 0.5 if branches balanced | 🔨 `route_error` *(new)* | to build |
| 4 | **Compensating front/back XOR** (swap front-/back-loaded branches) | **timeline** | **State** (activity/case/activity_case 0.94–0.96); cycle_time/**red**/rtd/ngd all ~0.5 | ✅ `front_back_swap` | validated — beats RED |
| 1 | **Single-activity rename** (pure composition) | composition | **State** (activity/activity_case); cycle_time/ngd/red/rtd 0.5 | ✅ `relabel` | keep — needs justification |
| 7 | **Per-case timeline jitter** (slide each case's whole timeline) | timeline | **State**; ctd/ngd/red exactly 0 | ✅ `rephase` | validated |

**Removed / de-featured:** old #3 XOR redistribution (`mix_ratio`/`gateway`, ngd
wins — replaced by model-error routing); #5 asymmetric case-type as a log
tag-swap (`label_swap` — folded into #3's model-error form); #6 temporal
case-type drift (`case_type_drift` — Prosimos has no time-dependent case
attribute); #7-old activity time-shift (`calendar_shift` — changes cycle time,
no state edge). Code retained for provenance; not featured.

## Scenario descriptions

### 2 · Parallel-branch automation — `parallel_auto`  *(David: "very good")*
- **Process:** an AND-split runs a long **Critical** task (10 min) and a short
  **Non-critical** chain (3×2 min) on a *separate* team, then they join. The
  critical task alone sets the cycle time.
- **Change (model):** "automate" the non-critical branch — shrink its durations
  toward ~0.
- **Kept identical:** total cycle time (critical path unmoved); the non-critical
  activity still appears once per case (sequence/bigrams unchanged).
- **Why baselines miss it:** cycle_time unchanged; ngd sees the same sequence.
- **Why state catches it:** that task used to occupy the active set for 6 min/case
  and now occupies ~0 → at any instant less is genuinely happening in parallel.

### 3 · Model-error routing — `route_error`  *(replaces XOR redistribution)*
- **Process:** cases carry a real `case_type` ∈ {red, blue} attribute driving a
  chain of 2–3 XOR splits separated by common activities. Branches are
  **length-asymmetric** (one path longer) but branch totals are balanced, so a
  50/50 population is cycle-time-neutral.
- **Change (model):** **invert** the routing rules (red ↔ blue) for a graded
  fraction of gateways; continue the ongoing cases with this wrong model.
- **Kept identical:** red/blue ratio (50/50), each gateway's path marginal
  (50/50), activity counts, aggregate cycle time.
- **Why baselines miss it:** no marginal moved; common separators keep the 2-gram
  histogram unchanged (`ngd_n2` blind, and it is a long-term dependency ngd can't
  see); 50/50 compensation keeps cycle_time flat.
- **Why state catches it:** pairing by `case_id` sees each ongoing case leave its
  ground-truth path — plain `(case_id, activity)` (`activity_case`) fires;
  `(activity, case_type)` (`activity_type`) fires too. *(David: `(case_id,
  activity)` alone suffices; modifying the model, not the log, is the clean
  framing.)*

### 4 · Compensating front/back XOR — `front_back_swap`  *(validated)*
- **Process:** an XOR with **two branches of equal total duration** — one
  front-loaded (P), one back-loaded (Q). 50/50 population.
- **Change (model):** **swap** the two loadings (`level` = percent swapped, each
  branch total held exactly fixed). Each case's total duration is unchanged.
- **Kept identical:** per-case total → cycle time; the activity sequence; **and
  the aggregate relative-event-distribution** (the branches compensate).
- **Why baselines miss it — including RED:** same total/steps for cycle_time and
  ngd; and because front/back compensate in aggregate, **`red` is blind too**
  (the key upgrade over the old single-chain version, which RED would detect).
- **Why state catches it:** *where in time the work sits* changed per case — the
  per-instant active-set view localises it. The "where is the error" story.
- **Result** (Scope A, 5 runs): cycle_time/red/rtd c-index **0.500** (exactly 0
  every level), ngd 0.55; state activity 0.956 / case 0.952 / activity_case
  0.936. All four baselines — RED included — blind.

### 1 · Single-activity rename — `relabel`  *(keep — needs justification)*
- **Change:** rename a fraction of one activity's occurrences; every timestamp,
  case, resource, duration byte-identical.
- **Why state catches it:** the set of *what's active* holds the new label — a
  pure composition shift. *(David: just a composition change; include only with a
  realistic justification. Warm-up, not headline.)*

### 7 · Per-case timeline jitter — `rephase`
- **Change:** slide each case's *entire* timeline earlier/later by a small random
  offset — internally unchanged (same durations, gaps, order).
- **Kept identical (exactly 0 by construction):** per-case cycle time, activity
  sequence, relative event positions → ctd / ngd / red exactly 0.
- **Why state catches it:** across cases the overlap pattern changes — at each
  instant a different set of cases is active. state c-index 0.94–1.0.
