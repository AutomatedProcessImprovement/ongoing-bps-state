"""Build perturbed Prosimos JSON params.

Perturbation families:

Magnitude perturbations (uniform shift of throughput):
- ``build_perturbed_params`` — add/remove resources from a profile.
- ``build_duration_scaled_params`` — scale every activity duration.

Structural perturbations (added 2026-05 — designed to expose state metrics'
WHAT/WHEN advantages where magnitude metrics like cycle_time are blind):
- ``build_role_swap_params`` — re-route N tasks from one role to another.
  Cycle time is approximately preserved (same per-resource distributions
  on a different worker pool); the ``activity_role`` projection sees an
  immediate, persistent change.
- ``build_calendar_shifted_params`` — shift one resource calendar by N hours.
  Throughput unchanged; the *timing* of activity execution within the day
  shifts.
- ``build_gateway_biased_params`` — push one gateway's probabilities toward
  a more skewed distribution. Cycle time mostly preserved; activity mix at
  any instant changes.
- ``build_arrival_burstier_params`` — for a gamma arrival distribution,
  scale variance up while keeping the mean constant. Long-run throughput
  unchanged; the WIP envelope becomes burstier.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path


def build_perturbed_params(
    base_json_path: str | Path,
    *,
    remove_from_profile: str,
    n_to_remove: int,
    out_json_path: str | Path,
) -> dict:
    """Shrink or grow the named profile by ``|n_to_remove|`` resources.

    Sign convention:
      * ``n_to_remove > 0`` — drop the first N resources from the profile
        and prune their references in ``task_resource_distribution``.
      * ``n_to_remove == 0`` — no-op (copies params verbatim).
      * ``n_to_remove < 0`` — clone the last resource in the profile
        ``|n_to_remove|`` times and append, mirroring each clone into every
        ``task_resource_distribution`` entry that referenced the template.
        Clones get unique IDs derived from the template's name (e.g.
        ``Clerk-000010``, ``Clerk-000011``, ...).

    Returns a small manifest with ``profile``, ``removed`` (names of dropped
    resources, empty when adding), ``added`` (names of new clones), and
    ``remaining`` (final resource count in the profile).
    """
    with open(base_json_path, encoding="utf-8") as f:
        params = json.load(f)

    params = copy.deepcopy(params)

    profile = next(
        (p for p in params.get("resource_profiles", []) if p.get("name") == remove_from_profile),
        None,
    )
    if profile is None:
        raise ValueError(f"profile {remove_from_profile!r} not found")
    resources = profile.get("resource_list", [])

    if n_to_remove > 0:
        if n_to_remove >= len(resources):
            # Forbid emptying a profile entirely — Prosimos tasks assigned
            # to this profile would have no runnable resources.
            raise ValueError("refusing to empty the profile; keep at least one resource")
        dropped = resources[:n_to_remove]
        dropped_ids = {r["id"] for r in dropped}
        dropped_names = [r.get("name", r["id"]) for r in dropped]
        profile["resource_list"] = resources[n_to_remove:]
        for task in params.get("task_resource_distribution", []):
            task["resources"] = [
                r for r in task.get("resources", []) if r.get("resource_id") not in dropped_ids
            ]
        added_names: list[str] = []
    elif n_to_remove < 0:
        if not resources:
            raise ValueError(f"profile {remove_from_profile!r} has no template resource to clone")
        n_to_add = -n_to_remove
        template = resources[-1]
        added_names = []
        added_ids_to_template: dict[str, str] = {}
        # Derive a base name like "Clerk" from a template named "Clerk-000009".
        tmpl_name = template.get("name", template["id"])
        base, sep, suffix = tmpl_name.rpartition("-")
        # If the suffix is numeric, continue the numbering; else just append _add_K.
        next_idx = int(suffix) + 1 if (sep and suffix.isdigit()) else len(resources) + 1
        for i in range(n_to_add):
            clone = copy.deepcopy(template)
            new_idx = next_idx + i
            new_name = f"{base}-{new_idx:06d}" if sep else f"{tmpl_name}_add_{i + 1}"
            new_id = new_name
            clone["id"] = new_id
            clone["name"] = new_name
            profile["resource_list"].append(clone)
            added_names.append(new_name)
            added_ids_to_template[new_id] = template["id"]
        # Mirror task distributions: every task that referenced the template
        # gets a parallel entry for each clone using the same distribution.
        template_id = template["id"]
        for task in params.get("task_resource_distribution", []):
            template_entry = next(
                (r for r in task.get("resources", []) if r.get("resource_id") == template_id),
                None,
            )
            if template_entry is None:
                continue
            for new_id in added_ids_to_template:
                clone_entry = copy.deepcopy(template_entry)
                clone_entry["resource_id"] = new_id
                task["resources"].append(clone_entry)
        dropped_names: list[str] = []
    else:
        # n_to_remove == 0 — no-op.
        dropped_names = []
        added_names = []

    out_json_path = Path(out_json_path)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with out_json_path.open("w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    return {
        "profile": remove_from_profile,
        "removed": dropped_names,
        "added": added_names,
        "remaining": len(profile["resource_list"]),
    }


def build_duration_scaled_params(
    base_json_path: str | Path,
    *,
    scale_factor: float,
    out_json_path: str | Path,
) -> dict:
    """Multiply every task-resource duration-distribution param by ``scale_factor``.

    ``scale_factor = 1.0`` is a no-op. ``1.5`` makes every service time 50%
    longer on average (preserving shape). Applies to every entry in
    ``task_resource_distribution[*].resources[*].distribution_params``.

    Prosimos params are ``[{"value": x}, ...]`` — we rewrite each ``value``.
    Returns a small manifest for provenance.
    """
    if scale_factor <= 0:
        raise ValueError("scale_factor must be > 0")

    with open(base_json_path, encoding="utf-8") as f:
        params = json.load(f)
    params = copy.deepcopy(params)

    n_tasks = 0
    n_resources = 0
    n_params = 0
    for task in params.get("task_resource_distribution", []):
        n_tasks += 1
        for r in task.get("resources", []):
            n_resources += 1
            new_params = []
            for p in r.get("distribution_params", []):
                new_params.append({"value": p["value"] * scale_factor})
                n_params += 1
            r["distribution_params"] = new_params

    out_json_path = Path(out_json_path)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with out_json_path.open("w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    return {
        "scale_factor": scale_factor,
        "tasks_touched": n_tasks,
        "resource_entries_touched": n_resources,
        "params_rewritten": n_params,
    }


# ---------------------------------------------------------------------------
# Structural perturbations
# ---------------------------------------------------------------------------

def _load_params(base_json_path: str | Path) -> dict:
    with open(base_json_path, encoding="utf-8") as f:
        return copy.deepcopy(json.load(f))


def _write_params(params: dict, out_json_path: str | Path) -> None:
    out_json_path = Path(out_json_path)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with out_json_path.open("w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)


def _profile_resource_ids(params: dict, profile_name: str) -> list[str]:
    prof = next(
        (p for p in params.get("resource_profiles", []) if p.get("name") == profile_name),
        None,
    )
    if prof is None:
        raise ValueError(f"profile {profile_name!r} not found")
    return [r["id"] for r in prof.get("resource_list", [])]


def build_role_swap_params(
    base_json_path: str | Path,
    *,
    from_profile: str,
    to_profile: str,
    n_activities: int,
    out_json_path: str | Path,
) -> dict:
    """Re-route ``n_activities`` tasks from one role to another.

    For each affected task, every resource entry whose ``resource_id``
    belongs to ``from_profile`` is replaced with a parallel entry for each
    resource in ``to_profile`` — preserving the original distribution.
    Cycle time is approximately preserved (same per-instance distribution
    on a different pool); the ``activity_role`` projection sees the change.

    Activity selection: deterministic — the first ``n_activities`` tasks
    in ``task_resource_distribution`` that currently route to a resource in
    ``from_profile`` and *don't* already include any ``to_profile``
    resource. This avoids picking tasks that are already multi-role.

    ``n_activities == 0`` is a no-op.
    """
    if n_activities < 0:
        raise ValueError("n_activities must be >= 0")
    params = _load_params(base_json_path)
    if from_profile == to_profile:
        raise ValueError("from_profile and to_profile must differ")

    from_ids = set(_profile_resource_ids(params, from_profile))
    to_resources = next(
        p for p in params["resource_profiles"] if p["name"] == to_profile
    )["resource_list"]
    if not to_resources:
        raise ValueError(f"to_profile {to_profile!r} is empty")
    to_ids = {r["id"] for r in to_resources}

    swapped_tasks: list[str] = []
    if n_activities == 0:
        _write_params(params, out_json_path)
        return {
            "from_profile": from_profile, "to_profile": to_profile,
            "n_activities": 0, "swapped_tasks": [],
        }

    for task in params.get("task_resource_distribution", []):
        if len(swapped_tasks) >= n_activities:
            break
        task_resources = task.get("resources", [])
        from_entries = [r for r in task_resources if r["resource_id"] in from_ids]
        if not from_entries:
            continue
        # Skip already-multi-role tasks (one of to_profile's resources is
        # already a candidate) to keep the swap unambiguous.
        if any(r["resource_id"] in to_ids for r in task_resources):
            continue
        # Use the first from_entry's distribution as the template — the
        # whole point of role swap is to preserve per-instance distribution
        # while moving the worker pool.
        template = from_entries[0]
        kept = [r for r in task_resources if r["resource_id"] not in from_ids]
        new_entries = []
        for to_res in to_resources:
            clone = copy.deepcopy(template)
            clone["resource_id"] = to_res["id"]
            new_entries.append(clone)
        task["resources"] = kept + new_entries
        swapped_tasks.append(task["task_id"])

    if len(swapped_tasks) < n_activities:
        raise ValueError(
            f"only {len(swapped_tasks)} task(s) eligible for swap from "
            f"{from_profile!r} → {to_profile!r}; requested {n_activities}"
        )

    _write_params(params, out_json_path)
    return {
        "from_profile": from_profile, "to_profile": to_profile,
        "n_activities": n_activities, "swapped_tasks": swapped_tasks,
    }


# Calendar timestamps in Prosimos JSON look like "09:00:00" or
# "23:59:59.999000". Parse to seconds-since-midnight, mutate, format back.
_TIME_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,6}))?$")


def _time_to_seconds(t: str) -> float:
    m = _TIME_RE.match(t)
    if not m:
        raise ValueError(f"unrecognised time literal: {t!r}")
    h, mi, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
    frac = float("0." + (m.group(4) or "0"))
    return h * 3600 + mi * 60 + s + frac


def _seconds_to_time(s: float) -> str:
    s = s % 86400
    hours = int(s // 3600)
    rem = s - hours * 3600
    minutes = int(rem // 60)
    secs = rem - minutes * 60
    whole_secs = int(secs)
    frac = secs - whole_secs
    if frac > 0:
        # Always emit microseconds for fractional values, matching Prosimos style.
        return f"{hours:02d}:{minutes:02d}:{whole_secs:02d}.{int(round(frac * 1_000_000)):06d}"
    return f"{hours:02d}:{minutes:02d}:{whole_secs:02d}"


def _shift_periods_inplace(time_periods: list, shift_hours: float, *, label: str) -> None:
    """Shift every period in ``time_periods`` by ``shift_hours`` (in place).

    Guards: rejects full-day (>=24h) periods (shifting them is a no-op and
    produces a midnight-wrapping range that crashes Prosimos) and any shift
    that would push a period across midnight (split-period output is not
    implemented). ``label`` is used only in error messages.
    """
    shift_seconds = shift_hours * 3600.0
    for p in time_periods:
        begin = _time_to_seconds(p["beginTime"])
        end = _time_to_seconds(p["endTime"])
        if end - begin >= 24 * 3600 - 1:  # leave 1s slack for ...59.999 quirks
            raise ValueError(
                f"calendar_shift requires sub-24h periods; period "
                f"{p['beginTime']}-{p['endTime']} on {p['from']} in {label} "
                f"covers the full day and shifting is a no-op. Use a dataset "
                f"with business-hours calendars."
            )
        new_begin = begin + shift_seconds
        new_end = end + shift_seconds
        if (new_begin < 0 or new_end > 86400) or \
           (new_begin % 86400 > new_end % 86400):
            raise NotImplementedError(
                f"calendar_shift would wrap midnight ({p['beginTime']}-"
                f"{p['endTime']} on {p['from']} in {label} shifted by "
                f"{shift_hours}h); split-period output is not yet implemented. "
                f"Use a smaller |shift_hours|."
            )
        p["beginTime"] = _seconds_to_time(new_begin)
        p["endTime"] = _seconds_to_time(new_end)


def build_all_calendars_shifted_params(
    base_json_path: str | Path,
    *,
    shift_hours: float,
    out_json_path: str | Path,
    shift_arrival: bool = True,
) -> dict:
    """Shift EVERY resource calendar (and, by default, the arrival calendar)
    by the same ``shift_hours``.

    This is a pure *phase translation*: because all roles move together, the
    relative availability between roles is unchanged, so per-case service
    dynamics (and hence cycle time, especially for cases that arrive entirely
    within the shifted regime) are preserved. Only the wall-clock alignment of
    the whole process moves. Contrast with ``build_calendar_shifted_params``,
    which shifts one profile and therefore *desynchronises* hand-offs and does
    change cycle time.

    Shifting the arrival calendar too keeps new arrivals fully translated; set
    ``shift_arrival=False`` to keep arrivals fixed (then new arrivals see the
    same wall-clock arrival pattern but a shifted service window).
    """
    params = _load_params(base_json_path)
    calendars = params.get("resource_calendars", [])
    if not calendars:
        raise ValueError("no resource_calendars to shift")
    if shift_hours != 0:
        for cal in calendars:
            _shift_periods_inplace(cal.get("time_periods", []), shift_hours,
                                   label=str(cal.get("id")))
        if shift_arrival and params.get("arrival_time_calendar"):
            _shift_periods_inplace(params["arrival_time_calendar"], shift_hours,
                                   label="arrival_time_calendar")
    _write_params(params, out_json_path)
    return {
        "shift_hours": shift_hours,
        "n_calendars_shifted": len(calendars),
        "arrival_shifted": bool(shift_arrival and shift_hours != 0),
    }


def build_calendar_shifted_params(
    base_json_path: str | Path,
    *,
    profile_name: str,
    shift_hours: float,
    out_json_path: str | Path,
) -> dict:
    """Shift one resource calendar's time periods by ``shift_hours``.

    Matches the calendar whose ``id`` or ``name`` is ``{profile_name}_calendar``
    (the convention used by Loan-stable and friends) OR whose ``id`` literally
    equals ``profile_name``. The total work-hours per period are preserved
    — only ``beginTime`` / ``endTime`` shift.

    Wraps modulo 24 hours: a period 09:00-17:00 shifted by +10 becomes
    19:00-03:00. Prosimos appears to handle wraparound periods correctly,
    but we don't currently split a wrapped period into two — if your
    Prosimos build doesn't accept wraparound, choose ``shift_hours`` so the
    shifted range stays within a single day.

    ``shift_hours == 0`` is a no-op.
    """
    params = _load_params(base_json_path)
    calendars = params.get("resource_calendars", [])
    candidates = [
        c for c in calendars
        if c.get("id") in (profile_name, f"{profile_name}_calendar")
        or c.get("name") in (profile_name, f"{profile_name}_calendar")
    ]
    if not candidates:
        raise ValueError(
            f"no resource_calendar matching profile {profile_name!r}"
        )
    if len(candidates) > 1:
        raise ValueError(
            f"multiple calendars match profile {profile_name!r}: "
            f"{[c.get('id') for c in candidates]}"
        )
    cal = candidates[0]

    original_periods = [
        {"from": p["from"], "to": p["to"],
         "beginTime": p["beginTime"], "endTime": p["endTime"]}
        for p in cal.get("time_periods", [])
    ]
    if shift_hours != 0:
        _shift_periods_inplace(cal.get("time_periods", []), shift_hours,
                               label=str(cal.get("id")))

    _write_params(params, out_json_path)
    return {
        "calendar_id": cal.get("id"),
        "profile_name": profile_name,
        "shift_hours": shift_hours,
        "original_periods": original_periods,
        "shifted_periods": [
            {"from": p["from"], "to": p["to"],
             "beginTime": p["beginTime"], "endTime": p["endTime"]}
            for p in cal.get("time_periods", [])
        ],
    }


def _select_balanced_gateway(gateways: list[dict]) -> int:
    """Return index of the gateway with the closest-to-uniform probabilities.

    Measured by max probability across paths — the smaller the max, the more
    balanced the gateway. 2-way only is preferred when a tie exists, since
    bias is unambiguous for 2-way.
    """
    if not gateways:
        raise ValueError("no gateways in params")
    best_idx = 0
    best_max = float("inf")
    for i, g in enumerate(gateways):
        probs = [pr["value"] for pr in g.get("probabilities", [])]
        if not probs:
            continue
        m = max(probs)
        if m < best_max:
            best_max = m
            best_idx = i
    return best_idx


def build_gateway_biased_params(
    base_json_path: str | Path,
    *,
    gateway_id: str | None,
    bias_level: int,
    out_json_path: str | Path,
) -> dict:
    """Bias one gateway's probabilities by ``bias_level`` steps.

    Picks the gateway by ``gateway_id`` if provided, otherwise the most
    balanced gateway (closest-to-uniform path probabilities).

    Each bias step shifts probability mass toward the currently-majority
    path. For 2-way gateways the encoding is intuitive:

        level=0 → original
        level=1 → push majority share +0.1 (capped at 0.95)
        level=2 → push +0.2
        ...

    For >2-way gateways, the majority path takes the full shift and other
    paths shrink proportionally to keep the total at 1.0.
    """
    if bias_level < 0:
        raise ValueError("bias_level must be >= 0")
    params = _load_params(base_json_path)
    gateways = params.get("gateway_branching_probabilities", [])
    if not gateways:
        raise ValueError("params has no gateway_branching_probabilities")

    if gateway_id is not None:
        idx = next(
            (i for i, g in enumerate(gateways) if g.get("gateway_id") == gateway_id),
            None,
        )
        if idx is None:
            raise ValueError(f"gateway {gateway_id!r} not found")
    else:
        idx = _select_balanced_gateway(gateways)

    gateway = gateways[idx]
    probs = gateway.get("probabilities", [])
    if len(probs) < 2:
        raise ValueError(
            f"gateway {gateway.get('gateway_id')} has <2 paths; cannot bias"
        )
    original_probs = [
        {"path_id": p["path_id"], "value": float(p["value"])} for p in probs
    ]

    if bias_level > 0:
        shift = min(0.1 * bias_level, 0.95)
        # Majority path indices are picked by current value (ties broken by
        # path_id for determinism).
        majority_idx = max(
            range(len(probs)),
            key=lambda i: (probs[i]["value"], probs[i]["path_id"]),
        )
        majority_val = probs[majority_idx]["value"]
        new_majority = min(majority_val + shift, 0.99)
        actual_shift = new_majority - majority_val
        # Distribute the negative shift across the other paths proportional to
        # their current value (so already-tiny paths stay tiny).
        other_total = sum(p["value"] for i, p in enumerate(probs) if i != majority_idx)
        if other_total <= 0:
            raise ValueError("non-majority paths have zero mass; cannot shrink")
        for i, p in enumerate(probs):
            if i == majority_idx:
                p["value"] = new_majority
            else:
                p["value"] = p["value"] - actual_shift * (p["value"] / other_total)
                if p["value"] < 0:
                    p["value"] = 0.0

    # Renormalise defensively against floating drift.
    total = sum(p["value"] for p in probs)
    if total > 0:
        for p in probs:
            p["value"] = p["value"] / total

    _write_params(params, out_json_path)
    return {
        "gateway_id": gateway["gateway_id"],
        "gateway_index": idx,
        "bias_level": bias_level,
        "original_probs": original_probs,
        "new_probs": [
            {"path_id": p["path_id"], "value": float(p["value"])} for p in probs
        ],
    }


def build_case_route_params(
    base_json_path: str | Path,
    *,
    n_gateways_ruled: int,
    out_json_path: str | Path,
) -> dict:
    """Make the first ``n_gateways_ruled`` XOR splits route by ``case_type``.

    Designed for the *case-route* synthetic (see
    ``tools/generate_case_route.py``): every split gateway has two
    duration-symmetric branches whose flow ids end in ``_a`` / ``_b``. In the
    base params each split routes 50/50 by static probability, so ``case_type``
    is independent of the path. This builder rewrites the first
    ``n_gateways_ruled`` split gateways (ordered by gateway id) to bind the
    ``_a`` branch to ``rule_red`` and the ``_b`` branch to ``rule_blue`` (the
    equality branch_rules already declared in the base params), so red cases
    take the A-branch and blue cases the B-branch at those gateways.

    Each gateway's path marginal stays 50/50 (population is 50/50 red/blue), so
    cycle time and the 2-gram distribution are unchanged; only the joint
    (activity, case_type) distribution shifts. ``n_gateways_ruled == 0`` is a
    no-op (copies the base params verbatim).

    Returns a manifest listing which gateways were ruled.
    """
    if n_gateways_ruled < 0:
        raise ValueError("n_gateways_ruled must be >= 0")
    params = _load_params(base_json_path)

    # A "split" gateway here is any branching entry with exactly two paths whose
    # ids end in _a / _b — the convention emitted by the case-route generator.
    def _is_split(entry: dict) -> bool:
        ids = [p["path_id"] for p in entry.get("probabilities", [])]
        return len(ids) == 2 and any(i.endswith("_a") for i in ids) and any(
            i.endswith("_b") for i in ids
        )

    splits = sorted(
        (e for e in params.get("gateway_branching_probabilities", []) if _is_split(e)),
        key=lambda e: e["gateway_id"],
    )
    if n_gateways_ruled > len(splits):
        raise ValueError(
            f"requested {n_gateways_ruled} ruled gateways but only "
            f"{len(splits)} split gateways exist"
        )

    rule_ids = {r["id"] for r in params.get("branch_rules", [])}
    for needed in ("rule_red", "rule_blue"):
        if needed not in rule_ids:
            raise ValueError(f"base params is missing branch rule {needed!r}")

    ruled: list[str] = []
    for entry in splits[:n_gateways_ruled]:
        for p in entry["probabilities"]:
            cond = "rule_red" if p["path_id"].endswith("_a") else "rule_blue"
            # Bind to the rule; drop the static probability so the engine uses
            # the condition (a probabilities entry is either value- or
            # condition-keyed, never both).
            p.pop("value", None)
            p["condition_id"] = cond
        ruled.append(entry["gateway_id"])

    _write_params(params, out_json_path)
    return {
        "n_gateways_ruled": n_gateways_ruled,
        "ruled_gateways": ruled,
        "n_split_gateways": len(splits),
    }


def build_route_error_params(
    base_json_path: str | Path,
    *,
    n_gateways_inverted: int,
    out_json_path: str | Path,
) -> dict:
    """Invert the ``case_type`` routing on the first ``n_gateways_inverted``
    XOR splits — the *model error* for scenario #3.

    Designed for the *route-error* synthetic (see
    ``tools/generate_route_error.py``), whose base params already route
    CORRECTLY by case_type at every split gateway: the ``_a`` (long) branch is
    bound to ``rule_red`` and the ``_b`` (short) branch to ``rule_blue`` via
    ``condition_id``. This builder rewrites the first ``n_gateways_inverted``
    split gateways (ordered by gateway id) to **swap** those bindings — the
    ``_a`` branch now fires on ``rule_blue`` and the ``_b`` branch on
    ``rule_red`` — so red cases take the short branch and blue cases the long
    branch at those gateways.

    Because the population is 50/50 red/blue, each gateway's path marginal stays
    50/50, so cycle time, the 2-gram distribution and the aggregate
    relative-event-distribution are all unchanged; only each case's *own* path
    (paired by case_id) flips. ``n_gateways_inverted == 0`` is a no-op (copies
    the base params verbatim).

    Returns a manifest listing which gateways were inverted.
    """
    if n_gateways_inverted < 0:
        raise ValueError("n_gateways_inverted must be >= 0")
    params = _load_params(base_json_path)

    # A "split" gateway here is any branching entry with exactly two paths whose
    # ids end in _a / _b — the convention emitted by the route-error generator.
    def _is_split(entry: dict) -> bool:
        ids = [p["path_id"] for p in entry.get("probabilities", [])]
        return len(ids) == 2 and any(i.endswith("_a") for i in ids) and any(
            i.endswith("_b") for i in ids
        )

    splits = sorted(
        (e for e in params.get("gateway_branching_probabilities", []) if _is_split(e)),
        key=lambda e: e["gateway_id"],
    )
    if n_gateways_inverted > len(splits):
        raise ValueError(
            f"requested {n_gateways_inverted} inverted gateways but only "
            f"{len(splits)} split gateways exist"
        )

    inverted: list[str] = []
    for entry in splits[:n_gateways_inverted]:
        for p in entry["probabilities"]:
            # Swap the rule so the long (_a) branch fires on blue and the short
            # (_b) branch on red — the inverse of the correct base routing.
            if "condition_id" not in p:
                raise ValueError(
                    f"gateway {entry['gateway_id']!r} path {p['path_id']!r} has no "
                    "condition_id; base params must route by case_type "
                    "(generate_route_error bakes this in)"
                )
            p["condition_id"] = (
                "rule_blue" if p["path_id"].endswith("_a") else "rule_red"
            )
        inverted.append(entry["gateway_id"])

    _write_params(params, out_json_path)
    return {
        "n_gateways_inverted": n_gateways_inverted,
        "inverted_gateways": inverted,
        "n_split_gateways": len(splits),
    }


def build_branch_automation_params(
    base_json_path: str | Path,
    *,
    automate_task_ids: list[str],
    level: int,
    floor_seconds: float = 1.0,
    out_json_path: str | Path,
) -> dict:
    """Automate (shrink) the durations of the non-critical branch tasks.

    Designed for the *parallel-automation* synthetic (see
    ``tools/generate_parallel_auto.py``): the listed ``automate_task_ids`` are
    the non-critical AND branch, whose total duration is dominated by the
    critical branch. Each task's duration params are scaled by
    ``(1 - level/100)``, with the location parameter floored at
    ``floor_seconds`` so the activity still appears once per case (keeping the
    bigram histogram and per-case activity counts identical, hence ``ngd_n2``
    blind). Because the critical branch sets the cycle time and runs on a
    separate pool, ``cycle_time`` is blind too — only the time-weighted state
    metric sees the non-critical branch shrink out of the concurrent active set.

    ``level == 0`` is a no-op (factor 1.0). ``level`` is read as a percentage in
    ``[0, 100]``; ``level == 100`` collapses every automated task to the floor.

    Returns a manifest with the scale factor and which tasks were touched.
    """
    if level < 0:
        raise ValueError("level must be >= 0")
    if level > 100:
        raise ValueError("level must be <= 100")
    if not automate_task_ids:
        raise ValueError("automate_task_ids must be non-empty")
    if floor_seconds <= 0:
        raise ValueError("floor_seconds must be > 0")

    params = _load_params(base_json_path)
    factor = 1.0 - level / 100.0
    targets = set(automate_task_ids)
    touched: list[str] = []
    for task in params.get("task_resource_distribution", []):
        if task.get("task_id") not in targets:
            continue
        touched.append(task["task_id"])
        for r in task.get("resources", []):
            dparams = r.get("distribution_params", [])
            new_params = []
            for i, p in enumerate(dparams):
                scaled = p["value"] * factor
                # Floor only the location/value parameter (index 0) so the
                # activity keeps a strictly positive duration and stays in the
                # trace; scale any shape/min/max params by the same factor.
                if i == 0:
                    scaled = max(scaled, floor_seconds)
                new_params.append({"value": scaled})
            r["distribution_params"] = new_params

    missing = targets - set(touched)
    if missing:
        raise ValueError(
            f"automate_task_ids not found in task_resource_distribution: {sorted(missing)}"
        )

    _write_params(params, out_json_path)
    return {
        "automate_task_ids": list(automate_task_ids),
        "level": level,
        "factor": factor,
        "floor_seconds": floor_seconds,
        "tasks_touched": touched,
    }


def build_front_back_load_params(
    base_json_path: str | Path,
    *,
    chain_task_ids: list[str],
    shift: int,
    out_json_path: str | Path,
) -> dict:
    """Redistribute duration mass along a sequential chain, total-invariant.

    Designed for the *linear-chain* synthetic (see
    ``tools/generate_linear_chain.py``): the ordered ``chain_task_ids`` form a
    sequence of comparable-duration activities on one pool. This builder
    reweights their mean durations by position so the per-case **total**
    duration (and hence cycle time and aggregate utilisation) is held constant,
    but the duration mass moves toward the front or the back of the chain.

    Signed ``shift`` (a percentage):
      * ``shift > 0`` *front-loads* — early tasks longer, late tasks shorter.
      * ``shift < 0`` *back-loads* — late tasks longer, early tasks shorter.
      * ``shift == 0`` is a no-op.

    For a chain of ``n`` tasks at ordered positions ``p_i = i/(n-1) ∈ [0, 1]``
    the per-task weight is ``w_i = 1 + (shift/100)·(1 - 2·p_i)`` — an
    antisymmetric ramp about the chain midpoint. New means are ``m_i·w_i``,
    then globally renormalised by ``Σm_i / Σ(m_i·w_i)`` so the total is held
    **exactly** regardless of any base-mean asymmetry.

    Because only *where* the duration sits changes (not the per-case total, the
    activities, the bigrams, or aggregate ρ), ``cycle_time`` and ``ngd_n2`` are
    blind, while the time-weighted state metric localises the moved mass.
    ``|shift|`` must be < 100 so every weight stays positive.

    Returns a manifest with the per-task factors actually applied.
    """
    if abs(shift) >= 100:
        raise ValueError("|shift| must be < 100 so all task weights stay positive")
    if len(chain_task_ids) < 2:
        raise ValueError("chain_task_ids must list at least two tasks (ordered)")

    params = _load_params(base_json_path)
    trd_by_id = {t.get("task_id"): t for t in params.get("task_resource_distribution", [])}
    missing = [tid for tid in chain_task_ids if tid not in trd_by_id]
    if missing:
        raise ValueError(f"chain_task_ids not found in params: {missing}")

    n = len(chain_task_ids)
    s = shift / 100.0

    def _task_mean(task: dict) -> float:
        # Representative mean = first resource's location parameter.
        resources = task.get("resources", [])
        if not resources or not resources[0].get("distribution_params"):
            raise ValueError(f"task {task.get('task_id')!r} has no distribution params")
        return float(resources[0]["distribution_params"][0]["value"])

    if shift == 0:
        _write_params(params, out_json_path)
        return {
            "chain_task_ids": list(chain_task_ids), "shift": 0,
            "factors": {tid: 1.0 for tid in chain_task_ids},
            "total_invariant": True,
        }

    weights: dict[str, float] = {}
    for i, tid in enumerate(chain_task_ids):
        p_i = i / (n - 1)
        weights[tid] = 1.0 + s * (1.0 - 2.0 * p_i)

    base_total = sum(_task_mean(trd_by_id[tid]) for tid in chain_task_ids)
    new_total = sum(_task_mean(trd_by_id[tid]) * weights[tid] for tid in chain_task_ids)
    g = base_total / new_total if new_total > 0 else 1.0

    factors: dict[str, float] = {}
    for tid in chain_task_ids:
        f = weights[tid] * g
        factors[tid] = f
        for r in trd_by_id[tid].get("resources", []):
            r["distribution_params"] = [
                {"value": p["value"] * f} for p in r.get("distribution_params", [])
            ]

    _write_params(params, out_json_path)
    return {
        "chain_task_ids": list(chain_task_ids),
        "shift": shift,
        "factors": factors,
        "base_total_mean": base_total,
        "renormalised_total_mean": sum(
            _task_mean(trd_by_id[tid]) for tid in chain_task_ids
        ),
        "total_invariant": True,
    }


def build_arrival_burstier_params(
    base_json_path: str | Path,
    *,
    cv2_multiplier: float,
    out_json_path: str | Path,
) -> dict:
    """Make arrival distribution burstier by scaling its CV² (variance/mean²).

    Currently supports gamma arrivals (Loan-stable and other ICPM-2025
    synthetics): scales the second distribution parameter (variance) by
    ``cv2_multiplier`` while leaving the mean (first parameter) unchanged.
    Long-run throughput is preserved; arrivals come in bigger bursts with
    longer quiet stretches.

    For other distributions a NotImplementedError is raised — adding
    explicit support per distribution avoids silently mis-perturbing them.

    Convention: the first two distribution_params for gamma in Prosimos are
    [mean, variance, min, max] (the same convention as exponential/normal in
    this codebase — see ``build_duration_scaled_params``). If your version
    of Prosimos differs, the manifest's ``mean_before``/``mean_after`` will
    reveal the mistake quickly.

    ``cv2_multiplier == 1.0`` is a no-op.
    """
    if cv2_multiplier <= 0:
        raise ValueError("cv2_multiplier must be > 0")
    params = _load_params(base_json_path)
    arrival = params.get("arrival_time_distribution")
    if arrival is None:
        raise ValueError("params has no arrival_time_distribution")

    name = arrival.get("distribution_name")
    if name != "gamma":
        raise NotImplementedError(
            f"arrival burstiness only supported for gamma (got {name!r})"
        )

    dparams = arrival.get("distribution_params", [])
    if len(dparams) < 2:
        raise ValueError("gamma arrival needs at least 2 distribution_params")
    mean_before = dparams[0]["value"]
    var_before = dparams[1]["value"]
    dparams[1] = {"value": var_before * cv2_multiplier}
    mean_after = dparams[0]["value"]
    var_after = dparams[1]["value"]

    _write_params(params, out_json_path)
    return {
        "distribution": name,
        "cv2_multiplier": cv2_multiplier,
        "mean_before": mean_before,
        "mean_after": mean_after,
        "variance_before": var_before,
        "variance_after": var_after,
    }
