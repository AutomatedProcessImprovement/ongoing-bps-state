import copy
from typing import List

MAX_POOL_SIZE = 500
MIN_PERCENT = -90
MAX_PERCENT = 300
PARAMETER_POWERS = {
    "fix": [1],
    "expon": [1, 1, 1],
    "norm": [1, 1, 1, 1],
    "uniform": [1, 1],
    "lognorm": [1, 2, 1, 1],
    "gamma": [1, 2, 1, 1],
}


class InvalidEdit(ValueError):
    pass


def pool_size(pool: dict) -> int:
    return sum(int(resource["amount"]) for resource in pool["resource_list"])


def pool_members(pool: dict) -> List[dict]:
    members = []
    for resource in pool["resource_list"]:
        amount = int(resource["amount"])
        names = [f"{resource['name']}_{i}" for i in range(amount)] if amount > 1 else [resource["name"]]
        members += [{"name": name, "calendar": resource["calendar"]} for name in names]
    return members


def can_edit_pools(parameters: dict) -> bool:
    return parameters.get("model_type") != "FUZZY"


def apply_edits(parameters: dict, edits: dict) -> dict:
    edited = copy.deepcopy(parameters)
    if not can_edit_pools(edited) and edits.get("pools"):
        raise InvalidEdit("Pool sizes cannot be changed for a model with fuzzy calendars.")

    for pool_id, size in (edits.get("pools") or {}).items():
        pool = next((pool for pool in edited["resource_profiles"] if pool["id"] == pool_id), None)
        if pool is None:
            raise InvalidEdit(f"Unknown resource pool: {pool_id}")
        if not 1 <= size <= MAX_POOL_SIZE:
            raise InvalidEdit(f"{pool['name']} must keep between 1 and {MAX_POOL_SIZE} resources.")
        _resize_pool(edited, pool, size)

    arrival_percent = edits.get("arrival_percent") or 0
    if arrival_percent:
        _check_percent(arrival_percent, "Arrivals")
        edited["arrival_time_distribution"] = _scaled(
            edited["arrival_time_distribution"], 100 / (100 + arrival_percent),
        )

    task_percents = edits.get("tasks") or {}
    known_tasks = {task["task_id"] for task in edited["task_resource_distribution"]}
    for task_id, percent in task_percents.items():
        if task_id not in known_tasks:
            raise InvalidEdit(f"Unknown task: {task_id}")
        _check_percent(percent, "Task durations")
    for task in edited["task_resource_distribution"]:
        percent = task_percents.get(task["task_id"])
        if percent:
            task["resources"] = [
                {**assignment, **_scaled(assignment, (100 + percent) / 100)}
                for assignment in task["resources"]
            ]
    return edited


def _check_percent(percent: int, label: str):
    if not MIN_PERCENT <= percent <= MAX_PERCENT:
        raise InvalidEdit(f"{label} can change between {MIN_PERCENT}% and +{MAX_PERCENT}%.")


def _scaled(distribution: dict, factor: float) -> dict:
    name = distribution.get("distribution_name")
    powers = PARAMETER_POWERS.get(name)
    if powers is None:
        raise InvalidEdit(f"The {name} distribution cannot be scaled.")
    return {
        "distribution_name": name,
        "distribution_params": [
            {"value": float(parameter["value"]) * factor ** power}
            for parameter, power in zip(distribution["distribution_params"], powers)
        ],
    }


def _resize_pool(parameters: dict, pool: dict, size: int):
    current = pool_size(pool)
    if size == current:
        return
    expand_amounts(parameters, pool)
    if size > current:
        _add_resources(parameters, pool, size - current)
    else:
        _remove_resources(parameters, pool, current - size)


def expand_amounts(parameters: dict, pool: dict):
    copies = {}
    expanded = []
    for resource in pool["resource_list"]:
        amount = int(resource["amount"])
        if amount == 1:
            expanded.append(resource)
            continue
        members = [
            {**resource, "id": f"{resource['id']}_{i}", "name": f"{resource['name']}_{i}", "amount": 1}
            for i in range(amount)
        ]
        copies[resource["id"]] = [member["id"] for member in members]
        expanded += members
    pool["resource_list"] = expanded
    for task in parameters["task_resource_distribution"]:
        assignments = []
        for assignment in task["resources"]:
            assignments += [
                {**assignment, "resource_id": member_id}
                for member_id in copies.get(assignment["resource_id"], [assignment["resource_id"]])
            ]
        task["resources"] = assignments


def _add_resources(parameters: dict, pool: dict, count: int):
    template = pool["resource_list"][-1]
    for number in range(1, count + 1):
        added = {
            **template,
            "id": f"{template['id']}_added_{number}",
            "name": f"{pool['name']} added {number}",
            "amount": 1,
        }
        pool["resource_list"].append(added)
        for task in parameters["task_resource_distribution"]:
            assignment = next((a for a in task["resources"] if a["resource_id"] == template["id"]), None)
            if assignment:
                task["resources"].append({**assignment, "resource_id": added["id"]})


def _remove_resources(parameters: dict, pool: dict, count: int):
    for resource in reversed(list(pool["resource_list"])):
        if count == 0:
            break
        assigned = [
            task for task in parameters["task_resource_distribution"]
            if any(a["resource_id"] == resource["id"] for a in task["resources"])
        ]
        if any(len(task["resources"]) == 1 for task in assigned):
            continue
        pool["resource_list"].remove(resource)
        for task in assigned:
            task["resources"] = [a for a in task["resources"] if a["resource_id"] != resource["id"]]
        count -= 1
    if count > 0:
        raise InvalidEdit(f"{pool['name']} cannot lose that many: some tasks would have nobody left to do them.")
