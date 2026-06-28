from collections import Counter

import pytest

from state_metrics.views import ActiveInstance, project


@pytest.fixture
def instances():
    return [
        ActiveInstance("c1", "A", "R1"),
        ActiveInstance("c2", "A", "R1"),
        ActiveInstance("c2", "B", "R2"),
    ]


def test_view_activity_case(instances):
    assert project(instances, "activity_case") == Counter(
        {("A", "c1"): 1, ("A", "c2"): 1, ("B", "c2"): 1}
    )


def test_view_case(instances):
    assert project(instances, "case") == Counter({"c1": 1, "c2": 2})


def test_view_activity(instances):
    assert project(instances, "activity") == Counter({"A": 2, "B": 1})


def test_view_activity_role(instances):
    role_map = {"R1": "junior", "R2": "senior"}
    assert project(instances, "activity_role", role_map=role_map) == Counter(
        {("A", "junior"): 2, ("B", "senior"): 1}
    )


def test_view_activity_role_requires_map(instances):
    with pytest.raises(ValueError):
        project(instances, "activity_role", role_map=None)


def test_view_activity_role_missing_resource(instances):
    with pytest.raises(KeyError):
        project(instances, "activity_role", role_map={"R1": "j"})


def test_view_unknown_raises(instances):
    with pytest.raises(ValueError):
        project(instances, "something_else")


def test_view_case_type():
    typed = [
        ActiveInstance("c1", "A", "R1", case_type="green"),
        ActiveInstance("c2", "A", "R1", case_type="red"),
        ActiveInstance("c3", "B", "R2", case_type="green"),
    ]
    assert project(typed, "case_type") == Counter({"green": 2, "red": 1})


def test_view_activity_type():
    typed = [
        ActiveInstance("c1", "A", "R1", case_type="green"),
        ActiveInstance("c2", "A", "R1", case_type="red"),
        ActiveInstance("c3", "B", "R2", case_type="green"),
    ]
    # (activity, case_type) — distinguishes same activity on different
    # case types, which is the projection the label_swap oracle relies on.
    assert project(typed, "activity_type") == Counter(
        {("A", "green"): 1, ("A", "red"): 1, ("B", "green"): 1}
    )


def test_active_instance_default_case_type_is_empty(instances):
    # Backward compat: pre-case_type code constructs ActiveInstance with three
    # positional args; case_type defaults to "" so the projection still works
    # (and yields a single bucket).
    assert project(instances, "case_type") == Counter({"": 3})
