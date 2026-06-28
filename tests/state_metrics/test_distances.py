from collections import Counter

from state_metrics.distances import cardinality, jaccard_multiset


def test_jaccard_identical():
    a = Counter({"x": 2, "y": 1})
    assert jaccard_multiset(a, a) == 0.0


def test_jaccard_disjoint():
    a = Counter({"x": 1})
    b = Counter({"y": 1})
    assert jaccard_multiset(a, b) == 1.0


def test_jaccard_one_empty():
    a = Counter({"x": 3})
    b = Counter()
    assert jaccard_multiset(a, b) == 1.0


def test_jaccard_both_empty():
    assert jaccard_multiset(Counter(), Counter()) == 0.0


def test_jaccard_partial_overlap():
    a = Counter({"x": 2, "y": 1})
    b = Counter({"x": 1, "z": 1})
    # min: x=1, y=0, z=0 -> 1 ; max: x=2, y=1, z=1 -> 4 ; d = 1 - 1/4 = 0.75
    assert jaccard_multiset(a, b) == 0.75


def test_cardinality_same_size_different_labels():
    a = Counter({"x": 2})
    b = Counter({"y": 2})
    # Same total -> cardinality distance is 0, even though labels differ.
    assert cardinality(a, b) == 0.0


def test_cardinality_size_mismatch():
    a = Counter({"x": 3})
    b = Counter({"y": 1})
    assert cardinality(a, b) == (3 - 1) / 3


def test_cardinality_both_empty():
    assert cardinality(Counter(), Counter()) == 0.0


def test_cardinality_one_empty():
    a = Counter({"x": 4})
    b = Counter()
    assert cardinality(a, b) == 1.0


def test_jaccard_vs_cardinality_diverge_on_labels():
    # Same size so cardinality=0, but disjoint labels so jaccard=1.
    a = Counter({"x": 1})
    b = Counter({"y": 1})
    assert cardinality(a, b) == 0.0
    assert jaccard_multiset(a, b) == 1.0
