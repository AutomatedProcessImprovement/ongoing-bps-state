import os
import tempfile
from typing import List, Set, Tuple

import pytest

from process_running_state.n_gram_index import NGramIndex
from test_bpmn_model_fixtures import _bpmn_model_with_loop_inside_AND, _bpmn_model_with_AND_and_nested_XOR, \
    _bpmn_model_with_XOR_within_AND, _bpmn_model_with_AND_and_XOR, \
    _bpmn_model_with_two_loops_inside_AND_followed_by_XOR_within_AND, \
    _bpmn_model_with_three_loops_inside_AND_two_of_them_inside_sub_AND


@pytest.fixture
def temp_file_path():
    # Create a temporary file and get the path
    tmp = tempfile.NamedTemporaryFile(mode='w+t', delete=False)  # Set delete=False
    path = tmp.name
    try:
        yield path
    finally:
        # Cleanup: close the file and remove it
        tmp.close()
        os.remove(path)


def _prepare(markings: List[Set[str]]) -> Set[Tuple[str]]:
    return {tuple(sorted(marking)) for marking in markings}


def test_build_simple_size_limit():
    bpmn_model = _bpmn_model_with_AND_and_XOR()
    reachability_graph = bpmn_model.get_reachability_graph()
    n_gram_index = NGramIndex(reachability_graph, n_gram_size_limit=1)
    n_gram_index.build()
    # Check the maximum size of the explored n-grams is under the limit
    n_gram_sizes = [len(n_gram) for n_gram in n_gram_index.markings]
    assert max(n_gram_sizes) <= 1
    # Check number of n-grams is the expected (num of labels + trace_start)
    assert len(n_gram_index.markings) == len(reachability_graph.activity_to_edges) + 1
    # Check specific markings
    assert n_gram_index.get_marking_state([NGramIndex.TRACE_START]) == [{"1"}]
    assert n_gram_index.get_marking_state(["A"]) == [{"5", "6"}]
    assert n_gram_index.get_marking_state(["B"]) == [{"9", "6"}, {"12"}]
    assert n_gram_index.get_marking_state(["C"]) == [{"5", "10"}, {"12"}]
    assert n_gram_index.get_marking_state(["D"]) == [{"21"}]
    assert n_gram_index.get_marking_state(["E"]) == [{"21"}]
    assert n_gram_index.get_marking_state(["F"]) == [{"23"}]


def test_build_simple():
    bpmn_model = _bpmn_model_with_AND_and_XOR()
    reachability_graph = bpmn_model.get_reachability_graph()
    n_gram_index = NGramIndex(reachability_graph, n_gram_size_limit=3)
    n_gram_index.build()
    # Check the maximum size of the explored n-grams is under the limit
    n_gram_sizes = [len(n_gram) for n_gram in n_gram_index.markings]
    assert max(n_gram_sizes) <= 3
    # Check number of n-grams is the expected
    assert len(n_gram_index.markings) == 11
    # Check specific markings
    assert n_gram_index.get_marking_state([NGramIndex.TRACE_START]) == [{"1"}]
    assert n_gram_index.get_marking_state(["A"]) == [{"5", "6"}]
    assert n_gram_index.get_marking_state(["B"]) == [{"9", "6"}, {"12"}]
    assert n_gram_index.get_marking_state(["C"]) == [{"5", "10"}, {"12"}]
    assert n_gram_index.get_marking_state(["A", "B"]) == [{"9", "6"}]
    assert n_gram_index.get_marking_state(["C", "B"]) == [{"12"}]
    assert n_gram_index.get_marking_state(["A", "C"]) == [{"5", "10"}]
    assert n_gram_index.get_marking_state(["B", "C"]) == [{"12"}]
    assert n_gram_index.get_marking_state(["D"]) == [{"21"}]
    assert n_gram_index.get_marking_state(["E"]) == [{"21"}]
    assert n_gram_index.get_marking_state(["F"]) == [{"23"}]


def test_build_XOR_within_AND():
    bpmn_model = _bpmn_model_with_XOR_within_AND()
    reachability_graph = bpmn_model.get_reachability_graph()
    n_gram_index = NGramIndex(reachability_graph, n_gram_size_limit=3)
    n_gram_index.build()
    # Check the maximum size of the explored n-grams is under the limit
    n_gram_sizes = [len(n_gram) for n_gram in n_gram_index.markings]
    assert max(n_gram_sizes) <= 3
    # Check number of n-grams is the expected
    assert len(n_gram_index.markings) == 9 + 6 + 24 + 24 + 48
    # Check 1-grams (9)
    assert n_gram_index.get_marking_state([NGramIndex.TRACE_START]) == [{"1"}]
    assert n_gram_index.get_marking_state(["A"]) == [{"5", "6", "7"}]
    assert _prepare(
        n_gram_index.get_marking_state(["B"])
    ) == _prepare([{"32", "6", "7"}, {"32", "33", "7"}, {"32", "6", "34"}, {"36"}])
    assert _prepare(
        n_gram_index.get_marking_state(["C"])
    ) == _prepare([{"32", "6", "7"}, {"32", "33", "7"}, {"32", "6", "34"}, {"36"}])
    assert _prepare(
        n_gram_index.get_marking_state(["D"])
    ) == _prepare([{"5", "33", "7"}, {"32", "33", "7"}, {"5", "33", "34"}, {"36"}])
    assert _prepare(
        n_gram_index.get_marking_state(["E"])
    ) == _prepare([{"5", "33", "7"}, {"32", "33", "7"}, {"5", "33", "34"}, {"36"}])
    assert _prepare(
        n_gram_index.get_marking_state(["F"])
    ) == _prepare([{"5", "6", "34"}, {"32", "6", "34"}, {"5", "33", "34"}, {"36"}])
    assert _prepare(
        n_gram_index.get_marking_state(["G"])
    ) == _prepare([{"5", "6", "34"}, {"32", "6", "34"}, {"5", "33", "34"}, {"36"}])
    assert n_gram_index.get_marking_state(["H"]) == [{"38"}]
    # 2-gram at the start of the trace (6)
    assert n_gram_index.get_marking_state(["A", "B"]) == [{"32", "6", "7"}]
    assert n_gram_index.get_marking_state(["A", "C"]) == [{"32", "6", "7"}]
    assert n_gram_index.get_marking_state(["A", "D"]) == [{"5", "33", "7"}]
    assert n_gram_index.get_marking_state(["A", "E"]) == [{"5", "33", "7"}]
    assert n_gram_index.get_marking_state(["A", "F"]) == [{"5", "6", "34"}]
    assert n_gram_index.get_marking_state(["A", "G"]) == [{"5", "6", "34"}]
    # 2-gram in the middle of the AND execution (24)
    assert _prepare(n_gram_index.get_marking_state(["B", "E"])) == _prepare([{"32", "33", "7"}, {"36"}])
    assert _prepare(n_gram_index.get_marking_state(["C", "F"])) == _prepare([{"32", "6", "34"}, {"36"}])
    assert _prepare(n_gram_index.get_marking_state(["D", "B"])) == _prepare([{"32", "33", "7"}, {"36"}])
    assert _prepare(n_gram_index.get_marking_state(["E", "G"])) == _prepare([{"5", "33", "34"}, {"36"}])
    assert _prepare(n_gram_index.get_marking_state(["F", "C"])) == _prepare([{"32", "6", "34"}, {"36"}])
    assert _prepare(n_gram_index.get_marking_state(["G", "D"])) == _prepare([{"5", "33", "34"}, {"36"}])
    # 3-gram at the start of the trace (24)
    assert n_gram_index.get_marking_state(["A", "B", "E"]) == [{"32", "33", "7"}]
    assert n_gram_index.get_marking_state(["A", "C", "F"]) == [{"32", "6", "34"}]
    assert n_gram_index.get_marking_state(["A", "D", "B"]) == [{"32", "33", "7"}]
    assert n_gram_index.get_marking_state(["A", "E", "G"]) == [{"5", "33", "34"}]
    assert n_gram_index.get_marking_state(["A", "F", "C"]) == [{"32", "6", "34"}]
    assert n_gram_index.get_marking_state(["A", "G", "D"]) == [{"5", "33", "34"}]
    # 3-gram in the middle of the AND execution (48)
    assert n_gram_index.get_marking_state(["F", "B", "E"]) == [{"36"}]
    assert n_gram_index.get_marking_state(["D", "C", "F"]) == [{"36"}]
    assert n_gram_index.get_marking_state(["F", "D", "B"]) == [{"36"}]
    assert n_gram_index.get_marking_state(["C", "E", "G"]) == [{"36"}]
    assert n_gram_index.get_marking_state(["D", "F", "C"]) == [{"36"}]
    assert n_gram_index.get_marking_state(["B", "G", "D"]) == [{"36"}]


def test_build_nested_XOR():
    bpmn_model = _bpmn_model_with_AND_and_nested_XOR()
    reachability_graph = bpmn_model.get_reachability_graph()
    n_gram_index = NGramIndex(reachability_graph, n_gram_size_limit=3)
    n_gram_index.build()
    # Check the maximum size of the explored n-grams is under the needed one
    n_gram_sizes = [len(n_gram) for n_gram in n_gram_index.markings]
    assert max(n_gram_sizes) <= 2
    # Check number of n-grams is the expected
    assert len(n_gram_index.markings) == 7 + 10
    # Check 1-gram markings (7)
    assert n_gram_index.get_marking_state([NGramIndex.TRACE_START]) == [{"1"}]
    assert n_gram_index.get_marking_state(["A"]) == [{"4", "6"}]
    assert _prepare(n_gram_index.get_marking_state(["B"])) == _prepare([{"21", "6"}, {"27"}])
    assert _prepare(n_gram_index.get_marking_state(["C"])) == _prepare([{"21", "6"}, {"27"}])
    assert _prepare(n_gram_index.get_marking_state(["D"])) == _prepare([{"21", "6"}, {"27"}])
    assert _prepare(n_gram_index.get_marking_state(["E"])) == _prepare([{"4", "20"}, {"27"}])
    assert n_gram_index.get_marking_state(["F"]) == [{"24"}]
    # Check 2-gram markings (10)
    assert n_gram_index.get_marking_state(["A", "B"]) == [{"21", "6"}]
    assert n_gram_index.get_marking_state(["A", "C"]) == [{"21", "6"}]
    assert n_gram_index.get_marking_state(["A", "D"]) == [{"21", "6"}]
    assert n_gram_index.get_marking_state(["A", "E"]) == [{"4", "20"}]
    assert n_gram_index.get_marking_state(["B", "E"]) == [{"27"}]
    assert n_gram_index.get_marking_state(["C", "E"]) == [{"27"}]
    assert n_gram_index.get_marking_state(["D", "E"]) == [{"27"}]
    assert n_gram_index.get_marking_state(["E", "B"]) == [{"27"}]
    assert n_gram_index.get_marking_state(["E", "C"]) == [{"27"}]
    assert n_gram_index.get_marking_state(["E", "D"]) == [{"27"}]


def test_build_loop_model():
    bpmn_model = _bpmn_model_with_loop_inside_AND()
    reachability_graph = bpmn_model.get_reachability_graph()
    n_gram_index = NGramIndex(reachability_graph, n_gram_size_limit=3)
    n_gram_index.build()
    # Check the maximum size of the explored n-grams is under the needed one
    n_gram_sizes = [len(n_gram) for n_gram in n_gram_index.markings]
    assert max(n_gram_sizes) <= 3
    # Check number of n-grams is the expected
    assert len(n_gram_index.markings) == 5 + 5 + 3
    # Check 1-gram markings (5)
    assert n_gram_index.get_marking_state([NGramIndex.TRACE_START]) == [{"1"}]
    assert n_gram_index.get_marking_state(["A"]) == [{"9", "6"}]
    assert _prepare(n_gram_index.get_marking_state(["B"])) == _prepare([{"11", "6"}, {"11", "15"}])
    assert _prepare(n_gram_index.get_marking_state(["C"])) == _prepare([{"9", "15"}, {"11", "15"}])
    assert n_gram_index.get_marking_state(["D"]) == [{"19"}]
    # Check 2-gram markings (5)
    assert n_gram_index.get_marking_state(["A", "B"]) == [{"11", "6"}]
    assert _prepare(n_gram_index.get_marking_state(["B", "B"])) == _prepare([{"11", "6"}, {"11", "15"}])
    assert n_gram_index.get_marking_state(["C", "B"]) == [{"11", "15"}]
    assert n_gram_index.get_marking_state(["A", "C"]) == [{"9", "15"}]
    assert n_gram_index.get_marking_state(["B", "C"]) == [{"11", "15"}]
    # Check 3-gram markings (1)
    assert n_gram_index.get_marking_state(["A", "B", "B"]) == [{"11", "6"}]
    assert _prepare(n_gram_index.get_marking_state(["B", "B", "B"])) == _prepare([{"11", "6"}, {"11", "15"}])
    assert n_gram_index.get_marking_state(["C", "B", "B"]) == [{"11", "15"}]


def test_build_double_loop_model():
    bpmn_model = _bpmn_model_with_two_loops_inside_AND_followed_by_XOR_within_AND()
    reachability_graph = bpmn_model.get_reachability_graph()
    n_gram_index = NGramIndex(reachability_graph, n_gram_size_limit=3)
    n_gram_index.build()
    # Check the maximum size of the explored n-grams is under the needed one
    n_gram_sizes = [len(n_gram) for n_gram in n_gram_index.markings]
    assert max(n_gram_sizes) <= 3
    # Check number of n-grams is the expected
    assert len(n_gram_index.markings) == 9 + 22 + 6
    # Check 1-gram markings (9)
    assert n_gram_index.get_marking_state([NGramIndex.TRACE_START]) == [{"1"}]
    assert n_gram_index.get_marking_state(["A"]) == [{"9", "10"}]
    assert _prepare(n_gram_index.get_marking_state(["B"])) == _prepare([{"13", "10"}, {"13", "14"}])
    assert _prepare(n_gram_index.get_marking_state(["C"])) == _prepare([{"9", "14"}, {"13", "14"}])
    assert n_gram_index.get_marking_state(["D"]) == [{"40", "25"}, {"43"}]
    assert n_gram_index.get_marking_state(["E"]) == [{"40", "25"}, {"43"}]
    assert n_gram_index.get_marking_state(["F"]) == [{"24", "41"}, {"43"}]
    assert n_gram_index.get_marking_state(["G"]) == [{"24", "41"}, {"43"}]
    assert n_gram_index.get_marking_state(["H"]) == [{"45"}]
    # Check 2-gram markings (22)
    assert n_gram_index.get_marking_state(["A", "B"]) == [{"13", "10"}]
    assert _prepare(n_gram_index.get_marking_state(["B", "B"])) == _prepare([{"13", "10"}, {"13", "14"}])
    assert n_gram_index.get_marking_state(["C", "B"]) == [{"13", "14"}]
    assert n_gram_index.get_marking_state(["A", "C"]) == [{"9", "14"}]
    assert n_gram_index.get_marking_state(["B", "C"]) == [{"13", "14"}]
    assert _prepare(n_gram_index.get_marking_state(["C", "C"])) == _prepare([{"9", "14"}, {"13", "14"}])
    assert n_gram_index.get_marking_state(["B", "D"]) == [{"40", "25"}]
    assert n_gram_index.get_marking_state(["C", "D"]) == [{"40", "25"}]
    assert n_gram_index.get_marking_state(["F", "D"]) == [{"43"}]
    assert n_gram_index.get_marking_state(["G", "D"]) == [{"43"}]
    assert n_gram_index.get_marking_state(["B", "E"]) == [{"40", "25"}]
    assert n_gram_index.get_marking_state(["C", "E"]) == [{"40", "25"}]
    assert n_gram_index.get_marking_state(["F", "E"]) == [{"43"}]
    assert n_gram_index.get_marking_state(["G", "E"]) == [{"43"}]
    assert n_gram_index.get_marking_state(["B", "F"]) == [{"24", "41"}]
    assert n_gram_index.get_marking_state(["C", "F"]) == [{"24", "41"}]
    assert n_gram_index.get_marking_state(["D", "F"]) == [{"43"}]
    assert n_gram_index.get_marking_state(["E", "F"]) == [{"43"}]
    assert n_gram_index.get_marking_state(["B", "G"]) == [{"24", "41"}]
    assert n_gram_index.get_marking_state(["C", "G"]) == [{"24", "41"}]
    assert n_gram_index.get_marking_state(["D", "G"]) == [{"43"}]
    assert n_gram_index.get_marking_state(["E", "G"]) == [{"43"}]
    # Check 3-gram markings (6)
    assert n_gram_index.get_marking_state(["A", "B", "B"]) == [{"13", "10"}]
    assert _prepare(n_gram_index.get_marking_state(["B", "B", "B"])) == _prepare([{"13", "10"}, {"13", "14"}])
    assert n_gram_index.get_marking_state(["C", "B", "B"]) == [{"13", "14"}]
    assert n_gram_index.get_marking_state(["A", "C", "C"]) == [{"9", "14"}]
    assert _prepare(n_gram_index.get_marking_state(["C", "C", "C"])) == _prepare([{"9", "14"}, {"13", "14"}])
    assert n_gram_index.get_marking_state(["B", "C", "C"]) == [{"13", "14"}]


def test_build_triple_loop_model():
    bpmn_model = _bpmn_model_with_three_loops_inside_AND_two_of_them_inside_sub_AND()
    reachability_graph = bpmn_model.get_reachability_graph()
    n_gram_index = NGramIndex(reachability_graph, n_gram_size_limit=3)
    n_gram_index.build()
    # Check the maximum size of the explored n-grams is under the needed one
    n_gram_sizes = [len(n_gram) for n_gram in n_gram_index.markings]
    assert max(n_gram_sizes) <= 3
    # Check number of n-grams is the expected
    assert len(n_gram_index.markings) == 7 + 19 + 48
    # Check 1-gram markings (7)
    assert n_gram_index.get_marking_state([NGramIndex.TRACE_START]) == [{"3", "26"}]
    assert _prepare(n_gram_index.get_marking_state(["A"])) == _prepare([{"13", "14", "26"}, {"13", "14", "28"}])
    assert _prepare(n_gram_index.get_marking_state(["B"])) == _prepare(
        [{"17", "14", "26"}, {"17", "14", "28"}, {"17", "18", "26"}, {"17", "18", "28"}])
    assert _prepare(n_gram_index.get_marking_state(["C"])) == _prepare(
        [{"13", "18", "26"}, {"17", "18", "26"}, {"13", "18", "28"}, {"17", "18", "28"}])
    assert _prepare(n_gram_index.get_marking_state(["D"])) == _prepare(
        [{"3", "28"}, {"13", "14", "28"}, {"17", "14", "28"}, {"13", "18", "28"}, {"17", "18", "28"}, {"33", "28"}])
    assert _prepare(n_gram_index.get_marking_state(["E"])) == _prepare([{"33", "26"}, {"33", "28"}])
    assert n_gram_index.get_marking_state(["F"]) == [{"38"}]
    # Check 2-gram markings (19)
    assert n_gram_index.get_marking_state([NGramIndex.TRACE_START, "A"]) == [{"13", "14", "26"}]
    assert n_gram_index.get_marking_state(["D", "A"]) == [{"13", "14", "28"}]

    assert _prepare(n_gram_index.get_marking_state(["A", "B"])) == _prepare(
        [{"17", "14", "26"}, {"17", "14", "28"}])
    assert _prepare(n_gram_index.get_marking_state(["B", "B"])) == _prepare(
        [{"17", "14", "26"}, {"17", "14", "28"}, {"17", "18", "26"}, {"17", "18", "28"}])
    assert _prepare(n_gram_index.get_marking_state(["C", "B"])) == _prepare(
        [{"17", "18", "26"}, {"17", "18", "28"}])
    assert _prepare(n_gram_index.get_marking_state(["D", "B"])) == _prepare(
        [{"17", "14", "28"}, {"17", "18", "28"}])

    assert n_gram_index.get_marking_state([NGramIndex.TRACE_START, "D"]) == [{"3", "28"}]
    assert n_gram_index.get_marking_state(["A", "D"]) == [{"13", "14", "28"}]
    assert _prepare(n_gram_index.get_marking_state(["B", "D"])) == _prepare(
        [{"17", "14", "28"}, {"17", "18", "28"}])
    assert _prepare(n_gram_index.get_marking_state(["C", "D"])) == _prepare(
        [{"13", "18", "28"}, {"17", "18", "28"}])
    assert _prepare(n_gram_index.get_marking_state(["D", "D"])) == _prepare(
        [{"3", "28"}, {"13", "14", "28"}, {"17", "14", "28"}, {"13", "18", "28"}, {"17", "18", "28"}, {"33", "28"}])
    assert n_gram_index.get_marking_state(["E", "D"]) == [{"33", "28"}]

    assert _prepare(n_gram_index.get_marking_state(["B", "E"])) == _prepare([{"33", "26"}, {"33", "28"}])
    assert _prepare(n_gram_index.get_marking_state(["C", "E"])) == _prepare([{"33", "26"}, {"33", "28"}])
    assert n_gram_index.get_marking_state(["D", "E"]) == [{"33", "28"}]
    # Check 3-gram markings (48)
    assert n_gram_index.get_marking_state([NGramIndex.TRACE_START, "A", "B"]) == [{"17", "14", "26"}]
    assert n_gram_index.get_marking_state(["D", "A", "B"]) == [{"17", "14", "28"}]

    assert _prepare(n_gram_index.get_marking_state(["B", "B", "B"])) == _prepare(
        [{"17", "14", "26"}, {"17", "14", "28"}, {"17", "18", "26"}, {"17", "18", "28"}])
    assert _prepare(n_gram_index.get_marking_state(["C", "C", "C"])) == _prepare(
        [{"13", "18", "26"}, {"13", "18", "28"}, {"17", "18", "26"}, {"17", "18", "28"}])
    assert _prepare(n_gram_index.get_marking_state(["D", "D", "D"])) == _prepare(
        [{"3", "28"}, {"13", "14", "28"}, {"17", "14", "28"}, {"13", "18", "28"}, {"17", "18", "28"}, {"33", "28"}])


def test_get_best_marking_state_for():
    bpmn_model = _bpmn_model_with_three_loops_inside_AND_two_of_them_inside_sub_AND()
    reachability_graph = bpmn_model.get_reachability_graph()
    n_gram_index = NGramIndex(reachability_graph, n_gram_size_limit=3)
    n_gram_index.build()
    # Assert n-grams that get one single state with the 1-gram
    assert n_gram_index.get_best_marking_state_for(["F"]) == {"38"}
    assert n_gram_index.get_best_marking_state_for(["E", "F"]) == {"38"}
    assert n_gram_index.get_best_marking_state_for(["E", "D", "F"]) == {"38"}
    assert n_gram_index.get_best_marking_state_for([NGramIndex.TRACE_START]) == {"3", "26"}
    # Assert n-grams that get one single state with the 2-gram
    assert n_gram_index.get_best_marking_state_for(["D", "A"]) == {"13", "14", "28"}
    assert n_gram_index.get_best_marking_state_for(["E", "D", "A"]) == {"13", "14", "28"}
    assert n_gram_index.get_best_marking_state_for(["D", "E"]) == {"33", "28"}
    assert n_gram_index.get_best_marking_state_for(["A", "D", "E"]) == {"33", "28"}
    assert n_gram_index.get_best_marking_state_for(["A", "D"]) == {"13", "14", "28"}
    assert n_gram_index.get_best_marking_state_for(["D", "A", "D"]) == {"13", "14", "28"}
    # Assert n-grams that don't get deterministic even with the complete n-gram
    assert n_gram_index.get_best_marking_state_for(["C", "D"]) in [{"13", "18", "28"}, {"17", "18", "28"}]
    assert n_gram_index.get_best_marking_state_for(["A", "C", "D"]) in [{"13", "18", "28"}, {"17", "18", "28"}]
    assert n_gram_index.get_best_marking_state_for(["B", "B", "B"]) in [{"17", "14", "26"}, {"17", "14", "28"},
                                                                             {"17", "18", "26"}, {"17", "18", "28"}]
    assert n_gram_index.get_best_marking_state_for(["C", "B", "B", "B"]) in [{"17", "14", "26"},
                                                                                  {"17", "14", "28"},
                                                                                  {"17", "18", "26"},
                                                                                  {"17", "18", "28"}]


def test_input_output_simple(temp_file_path):
    # Compute simple n-gram index
    bpmn_model = _bpmn_model_with_AND_and_nested_XOR()
    reachability_graph = bpmn_model.get_reachability_graph()
    n_gram_index = NGramIndex(reachability_graph, n_gram_size_limit=5)
    n_gram_index.build()
    # Export to file
    n_gram_index.to_self_contained_map_file(temp_file_path)
    # Import from file
    read_n_gram_index = NGramIndex.from_self_contained_map_file(temp_file_path, reachability_graph)
    # Assert equality
    assert n_gram_index.markings == read_n_gram_index.markings


def test_input_output_complex(temp_file_path):
    # Compute complex n-gram index
    bpmn_model = _bpmn_model_with_three_loops_inside_AND_two_of_them_inside_sub_AND()
    reachability_graph = bpmn_model.get_reachability_graph()
    n_gram_index = NGramIndex(reachability_graph, n_gram_size_limit=5)
    n_gram_index.build()
    # Export to file
    n_gram_index.to_self_contained_map_file(temp_file_path)
    # Import from file
    read_n_gram_index = NGramIndex.from_self_contained_map_file(temp_file_path, reachability_graph)
    # Assert equality
    assert n_gram_index.markings == read_n_gram_index.markings
