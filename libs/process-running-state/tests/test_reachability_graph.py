from typing import List, Set, Tuple

import pytest

from process_running_state.reachability_graph import ReachabilityGraph
from test_reachability_graph_fixtures import _simple_reachability_graph, _non_deterministic_reachability_graph


def _prepare(markings: List[Set[str]]) -> Set[Tuple[str]]:
    return {tuple(sorted(marking)) for marking in markings}


def test_reachability_graph_model():
    graph = ReachabilityGraph()
    # Add some markings
    graph.add_marking({"a", "b", "c"})
    graph.add_marking({"a", "b"})
    graph.add_marking({"b", "c"})
    graph.add_marking({"a", "c"})
    # Assert size of stored elements
    assert len(graph.markings) == 4
    assert len(graph.edges) == 0
    assert len(graph.activity_to_edges) == 0
    # Add rest of markings
    graph.add_marking({"a"})
    graph.add_marking({"b"})
    graph.add_marking({"c"})
    graph.add_marking({"d"})
    # Add edges
    graph.add_edge("A", {"a", "b", "c"}, {"b", "c"})
    graph.add_edge("B", {"a", "b", "c"}, {"a", "c"})
    graph.add_edge("C", {"a", "b", "c"}, {"a", "b"})
    graph.add_edge("A", {"a", "b"}, {"b"})
    graph.add_edge("A", {"a", "c"}, {"c"})
    graph.add_edge("B", {"a", "b"}, {"a"})
    graph.add_edge("B", {"b", "c"}, {"c"})
    graph.add_edge("C", {"a", "c"}, {"a"})
    graph.add_edge("C", {"b", "c"}, {"b"})
    graph.add_edge("A", {"a"}, {"d"})
    graph.add_edge("B", {"b"}, {"d"})
    graph.add_edge("C", {"c"}, {"d"})
    # Assert size of stored elements
    assert len(graph.markings) == 8
    assert len(graph.edges) == 12
    assert len(graph.activity_to_edges) == 3
    assert len(graph.activity_to_edges["A"]) == 4
    # Assert incoming and outgoing arcs
    marking_b_key = graph.marking_to_key[tuple(sorted({"b"}))]
    marking_d_key = graph.marking_to_key[tuple(sorted({"d"}))]
    marking_ab_key = graph.marking_to_key[tuple(sorted({"a", "b"}))]
    marking_bc_key = graph.marking_to_key[tuple(sorted({"b", "c"}))]
    assert {
               graph.edges[edge_id]
               for edge_id in graph.incoming_edges[marking_b_key]
           } == {(marking_ab_key, marking_b_key), (marking_bc_key, marking_b_key)}
    assert {
               graph.edges[edge_id]
               for edge_id in graph.outgoing_edges[marking_b_key]
           } == {(marking_b_key, marking_d_key)}
    marking_abc_key = graph.marking_to_key[tuple(sorted({"a", "b", "c"}))]
    marking_c_key = graph.marking_to_key[tuple(sorted({"c"}))]
    assert {
               graph.edges[edge_id]
               for edge_id in graph.incoming_edges[marking_bc_key]
           } == {(marking_abc_key, marking_bc_key)}
    assert {
               graph.edges[edge_id]
               for edge_id in graph.outgoing_edges[marking_bc_key]
           } == {(marking_bc_key, marking_b_key), (marking_bc_key, marking_c_key)}
    # Try to add an existing repeated edge
    graph.add_edge("C", {"b", "c"}, {"b"})
    assert len(graph.edges) == 12
    # Try to add an existing marking
    graph.add_marking({"b", "c"})
    assert len(graph.markings) == 8


def test_get_markings_from_activity_sequence_simple():
    # Instantiate graph
    graph = _simple_reachability_graph()
    # Assert replayable sequences
    assert graph.get_markings_from_activity_sequence(["A", "B"]) == [{"11", "6"}]
    assert graph.get_markings_from_activity_sequence(["A", "B", "B"]) == [{"11", "6"}]
    assert graph.get_markings_from_activity_sequence(["A", "B", "C"]) == [{"11", "15"}]
    assert graph.get_markings_from_activity_sequence(["A", "B", "C", "B"]) == [{"11", "15"}]
    assert graph.get_markings_from_activity_sequence(["A", "B", "B", "C", "B", "B"]) == [{"11", "15"}]
    assert graph.get_markings_from_activity_sequence(["A", "B", "C", "B", "B", "D"]) == [{"19"}]
    assert graph.get_markings_from_activity_sequence(["A", "C", "B"]) == [{"11", "15"}]
    assert graph.get_markings_from_activity_sequence(["A", "C", "B", "B", "B"]) == [{"11", "15"}]
    assert graph.get_markings_from_activity_sequence(["A", "C", "B", "D"]) == [{"19"}]
    # Assert sequences raising errors
    with pytest.raises(RuntimeError):
        graph.get_markings_from_activity_sequence(["C"])
    with pytest.raises(RuntimeError):
        graph.get_markings_from_activity_sequence(["A", "B", "C", "C"])
    with pytest.raises(RuntimeError):
        graph.get_markings_from_activity_sequence(["A", "B", "D"])


def test_get_markings_from_activity_sequence_non_deterministic():
    # Instantiate graph
    graph = _non_deterministic_reachability_graph()
    # Assert replayable sequences
    assert graph.get_markings_from_activity_sequence(["A"]) == [{"10", "7"}]
    assert graph.get_markings_from_activity_sequence(["A", "B"]) == [{"12", "7"}]
    assert graph.get_markings_from_activity_sequence(["A", "B", "B"]) == [{"12", "7"}]
    assert graph.get_markings_from_activity_sequence(["A", "C", "B"]) == [{"12", "17"}]
    assert graph.get_markings_from_activity_sequence(["A", "C", "B", "C"]) == [{"10", "17"}]
    assert graph.get_markings_from_activity_sequence(["A", "C", "B", "C", "B", "D"]) == [{"24"}]
    assert _prepare(
        graph.get_markings_from_activity_sequence(["A", "B", "B", "C", "B"])
    ) == _prepare([{"12", "7"}, {"12", "17"}])
    assert _prepare(
        graph.get_markings_from_activity_sequence(["A", "B", "B", "C", "B", "B"])
    ) == _prepare([{"12", "7"}, {"12", "17"}])
    assert _prepare(
        graph.get_markings_from_activity_sequence(["A", "B", "B", "C", "B", "C"])
    ) == _prepare([{"12", "17"}, {"10", "17"}])
    assert graph.get_markings_from_activity_sequence(["A", "B", "C", "B", "D"]) == [{"24"}]
    # Assert sequences raising errors
    with pytest.raises(RuntimeError):
        graph.get_markings_from_activity_sequence(["C"])
    with pytest.raises(RuntimeError):
        graph.get_markings_from_activity_sequence(["A", "C", "C"])
    with pytest.raises(RuntimeError):
        graph.get_markings_from_activity_sequence(["A", "B", "C", "B", "A"])
    with pytest.raises(RuntimeError):
        graph.get_markings_from_activity_sequence(["A", "B", "C", "B", "D", "D"])


def test_transformation_to_tgf():
    # Instantiate graph
    graph = _simple_reachability_graph()
    # Transform into string
    tgf_string = graph.to_tgf_format()
    # Assert certain lines are in the string
    tgf_lines = tgf_string.splitlines()
    assert tgf_lines[0] == "0 {'1'}"
    assert tgf_lines[2] == "2 {'11', '6'}" or tgf_lines[2] == "2 {'6', '11'}"
    assert tgf_lines[4] == "4 {'11', '15'}" or tgf_lines[4] == "4 {'15', '11'}"
    assert tgf_lines[6] == "#"
    edge_lines = tgf_lines[7:]
    assert "0 1 A" in edge_lines
    assert "2 2 B" in edge_lines
    assert "1 3 C" in edge_lines
    assert "2 4 C" in edge_lines
    assert "4 5 D" in edge_lines
    # Convert back to graph
    read_graph = ReachabilityGraph.from_tgf_format(tgf_string)
    # Assert some components
    assert graph == read_graph
