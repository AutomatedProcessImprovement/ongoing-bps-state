from process_running_state.reachability_graph import ReachabilityGraph


def _simple_reachability_graph() -> ReachabilityGraph:
    graph = ReachabilityGraph()
    graph.add_marking({"1"}, True)
    graph.add_marking({"9", "6"})
    graph.add_marking({"11", "6"})
    graph.add_marking({"9", "15"})
    graph.add_marking({"11", "15"})
    graph.add_marking({"19"})
    graph.add_edge("A", {"1"}, {"9", "6"})
    graph.add_edge("B", {"9", "6"}, {"11", "6"})
    graph.add_edge("B", {"11", "6"}, {"11", "6"})
    graph.add_edge("B", {"9", "15"}, {"11", "15"})
    graph.add_edge("B", {"11", "15"}, {"11", "15"})
    graph.add_edge("C", {"9", "6"}, {"9", "15"})
    graph.add_edge("C", {"11", "6"}, {"11", "15"})
    graph.add_edge("D", {"11", "15"}, {"19"})
    return graph


def _non_deterministic_reachability_graph() -> ReachabilityGraph:
    graph = ReachabilityGraph()
    graph.add_marking({"1"}, True)
    graph.add_marking({"10", "7"})
    graph.add_marking({"12", "7"})
    graph.add_marking({"10", "17"})
    graph.add_marking({"12", "17"})
    graph.add_marking({"24"})
    graph.add_edge("A", {"1"}, {"10", "7"})
    graph.add_edge("B", {"10", "7"}, {"12", "7"})
    graph.add_edge("B", {"12", "7"}, {"12", "7"})
    graph.add_edge("B", {"10", "17"}, {"12", "17"})
    graph.add_edge("B", {"12", "17"}, {"12", "7"})
    graph.add_edge("B", {"12", "17"}, {"12", "17"})
    graph.add_edge("C", {"10", "7"}, {"10", "17"})
    graph.add_edge("C", {"12", "17"}, {"10", "17"})
    graph.add_edge("C", {"12", "7"}, {"12", "17"})
    graph.add_edge("D", {"12", "17"}, {"24"})
    return graph
