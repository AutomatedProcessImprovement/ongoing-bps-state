from enum import Enum
from itertools import combinations
from typing import List, Set, Dict, Tuple, Optional

from process_running_state.reachability_graph import ReachabilityGraph


class BPMNNodeType(Enum):
    TASK = "TASK"
    START_EVENT = "START-EVENT"
    INTERMEDIATE_EVENT = "INTERMEDIATE-EVENT"
    END_EVENT = "END-EVENT"
    EXCLUSIVE_GATEWAY = "EXCLUSIVE-GATEWAY"
    INCLUSIVE_GATEWAY = "INCLUSIVE-GATEWAY"
    PARALLEL_GATEWAY = "PARALLEL-GATEWAY"
    UNDEFINED = "UNDEFINED"


class Node:
    def __init__(self, node_type: BPMNNodeType, node_id: str, node_name: str):
        self.id: str = node_id
        self.name: str = node_name
        self.type: BPMNNodeType = node_type
        self.incoming_flows: Set[str] = set()
        self.outgoing_flows: Set[str] = set()

    def is_split(self) -> bool:
        return len(self.outgoing_flows) > 1

    def is_join(self) -> bool:
        return len(self.incoming_flows) > 1

    def is_task(self) -> bool:
        return self.type == BPMNNodeType.TASK

    def is_event(self) -> bool:
        return self.type in [
            BPMNNodeType.START_EVENT,
            BPMNNodeType.INTERMEDIATE_EVENT,
            BPMNNodeType.END_EVENT,
        ]

    def is_start_event(self) -> bool:
        return self.type == BPMNNodeType.START_EVENT

    def is_intermediate_event(self) -> bool:
        return self.type == BPMNNodeType.INTERMEDIATE_EVENT

    def is_end_event(self) -> bool:
        return self.type == BPMNNodeType.END_EVENT

    def is_gateway(self) -> bool:
        return self.type in [
            BPMNNodeType.EXCLUSIVE_GATEWAY,
            BPMNNodeType.PARALLEL_GATEWAY,
            BPMNNodeType.INCLUSIVE_GATEWAY,
        ]

    def is_AND(self) -> bool:
        return self.type == BPMNNodeType.PARALLEL_GATEWAY

    def is_OR(self) -> bool:
        return self.type == BPMNNodeType.INCLUSIVE_GATEWAY

    def is_XOR(self) -> bool:
        return self.type == BPMNNodeType.EXCLUSIVE_GATEWAY


class Flow:
    def __init__(self, flow_id: str, flow_name: str, source_id: str, target_id: str):
        self.id: str = flow_id
        self.name: str = flow_name
        self.source: str = source_id
        self.target: str = target_id


class BPMNModel:

    def __init__(self):
        self.nodes: Set[Node] = set()
        self.id_to_node: Dict[str, Node] = dict()
        self.flows: Set[Flow] = set()
        self.id_to_flow: Dict[str, Flow] = dict()
        # Params for cached reachability graph search
        self._cached_search: bool = True
        self._advance_marking_cache = dict()
        self._advance_combination_cache = dict()

    def add_task(self, task_id: str, task_name: str):
        if task_id not in self.id_to_node:
            node = Node(BPMNNodeType.TASK, task_id, task_name)
            self.nodes |= {node}
            self.id_to_node[task_id] = node

    def add_event(self, event_type: BPMNNodeType, event_id: str, event_name: str):
        if event_id not in self.id_to_node:
            node = Node(event_type, event_id, event_name)
            if node.is_event():
                self.nodes |= {node}
                self.id_to_node[event_id] = node

    def add_gateway(self, gateway_type: BPMNNodeType, gateway_id: str, gateway_name: str):
        if gateway_type is BPMNNodeType.INCLUSIVE_GATEWAY:
            raise AttributeError("Current implementation does not support Inclusive Gateways!")
        if gateway_id not in self.id_to_node:
            node = Node(gateway_type, gateway_id, gateway_name)
            if node.is_gateway():
                self.nodes |= {node}
                self.id_to_node[gateway_id] = node

    def add_flow(self, flow_id: str, flow_name: str, source_id: str, target_id: str):
        if flow_id not in self.id_to_flow:
            source = self.id_to_node[source_id]
            target = self.id_to_node[target_id]
            # Check correctness
            if (source.is_task() or source.is_event()) and len(source.outgoing_flows) > 0:
                raise RuntimeError(
                    f"Error when adding flow (id: {flow_id}). Tasks and events must have one single outgoing flow arc."
                )
            if target.is_start_event():
                raise RuntimeError(
                    f"Error when adding flow (id: {flow_id}). Start events cannot have incoming flow arcs."
                )
            if source.is_end_event():
                raise RuntimeError(
                    f"Error when adding flow (id: {flow_id}). End events cannot have outgoing flow arcs."
                )
            # Add flow to model
            flow = Flow(flow_id, flow_name, source_id, target_id)
            self.flows |= {flow}
            self.id_to_flow[flow_id] = flow
            source.outgoing_flows |= {flow_id}
            target.incoming_flows |= {flow_id}

    def get_initial_marking(self) -> Set[str]:
        """
        Get initial marking, which corresponds to the execution of the start events of the process model.

        :return: marking (set of flows) corresponding to the initial marking of this BPMN model.
        """
        initial_marking = set()
        start_nodes = [node for node in self.nodes if node.is_start_event()]
        for node in start_nodes:
            initial_marking |= node.outgoing_flows  # It always has only one outgoing flow (at most)
        return initial_marking

    def simulate_execution(self, node_id: str, marking: Set[str]) -> List[Set[str]]:
        """
        Simulate the execution of [node_id], if possible, given the current [marking], and return the possible markings
        result of such execution.

        :param node_id: Identifier of the node to execute.
        :param marking: Current marking to simulate the execution over it.
        :return: when it is possible to execute [node_id], list with the different markings result of such execution,
        otherwise, return empty list.
        """
        node = self.id_to_node[node_id]
        if node.is_task() or node.is_event():
            # Task/Event: consume active incoming flow and enable the outgoing flow
            active_incoming_flows = node.incoming_flows & marking
            if len(active_incoming_flows) > 1:
                print(f"Warning! Node '{node_id}' has more than one incoming flow enabled (consuming only one).")
            if len(active_incoming_flows) > 0:
                consumed_flow = active_incoming_flows.pop()
                new_marking = marking - {consumed_flow}
                return [new_marking | node.outgoing_flows]
        elif node.is_XOR():
            # Exclusive gateway: consume active incoming flow and enable one of the outgoing flows
            active_incoming_flows = node.incoming_flows & marking
            if len(active_incoming_flows) > 1:
                print(f"Warning! ExclGateway '{node_id}' has more than one incoming flow enabled (consuming only one).")
            if len(active_incoming_flows) > 0:
                consumed_flow = active_incoming_flows.pop()
                new_marking = marking - {consumed_flow}
                return [new_marking | {outgoing_flow} for outgoing_flow in node.outgoing_flows]
        elif node.is_AND():
            # Parallel gateway: consume all incoming and enable all outgoing
            if node.incoming_flows.issubset(marking):
                new_marking = marking - node.incoming_flows | node.outgoing_flows
                return [new_marking]
        elif node.is_OR():
            # Inclusive gateway: consume all active incoming edges and enable all combinations of outgoing
            active_incoming_flows = node.incoming_flows & marking
            if len(active_incoming_flows) > 0:
                new_marking = marking - active_incoming_flows
                return [
                    new_marking | outgoing_flows
                    for outgoing_flows in _powerset(node.outgoing_flows)
                    if len(outgoing_flows) > 0
                ]
        # Unknown element or unable to execute
        return []

    def get_enabled_nodes(self, marking: Set[str]) -> Set[str]:
        """
        Compute the set of enabled nodes (excluding start/end events) given the current [marking]. A node (task,
        gateway, or event) is considered to be enabled when it can be fired.

        :param marking: marking considered as reference to compute the enabled nodes.
        :return: a set with the IDs of the enabled nodes (no start or end events).
        """
        return {
            node.id
            for node in self.nodes
            if (not node.is_start_event() and not node.is_end_event()) and (
                    (node.is_AND() and node.incoming_flows.issubset(marking)) or
                    (not node.is_AND() and len(node.incoming_flows & marking) > 0)
            )
        }

    def get_enabled_tasks_events(self, marking: Set[str]) -> Set[str]:
        """
        Compute the set of enabled tasks or events (excluding start/end events) given the current [marking].

        :param marking: marking considered as reference to compute the enabled nodes.
        :return: a set with the IDs of the enabled tasks/events (no start or end events).
        """
        return {
            node.id
            for node in self.nodes
            if ((node.is_task() or node.is_event()) and
                (not node.is_start_event() and not node.is_end_event()) and
                len(node.incoming_flows & marking) > 0)
        }

    def advance_marking_until_decision_point(self, marking: Set[str]) -> Set[str]:
        """
        Advance the current marking (every branch in it) as much as possible without executing any task, event, or
        decision point, i.e., execute AND-split and all join gateways until there are none enabled.

        :param marking: marking to consider as starting point to perform the advance operation.
        :return: marking after (recursively) executing all non-decision-point gateways (AND-split, AND-join, XOR-join,
        OR-join).
        """
        advanced_marking = marking.copy()
        # Get enabled gateways (AND-split, AND-join, XOR-join, OR-join)
        enabled_gateways = [
            node_id
            for node_id in self.get_enabled_nodes(advanced_marking) if
            self.id_to_node[node_id].is_gateway() and
            (self.id_to_node[node_id].is_AND() or not self.id_to_node[node_id].is_split())
        ]
        # Run propagation until no more gateways (AND-split, AND-join, XOR-join, OR-join) can be fired
        while len(enabled_gateways) > 0:
            # Execute one of the enabled gateways and save result for next iteration
            [advanced_marking] = self.simulate_execution(enabled_gateways[0], advanced_marking)
            # Get enabled gateways (exclude XOR-splits & OR-splits)
            enabled_gateways = [
                node_id
                for node_id in self.get_enabled_nodes(advanced_marking) if
                self.id_to_node[node_id].is_gateway() and
                (self.id_to_node[node_id].is_AND() or not self.id_to_node[node_id].is_split())
            ]
        # Return final set
        return advanced_marking

    def advance_full_marking(
            self,
            marking: Set[str],
            explored_markings: Optional[Set[Tuple[str]]] = None,
    ) -> List[Tuple[str, Set[str]]]:
        """
        Advance the current marking as much as possible without executing any task, i.e., execute gateways until there
        are none enabled. If there are multiple (parallel) branches, first advance in each of them individually, and
        only advance in more than one branch if needed to trigger an AND-join and advance further. For example, if the
        marking contains three enabled branches, but one of them cannot advance to close the AND-join, the result will
        be the markings after individually advancing each branch.

        :param marking: marking to consider as starting point to perform the advance operation.
        :param explored_markings: if recursive call, set of previously explored markings to avoid infinite loop.
        :return: list of tuples with the ID of an enabled task/event as first element and the advanced marking that
        enabled it as second element.
        """
        # Instantiate list for advanced markings
        tuples_final_markings = set()
        # First advance all branches at the same time until tasks, events, or decision points (XOR-split/OR-split)
        advanced_marking = self.advance_marking_until_decision_point(marking)
        # Advance all branches together (getting all combinations of advancements)
        tuples_fully_advanced_markings = self._advance_marking(advanced_marking, explored_markings)
        # Save only advanced marking that enabled new tasks/events
        for enabled_node_id, fully_advanced_marking in tuples_fully_advanced_markings:
            # Try to rollback the advancements in other branches as much as possible
            rollbacked_marking = self._try_rollback(fully_advanced_marking, advanced_marking, enabled_node_id)
            # Save rollbacked marking
            tuples_final_markings |= {(enabled_node_id, tuple(sorted(rollbacked_marking)))}
        # Return final markings (if none of them enabled any new tasks/events return original marking)
        return [(enabled_node_id, set(final_marking)) for enabled_node_id, final_marking in tuples_final_markings]

    def _advance_marking(
            self,
            marking: Set[str],
            explored_markings: Optional[Set[Tuple[str]]] = None,
    ) -> List[Tuple[str, Set[str]]]:
        """
        Advance the current marking as much as possible without executing any task, i.e., execute gateways until there
        are none enabled.

        When traversing AND-split or OR-split gateways, the process has to start again (recursion) to consider the
        possibility of many branches with decision points (we do not want to traverse all of them creating all
        possible combinations, but branch by branch).

        :param marking: marking to consider as starting point to perform the advance operation.
        :param explored_markings: if recursive call, set of previously explored markings to avoid infinite loop.
        :return: list of tuples with the ID of the enabled task/event as first element, and the advanced marking that
        enabled it as second element.
        """
        # If result in cache, retrieve, otherwise compute
        marking_key = tuple(sorted(marking))
        if self._cached_search and marking_key in self._advance_marking_cache:
            tuples_final_markings = self._advance_marking_cache[marking_key]
        else:
            # Initialize breath-first search list
            current_marking_stack = [marking]
            explored_markings = set() if explored_markings is None else explored_markings
            set_tuples_final_markings = set()
            # Run propagation until no more gateways can be fired
            while current_marking_stack:
                next_marking_stack = []
                # For each marking
                for current_marking in current_marking_stack:
                    # If it hasn't been explored
                    current_marking_key = tuple(sorted(current_marking))
                    if current_marking_key not in explored_markings:
                        # Add it to explored
                        explored_markings.add(current_marking_key)
                        # Get enabled gateways
                        enabled_gateways = [
                            node_id
                            for node_id in self.get_enabled_nodes(current_marking)
                            if self.id_to_node[node_id].is_gateway()
                        ]
                        # If no enabled gateways, save fully advanced marking
                        if len(enabled_gateways) == 0:
                            set_tuples_final_markings |= {
                                (enabled_node_id, tuple(sorted(current_marking)))
                                for enabled_node_id in self.get_enabled_nodes(current_marking)
                            }
                        else:
                            # Otherwise, execute one of the enabled gateways and save result for next iteration
                            gateway_id = enabled_gateways.pop()
                            gateway = self.id_to_node[gateway_id]
                            if (gateway.is_AND() or gateway.is_OR()) and gateway.is_split():
                                # AND-split/OR-split: traverse it
                                advanced_markings = self.simulate_execution(gateway_id, current_marking)
                                # For each advanced markings (after gateway split)
                                for advanced_marking in advanced_markings:
                                    # Save advancements that were needed to enable new activities
                                    set_tuples_final_markings |= {
                                        (enabled_node_id, tuple(sorted(fully_advanced_marking)))
                                        for enabled_node_id, fully_advanced_marking
                                        in self.advance_full_marking(advanced_marking, explored_markings)
                                    }
                            else:
                                # JOINs or XOR-split, execute and continue with advancement
                                next_marking_stack += self.simulate_execution(gateway_id, current_marking)
                # Update new marking stack
                current_marking_stack = next_marking_stack
            # Remove redundant tuples (str, Set[str])
            tuples_final_markings = [
                (node_id, set(final_marking))
                for node_id, final_marking in set_tuples_final_markings
            ]
            # Save if using cache
            if self._cached_search:
                self._advance_marking_cache[marking_key] = tuples_final_markings
        # Return final set
        return tuples_final_markings

    def _try_rollback(self, advanced_marking: Set[str], marking: Set[str], enabled_node_id: str) -> Set[str]:
        """
        Given an advanced marking and the node for which it advanced, try to rollback the advancement of as much
        branches as possible while keeping the node enabled. In this way, the marking still enables the desired node,
        and all the other branches remain as if they did not advance, only the required ones to enabled the desired
        node remain advanced.

        For example, imagine marking={1,2}, {1} advances to {3} and {4}, and {2} advances to {5} and {6}.
        Then, advanced_markings=[{3,5},{3,6},{4,5},{4,6}], and the objective is to rollback first the advancements of
        {1}, and then of {2}, so the result is [{1,5},{1,6},{3,2},{4,2}].

        For this, the method has to find the smaller combination of branches that, when advanced isolated, enable the
        desired node. Then, identify the remaining branches, advance them individually, and rollback such advancement.
        In this way, we ensure that when many parallel branches need to join (in an AND-join), they are advanced
        together.

        :param advanced_marking: advanced marking result of advancing the branches in [marking] as much as possible.
        :param marking: marking considered as starting point.
        :param enabled_node_id: identifier of the node that is enabled and must remain enabled.
        :return: rollbacked marking.
        """
        rollbacked_marking = set()
        # Retrieve edge enabling current activity
        enabling_flows = self.id_to_node[enabled_node_id].incoming_flows & advanced_marking
        assert len(enabling_flows) == 1, f"Many enabled flows ({enabling_flows}) for one task/event ({enabled_node_id})"
        enabling_flow_id = enabling_flows.pop()
        # Generate all possible branch combinations to explore individually
        branch_combinations = [
            combination
            for combination in _powerset(marking)
            if combination != marking
        ]
        branch_combinations.sort(key=len)  # Sort ascending to find the smaller one first
        # Identify the combination of branches needed for the enabling branch to advance
        advanced_combination = set(marking)  # If no other smaller combination found, all branches were needed
        found = False
        for branch_combination in branch_combinations:
            if not found:
                # Advance with this branch combination
                for advanced_marking_with_branch_combination in self._advance_combination(branch_combination):
                    # If the advancement reached the enabling flow
                    reached_enabling_flow = enabling_flow_id in advanced_marking_with_branch_combination
                    # and the advanced marking with these branch combination is all in the advanced marking
                    advanced_is_subset = advanced_marking_with_branch_combination.issubset(advanced_marking)
                    if not found and reached_enabling_flow and advanced_is_subset:
                        # All these branches were needed to advance current one, save to not rollback them
                        found = True
                        advanced_combination = branch_combination
        # Rollback the advancements that were also reached when advancing the other branches
        other_branches = marking - advanced_combination
        rollbacked = False
        if len(other_branches) > 0:
            for advanced_marking_other_branches in self._advance_combination(other_branches):
                if not rollbacked and advanced_marking_other_branches.issubset(advanced_marking):
                    # This advancement is independent of the current branch, rollback it
                    rollbacked = True
                    rollbacked_marking = advanced_marking - advanced_marking_other_branches | other_branches
            if not rollbacked:
                rollbacked_marking = advanced_marking
        else:
            # If it was not rollbacked (i.e., all branches needed to advance until that point), keep it
            rollbacked_marking = advanced_marking
        # Return final markings
        return rollbacked_marking

    def _advance_combination(self, combination: Set[str]) -> List[Set[str]]:
        """
        Advance a combination of branches as much as possible, executing all enabled gateways, generating all
        combinations of advanced branches until no more gateways are enabled.

        :param combination: marking to consider as starting point to perform the advance operation.
        :return: list with the different markings result of such advancement.
        """
        # If result in cache, retrieve, otherwise compute
        combination_key = tuple(sorted(combination))
        if self._cached_search and combination_key in self._advance_combination_cache:
            final_markings = self._advance_combination_cache[combination_key]
        else:
            # Initialize breath-first search list
            current_marking_stack = [combination]
            explored_markings = set()
            final_markings = []
            # Run propagation until no more gateways can be fired
            while current_marking_stack:
                next_marking_stack = []
                # For each marking
                for current_marking in current_marking_stack:
                    # If it hasn't been explored
                    current_marking_key = tuple(sorted(current_marking))
                    if current_marking_key not in explored_markings:
                        # Add it to explored
                        explored_markings.add(current_marking_key)
                        # Get enabled gateways
                        enabled_gateways = [
                            node_id
                            for node_id in self.get_enabled_nodes(current_marking) if
                            self.id_to_node[node_id].is_gateway()
                        ]
                        # If no enabled gateways, save fully advanced marking
                        if len(enabled_gateways) == 0:
                            final_markings += [current_marking]
                        else:
                            # Otherwise, execute one of the enabled gateways and save result for next iteration
                            gateway_id = enabled_gateways.pop()
                            next_marking_stack += self.simulate_execution(gateway_id, current_marking)
                # Update new marking stack
                current_marking_stack = next_marking_stack
            # Save if using cache
            if self._cached_search:
                self._advance_combination_cache[combination_key] = final_markings
        # Return final set
        return final_markings

    def get_reachability_graph(self, cached_search: bool = True) -> ReachabilityGraph:
        """
        Compute the reachability graph of this BPMN model. Each marking in the reachability graph contains the enabled
        flows of that state, and corresponds to a state of the process where the only enabled elements are tasks,
        events, and decision points (XOR-split/OR-split).

        :return: the reachability graph of this BPMN model.
        """
        self._cached_search = cached_search
        self._advance_marking_cache = dict()
        self._advance_combination_cache = dict()
        # Get initial BPMN marking and instantiate reachability graph
        initial_marking = self.get_initial_marking()
        initial_advanced_marking = self.advance_marking_until_decision_point(initial_marking)
        graph = ReachabilityGraph()
        graph.add_marking(initial_advanced_marking, is_initial=True)
        # Start exploration, for each "reference" marking, simulate in its corresponding advanced markings
        marking_stack = [initial_advanced_marking]
        explored_markings = set()
        while len(marking_stack) > 0:
            # Retrieve current markings
            current_marking = marking_stack.pop()  # This marking is already advanced to decision points
            # If this marking hasn't been explored (reference marking + advanced marking)
            exploration_key = tuple(sorted(current_marking))
            if exploration_key not in explored_markings:
                # Add it to explored
                explored_markings.add(exploration_key)
                # Advance the current marking, executing enabled gateways, obtaining:
                #   An activity enabled by the advancement
                #   The advanced marking needed to execute the activity
                tuples_advanced_markings = self.advance_full_marking(current_marking)
                # For each pair of enabled activity and advanced marking that enables it
                for enabled_node_id, advanced_marking in tuples_advanced_markings:
                    enabled_node = self.id_to_node[enabled_node_id]
                    # Fire task/event (always returns 1 marking)
                    [new_marking] = self.simulate_execution(enabled_node_id, advanced_marking)
                    # Advance the marking as much as possible without executing decision points (XOR-split/OR-split)
                    new_advanced_marking = self.advance_marking_until_decision_point(new_marking)
                    # Update reachability graph
                    graph.add_marking(new_advanced_marking)  # Add reference marking to graph
                    graph.add_edge(enabled_node.name, current_marking, new_advanced_marking)
                    # Save to continue exploring it
                    marking_stack += [new_advanced_marking]
        # Return reachability graph
        return graph


def _powerset(iterable):
    # powerset({1,2,3}) --> {1} {2} {3} {1,2} {1,3} {2,3} {1,2,3}
    return [
        set(combination)
        for r in range(len(iterable) + 1)
        for combination in combinations(iterable, r)
        if len(combination) > 0
    ]
