import ast
import random
from pathlib import Path

import pandas as pd
from ongoing_process_state.bpmn_model import BPMNModel
from ongoing_process_state.reachability_graph import ReachabilityGraph
from pix_framework.io.event_log import EventLogIDs
from typing import List, Dict, Set, Tuple, Optional

sim_log_ids = EventLogIDs(
    case="case_id",
    activity="activity",
    enabled_time="enable_time",
    start_time="start_time",
    end_time="end_time",
    resource="resource",
)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Function to compute the reachability graph of a BPMN model, considering   #
# the START and COMPLETE events of each node, and all markings in the model #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def compute_complete_reachability_graph(model: BPMNModel) -> ReachabilityGraph:
    # Get initial BPMN marking and instantiate reachability graph
    initial_marking = {node.id for node in model.nodes if node.is_start_event()}
    graph = ReachabilityGraph()
    graph.add_marking(initial_marking, is_initial=True)
    # Start exploration
    marking_stack = [initial_marking]
    explored_markings = set()
    while len(marking_stack) > 0:
        # Retrieve current marking
        current_marking = marking_stack.pop()
        # If this marking hasn't been explored
        exploration_key = tuple(sorted(current_marking))
        if exploration_key not in explored_markings:
            # Add it to explored
            explored_markings.add(exploration_key)
            # Retrieve all enabled nodes and process them
            enabled_node_ids = {
                node.id
                for node in model.nodes
                if not node.is_start_event() and (
                        (node.is_AND() and node.incoming_flows <= current_marking) or
                        (not node.is_AND() and len(node.incoming_flows & current_marking) > 0)
                )
            }
            for enabled_node_id in enabled_node_ids:
                enabled_node = model.id_to_node[enabled_node_id]
                # If task or event, process their START
                if enabled_node.is_task() or enabled_node.is_event():
                    # Update marking (entering the node)
                    new_marking = current_marking - enabled_node.incoming_flows | {enabled_node_id}
                    # Update reachability graph
                    graph.add_marking(new_marking)
                    graph.add_edge((enabled_node.name, "START"), current_marking, new_marking)
                    # Save to continue exploring it
                    marking_stack += [new_marking]
                elif enabled_node.is_gateway():  # If gateway, process traversing at once
                    # Update marking (traversing gateway)
                    for new_marking in model.simulate_execution(enabled_node_id, current_marking):
                        # Update reachability graph
                        graph.add_marking(new_marking)
                        graph.add_edge((enabled_node.name, "GATEWAY"), current_marking, new_marking)
                        # Save to continue exploring it
                        marking_stack += [new_marking]
            # Retrieve all ongoing tasks&events and complete their execution
            ongoing_node_ids = {node_id for node_id in current_marking if node_id in model.id_to_node}
            for ongoing_node_id in ongoing_node_ids:
                ongoing_node = model.id_to_node[ongoing_node_id]
                # If task or event, process their COMPLETE
                if ongoing_node.is_task() or ongoing_node.is_event():
                    # Update marking (leaving the node)
                    new_marking = current_marking - {ongoing_node_id} | ongoing_node.outgoing_flows
                    # Update reachability graph
                    graph.add_marking(new_marking)
                    graph.add_edge((ongoing_node.name, "COMPLETE"), current_marking, new_marking)
                    # Save to continue exploring it
                    marking_stack += [new_marking]
    # Return reachability graph
    return graph


def read_reachability_graph(reachability_graph_path: Path, names_are_tuples: bool = False) -> ReachabilityGraph:
    with open(reachability_graph_path, "r") as graph_file:
        # Read graph
        reachability_graph = ReachabilityGraph.from_tgf_format(graph_file.read())
        if names_are_tuples:
            # Tuples are read as str, convert them to be tuples
            reachability_graph.activity_to_edges = {
                ast.literal_eval(label): reachability_graph.activity_to_edges[label]
                for label in reachability_graph.activity_to_edges
            }
            reachability_graph.edge_to_activity = {
                edge_id: ast.literal_eval(reachability_graph.edge_to_activity[edge_id])
                for edge_id in reachability_graph.edge_to_activity
            }
    return reachability_graph


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Function to sort the events of a trace (enable, start, complete) by #
# their timestamp, keeping the original order when similar timestamps #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def sorted_trace_events(trace: pd.DataFrame, start_event: str, end_event: str) -> List[dict]:
    sorted_events = []
    for _, event in trace.iterrows():
        if event[sim_log_ids.activity] == start_event:
            # Process case arrival
            processed_event = {
                'case_id': event[sim_log_ids.case],
                'lifecycle': "CASE_ARRIVAL",
                'timestamp': event[sim_log_ids.enabled_time],
                'node_name': event[sim_log_ids.activity]
            }
            add_to_sorted_events(sorted_events, processed_event)
        elif event[sim_log_ids.activity] == end_event:
            # Process case end
            processed_event = {
                'case_id': event[sim_log_ids.case],
                'lifecycle': "CASE_END",
                'timestamp': event[sim_log_ids.enabled_time],
                'node_name': event[sim_log_ids.activity]
            }
            add_to_sorted_events(sorted_events, processed_event)
            pass
        else:
            # Process enablement
            if not pd.isna(event[sim_log_ids.enabled_time]):
                processed_event = {
                    'case_id': event[sim_log_ids.case],
                    'lifecycle': "ENABLE",
                    'timestamp': event[sim_log_ids.enabled_time],
                    'node_name': event[sim_log_ids.activity]
                }
                add_to_sorted_events(sorted_events, processed_event)
            # Process start
            if not pd.isna(event[sim_log_ids.start_time]):
                processed_event = {
                    'case_id': event[sim_log_ids.case],
                    'lifecycle': "START",
                    'timestamp': event[sim_log_ids.start_time],
                    'node_name': event[sim_log_ids.activity]
                }
                add_to_sorted_events(sorted_events, processed_event)
            # Process end
            if not pd.isna(event[sim_log_ids.end_time]):
                processed_event = {
                    'case_id': event[sim_log_ids.case],
                    'lifecycle': "COMPLETE",
                    'timestamp': event[sim_log_ids.end_time],
                    'node_name': event[sim_log_ids.activity]
                }
                add_to_sorted_events(sorted_events, processed_event)
    return sorted_events


def add_to_sorted_events(sorted_events: List[dict], event: dict):
    index = 0
    for i, other_event in enumerate(sorted_events):
        if other_event['timestamp'] <= event['timestamp']:
            index = i
    sorted_events.insert(index + 1, event)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Function to compute, given the reachability graph and #
# the sorted sequence of events, the path in the graph  #
# that represents the execution of the recorded events  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def compute_marking_sequence(
        graph: ReachabilityGraph,
        sorted_events: List[dict],
        ongoing_marking_id: Optional[int] = None
) -> List[int]:
    # If the trace is an ongoing one, instantiate the ongoing marking as current path
    current_marking_paths = None if ongoing_marking_id is None else [[ongoing_marking_id]]
    # Compute all marking sequences that fulfill the event sequence
    for event in sorted_events:
        next_marking_paths = []
        if event['lifecycle'] == "CASE_ARRIVAL":
            # Create a marking path per outgoing edge (should be 1) of initial marking
            for edge_id in graph.outgoing_edges[graph.initial_marking_id]:
                next_marking = graph.edges[edge_id][1]
                next_marking_paths += [[graph.initial_marking_id, next_marking]]
        else:
            edges_to_reach = []
            traverse = False
            if event['lifecycle'] == "ENABLE":
                # Retrieve edges with START of recorded event as label
                edges_to_reach = graph.activity_to_edges[(event['node_name'], "START")]
                traverse = False  # Do not traverse edge after reaching source marking
            elif event['lifecycle'] == "START":
                # Retrieve edges with START of recorded event as label
                edges_to_reach = graph.activity_to_edges[(event['node_name'], "START")]
                traverse = True  # Traverse edge after reaching source marking
            elif event['lifecycle'] == "COMPLETE":
                # Retrieve edges with START of recorded event as label
                edges_to_reach = graph.activity_to_edges[(event['node_name'], "COMPLETE")]
                traverse = True  # Traverse edge after reaching source marking
            elif event['lifecycle'] == "CASE_END":
                # Retrieve edges with START of recorded event as label
                edges_to_reach = graph.activity_to_edges[(event['node_name'], "START")]
                traverse = True  # Traverse edge after reaching source marking
            # Compute paths to the source markings of these edges
            for current_path in current_marking_paths:
                current_marking = current_path[-1]
                for edge_id in edges_to_reach:
                    marking_to_reach = graph.edges[edge_id][0]
                    # Search if there are any paths from current_marking to marking_to_reach
                    for path in find_paths(graph, current_marking, marking_to_reach):
                        # Store the path + crossing event
                        if traverse:
                            next_marking_paths += [current_path + path + [graph.edges[edge_id][1]]]
                        else:
                            next_marking_paths += [current_path + path]
        # Update marking paths
        current_marking_paths = next_marking_paths
    # Return path
    return random.choice(current_marking_paths)


def find_paths(graph: ReachabilityGraph, start_marking: int, end_marking: int) -> List[List[int]]:
    # Initialize final paths
    final_paths = []
    # Check if already reached
    if end_marking == start_marking:
        # Reached, add empty path
        final_paths = [[]]
    else:
        # Find paths
        paths = [
            [graph.edges[edge_id][1]]
            for edge_id in graph.outgoing_edges[start_marking]
            if graph.edge_to_activity[edge_id][1] == "GATEWAY"
        ]
        while len(paths) > 0:
            current_path = paths.pop()
            current_marking = current_path[-1]
            if current_marking == end_marking:
                # Reached, add to final paths
                final_paths += [current_path]
            else:
                # Keep expanding through gateway edges (avoiding loops)
                paths += [
                    current_path + [graph.edges[edge_id][1]]
                    for edge_id in graph.outgoing_edges[current_marking]
                    if graph.edge_to_activity[edge_id][1] == "GATEWAY" and graph.edges[edge_id][1] not in current_path
                ]

    # return [[]] if already there
    return final_paths


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Function to, given a BPMN model, its reachability graph,    #
# and a trace, compute the sequence of events (ENABLE, START, #
# COMPLETE) and the token movements associated to each event  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def compute_token_movements(
        model: BPMNModel,
        reach_graph: ReachabilityGraph,
        trace: pd.DataFrame,
        ongoing_token_status: Optional[dict] = None
) -> List[dict]:
    """
    Warning, the BPMN model must only have one start and one end events (workflow net), and no duplicated labels.
    """
    # If ongoing marking, obtain ongoing marking from there
    if ongoing_token_status is not None:
        token_idx = len(ongoing_token_status)
        current_token_status = ongoing_token_status.copy()
        ongoing_marking_id = reach_graph.marking_to_key[tuple(sorted(ongoing_token_status.values()))]
    else:
        token_idx = 0
        current_token_status = dict()
        ongoing_marking_id = None
    # Obtain labels for start and end events
    start_event = [node for node in model.nodes if node.is_start_event()][0]
    end_event = [node for node in model.nodes if node.is_end_event()][0]
    # Compute sorted events and marking sequence for this trace
    sorted_events = sorted_trace_events(trace, start_event.name, end_event.name)
    marking_sequence = compute_marking_sequence(reach_graph, sorted_events, ongoing_marking_id)
    # Process event by event computing the paths by advancing through the marking sequence
    events = []
    i = 0
    for event in sorted_events:
        paths = dict()
        if event['lifecycle'] == "CASE_ARRIVAL":
            # Initiate a token per enabled element in initial marking
            for enabled_element in reach_graph.markings[marking_sequence[i]]:
                current_token_status[f'token_{token_idx}'] = enabled_element
                paths[f'token_{token_idx}'] = [enabled_element]
                token_idx += 1
            # Process first movement
            marking = reach_graph.markings[marking_sequence[i + 1]]
            current_token_status, paths, token_idx = update_token_status(
                current_token_status,
                marking,
                paths,
                token_idx
            )
            # Advance marking for next iteration
            i += 1
        else:
            if event['lifecycle'] == "START":
                # Process advancements until the enabled activity can be fired
                next_edge_label = get_next_edge_label(reach_graph, marking_sequence, i)
                while next_edge_label[1] == "GATEWAY":
                    # Retrieve gateway ID
                    gateway_id = [node.id for node in model.nodes if node.name == next_edge_label[0]][0]
                    # Advance tokens in marking
                    marking = reach_graph.markings[marking_sequence[i + 1]]
                    current_token_status, paths, token_idx = update_token_status(
                        current_token_status,
                        marking,
                        paths,
                        token_idx,
                        gateway_id
                    )
                    # Advance marking and compute next edge label
                    i += 1
                    next_edge_label = get_next_edge_label(reach_graph, marking_sequence, i)
                # Process activity entrance (only one token moving)
                marking = reach_graph.markings[marking_sequence[i + 1]]
                current_token_status, paths, token_idx = update_token_status(
                    current_token_status,
                    marking,
                    paths,
                    token_idx
                )
                # Advance marking for next iteration
                i += 1
            elif event['lifecycle'] == "ENABLE":
                # Process advancements until the enabled activity can be fired
                next_edge_label = get_next_edge_label(reach_graph, marking_sequence, i)
                while next_edge_label[1] == "GATEWAY":
                    # Retrieve gateway ID
                    gateway_id = [node.id for node in model.nodes if node.name == next_edge_label[0]][0]
                    # Advance tokens in marking
                    marking = reach_graph.markings[marking_sequence[i + 1]]
                    current_token_status, paths, token_idx = update_token_status(
                        current_token_status,
                        marking,
                        paths,
                        token_idx,
                        gateway_id
                    )
                    # Advance marking and compute next edge label
                    i += 1
                    next_edge_label = get_next_edge_label(reach_graph, marking_sequence, i)
            elif event['lifecycle'] == "COMPLETE":
                # Exit activity and process gateway advancements
                next_edge_label = get_next_edge_label(reach_graph, marking_sequence, i)
                while next_edge_label == (event['node_name'], "COMPLETE") or next_edge_label[1] == "GATEWAY":
                    # Retrieve gateway/node ID
                    gateway_id = [node.id for node in model.nodes if node.name == next_edge_label[0]][0]
                    # Advance tokens in marking
                    marking = reach_graph.markings[marking_sequence[i + 1]]
                    current_token_status, paths, token_idx = update_token_status(
                        current_token_status,
                        marking,
                        paths,
                        token_idx,
                        gateway_id
                    )
                    # Advance marking and compute next edge label
                    i += 1
                    next_edge_label = get_next_edge_label(reach_graph, marking_sequence, i)
            elif event['lifecycle'] == "CASE_END":
                # Process gateway advancement until the end event starts + cross it
                next_edge_label = get_next_edge_label(reach_graph, marking_sequence, i)
                while next_edge_label[1] == "GATEWAY" or next_edge_label == (event['node_name'], "START"):
                    # Retrieve gateway ID
                    if next_edge_label[1] == "GATEWAY":
                        gateway_id = [node.id for node in model.nodes if node.name == next_edge_label[0]][0]
                    else:
                        gateway_id = None
                    # Advance tokens in marking
                    marking = reach_graph.markings[marking_sequence[i + 1]]
                    current_token_status, paths, token_idx = update_token_status(
                        current_token_status,
                        marking,
                        paths,
                        token_idx,
                        gateway_id
                    )
                    # Advance marking and compute next edge label
                    i += 1
                    next_edge_label = get_next_edge_label(reach_graph, marking_sequence, i)
        # Create event structure and add it to final list
        processed_event = {
            'case_id': event['case_id'],
            'lifecycle': event['lifecycle'],
            'timestamp': event['timestamp'],
            'node_id': [node.id for node in model.nodes if node.name == event['node_name']][0],
            'paths': paths
        }
        events += [processed_event]
    # Return final list
    return events


def update_token_status(
        current_token_status: Dict[str, str],
        marking: Set[str],
        paths: Dict[str, List[str]],
        token_idx: int,
        gateway_id: Optional[str] = None
) -> Tuple[Dict[str, str], Dict[str, List[str]], int]:
    # Instantiate updated token_status
    updated_token_status = dict()
    # Check if the tokens moved or appeared/disappeared
    if len(marking) == len(current_token_status):
        # Tokens move
        new_enabled_elements = marking - set(current_token_status.values())
        for token_id in current_token_status:
            if current_token_status[token_id] in marking:
                # Didn't move, keep it for next iteration
                updated_token_status[token_id] = current_token_status[token_id]
            else:
                # Moved, update token_status and record "path"
                new_enabled_element = new_enabled_elements.pop()
                updated_token_status[token_id] = new_enabled_element
                if gateway_id is None:
                    paths[token_id] = paths.get(token_id, []) + [new_enabled_element]
                else:
                    paths[token_id] = paths.get(token_id, []) + [gateway_id, new_enabled_element]
    else:
        # Tokens appear/disappear
        new_enabled_elements = marking - set(current_token_status.values())
        for token_id in current_token_status:
            if current_token_status[token_id] in marking:
                # Didn't move, keep it for next iteration
                updated_token_status[token_id] = current_token_status[token_id]
            else:
                # Disappearing token, add gateway ID to path
                paths[token_id] = paths.get(token_id, []) + [gateway_id]
        for new_enabled_element in new_enabled_elements:
            # New token, add to token_status and record "path"
            updated_token_status[f'token_{token_idx}'] = new_enabled_element
            paths[f'token_{token_idx}'] = [gateway_id, new_enabled_element]
            token_idx += 1
    return updated_token_status, paths, token_idx


def get_next_edge_label(reachability_graph: ReachabilityGraph, marking_sequence: List[int], i: int) -> Tuple[str, str]:
    if i + 1 < len(marking_sequence):
        return [
            reachability_graph.edge_to_activity[edge_id]
            for edge_id in reachability_graph.edges
            if reachability_graph.edges[edge_id] == (marking_sequence[i], marking_sequence[i + 1])
        ][0]  # Should be only 1 by construction
    else:
        return "", ""
