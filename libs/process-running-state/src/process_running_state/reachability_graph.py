import ast
from dataclasses import dataclass
from typing import List, Set


@dataclass
class ReachabilityGraph:
    def __init__(self):
        self.edges = {}  # Dict with edge ID as key, and tuple source-target marking IDs as value
        self.activity_to_edges = {}  # Dict with activity label as key, and list of edge IDs as value
        self.edge_to_activity = {}  # Dict with edge ID as key, and activity label as value
        self.markings = {}  # Dict with ID as key, and set of place IDs (marking) as value
        self.marking_to_key = {}  # Dict with marking (sorted tuple) as key, and ID as value
        self.incoming_edges = {}  # Dict with marking ID as key, and set of incoming edge IDs as value
        self.outgoing_edges = {}  # Dict with marking ID as key, and set of outgoing edge IDs as value
        self.initial_marking_id = None  # ID of the initial marking

    def add_marking(self, marking: set, is_initial=False):
        marking_key = tuple(sorted(marking))
        if marking_key not in self.marking_to_key:
            marking_id = len(self.markings)
            self.markings[marking_id] = marking
            self.marking_to_key[marking_key] = marking_id
            self.incoming_edges[marking_id] = set()
            self.outgoing_edges[marking_id] = set()
            if is_initial:
                self.initial_marking_id = marking_id

    def add_edge(self, activity: str, source_marking: set, target_marking: set):
        # Get edge components
        edge_id = len(self.edges)
        source_id = self.marking_to_key[tuple(sorted(source_marking))]
        target_id = self.marking_to_key[tuple(sorted(target_marking))]
        # Check if edge already in the graph
        existent_edges = [
            self.edges[existent_edge_id]
            for existent_edge_id in self.activity_to_edges.get(activity, set())
        ]
        if (source_id, target_id) not in existent_edges:
            # Update graph elements
            self.edges[edge_id] = (source_id, target_id)
            self.activity_to_edges[activity] = self.activity_to_edges.get(activity, set()) | {edge_id}
            self.edge_to_activity[edge_id] = activity
            self.incoming_edges[target_id] |= {edge_id}
            self.outgoing_edges[source_id] |= {edge_id}

    def get_markings_from_activity_sequence(self, activity_sequence: List[str]) -> List[Set[str]]:
        # Initiate search in the initial marking
        current_marking_ids = {self.initial_marking_id}
        # Iterate over the activity sequence advancing in the reachability graph
        for activity in activity_sequence:
            next_marking_ids = set()
            errors = []  # List with paths that could not continue propagation
            # Process each current marking
            for current_marking_id in current_marking_ids:
                # Retrieve edges leaving current marking with the activity as label
                potential_edges = [
                    edge
                    for edge in self.outgoing_edges[current_marking_id]
                    if self.edge_to_activity[edge] == activity
                ]
                # Advance through these edges if no errors
                if len(potential_edges) > 0:
                    # Correct, advance to target marking of each edge
                    next_marking_ids |= {self.edges[edge][1] for edge in potential_edges}
                else:
                    # Error, the activity is not enabled in the current marking
                    errors += [str(current_marking_id)]
            # Raise error if all paths ended up in an error
            if len(next_marking_ids) == 0:
                error_markings = ", ".join(errors)
                raise RuntimeError(f"Error, '{activity}' is not enabled from markings {error_markings}.")
            # Replace current with next marking ids
            current_marking_ids = next_marking_ids
        # Return last reached marking(s)
        return [self.markings[marking_id] for marking_id in current_marking_ids]

    def to_tgf_format(self) -> str:
        """
        Stores the reachability graph in a string following the Trivial Graph Format. As a reachability graph is a
        directed graph where the nodes store a marking (set of IDs) and the edges the name of an activity, the label
        of each node corresponds to the serialization of a set of strings (the marking) and the label of an edge the
        name of the corresponding activity.

        ** Note: the first node is assumed to be the initial marking.

        Example:

        0 {'1'}
        1 {'9', '6'}
        2 {'11', '6'}
        3 {'9', '15'}
        4 {'11', '15'}
        5 {'19'}
        #
        0 1 Invoice Received
        1 2 Notify Acceptance
        1 3 Post Invoice
        3 4 Notify Acceptance
        2 4 Post Invoice
        4 5 Pay Invoice
        """
        # Instantiate string to store conversion
        tgf_string = ""
        # Store markings
        for marking_id in self.markings:
            tgf_string += f"{marking_id} {self.markings[marking_id]}\n"
        # Delimiter
        tgf_string += "#\n"
        # Store edges
        for edge_id in self.edges:
            (source_id, target_id) = self.edges[edge_id]
            label = self.edge_to_activity[edge_id]
            tgf_string += f"{source_id} {target_id} {label}\n"
        # Return TGF formatted graph
        return tgf_string

    @staticmethod
    def from_tgf_format(tgf_string: str) -> 'ReachabilityGraph':
        """
        Instantiate a reachability graph from a string containing its nodes and edges stored in Trivial Graph Format,
        following the description in method "to_tgf_format".

        :param tgf_string: string with the nodes and edges of the graph in a TGF format.
        :return: an instance of the reachability graph.
        """
        # Instantiate empty graph and map for IDs
        graph = ReachabilityGraph()
        node_id_file_to_marking = dict()
        lines = [line.strip() for line in tgf_string.splitlines() if line.strip() != ""]
        # Go over the lines processing first nodes and then edges
        state = 0  # processing nodes or edges
        initial_marking = True
        for line in lines:
            if line == "#":
                # Delimiter, switch to edges
                state = 1
            elif state == 0:
                # -- Processing nodes
                # Get ID and marking
                id_in_file = line.split(" ")[0]
                marking = ast.literal_eval(line[len(id_in_file) + 1:])
                # Add node to graph
                graph.add_marking(marking, initial_marking)
                initial_marking = False
                # Update ID conversion
                node_id_file_to_marking[id_in_file] = marking
            else:
                # -- Processing edges
                # Get source, target, and activity name
                split_line = line.split(" ")
                source_id = split_line[0]
                target_id = split_line[1]
                label = line[len(source_id) + len(target_id) + 2:]
                # Add edge to graph
                graph.add_edge(label, node_id_file_to_marking[source_id], node_id_file_to_marking[target_id])
        # Return reachability graph
        return graph
