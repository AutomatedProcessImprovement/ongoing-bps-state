import ast
import re
from pathlib import Path
from typing import Set, List

from process_running_state.reachability_graph import ReachabilityGraph


class NGramIndex:
    TRACE_START = "DEFAULT_TRACE_START_LABEL"

    def __init__(self, graph: ReachabilityGraph, n_gram_size_limit: int = 5):
        self.graph = graph
        self.n_gram_size_limit = n_gram_size_limit
        self.markings = {}  # Dict with the N-Gram (tuple) as key and list with marking ID(s) as value

    def add_associations(self, n_gram: List[str], markings: Set[str]):
        n_gram_key = tuple(n_gram)
        if n_gram_key in self.markings:
            self.markings[n_gram_key] |= markings
        else:
            self.markings[n_gram_key] = markings

    def add_association(self, n_gram: List[str], marking: str):
        n_gram_key = tuple(n_gram)
        if n_gram_key in self.markings:
            self.markings[n_gram_key] |= {marking}
        else:
            self.markings[n_gram_key] = {marking}

    def get_marking_state(self, n_gram: List[str]) -> List[Set[str]]:
        """
        Retrieve, given an n-gram representing the last N activities executed in a trace, the list of markings (set of
        enabled flows) associated to that state.

        :param n_gram: list of activity labels representing the last N activities recorded in the trace.
        :return: a list with the marking(s) corresponding to the state of the process.
        """
        n_gram_key = tuple(n_gram)
        markings = {}
        # If present, retrieve marking IDs associated to this n-gram
        if n_gram_key in self.markings:
            markings = self.markings[n_gram_key]
        # Return set of markings
        return [self.graph.markings[marking] for marking in markings]

    def get_best_marking_state_for(self, n_gram: List[str]) -> Set[str]:
        """
        Retrieve, given an n-gram representing the last N activities executed in a trace, the marking (set of enabled
        flows) that has higher probability to be the one associated to that state. To do this, the function retrieves
        the marking(s) of each k-gram (being k in 1..n) until the associated marking(s) is deterministic (only one
        marking), or k = n (limit reached). If maximum size n-gram is reached and more than one marking are associated
        to it, return one of them randomly.

        If the n-gram contains activities that are not in the reachability graph (or marking n-grams), filter them out.

        :param n_gram: list of activity labels representing the last N activities recorded in the trace.
        :return: the marking corresponding to the state of the case given the last N activities.
        """
        # Filter out nonexistent (in the reachability graph) activities
        n_gram = [label for label in n_gram if label in self.graph.activity_to_edges]
        # Initialize estimated marking to initial marking (if no other marking found, that's default)
        final_marking = self.get_marking_state([NGramIndex.TRACE_START])[0]
        stop_search = False
        k = 1
        # Search iteratively for a deterministic marking
        while not stop_search and k <= len(n_gram):
            # Get marking(s) corresponding last K activities of the n-gram
            markings = self.get_marking_state(n_gram[-k:])
            if len(markings) == 1:
                # Deterministic marking, stop search
                final_marking = markings[0]
                stop_search = True
            elif len(markings) > 1:
                # More than one marking, keep first and continue expanding n-gram
                final_marking = markings[0]
                k += 1
            else:
                # No marking(s) found, stop search
                stop_search = True
        # Return found marking
        return final_marking

    def build(self):
        """
        Build the n-gram index mapping for the reachability graph in [self.graph] and with the n-limit stored in
        [self.n_gram_size_limit].
        """
        # Initialize stacks
        marking_stack = list(self.graph.markings)  # Stack of markings to explore (incoming edges)
        n_gram_stack = [[] for _ in marking_stack]  # n-gram (list of str) explored to reach each marking in the stack
        target_marking_stack = marking_stack.copy()  # List of marking that each n-gram points to
        # Continue with expansion while there are markings in the stack
        while len(marking_stack) > 0:
            # Initialize lists for next iteration
            next_marking_stack = []
            next_n_gram_stack = []
            next_target_marking_stack = []
            # Expand each of the markings in the stack backwards
            while len(marking_stack) > 0:
                # Retrieve marking to explore, n-gram that led (backwards) to it, and marking at the end of the n-gram
                marking_id = marking_stack.pop()
                previous_n_gram = n_gram_stack.pop()
                target_marking = target_marking_stack.pop()
                # If this marking is the initial marking, save corresponding association
                if marking_id == self.graph.initial_marking_id:
                    current_n_gram = [NGramIndex.TRACE_START] + previous_n_gram
                    self.add_association(current_n_gram, target_marking)
                # Grow n-gram with each incoming edge
                for edge_id in self.graph.incoming_edges[marking_id]:
                    # Add association
                    current_n_gram = [self.graph.edge_to_activity[edge_id]] + previous_n_gram
                    self.add_association(current_n_gram, target_marking)
                    # Save source marking for exploration if necessary
                    if len(current_n_gram) < self.n_gram_size_limit:
                        (source_marking_id, _) = self.graph.edges[edge_id]
                        next_marking_stack += [source_marking_id]
                        next_n_gram_stack += [current_n_gram]
                        next_target_marking_stack += [target_marking]
            # Update search stacks when necessary to keep expanding backwards
            while len(next_marking_stack) > 0:
                # Retrieve marking to explore, n-gram that led (backwards) to it, and marking at the end of the n-gram
                marking_id = next_marking_stack.pop()
                previous_n_gram = next_n_gram_stack.pop()
                target_marking = next_target_marking_stack.pop()
                # If the n-gram is not deterministic, add it to search further
                markings = self.get_marking_state(previous_n_gram)
                if len(markings) > 1:
                    marking_stack += [marking_id]
                    n_gram_stack += [previous_n_gram]
                    target_marking_stack += [target_marking]

    def get_self_contained_map(self) -> dict:
        return {
            key: [self.graph.markings[marking] for marking in self.markings[key]]
            for key in self.markings
        }

    def to_self_contained_map_file(self, file_path: Path):
        with open(file_path, "w") as output_file:
            for n_gram in self.markings:
                markings = [self.graph.markings[marking] for marking in self.markings[n_gram]]
                output_file.write(f"{str(n_gram)} : {str(markings)}\n")

    @staticmethod
    def from_self_contained_map_file(file_path: Path, reachability_graph: ReachabilityGraph) -> 'NGramIndex':
        # Instantiate the marking
        n_gram_index = NGramIndex(reachability_graph, 0)
        # Go over file line by line
        with open(file_path, "r") as input_file:
            for line in input_file:
                # Retrieve key and value
                if line != "":
                    match = re.search(r'\((.*?)\)\s*:\s*\[(.*)\]', line)
                    if match:
                        key = ast.literal_eval(f"({match.group(1)})")
                        value = ast.literal_eval(f"[{match.group(2)}]")
                        n_gram_index.markings[key] = {
                            reachability_graph.marking_to_key[tuple(sorted(marking))]
                            for marking in value
                        }
                    else:
                        raise RuntimeError(f"Problem with format of line: {line}.")
        # Return read n-gram index
        return n_gram_index
