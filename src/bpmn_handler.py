# src/bpmn_handler.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'libs')))

from process_running_state.n_gram_index import NGramIndex

class BPMNHandler:
    """Handles BPMN model operations."""
    def __init__(self, bpmn_model):
        self.bpmn_model = bpmn_model

    def build_n_gram_index(self, n_gram_size_limit=5):
        """Builds the N-Gram index from the BPMN model."""
        reachability_graph = self.bpmn_model.get_reachability_graph()
        n_gram_index = NGramIndex(reachability_graph, n_gram_size_limit)
        n_gram_index.build()
        return n_gram_index
