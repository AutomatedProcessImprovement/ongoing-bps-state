# src/bpmn_handler.py

from ongoing_process_state.n_gram_index import NGramIndex
import xml.etree.ElementTree as ET

class BPMNHandler:
    """Handles BPMN model operations."""
    def __init__(self, bpmn_model, bpmn_parameters, bpmn_model_path):
        self.bpmn_model = bpmn_model
        self.bpmn_parameters = bpmn_parameters
        self.bpmn_model_path = bpmn_model_path
        self.sequence_flows = {}
        self.activities = {}
        self.parse_bpmn_xml()
    
    def parse_bpmn_xml(self):
        """Parses the BPMN XML file to extract sequence flows and activities."""
        tree = ET.parse(self.bpmn_model_path)
        root = tree.getroot()
        ns = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}
        
        # Extract activities (tasks)
        for task in root.findall('.//bpmn:task', ns):
            self.activities[task.attrib['id']] = task.attrib.get('name', f"Unnamed Task {task.attrib['id']}")
        # Extract sequence flows
        for seq_flow in root.findall('.//bpmn:sequenceFlow', ns):
            self.sequence_flows[seq_flow.attrib['id']] = seq_flow.attrib['targetRef']
    
    def build_n_gram_index(self, n_gram_size_limit=5):
        """Builds the N-Gram index from the BPMN model."""
        reachability_graph = self.bpmn_model.get_reachability_graph()
        n_gram_index = NGramIndex(reachability_graph, n_gram_size_limit)
        n_gram_index.build()
        self.reachability_graph = reachability_graph
        return n_gram_index
    
    def get_reachability_graph(self):
        """Returns the reachability graph."""
        return self.reachability_graph
