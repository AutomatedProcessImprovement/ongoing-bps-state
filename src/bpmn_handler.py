# src/bpmn_handler.py

from ongoing_process_state.bpmn_model import BPMNModel
from ongoing_process_state.n_gram_index import NGramIndex
import xml.etree.ElementTree as ET
from pathlib import Path

class BPMNHandler:
    """Handles BPMN model operations."""
    def __init__(self, bpmn_model, bpmn_parameters, bpmn_model_path):
        self.bpmn_model = bpmn_model
        self.bpmn_parameters = bpmn_parameters
        self.bpmn_model_path = bpmn_model_path
        self.sequence_flows = {}
        self.activities = {}
        self.flow_sources = {}
        self.task_name_to_id = {}
        self.end_events = set()
        self.events = {}
        self.gateways = {}
        self.parse_bpmn_xml()
    
    def parse_bpmn_xml(self):
        """Parses the BPMN XML file to extract sequence flows, activities, events, and gateways."""
        tree = ET.parse(self.bpmn_model_path)
        root = tree.getroot()
        ns = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}
        
        # Extract activities (tasks)
        for task in root.findall('.//bpmn:task', ns):
            t_id = task.attrib['id']
            t_name = task.attrib.get('name', f"Unnamed Task {t_id}")
            self.activities[t_id] = t_name
            self.task_name_to_id[t_name] = t_id

        # Extract events (all types: start, intermediate, end)
        event_tags = ['startEvent', 'intermediateCatchEvent', 'intermediateThrowEvent', 'endEvent']
        for tag in event_tags:
            # Using the namespace
            for event in root.findall(f'.//bpmn:{tag}', ns):
                event_id = event.attrib['id']
                event_name = event.attrib.get('name', f"Unnamed {tag} {event_id}")
                self.events[event_id] = event_name
                if tag == 'endEvent':
                    self.end_events.add(event_id)
            # Also attempt to extract without namespace, if needed
            for event in root.findall(f'.//{tag}'):
                event_id = event.attrib['id']
                if event_id not in self.events:
                    event_name = event.attrib.get('name', f"Unnamed {tag} {event_id}")
                    self.events[event_id] = event_name
                    if tag == 'endEvent':
                        self.end_events.add(event_id)
        
        # Extract gateways
        gateway_tags = ['exclusiveGateway', 'parallelGateway', 'inclusiveGateway', 'complexGateway', 'eventBasedGateway']
        for tag in gateway_tags:
            # Using the namespace
            for gateway in root.findall(f'.//bpmn:{tag}', ns):
                gw_id = gateway.attrib['id']
                self.gateways[gw_id] = tag  # e.g., "exclusiveGateway"
            # Also attempt to extract without namespace, if needed
            for gateway in root.findall(f'.//{tag}'):
                gw_id = gateway.attrib['id']
                if gw_id not in self.gateways:
                    self.gateways[gw_id] = tag

        # Extract sequence flows
        for seq_flow in root.findall('.//bpmn:sequenceFlow', ns):
            sf_id = seq_flow.attrib['id']
            self.sequence_flows[sf_id] = seq_flow.attrib['targetRef']
            self.flow_sources[sf_id] = seq_flow.attrib['sourceRef']
    
    def is_end_event(self, element_id):
        """
        Returns True if the provided element_id corresponds to an end event in the BPMN model.
        """
        return element_id in self.end_events

    def get_upstream_tasks_through_gateways(self, gateway_id):
        visited = set()
        stack = [gateway_id]
        tasks_found = set()
        while stack:
            current = stack.pop()
            if current in self.activities:
                tasks_found.add(current)
                continue
            for sf_id, tgt in self.sequence_flows.items():
                if tgt == current:
                    src = self.flow_sources[sf_id]
                    if src not in visited:
                        visited.add(src)
                        stack.append(src)
        return tasks_found

    def build_n_gram_index(self, n_gram_size_limit=10):
        """Builds the N-Gram index from the BPMN model."""
        extended_bpmn_model = compute_extended_bpmn_model(self.bpmn_model)
        reachability_graph = extended_bpmn_model.get_reachability_graph()
        n_gram_index = NGramIndex(reachability_graph, n_gram_size_limit)
        n_gram_index.build()
        # n_gram_index.to_self_contained_map_file(Path("./n_gram_index.map"))
        self.reachability_graph = reachability_graph
        return n_gram_index

    def get_reachability_graph(self):
        """Returns the reachability graph."""
        return self.reachability_graph

    def get_task_id_by_name(self, name):
        return self.task_name_to_id.get(name)
    
    def get_node_type(self, element_id):
        """
        Returns the type of the BPMN element with the given ID.
        Possible return values include:
          - "Task" for tasks,
          - "Event" for non-end events,
          - "EndEvent" for end events,
          - A gateway type string (e.g., "exclusiveGateway", "parallelGateway", etc.) if it's a gateway,
          - None if the element is not found.
        """
        if element_id in self.activities:
            return "Task"
        if element_id in self.events:
            if element_id in self.end_events:
                return "EndEvent"
            else:
                return "Event"
        if element_id in self.gateways:
            return self.gateways[element_id]
        return None
    
    def get_incoming_flows(self, activity_id):
        """
        Returns the list of sequence flow IDs whose targetRef is the given activity (activity_id).
        """
        incoming_flows = []
        for flow_id, target_ref in self.sequence_flows.items():
            if target_ref == activity_id:
                incoming_flows.append(flow_id)
        return incoming_flows


def compute_extended_bpmn_model(bpmn_model: BPMNModel, treat_event_as_task: bool = False) -> BPMNModel:
    # Build extended BPMN model where each activity is split in Start/End activities
    extended_bpmn_model = BPMNModel()
    split_node_ids = set()
    # Add nodes, splitting when necessary
    for node in bpmn_model.nodes:
        if node.is_task():
            # Task, split in two
            node_start_id = f"{node.id}+START"
            node_start_name = f"{node.name}+START"
            extended_bpmn_model.add_task(node_start_id, node_start_name)
            node_complete_id = f"{node.id}+COMPLETE"
            node_complete_name = f"{node.name}+COMPLETE"
            extended_bpmn_model.add_task(node_complete_id, node_complete_name)
            extended_bpmn_model.add_flow(node.id, node.name, node_start_id, node_complete_id)
            split_node_ids |= {node.id}
        elif node.is_event():
            # Event
            if node.is_intermediate_event() and treat_event_as_task:
                # Intermediate event and we are treating them as tasks, split
                node_start_id = f"{node.id}+START"
                node_start_name = f"{node.name}+START"
                extended_bpmn_model.add_task(node_start_id, node_start_name)
                node_complete_id = f"{node.id}+COMPLETE"
                node_complete_name = f"{node.name}+COMPLETE"
                extended_bpmn_model.add_task(node_complete_id, node_complete_name)
                extended_bpmn_model.add_flow(node.id, node.name, node_start_id, node_complete_id)
                split_node_ids |= {node.id}
            else:
                # Add without splitting
                extended_bpmn_model.add_event(node.type, node.id, node.name)
        elif node.is_gateway():
            # Gateway, add without splitting
            extended_bpmn_model.add_gateway(node.type, node.id, node.name)
    # Add original flows, updating source/target when it is split node
    for flow in bpmn_model.flows:
        flow_source_id = flow.source if flow.source not in split_node_ids else f"{flow.source}+COMPLETE"
        flow_target_id = flow.target if flow.target not in split_node_ids else f"{flow.target}+START"
        extended_bpmn_model.add_flow(flow.id, flow.name, flow_source_id, flow_target_id)
    # Return extended BPMN model
    return extended_bpmn_model
    