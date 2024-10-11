# src/state_computer.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'libs')))

import pandas as pd
from process_running_state.n_gram_index import NGramIndex

class StateComputer:
    """Computes the state of each case using the N-Gram index."""
    def __init__(self, n_gram_index, reachability_graph, event_log_df, bpmn_handler, concurrency_oracle, event_log_ids):
        self.n_gram_index = n_gram_index
        self.reachability_graph = reachability_graph
        self.event_log_df = event_log_df
        self.bpmn_handler = bpmn_handler
        self.concurrency_oracle = concurrency_oracle
        self.event_log_ids = event_log_ids

    def compute_case_states(self):
        """Computes states and active activities for all cases."""
        case_states = {}
        ids = self.event_log_ids  # For convenience
        # Group the event log by CaseId
        grouped = self.event_log_df.groupby(ids.case)
        for case_id, group in grouped:
            # Sort activities by StartTime
            group = group.sort_values(ids.start_time)
            # Identify ongoing activities
            ongoing_activities_df = group[group[ids.end_time].isna()]
            ongoing_activities = ongoing_activities_df[[ids.activity, ids.start_time, ids.resource]].rename(columns={
                ids.activity: 'name',
                ids.start_time: 'start_time',
                ids.resource: 'resource'
            }).to_dict('records')
            # Get the entire sequence of activities (including ongoing ones)
            activities = group[ids.activity].tolist()
            n_gram = [NGramIndex.TRACE_START] + activities
            # Compute the state using N-Gram index
            state_marking = self.n_gram_index.get_best_marking_state_for(n_gram)
            state_flows = state_marking.copy()
            # Get the current marking ID
            current_marking_key = tuple(sorted(state_marking))
            current_marking_id = self.reachability_graph.marking_to_key.get(current_marking_key)
            if current_marking_id is not None:
                for activity in ongoing_activities:
                    activity_name = activity['name']
                    # Get incoming edges to the current marking
                    incoming_edges = self.reachability_graph.incoming_edges.get(current_marking_id, [])
                    # Find the edge with the activity label
                    for edge_id in incoming_edges:
                        edge_activity = self.reachability_graph.edge_to_activity.get(edge_id)
                        if edge_activity == activity_name:
                            # Get the source marking of that edge
                            source_marking_id, _ = self.reachability_graph.edges[edge_id]
                            source_marking = self.reachability_graph.markings[source_marking_id]
                            # Intersect the source marking with the current state marking
                            state_flows = state_flows.intersection(source_marking)
                            break  # Stop after finding the first matching edge
            # Add IDs of ongoing activities to the state
            ongoing_activity_ids = set([activity['name'] for activity in ongoing_activities])
            state_activities = ongoing_activity_ids
            # Compute enabled activities
            enabled_activities = []
            for flow_id in state_flows:
                target_ref = self.bpmn_handler.sequence_flows.get(flow_id)
                activity_name = self.bpmn_handler.activities.get(target_ref)
                if activity_name:
                    # Create temporary event
                    finished_activities = group[group[ids.end_time].notna()]
                    max_end_time = finished_activities[ids.end_time].max() if not finished_activities.empty else None
                    temp_event = pd.Series({
                        ids.activity: activity_name,
                        ids.start_time: max_end_time,
                        ids.end_time: max_end_time
                    })
                    # Compute enabled_time using the concurrency oracle
                    enabled_time = self.concurrency_oracle.enabled_since(trace=group, event=temp_event)
                    enabled_activities.append({
                        'name': activity_name,
                        'enabled_time': enabled_time
                    })
            # Store case information
            case_states[case_id] = {
                'control_flow_state': {
                    'flows': list(state_flows),
                    'activities': list(state_activities)
                },
                'ongoing_activities': ongoing_activities,
                'enabled_activities': enabled_activities
            }
        return case_states
