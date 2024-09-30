# src/state_computer.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'libs')))

from process_running_state.n_gram_index import NGramIndex
from collections import defaultdict

class StateComputer:
    """Computes the state of each case using the N-Gram index."""
    def __init__(self, n_gram_index, reachability_graph, event_log_df, bpmn_handler):
        self.n_gram_index = n_gram_index
        self.reachability_graph = reachability_graph
        self.event_log_df = event_log_df
        self.bpmn_handler = bpmn_handler
    
    def compute_case_states(self):
        """Computes states and active activities for all cases."""
        case_states = {}
        # Group the event log by CaseId
        grouped = self.event_log_df.groupby('CaseId')
        for case_id, group in grouped:
            # Sort activities by StartTime
            group = group.sort_values('StartTime')
            # Identify ongoing activities
            ongoing_activities_df = group[group['EndTime'].isna()]
            ongoing_activities = ongoing_activities_df[['Activity', 'StartTime', 'Resource']].rename(columns={
                'Activity': 'name',
                'StartTime': 'start_time',
                'Resource': 'resource'
            }).to_dict('records')
            # Get last 5 finished activities (excluding ongoing ones)
            finished_activities = group[group['EndTime'].notna()]
            last_activities = finished_activities['Activity'].tolist()[-self.n_gram_index.n_gram_size_limit:]
            # Compute the state using N-Gram index
            if last_activities:
                n_gram = last_activities
                if len(n_gram) < self.n_gram_index.n_gram_size_limit:
                    n_gram = [NGramIndex.TRACE_START] + n_gram
                state_marking = self.n_gram_index.get_best_marking_state_for(n_gram)
                state_flows = state_marking
            else:
                # No finished activities, start from initial marking
                state_flows = self.reachability_graph.markings[self.reachability_graph.initial_marking_id]
            # Post-process the state to replace active flows consumed by ongoing activities
            for activity in ongoing_activities:
                activity_name = activity['name']
                # Get edges corresponding to the activity
                edges = self.reachability_graph.activity_to_edges.get(activity_name, [])
                for edge_id in edges:
                    edge_activity = self.reachability_graph.edge_to_activity.get(edge_id)
                    if edge_activity == activity_name:
                        source_marking_id, _ = self.reachability_graph.edges[edge_id]
                        source_marking = self.reachability_graph.markings[source_marking_id]
                        # Intersect the state_flows with the source marking
                        state_flows = state_flows.intersection(source_marking)
                        break  # Found the edge, no need to check further
            # Add IDs of ongoing activities to the state
            ongoing_activity_ids = set([activity['name'] for activity in ongoing_activities])
            state_activities = ongoing_activity_ids
            # Compute enabled activities
            enabled_activities = []
            for flow_id in state_flows:
                target_ref = self.bpmn_handler.sequence_flows.get(flow_id)
                activity_name = self.bpmn_handler.activities.get(target_ref)
                if activity_name:
                    # Get enabled_time as end time of last finished activity
                    if not finished_activities.empty:
                        enabled_time = finished_activities['EndTime'].max()
                    else:
                        enabled_time = None
                    enabled_activities.append({
                        'name': activity_name,
                        'enabled_time': enabled_time
                    })
            # Store case information
            case_states[case_id] = {
                'state': {
                    'flows': state_flows,
                    'activities': state_activities
                },
                'ongoing_activities': ongoing_activities,
                'enabled_activities': enabled_activities
            }
        return case_states
