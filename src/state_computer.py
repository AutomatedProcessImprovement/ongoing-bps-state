# src/state_computer.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'libs')))

from process_running_state.n_gram_index import NGramIndex

class StateComputer:
    """Computes the state of each case using the N-Gram index."""
    def __init__(self, n_gram_index, event_log_df):
        self.n_gram_index = n_gram_index
        self.event_log_df = event_log_df

    def compute_case_states(self):
        """Computes states and active activities for all cases."""
        case_states = {}
        # Group the event log by CaseId
        grouped = self.event_log_df.groupby('CaseId')
        for case_id, group in grouped:
            # Sort activities by StartTime
            group = group.sort_values('StartTime')
            # Get the last up to 5 activities
            last_activities = group['Activity'].tolist()[-self.n_gram_index.n_gram_size_limit:]
            # Prepend TRACE_START if necessary
            if len(last_activities) < self.n_gram_index.n_gram_size_limit:
                last_activities = [NGramIndex.TRACE_START] + last_activities
            # Compute the state using N-Gram index
            state = self.n_gram_index.get_best_marking_state_for(last_activities)
            # Identify ongoing activities
            ongoing_activities = group[group['EndTime'].isna()][['Activity', 'StartTime', 'Resource']]
            active_activities = ongoing_activities.rename(columns={
                'Activity': 'name',
                'StartTime': 'startTime',
                'Resource': 'resource'
            }).to_dict('records')
            # Store case information
            case_states[case_id] = {
                'state': state,
                'active_activities': active_activities
            }
        return case_states
