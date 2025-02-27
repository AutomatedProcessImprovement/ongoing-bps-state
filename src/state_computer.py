# src/state_computer.py
from typing import List

import pandas as pd
from ongoing_process_state.n_gram_index import NGramIndex

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
        ids = self.event_log_ids 
        # Group the event log by CaseId
        grouped = self.event_log_df.groupby(ids.case)
        for case_id, group in grouped:
            # Sort activities by StartTime
            group = group.sort_values(ids.start_time)

            # Identify ongoing activities
            ongoing_activities_df = group[group[ids.end_time].isna()]
            ongoing_activities = []
            for _, row in ongoing_activities_df.iterrows():
                original_activity = row[ids.activity]
                task_id = self.bpmn_handler.get_task_id_by_name(original_activity)
                stime = row[ids.start_time]
                if pd.isna(stime) or task_id is None:
                    enabled_time = None
                else:
                    # Use the original activity name for the concurrency oracle lookup.
                    if original_activity not in getattr(self.concurrency_oracle, 'concurrency', {}):
                        enabled_time = None
                    else:
                        temp_event = pd.Series({
                            ids.activity: original_activity,
                            ids.start_time: stime,
                            ids.end_time: stime
                        })
                        enabled_time = row[ids.enabled_time]

                ongoing_activities.append({
                    "id": task_id,
                    "label": original_activity,
                    "start_time": stime,
                    "resource": row[ids.resource],
                    "enabled_time": enabled_time
                })

            # Get the entire sequence of activities (including ongoing ones)
            sorted_events = list()
            group.apply(
                lambda activity_instance:
                _add_to_sorted_events(
                    sorted_events=sorted_events,
                    activity_label=activity_instance[ids.activity],
                    start_time=activity_instance[ids.start_time],
                    end_time=activity_instance[ids.end_time],
                ), axis=1
            )
            n_gram = [NGramIndex.TRACE_START] + [event['label'] for event in sorted_events]
            # Compute the state using N-Gram index
            state_marking = self.n_gram_index.get_best_marking_state_for(n_gram)
            state_flows = {el_id for el_id in state_marking if el_id in self.bpmn_handler.sequence_flows}
            state_activities = {el_id for el_id in state_marking if el_id in self.bpmn_handler.activities}

            # Compute enabled activities
            finished_activities = group[group[ids.end_time].notna()]
            enabled_activities = []
            for flow_id in state_flows:
                target_ref = self.bpmn_handler.sequence_flows.get(flow_id)
                if target_ref in self.bpmn_handler.activities:
                    activity_name = self.bpmn_handler.activities.get(target_ref)
                    if len(finished_activities)== 0:
                        enabled_time = min(group[ids.start_time])
                    else:
                        if activity_name not in getattr(self.concurrency_oracle, 'concurrency', {}):
                            self.concurrency_oracle.concurrency[activity_name] = {}
                        max_end_time = finished_activities[ids.end_time].max() if not finished_activities.empty else None
                        temp_event = pd.Series({
                            ids.activity: activity_name,
                            ids.start_time: max_end_time + pd.Timedelta(seconds=1),
                            ids.end_time: max_end_time+ pd.Timedelta(seconds=1),
                        })
                        enabled_time = self.concurrency_oracle.enabled_since(trace=finished_activities, event=temp_event)
                    enabled_activities.append({
                        "id": target_ref,
                        "enabled_time": enabled_time
                    })

            # Compute enabled gateways, skipping those that have upstream tasks still ongoing
            enabled_gateways = []
            for flow_id in state_flows:
                gw_id = self.bpmn_handler.sequence_flows.get(flow_id, None)
                # Only consider if the target is an exclusive gateway.
                if gw_id and self.bpmn_handler.get_node_type(gw_id) == 'exclusiveGateway':
                    # Get upstream tasks for this gateway.
                    tasks_upstream = self.bpmn_handler.get_upstream_tasks_through_gateways(gw_id)
                    # If any upstream task is still ongoing (i.e. token present in state_activities), skip this gateway.
                    if tasks_upstream.intersection(state_activities):
                        continue
                    # If there are finished activities, use them to compute the enabled time;
                    # otherwise, default to the earliest start time in the case.
                    if not finished_activities.empty:
                        gw_enabled_time = self._compute_gateway_enabled_time(gw_id, group, finished_activities)
                    else:
                        gw_enabled_time = min(group[ids.start_time])
                    if pd.notna(gw_enabled_time):
                        enabled_gateways.append({
                            "id": gw_id,
                            "enabled_time": gw_enabled_time
                        })

            # --- Exclude cases with enabled gateways that are end events ---
            if any(self.bpmn_handler.is_end_event(gateway["id"]) for gateway in enabled_gateways):
                # Skip this case from the process state as it has an enabled gateway that is an end event.
                continue

            # --- Compute enabled events ---
            enabled_events = []
            for flow_id in state_flows:
                target_ref = self.bpmn_handler.sequence_flows.get(flow_id)
                # Check if the target is an event
                if target_ref in self.bpmn_handler.events:
                    # Compute the enabled time for the event
                    if finished_activities.empty:
                        event_enabled_time = min(group[ids.start_time])
                    else:
                        # Look up the event name; if not found, fallback to target_ref
                        event_name = self.bpmn_handler.events.get(target_ref, target_ref)
                        # Ensure the concurrency oracle has an entry for this event
                        if event_name not in getattr(self.concurrency_oracle, 'concurrency', {}):
                            self.concurrency_oracle.concurrency[event_name] = {}
                        # Get the last finished activity’s end time and add a small delta
                        max_end_time = finished_activities[ids.end_time].max()
                        temp_event = pd.Series({
                            ids.activity: event_name,
                            ids.start_time: max_end_time + pd.Timedelta(seconds=1),
                            ids.end_time: max_end_time + pd.Timedelta(seconds=1)
                        })
                        event_enabled_time = self.concurrency_oracle.enabled_since(trace=finished_activities, event=temp_event)
                    enabled_events.append({
                        "id": target_ref,
                        "enabled_time": event_enabled_time
                    })


            # Store case information
            case_states[case_id] = {
                "control_flow_state": {
                    "flows": list(state_flows),
                    "activities": list(state_activities)
                },
                "ongoing_activities": ongoing_activities,
                "enabled_activities": enabled_activities,
                "enabled_gateways": enabled_gateways,
                "enabled_events": enabled_events
            }
        return case_states

    def _compute_gateway_enabled_time(self, gateway_id, group_df, ended_df):
        tasks_upstream = self.bpmn_handler.get_upstream_tasks_through_gateways(gateway_id)
        if not tasks_upstream:
            return ended_df[self.event_log_ids.end_time].max()
        task_names = [
            self.bpmn_handler.activities[t_id]
            for t_id in tasks_upstream
            if t_id in self.bpmn_handler.activities
        ]
        sub_df = ended_df[ended_df[self.event_log_ids.activity].isin(task_names)]
        max_et = sub_df[self.event_log_ids.end_time].max()
        if pd.isna(max_et):
            max_et = ended_df[self.event_log_ids.end_time].max()
        return max_et


def _add_to_sorted_events(
        sorted_events: List[dict],
        activity_label: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp
):
    # Insert start
    if not pd.isna(start_time):
        index = 0
        event = {'label': f"{activity_label}+START", 'timestamp': start_time}
        for i, other_event in enumerate(sorted_events):
            if other_event['timestamp'] <= event['timestamp']:
                index = i
        sorted_events.insert(index + 1, event)
    # Insert end
    if not pd.isna(end_time):
        index = 0
        event = {'label': f"{activity_label}+COMPLETE", 'timestamp': end_time}
        for i, other_event in enumerate(sorted_events):
            if other_event['timestamp'] <= event['timestamp']:
                index = i
        sorted_events.insert(index + 1, event)
