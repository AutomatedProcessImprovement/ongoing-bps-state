# main.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'libs')))

import argparse
import json
import pandas as pd
from src.input_handler import InputHandler
from src.event_log_processor import EventLogProcessor
from src.bpmn_handler import BPMNHandler
from src.state_computer import StateComputer

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process Event Log and BPMN Model')
    parser.add_argument('event_log', help='Path to the event log CSV file')
    parser.add_argument('bpmn_model', help='Path to the BPMN model file')
    parser.add_argument('bpmn_parameters', help='BPMN parameters in JSON format')
    parser.add_argument('--start_time', help='Optional starting point datetime in ISO format')
    parser.add_argument('--column_mapping', help='Optional JSON string for column mapping')
    args = parser.parse_args()
    
    # Handle inputs
    input_handler = InputHandler(args)
    event_log_ids = input_handler.event_log_ids
    event_log_df = input_handler.event_log_df
    bpmn_model = input_handler.read_bpmn_model()
    bpmn_parameters = input_handler.parse_bpmn_parameters()
    
    # Process event log
    event_log_processor = EventLogProcessor(event_log_df, args.start_time, event_log_ids)
    processed_event_log = event_log_processor.process()
    concurrency_oracle = event_log_processor.concurrency_oracle
    
    # Build N-Gram Index from BPMN model
    bpmn_handler = BPMNHandler(bpmn_model, bpmn_parameters, input_handler.bpmn_model_path)
    n_gram_index = bpmn_handler.build_n_gram_index()
    reachability_graph = bpmn_handler.get_reachability_graph()
    
    # Compute states for each case
    state_computer = StateComputer(n_gram_index, reachability_graph, processed_event_log, bpmn_handler, concurrency_oracle, event_log_ids)
    case_states = state_computer.compute_case_states()
    
    # Compute the last case arrival datetime
    ids = event_log_ids
    # Get the earliest start_time per case
    case_arrival_times = processed_event_log.groupby(ids.case)[ids.start_time].min()
    # Get the maximum of these earliest start_times
    last_case_arrival = case_arrival_times.max()
    # Convert to ISO format string
    last_case_arrival_str = last_case_arrival.isoformat() if pd.notnull(last_case_arrival) else None
    
    # Compute the last recorded end_time per resource
    # Drop rows where end_time is NaT
    end_times = processed_event_log.dropna(subset=[ids.end_time])
    # Group by resource and get the maximum end_time per resource
    resource_last_end_times = end_times.groupby(ids.resource)[ids.end_time].max()
    # Convert to dictionary with ISO format strings
    resource_last_end_times_dict = {
        resource: end_time.isoformat() if pd.notnull(end_time) else None
        for resource, end_time in resource_last_end_times.items()
    }
    
    # Output results
    output_data = {
        'last_case_arrival': last_case_arrival_str,
        'resource_last_end_times': resource_last_end_times_dict,
        'cases': {}
    }
    
    for case_id, case_info in case_states.items():
        output_data['cases'][str(case_id)] = {
            'control_flow_state': {
                'flows': list(case_info['control_flow_state']['flows']),
                'activities': list(case_info['control_flow_state']['activities'])
            },
            'ongoing_activities': case_info['ongoing_activities'],
            'enabled_activities': case_info['enabled_activities']
        }
    
    # Write output to JSON file
    with open('output.json', 'w') as json_file:
        json.dump(output_data, json_file, default=str, indent=4)
    
    print("Results have been written to output.json")

if __name__ == '__main__':
    main()
