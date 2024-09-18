# main.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'libs')))

import argparse
import json
from src.input_handler import InputHandler
from src.event_log_processor import EventLogProcessor
from src.bpmn_handler import BPMNHandler
from src.state_computer import StateComputer

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process Event Log and BPMN Model')
    parser.add_argument('event_log', help='Path to the event log CSV file')
    parser.add_argument('bpmn_model', help='Path to the BPMN model file')
    parser.add_argument('--start_time', help='Optional starting point datetime in ISO format')
    parser.add_argument('--column_mapping', help='Optional JSON string for column mapping')
    args = parser.parse_args()
    
    # Handle inputs
    input_handler = InputHandler(args)
    event_log_df = input_handler.read_event_log()
    bpmn_model = input_handler.read_bpmn_model()
    
    # Process event log
    event_log_processor = EventLogProcessor(event_log_df, args.start_time)
    processed_event_log = event_log_processor.process()
    
    # Build N-Gram Index from BPMN model
    bpmn_handler = BPMNHandler(bpmn_model)
    n_gram_index = bpmn_handler.build_n_gram_index()
    
    # Compute states for each case
    state_computer = StateComputer(n_gram_index, processed_event_log)
    case_states = state_computer.compute_case_states()
    
    # Output results
    output_data = {}
    for case_id, case_info in case_states.items():
        print(f"Case ID: {case_id}")
        print(f"State: {case_info['state']}")
        print("Active Activities:")
        for activity in case_info['active_activities']:
            print(f"\tName: {activity['name']}, StartTime: {activity['startTime']}, Resource: {activity['resource']}")
        print("")
        # Prepare data for JSON output
        output_data[case_id] = {
            'state': str(case_info['state']),
            'active_activities': case_info['active_activities']
        }
    
    # Write output to JSON file
    with open('output.json', 'w') as json_file:
        json.dump(output_data, json_file, default=str, indent=4)
    
    print("Results have been written to output.json")

if __name__ == '__main__':
    main()
