# src/input_handler.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'libs')))

import pandas as pd
from pathlib import Path
import json
from process_running_state.utils import read_bpmn_model

class InputHandler:
    """Handles the input files and parameter parsing."""
    def __init__(self, args):
        self.event_log_path = Path(args.event_log)
        self.bpmn_model_path = Path(args.bpmn_model)
        self.start_time = args.start_time
        self.column_mapping = self.parse_column_mapping(args.column_mapping)

    def parse_column_mapping(self, column_mapping_str):
        """Parses the column mapping from a JSON string."""
        if column_mapping_str:
            return json.loads(column_mapping_str)
        else:
            # Default mapping
            return {
                'CaseId': 'CaseId',
                'Resource': 'Resource',
                'Activity': 'Activity',
                'StartTime': 'StartTime',
                'EndTime': 'EndTime'
            }
    
    def read_event_log(self):
        """Reads the event log CSV file into a DataFrame."""
        df = pd.read_csv(self.event_log_path)
        # Apply column mapping
        df = df.rename(columns=self.column_mapping)
        # Validate required columns
        required_columns = ['CaseId', 'Resource', 'Activity', 'StartTime', 'EndTime']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        # Convert StartTime and EndTime to datetime
        df['StartTime'] = pd.to_datetime(df['StartTime'], utc=True)
        df['EndTime'] = pd.to_datetime(df['EndTime'], utc=True, errors='coerce')  # Allow NaT for missing EndTime
        return df

    def read_bpmn_model(self):
        """Reads the BPMN model file."""
        bpmn_model = read_bpmn_model(self.bpmn_model_path)
        return bpmn_model
