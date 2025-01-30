# src/input_handler.py

import pandas as pd
from pathlib import Path
import json
from ongoing_process_state.utils import read_bpmn_model
from pix_framework.io.event_log import EventLogIDs

class InputHandler:
    """Handles the input files and parameter parsing."""
    def __init__(self, args):
        self.event_log_path = Path(args.event_log)
        self.bpmn_model_path = Path(args.bpmn_model)
        self.bpmn_parameters_str = args.bpmn_parameters
        self.start_time = args.start_time
        self.column_mapping_str = args.column_mapping
        self.column_mapping = self.parse_column_mapping()
        self.event_log_df = self.read_event_log()  # Read the event log here
        self.event_log_ids = self.get_event_log_ids()
    
    def parse_column_mapping(self):
        """Parses the column mapping from a JSON string."""
        if self.column_mapping_str:
            mapping = json.loads(self.column_mapping_str)
            # Ensure all required standard names are present in the mapping values
            required_standard_names = ['CaseId', 'Resource', 'Activity', 'StartTime', 'EndTime']
            provided_standard_names = set(mapping.values())
            for std_name in required_standard_names:
                if std_name not in provided_standard_names:
                    # If the standard name is not in the mapping, assume the column is already named as the standard name
                    mapping[std_name] = std_name
            return mapping
        else:
            # Use default mapping (columns are already named as standard names)
            return {
                'CaseId': 'CaseId',
                'Resource': 'Resource',
                'Activity': 'Activity',
                'StartTime': 'StartTime',
                'EndTime': 'EndTime'
            }
    
    def get_event_log_ids(self):
        """Returns the EventLogIDs instance with actual column names after mapping."""
        return EventLogIDs(
            case='CaseId',
            activity='Activity',
            resource='Resource',
            start_time='StartTime',
            end_time='EndTime',
            enabled_time='enabled_time'
        )

    def parse_bpmn_parameters(self):
        """
        Interpret self.bpmn_parameters_str as a filename.
        We'll open it and parse the JSON content from that file.
        """
        with open(self.bpmn_parameters_str, 'r') as f:
            return json.load(f)

    
    def read_event_log(self):
        """Reads the event log CSV file into a DataFrame."""
        df = pd.read_csv(self.event_log_path)
        # Apply column mapping to rename columns to standard names
        df = df.rename(columns=self.column_mapping)
        # Validate required columns using standard names
        required_columns = ['CaseId', 'Resource', 'Activity', 'StartTime', 'EndTime']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        # Convert timestamps to datetime with UTC timezone
        df['StartTime'] = pd.to_datetime(df['StartTime'], utc=True)
        df['EndTime'] = pd.to_datetime(df['EndTime'], utc=True, errors='coerce')  # Allow NaT for missing EndTime
        return df
    
    def read_bpmn_model(self):
        """Reads the BPMN model file."""
        bpmn_model = read_bpmn_model(self.bpmn_model_path)
        return bpmn_model
