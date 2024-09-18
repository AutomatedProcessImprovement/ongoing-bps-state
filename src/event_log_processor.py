# src/event_log_processor.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'libs')))

import pandas as pd

class EventLogProcessor:
    """Processes the event log DataFrame based on the starting point."""
    def __init__(self, event_log_df, start_time):
        self.event_log_df = event_log_df
        self.start_time = pd.to_datetime(start_time, utc=True) if start_time else None

    def process(self):
        """Processes the event log according to the provided starting point."""
        df = self.event_log_df.copy()
        if self.start_time:
            # Exclude activities that start after the starting point
            df = df[df['StartTime'] <= self.start_time]
            # Update EndTime for ongoing activities at the starting point
            df.loc[(df['EndTime'].isna()) | (df['EndTime'] > self.start_time), 'EndTime'] = pd.NaT

        print(df)
        return df
