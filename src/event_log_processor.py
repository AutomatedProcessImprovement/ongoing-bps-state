# src/event_log_processor.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'libs')))

import pandas as pd
import numpy as np
from pix_framework.enhancement.concurrency_oracle import OverlappingConcurrencyOracle
from pix_framework.enhancement.start_time_estimator.config import Configuration, ConcurrencyThresholds
from pix_framework.io.event_log import EventLogIDs

class EventLogProcessor:
    """Processes the event log DataFrame based on the starting point."""
    def __init__(self, event_log_df, start_time):
        self.event_log_df = event_log_df
        self.start_time = pd.to_datetime(start_time, utc=True) if start_time else None
    
    def process(self):
        """Processes the event log according to the provided starting point and computes enabled times."""
        df = self.event_log_df.copy()
        if self.start_time:
            # Exclude activities that start after the starting point
            df = df[df['StartTime'] <= self.start_time]
            # Update EndTime for ongoing activities at the starting point
            df.loc[(df['EndTime'].isna()) | (df['EndTime'] > self.start_time), 'EndTime'] = pd.NaT
        else:
            # Handle cases where EndTime is missing or activities are ongoing
            pass  # No changes needed as per requirements
        
        # Compute enabled time if not present
        if 'enabled_time' not in df.columns:
            # Step i: Set all empty/NULL EndTime to last timestamp + 1h
            last_timestamp = df['EndTime'].max()
            last_timestamp_plus_one_hour = last_timestamp + pd.Timedelta(hours=1)
            df['EndTime'] = df['EndTime'].fillna(last_timestamp_plus_one_hour)
            # Instantiate OverlappingConcurrencyOracle
            config = Configuration(
                log_ids=EventLogIDs(
                    case='CaseId',
                    activity='Activity',
                    start_time='StartTime',
                    end_time='EndTime',
                    resource='Resource',
                    enabled_time='enabled_time'
                ),
                concurrency_thresholds=ConcurrencyThresholds(df=0.5)
            )
            concurrency_oracle = OverlappingConcurrencyOracle(df, config)
            # Call add_enabled_times function
            concurrency_oracle.add_enabled_times(df)
            # Step iii: Revert EndTime to empty/NULL where it was originally
            df.loc[df['EndTime'] == last_timestamp_plus_one_hour, 'EndTime'] = pd.NaT
        return df
