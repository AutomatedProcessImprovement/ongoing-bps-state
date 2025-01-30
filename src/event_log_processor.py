# src/event_log_processor.py

import pandas as pd
import numpy as np
from pix_framework.enhancement.concurrency_oracle import OverlappingConcurrencyOracle
from pix_framework.enhancement.start_time_estimator.config import Configuration, ConcurrencyThresholds

class EventLogProcessor:
    """Processes the event log DataFrame based on the starting point."""
    def __init__(self, event_log_df, start_time, event_log_ids):
        self.event_log_df = event_log_df
        self.start_time = pd.to_datetime(start_time, utc=True) if start_time else None
        self.event_log_ids = event_log_ids
        self.concurrency_oracle = None  # Initialize the concurrency oracle
    
    def process(self):
        """Processes the event log according to the provided starting point and computes enabled times."""
        df = self.event_log_df.copy()
        ids = self.event_log_ids  # For convenience
        if self.start_time:
            # Exclude activities that start after the starting point
            df = df[df[ids.start_time] <= self.start_time]
            # Update EndTime for ongoing activities at the starting point
            df.loc[(df[ids.end_time].isna()) | (df[ids.end_time] > self.start_time), ids.end_time] = pd.NaT
        
        # Compute enabled time if not present
        if ids.enabled_time not in df.columns:
            # Step i: Set all empty/NULL EndTime to last timestamp + 1h
            last_timestamp = df[ids.end_time].max()
            last_timestamp_plus_one_hour = last_timestamp + pd.Timedelta(hours=1)
            df[ids.end_time] = df[ids.end_time].fillna(last_timestamp_plus_one_hour)
            # Instantiate OverlappingConcurrencyOracle
            config = Configuration(
                log_ids=self.event_log_ids,
                concurrency_thresholds=ConcurrencyThresholds(df=0.5)
            )
            self.concurrency_oracle = OverlappingConcurrencyOracle(df, config)
            # Call add_enabled_times function
            self.concurrency_oracle.add_enabled_times(df)
            # Step iii: Revert EndTime to empty/NULL where it was originally
            df.loc[df[ids.end_time] == last_timestamp_plus_one_hour, ids.end_time] = pd.NaT
        else:
            # If enabled_time exists, still create the concurrency oracle for later use
            config = Configuration(
                log_ids=self.event_log_ids,
                concurrency_thresholds=ConcurrencyThresholds(df=0.5)
            )
            self.concurrency_oracle = OverlappingConcurrencyOracle(df, config)
        return df
