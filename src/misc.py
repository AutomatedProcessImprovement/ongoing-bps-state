#!/usr/bin/env python
"""
process_event_log_recalc_enabled.py

This script reads an event log CSV, remaps case_id to consecutive integers starting at 0,
recalculates the enabled_time using pix‑framework’s OverlappingConcurrencyOracle, and formats
the enabled_time to ISO8601 (with a T separator and three-digit milliseconds).

IMPORTANT: This script does not change the start_time or end_time values from the input.
They are used (after conversion) only for computing enabled_time, and then the original string
values are restored in the output.
"""

import pandas as pd
from pathlib import Path
import json

# --- PIX-FRAMEWORK IMPORTS ---
from pix_framework.io.event_log import EventLogIDs
from pix_framework.enhancement.start_time_estimator.config import Configuration, ConcurrencyThresholds
from pix_framework.enhancement.concurrency_oracle import OverlappingConcurrencyOracle

# ============================
# Configuration: update these paths.
# ============================
INPUT_FILE = "samples/real_life/BPIC_2012.csv"    # Path to your input CSV event log
OUTPUT_FILE = "samples/real_life/BPIC_2012_fixed.csv"  # Path to the output CSV file

def format_enabled_time(ts):
    """If ts is a pandas Timestamp, return ISO8601 string with T and three-digit ms."""
    if pd.isna(ts):
        return ""
    return ts.isoformat(timespec='milliseconds')

def main():
    print(f"Reading input CSV from: {INPUT_FILE}")
    # Read the original CSV
    df_orig = pd.read_csv(INPUT_FILE)
    # Keep copies of the original start_time and end_time strings.
    original_start = df_orig['start_time'].copy() if 'start_time' in df_orig.columns else None
    original_end = df_orig['end_time'].copy() if 'end_time' in df_orig.columns else None

    # Work on a copy for processing.
    df = df_orig.copy()

    # --- Step 1: Remap case_id to consecutive integers starting at 0 ---
    original_case_ids = df['case_id'].unique()
    case_mapping = {orig: new for new, orig in enumerate(original_case_ids)}
    print("Mapping of original case_id to new case_id:")
    for orig, new in case_mapping.items():
        print(f"  {orig} -> {new}")
    df['case_id'] = df['case_id'].map(case_mapping)

    # --- Step 2: Parse start_time and end_time for concurrency calculations ---
    # We do not want to change the original text for these columns, so we use our own parsed copies.
    # Use pd.to_datetime without specifying a strict format so that various ISO-like strings are accepted.
    df['start_time_parsed'] = pd.to_datetime(df['start_time'], utc=True, errors='coerce')
    df['end_time_parsed'] = pd.to_datetime(df['end_time'], utc=True, errors='coerce')

    # --- Step 3: Prepare pix-framework’s EventLogIDs ---
    ids = EventLogIDs(
        case="case_id",
        activity="activity",
        resource="resource",
        start_time="start_time_parsed",
        end_time="end_time_parsed",
        enabled_time="enabled_time"  # This column will be updated.
    )

    # --- Step 4: For events missing an end_time (i.e. ongoing events), fill temporarily ---
    if df['end_time_parsed'].isna().any():
        filler = df['end_time_parsed'].max() + pd.Timedelta(hours=1)
        df["_was_missing_end"] = df['end_time_parsed'].isna()
        df['end_time_parsed'] = df['end_time_parsed'].fillna(filler)
    else:
        df["_was_missing_end"] = False

    # --- Step 5: Create the concurrency oracle and compute enabled_time ---
    config = Configuration(
        log_ids=ids,
        concurrency_thresholds=ConcurrencyThresholds(df=0.5)  # Example threshold; adjust if needed.
    )
    concurrency_oracle = OverlappingConcurrencyOracle(df, config)
    print("Calculating enabled_time using OverlappingConcurrencyOracle.add_enabled_times() ...")
    concurrency_oracle.add_enabled_times(df)

    # --- Step 6: Revert temporary end_time fillers if used ---
    if df["_was_missing_end"].any():
        df.loc[df["_was_missing_end"], "end_time_parsed"] = pd.NaT
    df.drop(columns="_was_missing_end", inplace=True)

    # --- Step 7: Format the enabled_time column in ISO8601 with T and three-digit milliseconds ---
    df['enabled_time'] = df['enabled_time'].apply(format_enabled_time)

    # --- Step 8: Restore the original start_time and end_time text values (unchanged) ---
    if original_start is not None:
        df['start_time'] = original_start
    if original_end is not None:
        df['end_time'] = original_end
    # Optionally, remove our temporary parsed columns.
    df.drop(columns=['start_time_parsed', 'end_time_parsed'], inplace=True)
    df.drop(columns=['variant_index', 'variant', 'creator'], inplace=True)

    # --- Step 9: Write the updated DataFrame to the output CSV ---
    print(f"Writing output CSV to: {OUTPUT_FILE}")
    df.to_csv(OUTPUT_FILE, index=False)
    print("Processing complete.")

if __name__ == "__main__":
    main()
