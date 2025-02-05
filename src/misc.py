# import pandas as pd

# def transform_case_ids(
#     input_csv_path: str,
#     output_csv_path: str,
#     case_column: str = "case_id"
# ) -> None:
#     """
#     Reads an event log from `input_csv_path`,
#     converts the case IDs in `case_column` to integers 0..N-1,
#     writes the transformed event log to `output_csv_path`.

#     - Any events sharing the original case ID remain grouped under the same new integer ID.
#     - The order of the new IDs depends on the sorted order of the *unique* original case IDs.
#       (You can modify if you prefer first-appearance order.)
#     """

#     # 1) Read the event log
#     df = pd.read_csv(input_csv_path)

#     # 2) Extract unique case IDs and sort them
#     unique_ids = df[case_column].unique()
#     unique_ids_sorted = sorted(unique_ids)  # or just `unique_ids` if you want first-appearance order

#     # 3) Build a mapping from old -> new
#     #    E.g., if unique_ids_sorted == [100, 198639, 200, 300],
#     #    then the mapping is {100: 0, 198639: 1, 200: 2, 300: 3}.
#     mapping = {old_id: new_id for new_id, old_id in enumerate(unique_ids_sorted)}

#     # 4) Apply the mapping to the DataFrame
#     df[case_column] = df[case_column].map(mapping)

#     # 5) Save the transformed DataFrame to disk
#     df.to_csv(output_csv_path, index=False)

#     # 6) (Optional) print or return the mapping if you want
#     print("Mapping of old CaseIds -> new CaseIds:")
#     for old_id, new_id in mapping.items():
#         print(f"  {old_id} -> {new_id}")


# if __name__ == "__main__":
#     # Example usage
#     input_csv = "samples/real_life/BPIC_2012.csv"    # Replace with your actual input file
#     output_csv = "samples/real_life/BPIC_2012_changed.csv"  # Replace with the desired output file
#     transform_case_ids(input_csv, output_csv, case_column="case_id")
import pandas as pd
import re
import json

"""
This script tries to replicate the transformations in your input_handler.py, including:
1) optional JSON column mapping
2) renaming columns to standard names
3) ensure fractional seconds
4) parsing with pd.to_datetime
5) printing which rows end up as NaT
6) investigating distinct timestamp formats
"""

def parse_column_mapping(mapping_str: str):
    """
    Replicates your parse_column_mapping logic:
    - loads from JSON string
    - ensures we have 'CaseId', 'Resource', 'Activity', 'StartTime', 'EndTime'
    """
    mapping = json.loads(mapping_str)
    required_standard_names = ['CaseId', 'Resource', 'Activity', 'StartTime', 'EndTime']
    provided_standard_names = set(mapping.values())
    for std_name in required_standard_names:
        if std_name not in provided_standard_names:
            # If the standard name is not in the mapping, assume the column is already named as that standard name
            mapping[std_name] = std_name
    return mapping


def ensure_fractional_seconds(ts: str) -> str:
    """
    Same logic as in your input_handler: insert ".000" if there's no decimal at all.
    We'll do a slightly flexible approach: if there's a 'Txx:xx:xx'
    then we insert .000. Otherwise just append .000
    """
    if pd.isnull(ts):
        return ts  # preserve NaN / None

    # if there's already a decimal, do nothing
    if '.' in ts:
        return ts

    # Attempt to insert .000 just before any offset or 'Z'
    match = re.match(r'^(.*T\d{2}:\d{2}:\d{2})(.*)$', ts)
    if match:
        main_time = match.group(1)
        remainder = match.group(2)  # might be '', 'Z', '+01:00', etc.
        ts = main_time + ".000" + remainder
    else:
        # If it doesn't match that pattern, just append ".000"
        ts += ".000"

    return ts


def investigate_like_input_handler(
    csv_path: str,
    column_mapping_str: str = '{"CaseId":"CaseId","Resource":"Resource","Activity":"Activity","StartTime":"StartTime","EndTime":"EndTime"}',
    start_col_std: str = "StartTime",
    end_col_std: str = "EndTime",
    max_show: int = 10
):
    """
    1. Reads CSV
    2. Applies column mapping
    3. Validates required columns
    4. For each time col, does the ensure_fractional_seconds + to_datetime(utc=True, errors='coerce')
    5. Prints how many are NaT, plus examples
    6. Prints distinct example formats of original strings
    """

    print(f"=== Reading CSV from: {csv_path} ===")
    df = pd.read_csv(csv_path)

    print("\n=== 1) Parsing column mapping from JSON string ===")
    colmap = parse_column_mapping(column_mapping_str)
    print("Column mapping resolved as:", colmap)

    print("\n=== 2) Renaming columns to standard names ===")
    df = df.rename(columns=colmap)

    print(f"DataFrame columns after rename: {df.columns.tolist()}")

    required_cols = ["CaseId", "Activity", "Resource", "StartTime", "EndTime"]
    for col in required_cols:
        if col not in df.columns:
            print(f"ERROR: Missing required column: {col}")
            return

    # Let's store the original strings in separate columns for debugging
    df["_original_start"] = df["StartTime"].astype(str)
    df["_original_end"]   = df["EndTime"].astype(str)

    print("\n=== 3) Ensuring fractional seconds before parsing ===")
    df["StartTime"] = df["StartTime"].astype(str).apply(ensure_fractional_seconds)
    df["EndTime"]   = df["EndTime"].astype(str).apply(ensure_fractional_seconds)

    print("\n=== 4) Parsing with pd.to_datetime(utc=True, errors='coerce') ===")
    df["StartTime"] = pd.to_datetime(df["StartTime"], utc=True, errors="coerce")
    df["EndTime"]   = pd.to_datetime(df["EndTime"],   utc=True, errors="coerce")

    print("\n=== 5) Checking for NaT (unparsed timestamps) ===")
    bad_start = df[df["StartTime"].isna()]
    bad_end   = df[df["EndTime"].isna()]

    # Summaries
    print(f"[StartTime]: {len(bad_start)} rows with NaT out of {len(df)} total.")
    print(f"[EndTime]: {len(bad_end)} rows with NaT out of {len(df)} total.")

    # Show examples
    if not bad_start.empty:
        print(f"\n--- Rows with NaT in StartTime (showing up to {max_show}):")
        print(bad_start[["_original_start"]].head(max_show))
    if not bad_end.empty:
        print(f"\n--- Rows with NaT in EndTime (showing up to {max_show}):")
        print(bad_end[["_original_end"]].head(max_show))

    print("\n=== 6) Distinct example formats of original strings ===")
    # We'll see a sample of distinct formats from the original columns
    # for both StartTime and EndTime, to see the variety in your data
    # We'll just do a small sample for brevity
    n_sample = 20

    start_strings = df["_original_start"].dropna().unique().tolist()
    end_strings   = df["_original_end"].dropna().unique().tolist()

    # We'll show up to 20 from each, but you can tweak
    sample_start_strings = start_strings[:n_sample]
    sample_end_strings   = end_strings[:n_sample]

    print(f"\nDistinct sample of StartTime strings ({len(start_strings)} unique total, showing {len(sample_start_strings)}):")
    for s in sample_start_strings:
        print(" ", s)

    print(f"\nDistinct sample of EndTime strings ({len(end_strings)} unique total, showing {len(sample_end_strings)}):")
    for s in sample_end_strings:
        print(" ", s)

    print("\n=== Investigation complete. ===")
    return df, bad_start, bad_end


if __name__ == "__main__":
    # Example usage:
    csv_path = "samples/real_life/BPIC_2012.csv"  # adapt this as needed
    column_mapping_str = '{"case_id":"CaseId","activity":"Activity","resource":"Resource","start_time":"StartTime","end_time":"EndTime"}'

    investigate_like_input_handler(
        csv_path=csv_path,
        column_mapping_str=column_mapping_str,
        start_col_std="start_time",
        end_col_std="end_time",
    )
