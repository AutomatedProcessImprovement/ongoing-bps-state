#!/usr/bin/env python
"""
Fix canonical log timestamps to match the "good" format:

Input example (current):
    case_id,resource,role,activity,FixedCost,ResourceCost,start_time,end_time,enabled_time
    0,Clerk-000001,Clerk,Check application form completeness,0.0,0.4,
      2025-01-01T07:00:00.762000Z,2025-01-01T07:23:43.902000Z,2025-01-01 07:00:00.762000+0000

Output example (desired):
    case_id,resource,role,activity,FixedCost,ResourceCost,start_time,end_time,enable_time
    0,Clerk-000001,Clerk,Check application form completeness,0.0,0.4,
      2025-01-01T07:00:00.762,2025-01-01T07:23:43.902,2025-01-01T07:00:00.762

Usage:
    python fix_canonical_timestamps.py path/to/log.csv
    python fix_canonical_timestamps.py path/to/log.csv --inplace
"""

import argparse
from pathlib import Path
import sys

import pandas as pd


def _format_to_ms(dt: pd.Series) -> pd.Series:
    """
    Convert a datetime series to a string with milliseconds precision:
        YYYY-MM-DDTHH:MM:SS.sss
    """
    # Full microseconds
    s = dt.dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
    # Keep only 3 decimal places (strip last 3 digits)
    return s.str.slice(0, 23)


def fix_canonical_log(
    input_path: Path,
    output_path: Path,
    start_col: str = "start_time",
    end_col: str = "end_time",
    enabled_col: str = "enabled_time",
    enable_col_out: str = "enable_time",
) -> None:
    print(f"[fix_canonical] Reading {input_path}")
    df = pd.read_csv(input_path)

    missing = [c for c in [start_col, end_col, enabled_col] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input must contain {start_col}, {end_col}, {enabled_col}. "
            f"Missing: {missing}. Columns found: {list(df.columns)}"
        )

    # Parse all three columns as datetimes (they might have Z, +0000, etc.)
    print("[fix_canonical] Parsing timestamps as datetimes (utc=True)...")
    start_dt = pd.to_datetime(df[start_col], utc=True, errors="coerce")
    end_dt = pd.to_datetime(df[end_col], utc=True, errors="coerce")
    enabled_dt = pd.to_datetime(df[enabled_col], utc=True, errors="coerce")

    # optional debug
    n_bad_start = int(start_dt.isna().sum())
    n_bad_end = int(end_dt.isna().sum())
    n_bad_enabled = int(enabled_dt.isna().sum())
    if n_bad_start or n_bad_end or n_bad_enabled:
        print(
            f"[fix_canonical] WARNING: failed to parse "
            f"{n_bad_start} start_time, {n_bad_end} end_time, "
            f"{n_bad_enabled} enabled_time values."
        )

    # Drop timezone info (naive) and format to "YYYY-MM-DDTHH:MM:SS.sss"
    start_dt = start_dt.dt.tz_convert("UTC").dt.tz_localize(None)
    end_dt = end_dt.dt.tz_convert("UTC").dt.tz_localize(None)
    enabled_dt = enabled_dt.dt.tz_convert("UTC").dt.tz_localize(None)

    df[start_col] = _format_to_ms(start_dt)
    df[end_col] = _format_to_ms(end_dt)

    # rename enabled_time → enable_time
    df[enable_col_out] = _format_to_ms(enabled_dt)
    if enabled_col != enable_col_out and enabled_col in df.columns:
        df = df.drop(columns=[enabled_col])

    print(f"[fix_canonical] Writing cleaned log to {output_path}")
    df.to_csv(output_path, index=False)
    print("[fix_canonical] Done.")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Fix canonical event-log timestamps (start/end/enable)."
    )
    parser.add_argument("input_csv", type=str, help="Path to the input CSV.")
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite the input file instead of creating *-fixed.csv.",
    )
    parser.add_argument(
        "--start-col",
        type=str,
        default="start_time",
        help="Name of the start timestamp column (default: start_time).",
    )
    parser.add_argument(
        "--end-col",
        type=str,
        default="end_time",
        help="Name of the end timestamp column (default: end_time).",
    )
    parser.add_argument(
        "--enabled-col",
        type=str,
        default="enabled_time",
        help="Name of the 'enabled' timestamp column (default: enabled_time).",
    )
    parser.add_argument(
        "--enable-col-out",
        type=str,
        default="enable_time",
        help="Name of the output column for enabled timestamps (default: enable_time).",
    )

    args = parser.parse_args(argv)
    input_path = Path(args.input_csv)

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if args.inplace:
        output_path = input_path
    else:
        output_path = input_path.with_name(
            input_path.stem + "-fixed" + input_path.suffix
        )

    fix_canonical_log(
        input_path=input_path,
        output_path=output_path,
        start_col=args.start_col,
        end_col=args.end_col,
        enabled_col=args.enabled_col,
        enable_col_out=args.enable_col_out,
    )


if __name__ == "__main__":
    main()
