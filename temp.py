# tools/split_synthetic_logs_timewise_50_50.py
"""
Time-based 50/50 split for ICPM-2025 synthetic logs, while keeping full cases intact
(i.e., no case appears in both splits).

Split rule:
- Order cases by their first start timestamp (chronological case order).
- TRAIN gets the earliest 50% of cases, TEST gets the latest 50%.
- All events of a case stay in its split.

Also:
- Reindex case_id to 1..N within each split.

Example:
  in : samples/icpm-2025/synthetic/Loan-stable.csv
  out: samples/extension-uncertainty/synthetic-v2/Loan-stable/Loan-stable-train.csv
       samples/extension-uncertainty/synthetic-v2/Loan-stable/Loan-stable-test.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


DEFAULT_INPUTS = [
    "samples/icpm-2025/synthetic/Loan-stable.csv",
    "samples/icpm-2025/synthetic/Loan-circadian.csv",
    "samples/icpm-2025/synthetic/Loan-unpredictable.csv",
    "samples/icpm-2025/synthetic/P2P-stable.csv",
    "samples/icpm-2025/synthetic/P2P-circadian.csv",
    "samples/icpm-2025/synthetic/P2P-unstable.csv",
]


CASE_COL_CANDIDATES = ["case_id", "CaseId", "caseid", "Case ID", "Case_ID"]
START_COL_CANDIDATES = ["start_time", "StartTime", "start", "Start", "startTimestamp"]


def detect_col(df: pd.DataFrame, candidates: list[str], what: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find {what} column. Tried {candidates}. Found {list(df.columns)}")


def read_csv_any(path: Path) -> pd.DataFrame:
    # Works for .csv and .csv.gz
    return pd.read_csv(path, low_memory=False)


def timewise_case_split_50_50(df: pd.DataFrame, case_col: str, start_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Parse timestamps (robust)
    start_dt = pd.to_datetime(df[start_col], errors="coerce", utc=True)
    if start_dt.isna().all():
        raise ValueError(f"Could not parse any datetimes from column '{start_col}'")

    df = df.copy()
    df["_start_dt"] = start_dt

    # Case order by first start time
    case_first_start = (
        df.groupby(case_col, sort=False)["_start_dt"]
          .min()
          .sort_values(kind="mergesort")  # stable sort
    )
    ordered_cases = case_first_start.index.tolist()
    n_cases = len(ordered_cases)
    if n_cases < 2:
        raise ValueError(f"Not enough cases to split (found {n_cases})")

    n_train = n_cases // 2  # earliest half
    train_cases = set(ordered_cases[:n_train])
    test_cases = set(ordered_cases[n_train:])

    train_df = df[df[case_col].isin(train_cases)].copy()
    test_df = df[df[case_col].isin(test_cases)].copy()

    # Drop helper col
    train_df.drop(columns=["_start_dt"], inplace=True)
    test_df.drop(columns=["_start_dt"], inplace=True)

    return train_df, test_df


def reindex_case_ids_by_time(df: pd.DataFrame, case_col: str, start_col: str) -> pd.DataFrame:
    df = df.copy()
    start_dt = pd.to_datetime(df[start_col], errors="coerce", utc=True)
    df["_start_dt"] = start_dt

    # Order cases within this split
    case_order = (
        df.groupby(case_col, sort=False)["_start_dt"]
          .min()
          .sort_values(kind="mergesort")
          .index.tolist()
    )
    mapping = {old: i + 1 for i, old in enumerate(case_order)}
    df[case_col] = df[case_col].map(mapping).astype(int)

    df.drop(columns=["_start_dt"], inplace=True)
    return df


def split_one_file(input_csv: Path, out_root: Path) -> tuple[Path, Path]:
    df = read_csv_any(input_csv)

    case_col = detect_col(df, CASE_COL_CANDIDATES, "case-id")
    start_col = detect_col(df, START_COL_CANDIDATES, "start-time")

    train_df, test_df = timewise_case_split_50_50(df, case_col=case_col, start_col=start_col)

    # Reindex case ids starting from 1 in each split
    train_df = reindex_case_ids_by_time(train_df, case_col=case_col, start_col=start_col)
    test_df = reindex_case_ids_by_time(test_df, case_col=case_col, start_col=start_col)

    stem = input_csv.stem
    ds_out_dir = out_root / stem
    ds_out_dir.mkdir(parents=True, exist_ok=True)

    train_out = ds_out_dir / f"{stem}-train.csv"
    test_out = ds_out_dir / f"{stem}-test.csv"

    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)

    # Quick sanity check (no overlap)
    # train_cases = set(train_df[case_col].unique())
    # test_cases = set(test_df[case_col].unique())
    # if train_cases.intersection(test_cases):
    #     raise RuntimeError("Sanity check failed: case ids overlap after reindexing (should never happen).")

    return train_out, test_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-root",
        type=str,
        default="samples/extension-uncertainty/synthetic-v2",
        help="Root directory where per-dataset folders will be created.",
    )
    ap.add_argument(
        "--inputs",
        nargs="*",
        default=DEFAULT_INPUTS,
        help="Input logs to split (default: all ICPM-2025 synthetic logs).",
    )
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for inp in args.inputs:
        inp_path = Path(inp)
        print(f"\n[split] Input: {inp_path}")
        train_out, test_out = split_one_file(inp_path, out_root=out_root)
        print(f"[split] Wrote: {train_out}")
        print(f"[split] Wrote: {test_out}")


if __name__ == "__main__":
    main()
