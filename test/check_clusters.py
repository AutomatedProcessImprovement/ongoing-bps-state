from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main() -> None:
    # ------------------------------------------------------------------
    # CONFIG – adjust these two to point to the run you want to analyse
    # ------------------------------------------------------------------
    DATASET = "LOAN_STABLE"
    RUN_ID = "su0f4f"  # folder created by clustered_short_term_simulation.py
    ROOT = Path("outputs_confidence") / DATASET / RUN_ID

    test_eval_path = ROOT / "test_evaluation.csv"
    test_samples_path = ROOT / "test_samples.csv"

    print(f"Loading:\n  {test_eval_path}\n  {test_samples_path}\n")

    df_eval = pd.read_csv(test_eval_path)
    df_samples = pd.read_csv(test_samples_path)

    # ------------------------------------------------------------------
    # 1. Make sure we have a common key for merging: cut_time_iso
    #    - samples: cut_time_iso already there
    #    - eval   : has 'cut_time' column; copy/rename to cut_time_iso
    # ------------------------------------------------------------------
    if "cut_time_iso" not in df_eval.columns and "cut_time" in df_eval.columns:
        df_eval["cut_time_iso"] = df_eval["cut_time"]

    # sanity check
    if "cut_time_iso" not in df_eval.columns or "cut_time_iso" not in df_samples.columns:
        raise RuntimeError("Could not find 'cut_time_iso' in both files – "
                           "check column names in test_evaluation.csv and test_samples.csv")

    # ------------------------------------------------------------------
    # 2. Merge features + evaluation so we have everything in one DataFrame
    # ------------------------------------------------------------------
    df = df_samples.merge(
        df_eval,
        on="cut_time_iso",
        suffixes=("_feat", "_eval"),  # guard against accidental collisions
    )

    print(f"Merged dataframe shape: {df.shape}")
    print("Columns available:", ", ".join(df.columns))
    print()

    # ------------------------------------------------------------------
    # 3. Method-wise statistics: coverage & average CI width
    # ------------------------------------------------------------------
    # name, in_ci_column, ci_width_column, human_readable_label
    METHODS = [
        ("baseline", "baseline_in_ci", "baseline_ci", "Global baseline"),
        ("wip_deciles", "wip_decile_in_ci", "wip_decile_ci", "WIP decile groups"),
        ("kmeans_simple", "kmeans_simple_in_ci", "kmeans_simple_ci",
         "K-means (simple features)"),
        ("kmeans_advanced_wip", "kmeans_advanced_wip_in_ci", "kmeans_advanced_wip_ci",
         "K-means (advanced WIP)"),
        ("kmeans_statevec", "kmeans_statevec_in_ci", "kmeans_statevec_ci",
         "K-means (activity state vector)"),
    ]

    summaries = []
    for key, in_col, ci_col, label in METHODS:
        if in_col not in df_eval.columns or ci_col not in df_eval.columns:
            # this method wasn't trained or produced no results
            continue

        mask = df_eval[in_col].notna() & df_eval[ci_col].notna()
        if not mask.any():
            continue

        coverage = df_eval.loc[mask, in_col].mean()  # fraction of True
        avg_ci = df_eval.loc[mask, ci_col].mean()
        avg_abs_err = df_eval.loc[mask, "true_error"].abs().mean()

        summaries.append({
            "method_key": key,
            "method_label": label,
            "n_points": int(mask.sum()),
            "coverage": float(coverage),
            "avg_ci_width": float(avg_ci),
            "avg_abs_error": float(avg_abs_err),
        })

    summary_df = pd.DataFrame(summaries)
    if summary_df.empty:
        print("No methods found in test_evaluation.csv – nothing to summarise.")
        return

    # nicer order: sort by coverage descending, then by avg_ci_width ascending
    summary_df = summary_df.sort_values(["coverage", "avg_ci_width"],
                                        ascending=[False, True])

    print("=== Method comparison (higher coverage, smaller CI width is better) ===")
    print(summary_df.to_string(index=False))
    print()

    # ------------------------------------------------------------------
    # 4. Plots – coverage & CI width per method
    # ------------------------------------------------------------------
    # 4.1 Coverage bar plot
    plt.figure(figsize=(8, 4))
    plt.bar(summary_df["method_label"], summary_df["coverage"])
    plt.ylim(0, 1.05)
    plt.axhline(0.95, linestyle="--")  # nominal 95% line (just a reference)
    plt.ylabel("Coverage (fraction of test points within CI)")
    plt.title(f"{DATASET} – CI coverage per method")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.show()

    # 4.2 Average CI width bar plot
    plt.figure(figsize=(8, 4))
    plt.bar(summary_df["method_label"], summary_df["avg_ci_width"])
    plt.ylabel("Average CI half-width")
    plt.title(f"{DATASET} – Average CI width per method")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # 5. Scatter plot: WIP vs true error, coloured by K-means-simple cluster
    # ------------------------------------------------------------------
    if "kmeans_simple_id" in df_eval.columns:
        merged_scatter = df_samples.merge(
            df_eval[["cut_time_iso", "kmeans_simple_id", "true_error"]],
            on="cut_time_iso",
        )

        # drop NaNs in the plotted columns
        mask = merged_scatter["wip"].notna() & merged_scatter["true_error"].notna()
        merged_scatter = merged_scatter.loc[mask]

        if not merged_scatter.empty:
            plt.figure(figsize=(7, 5))
            sc = plt.scatter(
                merged_scatter["wip"],
                merged_scatter["true_error"],
                c=merged_scatter["kmeans_simple_id"],
                alpha=0.75,
            )
            plt.xlabel("WIP (number of ongoing cases)")
            plt.ylabel("True error (target metric)")
            plt.title(f"{DATASET} – WIP vs error coloured by K-means (simple)")
            cbar = plt.colorbar(sc)
            cbar.set_label("K-means (simple) cluster ID")
            plt.tight_layout()
            plt.show()
        else:
            print("No non-NaN data available to plot WIP vs error scatter.")
    else:
        print("Column 'kmeans_simple_id' not found – skipping WIP vs error scatter.")

    # ------------------------------------------------------------------
    # 6. Optional: you can add more plots here, for example:
    #    - per-cluster coverage/CI (for a specific method),
    #    - histograms of true_error,
    #    - reliability diagrams, etc.
    # ------------------------------------------------------------------


if __name__ == "__main__":
    main()
