#!/usr/bin/env python3
import pandas as pd
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parents[2]
    input_csv = repo_root / "output" / "ml_results" / "nn_stageA_inventory.csv"
    output_csv = repo_root / "output" / "ml_results" / "nn_stageA_summary_by_mode_and_constraint.csv"
    shortlist_path = repo_root / "output" / "ml_results" / "nn_stageA_stageB_shortlist.txt"

    df = pd.read_csv(input_csv)

    # Normalize column names to requested output labels
    column_map = {
        "precision": "test_precision",
        "recall": "test_recall",
        "specificity": "test_specificity",
        "f1": "test_f1_score",
        "balanced_accuracy": "test_balanced_accuracy",
        "roc_auc": "test_roc_auc",
        "pr_auc": "test_pr_auc",
        "decision_threshold": "decision_threshold",
    }

    for src, dst in column_map.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]

    required_cols = [
        "nn_imbalance_mode",
        "min_recall",
        "test_precision",
        "test_recall",
        "test_specificity",
        "test_f1_score",
        "test_balanced_accuracy",
        "test_roc_auc",
        "test_pr_auc",
        "decision_threshold",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in inventory: {missing}")

    summary = (
        df[required_cols]
        .groupby(["nn_imbalance_mode", "min_recall"], as_index=False)
        .mean(numeric_only=True)
    )

    # Rank within each min_recall
    summary["rank_specificity_precision"] = (
        summary.sort_values(
            ["min_recall", "test_specificity", "test_precision"],
            ascending=[True, False, False]
        )
        .groupby("min_recall")
        .cumcount() + 1
    )

    summary.to_csv(output_csv, index=False)

    for min_recall in [0.85, 0.9]:
        subset = summary[summary["min_recall"] == min_recall].copy()
        if subset.empty:
            print(f"min_recall={min_recall}: no rows found")
            continue

        subset = subset.sort_values(
            ["test_specificity", "test_precision"],
            ascending=[False, False]
        )

        print(f"\nRanking for min_recall={min_recall}")
        print(subset[[
            "nn_imbalance_mode",
            "test_specificity",
            "test_precision",
            "rank_specificity_precision",
        ]].to_string(index=False))

        winner = subset.iloc[0]
        print(
            f"Winner for min_recall={min_recall}: {winner['nn_imbalance_mode']} "
            f"(specificity={winner['test_specificity']:.4f}, "
            f"precision={winner['test_precision']:.4f})"
        )

    # Stage B shortlist
    shortlist = {}
    for min_recall in [0.85, 0.9]:
        subset = summary[summary["min_recall"] == min_recall].copy()
        if subset.empty:
            continue
        subset = subset.sort_values(
            ["test_specificity", "test_precision"],
            ascending=[False, False]
        )
        top_modes = subset["nn_imbalance_mode"].head(2).tolist()
        shortlist[min_recall] = top_modes

    shortlist_modes = []
    if 0.85 in shortlist and 0.9 in shortlist:
        if shortlist[0.85] and shortlist[0.9] and shortlist[0.85][0] == shortlist[0.9][0]:
            shortlist_modes.append(shortlist[0.85][0])
            if len(shortlist[0.85]) > 1:
                shortlist_modes.append(shortlist[0.85][1])
        else:
            shortlist_modes.extend(shortlist.get(0.85, []))
            shortlist_modes.extend(shortlist.get(0.9, []))
    else:
        for modes in shortlist.values():
            shortlist_modes.extend(modes)

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for mode in shortlist_modes:
        if mode not in seen:
            seen.add(mode)
            deduped.append(mode)

    if deduped:
        print("\nStage B shortlist:")
        for mode in deduped:
            print(f"- {mode}")

        with shortlist_path.open("w", encoding="utf-8") as f:
            f.write("Stage B shortlist (modes):\n")
            for mode in deduped:
                f.write(f"- {mode}\n")
    else:
        print("\nStage B shortlist: no candidates found")
        with shortlist_path.open("w", encoding="utf-8") as f:
            f.write("Stage B shortlist: no candidates found\n")


if __name__ == "__main__":
    main()
