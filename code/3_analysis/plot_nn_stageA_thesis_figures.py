#!/usr/bin/env python3
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def add_bar_labels(ax):
    for bar in ax.patches:
        height = bar.get_height()
        if math.isnan(height):
            continue
        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_bar_by_min_recall(df, min_recall, metric, out_dir, filename_prefix, title):
    sub = df[df["min_recall"] == min_recall].copy()
    if sub.empty:
        return None

    sub = sub.sort_values("nn_imbalance_mode")
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(sub["nn_imbalance_mode"], sub[metric], color="#4C72B0")
    ax.set_title(title)
    ax.set_xlabel("Imbalance mode")
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=25)
    add_bar_labels(ax)
    fig.tight_layout()

    out_path = out_dir / f"{filename_prefix}_minrecall{min_recall}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_scatter_by_min_recall(df, min_recall, out_dir, filename_prefix, title):
    sub = df[df["min_recall"] == min_recall].copy()
    if sub.empty:
        return None

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(sub["test_recall"], sub["test_specificity"], color="#55A868")

    for _, row in sub.iterrows():
        ax.annotate(
            row["nn_imbalance_mode"],
            (row["test_recall"], row["test_specificity"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    ax.set_title(title)
    ax.set_xlabel("Test recall")
    ax.set_ylabel("Test specificity")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()

    out_path = out_dir / f"{filename_prefix}_minrecall{min_recall}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main():
    repo_root = Path(__file__).resolve().parents[2]
    input_csv = repo_root / "output" / "ml_results" / "nn_stageA_inventory.csv"
    plots_dir = repo_root / "output" / "ml_results" / "plots" / "phase2_nn_seed42_stageA"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    # Normalize columns from inventory
    if "test_specificity" not in df.columns and "specificity" in df.columns:
        df["test_specificity"] = df["specificity"]
    if "test_recall" not in df.columns and "recall" in df.columns:
        df["test_recall"] = df["recall"]

    min_recalls = [0.85, 0.9]
    saved = []

    for min_recall in min_recalls:
        saved.append(plot_bar_by_min_recall(
            df,
            min_recall,
            metric="test_specificity",
            out_dir=plots_dir,
            filename_prefix="FIG1_test_specificity_by_mode",
            title=f"Test Specificity by Mode (min_recall={min_recall})",
        ))
        saved.append(plot_bar_by_min_recall(
            df,
            min_recall,
            metric="test_recall",
            out_dir=plots_dir,
            filename_prefix="FIG2_test_recall_by_mode",
            title=f"Test Recall by Mode (min_recall={min_recall})",
        ))
        saved.append(plot_scatter_by_min_recall(
            df,
            min_recall,
            out_dir=plots_dir,
            filename_prefix="FIG3_recall_vs_specificity",
            title=f"Recall vs Specificity (min_recall={min_recall})",
        ))

    for path in saved:
        if path:
            print(f"Saved: {path}")


if __name__ == "__main__":
    main()
