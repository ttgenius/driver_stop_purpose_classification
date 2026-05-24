from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHAPTER4_CSV_PATH = (
    PROJECT_ROOT
    / "output"
    / "ml_results"
    / "saved_ml_performance_metrics"
    / "phase1_lgbm_3seeds"
    / "phase1_lgbm_3seeds_metrics_summary_with_std.csv"
)
CHAPTER6_CSV_PATH = (
    PROJECT_ROOT
    / "output"
    / "ml_results"
    / "saved_ml_performance_metrics"
    / "phase3_with_naics"
    / "phase3_with_naics_metrics_summary_with_std.csv"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "output"
    / "ml_results"
    / "saved_ml_performance_metrics"
    / "phase3_with_naics"
    / "chapter4_baseline_vs_chapter6_final_lgbm_metrics_with_std.png"
)

CHAPTER4_LABEL = "Chapter 4 Baseline LightGBM"
CHAPTER6_LABEL = "Chapter 6 Final LightGBM"

METRICS = [
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("specificity", "Specificity"),
    ("f1_score", "F1 Score"),
    ("balanced_accuracy", "Balanced Accuracy"),
    ("roc_auc", "ROC AUC"),
    ("pr_auc", "PR AUC"),
]


def load_lgbm_row(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"No rows found in {csv_path}")

    if "model_name" in df.columns:
        lgbm_rows = df.loc[df["model_name"].astype(str).str.upper().eq("LGBM")]
        if not lgbm_rows.empty:
            return lgbm_rows.iloc[0]

    if len(df) == 1:
        return df.iloc[0]

    raise ValueError(f"No model_name='LGBM' row found in {csv_path}")


def build_metric_frame(chapter4_row: pd.Series, chapter6_row: pd.Series) -> pd.DataFrame:
    records = []
    for metric_key, metric_label in METRICS:
        mean_col = f"{metric_key}_mean"
        std_col = f"{metric_key}_std"
        missing = [
            col
            for col in (mean_col, std_col)
            if col not in chapter4_row.index or col not in chapter6_row.index
        ]
        if missing:
            raise ValueError(f"Missing expected columns for {metric_label}: {missing}")

        records.append(
            {
                "metric": metric_label,
                "chapter4_mean": chapter4_row[mean_col],
                "chapter4_std": chapter4_row[std_col],
                "chapter6_mean": chapter6_row[mean_col],
                "chapter6_std": chapter6_row[std_col],
                "delta_mean": chapter6_row[mean_col] - chapter4_row[mean_col],
            }
        )
    return pd.DataFrame(records)


def plot_metric_comparison(
    metrics_df: pd.DataFrame,
    chapter4_row: pd.Series,
    chapter6_row: pd.Series,
    output_path: Path,
) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
        }
    )

    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)

    x = np.arange(len(metrics_df))
    width = 0.36
    chapter4_color = "#4C78A8"
    chapter6_color = "#F58518"

    ax.bar(
        x - width / 2,
        metrics_df["chapter4_mean"],
        width=width,
        yerr=metrics_df["chapter4_std"],
        capsize=4,
        label=CHAPTER4_LABEL,
        color=chapter4_color,
        alpha=0.92,
    )
    ax.bar(
        x + width / 2,
        metrics_df["chapter6_mean"],
        width=width,
        yerr=metrics_df["chapter6_std"],
        capsize=4,
        label=CHAPTER6_LABEL,
        color=chapter6_color,
        alpha=0.92,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df["metric"])
    ax.set_ylabel("Mean score")
    ax.set_title("Chapter 4 Baseline LightGBM vs Chapter 6 Final LightGBM")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right")

    metric_min = min(
        (metrics_df["chapter4_mean"] - metrics_df["chapter4_std"]).min(),
        (metrics_df["chapter6_mean"] - metrics_df["chapter6_std"]).min(),
    )
    metric_max = max(
        (metrics_df["chapter4_mean"] + metrics_df["chapter4_std"]).max(),
        (metrics_df["chapter6_mean"] + metrics_df["chapter6_std"]).max(),
    )
    margin = max(0.02, (metric_max - metric_min) * 0.18)
    ax.set_ylim(max(0.0, metric_min - margin), min(1.0, metric_max + margin))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    chapter4_row = load_lgbm_row(CHAPTER4_CSV_PATH)
    chapter6_row = load_lgbm_row(CHAPTER6_CSV_PATH)
    metrics_df = build_metric_frame(chapter4_row, chapter6_row)
    plot_metric_comparison(metrics_df, chapter4_row, chapter6_row, OUTPUT_PATH)
    print(f"Saved plot to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
