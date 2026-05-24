from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MLP_SUMMARY_PATH = (
    PROJECT_ROOT
    / "output"
    / "ml_results"
    / "saved_ml_performance_metrics"
    / "phase2_nn_baseline_diff_seed"
    / "nn_baseline_threshold_final_summary_0.85.csv"
)
LGBM_SUMMARY_PATH = (
    PROJECT_ROOT
    / "output"
    / "ml_results"
    / "saved_ml_performance_metrics"
    / "phase2_threshold_tunning_diff_seed"
    / "lgbm_threshold_summary_by_constraint.csv"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "output"
    / "ml_results"
    / "saved_ml_performance_metrics"
    / "phase2_threshold_tunning_diff_seed"
    / "nn_vs_lgbm_metrics_comparison_min_recall_0.85.png"
)
TRAINING_TIME_CSV_PATH = (
    PROJECT_ROOT
    / "input"
    / "ml_results"
    / "saved_ml_performance_metrics"
    / "phase2_training_time_comparison_by_constraint.csv"
)

MIN_RECALL = 0.85
METRICS = [
    ("test_precision", "Precision"),
    ("test_recall", "Recall"),
    ("test_specificity", "Specificity"),
    ("test_f1_score", "F1 Score"),
    ("test_balanced_accuracy", "Balanced Accuracy"),
    ("test_roc_auc", "ROC AUC"),
    ("test_pr_auc", "PR AUC"),
]


def load_single_row(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"No rows found in {csv_path}")
    return df.iloc[0]


def load_lgbm_row(csv_path: Path, min_recall: float) -> pd.Series:
    df = pd.read_csv(csv_path)
    row = df.loc[df["min_recall"].round(4) == round(min_recall, 4)]
    if row.empty:
        raise ValueError(f"No LightGBM row found for min_recall={min_recall} in {csv_path}")
    return row.iloc[0]


def load_training_time_row(csv_path: Path, min_recall: float) -> pd.Series:
    df = pd.read_csv(csv_path)
    row = df.loc[df["min_recall"].round(4) == round(min_recall, 4)]
    if row.empty:
        raise ValueError(f"No training-time row found for min_recall={min_recall} in {csv_path}")
    return row.iloc[0]


def build_metric_frame(MLP_row: pd.Series, lgbm_row: pd.Series) -> pd.DataFrame:
    records = []
    for metric_key, metric_label in METRICS:
        MLP_mean = MLP_row[f"{metric_key}_mean"]
        lgbm_mean = lgbm_row[f"{metric_key}_mean"]
        records.append(
            {
                "metric": metric_label,
                "MLP_mean": MLP_mean,
                "MLP_std": MLP_row[f"{metric_key}_std"],
                "lgbm_mean": lgbm_mean,
                "lgbm_std": lgbm_row[f"{metric_key}_std"],
                "delta_mean": lgbm_mean - MLP_mean,
            }
        )
    return pd.DataFrame(records)


def plot_metric_comparison(
    metrics_df: pd.DataFrame,
    MLP_row: pd.Series,
    lgbm_row: pd.Series,
    training_time_row: pd.Series,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(13, 16),
        gridspec_kw={"height_ratios": [4.2, 2.2, 1.8, 1.8]},
        constrained_layout=True,
    )

    x = range(len(metrics_df))
    width = 0.36

    axes[0].bar(
        [i - width / 2 for i in x],
        metrics_df["MLP_mean"],
        width=width,
        yerr=metrics_df["MLP_std"],
        capsize=4,
        label="MLP",
        color="#1f77b4",
        alpha=0.9,
    )
    axes[0].bar(
        [i + width / 2 for i in x],
        metrics_df["lgbm_mean"],
        width=width,
        yerr=metrics_df["lgbm_std"],
        capsize=4,
        label="LightGBM",
        color="#ff7f0e",
        alpha=0.9,
    )
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(metrics_df["metric"], rotation=20, ha="right")
    axes[0].set_ylabel("Score")
    axes[0].set_title("MLP vs LightGBM Test Metrics at Minimum Recall Constraint 0.85")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend()

    metric_min = min(
        (metrics_df["MLP_mean"] - metrics_df["MLP_std"]).min(),
        (metrics_df["lgbm_mean"] - metrics_df["lgbm_std"]).min(),
    )
    metric_max = max(
        (metrics_df["MLP_mean"] + metrics_df["MLP_std"]).max(),
        (metrics_df["lgbm_mean"] + metrics_df["lgbm_std"]).max(),
    )
    margin = max(0.01, (metric_max - metric_min) * 0.2)
    axes[0].set_ylim(max(0.0, metric_min - margin), min(1.0, metric_max + margin))

    delta_colors = ["#2ca02c" if value >= 0 else "#d62728" for value in metrics_df["delta_mean"]]
    axes[1].bar(metrics_df["metric"], metrics_df["delta_mean"], color=delta_colors, alpha=0.9)
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_ylabel("LightGBM - MLP")
    axes[1].set_title("Metric Difference")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].tick_params(axis="x", rotation=20)

    delta_abs_max = max(abs(metrics_df["delta_mean"]).max(), 0.002)
    axes[1].set_ylim(-delta_abs_max * 1.35, delta_abs_max * 1.35)
    for idx, delta in enumerate(metrics_df["delta_mean"]):
        va = "bottom" if delta >= 0 else "top"
        offset = delta_abs_max * 0.06 if delta >= 0 else -delta_abs_max * 0.06
        axes[1].text(idx, delta + offset, f"{delta:+.4f}", ha="center", va=va, fontsize=9)

    threshold_labels = ["MLP Threshold", "LightGBM Threshold"]
    threshold_means = [MLP_row["decision_threshold_mean"], lgbm_row["decision_threshold_mean"]]
    threshold_stds = [MLP_row["decision_threshold_std"], lgbm_row["decision_threshold_std"]]
    threshold_colors = ["#1f77b4", "#ff7f0e"]

    axes[2].bar(
        threshold_labels,
        threshold_means,
        yerr=threshold_stds,
        capsize=4,
        color=threshold_colors,
        alpha=0.9,
    )
    threshold_min = min(mean - std for mean, std in zip(threshold_means, threshold_stds))
    threshold_max = max(mean + std for mean, std in zip(threshold_means, threshold_stds))
    threshold_margin = max(0.01, (threshold_max - threshold_min) * 0.35)
    axes[2].set_ylim(max(0.0, threshold_min - threshold_margin), min(1.0, threshold_max + threshold_margin))
    axes[2].set_ylabel("Threshold")
    axes[2].set_title("Tuned Decision Threshold")
    axes[2].grid(axis="y", alpha=0.3)
    for idx, value in enumerate(threshold_means):
        axes[2].text(idx, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    training_labels = ["MLP Training Time", "LightGBM Training Time"]
    training_means = [
        training_time_row["mlp_training_time_mean"],
        training_time_row["lgbm_training_time_mean"],
    ]
    training_stds = [
        training_time_row["mlp_training_time_std"],
        training_time_row["lgbm_training_time_std"],
    ]
    axes[3].bar(
        training_labels,
        training_means,
        yerr=training_stds,
        capsize=4,
        color=threshold_colors,
        alpha=0.9,
    )
    training_min = min(mean - std for mean, std in zip(training_means, training_stds))
    training_max = max(mean + std for mean, std in zip(training_means, training_stds))
    training_margin = max(5.0, (training_max - training_min) * 0.2)
    axes[3].set_ylim(max(0.0, training_min - training_margin), training_max + training_margin)
    axes[3].set_ylabel("Seconds")
    axes[3].set_title("Training Time Comparison")
    axes[3].grid(axis="y", alpha=0.3)
    for idx, value in enumerate(training_means):
        axes[3].text(idx, value, f"{value:.1f}s", ha="center", va="bottom", fontsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    MLP_row = load_single_row(MLP_SUMMARY_PATH)
    lgbm_row = load_lgbm_row(LGBM_SUMMARY_PATH, MIN_RECALL)
    training_time_row = load_training_time_row(TRAINING_TIME_CSV_PATH, MIN_RECALL)
    metrics_df = build_metric_frame(MLP_row, lgbm_row)
    plot_metric_comparison(metrics_df, MLP_row, lgbm_row, training_time_row, OUTPUT_PATH)
    print(f"Saved plot to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
