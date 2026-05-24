import pandas as pd
import matplotlib.pyplot as plt


# Specificity vs Recall (mean ± std across seeds)
#summary = pd.read_csv("../../output/ml_results/saved_ml_performance_metrics/phase2_threshold_tunning_diff_seed/lgbm_threshold_summary_by_constraint.csv")
summary = pd.read_csv("../../output/ml_results/saved_ml_performance_metrics/phase2_nn_baseline_diff_seed/nn_baseline_threshold_summary_by_constraint.csv")

plt.figure(figsize=(8, 6))

plt.errorbar(
    summary["min_recall"],
    summary["test_specificity_mean"],
    yerr=summary["test_specificity_std"],
    fmt="o-",
    capsize=4,
    label="Specificity (mean ± std)"
)

plt.errorbar(
    summary["min_recall"],
    summary["test_recall_mean"],
    yerr=summary["test_recall_std"],
    fmt="s--",
    capsize=4,
    label="Recall (mean ± std)"
)

plt.xlabel("Minimum Recall Constraint")
plt.ylabel("Metric Value")
plt.title("Recall–Specificity Trade-off (Threshold Tuning, LightGBM)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("../../output/ml_results/saved_ml_performance_metrics/phase2_nn_baseline_diff_seed/nn_baseline_threshold_summary_by_constraint_specificity_vs_recall_plot.png")
plt.show()

#Decision threshold stability
plt.figure(figsize=(8, 5))

plt.errorbar(
    summary["min_recall"],
    summary["decision_threshold_mean"],
    yerr=summary["decision_threshold_std"],
    fmt="o-",
    capsize=4
)

plt.xlabel("Minimum Recall Constraint")
plt.ylabel("Decision Threshold")
plt.title("Stability of Tuned Decision Threshold Across Seeds")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("../../output/ml_results/saved_ml_performance_metrics/phase2_nn_baseline_diff_seed/nn_baseline_threshold_summary_by_constraint_decision_threshold_stability_plot.png")

plt.show()

# Model quality invariance roc auc & pr auc
plt.figure(figsize=(8, 5))

plt.errorbar(
    summary["min_recall"],
    summary["test_roc_auc_mean"],
    yerr=summary["test_roc_auc_std"],
    fmt="o-",
    capsize=4,
    label="ROC AUC"
)

plt.errorbar(
    summary["min_recall"],
    summary["test_pr_auc_mean"],
    yerr=summary["test_pr_auc_std"],
    fmt="s--",
    capsize=4,
    label="PR AUC"
)

plt.xlabel("Minimum Recall Constraint")
plt.ylabel("AUC")
plt.title("Model Discriminative Performance vs Recall Constraint")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("../../output/ml_results/saved_ml_performance_metrics/phase2_nn_baseline_diff_seed/nn_baseline_threshold_summary_by_constraint_model_quality_invariance_roc_auc_pr_auc_plot.png")

plt.show()

