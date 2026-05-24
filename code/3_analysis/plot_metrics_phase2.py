import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = "../../output/ml_results/"
CSV_PATH = os.path.join(RESULTS_DIR, "phase2_rf_lgbm_performance_log.csv")
OUT_DIR = os.path.join(RESULTS_DIR, "phase2_plots")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# Convenience columns
df["family"] = df["model_name"].str.extract(r"^(RF|LGBM)", expand=False)
df["feature_set"] = df["model_name"].str.replace(r"^(RF|LGBM)_", "", regex=True)

# Add balanced accuracy for analysis
df["test_balanced_accuracy"] = (df["test_recall"] + df["test_specificity"]) / 2.0

# -----------------------------
# 1) Specificity vs Recall
# -----------------------------
plt.figure(figsize=(7, 5))
for (fam, feat), g in df.groupby(["family", "feature_set"]):
    plt.scatter(g["test_recall"], g["test_specificity"], s=90, label=f"{fam}_{feat}")
    for _, r in g.iterrows():
        plt.text(
            r["test_recall"] + 0.001,
            r["test_specificity"] + 0.002,
            f"minR={r['min_recall_constraint']:.2f}",
            fontsize=8
        )

plt.xlabel("Test Recall (Sensitivity)")
plt.ylabel("Test Specificity")
plt.title("Phase 2: Specificity vs Recall (threshold tuned under recall constraint)")
plt.xlim(0.88, 0.94)
plt.ylim(0.15, 0.36)
plt.legend(fontsize=8, loc="lower left")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "01_specificity_vs_recall.png"), dpi=300)
plt.close()

# -----------------------------
# 2) Specificity bars by constraint
# -----------------------------
pivot = df.pivot_table(
    index=["family", "feature_set"],
    columns="min_recall_constraint",
    values="test_specificity",
    aggfunc="mean"
).sort_index()

plt.figure(figsize=(7, 4))
x = np.arange(len(pivot.index))
width = 0.35
cols = sorted(pivot.columns)

plt.bar(x - width/2, pivot[cols[0]].values, width, label=f"min_recall={cols[0]:.2f}")
plt.bar(x + width/2, pivot[cols[1]].values, width, label=f"min_recall={cols[1]:.2f}")

plt.xticks(x, [f"{a}_{b}" for a, b in pivot.index], rotation=30, ha="right")
plt.ylabel("Test Specificity")
plt.title("Specificity under different recall constraints")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "02_specificity_bars.png"), dpi=300)
plt.close()

# -----------------------------
# 3) PR-AUC (should be threshold-independent)
# -----------------------------
order = df.sort_values(["family", "feature_set", "min_recall_constraint"]).reset_index(drop=True)
labels = [f"{r.family}_{r.feature_set}\nminR={r.min_recall_constraint:.2f}" for r in order.itertuples()]

plt.figure(figsize=(7, 4))
plt.bar(np.arange(len(order)), order["test_pr_auc"].values)
plt.xticks(np.arange(len(order)), labels, rotation=30, ha="right")
plt.ylabel("Test PR-AUC")
plt.title("PR-AUC across runs (threshold-independent)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "03_pr_auc.png"), dpi=300)
plt.close()

# -----------------------------
# 4) Tuned threshold per run
# -----------------------------
plt.figure(figsize=(7, 4))
plt.plot(np.arange(len(order)), order["decision_threshold"].values, marker="o")
plt.xticks(np.arange(len(order)), labels, rotation=30, ha="right")
plt.ylabel("Decision threshold")
plt.title("Chosen threshold (maximize specificity subject to recall constraint)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "04_thresholds.png"), dpi=300)
plt.close()

# -----------------------------
# 5) FPR vs TPR (same as ROC operating points)
# -----------------------------
df["test_fpr"] = df["test_false_positives"] / (df["test_false_positives"] + df["test_true_negatives"])
df["test_tpr"] = df["test_true_positives"] / (df["test_true_positives"] + df["test_false_negatives"])

plt.figure(figsize=(6, 5))
for (fam, feat), g in df.groupby(["family", "feature_set"]):
    plt.scatter(g["test_fpr"], g["test_tpr"], s=90, label=f"{fam}_{feat}")
    for _, r in g.iterrows():
        plt.text(
            r["test_fpr"] + 0.002,
            r["test_tpr"] - 0.004,
            f"minR={r['min_recall_constraint']:.2f}",
            fontsize=8
        )

plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Recall)")
plt.title("Operating points (FPR, TPR) after threshold tuning")
plt.xlim(0.60, 0.85)
plt.ylim(0.88, 0.94)
plt.legend(fontsize=8, loc="lower left")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "05_fpr_tpr.png"), dpi=300)
plt.close()

print(f"Saved plots to: {OUT_DIR}")
