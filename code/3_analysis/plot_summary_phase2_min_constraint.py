import pandas as pd
import matplotlib.pyplot as plt

# ---- Paths: replace with your actual files ----
# all_09 = ../../output/ml_results/tables/LGBM_ALL_all_20260111_145152_val_threshold_sweep_all_without_scale_pos_weight.csv"
# con_09 = ../../output/ml_results/tables/LGBM_ALL_all_20260111_145152_val_threshold_sweep_constrained_0.9_without_scale_pos_weight.csv"  # recall>=0.9
#
# all_085 = ../../output/ml_results/tables/LGBM_ALL_all_20260111_175258_val_threshold_sweep_all.csv"
# con_085 = ../../output/ml_results/tables/LGBM_ALL_all_20260111_175258_val_threshold_sweep_constrained_0.85.csv"
#
# all_08 = ../../output/ml_results/tables/LGBM_ALL_all_20260111_182429_val_threshold_sweep_all.csv"
# con_08 = ../../output/ml_results/tables/LGBM_ALL_all_20260111_182429_val_threshold_sweep_constrained_0.8.csv"
#
# all_095 = ../../output/ml_results/tables/LGBM_ALL_all_20260111_203113_val_threshold_sweep_all.csv"
# con_095 = ../../output/ml_results/tables/LGBM_ALL_all_20260111_203113_val_threshold_sweep_constrained_0.95.csv"

all_09 = "../../output/ml_results/tables/LGBM_ALL_all_20260116_222753_val_threshold_sweep_all.csv"
con_09 = "../../output/ml_results/tables/LGBM_ALL_all_20260116_222753_val_threshold_sweep_constrained_0.9_seed25.csv"  # recall>=0.9

all_085 = "../../output/ml_results/tables/LGBM_ALL_all_20260116_222416_val_threshold_sweep_all.csv"
con_085 = "../../output/ml_results/tables/LGBM_ALL_all_20260116_222416_val_threshold_sweep_constrained_0.85_seed25.csv"

all_08 = "../../output/ml_results/tables/LGBM_ALL_all_20260116_222042_val_threshold_sweep_all.csv"
con_08 = "../../output/ml_results/tables/LGBM_ALL_all_20260116_222042_val_threshold_sweep_constrained_0.8_seed25.csv"

all_095 = "../../output/ml_results/tables/LGBM_ALL_all_20260116_223138_val_threshold_sweep_all.csv"
con_095 = "../../output/ml_results/tables/LGBM_ALL_all_20260116_223138_val_threshold_sweep_constrained_0.95_seed25.csv"

# ---- Load ----
df_all = pd.read_csv(all_09)  # full sweep is the same model; any "all" file is fine if same run
df_09 = pd.read_csv(con_09)
df_085 = pd.read_csv(con_085)
df_08 = pd.read_csv(con_08)
df_095 = pd.read_csv(con_095)

# If your constrained CSVs include meets_min_recall, filter to be safe
for name, df in [("0.95", df_095), ("0.9", df_09), ("0.85", df_085), ("0.8", df_08)]:
    if "meets_min_recall" in df.columns:
        df = df[df["meets_min_recall"]].copy()
    # reassign
    if name == "0.95":
        df_095 = df
    if name == "0.9":
        df_09 = df
    elif name == "0.85":
        df_085 = df
    else:
        df_08 = df

# ---- Helper: select the "tuned" operating point
# Your tuner is "max specificity subject to recall>=min_recall", so replicate that:
def pick_op_point(df_constrained):
    # (assumes df already satisfies min_recall if filtered)
    idx = df_constrained["specificity"].idxmax()
    return df_constrained.loc[idx]

op_095 = pick_op_point(df_095)
op_09 = pick_op_point(df_09)
op_085 = pick_op_point(df_085)
op_08 = pick_op_point(df_08)

# ---- Plot ----
plt.figure(figsize=(9, 6))

# Full frontier (all thresholds)
plt.plot(df_all["recall"], df_all["specificity"], alpha=0.25, linewidth=2, label="All thresholds (frontier)")

# Feasible regions
plt.plot(df_095["recall"], df_095["specificity"], linewidth=2.5, label="Feasible: recall ≥ 0.95")
plt.plot(df_09["recall"], df_09["specificity"], linewidth=2.5, label="Feasible: recall ≥ 0.90")
plt.plot(df_085["recall"], df_085["specificity"], linewidth=2.5, label="Feasible: recall ≥ 0.85")
plt.plot(df_08["recall"], df_08["specificity"], linewidth=2.5, label="Feasible: recall ≥ 0.80")

# Operating points (tuned thresholds)
plt.scatter(op_095["recall"], op_095["specificity"], s=90, marker="o", label=f"Chosen @0.95 (thr={op_095['threshold']:.3f})")
plt.scatter(op_09["recall"], op_09["specificity"], s=90, marker="o", label=f"Chosen @0.90 (thr={op_09['threshold']:.3f})")
plt.scatter(op_085["recall"], op_085["specificity"], s=90, marker="o", label=f"Chosen @0.85 (thr={op_085['threshold']:.3f})")
plt.scatter(op_08["recall"], op_08["specificity"], s=90, marker="o", label=f"Chosen @0.80 (thr={op_08['threshold']:.3f})")

# Reference lines
for r in [0.8, 0.85, 0.9, 0.95]:
    plt.axvline(r, linestyle="--", alpha=0.35)

plt.xlabel("Recall (Sensitivity)")
plt.ylabel("Specificity")
plt.title("LightGBM (All Features) — Operating Points under Recall Constraints (Validation)")
plt.grid(True, alpha=0.25)
plt.legend(loc="lower left", fontsize=9)
# plt.annotate(
#     f"R={op_09['recall']:.2f}, S={op_09['specificity']:.2f}",
#     (op_09["recall"], op_09["specificity"]),
#     textcoords="offset points", xytext=(8, 8)
# )
plt.tight_layout()

# Save if you want
plt.savefig("../../output/ml_results/plots/lgbm_operating_points_summary.png", dpi=250)

plt.show()
