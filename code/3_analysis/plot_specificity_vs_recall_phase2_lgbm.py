import pandas as pd
import matplotlib.pyplot as plt

# Paths
all_csv = "../../output/ml_results/tables/LGBM_ALL_all_20260115_222716_val_threshold_sweep_all_seed10.csv"
con_csv = "../../output/ml_results/tables/LGBM_ALL_all_20260115_222716_val_threshold_sweep_constrained_0.9_seed10.csv"

# Load data
df_all = pd.read_csv(all_csv)
df_con = pd.read_csv(con_csv)

# Optional: filter constrained explicitly if column exists
if "meets_min_recall" in df_con.columns:
    df_con = df_con[df_con["meets_min_recall"]]

# ---- Plot ----
plt.figure(figsize=(8, 6))

plt.plot(
    df_all["recall"],
    df_all["specificity"],
    label="All thresholds",
    alpha=0.6,
)

plt.plot(
    df_con["recall"],
    df_con["specificity"],
    label="Recall ≥ 0.9",
    linewidth=2.5,
)

plt.axvline(
    x=0.9,
    linestyle="--",
    color="gray",
    label="Min recall = 0.9"
)

plt.xlabel("Recall (Sensitivity)")
plt.ylabel("Specificity")
plt.title("LightGBM (All Features) – Specificity vs Recall (Validation)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
