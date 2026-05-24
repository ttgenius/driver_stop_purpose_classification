import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
CSV_PATH = "phase1_all_models_performance_log.csv"
OUT_DIR = "phase1_plots"
os.makedirs(OUT_DIR, exist_ok=True)

# Publication-ish defaults (no seaborn)
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
})

FEATURE_ORDER = ["TIME_ONLY", "POI_TIME", "ROAD_TIME", "INTERSECTION_TIME", "ALL"]
MODEL_ORDER   = ["LR", "RF", "LGBM", "NN", "CNN", "TabTransformer"]

def parse_name(name: str):
    if name.startswith("TabTransformer"):
        model = "TabTransformer"
        feature = name[len("TabTransformer_"):]
    else:
        parts = name.split("_")
        model = parts[0]
        feature = "_".join(parts[1:])
    return model, feature

def savefig(filename: str):
    path = os.path.join(OUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print("Saved:", path)

# ---------- LOAD ----------
df = pd.read_csv(CSV_PATH)
df[["model", "feature_set"]] = df["model_name"].apply(lambda s: pd.Series(parse_name(s)))

df["feature_set"] = pd.Categorical(df["feature_set"], categories=FEATURE_ORDER, ordered=True)
df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)

# ---------- 1) PR-AUC HEATMAP ----------
pivot_pr = df.pivot(index="model", columns="feature_set", values="test_pr_auc").loc[MODEL_ORDER, FEATURE_ORDER]

plt.figure(figsize=(10, 4.5))
im = plt.imshow(pivot_pr.values, aspect="auto")
plt.colorbar(im, label="Test PR-AUC")
plt.xticks(range(len(FEATURE_ORDER)), FEATURE_ORDER, rotation=25, ha="right")
plt.yticks(range(len(MODEL_ORDER)), MODEL_ORDER)
plt.title("Phase 1: Test PR-AUC by Model and Feature Set")

for i in range(pivot_pr.shape[0]):
    for j in range(pivot_pr.shape[1]):
        v = pivot_pr.values[i, j]
        if pd.notna(v):
            plt.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8)

savefig("01_pr_auc_heatmap.png")

# ---------- 2) PR-AUC GROUPED BARS ----------
plt.figure(figsize=(11, 4.5))
x = np.arange(len(MODEL_ORDER))
width = 0.15

for k, fs in enumerate(FEATURE_ORDER):
    vals = (
        df[df["feature_set"] == fs]
        .set_index("model")
        .loc[MODEL_ORDER, "test_pr_auc"]
        .values
    )
    plt.bar(x + (k - (len(FEATURE_ORDER)-1)/2)*width, vals, width, label=fs)

plt.xticks(x, MODEL_ORDER)
plt.ylabel("Test PR-AUC")
plt.title("Test PR-AUC across Models (grouped by Feature Set)")
plt.legend(ncol=3)
savefig("02_pr_auc_grouped_bars.png")

# ---------- 3) DELTA PR-AUC VS BASELINE ----------
baseline = df[df["feature_set"] == "TIME_ONLY"].set_index("model")["test_pr_auc"]

deltas = []
for fs in FEATURE_ORDER:
    cur = df[df["feature_set"] == fs].set_index("model")["test_pr_auc"]
    deltas.append((cur - baseline).loc[MODEL_ORDER].values)

deltas = np.vstack(deltas)

plt.figure(figsize=(10, 4.5))
im = plt.imshow(deltas, aspect="auto")
plt.colorbar(im, label="Δ PR-AUC vs TIME_ONLY")
plt.xticks(range(len(MODEL_ORDER)), MODEL_ORDER)
plt.yticks(range(len(FEATURE_ORDER)), FEATURE_ORDER)
plt.title("PR-AUC Improvement over Baseline (TIME_ONLY)")

for i in range(deltas.shape[0]):
    for j in range(deltas.shape[1]):
        v = deltas[i, j]
        if np.isfinite(v):
            plt.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=8)

savefig("03_pr_auc_delta_vs_time_only.png")

# ---------- 4) SPECIFICITY VS RECALL ----------
plt.figure(figsize=(8, 5))
markers = {"LR":"o","RF":"s","LGBM":"^","NN":"D","CNN":"v","TabTransformer":"P"}

for fs in FEATURE_ORDER:
    sub = df[df["feature_set"] == fs]
    for m in MODEL_ORDER:
        row = sub[sub["model"] == m]
        if len(row) == 1:
            plt.scatter(
                row["test_specificity"].values[0],
                row["test_recall"].values[0],
                marker=markers[m],
                label=fs if m == MODEL_ORDER[0] else None
            )

plt.xlabel("Test Specificity")
plt.ylabel("Test Recall (Sensitivity)")
plt.title("Specificity vs Recall (Phase 1)")
plt.legend(title="Feature set")
savefig("04_specificity_vs_recall.png")

# ---------- 5) TRAINING TIME HEATMAP (log10) ----------
pivot_time = df.pivot(index="model", columns="feature_set", values="training_time_seconds").loc[MODEL_ORDER, FEATURE_ORDER]

plt.figure(figsize=(10, 4.5))
im = plt.imshow(np.log10(pivot_time.values), aspect="auto")
plt.colorbar(im, label="log10(training_time_seconds)")
plt.xticks(range(len(FEATURE_ORDER)), FEATURE_ORDER, rotation=25, ha="right")
plt.yticks(range(len(MODEL_ORDER)), MODEL_ORDER)
plt.title("Training Time (log10 seconds) by Model and Feature Set")

for i in range(pivot_time.shape[0]):
    for j in range(pivot_time.shape[1]):
        v = pivot_time.values[i, j]
        if pd.notna(v):
            plt.text(j, i, f"{v/60:.1f}m", ha="center", va="center", fontsize=8)

savefig("05_training_time_heatmap.png"