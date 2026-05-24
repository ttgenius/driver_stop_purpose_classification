"""
Aggregate LightGBM threshold-tuning results across random seeds + recall constraints.

Assumptions:
- You have multiple JSON metric files with the SAME structure as the example.
- We aggregate ONLY the metrics in metrics["test"].
- We also record the tuned decision_threshold + val operating point from hyperparameters.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# --------- helpers ---------

def parse_filename_tags(fp: Path) -> Tuple[Optional[int], Optional[float]]:
    """
    Parse filenames like:
      LGBM_ALL_all_20260111_175258_metrics_seed42_minrecall0.85.json
      LGBM_ALL_all_20260116_222753_metrics_seed25_minrecall0.9.json
      LGBM_ALL_all_20260116_214355_metrics_seed10_minrecall0.95.json

    Returns: (seed, min_recall)
    """
    name = fp.name

    seed = None
    mr = None

    m_seed = re.search(r"_seed(\d+)(?:_|\.|$)", name, flags=re.IGNORECASE)
    if m_seed:
        seed = int(m_seed.group(1))

    m_mr = re.search(r"_minrecall([0-9]*\.?[0-9]+)(?:_|\.|$)", name, flags=re.IGNORECASE)
    if m_mr:
        mr = float(m_mr.group(1))

    return seed, mr


def _infer_seed(obj: Dict[str, Any], filepath: Path) -> Optional[int]:
    # Prefer JSON if present
    rs = obj.get("hyperparameters", {}).get("random_state")
    if isinstance(rs, int):
        return rs

    # Fallback: filename (deterministic)
    seed, _ = parse_filename_tags(filepath)
    return seed


def _infer_min_recall(obj: Dict[str, Any], filepath: Path) -> Optional[float]:
    # Prefer JSON if present
    mr = obj.get("hyperparameters", {}).get("min_recall_constraint")
    if isinstance(mr, (int, float)):
        return float(mr)

    # Fallback: filename (deterministic)
    _, min_recall = parse_filename_tags(filepath)
    return min_recall


def _safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_results_from_json_dir(
    json_dir,
    glob_pattern: str = "*.json",
) -> pd.DataFrame:
    """
    Loads all JSON files and returns a flat dataframe with test metrics + key hyperparams.
    """
    json_dir = Path(json_dir)
    rows: List[Dict[str, Any]] = []

    for fp in sorted(json_dir.glob(glob_pattern)):
        with fp.open("r", encoding="utf-8") as f:
            obj = json.load(f)

        test = _safe_get(obj, ["metrics", "test"], default={})
        if not isinstance(test, dict) or len(test) == 0:
            continue

        row: Dict[str, Any] = {}
        row["file"] = str(fp)
        row["timestamp"] = obj.get("timestamp")
        row["model_name"] = obj.get("model_name")
        row["feature_mode"] = obj.get("feature_mode")

        row["seed"] = _infer_seed(obj, fp)
        row["min_recall"] = _infer_min_recall(obj, fp)

        # tuned threshold + val op point (useful to report stability)
        row["decision_threshold"] = _safe_get(obj, ["hyperparameters", "decision_threshold"])
        row["val_best_threshold"] = _safe_get(obj, ["hyperparameters", "val_best_threshold"])
        row["val_specificity_at_threshold"] = _safe_get(obj, ["hyperparameters", "val_specificity_at_threshold"])
        row["val_recall_at_threshold"] = _safe_get(obj, ["hyperparameters", "val_recall_at_threshold"])

        # copy all test metrics
        for k, v in test.items():
            if k == "confusion_matrix":
                continue
            row[f"test_{k}"] = v

        # confusion matrix parts (if present)
        row["test_tp"] = test.get("true_positives")
        row["test_tn"] = test.get("true_negatives")
        row["test_fp"] = test.get("false_positives")
        row["test_fn"] = test.get("false_negatives")

        rows.append(row)

    df = pd.DataFrame(rows)

    # ensure numeric types for aggregation columns
    for c in df.columns:
        if c.startswith("test_") or c in ("seed", "min_recall", "decision_threshold", "val_best_threshold",
                                          "val_specificity_at_threshold", "val_recall_at_threshold"):
            df[c] = pd.to_numeric(df[c], errors="ignore")

    return df


def summarize_by_constraint(
    df: pd.DataFrame,
    metric_cols: Optional[List[str]] = None,
    group_cols: Tuple[str, ...] = ("min_recall",),
) -> pd.DataFrame:
    """
    Returns mean/std across seeds for each recall constraint.
    """
    if metric_cols is None:
        # default: key thesis metrics + threshold stability
        metric_cols = [
            "test_precision",
            "test_recall",
            "test_specificity",
            "test_f1_score",
            "test_balanced_accuracy",
            "test_roc_auc",
            "test_pr_auc",
            "decision_threshold",
        ]

    # keep only existing columns
    metric_cols = [c for c in metric_cols if c in df.columns]

    agg = {}
    for c in metric_cols:
        agg[c] = ["mean", "std", "min", "max"]

    out = (
        df.dropna(subset=list(group_cols))
          .groupby(list(group_cols), dropna=False)
          .agg(agg)
    )

    # flatten multiindex columns
    out.columns = [f"{c}_{stat}" for c, stat in out.columns]
    out = out.reset_index()

    return out


def summarize_single_choice(
    df: pd.DataFrame,
    min_recall: float,
    metric_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Convenience: summary table for your FINAL chosen constraint (e.g., 0.85).
    """
    sub = df[df["min_recall"].round(2) == round(min_recall, 2)].copy()
    if sub.empty:
        raise ValueError(f"No rows found for min_recall={min_recall}. Check parsing or filenames.")
    return summarize_by_constraint(sub, metric_cols=metric_cols, group_cols=("min_recall",))


# --------- example usage ---------
if __name__ == "__main__":
    # 1) Point this to your folder containing the metrics JSONs
    #JSON_DIR = "../../output/ml_results/saved_ml_performance_metrics/phase2_threshold_tunning_diff_seed"
    JSON_DIR = "../../output/ml_results/saved_ml_performance_metrics/phase2_nn_baseline_diff_seed"
    df = load_results_from_json_dir(JSON_DIR, glob_pattern="*.json")

    #Optional sanity check:
    print(df[["seed", "min_recall", "test_recall", "test_specificity", "decision_threshold", "file"]]
          .sort_values(["min_recall", "seed"]))

    # 2) Summary across all constraints (mean/std across seeds)
    summary_all = summarize_by_constraint(df)
    print("\n=== Summary across recall constraints (test metrics, mean/std/min/max across seeds) ===")
    print(summary_all.to_string(index=False))

    # 3) Final report table for your chosen constraint (recommend: 0.85)
    final_summary = summarize_single_choice(df, min_recall=0.85)
    print("\n=== FINAL report (min_recall=0.85) ===")
    print(final_summary.to_string(index=False))

    # 4) Save outputs
    out_dir = Path(JSON_DIR)
    # df.to_csv(out_dir / "lgbm_threshold_runs_flat.csv", index=False)
    # summary_all.to_csv(out_dir / "lgbm_threshold_summary_by_constraint.csv", index=False)
    # final_summary.to_csv(out_dir / "lgbm_threshold_final_summary_0.85.csv", index=False)

    # print("\nSaved:")
    # print(out_dir / "lgbm_threshold_runs_flat.csv")
    # print(out_dir / "lgbm_threshold_summary_by_constraint.csv")
    # print(out_dir / "lgbm_threshold_final_summary_0.85.csv")

    df.to_csv(out_dir / "nn_baseline_threshold_runs_flat.csv", index=False)
    summary_all.to_csv(out_dir / "nn_baseline_threshold_summary_by_constraint.csv", index=False)
    final_summary.to_csv(out_dir / "nn_baseline_threshold_final_summary_0.85.csv", index=False)

    print("\nSaved:")
    print(out_dir / "nn_baseline_threshold_runs_flat.csv")
    print(out_dir / "nn_baseline_threshold_summary_by_constraint.csv")
    print(out_dir / "nn_baseline_threshold_final_summary_0.85.csv")
