#!/usr/bin/env python3
import json
import re
from pathlib import Path
import os

import pandas as pd


def parse_run_id_from_name(name: str):
    """
    Parse identifiers from filenames like:
    NN_ALL_balanced_focal_20260117_101500_seed42_minrecall0.85_val_roc_0.85_seed42.png
    NN_ALL_balanced_focal_20260117_101500_seed42_minrecall0.85_val_threshold_sweep_all.csv
    """
    pattern = re.compile(
        r"NN_(?P<feature>[^_]+)_(?P<mode>.+?)_(?P<ts>\d{8}_\d{6})_seed(?P<seed>\d+)_minrecall(?P<minrecall>[0-9.]+)"
    )
    match = pattern.search(name)
    if not match:
        return {}
    data = match.groupdict()
    return {
        "feature": data.get("feature"),
        "mode": data.get("mode"),
        "timestamp": data.get("ts"),
        "seed": int(data.get("seed")),
        "min_recall": float(data.get("minrecall")),
    }


def parse_seed_minrecall_from_artifacts(artifacts: dict):
    for key in ("val_threshold_sweep_all_csv", "val_threshold_sweep_constrained_csv", "val_curves"):
        val = artifacts.get(key)
        if isinstance(val, dict):
            # val_curves has roc/pr paths
            for path in val.values():
                if isinstance(path, str):
                    parsed = parse_run_id_from_name(path)
                    if parsed:
                        return parsed
        elif isinstance(val, str):
            parsed = parse_run_id_from_name(val)
            if parsed:
                return parsed
    return {}


def build_file_index(paths):
    index = {}
    for path in paths:
        parsed = parse_run_id_from_name(path.name)
        if not parsed:
            continue
        key = (
            parsed.get("mode"),
            parsed.get("min_recall"),
            parsed.get("seed"),
            parsed.get("timestamp"),
        )
        index.setdefault(key, []).append(path)
    return index


def main():
    base_dir = "../../output/ml_results"

    metrics_dir = Path(os.path.join(base_dir, "saved_ml_performance_metrics/phase2_nn_seed42_stageA"))
    tables_dir =Path(os.path.join(base_dir, "tables/phase2_nn_seed42_stageA"))
    plots_dir =Path(os.path.join(base_dir, "plots/phase2_nn_seed42_stageA"))

    metrics_files = sorted(metrics_dir.glob("*.json"))
    sweep_files = sorted(tables_dir.glob("*threshold_sweep*.csv"))
    plot_files = sorted(list(plots_dir.glob("*roc*.png")) + list(plots_dir.glob("*pr*.png")))

    sweep_index = build_file_index(sweep_files)
    plot_index = build_file_index(plot_files)

    rows = []
    for mf in metrics_files:
        with mf.open("r", encoding="utf-8") as f:
            data = json.load(f)

        model_name = data.get("model_name")
        timestamp = data.get("timestamp")
        hyper = data.get("hyperparameters") or {}
        metrics = (data.get("metrics") or {}).get("test", {})
        artifacts = data.get("artifacts") or {}

        nn_mode = hyper.get("nn_imbalance_mode")
        min_recall = hyper.get("min_recall_constraint")
        decision_threshold = hyper.get("decision_threshold")
        val_spec = hyper.get("val_specificity_at_threshold")
        val_recall = hyper.get("val_recall_at_threshold")

        parsed_artifacts = parse_seed_minrecall_from_artifacts(artifacts) if artifacts else {}
        seed = parsed_artifacts.get("seed")
        if min_recall is None:
            min_recall = parsed_artifacts.get("min_recall")

        # Fallback: parse from model_name if needed
        if nn_mode is None and isinstance(model_name, str):
            parts = model_name.split("_")
            if len(parts) >= 3:
                nn_mode = "_".join(parts[2:]).lower()

        # File paths from artifacts (preferred)
        val_curves = artifacts.get("val_curves") or {}
        test_curves = artifacts.get("test_curves") or {}
        val_roc = val_curves.get("roc_path") if isinstance(val_curves, dict) else None
        val_pr = val_curves.get("pr_path") if isinstance(val_curves, dict) else None
        test_roc = test_curves.get("roc_path") if isinstance(test_curves, dict) else None
        test_pr = test_curves.get("pr_path") if isinstance(test_curves, dict) else None
        sweep_all = artifacts.get("val_threshold_sweep_all_csv")
        sweep_con = artifacts.get("val_threshold_sweep_constrained_csv")

        # If artifacts missing, try to find by key
        key = (nn_mode, min_recall, seed, timestamp)
        if not val_roc or not val_pr or not test_roc or not test_pr:
            matches = plot_index.get(key, [])
            for path in matches:
                lname = path.name.lower()
                if "val" in lname and "roc" in lname and not val_roc:
                    val_roc = str(path)
                if "val" in lname and "pr" in lname and not val_pr:
                    val_pr = str(path)
                if "test" in lname and "roc" in lname and not test_roc:
                    test_roc = str(path)
                if "test" in lname and "pr" in lname and not test_pr:
                    test_pr = str(path)

        if not sweep_all or not sweep_con:
            matches = sweep_index.get(key, [])
            for path in matches:
                lname = path.name.lower()
                if "sweep_all" in lname and not sweep_all:
                    sweep_all = str(path)
                if "sweep_constrained" in lname and not sweep_con:
                    sweep_con = str(path)

        rows.append({
            "seed": seed,
            "min_recall": min_recall,
            "nn_imbalance_mode": nn_mode,
            "model_name": model_name,
            "timestamp": timestamp,
            "training_time_seconds": data.get("training_time_seconds"),
            "accuracy": metrics.get("accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "specificity": metrics.get("specificity"),
            "f1": metrics.get("f1_score"),
            "roc_auc": metrics.get("roc_auc"),
            "pr_auc": metrics.get("pr_auc"),
            "balanced_accuracy": metrics.get("balanced_accuracy"),
            "decision_threshold": decision_threshold,
            "val_specificity_at_threshold": val_spec,
            "val_recall_at_threshold": val_recall,
            "val_roc_plot": val_roc,
            "val_pr_plot": val_pr,
            "test_roc_plot": test_roc,
            "test_pr_plot": test_pr,
            "val_threshold_sweep_all_csv": sweep_all,
            "val_threshold_sweep_constrained_csv": sweep_con,
            "metrics_json": str(mf),
        })

    df = pd.DataFrame(rows)
    out_path = os.path.join(base_dir, "nn_stageA_inventory.csv")
    df.to_csv(out_path, index=False)

    if not df.empty:
        summary = df.groupby(["nn_imbalance_mode", "min_recall"]).size().reset_index(name="count")
        print(summary.to_string(index=False))
    else:
        print("No metrics files found.")


if __name__ == "__main__":
    main()
