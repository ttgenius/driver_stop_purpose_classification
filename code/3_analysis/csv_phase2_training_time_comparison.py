from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LGBM_JSON_DIR = (
    PROJECT_ROOT
    / "output"
    / "ml_results"
    / "saved_ml_performance_metrics"
    / "phase2_threshold_tunning_diff_seed"
)
MLP_JSON_DIR = (
    PROJECT_ROOT
    / "output"
    / "ml_results"
    / "saved_ml_performance_metrics"
    / "phase2_nn_baseline_diff_seed"
)
OUTPUT_CSV_PATH = (
    PROJECT_ROOT
    / "output"
    / "ml_results"
    / "saved_ml_performance_metrics"
    / "phase2_training_time_comparison_by_constraint.csv"
)


def parse_filename_tags(filepath: Path) -> tuple[int | None, float | None]:
    seed_match = re.search(r"_seed(\d+)(?:_|\.|$)", filepath.name, flags=re.IGNORECASE)
    min_recall_match = re.search(
        r"_minrecall([0-9]*\.?[0-9]+)(?:_|\.|$)",
        filepath.name,
        flags=re.IGNORECASE,
    )

    seed = int(seed_match.group(1)) if seed_match else None
    min_recall = float(min_recall_match.group(1)) if min_recall_match else None
    return seed, min_recall


def load_training_times(json_dir: Path, model_label: str) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    for filepath in sorted(json_dir.glob("*.json")):
        with filepath.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        seed, min_recall = parse_filename_tags(filepath)
        training_time = payload.get("training_time_seconds")
        if min_recall is None or training_time is None:
            continue

        rows.append(
            {
                "model": model_label,
                "seed": seed,
                "min_recall": float(min_recall),
                "training_time_seconds": float(training_time),
                "file_name": filepath.name,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No training-time rows found in {json_dir}")
    return df.sort_values(["min_recall", "seed"]).reset_index(drop=True)


def summarize_training_times(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    summary = (
        df.groupby("min_recall", dropna=False)["training_time_seconds"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    return summary.rename(
        columns={
            "mean": f"{prefix}_training_time_mean",
            "std": f"{prefix}_training_time_std",
            "min": f"{prefix}_training_time_min",
            "max": f"{prefix}_training_time_max",
            "count": f"{prefix}_seed_count",
        }
    )


def main() -> None:
    lgbm_df = load_training_times(LGBM_JSON_DIR, model_label="LightGBM")
    mlp_df = load_training_times(MLP_JSON_DIR, model_label="MLP")

    lgbm_summary = summarize_training_times(lgbm_df, prefix="lgbm")
    mlp_summary = summarize_training_times(mlp_df, prefix="mlp")

    comparison = lgbm_summary.merge(mlp_summary, on="min_recall", how="outer").sort_values("min_recall")

    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUTPUT_CSV_PATH, index=False)

    print(f"Saved comparison CSV to: {OUTPUT_CSV_PATH}")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
