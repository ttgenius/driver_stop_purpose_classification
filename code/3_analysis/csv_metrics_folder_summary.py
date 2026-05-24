from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


METRIC_COLUMNS = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
    "pr_auc",
    "specificity",
    "sensitivity",
    "balanced_accuracy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read all metrics JSON files in a folder, compute mean/std for training time "
            "and test metrics, and save the summary CSV in the same folder."
        )
    )
    parser.add_argument(
        "input_folder",
        type=Path,
        help="Folder containing metrics JSON files.",
    )
    return parser.parse_args()


def safe_nested_get(payload: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def model_name_from_filename(filepath: Path) -> str:
    return filepath.stem.split("_")[0]


def build_rows(input_folder: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for filepath in sorted(input_folder.glob("*.json")):
        with filepath.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        test_metrics = safe_nested_get(payload, ["metrics", "test"], default={})
        if not isinstance(test_metrics, dict):
            test_metrics = {}

        row: dict[str, Any] = {
            "model_name": model_name_from_filename(filepath),
            "file_name": filepath.name,
            "training_time_seconds": payload.get("training_time_seconds"),
        }

        for metric_name in METRIC_COLUMNS:
            row[metric_name] = test_metrics.get(metric_name)

        rows.append(row)

    return rows


def summarize_metrics(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No JSON files were found, or no rows could be created.")

    numeric_columns = ["training_time_seconds", *METRIC_COLUMNS]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    agg_spec: dict[str, list[str]] = {
        "training_time_seconds": ["mean", "std"],
    }
    for metric_name in METRIC_COLUMNS:
        agg_spec[metric_name] = ["mean", "std"]

    summary = (
        df.groupby("model_name", dropna=False)
        .agg(agg_spec)
        .reset_index()
    )

    renamed_columns = ["model_name"]
    for column_name, stat_name in summary.columns.tolist()[1:]:
        if column_name == "training_time_seconds":
            base_name = "training_time"
        else:
            base_name = column_name
        suffix = "std" if stat_name == "std" else stat_name
        renamed_columns.append(f"{base_name}_{suffix}")

    summary.columns = renamed_columns
    return summary.sort_values("model_name").reset_index(drop=True)


def output_csv_path(input_folder: Path) -> Path:
    return input_folder / f"{input_folder.name}_metrics_summary_with_std.csv"


def main() -> None:
    args = parse_args()
    input_folder = args.input_folder.resolve()

    if not input_folder.exists() or not input_folder.is_dir():
        raise FileNotFoundError(f"Input folder does not exist or is not a directory: {input_folder}")

    rows = build_rows(input_folder)
    summary = summarize_metrics(rows)

    output_path = output_csv_path(input_folder)
    summary.to_csv(output_path, index=False)

    print(f"Saved summary CSV to: {output_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
