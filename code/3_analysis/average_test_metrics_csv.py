from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_INPUT_DIR = "../../output/ml_results/saved_ml_performance_metrics/phase3_nn_with_naics"

DEFAULT_OUTPUT_NAME = "average_test_metrics_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Average the numeric values under metrics.test across JSON metric files in a folder."
        )
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=DEFAULT_INPUT_DIR,
        help="Folder containing metrics JSON files.",
    )
    parser.add_argument(
        "--glob",
        default="*_metrics.json",
        help="Glob pattern used to discover JSON files inside the input folder.",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help="Output CSV filename written into the input folder.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def average_nested_values(values: list[Any]) -> Any:
    first_value = values[0]

    if is_number(first_value):
        return sum(float(value) for value in values) / len(values)

    if isinstance(first_value, list):
        expected_length = len(first_value)
        for value in values:
            if not isinstance(value, list) or len(value) != expected_length:
                raise ValueError("Encountered inconsistent list shapes while averaging test metrics.")
        return [
            average_nested_values([value[index] for value in values])
            for index in range(expected_length)
        ]

    raise TypeError(f"Unsupported test metric type for averaging: {type(first_value).__name__}")


def infer_seed(obj: dict[str, Any], path: Path) -> Any:
    hyperparameters = obj.get("hyperparameters", {})
    if isinstance(hyperparameters, dict) and "random_state" in hyperparameters:
        return hyperparameters["random_state"]
    return path.name


def flatten_metric_for_csv(metric_name: str, value: Any) -> dict[str, Any]:
    if is_number(value):
        return {metric_name: value}

    if isinstance(value, list):
        flattened: dict[str, Any] = {}
        for index, nested_value in enumerate(value):
            nested_name = f"{metric_name}_{index}"
            flattened.update(flatten_metric_for_csv(nested_name, nested_value))
        return flattened

    raise TypeError(f"Unsupported averaged metric type for CSV output: {type(value).__name__}")


def should_include_metric(metric_name: str) -> bool:
    excluded_metrics = {
        "confusion_matrix",
    }
    return metric_name not in excluded_metrics


def summarize_test_metrics(json_files: list[Path]) -> dict[str, Any]:
    loaded_objects = [load_json(path) for path in json_files]
    test_metrics_by_file: list[dict[str, Any]] = []

    for path, obj in zip(json_files, loaded_objects):
        metrics = obj.get("metrics", {})
        test_metrics = metrics.get("test", {}) if isinstance(metrics, dict) else {}
        if not isinstance(test_metrics, dict) or not test_metrics:
            raise ValueError(f"Missing or empty metrics.test in {path}")
        test_metrics_by_file.append(test_metrics)

    metric_names = list(test_metrics_by_file[0].keys())
    for path, test_metrics in zip(json_files, test_metrics_by_file):
        if list(test_metrics.keys()) != metric_names:
            raise ValueError(
                f"metrics.test keys do not match across files. Mismatch found in {path}"
            )

    averaged_test_metrics = {
        metric_name: average_nested_values(
            [test_metrics[metric_name] for test_metrics in test_metrics_by_file]
        )
        for metric_name in metric_names
    }

    first_obj = loaded_objects[0]
    model_name = first_obj.get("model_name")
    min_recall = first_obj.get("hyperparameters", {}).get("min_recall_constraint")
    feature_name = json_files[0].parent.name

    summary_row: dict[str, Any] = {
        "model": model_name,
        "min_recall": min_recall,
        "feature": feature_name,
    }
    for metric_name, value in averaged_test_metrics.items():
        if not should_include_metric(metric_name):
            continue
        summary_row.update(flatten_metric_for_csv(metric_name, value))

    return {
        "summary_row": summary_row,
        "num_files": len(json_files),
        "input_files": [path.name for path in json_files],
        "seeds": [infer_seed(obj, path) for path, obj in zip(json_files, loaded_objects)],
    }


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")

    json_files = sorted(path for path in input_dir.glob(args.glob) if path.is_file())
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {input_dir} with glob '{args.glob}'")

    summary = summarize_test_metrics(json_files)
    output_path = input_dir / args.output_name
    fieldnames = list(summary["summary_row"].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(summary["summary_row"])

    print(f"Found {len(json_files)} JSON files in {input_dir}")
    print(f"Wrote averaged test metrics CSV to {output_path}")


if __name__ == "__main__":
    main()
