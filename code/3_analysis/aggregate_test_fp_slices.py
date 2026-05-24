from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


DEFAULT_INPUT_SUFFIX = "test_fp_slices.csv"
DEFAULT_OUTPUT_NAME = "average_test_fp_slices.csv"
NUMERIC_COLUMNS = ("n_nonwork", "fp", "fpr", "fp_share")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Average false-positive slice metrics across seed-specific CSV files in a folder."
        )
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=r"C:\Users\masters_research\driver_stop_purpose_classification\fp_diagnostics_naics",
        help="Folder containing seed-specific FP slice CSV files.",
    )
    parser.add_argument(
        "--input-suffix",
        default=DEFAULT_INPUT_SUFFIX,
        help="Filename suffix used to discover input CSV files.",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help="Output CSV filename written into the input folder.",
    )
    return parser.parse_args()


def find_input_files(input_dir: Path, input_suffix: str) -> list[Path]:
    files = sorted(path for path in input_dir.glob(f"*{input_suffix}") if path.is_file())
    if not files:
        raise FileNotFoundError(
            f"No files ending with '{input_suffix}' were found in {input_dir}"
        )
    return files


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_slice_metrics(input_files: list[Path]) -> list[dict[str, object]]:
    grouped_metrics: dict[tuple[str, str], dict[str, object]] = defaultdict(
        lambda: {
            "slice": None,
            "bucket": None,
            "n_nonwork": [],
            "fp": [],
            "fpr": [],
            "fp_share": [],
        }
    )
    order_by_key: dict[tuple[str, str], int] = {}
    next_order = 0

    for csv_path in input_files:
        for row in load_rows(csv_path):
            key = (row["slice"], row["bucket"])
            metrics = grouped_metrics[key]
            metrics["slice"] = row["slice"]
            metrics["bucket"] = row["bucket"]

            if key not in order_by_key:
                order_by_key[key] = next_order
                next_order += 1

            for column in NUMERIC_COLUMNS:
                metrics[column].append(float(row[column]))

    summary_rows: list[dict[str, object]] = []
    for key, metrics in grouped_metrics.items():
        num_files = len(metrics["n_nonwork"])
        summary_rows.append(
            {
                "slice": metrics["slice"],
                "bucket": metrics["bucket"],
                "avg_n_nonwork": sum(metrics["n_nonwork"]) / num_files,
                "avg_fp": sum(metrics["fp"]) / num_files,
                "avg_fpr": sum(metrics["fpr"]) / num_files,
                "avg_fp_share": sum(metrics["fp_share"]) / num_files,
                "num_seeds": num_files,
            }
        )

    summary_rows.sort(key=lambda row: order_by_key[(row["slice"], row["bucket"])])
    return summary_rows


def write_summary_csv(summary_rows: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "slice",
        "bucket",
        "avg_n_nonwork",
        "avg_fp",
        "avg_fpr",
        "avg_fp_share",
        "num_seeds",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")

    input_files = find_input_files(input_dir, args.input_suffix)
    summary_rows = summarize_slice_metrics(input_files)
    output_path = input_dir / args.output_name
    write_summary_csv(summary_rows, output_path)

    print(f"Found {len(input_files)} input files in {input_dir}")
    print(f"Wrote {len(summary_rows)} summary rows to {output_path}")


if __name__ == "__main__":
    main()
