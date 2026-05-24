from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


BASE_DIR = Path("../../fp_diagnostics_naics")
INPUT_FILES = [
    BASE_DIR / "seed42_test_fp_vs_tn_feature_separation.csv",
    BASE_DIR / "seed25_test_fp_vs_tn_feature_separation.csv",
    BASE_DIR / "seed10_test_fp_vs_tn_feature_separation.csv",
]
OUTPUT_FILE = BASE_DIR / "average_test_fp_vs_tn_feature_separation.csv"


def load_feature_metrics(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def compute_average_metrics(input_files: list[Path]) -> list[dict[str, object]]:
    feature_metrics: dict[tuple[str, str], dict[str, object]] = defaultdict(
        lambda: {"smd_values": [], "ks_values": []}
    )

    for csv_path in input_files:
        rows = load_feature_metrics(csv_path)
        for row in rows:
            feature_name = row["feature"]
            feature_column = row["feature_column_used"]
            key = (feature_name, feature_column)
            feature_metrics[key]["smd_values"].append(float(row["smd"]))
            feature_metrics[key]["ks_values"].append(float(row["ks_stat"]))

    summary_rows = []
    for (feature_name, feature_column), metrics in feature_metrics.items():
        smd_values = metrics["smd_values"]
        ks_values = metrics["ks_values"]
        summary_rows.append(
            {
                "feature": feature_name,
                "feature_column_used": feature_column,
                "avg_smd": sum(smd_values) / len(smd_values),
                "avg_abs_smd": sum(abs(value) for value in smd_values) / len(smd_values),
                "avg_ks_stat": sum(ks_values) / len(ks_values),
                "num_seeds": len(smd_values),
            }
        )

    summary_rows.sort(key=lambda row: row["avg_abs_smd"], reverse=True)
    return summary_rows


def write_summary_csv(summary_rows: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "feature",
        "feature_column_used",
        "avg_smd",
        "avg_abs_smd",
        "avg_ks_stat",
        "num_seeds",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> None:
    missing_files = [str(path) for path in INPUT_FILES if not path.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing input file(s): {', '.join(missing_files)}")

    summary_rows = compute_average_metrics(INPUT_FILES)
    write_summary_csv(summary_rows, OUTPUT_FILE)
    print(f"Wrote {len(summary_rows)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
