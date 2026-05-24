#!/usr/bin/env python3
"""
Print the percentage of rows with no value in the POIs column.

The script scans all CSV files in:
input/stop_records_with_poi_features
"""

from pathlib import Path

import pandas as pd


CHUNK_SIZE = 100_000


def count_missing_pois(csv_path):
    """Return (total_rows, missing_pois_rows) for one CSV file."""
    total_rows = 0
    missing_rows = 0

    try:
        chunks = pd.read_csv(
            csv_path,
            usecols=["POIs"],
            chunksize=CHUNK_SIZE,
            keep_default_na=True,
        )

        for chunk in chunks:
            pois = chunk["POIs"]
            missing_mask = pois.isna() | (pois.astype(str).str.strip() == "")
            total_rows += len(chunk)
            missing_rows += int(missing_mask.sum())

    except ValueError as exc:
        if "Usecols do not match columns" in str(exc):
            print(f"Skipping {csv_path.name}: missing required column 'POIs'")
            return 0, 0
        raise

    return total_rows, missing_rows


def format_percentage(part, whole):
    if whole == 0:
        return "0.00%"
    return f"{part / whole * 100:.2f}%"


def main():
    base_dir = Path(__file__).resolve().parents[3]
    input_dir = base_dir / "input" / "stop_records_with_poi_features"

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in: {input_dir}")
        return

    print(f"Scanning {len(csv_files)} CSV file(s) in: {input_dir}")
    print()

    grand_total_rows = 0
    grand_missing_rows = 0

    for csv_path in csv_files:
        total_rows, missing_rows = count_missing_pois(csv_path)
        grand_total_rows += total_rows
        grand_missing_rows += missing_rows

        print(
            f"{csv_path.name}: {missing_rows:,}/{total_rows:,} rows "
            f"without POIs ({format_percentage(missing_rows, total_rows)})"
        )

    print()
    print(
        f"Overall: {grand_missing_rows:,}/{grand_total_rows:,} rows "
        f"without POIs ({format_percentage(grand_missing_rows, grand_total_rows)})"
    )


if __name__ == "__main__":
    main()
