#!/usr/bin/env python3
"""
Extract all unique POI categories into a single CSV.

Input:
    input/feature_classification_metadata/poi_category_classification.csv

Output:
    input/poi_category/all_poi_categories.csv
"""

import argparse
import csv
from pathlib import Path


def get_default_paths():
    """Return default input and output paths based on the project layout."""
    base_dir = Path(__file__).resolve().parents[3]
    input_file = base_dir / "input/feature_classification_metadata/poi_category_classification.csv"
    output_file = base_dir / "input/poi_category/all_poi_categories.csv"
    return input_file, output_file


def extract_unique_categories(input_file):
    """
    Read poi_category_classification.csv and return sorted unique category values.

    All columns with names starting with 'poi_category' are included.
    """
    categories = set()

    with input_file.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"Input file has no header row: {input_file}")

        category_columns = [
            column for column in reader.fieldnames if column.startswith("poi_category")
        ]
        if not category_columns:
            raise ValueError(f"No poi_category columns found in: {input_file}")

        for row in reader:
            for column in category_columns:
                category = (row.get(column) or "").strip()
                if category:
                    categories.add(category)

    return sorted(categories)


def write_categories(output_file, categories):
    """Write categories to a one-column CSV."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["poi_category"])
        for category in categories:
            writer.writerow([category])


def main():
    default_input_file, default_output_file = get_default_paths()

    parser = argparse.ArgumentParser(
        description="Extract all unique POI categories from poi_category_classification.csv."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input_file,
        help=f"Input CSV path. Default: {default_input_file}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output_file,
        help=f"Output CSV path. Default: {default_output_file}",
    )
    args = parser.parse_args()

    categories = extract_unique_categories(args.input)
    write_categories(args.output, categories)

    print(f"Read categories from: {args.input}")
    print(f"Wrote {len(categories)} unique categories to: {args.output}")


if __name__ == "__main__":
    main()
