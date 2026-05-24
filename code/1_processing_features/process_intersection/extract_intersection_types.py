import pandas as pd
import ast
import os
import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
INPUT_DIR = BASE_DIR / 'input' / 'stop_records'
OUTPUT_FILE = BASE_DIR / 'input' / 'intesection_types'/ 'intersection_types.csv'


def get_intersection_types():
    """Extracts intersection features, including distance-weighted and density-based metrics."""
    intersection_types = set()
    for f in os.listdir(INPUT_DIR):
        print("file to extract", f)
        df = pd.read_csv(os.path.join(INPUT_DIR, f))
        for _, row in df["Nearest Intersections"].fillna("[]").items():
            try:
                intersections = ast.literal_eval(row)
                if not isinstance(intersections, list):
                    intersections = []
            except Exception:
                intersections = []

            for inter in intersections:
                if not isinstance(inter, dict):
                    continue
                intersection_type = inter.get("intersection_type")
                if intersection_type:
                    intersection_types.add(intersection_type)

    print("intersection types")
    print(intersection_types)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(["intersection_type"])
        types = sorted(intersection_types)
        for t in types:
            csvwriter.writerow([t])
    print(f"saved intersection types to {OUTPUT_FILE}")

if __name__ == "__main__":
    get_intersection_types()
