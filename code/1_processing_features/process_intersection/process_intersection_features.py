"""
Efficient parallel script to extract intersection features from the 'Nearest Intersections' column.

✅ Adds nearest_intersection_sign (distance to nearest intersection with a sign)
✅ Converts all speeds to km/h
✅ Keeps intersection_max_speed_numeric in km/h
✅ Categorizes speed consistently with process_road_features_updated.py
✅ Produces ML/DL-ready CSV outputs
"""

import os
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from pathlib import Path


# ============================================================
# === Helper Functions ===
# ============================================================

def parse_intersections(cell):
    """Safely parse the 'Nearest Intersections' list-of-dicts from string."""
    try:
        if pd.isna(cell):
            return []
        data = ast.literal_eval(cell)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def convert_speed_to_kmh(speed_str):
    """
    Convert speed strings like '40 mph' or '50 km/h' to float in km/h.
    """
    if not isinstance(speed_str, str) or not speed_str.strip():
        return np.nan

    try:
        value = ''.join(c for c in speed_str if c.isdigit() or c == '.')
        if not value:
            return np.nan
        value = float(value)
        if "mph" in speed_str.lower():
            return round(value * 1.60934, 2)  # Convert mph → km/h
        return value  # Already in km/h
    except Exception:
        return np.nan


def extract_features_from_intersections(intersections):
    """Extracts numeric, categorical, and boolean features from a list of intersection dictionaries."""
    intersection_dict = {
            "has_intersections": 0,
            "closest_intersection_distance": 151,
            "average_intersection_distance": 151,
            "total_intersections": 0,
            "nearest_intersection_sign_distance": 151,
            "intersection_has_speed_limit": 0,
            "intersection_max_speed_numeric": -1,
            "intersection_low_speed_roads": 0,
            "intersection_medium_speed_roads": 0,
            "intersection_high_speed_roads": 0,
            "intersection_has_multi_lane": 0,
            "total_signs": 0,
            "total_stop_sign": 0,
            "total_give_way_sign": 0,
            "has_crossing": 0,
            "has_railway": 0,
            "has_barrier": 0,
        }
    if not intersections:
        # Default values for empty intersection list
        return intersection_dict


    # Initialize accumulators
    distances, max_speeds, lanes = [], [], []
    sign_counts = {
        "traffic_sign": 0, "pedestrian_traffic_signals": 0, "stop": 0,
        "crossing": 0, "traffic_signals": 0, "give_way": 0, "railway_crossing": 0
    }

    has_flags = {"has_crossing": 0, "has_railway": 0, "has_barrier": 0}

    intersections_with_signs = []

    for inter in intersections:
        # --- Distance --- #
        dist = inter.get("distance_m")
        if isinstance(dist, (int, float)):
            distances.append(dist)

        # --- Speed (convert mph → km/h) --- #
        spd = convert_speed_to_kmh(inter.get("max_speed", ""))
        if not np.isnan(spd):
            max_speeds.append(spd)

        # --- Lanes --- #
        ln = inter.get("intersection_lanes", "")
        if isinstance(ln, str) and ln.isdigit():
            lanes.append(int(ln))

        # --- Signs --- #
        signs = inter.get("signs", [])
        if signs:
            intersections_with_signs.append(dist if isinstance(dist, (int, float)) else np.nan)
            for s in signs:
                if s in sign_counts:
                    sign_counts[s] += 1
                if s == 'railway_crossing':
                    has_flags['has_railway'] = 1
                    has_flags['has_crossing'] = 1
                if s == 'crossing':
                    has_flags['has_crossing'] = 1

        # --- Boolean flags --- #
        for key in has_flags:
            if inter.get(key.replace("has_", ""), ""):
                has_flags[key] = 1

    # --- Aggregates --- #
    total_intersections = len(intersections)
    closest_distance = min(distances) if distances else 151
    avg_distance = np.mean(distances) if distances else 151

    # Nearest intersection with ≥1 sign
    nearest_intersection_sign = (
        np.nanmin(intersections_with_signs)
        if intersections_with_signs and not np.all(np.isnan(intersections_with_signs))
        else 151
    )

    # --- Speed Categorization --- #
    if not max_speeds:
        has_intersection_speed_limit = 0
        low = 0
        med = 0
        high = 0
        intersection_max_speed = -1
    else:
        has_intersection_speed_limit = 1
        low = sum(1 for s in max_speeds if s <= 40)
        med = sum(1 for s in max_speeds if 40 < s <= 80)
        high = sum(1 for s in max_speeds if s > 80)
        intersection_max_speed = max(max_speeds)


    max_lanes = np.nanmax(lanes) if lanes else 0
    has_multi_lane = 1 if max_lanes > 1 else 0

    intersection_dict["has_intersections"] = 1
    intersection_dict["closest_intersection_distance"] = closest_distance
    intersection_dict["average_intersection_distance"] = avg_distance
    intersection_dict["total_intersections"] = total_intersections
    intersection_dict["nearest_intersection_sign_distance"] = nearest_intersection_sign
    intersection_dict['intersection_has_speed_limit'] = has_intersection_speed_limit
    intersection_dict['intersection_max_speed_numeric'] = intersection_max_speed
    intersection_dict['intersection_low_speed_roads'] = low
    intersection_dict['intersection_medium_speed_roads'] = med
    intersection_dict['intersection_high_speed_roads'] = high
    intersection_dict['intersection_has_multi_lane'] = has_multi_lane
    intersection_dict['total_signs'] = sum(sign_counts.values())
    intersection_dict['total_stop_sign'] = sign_counts['stop']
    intersection_dict['total_give_way_sign'] = sign_counts['give_way']
    intersection_dict['has_crossing'] = has_flags['has_crossing']
    intersection_dict['has_railway'] = has_flags['has_railway']
    intersection_dict['has_barrier'] = has_flags['has_barrier']

    return intersection_dict


# ============================================================
# === File-level Processing ===
# ============================================================

def process_single_file(file_tuple):
    input_path, output_path = file_tuple
    try:
        df = pd.read_csv(input_path)
        if "Nearest Intersections" not in df.columns:
            print(f"⚠️ Skipped {input_path}: No 'Nearest Intersections' column found.")
            return

        df["Nearest Intersections"] = df["Nearest Intersections"].apply(parse_intersections)
        features_df = df["Nearest Intersections"].apply(extract_features_from_intersections).apply(pd.Series)
        df = pd.concat([df, features_df], axis=1)

        # Replace NaN with -1 and add _has_value indicators
        # numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        # for col in numeric_cols:
        #     df[f"{col}_has_value"] = (~df[col].isna()).astype(int)
        #     df[col] = df[col].fillna(-1)

        df.to_csv(output_path, index=False)
        return output_path

    except Exception as e:
        print(f"❌ Error processing {input_path}: {e}")
        return None


# ============================================================
# === Parallel Processing for All CSVs ===
# ============================================================

def process_all_files(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
    tasks =[]
    for f in csv_files:
        f1 = f.split('.')[0]
        new_f = "{}_with_intersection_features.csv".format(f1)
        tasks.append((os.path.join(input_dir, f), os.path.join(output_dir, new_f)))
    # tasks = [(os.path.join(input_dir, f), os.path.join(output_dir, f)) for f in csv_files]
    print(f"🧮 Processing {len(tasks)} files using {min(cpu_count(), 8)} CPU cores...")

    with Pool(processes=min(cpu_count(), 8)) as pool:
        list(tqdm(pool.imap_unordered(process_single_file, tasks), total=len(tasks)))

    print("✅ All intersection feature files processed successfully.")


# ============================================================
# === Main Entry Point ===
# ============================================================

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[3]
    input_dir = base_dir / "input" / "stop_records"
    output_dir = base_dir / "input" / "stop_records_with_intersection_features"
    process_all_files(input_dir, output_dir)
