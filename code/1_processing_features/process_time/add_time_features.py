import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Input and output directories
BASE_DIR = Path(__file__).resolve().parents[3]
INPUT_DIR = BASE_DIR / "input" / "stop_records"
OUTPUT_DIR = BASE_DIR / "input" / "stop_records_with_time_features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Define time columns to process
TIME_COLUMNS = [
    "stop_start_time_local",
    "stop_end_time_local",
]

def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based numerical and cyclical features."""
    for col in TIME_COLUMNS:
        if col not in df.columns:
            continue

        # Convert to datetime safely
        df[col] = pd.to_datetime(df[col], errors="coerce")

        # Drop NaT to avoid errors later
        valid_mask = df[col].notna()

        # Seconds since midnight
        df[f"{col}_seconds"] = np.where(
            valid_mask,
            df[col].dt.hour * 3600 + df[col].dt.minute * 60 + df[col].dt.second,
            -1  # -1 for missing values
        )

        # Cyclical encoding (sin and cos)
        seconds_in_day = 24 * 3600
        radians = 2 * np.pi * df[f"{col}_seconds"].clip(lower=0) / seconds_in_day

        df[f"{col}_sin"] = np.sin(radians)
        df[f"{col}_cos"] = np.cos(radians)

    # Add is_weekend feature (based on stop_start_time_local if available)
    if "stop_start_time_local" in df.columns:
        df["is_weekend"] = np.where(
            df["stop_start_time_local"].dt.dayofweek.isin([5, 6]), 1, 0
        )
    else:
        df["is_weekend"] = -1  # fallback if missing

    return df


def process_stop_dataset(input_dir: str, output_dir: str):
    """Process all CSVs in input_dir and save enhanced ones to output_dir."""
    csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]

    for i, filename in enumerate(csv_files, start=1):
        input_path = os.path.join(input_dir, filename)
        f1 = filename.split('.')[0]
        new_f = "{}_with_time_features.csv".format(f1)
        output_path = os.path.join(output_dir, new_f)

        print(f"[{i}/{len(csv_files)}] Processing {filename}...")

        try:
            df = pd.read_csv(input_path)
            df_enhanced = extract_time_features(df)
            df_enhanced.to_csv(output_path, index=False)
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            continue

    print(f"\n✅ Done! Enhanced CSVs saved to '{output_dir}'")


if __name__ == "__main__":
    process_stop_dataset(INPUT_DIR, OUTPUT_DIR)
