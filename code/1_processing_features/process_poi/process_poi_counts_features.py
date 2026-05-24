#!/usr/bin/env python3
"""
Script to extract POI category counts features for the classification dataset.

This script:
1. Reads poi_category_classification.csv to create a poi dictionary
2. Processes the first 3 CSV files from stop_records folder
3. Adds POI category columns with COUNT values based on POIs column
"""

import pandas as pd
import os
import re
from pathlib import Path

def create_poi_dictionary(poi_file_path):
    """
    Read POI categories file and create dictionary mapping poi_type to poi_category_1
    
    Args:
        poi_file_path (str): Path to poi_category_classification.csv
        
    Returns:
        dict: Dictionary with poi_type as key and poi_category_1 as value
    """
    print(f"Reading POI categories from: {poi_file_path}")
    
    # Read the POI categories CSV
    poi_df = pd.read_csv(poi_file_path)
    
    # Create dictionary mapping poi_type to poi_category_1
    poi_dict = {}
    for _, row in poi_df.iterrows():
        poi_type = row['poi_type']
        poi_category = row['poi_category_1']
        
        # Skip empty or NaN values
        if pd.notna(poi_type) and pd.notna(poi_category):
            poi_dict[poi_type] = poi_category
    
    print(f"Created POI dictionary with {len(poi_dict)} entries")
    return poi_dict

def get_unique_categories(poi_dict):
    """
    Get unique POI categories from the dictionary
    
    Args:
        poi_dict (dict): POI dictionary
        
    Returns:
        list: Sorted list of unique categories
    """
    categories = list(set(poi_dict.values()))
    categories.sort()
    return categories

def parse_pois_column(pois_text):
    """
    Parse POIs column to extract poi types
    
    Args:
        pois_text (str): POIs column content like "restaurant - 128.744m; convenience - 131.555m"
        
    Returns:
        list: List of poi types found
    """
    if pd.isna(pois_text) or pois_text == "":
        return []
    
    poi_types = []
    # Split by semicolon to get individual poi-distance pairs
    pairs = pois_text.split(';')
    
    for pair in pairs:
        pair = pair.strip()
        if pair:
            # Extract poi type (everything before " - ")
            match = re.match(r'^([^-]+)\s*-\s*[\d.]+m$', pair)
            if match:
                poi_type = match.group(1).strip()
                poi_types.append(poi_type)
    
    return poi_types

def process_csv_file(input_file, output_file, poi_dict, categories):
    """
    Process a single CSV file: copy content and add POI category columns with COUNTS
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
        poi_dict (dict): POI dictionary
        categories (list): List of unique categories
    """
    print(f"Processing {input_file} -> {output_file}")
    
    # Read the input CSV
    df = pd.read_csv(input_file)
    
    # Add new columns for each POI category with default value 0
    for category in categories:
        df[category] = 0
    
    # Add total_pois column with default value 0
    df['total_pois'] = 0
    
    # Process each row to set POI category COUNTS and total POIs
    for idx, row in df.iterrows():
        pois_text = row.get('POIs', '')
        poi_types = parse_pois_column(pois_text)
        
        # Set the total_pois count
        df.at[idx, 'total_pois'] = len(poi_types)
        
        # Count POIs by category
        category_counts = {}
        for poi_type in poi_types:
            if poi_type in poi_dict:
                category = poi_dict[poi_type]
                if category in category_counts:
                    category_counts[category] += 1
                else:
                    category_counts[category] = 1
        
        # Set the counts in the DataFrame
        for category, count in category_counts.items():
            if category in df.columns:
                df.at[idx, category] = count
    
    # Save the processed DataFrame
    df.to_csv(output_file, index=False)
    print(f"Saved processed file: {output_file}")
    print(f"Added {len(categories)} POI category columns (with counts) + 1 total_pois column")

def main():
    """Main function to orchestrate the processing"""
    
    # Define paths
    base_dir = Path(__file__).resolve().parents[3]
    poi_file = base_dir / "input/feature_classification_metadata/poi_category_classification.csv"
    input_dir = base_dir / "input/stop_records"
    output_dir = base_dir / "input/stop_records_with_poi_features"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    
    # Step 1: Create POI dictionary
    poi_dict = create_poi_dictionary(poi_file)
    
    # Step 2: Get unique categories
    categories = get_unique_categories(poi_dict)
    print(f"Found {len(categories)} unique POI categories:")
    for i, cat in enumerate(categories, 1):
        print(f"  {i}. {cat}")
    
    # Step 3: Process all CSV files in the stop_records folder
    csv_files = [f for f in input_dir.iterdir() if f.suffix == '.csv' and f.name.startswith('anonymised_part_')]
    csv_files.sort()  # Sort to process in order
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    for i, csv_file in enumerate(csv_files, 1):
        input_file = csv_file  # csv_file is already a Path object
        output_file = output_dir / csv_file.name.replace('.csv', '_with_poi_features.csv')
        
        print(f"\nProcessing file {i}/{len(csv_files)}: {csv_file.name}")
        
        if input_file.exists():
            process_csv_file(input_file, output_file, poi_dict, categories)
        else:
            print(f"Warning: Input file not found: {input_file}")
    
    print("\nProcessing completed!")
    
    # Display sample of processed data
    sample_file = output_dir / "anonymised_part_001_with_poi_features.csv"
    if sample_file.exists():
        print(f"\nSample of processed data from {sample_file}:")
        sample_df = pd.read_csv(sample_file)
        print(f"Shape: {sample_df.shape}")
        
        # Show first few rows with POI data
        poi_rows = sample_df[sample_df['POIs'].notna() & (sample_df['POIs'] != '')].head(3)
        if not poi_rows.empty:
            print(f"\nSample rows with POI data (showing counts):")
            for idx, row in poi_rows.iterrows():
                print(f"\nRow {idx}:")
                print(f"  POIs: {row['POIs'][:100]}{'...' if len(row['POIs']) > 100 else ''}")
                print(f"  Total POIs: {row['total_pois']}")
                
                # Show category counts > 0
                category_counts = []
                for category in categories:
                    count = row.get(category, 0)
                    if count > 0:
                        category_counts.append(f"{category}: {count}")
                
                print(f"  Category counts: {', '.join(category_counts[:5])}")
                if len(category_counts) > 5:
                    print(f"    ... and {len(category_counts) - 5} more categories")

if __name__ == "__main__":
    main()
