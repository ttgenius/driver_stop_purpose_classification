#!/usr/bin/env python3
"""
Extract distinct POI types from all CSV files in the stop_records_parts folder.
The POIs column contains values like "platform - 92.782m; platform - 118.193m; stop_position - 129.562m"
where POI types are separated from distances by ' - ' and pairs are separated by ';'.
"""

import os
import csv
from pathlib import Path

def extract_poi_types_from_csv(csv_file_path):
    """
    Extract all POI types from a single CSV file.
    
    Args:
        csv_file_path (str): Path to the CSV file
    
    Returns:
        set: Set of unique POI types found in this file
    """
    poi_types = set()
    
    try:
        # Read the CSV file
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            for row in reader:
                pois_column = row.get('POIs', '')
                
                # Skip empty POIs columns
                if not pois_column or pois_column.strip() == '':
                    continue
                
                # Split by semicolon to get individual POI entries
                poi_entries = pois_column.split(';')
                
                for entry in poi_entries:
                    entry = entry.strip()
                    if entry:
                        # Split by ' - ' to separate POI type from distance
                        parts = entry.split(' - ')
                        if len(parts) >= 2:
                            poi_type = parts[0].strip()
                            if poi_type:
                                poi_types.add(poi_type)
    
    except Exception as e:
        print(f"Error processing {csv_file_path}: {e}")
    
    return poi_types

def extract_all_poi_types(input_folder):
    """
    Extract all distinct POI types from all CSV files in the input folder.
    
    Args:
        input_folder (str): Path to the folder containing CSV files
    
    Returns:
        set: Set of all unique POI types found across all files
    """
    all_poi_types = set()
    
    # Get all CSV files in the folder
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    
    print(f"Found {len(csv_files)} CSV files to process...")
    
    for i, csv_file in enumerate(csv_files, 1):
        csv_file_path = os.path.join(input_folder, csv_file)
        print(f"Processing {i}/{len(csv_files)}: {csv_file}")
        
        file_poi_types = extract_poi_types_from_csv(csv_file_path)
        all_poi_types.update(file_poi_types)
        
        print(f"  Found {len(file_poi_types)} unique POI types in this file")
        print(f"  Total unique POI types so far: {len(all_poi_types)}")
    
    return all_poi_types

def save_poi_types_to_csv(poi_types, output_file_path):
    """
    Save the distinct POI types to a CSV file.
    
    Args:
        poi_types (set): Set of unique POI types
        output_file_path (str): Path to the output CSV file
    """
    # Sort POI types alphabetically for better readability
    sorted_poi_types = sorted(poi_types)
    
    with open(output_file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['POI_Type'])  # Header
        
        for poi_type in sorted_poi_types:
            writer.writerow([poi_type])
    
    print(f"Saved {len(sorted_poi_types)} distinct POI types to {output_file_path}")

def main():
    # Define input and output paths (relative to the current script location)
    base_dir = Path(__file__).resolve().parents[3]
    input_folder = base_dir / "input" / "stop_records"
    output_folder = base_dir / "input" / "poi_category" / "stop_records_output"
    output_file = os.path.join(output_folder, "POI_types.csv")
    
    print("Starting POI type extraction...")
    print(f"Input folder: {input_folder}")
    print(f"Output file: {output_file}")
    print("-" * 50)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Extract all distinct POI types
    all_poi_types = extract_all_poi_types(input_folder)
    
    print("-" * 50)
    print(f"Extraction complete! Found {len(all_poi_types)} distinct POI types:")
    
    # Display the first 20 POI types as a preview
    sorted_types = sorted(all_poi_types)
    for i, poi_type in enumerate(sorted_types[:20]):
        print(f"  {i+1}. {poi_type}")
    
    if len(sorted_types) > 20:
        print(f"  ... and {len(sorted_types) - 20} more")
    
    # Save to CSV
    save_poi_types_to_csv(all_poi_types, output_file)
    
    print("-" * 50)
    print("Process completed successfully!")

if __name__ == "__main__":
    main()
