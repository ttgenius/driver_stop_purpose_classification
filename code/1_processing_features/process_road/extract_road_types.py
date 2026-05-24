#!/usr/bin/env python3
"""
Road Types Extraction Script

This script extracts all unique road types from the 'road_type' attribute 
in the 'Stop Road Info' column across all CSV files in the stop_records folder.
Outputs the results to a CSV file with counts and statistics.
"""

import pandas as pd
import ast
from pathlib import Path
from collections import Counter, defaultdict
import json

def extract_road_types_from_file(csv_file):
    """Extract road types from a single CSV file"""
    
    print(f"Processing: {csv_file.name}")
    
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"   Error loading {csv_file}: {e}")
        return Counter(), 0, 0
    
    road_types = Counter()
    valid_entries = 0
    error_entries = 0
    
    for idx, row in df.iterrows():
        road_info = row.get('Stop Road Info', '')
        
        if pd.isna(road_info) or road_info == '' or road_info == '[]':
            continue
        
        try:
            # Parse the road info string
            if isinstance(road_info, str):
                road_data = ast.literal_eval(road_info)
            else:
                road_data = road_info
            
            if not isinstance(road_data, list):
                error_entries += 1
                continue
            
            valid_entries += 1
            
            # Extract road types from each road in the list
            for road in road_data:
                if isinstance(road, dict):
                    road_type = road.get('road_type', 'unknown')
                    if road_type:  # Only count non-empty road types
                        road_types[road_type] += 1
                        
        except Exception as e:
            error_entries += 1
            if error_entries <= 3:  # Show first 3 errors per file
                print(f"   Error parsing row {idx}: {e}")
    
    print(f"   Valid entries: {valid_entries}, Errors: {error_entries}")
    print(f"   Found {len(road_types)} unique road types")
    
    return road_types, valid_entries, error_entries

def main():
    """Main function to extract road types from all CSV files"""
    
    print("🚀 ROAD TYPES EXTRACTION")
    print("=" * 60)
    
    # Paths
    base_dir = Path(__file__).resolve().parents[3]
    input_dir = base_dir / "input/stop_records"
    output_dir = base_dir / "input/road_types"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find CSV files
    csv_files = [f for f in input_dir.iterdir() if f.suffix == '.csv' and f.name.startswith('anonymised_part_')]
    csv_files.sort()
    
    if not csv_files:
        print("❌ No CSV files found!")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Process all files
    all_road_types = Counter()
    file_statistics = []
    total_valid_entries = 0
    total_error_entries = 0
    
    for i, csv_file in enumerate(csv_files, 1):
        print(f"\nProcessing file {i}/{len(csv_files)}: {csv_file.name}")
        
        road_types, valid_entries, error_entries = extract_road_types_from_file(csv_file)
        
        # Accumulate results
        all_road_types.update(road_types)
        total_valid_entries += valid_entries
        total_error_entries += error_entries
        
        # Store file statistics
        file_statistics.append({
            'file_name': csv_file.name,
            'valid_entries': valid_entries,
            'error_entries': error_entries,
            'unique_road_types': len(road_types),
            'total_road_instances': sum(road_types.values())
        })
        
        # Progress update every 20 files
        if i % 20 == 0:
            print(f"   Progress: {i}/{len(csv_files)} files processed")
            print(f"   Total unique road types so far: {len(all_road_types)}")
    
    # Create results summary
    print(f"\n📊 EXTRACTION SUMMARY")
    print("=" * 50)
    print(f"Files processed: {len(csv_files)}")
    print(f"Total valid entries: {total_valid_entries}")
    print(f"Total error entries: {total_error_entries}")
    print(f"Unique road types found: {len(all_road_types)}")
    print(f"Total road instances: {sum(all_road_types.values())}")
    
    # Create road types DataFrame
    road_types_data = []
    for road_type, count in all_road_types.most_common():
        percentage = (count / sum(all_road_types.values())) * 100
        road_types_data.append({
            'road_type': road_type,
            'count': count,
            'percentage': round(percentage, 4),
            'rank': len(road_types_data) + 1
        })
    
    road_types_df = pd.DataFrame(road_types_data)
    
    # Create file statistics DataFrame
    file_stats_df = pd.DataFrame(file_statistics)
    
    # Save road types to CSV
    road_types_output = output_dir / "all_road_types.csv"
    road_types_df.to_csv(road_types_output, index=False)
    print(f"\n✅ Road types saved to: {road_types_output}")
    
    # Save file statistics to CSV
    file_stats_output = output_dir / "file_statistics.csv"
    file_stats_df.to_csv(file_stats_output, index=False)
    print(f"✅ File statistics saved to: {file_stats_output}")
    
    # Create summary statistics
    summary_stats = {
        'extraction_date': pd.Timestamp.now().isoformat(),
        'total_files_processed': len(csv_files),
        'total_valid_entries': total_valid_entries,
        'total_error_entries': total_error_entries,
        'unique_road_types_count': len(all_road_types),
        'total_road_instances': sum(all_road_types.values()),
        'most_common_road_type': all_road_types.most_common(1)[0] if all_road_types else None,
        'least_common_road_types': [item for item in all_road_types.most_common() if item[1] == 1]
    }
    
    # Save summary to JSON
    summary_output = output_dir / "extraction_summary.json"
    with open(summary_output, 'w') as f:
        json.dump(summary_stats, f, indent=2, default=str)
    print(f"✅ Summary statistics saved to: {summary_output}")
    
    # Display top road types
    print(f"\n🏆 TOP 20 MOST COMMON ROAD TYPES:")
    print("-" * 50)
    print(f"{'Rank':<4} {'Road Type':<20} {'Count':<8} {'Percentage':<10}")
    print("-" * 50)
    
    for i, (road_type, count) in enumerate(all_road_types.most_common(20), 1):
        percentage = (count / sum(all_road_types.values())) * 100
        print(f"{i:<4} {road_type:<20} {count:<8} {percentage:<10.2f}%")
    
    # Display rare road types (count = 1)
    rare_road_types = [road_type for road_type, count in all_road_types.items() if count == 1]
    if rare_road_types:
        print(f"\n🔍 RARE ROAD TYPES (Count = 1): {len(rare_road_types)} types")
        print(f"Examples: {', '.join(rare_road_types[:10])}")
        if len(rare_road_types) > 10:
            print(f"... and {len(rare_road_types) - 10} more")
    
    # Create categorization suggestions
    print(f"\n💡 ROAD TYPE CATEGORIZATION SUGGESTIONS:")
    print("-" * 50)
    
    # Analyze road types for categorization
    major_roads = [rt for rt in all_road_types.keys() if any(keyword in rt.lower() 
                  for keyword in ['primary', 'secondary', 'trunk', 'motorway'])]
    local_roads = [rt for rt in all_road_types.keys() if any(keyword in rt.lower() 
                  for keyword in ['residential', 'tertiary', 'unclassified'])]
    service_roads = [rt for rt in all_road_types.keys() if any(keyword in rt.lower() 
                    for keyword in ['service', 'living_street'])]
    pedestrian_paths = [rt for rt in all_road_types.keys() if any(keyword in rt.lower() 
                       for keyword in ['footway', 'path', 'steps', 'pedestrian', 'cycleway'])]
    
    print(f"Major roads: {len(major_roads)} types")
    print(f"Local roads: {len(local_roads)} types")
    print(f"Service roads: {len(service_roads)} types")
    print(f"Pedestrian paths: {len(pedestrian_paths)} types")
    
    categorized_count = len(set(major_roads + local_roads + service_roads + pedestrian_paths))
    uncategorized_count = len(all_road_types) - categorized_count
    print(f"Uncategorized: {uncategorized_count} types")
    
    print(f"\n✅ Road types extraction completed!")
    print(f"📁 Output directory: {output_dir}")

if __name__ == "__main__":
    main()


