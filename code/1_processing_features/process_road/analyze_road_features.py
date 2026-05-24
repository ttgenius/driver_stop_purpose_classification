#!/usr/bin/env python3
"""
Road Feature Analysis Script

This script analyzes the road features from the Stop Road Info column
to understand the structure and create feature extraction categories.
"""

import pandas as pd
import ast
import json
from collections import Counter, defaultdict
from pathlib import Path

def analyze_road_data(csv_file, sample_size=1000):
    """Analyze road data structure from a sample CSV file"""
    
    print(f"📊 Analyzing road data from: {csv_file.name}")
    
    # Load sample data
    df = pd.read_csv(csv_file)
    
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    print(f"Sample size: {len(df)} records")
    
    # Analyze Stop Road Info column
    road_types = Counter()
    max_speeds = Counter()
    access_types = Counter()
    lane_counts = Counter()
    distance_stats = []
    
    valid_road_entries = 0
    empty_road_entries = 0
    error_entries = 0
    
    for idx, row in df.iterrows():
        road_info = row.get('Stop Road Info', '')
        
        if pd.isna(road_info) or road_info == '' or road_info == '[]':
            empty_road_entries += 1
            continue
            
        try:
            # Parse the road info (it's stored as string representation of list)
            if isinstance(road_info, str):
                road_data = ast.literal_eval(road_info)
            else:
                road_data = road_info
                
            if not isinstance(road_data, list):
                error_entries += 1
                continue
                
            valid_road_entries += 1
            
            # Analyze each road in the list
            for road in road_data:
                if isinstance(road, dict):
                    # Road type analysis
                    road_type = road.get('road_type', 'unknown')
                    road_types[road_type] += 1
                    
                    # Max speed analysis
                    max_speed = road.get('max_speed', 'unknown')
                    if max_speed == '':
                        max_speed = 'not_specified'
                    max_speeds[max_speed] += 1
                    
                    # Access type analysis
                    access = road.get('access', 'unknown')
                    if access == '':
                        access = 'not_specified'
                    access_types[access] += 1
                    
                    # Lane count analysis
                    lanes = road.get('lanes', 'unknown')
                    if lanes == '':
                        lanes = 'not_specified'
                    lane_counts[lanes] += 1
                    
                    # Distance analysis
                    distance = road.get('distance_m', 0)
                    if isinstance(distance, (int, float)) and distance > 0:
                        distance_stats.append(distance)
                        
        except Exception as e:
            error_entries += 1
            if error_entries <= 5:  # Show first 5 errors
                print(f"   Error parsing row {idx}: {e}")
    
    # Print analysis results
    print(f"\n📈 ROAD DATA ANALYSIS RESULTS")
    print("=" * 50)
    print(f"Valid road entries: {valid_road_entries}")
    print(f"Empty road entries: {empty_road_entries}")
    print(f"Error entries: {error_entries}")
    
    print(f"\n🛣️  ROAD TYPES (Top 15):")
    for road_type, count in road_types.most_common(15):
        print(f"   {road_type}: {count}")
    
    print(f"\n🚗 MAX SPEEDS (Top 10):")
    for speed, count in max_speeds.most_common(10):
        print(f"   {speed}: {count}")
    
    print(f"\n🚪 ACCESS TYPES (Top 10):")
    for access, count in access_types.most_common(10):
        print(f"   {access}: {count}")
    
    print(f"\n🛤️  LANE COUNTS (Top 10):")
    for lanes, count in lane_counts.most_common(10):
        print(f"   {lanes}: {count}")
    
    if distance_stats:
        import numpy as np
        print(f"\n📏 DISTANCE STATISTICS:")
        print(f"   Mean: {np.mean(distance_stats):.2f}m")
        print(f"   Median: {np.median(distance_stats):.2f}m")
        print(f"   Min: {np.min(distance_stats):.2f}m")
        print(f"   Max: {np.max(distance_stats):.2f}m")
        print(f"   Std: {np.std(distance_stats):.2f}m")
    
    return {
        'road_types': road_types,
        'max_speeds': max_speeds,
        'access_types': access_types,
        'lane_counts': lane_counts,
        'distance_stats': distance_stats,
        'valid_entries': valid_road_entries,
        'empty_entries': empty_road_entries,
        'error_entries': error_entries
    }

def suggest_road_features(analysis_results):
    """Suggest road features based on analysis results"""
    
    print(f"\n💡 SUGGESTED ROAD FEATURES FOR ML")
    print("=" * 50)
    
    # Road type categories
    road_types = analysis_results['road_types']
    print("🛣️  Road Type Features (Binary/Count):")
    
    # Group similar road types
    road_categories = {
        'major_roads': ['primary', 'secondary', 'trunk', 'motorway', 'motorway_link'],
        'local_roads': ['residential', 'tertiary', 'unclassified'],
        'service_roads': ['service', 'living_street'],
        'pedestrian_paths': ['footway', 'path', 'steps', 'pedestrian', 'cycleway'],
        'parking_areas': ['parking', 'parking_aisle'],
        'other_roads': []
    }
    
    # Categorize road types
    categorized = set()
    for category, road_list in road_categories.items():
        category_count = sum(road_types.get(rt, 0) for rt in road_list)
        if category_count > 0:
            print(f"   {category}: {category_count} occurrences")
            categorized.update(road_list)
    
    # Find uncategorized road types
    uncategorized = set(road_types.keys()) - categorized - {'unknown'}
    if uncategorized:
        print(f"   uncategorized_roads: {uncategorized}")
        road_categories['other_roads'] = list(uncategorized)
    
    # Speed-based features
    max_speeds = analysis_results['max_speeds']
    print(f"\n🚗 Speed-Based Features:")
    print(f"   has_speed_limit: Binary (whether any road has speed limit)")
    print(f"   max_speed_numeric: Continuous (highest speed limit in mph/kmh)")
    print(f"   speed_categories: Binary for low/medium/high speed roads")
    
    # Lane-based features
    lane_counts = analysis_results['lane_counts']
    print(f"\n🛤️  Lane-Based Features:")
    print(f"   total_lanes: Sum of all lane counts")
    print(f"   max_lanes: Maximum lanes on any road")
    print(f"   has_multi_lane: Binary (whether any road has >1 lane)")
    
    # Distance-based features
    if analysis_results['distance_stats']:
        print(f"\n📏 Distance-Based Features:")
        print(f"   closest_road_distance: Distance to nearest road")
        print(f"   avg_road_distance: Average distance to all roads")
        print(f"   total_roads_count: Number of roads within radius")
        print(f"   roads_within_50m: Count of roads within 50m")
        print(f"   roads_within_100m: Count of roads within 100m")
    
    return road_categories

def main():
    """Main function to analyze road features"""
    
    print("🚀 ROAD FEATURE ANALYSIS")
    print("=" * 60)
    
    # Path to kaggle dataset
    data_dir = Path(__file__).resolve().parents[3] / "input" / "stop_records"
    
    # Find CSV files
    csv_files = list(data_dir.glob("anonymised_part_*.csv"))
    
    if not csv_files:
        print("❌ No CSV files found!")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Analyze first few files
    all_results = {}
    for i, csv_file in enumerate(csv_files[:3], 1):
        print(f"\n--- Analyzing File {i}/3 ---")
        results = analyze_road_data(csv_file, sample_size=500)
        all_results[csv_file.name] = results
    
    # Combine results from all files
    combined_road_types = Counter()
    combined_max_speeds = Counter()
    combined_access_types = Counter()
    combined_lane_counts = Counter()
    combined_distance_stats = []
    
    for results in all_results.values():
        combined_road_types.update(results['road_types'])
        combined_max_speeds.update(results['max_speeds'])
        combined_access_types.update(results['access_types'])
        combined_lane_counts.update(results['lane_counts'])
        combined_distance_stats.extend(results['distance_stats'])
    
    combined_results = {
        'road_types': combined_road_types,
        'max_speeds': combined_max_speeds,
        'access_types': combined_access_types,
        'lane_counts': combined_lane_counts,
        'distance_stats': combined_distance_stats
    }
    
    # Suggest features
    road_categories = suggest_road_features(combined_results)
    
    print(f"\n✅ Analysis complete!")
    print(f"Next step: Create road feature extraction script using these categories")
    
    return road_categories

if __name__ == "__main__":
    main()
