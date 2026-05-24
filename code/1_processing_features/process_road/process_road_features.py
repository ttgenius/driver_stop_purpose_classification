#!/usr/bin/env python3
"""
Road Feature Extraction Script

This script extracts road features using the road type category mapping
from feature_classification_metadata/road_type_classification.csv and creates CSV files with road-based features.
"""

import pandas as pd
import numpy as np
import ast
import re
from pathlib import Path
from collections import defaultdict

class RoadFeatureExtractor:
    """Extract road features using new category mapping"""
    
    def __init__(self, mapping_file):
        # Load road type mapping from CSV
        self.load_road_mapping(mapping_file)
        
        # Speed categories (in mph and kmh)
        self.speed_categories = {
            'low_speed': {'mph': [10, 15, 20, 25], 'kmh': [15, 25, 30, 40]},
            'medium_speed': {'mph': [30, 35, 40, 45], 'kmh': [50, 55, 60, 70]},
            'high_speed': {'mph': [50, 55, 60, 65, 70, 80], 'kmh': [80, 90, 100, 110, 120, 130]}
        }
    
    def load_road_mapping(self, mapping_file):
        """Load road type to category mapping from CSV file"""
        try:
            mapping_df = pd.read_csv(mapping_file)
            # Remove any empty rows
            mapping_df = mapping_df.dropna()
            
            # Create mapping dictionary
            self.road_type_mapping = dict(zip(mapping_df['road_type'], mapping_df['category']))
            
            # Get unique categories for feature creation
            self.categories = sorted(mapping_df['category'].unique())
            
            print(f"✅ Loaded road mapping with {len(self.road_type_mapping)} road types")
            print(f"✅ Found {len(self.categories)} categories: {self.categories}")
            
        except Exception as e:
            print(f"❌ Error loading road mapping: {e}")
            # Fallback to original categories
            self.road_type_mapping = {}
            self.categories = ['major_roads', 'local_roads', 'service_roads', 'pedestrian_paths', 'other_roads']
    
    def parse_road_info(self, road_info_str):
        """Parse the road info string into a list of dictionaries"""
        if pd.isna(road_info_str) or road_info_str == '' or road_info_str == '[]':
            return []
        
        try:
            if isinstance(road_info_str, str):
                road_data = ast.literal_eval(road_info_str)
            else:
                road_data = road_info_str
                
            if not isinstance(road_data, list):
                return []
                
            return road_data
        except Exception:
            return []
    
    # def extract_speed_value(self, speed_str):
    #     """Extract numeric speed value from speed string"""
    #     if not speed_str or speed_str == 'not_specified':
    #         return None
    #
    #     # Extract numbers from speed string
    #     numbers = re.findall(r'\d+', str(speed_str))
    #     if numbers:
    #         return int(numbers[0])
    #     return None
    
    # def categorize_speed(self, speed_value, unit='mph'):
    #     """Categorize speed into low/medium/high"""
    #     if speed_value is None:
    #         return None
    #
    #     unit = unit.lower()
    #     if unit not in ['mph', 'kmh']:
    #         unit = 'mph'  # default
    #
    #     if speed_value in self.speed_categories['low_speed'][unit]:
    #         return 'low'
    #     elif speed_value in self.speed_categories['medium_speed'][unit]:
    #         return 'medium'
    #     elif speed_value in self.speed_categories['high_speed'][unit]:
    #         return 'high'
    #     else:
    #         # Categorize by ranges
    #         if speed_value <= 25:
    #             return 'low'
    #         elif speed_value <= 45:
    #             return 'medium'
    #         else:
    #             return 'high'

    def convert_speed_to_kmh(self, speed_str):
        """Convert speed-like strings to km/h float, or return np.nan if not available."""
        if speed_str is None:
            return np.nan
        if isinstance(speed_str, (int, float)):
            # assume numeric is km/h
            return float(speed_str)
        s = str(speed_str).strip().lower()
        if s == '' or s == 'not_specified':
            return np.nan

        # find first numeric token (allow decimals)
        m = re.search(r"(\d+(\.\d+)?)", s)
        if not m:
            return np.nan
        try:
            val = float(m.group(1))
        except Exception:
            return np.nan

        # if mph indicated, convert
        if 'mph' in s:
            return round(val * 1.60934, 2)
        # if km/h or kmh or kph indicated, treat as km/h
        # otherwise assume km/h by default
        return val
    
    def extract_lane_count(self, lanes_str):
        """Extract numeric lane count"""
        if not lanes_str or lanes_str == 'not_specified':
            return None
        
        try:
            return int(lanes_str)
        except (ValueError, TypeError):
            return None
    
    def extract_road_features(self, road_info_str):
        """Extract all road features from road info string using new categories"""
        
        # Initialize feature dictionary with new categories
        features = {}
        
        # Road type categories (counts) - dynamic based on mapping
        for category in self.categories:
            features[f'{category}_count'] = 0

        # Speed features
        features.update({
            'has_speed_limit': 0,
            'max_speed_numeric': -1,
            'low_speed_roads': 0,
            'medium_speed_roads': 0,
            'high_speed_roads': 0,
        })
        
        # Lane features
        features.update({
            'total_lanes': 1,
            'max_lanes': 1,
            'has_multi_lane': 0,
        })
        
        # Distance features
        features.update({
            'has_road_features': 0,
            'closest_road_distance': 151,
            'avg_road_distance': 151,
            'total_roads_count': 0,
        })
        
        # Parse road data
        road_data = self.parse_road_info(road_info_str)
        
        if not road_data:
            return features
        else:
            features['has_road_features'] = 1
        
        # Process each road
        distances = []
        speeds_kmh = []
        lanes = []
        
        for road in road_data:
            if not isinstance(road, dict):
                continue
            
            # Road type categorization using new mapping
            road_type = road.get('road_type', 'unknown').lower()
            
            if road_type in self.road_type_mapping:
                category = self.road_type_mapping[road_type]
                features[f'{category}_count'] += 1
            else:
                # Handle unmapped road types - could add to 'other' category if exists
                if 'other_count' in features:
                    features['other_count'] += 1
            
            # Speed processing
            raw_speed = road.get('max_speed', '')
            speed_kmh = self.convert_speed_to_kmh(raw_speed)
            if not np.isnan(speed_kmh):
                features['has_speed_limit'] = 1
                speeds_kmh.append(speed_kmh)
                # Categorize speed using km/h thresholds
                if speed_kmh <= 40:
                    features['low_speed_roads'] += 1
                elif 40 < speed_kmh <= 80:
                    features['medium_speed_roads'] += 1
                else:
                    features['high_speed_roads'] += 1
            
            # Lane processing
            lanes_str = road.get('lanes', '')
            lane_count = self.extract_lane_count(lanes_str)
            if lane_count is not None:
                lanes.append(lane_count)
            
            # Distance processing
            distance = road.get('distance_m', None)
            if isinstance(distance, (int, float)):
                distances.append(distance)
        
        # Calculate aggregate features
        features['total_roads_count'] = len(road_data)
        
        if distances:
            features['closest_road_distance'] = min(distances)
            features['avg_road_distance'] = np.mean(distances)
        else:
            features['closest_road_distance'] = 151  # max radius 150
            features['avg_road_distance'] = 151
        
        if speeds_kmh:
            features['max_speed_numeric'] = max(speeds_kmh)
        
        if lanes:
            features['total_lanes'] = int(np.sum(lanes))
            features['max_lanes'] = int(max(lanes))
            features['has_multi_lane'] = 1 if max(lanes) > 1 else 0

        return features

def process_csv_file(input_file, output_file, extractor):
    """Process a single CSV file to extract road features"""

    print(f"Processing: {input_file.name}")

    # Load CSV
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"   Error loading {input_file}: {e}")
        return False

    # Extract road features for each row
    road_features_list = []

    for idx, row in df.iterrows():
        road_info = row.get('Stop Road Info', '')
        features = extractor.extract_road_features(road_info)
        road_features_list.append(features)
        #
        # # Progress indicator
        # if (idx + 1) % 1000 == 0:
        #     print(f"   Processed {idx + 1}/{len(df)} rows")

    # Create DataFrame with road features
    road_features_df = pd.DataFrame(road_features_list)
    result_df = pd.concat([df, road_features_df], axis=1)

    # # Before concatenating, add numeric "_has_value" flags and fill missing numeric with -1
    # numeric_cols = road_features_df.select_dtypes(include=[np.number]).columns.tolist()
    #
    # # For reproducibility: ensure ordering of columns
    # # Convert NaNs to -1 but keep a has_value flag for each numeric column
    # for col in numeric_cols:
    #     has_col = f"{col}_has_value"
    #     road_features_df[has_col] = (~road_features_df[col].isna()).astype(int)
    #     road_features_df[col] = road_features_df[col].fillna(-1)

    # # Combine original data with road features
    # result_df = pd.concat([df.reset_index(drop=True), road_features_df.reset_index(drop=True)], axis=1)

    # Save to output file
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_file, index=False)
        print(f"   ✅ Saved: {output_file.name}")
        return True
    except Exception as e:
        print(f"   ❌ Error saving {output_file}: {e}")
        return False

def main():
    """Main function to process all CSV files with updated road categories"""
    
    print("🚀 UPDATED ROAD FEATURE EXTRACTION")
    print("=" * 60)
    
    # Paths
    base_dir = Path(__file__).resolve().parents[3]
    input_dir = base_dir / "input/stop_records"
    output_dir = base_dir / "input/stop_records_with_road_features"
    mapping_file = base_dir / "input/feature_classification_metadata/road_type_classification.csv"
    
    # Check if mapping file exists
    if not mapping_file.exists():
        print(f"❌ Road type mapping file not found: {mapping_file}")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find CSV files
    csv_files = [f for f in input_dir.iterdir() if f.suffix == '.csv' and f.name.startswith('anonymised_part_')]
    csv_files.sort()
    
    if not csv_files:
        print("❌ No CSV files found!")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Initialize feature extractor with new mapping
    extractor = RoadFeatureExtractor(mapping_file)
    
    # Process files
    successful = 0
    failed = 0
    
    for i, csv_file in enumerate(csv_files, 1):
        input_file = csv_file
        output_file = output_dir / csv_file.name.replace('.csv', '_with_road_features.csv')
        
        print(f"\nProcessing file {i}/{len(csv_files)}: {csv_file.name}")
        
        if input_file.exists():
            if process_csv_file(input_file, output_file, extractor):
                successful += 1
            else:
                failed += 1
        else:
            print(f"   Warning: Input file not found: {input_file}")
            failed += 1
    
    # Summary
    print(f"\n📊 PROCESSING SUMMARY")
    print("=" * 40)
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Output directory: {output_dir}")
    
    # Display sample features
    if successful > 0:
        print(f"\n📋 EXTRACTED ROAD FEATURES (Updated Categories):")
        sample_features = extractor.extract_road_features("[]")  # Get feature names
        
        # Group features by type
        road_type_features = [k for k in sample_features.keys() if k.endswith('_count')]
        speed_features = [k for k in sample_features.keys() if 'speed' in k]
        lane_features = [k for k in sample_features.keys() if 'lane' in k]
        distance_features = [k for k in sample_features.keys() if 'distance' in k or 'total_roads' in k]
        
        print(f"\nRoad Type Features ({len(road_type_features)}):")
        for feature in road_type_features:
            print(f"   - {feature}")
        
        print(f"\nSpeed Features ({len(speed_features)}):")
        for feature in speed_features:
            print(f"   - {feature}")
        
        print(f"\nLane Features ({len(lane_features)}):")
        for feature in lane_features:
            print(f"   - {feature}")
        
        print(f"\nDistance Features ({len(distance_features)}):")
        for feature in distance_features:
            print(f"   - {feature}")
    
    print(f"\n✅ Updated road feature extraction completed!")
    print(f"🔄 All files in {output_dir} have been updated with new road categories")

if __name__ == "__main__":
    main()


