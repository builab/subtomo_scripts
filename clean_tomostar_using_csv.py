#!/usr/bin/env python
"""
Script to remove entry in tomostar file using a csv file
csv file format
filename,zvalues_to_remove
File_001.tomostar,"0,1,40"
File_002.tomostar,""
File_003.tomostar,"0,1,2"
File_004.tomostar,"1-3,39-41"

Requires starfile package (pip install starfile)
2025/04/02 Huy Bui, McGill
"""

import os
import csv
import argparse
import shutil
import re
import starfile

def parse_range(range_str):
    """Parse range string like '1-3' or '5' into list of integers"""
    if '-' in range_str:
        start, end = map(int, range_str.split('-'))
        return list(range(start, end + 1))
    else:
        return [int(range_str)]

def parse_zvalues(zvalue_str):
    """Parse zvalues string with possible ranges (e.g., '1-3,39-41' or '1,2,3,39,40,41')"""
    if not zvalue_str:
        return []
    
    zvalues = []
    for part in zvalue_str.split(','):
        part = part.strip()
        if part:
            zvalues.extend(parse_range(part))
    
    # Convert to 0-based if needed (here assuming input is 1-based)
    return [z - 1 for z in zvalues]

def load_removal_csv(csv_path):
    """Load removal instructions from CSV"""
    removal_map = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            tomo_name = row['TomoName']
            zvalues_str = row['ExcludedViews']
            zvalues = parse_zvalues(zvalues_str)
            removal_map[tomo_name] = zvalues
    return removal_map

def process_tomostar(input_path, output_path, zvalues_to_remove):
    """Process a tomostar file, removing specified indices"""
    try:
        # Read star file
        df = starfile.read(input_path)
        
        # Remove specified indices (0-based)
        if zvalues_to_remove:
            df = df.drop(zvalues_to_remove).reset_index(drop=True)
        
        # Write output
        starfile.write(df, output_path)
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Clean tomostar files based on CSV instructions')
    parser.add_argument('--input_dir', required=True, help='Directory containing input tomostar files')
    parser.add_argument('--csv', required=True, help='CSV file with removal instructions')
    parser.add_argument('--output_dir', required=True, help='Output directory for cleaned files')
    args = parser.parse_args()
    
    
    print (f"\nClean tomostar from \"{args.input_dir}\" using excluded values from {args.csv}")
    print (f"Output folder {args.output_dir}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load removal instructions
    removal_map = load_removal_csv(args.csv)

    # Process all tomostar files in sorted order
    processed = 0
    for filename in sorted(os.listdir(args.input_dir)):
        if filename.endswith('.tomostar'):
            input_path = os.path.join(args.input_dir, filename)
            output_path = os.path.join(args.output_dir, filename)
            
            # Get removal instructions for this file
            tomo_name = os.path.splitext(filename)[0]
            zvalues_to_remove = removal_map.get(tomo_name, [])
            
            if zvalues_to_remove:
                print(f"Processing {filename} - removing indices (0-based): {zvalues_to_remove}")
                success = process_tomostar(input_path, output_path, zvalues_to_remove)
            else:
                # Just copy if no removals needed
                print(f"Copying {filename} (no removals)")
                shutil.copy2(input_path, output_path)
                success = True
            
            if success:
                processed += 1

    print(f"\nDone. Processed {processed} files.")

if __name__ == "__main__":
    main()