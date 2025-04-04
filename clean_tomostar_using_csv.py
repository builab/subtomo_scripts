#!/usr/bin/env python
"""
Script to remove entry in tomostar file using a csv file
csv file format
filename,zvalues_to_remove
File_001.tomostar,"0,1,40"
File_002.tomostar,""
File_003.tomostar,"0,1,2"

Requires starfile package (pip install starfile)

# NOT Tested YEt
"""

import os
import csv
import argparse
import shutil
import starfile

def load_removal_csv(csv_path):
    """Load removal instructions from CSV"""
    removal_map = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row['filename']
            zvalues = [int(z) for z in row['zvalues_to_remove'].split(',')] if row['zvalues_to_remove'] else []
            removal_map[filename] = zvalues
    return removal_map

def process_tomostar(input_path, output_path, zvalues_to_remove):
    """Process a tomostar file, removing specified indices"""
    try:
        # Read star file
        df = starfile.read(input_path)
        
        # Remove specified indices (0-based)
        if zvalues_to_remove:
            # Convert to 0-based if needed (depends on your CSV format)
            # Here assuming CSV uses 0-based indices
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

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load removal instructions
    removal_map = load_removal_csv(args.csv)

    # Process all tomostar files
    processed = 0
    for filename in os.listdir(args.input_dir):
        if filename.endswith('.tomostar'):
            input_path = os.path.join(args.input_dir, filename)
            output_path = os.path.join(args.output_dir, filename)
            
            # Get removal instructions for this file
            zvalues_to_remove = removal_map.get(filename, [])
            
            if zvalues_to_remove:
                print(f"Processing {filename} - removing indices: {zvalues_to_remove}")
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