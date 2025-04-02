#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# require mdocfile from https://pypi.org/project/mdocfile
# pip install mdocfile
# Might not write all entries due to limitation of mdocfile
# 2025/04/01, Huy Bui, McGill

import sys
import os
import argparse
import mdocfile
import pandas as pd
import numpy as np
from mdocfile.data_models import Mdoc, MdocGlobalData, MdocSectionData

def safe_get(df, column, index=0, default=None):
    """Safely get a value from DataFrame with fallback to default."""
    try:
        val = df[column].iloc[index]
        if isinstance(val, (list, np.ndarray)):
            return val if len(val) > 0 else default
        return val if not pd.isna(val) else default
    except (KeyError, IndexError):
        return default

def process_mdoc_file(input_path, output_dir):
    """Process a single mdoc file and save sorted version to output directory"""
    try:
        df = mdocfile.read(input_path)
    except Exception as e:
        print(f"Error reading {input_path}: {str(e)}")
        return False

    # Sort by TiltAngle and reset ZValue to maintain order
    df2 = df.sort_values('TiltAngle', ignore_index=True)
    df2['ZValue'] = df2.index

    # Prepare global data with safe field access
    global_data = MdocGlobalData(
        DataMode=safe_get(df2, 'DataMode', default=1),
        ImageSize=safe_get(df2, 'ImageSize', default=(5760, 4092)),
        PixelSpacing=safe_get(df2, 'PixelSpacing', default=1.66),
        Voltage=safe_get(df2, 'Voltage', default=300)
    )

    # Handle titles separately since it might be an array
    titles = safe_get(df2, 'titles', default=[""])
    if isinstance(titles, (list, np.ndarray)) and len(titles) == 0:
        titles = [""]

    # Get all available section columns
    section_columns = [col for col in df2.columns 
                      if col not in ['DataMode', 'ImageSize', 'PixelSpacing', 'Voltage', 'titles']]

    # Create Mdoc object with all section data
    mdoc = Mdoc(
        titles=titles,
        global_data=global_data,
        section_data=[]
    )

    for ind in df2.index:
        section_dict = {}
        for col in section_columns:
            val = safe_get(df2, col, ind)
            if val is not None:
                if isinstance(val, (list, np.ndarray)):
                    if len(val) > 0:
                        section_dict[col] = val
                else:
                    section_dict[col] = val
        
        try:
            current_section = MdocSectionData(**section_dict)
            mdoc.section_data.append(current_section)
        except TypeError as e:
            print(f"Warning: Couldn't create section in {input_path} index {ind}: {str(e)}")
            continue

    # Create output path with same filename in output directory
    output_path = os.path.join(output_dir, os.path.basename(input_path))
    
    try:
        with open(output_path, mode='w+') as file:
            file.write(mdoc.to_string())
        print(f"Successfully wrote: {output_path}")
        return True
    except IOError as e:
        print(f"Error writing {output_path}: {str(e)}")
        return False

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Sort mdoc files by tilt angle')
    parser.add_argument('--i', '--input', dest='input_dir', required=True,
                       help='Input directory containing mdoc files')
    parser.add_argument('--o', '--output', dest='output_dir', required=True,
                       help='Output directory for sorted mdoc files')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Process all .mdoc files in input directory
    processed_count = 0
    for filename in os.listdir(args.input_dir):
        if filename.lower().endswith('.mdoc'):
            input_path = os.path.join(args.input_dir, filename)
            if process_mdoc_file(input_path, args.output_dir):
                processed_count += 1

    print(f"\nSorting complete. Processed {processed_count} mdoc files.")

if __name__ == "__main__":
    main()
