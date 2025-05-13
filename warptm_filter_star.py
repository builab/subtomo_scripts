#!/usr/bin/env python
# Script to use filter warptm star as the native WarpTool has a bug.
# Written by DeepSeek
# HB, McGill 2025/05


import argparse
import glob
import os
import starfile
import pandas as pd
from tqdm import tqdm  # For progress bar (install with: pip install tqdm)


def filter_star_file(input_star_path, min_score):
    """Filter a single STAR file by autopick score"""
    try:
        # Read STAR file
        star = starfile.read(input_star_path, always_dict=True)
        
        # Handle legacy STAR format (empty key)
        if '' in star:
            star['particles'] = star.pop('')
        
        # Verify required columns exist
        if 'particles' not in star:
            raise ValueError("No particles table found")
        if 'rlnAutopickFigureOfMerit' not in star['particles']:
            raise ValueError("No rlnAutopickFigureOfMerit column found")
        
        # Filter particles
        particles = star['particles']
        filtered = particles[particles['rlnAutopickFigureOfMerit'] >= min_score]
        
        # Create output filename
        base_name = os.path.splitext(input_star_path)[0]
        output_path = f"{base_name}_clean.star"
        
        # Save filtered STAR file
        star['particles'] = filtered
        starfile.write(star, output_path)
        
        return len(particles), len(filtered)
    
    except Exception as e:
        print(f"Error processing {input_star_path}: {str(e)}")
        return 0, 0

def main():
    parser = argparse.ArgumentParser(
        description='Filter Warp TM STAR files by autopick figure of merit',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_pattern', required=True,
                       help='Pattern to match STAR files (e.g., "*MT13.star")')
    parser.add_argument('--min_score', type=float, default=3.0,
                       help='Minimum autopick figure of merit to keep')
    
    args = parser.parse_args()
    
    # Find all matching files
    star_files = glob.glob(args.input_pattern)
    if not star_files:
        print(f"No files found matching pattern: {args.input_pattern}")
        return
    
    print(f"Found {len(star_files)} STAR files to process")
    print(f"Keeping particles with rlnAutopickFigureOfMerit >= {args.min_score}")
    
    # Process files with progress bar
    total_initial = 0
    total_filtered = 0
    for star_file in tqdm(star_files, desc="Processing files"):
        initial, filtered = filter_star_file(star_file, args.min_score)
        total_initial += initial
        total_filtered += filtered
    
    # Print summary
    print(f"\nProcessing complete:")
    print(f"  Total particles before filtering: {total_initial}")
    print(f"  Total particles after filtering: {total_filtered}")
    print(f"  Percentage kept: {100*total_filtered/max(1,total_initial):.1f}%")

if __name__ == "__main__":
    main()