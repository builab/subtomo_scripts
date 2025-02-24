#!/usr/bin/python
# Script to generate star file by selected on a list of tomograms
# HB, McGill, 2025. Style coming from recenter_3d.py by Alister Burt

import argparse
import starfile
import pandas as pd


def select_by_list(input_star_file, list, output_star_file):
    star = starfile.read(input_star_file, always_dict=True)
    print(f"{input_star_file} read")
    
    if not all(key in star for key in ('particles', 'optics')):
        print("expected RELION 3.1+ style STAR file containing particles and optics blocks")

    df = star['particles']
    print(f"{len(df)} particles found")
    
    df['rlnTomoName_cleaned'] = df['rlnTomoName'].str.split('.').str[0]
    
    filtered_df = df[df['rlnTomoName_cleaned'].isin(filter_list)]
    filtered_df = filtered_df.drop(columns=['rlnTomoName_cleaned'])
    print(f"{len(filtered_df)} particles found after filtering")

    star['particles'] = filtered_df
    # write output
    starfile.write(star, output_star_file)
    print(f"Output with written to {output_star_file}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter a RELION STAR file based on a list of tomogram names.")
    
    # Add optional arguments with flags
    parser.add_argument("--i", "--input_star", required=True, help="Path to the input STAR file.")
    parser.add_argument("--l", "--input_list", required=True, help="Path to the list file containing tomogram names (one per line).")
    parser.add_argument("--o", "--output_star", required=True, help="Path to the output STAR file.")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Extract file paths from arguments
    input_star_file = args.i
    output_star_file = args.o  
    
    list_df = pd.read_csv(args.l, header=None)
    filter_list = list_df[0].tolist()
    
    select_by_list(args.i, filter_list, args.o)

