#!/usr/bin/env python3
# Script to convert star file to IMOD file for compatibility
"""
Tested
@Builab 2025
"""

import warnings
# Suppress the specific pydantic warning about protected namespaces
warnings.filterwarnings('ignore', module='pydantic')

import argparse
import starfile
import pandas as pd
import imodmodel


def convert_star_to_imod(star_path, output_mod, sort_y=False):
    # Load STAR file
    data = starfile.read(star_path)
    
    # Check for required columns
    required_columns = ['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ', 'rlnHelicalTubeID']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        print(f"❌ ERROR: Missing required columns: {', '.join(missing_columns)}")
        print(f"   Available columns: {', '.join(data.columns.tolist())}")
        sys.exit(1)

    # Create new dataframe with the required format
    df = pd.DataFrame({
        'object_id': 0,  # Default value 0 for all rows
        'contour_id': data['rlnHelicalTubeID'],
        'x': data['rlnCoordinateX'],
        'y': data['rlnCoordinateY'],
        'z': data['rlnCoordinateZ']
    })

    # Sort by contour_id (rlnHelicalTubeID)
    if sort_y:
        df = df.sort_values(['contour_id', 'y']).reset_index(drop=True)
    else:
        df = df.sort_values('contour_id').reset_index(drop=True)        
    
    # Save model
    imodmodel.write(df, output_mod)
    print(f"✅ IMOD model saved to {output_mod}")

def main():
    parser = argparse.ArgumentParser(description="Convert Relion helical STAR file to IMOD .mod model.")
    parser.add_argument("star_file", help="Input Relion STAR file")
    parser.add_argument("output_mod", help="Output IMOD model file (.mod)")
    parser.add_argument("--sortY", action="store_true", help="Sort points within each contour by Y-coordinate")
    args = parser.parse_args()

    convert_star_to_imod(args.star_file, args.output_mod, sort_y=args.sortY)

if __name__ == "__main__":
    main()
