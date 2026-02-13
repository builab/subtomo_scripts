#!/usr/bin/env python3
# Script to add column to Warp from a template star file
# This allows adding rlnHelicalTubeID, rlnOriginalIndex or rlnProtomerIndex

import argparse
import pandas as pd
import numpy as np
import starfile
from scipy.spatial import cKDTree
from pathlib import Path

def read_star_particles(star_data) -> pd.DataFrame:
    """Extracts the particles dataframe from the starfile dictionary."""
    if isinstance(star_data, dict) and 'particles' in star_data:
        return star_data['particles']
    return star_data

def get_pixel_size_from_optics(star_data, label="File") -> float:
    """Extracts pixel size from optics block and reports it."""
    px = 1.0
    if isinstance(star_data, dict) and 'optics' in star_data:
        optics = star_data['optics']
        if 'rlnImagePixelSize' in optics.columns:
            px = float(optics['rlnImagePixelSize'].iloc[0])
    print(f"Detected {label} Pixel Size: {px:.4f} Å")
    return px

def get_coordinates_in_angstrom(df: pd.DataFrame, pixel_size: float, use_origin: bool = False) -> np.ndarray:
    """
    Calculates coordinates in Angstrom.
    If use_origin is True: (Coord * px) - OriginAngst.
    """
    coords = df[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].values * pixel_size
    
    if use_origin:
        origin_cols = ['rlnOriginXAngst', 'rlnOriginYAngst', 'rlnOriginZAngst']
        if all(col in df.columns for col in origin_cols):
            origins = df[origin_cols].values
            coords = coords - origins
            print("Applying rlnOriginX/Y/ZAngst offsets for true coordinate comparison.")
        else:
            print("Warning: --compare_true_origin requested but origin columns missing. Using Coord * px.")
            
    return coords

def assign_random_subsets(df: pd.DataFrame) -> pd.Series:
    """Assigns 1 or 2 based on tube ID to avoid gold-standard bias."""
    random_subsets = pd.Series(index=df.index, dtype='int')
    tomo_names = sorted(df['rlnTomoName'].unique())
    
    for tomo_idx, tomo_name in enumerate(tomo_names):
        tomo_mask = df['rlnTomoName'] == tomo_name
        tube_ids = sorted(df.loc[tomo_mask, 'rlnHelicalTubeID'].unique())
        
        for tube_id in tube_ids:
            tube_mask = (df['rlnTomoName'] == tomo_name) & (df['rlnHelicalTubeID'] == tube_id)
            # Pattern alternates by tomo and tube to balance counts
            if tomo_idx % 2 == 0:
                subset_value = 1 if int(tube_id) % 2 == 1 else 2
            else:
                subset_value = 2 if int(tube_id) % 2 == 1 else 1
            random_subsets.loc[df.index[tube_mask]] = subset_value
    return random_subsets

def main():
    parser = argparse.ArgumentParser(description='Match particles with optional true origin comparison.')
    parser.add_argument('--input', required=True, help='Input warp STAR file')
    parser.add_argument('--template', required=True, help='Template STAR file')
    parser.add_argument('--columns', nargs='+', default=['rlnHelicalTubeID', 'rlnProtomerIndex', 'rlnOriginalIndex'],
                        help='Columns to copy from template')
    parser.add_argument('--compare_true_origin', action='store_true', 
                        help='Compare using (Coord * px) - OriginAngst instead of just (Coord * px)')
    
    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = input_path.parent / f"{input_path.stem}_matched{input_path.suffix}"
    
    # Load data
    warp_star_full = starfile.read(args.input)
    template_star_full = starfile.read(args.template)
    
    warp_df = read_star_particles(warp_star_full)
    template_df = read_star_particles(template_star_full)
    
    # Explicit Pixel Size Reporting
    w_px = get_pixel_size_from_optics(warp_star_full, "INPUT (Warp)")
    t_px = get_pixel_size_from_optics(template_star_full, "TEMPLATE")
    
    # Coordinate Calculation
    warp_coords = get_coordinates_in_angstrom(warp_df, w_px, args.compare_true_origin)
    template_coords = get_coordinates_in_angstrom(template_df, t_px, args.compare_true_origin)

    # Initialize columns
    existing_cols = [c for c in args.columns if c in template_df.columns]
    for col in existing_cols:
        warp_df[col] = np.nan

    # Process by tomogram
    for tomo_name in warp_df['rlnTomoName'].unique():
        warp_tomo_mask = warp_df['rlnTomoName'] == tomo_name
        temp_tomo_mask = template_df['rlnTomoName'] == tomo_name
        
        if not temp_tomo_mask.any():
            continue
            
        tree = cKDTree(template_coords[temp_tomo_mask])
        distances, nearest_indices = tree.query(warp_coords[warp_tomo_mask])
        
        matched_template_indices = template_df.index[temp_tomo_mask][nearest_indices]
        
        for col in existing_cols:
            warp_df.loc[warp_tomo_mask, col] = template_df.loc[matched_template_indices, col].values
            
        print(f"Matched {warp_tomo_mask.sum()} particles in {tomo_name} (Avg dist: {distances.mean():.4f} Å)")

    # Post-processing: Random Subsets
    if 'rlnHelicalTubeID' in warp_df.columns:
        warp_df['rlnRandomSubset'] = assign_random_subsets(warp_df)

    # Save output
    if isinstance(warp_star_full, dict):
        warp_star_full['particles'] = warp_df
        starfile.write(warp_star_full, output_path, overwrite=True)
    else:
        starfile.write(warp_df, output_path, overwrite=True)
    print(f"Done! Results saved to {output_path}")

if __name__ == '__main__':
    main()