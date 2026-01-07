#!/usr/bin/env python3
# Original add_id_to_warpstar.py
# Now expanded to more columns for symmetry expansion

import argparse
import pandas as pd
import numpy as np
import starfile
from scipy.spatial import cKDTree
from pathlib import Path
from typing import List, Optional

def read_star(file_path: str) -> pd.DataFrame:
    df = starfile.read(file_path)
    if isinstance(df, dict) and 'particles' in df:
        return df['particles']
    return df

def get_coordinates_in_angstrom(df: pd.DataFrame, pixel_size: float = None) -> np.ndarray:
    coords = df[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].values
    if pixel_size is not None:
        coords = coords * pixel_size
    return coords

def get_pixel_size_from_optics(star_data) -> float:
    if isinstance(star_data, dict) and 'optics' in star_data:
        optics = star_data['optics']
        if 'rlnImagePixelSize' in optics.columns:
            return optics['rlnImagePixelSize'].iloc[0]
    return None

def match_and_copy_columns(warp_df: pd.DataFrame, template_df: pd.DataFrame,
                             warp_coords: np.ndarray, template_coords: np.ndarray,
                             columns_to_copy: List[str]) -> pd.DataFrame:
    """
    Finds nearest neighbors and copies specified columns from template to warp.
    """
    # Initialize new columns in warp_df
    for col in columns_to_copy:
        warp_df[col] = None

    for tomo_name in warp_df['rlnTomoName'].unique():
        warp_mask = warp_df['rlnTomoName'] == tomo_name
        template_mask = template_df['rlnTomoName'] == tomo_name
        
        if not template_mask.any():
            print(f"Warning: No template particles for tomogram {tomo_name}")
            continue
        
        # KD-Tree for nearest neighbor
        tree = cKDTree(template_coords[template_mask])
        distances, nearest_indices = tree.query(warp_coords[warp_mask])
        
        # Get the actual indices in the template_df
        actual_template_indices = template_df.index[template_mask][nearest_indices]
        
        # Copy values
        for col in columns_to_copy:
            warp_df.loc[warp_mask, col] = template_df.loc[actual_template_indices, col].values
        
        print(f"Matched {warp_mask.sum()} particles in {tomo_name}")
    
    return warp_df

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
            subset_value = (1 if int(tube_id) % 2 == 1 else 2) if tomo_idx % 2 == 0 else (2 if int(tube_id) % 2 == 1 else 1)
            random_subsets.loc[tube_indices := df.index[tube_mask]] = subset_value
    return random_subsets

def main():
    parser = argparse.ArgumentParser(description='Copy multiple tags from template to warp STAR file.')
    parser.add_argument('--input', required=True, help='Input warp STAR file')
    parser.add_argument('--template', required=True, help='Template STAR file')
    parser.add_argument('--columns', nargs='+', default=['rlnHelicalTubeID', 'rlnProtomerIndex', 'rlnOriginalIndex'],
                        help='Columns to copy from template if they exist')
    
    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = input_path.parent / f"{input_path.stem}_expanded{input_path.suffix}"
    
    warp_star_data = starfile.read(args.input)
    warp_df = read_star(args.input)
    template_df = read_star(args.template)
    
    # Filter columns that actually exist in the template
    existing_cols = [c for c in args.columns if c in template_df.columns]
    print(f"Copying existing columns: {existing_cols}")

    # Coordinate setup
    w_px = get_pixel_size_from_optics(warp_star_data)
    warp_coords = get_coordinates_in_angstrom(warp_df, w_px)
    
    t_px = template_df['rlnImagePixelSize'].iloc[0] if 'rlnImagePixelSize' in template_df.columns else None
    template_coords = get_coordinates_in_angstrom(template_df, t_px)

    # Core matching
    warp_df = match_and_copy_columns(warp_df, template_df, warp_coords, template_coords, existing_cols)

    # Logic for RandomSubset
    if 'rlnHelicalTubeID' in existing_cols:
        print("Assigning rlnRandomSubset based on rlnHelicalTubeID...")
        warp_df['rlnRandomSubset'] = assign_random_subsets(warp_df)

    # Save output
    if isinstance(warp_star_data, dict):
        warp_star_data['particles'] = warp_df
        starfile.write(warp_star_data, output_path, overwrite=True)
    else:
        starfile.write(warp_df, output_path, overwrite=True)
    print(f"Done! Saved to {output_path}")

if __name__ == '__main__':
    main()