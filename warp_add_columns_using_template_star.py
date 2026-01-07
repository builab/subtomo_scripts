#!/usr/bin/env python3
# Original add_id_to_warpstar.py
# Now expanded to more columns for symmetry expansion
# Haven't tested thoroughly yet

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

def get_pixel_size_from_optics(star_data) -> float:
    """Extracts pixel size from the optics block if it exists."""
    if isinstance(star_data, dict) and 'optics' in star_data:
        optics = star_data['optics']
        if 'rlnImagePixelSize' in optics.columns:
            return optics['rlnImagePixelSize'].iloc[0]
    return None

def get_coordinates_in_angstrom(df: pd.DataFrame, pixel_size: float = None) -> np.ndarray:
    """Converts pixel coordinates to Angstroms using the provided pixel size."""
    coords = df[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].values
    if pixel_size is not None:
        coords = coords * pixel_size
    return coords

def assign_random_subsets(df: pd.DataFrame) -> pd.Series:
    """Assigns 1 or 2 based on tube ID, alternating by tomogram to balance sets."""
    random_subsets = pd.Series(index=df.index, dtype='int')
    tomo_names = sorted(df['rlnTomoName'].unique())
    
    for tomo_idx, tomo_name in enumerate(tomo_names):
        tomo_mask = df['rlnTomoName'] == tomo_name
        # Get unique tube IDs for this tomogram
        tube_ids = sorted(df.loc[tomo_mask, 'rlnHelicalTubeID'].unique())
        
        for tube_id in tube_ids:
            tube_mask = (df['rlnTomoName'] == tomo_name) & (df['rlnHelicalTubeID'] == tube_id)
            # Pattern alternates by tomo index to balance half-sets
            if tomo_idx % 2 == 0:
                subset_value = 1 if int(tube_id) % 2 == 1 else 2
            else:
                subset_value = 2 if int(tube_id) % 2 == 1 else 1
            
            random_subsets.loc[df.index[tube_mask]] = subset_value
    return random_subsets

def main():
    parser = argparse.ArgumentParser(description='Match particles and copy metadata columns using optics-aware pixel sizes.')
    parser.add_argument('--input', required=True, help='Input warp STAR file')
    parser.add_argument('--template', required=True, help='Template STAR file')
    parser.add_argument('--columns', nargs='+', default=['rlnHelicalTubeID', 'rlnProtomerIndex', 'rlnOriginalIndex'],
                        help='Columns to copy from template')
    
    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = input_path.parent / f"{input_path.stem}_matched{input_path.suffix}"
    
    # Load full STAR structures
    warp_star_full = starfile.read(args.input)
    template_star_full = starfile.read(args.template)
    
    # Extract dataframes and pixel sizes from optics
    warp_df = read_star_particles(warp_star_full)
    template_df = read_star_particles(template_star_full)
    
    w_px = get_pixel_size_from_optics(warp_star_full)
    t_px = get_pixel_size_from_optics(template_star_full)
    
    print(f"Input pixel size: {w_px} A | Template pixel size: {t_px} A")

    # Initialize columns in warp_df
    existing_cols = [c for c in args.columns if c in template_df.columns]
    for col in existing_cols:
        warp_df[col] = np.nan

    # Convert coordinates to Angstrom for both
    warp_coords_ang = get_coordinates_in_angstrom(warp_df, w_px)
    template_coords_ang = get_coordinates_in_angstrom(template_df, t_px)

    # Process by tomogram
    for tomo_name in warp_df['rlnTomoName'].unique():
        warp_tomo_mask = warp_df['rlnTomoName'] == tomo_name
        temp_tomo_mask = template_df['rlnTomoName'] == tomo_name
        
        if not temp_tomo_mask.any():
            print(f"Skipping {tomo_name}: No particles in template.")
            continue
            
        # Build KD-Tree on template Angstrom coordinates
        tree = cKDTree(template_coords_ang[temp_tomo_mask])
        distances, nearest_indices = tree.query(warp_coords_ang[warp_tomo_mask])
        
        # Map back to original template dataframe indices
        matched_template_indices = template_df.index[temp_tomo_mask][nearest_indices]
        
        for col in existing_cols:
            warp_df.loc[warp_tomo_mask, col] = template_df.loc[matched_template_indices, col].values
            
        print(f"Matched {warp_tomo_mask.sum()} particles in {tomo_name} (Avg dist: {distances.mean():.4f} A)")

    # Post-processing: Random Subsets
    if 'rlnHelicalTubeID' in warp_df.columns:
        warp_df['rlnRandomSubset'] = assign_random_subsets(warp_df)

    # Save output preserving the input structure
    if isinstance(warp_star_full, dict):
        warp_star_full['particles'] = warp_df
        starfile.write(warp_star_full, output_path, overwrite=True)
    else:
        starfile.write(warp_df, output_path, overwrite=True)
    print(f"Done! Saved to {output_path}")

if __name__ == '__main__':
    main()