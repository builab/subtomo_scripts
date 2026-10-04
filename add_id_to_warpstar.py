#!/usr/bin/env python
# coding: utf-8

"""
Add rlnHelicalTubeID to star file exported using Warp using original file.
1. Add rlnHelicalTubeID based on the nearest points from original file.
2. Also add rlnRandomSubset grouped based on rlnHelicalTubeID to avoid inflated resolution.
3. Add rlnOriginalIndex reflecting the position along the tube (sorted by Z coordinate).

Tested good.
@Builab 2025
"""


import argparse
import pandas as pd
import numpy as np
import starfile
from scipy.spatial import cKDTree
from pathlib import Path


def read_star(file_path: str) -> pd.DataFrame:
    """
    Read STAR file into DataFrame.
    Take care of name
    
    Args:
        file_path: Path to STAR file.
    
    Returns:
        DataFrame containing STAR file data.
    """
    df = starfile.read(file_path)
    if 'rlnCoordinateX' not in df and 'particles' in df:
        df = df['particles']
    
    return df


def get_coordinates_in_angstrom(df: pd.DataFrame, pixel_size: float = None) -> np.ndarray:
    """
    Get particle coordinates in Angstrom.
    
    Args:
        df: DataFrame with particle data
        pixel_size: Pixel size in Angstrom (if None, coordinates assumed already in Angstrom)
    
    Returns:
        Nx3 array of coordinates in Angstrom
    """
    coords = df[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].values
    
    if pixel_size is not None:
        coords = coords * pixel_size
    
    return coords


def get_pixel_size_from_optics(star_data) -> float:
    """
    Extract pixel size from optics block if it exists.
    
    Args:
        star_data: Output from starfile.read()
    
    Returns:
        Pixel size in Angstrom, or None if not found
    """
    if isinstance(star_data, dict) and 'optics' in star_data:
        optics = star_data['optics']
        if 'rlnImagePixelSize' in optics.columns:
            # Assume single optics group or use first
            return optics['rlnImagePixelSize'].iloc[0]
    return None


def create_template_position_map(template_df: pd.DataFrame) -> dict:
    """
    Create a mapping from (tomoName, tubeID, particle_index) to position index along tube.
    Assumes particles are already sorted by Y coordinate within each tube in the template file.
    
    Args:
        template_df: DataFrame with template particles (pre-sorted by Y coordinate)
    
    Returns:
        Dictionary mapping (tomoName, tubeID, template_df_index) -> position_index (1-based)
    """
    position_map = {}
    
    # Group by tomogram and tube
    for (tomo_name, tube_id), group in template_df.groupby(['rlnTomoName', 'rlnHelicalTubeID']):
        # Use existing order (already sorted by Y coordinate)
        # Assign position indices (1-based)
        for position_idx, original_idx in enumerate(group.index, start=1):
            position_map[(tomo_name, tube_id, original_idx)] = position_idx
    
    return position_map


def match_particles_by_tomogram(warp_df: pd.DataFrame, template_df: pd.DataFrame,
                                  warp_coords: np.ndarray, template_coords: np.ndarray,
                                  position_map: dict) -> tuple:
    """
    For each particle in warp_df, find the nearest particle in template_df
    within the same tomogram and return the corresponding rlnHelicalTubeID and rlnOriginalIndex.
    
    Args:
        warp_df: DataFrame with warp particles
        template_df: DataFrame with template particles
        warp_coords: Coordinates of warp particles in Angstrom (Nx3)
        template_coords: Coordinates of template particles in Angstrom (Mx3)
        position_map: Dictionary mapping (tomoName, tubeID, template_idx) -> position_index
    
    Returns:
        Tuple of (Series of rlnHelicalTubeID, Series of rlnOriginalIndex) for each warp particle
    """
    helical_tube_ids = pd.Series(index=warp_df.index, dtype='Int64')
    original_indices = pd.Series(index=warp_df.index, dtype='Int64')
    
    # Group by tomogram name
    for tomo_name in warp_df['rlnTomoName'].unique():
        # Get indices for this tomogram
        warp_mask = warp_df['rlnTomoName'] == tomo_name
        template_mask = template_df['rlnTomoName'] == tomo_name
        
        warp_indices = warp_df.index[warp_mask]
        template_indices = template_df.index[template_mask]
        
        if len(template_indices) == 0:
            print(f"Warning: No template particles found for tomogram {tomo_name}")
            continue
        
        # Get coordinates for this tomogram
        warp_tomo_coords = warp_coords[warp_mask]
        template_tomo_coords = template_coords[template_mask]
        
        # Build KD-tree for efficient nearest neighbor search
        tree = cKDTree(template_tomo_coords)
        
        # Find nearest neighbor for each warp particle
        distances, nearest_indices = tree.query(warp_tomo_coords)
        
        # Map back to original template indices and get tube IDs and position indices
        template_orig_indices = template_indices[nearest_indices]
        tube_ids = template_df.loc[template_orig_indices, 'rlnHelicalTubeID'].values
        
        # Get position indices from the map
        pos_indices = np.array([
            position_map.get((tomo_name, tube_id, template_idx), None)
            for tube_id, template_idx in zip(tube_ids, template_orig_indices)
        ])
        
        # Assign to output series
        helical_tube_ids.loc[warp_indices] = tube_ids
        original_indices.loc[warp_indices] = pos_indices
        
        print(f"Matched {len(warp_indices)} particles in tomogram {tomo_name} "
              f"(mean distance: {distances.mean():.2f} Å)")
    
    return helical_tube_ids, original_indices


def assign_random_subsets(df: pd.DataFrame) -> pd.Series:
    """
    Assign rlnRandomSubset values (1 or 2) to particles.
    Particles with the same rlnTomoName and rlnHelicalTubeID get the same subset.
    Assignment alternates by rlnHelicalTubeID within each tomogram, and the pattern
    alternates between tomograms to balance particle counts.
    
    Args:
        df: DataFrame with particles containing rlnTomoName and rlnHelicalTubeID
    
    Returns:
        Series of rlnRandomSubset values (1 or 2) for each particle
    """
    random_subsets = pd.Series(index=df.index, dtype='int')
    
    # Process each tomogram separately
    tomo_names = sorted(df['rlnTomoName'].unique())
    
    for tomo_idx, tomo_name in enumerate(tomo_names):
        tomo_mask = df['rlnTomoName'] == tomo_name
        tomo_df = df[tomo_mask]
        
        # Get unique tube IDs for this tomogram, sorted
        tube_ids = sorted(tomo_df['rlnHelicalTubeID'].unique())
        
        # Alternate pattern between tomograms
        # Even tomogram index: odd tubes→1, even tubes→2
        # Odd tomogram index: odd tubes→2, even tubes→1
        for tube_id in tube_ids:
            tube_mask = (df['rlnTomoName'] == tomo_name) & (df['rlnHelicalTubeID'] == tube_id)
            tube_indices = df.index[tube_mask]
            
            if tomo_idx % 2 == 0:
                # Even tomogram: odd tubes→1, even tubes→2
                subset_value = 1 if int(tube_id) % 2 == 1 else 2
            else:
                # Odd tomogram: odd tubes→2, even tubes→1 (flip the pattern)
                subset_value = 2 if int(tube_id) % 2 == 1 else 1
            
            random_subsets.loc[tube_indices] = subset_value
    
    # Print statistics
    n_subset_1 = (random_subsets == 1).sum()
    n_subset_2 = (random_subsets == 2).sum()
    
    # Count groups per subset
    grouped = df.groupby(['rlnTomoName', 'rlnHelicalTubeID'])
    n_groups = len(grouped)
    n_groups_1 = sum(1 for _, group in grouped if random_subsets.loc[group.index[0]] == 1)
    n_groups_2 = n_groups - n_groups_1
    
    print(f"\nRandom subset assignment (alternating by tubeID, pattern flips per tomogram):")
    print(f"  Total unique (tomo, tubeID) groups: {n_groups}")
    print(f"  Groups in subset 1: {n_groups_1} ({n_subset_1} particles)")
    print(f"  Groups in subset 2: {n_groups_2} ({n_subset_2} particles)")
    print(f"  Difference: {abs(n_subset_1 - n_subset_2)} particles ({abs(n_subset_1 - n_subset_2) / len(df) * 100:.1f}%)")
    
    return random_subsets


def main():
    parser = argparse.ArgumentParser(
        description='Copy rlnHelicalTubeID from template to warp STAR file by matching nearest particles. '
                    'Automatically adds rlnRandomSubset for balanced half-set assignment and '
                    'rlnOriginalIndex reflecting position along tube (from pre-sorted template).'
    )
    parser.add_argument('--input', required=True,
                        help='Input warp STAR file')
    parser.add_argument('--template', required=True,
                        help='Template STAR file with rlnHelicalTubeID')
    
    args = parser.parse_args()
    
    # Generate output filename
    input_path = Path(args.input)
    output_path = input_path.parent / f"{input_path.stem}_with_id{input_path.suffix}"
    
    print(f"Reading warp STAR file: {args.input}")
    warp_star_data = starfile.read(args.input)
    warp_df = read_star(args.input)
    
    print(f"Reading template STAR file: {args.template}")
    template_df = read_star(args.template)
    
    # Check required columns
    required_cols = ['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ', 'rlnTomoName']
    for col in required_cols:
        if col not in warp_df.columns:
            raise ValueError(f"Missing column {col} in warp file")
        if col not in template_df.columns:
            raise ValueError(f"Missing column {col} in template file")
    
    if 'rlnHelicalTubeID' not in template_df.columns:
        raise ValueError("Missing rlnHelicalTubeID in template file")
    
    # Create position map from template (sorted by Y coordinate)
    print("\nCreating position map from template (assumes pre-sorted by Y coordinate)...")
    position_map = create_template_position_map(template_df)
    
    # Get pixel sizes and compute coordinates in Angstrom
    warp_pixel_size = get_pixel_size_from_optics(warp_star_data)
    print(f"Warp pixel size from optics: {warp_pixel_size} Å")
    warp_coords = get_coordinates_in_angstrom(warp_df, warp_pixel_size)
    
    # Template coordinates are already in Angstrom (pixels * pixel size in column)
    # Check if template has rlnImagePixelSize column
    if 'rlnImagePixelSize' in template_df.columns:
        template_pixel_size = template_df['rlnImagePixelSize'].iloc[0]
        print(f"Template pixel size from particle data: {template_pixel_size} Å")
        template_coords = get_coordinates_in_angstrom(template_df, template_pixel_size)
    else:
        print("Assuming template coordinates are already in Angstrom")
        template_coords = get_coordinates_in_angstrom(template_df, None)
    
    print(f"\nMatching {len(warp_df)} warp particles to {len(template_df)} template particles...")
    
    # Match particles and get tube IDs and position indices
    helical_tube_ids, original_indices = match_particles_by_tomogram(
        warp_df, template_df, warp_coords, template_coords, position_map
    )
    
    # Add rlnHelicalTubeID and rlnOriginalIndex to warp dataframe
    warp_df['rlnHelicalTubeID'] = helical_tube_ids
    warp_df['rlnOriginalIndex'] = original_indices
    
    # Check for any unmatched particles
    unmatched = helical_tube_ids.isna().sum()
    if unmatched > 0:
        print(f"\nWarning: {unmatched} particles could not be matched")
    
    # Assign random subsets (now default)
    print("\nAssigning random subsets...")
    random_subsets = assign_random_subsets(warp_df)
    warp_df['rlnRandomSubset'] = random_subsets
    
    # Write output
    print(f"\nWriting output to: {output_path}")
    if isinstance(warp_star_data, dict):
        # Preserve structure with optics block
        warp_star_data['particles'] = warp_df
        starfile.write(warp_star_data, output_path, overwrite=True)
    else:
        # Simple dataframe
        starfile.write(warp_df, output_path, overwrite=True)
    
    print("Done!")


if __name__ == '__main__':
    main()