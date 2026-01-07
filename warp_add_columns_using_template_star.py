import argparse
import pandas as pd
import numpy as np
import starfile
from scipy.spatial import cKDTree
from pathlib import Path
from typing import List

def read_star(file_path: str) -> pd.DataFrame:
    """Reads STAR file and returns the particles dataframe."""
    df = starfile.read(file_path)
    if isinstance(df, dict) and 'particles' in df:
        return df['particles']
    return df

def get_coordinates(df: pd.DataFrame, pixel_size: float = None) -> np.ndarray:
    """Gets coordinates, optionally converting to Angstroms."""
    coords = df[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].values
    if pixel_size is not None:
        coords = coords * pixel_size
    return coords

def get_pixel_size(star_data) -> float:
    """Extracts pixel size from the optics block."""
    if isinstance(star_data, dict) and 'optics' in star_data:
        optics = star_data['optics']
        if 'rlnImagePixelSize' in optics.columns:
            return optics['rlnImagePixelSize'].iloc[0]
    return None

def assign_random_subsets(df: pd.DataFrame) -> pd.Series:
    """Assigns 1 or 2 based on tube ID to avoid gold-standard bias."""
    random_subsets = pd.Series(index=df.index, dtype='int')
    tomo_names = sorted(df['rlnTomoName'].unique())
    
    for tomo_idx, tomo_name in enumerate(tomo_names):
        tomo_mask = df['rlnTomoName'] == tomo_name
        # Use .loc to avoid SettingWithCopy warnings
        tomo_df = df.loc[tomo_mask]
        
        if 'rlnHelicalTubeID' not in tomo_df.columns:
            continue
            
        tube_ids = sorted(tomo_df['rlnHelicalTubeID'].unique())
        
        for tube_id in tube_ids:
            tube_mask = (df['rlnTomoName'] == tomo_name) & (df['rlnHelicalTubeID'] == tube_id)
            tube_indices = df.index[tube_mask]
            
            # Alternate pattern by tomogram to balance sets
            if tomo_idx % 2 == 0:
                subset_value = 1 if int(tube_id) % 2 == 1 else 2
            else:
                subset_value = 2 if int(tube_id) % 2 == 1 else 1
            
            random_subsets.loc[tube_indices] = subset_value
    return random_subsets

def main():
    parser = argparse.ArgumentParser(description='Match particles and copy metadata columns.')
    parser.add_argument('--input', required=True, help='Warp STAR file to update')
    parser.add_argument('--template', required=True, help='Template STAR file with metadata')
    parser.add_argument('--columns', nargs='+', default=['rlnHelicalTubeID', 'rlnProtomerIndex', 'rlnOriginalIndex'],
                        help='Columns to copy from template')
    
    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = input_path.parent / f"{input_path.stem}_matched{input_path.suffix}"
    
    # Load data
    warp_data = starfile.read(args.input)
    warp_df = read_star(args.input)
    template_df = read_star(args.template)
    
    # Determine which requested columns actually exist in template
    cols_to_copy = [c for c in args.columns if c in template_df.columns]
    print(f"Columns to be copied: {cols_to_copy}")

    # Initialize columns in warp_df with appropriate types
    for col in cols_to_copy:
        warp_df[col] = np.nan

    # Coordinate setup
    w_px = get_pixel_size(warp_data)
    warp_coords_all = get_coordinates(warp_df, w_px)
    
    # Template pixel size (often 1.0 if already in Angstroms)
    t_px = template_df['rlnImagePixelSize'].iloc[0] if 'rlnImagePixelSize' in template_df.columns else None
    template_coords_all = get_coordinates(template_df, t_px)

    # Process by tomogram to ensure correct matching
    for tomo_name in warp_df['rlnTomoName'].unique():
        warp_mask = warp_df['rlnTomoName'] == tomo_name
        temp_mask = template_df['rlnTomoName'] == tomo_name
        
        if not temp_mask.any():
            print(f"Skipping {tomo_name}: No particles in template.")
            continue
            
        # Build tree for template points in THIS tomogram
        tree = cKDTree(template_coords_all[temp_mask])
        
        # Query tree with warp points in THIS tomogram
        distances, nearest_indices = tree.query(warp_coords_all[warp_mask])
        
        # Map the tree result back to the actual indices of the template dataframe
        template_indices_in_tomo = template_df.index[temp_mask][nearest_indices]
        
        # Copy the values row by row for the matched indices
        for col in cols_to_copy:
            warp_df.loc[warp_mask, col] = template_df.loc[template_indices_in_tomo, col].values
            
        print(f"Matched {warp_mask.sum()} particles in {tomo_name} (Avg dist: {distances.mean():.2f} A)")

    # Post-processing: Random Subsets
    if 'rlnHelicalTubeID' in warp_df.columns:
        print("Assigning rlnRandomSubset...")
        warp_df['rlnRandomSubset'] = assign_random_subsets(warp_df)

    # Save
    if isinstance(warp_data, dict):
        warp_data['particles'] = warp_df
        starfile.write(warp_data, output_path, overwrite=True)
    else:
        starfile.write(warp_df, output_path, overwrite=True)
    print(f"Successfully saved to {output_path}")

if __name__ == '__main__':
    main()