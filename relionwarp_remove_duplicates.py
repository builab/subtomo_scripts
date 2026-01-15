# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas",
#     "scipy",
#     "starfile",
#     "typer",
#     "einops",
#     "rich",
#     "scikit-learn",
# ]
# [tool.uv]
# exclude-newer = "2025-01-01T00:00:00Z"
# ///

# Script by Huy Bui & DeepSeek & Claude
# Modified to remove duplicates only within the same rlnTomoName groups
# running with uv run relionwarp_remove_duplicates.py -i input.star -o output.star -d 45
# 2026/01/15 not resetshift anymore

from pathlib import Path

import numpy as np
import rich
import starfile
import typer
import pandas as pd
from sklearn.cluster import DBSCAN

console = rich.console.Console()


def remove_duplicates_in_group(group_df, original_group_df, min_distance_pixels, group_name):
    """Remove duplicates within a single tomogram group
    
    Args:
        group_df: DataFrame with merged optics for calculations
        original_group_df: Original DataFrame to preserve exact values
        min_distance_pixels: Minimum distance threshold in pixels
        group_name: Name of the tomogram group
    """
    if len(group_df) <= 1:
        return original_group_df
    
    # Get coordinates for this group
    xyz = group_df[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].to_numpy()
    
    # Get pixel spacing and shifts for this group
    pixel_spacing = group_df['rlnImagePixelSize'].to_numpy()
    
    if all(col in group_df.columns for col in ['rlnOriginXAngst', 'rlnOriginYAngst', 'rlnOriginZAngst']):
        shifts = group_df[['rlnOriginXAngst', 'rlnOriginYAngst', 'rlnOriginZAngst']].to_numpy()
    else:
        shifts = np.zeros(shape=(len(group_df), 3))
    
    # Apply shifts to calculate absolute particle positions
    pixel_spacing = pixel_spacing[:, np.newaxis]  # Shape: (n, 1)
    shifts_pixels = shifts / pixel_spacing
    xyz -= shifts_pixels
    
    # Perform clustering
    db = DBSCAN(eps=min_distance_pixels, min_samples=1).fit(xyz)
    
    # Get indices of first occurrence in each cluster
    unique_indices = []
    seen_clusters = set()
    for idx, cluster_id in enumerate(db.labels_):
        if cluster_id not in seen_clusters:
            unique_indices.append(idx)
            seen_clusters.add(cluster_id)
    
    # Select from ORIGINAL dataframe using these indices
    unique_df = original_group_df.iloc[unique_indices].reset_index(drop=True)
            
    console.log(f"  {group_name}: {len(group_df)} -> {len(unique_df)} particles")
    
    return unique_df

def cli(
    input_star_file: Path = typer.Option(..., '--input', '-i', help="input star file"),
    min_distance: float = typer.Option(..., '--min_d', '-d', help="min distance in Angstrom"),
    output_star_file: Path = typer.Option(..., '--output', '-o', help="output star file"),
):
    star = starfile.read(input_star_file, always_dict=True)
    console.log(f"{input_star_file} read")

    if not all(key in star for key in ('particles', 'optics')):
        console.log("expected RELION 3.1+ style STAR file containing particles and optics blocks", style="bold red")
        raise typer.Exit(1)

    # Keep original particles dataframe separate
    original_particles = star['particles'].copy()
    
    # Create merged dataframe for calculations only
    df = star['particles'].merge(star['optics'], on='rlnOpticsGroup')
    console.log("optics table merged")
    console.log(f"{len(df)} particles found")

    # Check if rlnTomoName column exists
    if 'rlnTomoName' not in df.columns:
        console.log("rlnTomoName column not found, treating all particles as one group", style="bold yellow")
        tomo_groups = [('all_particles', df)]
        original_groups = [('all_particles', original_particles)]
    else:
        # Group by tomogram name
        tomo_groups = list(df.groupby('rlnTomoName'))
        original_groups = list(original_particles.groupby('rlnTomoName'))
        console.log(f"found {len(tomo_groups)} tomogram groups")

    # Get pixel spacing for distance conversion
    angpix = df['rlnImagePixelSize'].iloc[0]
    min_distance_pixels = min_distance / angpix
    console.log(f"minimum distance: {min_distance} Å = {min_distance_pixels:.2f} pixels")

    # Process each tomogram group separately
    console.log("removing duplicates within each tomogram group...")
    processed_groups = []
    
    # Create dict for easy lookup of original groups
    original_groups_dict = {name: group for name, group in original_groups}
    
    for tomo_name, group_df in tomo_groups:
        original_group_df = original_groups_dict[tomo_name]
        unique_group = remove_duplicates_in_group(group_df, original_group_df, min_distance_pixels, tomo_name)
        processed_groups.append(unique_group)
    
    # Combine all processed groups - these are from original particles
    df_output = pd.concat(processed_groups, ignore_index=True)
    
    console.log(f"total particles after duplicate removal: {len(df_output)}")
    
    # Update the star file with original values
    star['particles'] = df_output
    
    # write output
    with console.status(f"writing output STAR file {output_star_file}", spinner="arc"):
        starfile.write(star, output_star_file)
    console.log(f"Output written to {output_star_file}")


if __name__ == "__main__":
    app = typer.Typer(add_completion=False)
    app.command(no_args_is_help=True)(cli)
    app()