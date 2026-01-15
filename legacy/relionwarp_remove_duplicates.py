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


from pathlib import Path

import numpy as np
import rich
import starfile
import typer
import pandas as pd
from sklearn.cluster import DBSCAN

console = rich.console.Console()


def remove_duplicates_in_group(group_df, min_distance_pixels, group_name):
    """Remove duplicates within a single tomogram group"""
    if len(group_df) <= 1:
        return group_df
    
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
    
    # Create a temporary dataframe with cluster labels
    temp_df = group_df.copy()
    temp_df['Cluster'] = db.labels_
    
    # Keep one representative point per cluster (first occurrence)
    unique_df = temp_df.groupby('Cluster').first().reset_index(drop=True)

    # Remove cluster column if it exists
    if 'Cluster' in unique_df.columns:
        unique_df = unique_df.drop(columns=['Cluster'])
            
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

    df = star['particles'].merge(star['optics'], on='rlnOpticsGroup')
    console.log("optics table merged")
    console.log(f"{len(df)} particles found")

    # Check if rlnTomoName column exists
    if 'rlnTomoName' not in df.columns:
        console.log("rlnTomoName column not found, treating all particles as one group", style="bold yellow")
        tomo_groups = [('all_particles', df)]
    else:
        # Group by tomogram name
        tomo_groups = list(df.groupby('rlnTomoName'))
        console.log(f"found {len(tomo_groups)} tomogram groups")

    # Get pixel spacing for distance conversion
    angpix = df['rlnImagePixelSize'].iloc[0]
    min_distance_pixels = min_distance / angpix
    console.log(f"minimum distance: {min_distance} Å = {min_distance_pixels:.2f} pixels")

    # Process each tomogram group separately
    console.log("removing duplicates within each tomogram group...")
    processed_groups = []
    
    for tomo_name, group_df in tomo_groups:
        unique_group = remove_duplicates_in_group(group_df, min_distance_pixels, tomo_name)
        processed_groups.append(unique_group)
    
    # Combine all processed groups
    df_combined = pd.concat(processed_groups, ignore_index=True)
    
    # Get the original particle columns (without the merged optics columns)
    original_columns = star['particles'].columns
    df_output = df_combined[original_columns]
    
    console.log(f"total particles after duplicate removal: {len(df_output)}")
    
    # Update the star file
    star['particles'] = df_output
    
    # write output
    with console.status(f"writing output STAR file {output_star_file}", spinner="arc"):
        starfile.write(star, output_star_file)
    console.log(f"Output written to {output_star_file}")


if __name__ == "__main__":
    app = typer.Typer(add_completion=False)
    app.command(no_args_is_help=True)(cli)
    app()