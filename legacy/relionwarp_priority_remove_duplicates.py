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

# Script by Huy Bui & DeepSeek
# Modified for priority-based duplicate removal between two files
# running with uv run relionwarp_remove_duplicates_priority.py -i starfile1.star -i2 starfile2.star -o output.star -d 30

from pathlib import Path

import numpy as np
import rich
import starfile
import typer
import pandas as pd
from sklearn.cluster import DBSCAN

console = rich.console.Console()


def process_star_file(star_file_path: Path, file_label: str):
    """Process a single STAR file and return coordinates and dataframe"""
    star = starfile.read(star_file_path, always_dict=True)
    console.log(f"{star_file_path} ({file_label}) read")

    if not all(key in star for key in ('particles', 'optics')):
        console.log(f"expected RELION 3.1+ style STAR file containing particles and optics blocks in {file_label}", style="bold red")
        raise typer.Exit(1)

    df = star['particles'].merge(star['optics'], on='rlnOpticsGroup')
    console.log(f"optics table merged for {file_label}")
    console.log(f"{len(df)} particles found in {file_label}")

    # get relevant info from star file as numpy arrays
    console.log(f'grabbing relevant info from {file_label}...')

    xyz = df[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].to_numpy()
    console.log(f"got coordinates from 'rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ' for {file_label}")

    pixel_spacing = df['rlnImagePixelSize'].to_numpy()
    angpix = pixel_spacing[0]
    console.log(f"got pixel spacing from 'rlnImagePixelSize' for {file_label}: {angpix}")

    if all(col in df.columns for col in ['rlnOriginXAngst', 'rlnOriginYAngst', 'rlnOriginZAngst']):
        shifts = df[['rlnOriginXAngst', 'rlnOriginYAngst', 'rlnOriginZAngst']].to_numpy()
        console.log(f"got shifts from 'rlnOriginXAngst', 'rlnOriginYAngst', 'rlnOriginZAngst' for {file_label}")
    else:
        shifts = np.zeros(shape=(len(df), 3))
        console.log(f"no shifts found in 'rlnOriginXAngst', 'rlnOriginYAngst', 'rlnOriginZAngst' for {file_label}, setting to 0")

    # convert shifts to pixels then apply shifts to calculate absolute particle position
    pixel_spacing = pixel_spacing[:, np.newaxis]  # Shape: (n, 1)
    shifts_pixels = shifts / pixel_spacing
    console.log(f"converted shifts to pixels for {file_label}")
    xyz -= shifts_pixels
    console.log(f"applied shifts to particle positions for {file_label}")
    
    return xyz, df, star, angpix


def cli(
    input_star_file1: Path = typer.Option(..., '--input', '-i', help="first input star file (priority)"),
    input_star_file2: Path = typer.Option(..., '--input2', '-i2', help="second input star file"),
    min_distance: float = typer.Option(..., '--min_d', '-d', help="min distance in Angstrom"),
    output_star_file: Path = typer.Option(..., '--output', '-o', help="output star file"),
):
    # Process both files
    xyz1, df1, star1, angpix1 = process_star_file(input_star_file1, "file1")
    xyz2, df2, star2, angpix2 = process_star_file(input_star_file2, "file2")
    
    # Check if pixel sizes are compatible
    if abs(angpix1 - angpix2) > 0.001:
        console.log(f"Warning: pixel sizes differ between files: {angpix1} vs {angpix2}", style="bold yellow")
    
    # Use the pixel size from the first file as reference
    angpix = angpix1
    
    # Convert minimum distance to pixels
    min_distance_pixels = min_distance / angpix
    console.log(f"minimum distance: {min_distance} Å = {min_distance_pixels:.2f} pixels")
    
    # Combine coordinates for clustering
    xyz_combined = np.vstack([xyz1, xyz2])
    
    # Create labels to track which file each particle came from
    file_labels = np.concatenate([
        np.zeros(len(xyz1), dtype=int),  # 0 for file1
        np.ones(len(xyz2), dtype=int)    # 1 for file2
    ])
    
    # Perform clustering
    console.log("clustering particles using DBSCAN...")
    db = DBSCAN(eps=min_distance_pixels, min_samples=1).fit(xyz_combined)
    cluster_labels = db.labels_
    
    # Find which particles to keep (priority to file1)
    unique_clusters = np.unique(cluster_labels)
    keep_indices = []
    
    for cluster_id in unique_clusters:
        cluster_mask = cluster_labels == cluster_id
        cluster_files = file_labels[cluster_mask]
        cluster_indices = np.where(cluster_mask)[0]
        
        # If cluster contains particles from file1, keep the first one from file1
        if 0 in cluster_files:
            file1_in_cluster = cluster_indices[cluster_files == 0]
            keep_indices.append(file1_in_cluster[0])  # Keep first from file1
        else:
            # If cluster only contains particles from file2, keep the first one
            keep_indices.append(cluster_indices[0])
    
    keep_indices = np.array(keep_indices)
    keep_indices.sort()
    
    console.log(f"found {len(unique_clusters)} clusters")
    console.log(f"keeping {len(keep_indices)} particles")
    
    # Separate kept indices by file
    file1_keep = keep_indices[keep_indices < len(xyz1)]
    file2_keep = keep_indices[keep_indices >= len(xyz1)] - len(xyz1)
    
    console.log(f"keeping {len(file1_keep)} particles from file1")
    console.log(f"keeping {len(file2_keep)} particles from file2")
    
    # Create output dataframes
    df1_kept = star1['particles'].iloc[file1_keep].copy()
    df2_kept = star2['particles'].iloc[file2_keep].copy()
    
    # Combine the kept particles
    df_output = pd.concat([df1_kept, df2_kept], ignore_index=True)
    
    # Create output star file structure
    output_star = {
        'particles': df_output,
        'optics': star1['optics']  # Use optics from first file
    }
    
    # write output
    with console.status(f"writing output STAR file {output_star_file}", spinner="arc"):
        starfile.write(output_star, output_star_file)
    
    console.log(f"Output written to {output_star_file}")
    console.log(f"Total particles in output: {len(df_output)}")


if __name__ == "__main__":
    app = typer.Typer(add_completion=False)
    app.command(no_args_is_help=True)(cli)
    app()