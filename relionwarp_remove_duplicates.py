# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas",
#     "scipy",
#     "starfile",
#     "typer",
#     "einops",
#     "rich",
# ]
# [tool.uv]
# exclude-newer = "2025-01-01T00:00:00Z"
# ///

# Script by Huy Bui & DeepSeek
# running with uv run relionwarp_remove_duplicates.py -i input.star -o output.star -d 45


from pathlib import Path

import numpy as np
import rich
import starfile
import typer
import pandas as pd
from sklearn.cluster import DBSCAN

console = rich.console.Console()


def cli(
    input_star_file: Path = typer.Option(..., '--input', '-i', help="input star file"),
    min_distance: Path = typer.Option(..., '--min_d', '-d', help="min distance in Angstrom"),
    output_star_file: Path = typer.Option(..., '--output', '-o', help="output star file"),
):
    star = starfile.read(input_star_file, always_dict=True)
    console.log(f"{input_star_file} read")

    if not all(key in star for key in ('particles', 'optics')):
        console.log("expected RELION 3.1+ style STAR file containing particles and optics blocks", style="bold red")

    df = star['particles'].merge(star['optics'], on='rlnOpticsGroup')
    console.log("optics table merged")
    console.log(f"{len(df)} particles found")

    # get relevant info from star file as numpy arrays
    console.log('grabbing relevant info...')

    xyz = df[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].to_numpy()
    console.log("got shifts from 'rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ'")

    pixel_spacing = df['rlnImagePixelSize'].to_numpy()
    console.log("got pixel spacing from 'rlnImagePixelSize'")

    if all(col in df.columns for col in ['rlnOriginXAngst', 'rlnOriginYAngst', 'rlnOriginZAngst']):
        shifts = df[['rlnOriginXAngst', 'rlnOriginYAngst', 'rlnOriginZAngst']].to_numpy()
        console.log("got shifts from 'rlnOriginXAngst', 'rlnOriginYAngst', 'rlnOriginZAngst'")
    else:
        shifts = np.zeros(shape=(3,))
        console.log("no shifts found in 'rlnOriginXAngst', 'rlnOriginYAngst', 'rlnOriginZAngst', setting to 0")

    # convert shifts to angstrom then apply shifts to calculate absolute particle position
    pixel_spacing = pixel_spacing[:, np.newaxis]  # Shape: (b, 1)
    shifts = shifts / pixel_spacing
    console.log("converted shifts to angstroms")
    xyz -= shifts
    console.log("applied shifts to particle positions")
    
    # Remove duplicates by DeepSeek
    min_distance = float(min_distance) / pixel_spacing
    db = DBSCAN(eps=minD, min_samples=1).fit(coords)

    # Get cluster labels
    df['Cluster'] = db.labels_

    # Keep one representative point per cluster (e.g., first occurrence)
    df_unique = df.groupby('Cluster').first().reset_index()
    
    # Drop cluster column
    df_unique = df_unique.drop(columns=['Cluster'])
    
    star['particles'] = df_unique
    
    # write output
    with console.status(f"writing output STAR file {output_star_file}", spinner="arc"):
        starfile.write(star, output_star_file)
    console.log(f"Output with updated shifts written to {output_star_file}")


if __name__ == "__main__":
    app = typer.Typer(add_completion=False)
    app.command(no_args_is_help=True)(cli)
    app()