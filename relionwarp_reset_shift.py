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

# Script by Huy Bui & DeepSeek - Modified to reset shifts
# running with uv run relionwarp_reset_shift.py -i input.star -o output_resetshift.star


from pathlib import Path

import numpy as np
import rich
import starfile
import typer
import pandas as pd

console = rich.console.Console()


def cli(
    input_star_file: Path = typer.Option(..., '--input', '-i', help="input star file"),
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
    console.log("got coordinates from 'rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ'")

    pixel_spacing = df['rlnImagePixelSize'].to_numpy()
    console.log("got pixel spacing from 'rlnImagePixelSize'")

    if all(col in df.columns for col in ['rlnOriginXAngst', 'rlnOriginYAngst', 'rlnOriginZAngst']):
        shifts = df[['rlnOriginXAngst', 'rlnOriginYAngst', 'rlnOriginZAngst']].to_numpy()
        console.log("got shifts from 'rlnOriginXAngst', 'rlnOriginYAngst', 'rlnOriginZAngst'")
        
        # convert shifts from angstrom to pixels
        pixel_spacing = pixel_spacing[:, np.newaxis]  # Shape: (b, 1)
        shifts_pixels = shifts / pixel_spacing
        console.log("converted shifts from angstroms to pixels")
        
        # Update coordinates by adding the shifts
        xyz_updated = xyz - shifts_pixels
        console.log("updated coordinates by adding shifts")
        
        # Update the dataframe with new coordinates
        df['rlnCoordinateX'] = xyz_updated[:, 0]
        df['rlnCoordinateY'] = xyz_updated[:, 1]
        df['rlnCoordinateZ'] = xyz_updated[:, 2]
        console.log("updated coordinate columns in dataframe")
        
        # Reset shifts to zero
        df['rlnOriginXAngst'] = 0.0
        df['rlnOriginYAngst'] = 0.0
        df['rlnOriginZAngst'] = 0.0
        console.log("reset shifts to zero")
        
    else:
        console.log("no shifts found in 'rlnOriginXAngst', 'rlnOriginYAngst', 'rlnOriginZAngst', coordinates unchanged")

    # Update the particles table in the star dictionary
    # Remove the optics columns that were merged
    optics_columns = star['optics'].columns.tolist()
    optics_columns.remove('rlnOpticsGroup')  # Keep the group column for merging
    df_particles_updated = df.drop(columns=optics_columns)
    
    star['particles'] = df_particles_updated
    
    # write output
    with console.status(f"writing output STAR file {output_star_file}", spinner="arc"):
        starfile.write(star, output_star_file)
    console.log(f"Output reset shift written to {output_star_file}")


if __name__ == "__main__":
    app = typer.Typer(add_completion=False)
    app.command(no_args_is_help=True)(cli)
    app()