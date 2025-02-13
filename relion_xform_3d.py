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

# Script from Alister Burt, modified to take rot by Huy Bui
# running with uv run relion_xform_3d.py -i input.star -o output.star -s 0 0 1.073 -r 25.78 0 0


from pathlib import Path
from typing import Tuple

import einops
import numpy as np
import rich
import starfile
import typer
from scipy.spatial.transform import Rotation as R

console = rich.console.Console()


def cli(
    input_star_file: Path = typer.Option(..., '--input', '-i', help="input star file"),
    shift: tuple[float, float, float] = typer.Option(..., '--shift', '-s', help="shift x, y and z"),
    rots: tuple[float, float, float] = typer.Option(..., '--rot', '-r', help="eulers rot, tilt and psi"),
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

    euler_angles = df[['rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']].to_numpy()
    console.log("got euler angles from 'rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi'")

    if all(col in df.columns for col in ['rlnOriginXAngst', 'rlnOriginYAngst', 'rlnOriginZAngst']):
        shifts = df[['rlnOriginXAngst', 'rlnOriginYAngst', 'rlnOriginZAngst']].to_numpy()
        console.log("got shifts from 'rlnOriginXAngst', 'rlnOriginYAngst', 'rlnOriginZAngst'")
    else:
        shifts = np.zeros(shape=(3,))
        console.log("no shifts found in 'rlnOriginXAngst', 'rlnOriginYAngst', 'rlnOriginZAngst', setting to 0")

    # convert shifts to angstrom then apply shifts to calculate absolute particle position
    pixel_spacing = einops.rearrange(pixel_spacing, 'b -> b 1')
    shifts = shifts / pixel_spacing
    console.log("converted shifts to angstroms")
    xyz -= shifts
    console.log("applied shifts to particle positions")

    # get particle rotation matrices (column vectors are particle x/y/z in tomogram)
    rotation_matrices = R.from_euler(angles=euler_angles, seq='ZYZ', degrees=True).inv().as_matrix()
    console.log("calculated rotation matrices from euler angles")
    
    # Get in MT rotation of 25.7866
    rotation = R.from_euler(angles=np.asarray(rots), seq='ZYZ', degrees=True).as_matrix()
    #print(rotation)

    # recenter particles, we don't care about orientations so apply identity rotation
    new_xyz, updated_particle_orientations = shift_then_rotate_particles(
        particle_positions=xyz,
        particle_orientations=rotation_matrices,
        shift=np.asarray(shift),
        rotation=rotation,
    )
    console.log('calculated shifted particle positions')

    # express new positions relative to old positions in star file
    new_shifts = new_xyz - df[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].to_numpy()
    new_shifts = -1 * new_shifts * pixel_spacing
    console.log("calculated new shifts from shifted particle positions")
    
    star['particles'][['rlnOriginXAngst', 'rlnOriginYAngst', 'rlnOriginZAngst']] = new_shifts
    console.log("updated shift values in 'rlnOriginXAngst', 'rlnOriginYAngst', 'rlnOriginZAngst'")
    
    # express new orientation in star file
    new_euler_angles = R.from_matrix(updated_particle_orientations).inv().as_euler('ZYZ', degrees='True')
    console.log("calculated new eulers")
    
    star['particles'][['rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']] = new_euler_angles
    console.log("updated rotational angle values in 'rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi'")


    # write output
    with console.status(f"writing output STAR file {output_star_file}", spinner="arc"):
        starfile.write(star, output_star_file)
    console.log(f"Output with updated shifts written to {output_star_file}")


def shift_then_rotate_particles(
    particle_positions,  # (n, 3)
    particle_orientations,  # (n, 3, 3)
    shift,  # (3, )
    rotation,  # (3, 3)
) -> Tuple[np.ndarray, np.ndarray]:  # positions, orientations
    # goal: apply transformations in the local coordinate
    # system of each particle

    # transform the shifts into the local particle reference frame
    shift = einops.rearrange(shift, 'xyz -> xyz 1')
    local_shifts = particle_orientations @ shift
    local_shifts = einops.rearrange(local_shifts, 'b xyz 1 -> b xyz')

    # apply the shifts
    updated_particle_positions = particle_positions + local_shifts

    # transform the reference rotation to find the new particle orientation
    #print(particle_orientations)
    particle_orientations = particle_orientations @ rotation
    #print(particle_orientations)
    return updated_particle_positions, particle_orientations


if __name__ == "__main__":
    app = typer.Typer(add_completion=False)
    app.command(no_args_is_help=True)(cli)
    app()