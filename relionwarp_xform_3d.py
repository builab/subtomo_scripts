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

from pathlib import Path
from typing import Tuple, Optional

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
    add_protomer_index: Optional[int] = typer.Option(None, '--add_protomer_index', help="Optional: set rlnProtomerIndex to this value for all particles"),
):
    star = starfile.read(input_star_file, always_dict=True)
    console.log(f"{input_star_file} read")

    if not all(key in star for key in ('particles', 'optics')):
        console.log("expected RELION 3.1+ style STAR file containing particles and optics blocks", style="bold red")
        return

    particles = star['particles']
    
    # 1. Add rlnOriginalIndex as a copy of rlnTomoParticleId
    if 'rlnTomoParticleId' in particles.columns:
        particles['rlnOriginalIndex'] = particles['rlnTomoParticleId']
        console.log("added 'rlnOriginalIndex' as a copy of 'rlnTomoParticleId'")
    else:
        console.log("Warning: 'rlnTomoParticleId' not found; 'rlnOriginalIndex' not created", style="yellow")

    # 2. Add rlnProtomerIndex if the optional argument is provided
    if add_protomer_index is not None:
        particles['rlnProtomerIndex'] = add_protomer_index
        console.log(f"added 'rlnProtomerIndex' with value {add_protomer_index}")

    # Merge for transformation calculations
    df = particles.merge(star['optics'], on='rlnOpticsGroup')
    console.log("optics table merged")
    console.log(f"{len(df)} particles found")

    # --- Transformation Logic ---
    xyz = df[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].to_numpy()
    pixel_spacing = df['rlnImagePixelSize'].to_numpy()
    euler_angles = df[['rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']].to_numpy()

    if all(col in df.columns for col in ['rlnOriginXAngst', 'rlnOriginYAngst', 'rlnOriginZAngst']):
        shifts = df[['rlnOriginXAngst', 'rlnOriginYAngst', 'rlnOriginZAngst']].to_numpy()
    else:
        shifts = np.zeros(shape=(df.shape[0], 3))

    pixel_spacing = einops.rearrange(pixel_spacing, 'b -> b 1')
    shifts_in_pixels = shifts / pixel_spacing
    xyz -= shifts_in_pixels

    rotation_matrices = R.from_euler(angles=euler_angles, seq='ZYZ', degrees=True).inv().as_matrix()
    rotation = R.from_euler(angles=np.asarray(rots), seq='ZYZ', degrees=True).as_matrix()

    new_xyz, updated_particle_orientations = shift_then_rotate_particles(
        particle_positions=xyz,
        particle_orientations=rotation_matrices,
        shift=np.asarray(shift),
        rotation=rotation,
    )

    new_shifts = new_xyz - df[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].to_numpy()
    new_shifts = -1 * new_shifts * pixel_spacing
    
    particles[['rlnOriginXAngst', 'rlnOriginYAngst', 'rlnOriginZAngst']] = new_shifts
    new_euler_angles = R.from_matrix(updated_particle_orientations).inv().as_euler('ZYZ', degrees='True')
    particles[['rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']] = new_euler_angles

    # Write output
    with console.status(f"writing output STAR file {output_star_file}", spinner="arc"):
        starfile.write(star, output_star_file)
    console.log(f"Output written to {output_star_file}")

def shift_then_rotate_particles(particle_positions, particle_orientations, shift, rotation):
    shift = einops.rearrange(shift, 'xyz -> xyz 1')
    local_shifts = particle_orientations @ shift
    local_shifts = einops.rearrange(local_shifts, 'b xyz 1 -> b xyz')
    updated_particle_positions = particle_positions + local_shifts
    particle_orientations = particle_orientations @ rotation
    return updated_particle_positions, particle_orientations

if __name__ == "__main__":
    app = typer.Typer(add_completion=False)
    app.command(no_args_is_help=True)(cli)
    app()