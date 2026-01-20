#!

import subprocess
import argparse
import os
import sys
import numpy as np
import mrcfile


def boundingbox(volume):
    nz, ny, nx = volume.shape

    # z range
    z_min, z_max = nz, -1
    for z in range(nz):
        if np.any(volume[z] > 0):
            if z < z_min: z_min = z
            if z > z_max: z_max = z

    # No data at all?
    if z_max == -1:
        print("empty mrc: no xyz restriction applied")
        return ""

    # y range
    y_min, y_max = ny, 0
    for y in range(ny):
        if np.any(volume[:, y, :] > 0):
            if y < y_min: y_min = y
            if y > y_max: y_max = y

    # x range
    x_min, x_max = nx, 0
    for x in range(nx):
        if np.any(volume[:, :, x] > 0):
            if x < x_min: x_min = x
            if x > x_max: x_max = x

    return f"  --search-x {x_min} {x_max} \\\n" \
           f"  --search-y {y_min} {y_max} \\\n" \
           f"  --search-z {z_min} {z_max} \\\n"


def principal_axis_angle(volume):
    pts = np.column_stack(np.nonzero(volume))

    # center points
    pts_centered = pts - pts.mean(axis=0)

    # PCA via covariance (eigen vector with largest eigen value is the principle axis)
    cov = np.cov(pts_centered, rowvar=False)
    eig_values, eig_vectors = np.linalg.eigh(cov)

    # principal direction = eigenvector with max eigenvalue
    long_axis = eig_vectors[:, np.argmax(eig_values)]
    xy_pa = [0, long_axis[1], long_axis[2]]
    unit_xy_pa = xy_pa / np.linalg.norm(xy_pa)

    # angle relative to y-axis
    heading_angle = np.arccos(np.clip(np.dot(unit_xy_pa, [0, 1, 0]), -1, 1))
    # print(f"angle in the xy plane relative to y-axis: {heading_angle}")

    unit_xyz_pa = long_axis / np.linalg.norm(long_axis)
    tilt_angle = np.arccos(np.clip(np.dot(unit_xyz_pa, unit_xy_pa), -1, 1))

    # return angles in degrees and co-linear positive angle
    return np.degrees(heading_angle) % 360.0, np.degrees(tilt_angle) % 360.0


def create_shell_script(scriptdir, t, tm, i_mrc, bbx_volume, odir, angle_list_file):
    os.makedirs(odir, exist_ok=True)

    try:
        with open(scriptdir, "w", newline="", encoding="utf-8") as command:
            command.write(f"pytom_match_template.py \\\n")
            command.write(f"  -t {t} \\\n")
            command.write(f"  -m {tm} \\\n")
            command.write(f"  -v {i_mrc} \\\n")
            command.write(f"  -d {odir} \\\n")
            command.write(f"  -a xml/CU428lowmag_11.tlt \\\n")
            command.write(f"  -g 0 \\\n")
            command.write(f"  --low-pass 40 \\\n")
            command.write(f"  --defocus xml/CU428lowmag_11_defocus.txt \\\n")
            command.write(f"  --amplitude 0.07 \\\n")
            command.write(f"  --spherical 2.7 \\\n")
            command.write(f"  --voltage 300 \\\n")
            command.write(f"  --tomogram-ctf-model phase-flip \\\n")
            command.write(f"  --volume-split 2 2 1 \\\n")
            command.write(f"  --random-phase-correction \\\n")
            command.write(f"  --dose-accumulation xml/CU428lowmag_11_dose.txt \\\n")
            command.write(f"  --per-tilt-weighting \\\n")
            if bbx_volume:
                command.write(boundingbox(bbx_volume))
            command.write(f"  --angular-search {angle_list_file}\n")

    except PermissionError:
        print("\033[91mError: You do not have permission to write to this directory.\033[0m")
        sys.exit(1)
    except OSError as e:
        print(f"\033[91mOS error: {e}\033[0m")
    except Exception as e:
        print(f"\033[91mUnexpected error: {e}\033[0m")


def main():
    parser = argparse.ArgumentParser(description='generate shell script to run template matching with pytom')
    parser.add_argument('--t', help='template mrc file', required=True)
    parser.add_argument('--tmask', help='template mask mrc file', required=True)
    parser.add_argument('--inputmrc', help='target mrc file for template matching',
                        required=True)
    parser.add_argument('--maskmrc',
                        help='mrc file of same size as inputmrc with voxels of value 1 in the space occupied by target structure, used for bouding box calculation and orientation',
                        required=False)
    parser.add_argument('--outfolder', help='output folder directory', required=True)
    parser.add_argument("--a", type=float, default=4, help="Angular increment in degrees.")
    parser.add_argument("--tilt_margin", type=float, default=10,
                        help="Limit X (theta) to 90 ± this value (in degrees).")
    parser.add_argument("--psi_margin", type=float, default=10, help="Limit Z2 (psi) to 90±X and 270±X (in degrees).")

    args = parser.parse_args()

    template_mrc = args.t
    template_mask_mrc = args.tmask
    input_mrc = args.inputmrc
    mask_mrc = args.maskmrc
    odir = args.outfolder

    # ----- collecting parameters from mrc and generate angle list -----
    volume = None
    if mask_mrc:
        with mrcfile.open(mask_mrc) as mrc:
            volume = mrc.data.copy()
            principal_angles = principal_axis_angle(volume)

        print("measure: ", principal_angles)

        psi_range = principal_angles[0] - args.psi_margin, principal_angles[0] + args.psi_margin
        tilt_range = principal_angles[1] - args.tilt_margin, principal_angles[1] + args.tilt_margin
    else:
        psi_range = 0.0, 180.0
        tilt_range = -10, 10

    angle_list_name = f"angle_list_{os.path.splitext(os.path.basename(template_mrc))[0]}"  # append template file name

    print("psi: ", psi_range, "tilt: ", tilt_range, "name: ", angle_list_name)

    subprocess.run(
        [sys.executable, "generate_pytom_angle_list_range.py", '--a', f'{args.a}', '--psi_limit', f'{psi_range[0]}',
         f'{psi_range[1]}', '--tilt_limit',
         f'{tilt_range[0]}', f'{tilt_range[1]}', '--o', f'{odir}/{angle_list_name}.txt'], check=True, stdout=sys.stdout,
        stderr=sys.stderr, )

    # ----- writing shell script -----
    shell_script_path = os.path.join(".", f"tm_{os.path.splitext(os.path.basename(template_mrc))[0]}.sh")

    create_shell_script(shell_script_path, template_mrc, template_mask_mrc, input_mrc, volume, odir, f'{odir}/{angle_list_name}.txt')

    if input("run template matching? [y/n]? ") == 'y':
        subprocess.run(["bash", f"tm_{os.path.splitext(os.path.basename(template_mrc))[0]}.sh"], check=True) #UNCOMMENT THIS WHEN READY!!!!!!!!


if __name__ == '__main__':
    main()

