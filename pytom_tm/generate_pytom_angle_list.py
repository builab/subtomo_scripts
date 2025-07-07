#!/usr/bin/env python3
# Script to generate angle_list for pytom_match_pick
# With option --tilt_limit for filament template matching
# With option --psi_limit for cilia where we collect not too far away from the tilt axis

import numpy as np
import healpy as hp
import logging
import argparse

def angle_to_angle_list(angle_diff: float, sort_angles: bool = True, log_level: int = logging.DEBUG) -> list[tuple[float, float, float]]:
    npix = 4 * np.pi / (angle_diff * np.pi / 180) ** 2
    nside = 0
    while hp.nside2npix(nside) < npix:
        nside += 1
    used_npix = hp.nside2npix(nside)
    used_angle_diff = (4 * np.pi / used_npix) ** 0.5 * (180 / np.pi)
    logging.log(log_level, f"Using an angle difference of {used_angle_diff:.4f} for Z1 and X")
    
    theta, phi = hp.pix2ang(nside, np.arange(used_npix))
    
    n_psi_angles = int(np.ceil(360 / angle_diff))
    psi, used_psi_diff = np.linspace(0, 2 * np.pi, n_psi_angles, endpoint=False, retstep=True)
    
    logging.log(log_level, f"Using an angle difference of {np.rad2deg(used_psi_diff):.4f} for Z2")
    
    angle_list = [(ph, th, ps) for ph, th in zip(phi, theta) for ps in psi]
    if sort_angles:
        angle_list.sort()
    return angle_list

def main():
    parser = argparse.ArgumentParser(description="Generate a list of ZXZ Euler angles with optional tilt and psi filtering.")
    parser.add_argument("--a", type=float, required=True, help="Angular increment in degrees.")
    parser.add_argument("--tilt_limit", type=float, default=None, help="Limit X (theta) to 90 ± this value (in degrees).")
    parser.add_argument("--psi_limit", type=float, default=None, help="Limit Z2 (psi) to 90±X and 270±X (in degrees).")
    parser.add_argument("--o", type=str, required=True, help="Output filename.")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    angle_list = angle_to_angle_list(args.a)

    # Tilt filter (X)
    if args.tilt_limit is not None:
        lower = np.deg2rad(90 - args.tilt_limit)
        upper = np.deg2rad(90 + args.tilt_limit)
        angle_list = [a for a in angle_list if lower <= a[1] <= upper]

    # Psi filter (Z2), in 0 to 2π
    if args.psi_limit is not None:
        psi_center1 = np.deg2rad(0)
        psi_center2 = np.deg2rad(180)
        psi_center3 = np.deg2rad(360)
        psi_half_range = np.deg2rad(args.psi_limit)

        def in_psi_range(psi):
            return (
                psi_center1 - psi_half_range <= psi <= psi_center1 + psi_half_range
                or
                psi_center2 - psi_half_range <= psi <= psi_center2 + psi_half_range
                or
                psi_center3 - psi_half_range <= psi <= psi_center3 + psi_half_range
            )

        angle_list = [a for a in angle_list if in_psi_range(a[2])]

    with open(args.o, "w") as f:
        for z1, x, z2 in angle_list:
            f.write(f"{z1:.6f} {x:.6f} {z2:.6f}\n")

    print(f"Wrote {len(angle_list)} angles to {args.o}")

if __name__ == "__main__":
    main()
