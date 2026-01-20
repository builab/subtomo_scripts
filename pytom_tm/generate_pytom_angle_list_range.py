#!/usr/bin/env python3
# Script to generate angle_list for pytom_match_pick
# Now supports tilt and psi ranges: --tilt_limit min max, --psi_limit min max
# --tilt_limit min max, --psi_limit min max
# This doesn't work yet unfortunately :(

import numpy as np
import healpy as hp
import logging
import argparse

def angle_to_angle_list(angle_diff: float, sort_angles: bool = True, log_level: int = logging.DEBUG):
    # Determine nside based on angular step
    npix = 4 * np.pi / (angle_diff * np.pi / 180) ** 2
    nside = 0
    while hp.nside2npix(nside) < npix:
        nside += 1

    used_npix = hp.nside2npix(nside)
    used_angle_diff = (4 * np.pi / used_npix) ** 0.5 * (180 / np.pi)
    logging.log(log_level, f"Using angle diff {used_angle_diff:.4f} for Z1 & X")

    theta, phi = hp.pix2ang(nside, np.arange(used_npix))

    # psi (Z2)
    n_psi_angles = int(np.ceil(360 / angle_diff))
    psi, used_psi_diff = np.linspace(0, 2 * np.pi, n_psi_angles, endpoint=False, retstep=True)
    logging.log(log_level, f"Using angle diff {np.rad2deg(used_psi_diff):.4f} for Z2")

    angle_list = [(ph, th, ps) for ph, th in zip(phi, theta) for ps in psi]
    if sort_angles:
        angle_list.sort()

    return angle_list

def main():
    parser = argparse.ArgumentParser(description="Generate ZXZ Euler angles with tilt and psi ranges.")
    parser.add_argument("--a", type=float, required=True, help="Angular increment in degrees.")

    parser.add_argument("--tilt_limit", nargs=2, type=float, default=None,
                        help="Tilt (theta) range: min max (degrees)")

    parser.add_argument("--psi_limit", nargs=2, type=float, default=None,
                        help="Psi (Z2) range: min max (degrees)")

    parser.add_argument("--asym_psi", action="store_true",
                    help="Do NOT generate symmetric negative psi range")

    parser.add_argument("--o", type=str, required=True, help="Output filename.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    angle_list = angle_to_angle_list(args.a)

    # ---- TILT FILTER ----
    if args.tilt_limit is not None:
        tilt_min, tilt_max = args.tilt_limit
        tilt_min = np.deg2rad(tilt_min)
        tilt_max = np.deg2rad(tilt_max)

        angle_list = [a for a in angle_list if tilt_min <= a[1] <= tilt_max]

    # ---- PSI FILTER (symmetric and asymmetric) ----
    if args.psi_limit is not None:
        psi_min, psi_max = args.psi_limit
        psi_min = np.deg2rad(psi_min)
        psi_max = np.deg2rad(psi_max)

        def in_range(psi, lo, hi):
            return lo <= psi <= hi

        asym_list = []
        symmetric_list = []

        for z1, x, psi in angle_list:
            # Normalize psi into [-pi, pi]
            psi_norm = (psi + np.pi) % (2 * np.pi) - np.pi

            # --- Asymmetric: only the given range ---
            keep_asym = in_range(psi_norm, psi_min, psi_max)

            # --- Symmetric: include +/- cone ---
            keep_sym = keep_asym or in_range(psi_norm, psi_min - np.pi, psi_max - np.pi)

            if keep_asym:
                asym_list.append((z1, x, psi))

            if keep_sym:
                symmetric_list.append((z1, x, psi))

        # Replace original angle_list with symmetric version
        angle_list = symmetric_list
    else:
        asym_list = angle_list.copy()
        symmetric_list = angle_list.copy()

    # ---- WRITE OUTPUT FILES ----
    # 1. Symmetric version
    with open(args.o, "w") as f:
        for z1, x, z2 in symmetric_list:
            f.write(f"{z1:.6f} {x:.6f} {z2:.6f}\n")

    # 2. Asymmetric version
    asym_name = args.o.replace(".txt", "_asym.txt")
    with open(asym_name, "w") as f:
        for z1, x, z2 in asym_list:
            f.write(f"{z1:.6f} {x:.6f} {z2:.6f}\n")

    print(f"Wrote {len(symmetric_list)} symmetric angles to {args.o}")
    print(f"Wrote {len(asym_list)} asymmetric angles to {asym_name}")


if __name__ == "__main__":
    main()

