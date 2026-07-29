#!/usr/bin/env python3
# Script to generate angle_list for pytom_match_pick
# With option --tilt_limit for filament template matching
# With option --psi_limit for cilia where we collect not too far away from the tilt axis
# With option --imod_model to derive the tilt/psi reference directions from a
# 2-point IMOD model (instead of the fixed xy-plane / tilt-axis reference)

import numpy as np
import healpy as hp
import logging
import argparse

try:
    import imodmodel
except ImportError:
    imodmodel = None


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


def get_axis_from_model(model_path: str, object_id: int = None, contour_id: int = None) -> np.ndarray:
    """Read a 2-point IMOD model and return the (dx, dy, dz) axis vector between the points."""
    if imodmodel is None:
        raise ImportError(
            "The 'imodmodel' package is required to read IMOD model files. "
            "Install it with: pip install imodmodel"
        )

    df = imodmodel.read(model_path)

    if object_id is not None:
        df = df[df["object_id"] == object_id]
    if contour_id is not None:
        df = df[df["contour_id"] == contour_id]

    if len(df) != 2:
        raise ValueError(
            f"Expected exactly 2 points to define an axis, found {len(df)} "
            f"(after object_id={object_id}, contour_id={contour_id} filtering). "
            "Use --object_id and/or --contour_id to select a single 2-point contour."
        )

    points = df[["x", "y", "z"]].to_numpy()
    axis = points[1] - points[0]
    if np.linalg.norm(axis) == 0:
        raise ValueError("The two IMOD model points are identical; cannot define an axis.")
    return axis


def axis_to_reference_angles(axis: np.ndarray) -> tuple[float, float]:
    """
    Convert a 3D axis vector into the reference angles used to center the
    tilt and psi filters, in degrees.

    theta_ref: reference colatitude for the tilt (X) filter. The angle an
               (undirected) line makes with the xy-plane is a single,
               well-defined value in [0, 90] deg (deviation = 0 means the
               axis lies flat in the xy-plane, 90 means it points straight
               along Z), computed from |dz| so it doesn't depend on which
               model point was clicked first. theta_ref = 90 - deviation,
               so it reduces to 90 (the old default) when the axis is
               in-plane, and to 0 when the axis is along Z.
    phi_ref:   azimuth of the axis projected onto the xy-plane, in [0, 360).
               Unlike the tilt, the azimuth genuinely has two equally valid
               polarities for an undirected axis (phi_ref and phi_ref+180),
               which the caller uses both of when filtering psi.
    """
    dx, dy, dz = axis
    r = np.linalg.norm(axis)
    deviation_from_plane = np.degrees(np.arcsin(np.clip(abs(dz) / r, -1.0, 1.0)))
    theta_ref = 90 - deviation_from_plane
    phi_ref = np.degrees(np.arctan2(dy, dx)) % 360
    return theta_ref, phi_ref


def circular_diff_deg(a: float, b: float) -> float:
    """Smallest difference between two angles (in degrees) on a circle."""
    d = abs((a - b) % 360)
    return min(d, 360 - d)


def main():
    parser = argparse.ArgumentParser(description="Generate a list of ZXZ Euler angles with optional tilt and psi filtering.")
    parser.add_argument("--a", type=float, required=True, help="Angular increment in degrees.")
    parser.add_argument("--tilt_limit", type=float, default=None, help="Limit X (theta) to a band of this half-width around the reference tilt (in degrees).")
    parser.add_argument("--psi_limit", type=float, default=None, help="Limit Z2 (psi) to a band of this half-width around the reference psi (in degrees).")
    parser.add_argument("--imod_model", type=str, default=None, help="Path to a 2-point IMOD model file (.mod) defining a filament axis. If given, the tilt reference becomes the axis's deviation from the xy-plane and the psi reference becomes the axis's azimuth (both used symmetrically, since the axis has no polarity). If omitted, the script behaves as before (tilt centered at 90 deg / in-plane, psi centered at 0/180 deg).")
    parser.add_argument("--object_id", type=int, default=None, help="Restrict the IMOD model to this object_id, if the model contains more than one object/contour.")
    parser.add_argument("--contour_id", type=int, default=None, help="Restrict the IMOD model to this contour_id, if the model contains more than one object/contour.")
    parser.add_argument("--o", type=str, required=True, help="Output filename.")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    angle_list = angle_to_angle_list(args.a)

    # Determine the reference centers for the tilt (theta) and psi filters.
    if args.imod_model is not None:
        axis = get_axis_from_model(args.imod_model, args.object_id, args.contour_id)
        theta_ref, phi_ref = axis_to_reference_angles(axis)
        # Tilt (angle to the xy-plane) is single-valued for a line.
        # Azimuth (psi) has two equally valid polarities for an undirected
        # axis, so both phi_ref and phi_ref+180 are accepted below.
        tilt_centers = [theta_ref]
        psi_centers = sorted({phi_ref % 360, (phi_ref + 180) % 360})
        logging.info(
            f"Derived reference angles from IMOD model '{args.imod_model}': "
            f"tilt (theta) centers = {tilt_centers} deg, psi centers = {psi_centers} deg"
        )
    else:
        tilt_centers = [90.0]
        psi_centers = [0.0, 180.0]

    # Tilt filter (X)
    if args.tilt_limit is not None:
        angle_list = [
            a for a in angle_list
            if min(abs(np.degrees(a[1]) - c) for c in tilt_centers) <= args.tilt_limit
        ]

    # Psi filter (Z2), circular distance handles wraparound at 0/360 automatically
    if args.psi_limit is not None:
        angle_list = [
            a for a in angle_list
            if min(circular_diff_deg(np.degrees(a[2]), c) for c in psi_centers) <= args.psi_limit
        ]

    with open(args.o, "w") as f:
        for z1, x, z2 in angle_list:
            f.write(f"{z1:.6f} {x:.6f} {z2:.6f}\n")

    print(f"Wrote {len(angle_list)} angles to {args.o}")


if __name__ == "__main__":
    main()