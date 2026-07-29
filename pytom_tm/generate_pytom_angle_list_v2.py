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


def axis_to_cone_center(axis: np.ndarray) -> tuple[float, float]:
    """
    Convert a 3D axis vector into the (theta, phi) direction, in degrees,
    that the pointing direction (Z1=phi, X=theta) of the angle list should be
    restricted around to form a cone around the axis.

    The pointing direction of a candidate orientation is a point on the unit
    sphere given by (theta, phi) -- exactly like the healpix grid used to
    build angle_list. A cone around the axis vector (points 1 -> 2 in the
    IMOD model) is just the set of points within some angular radius of the
    axis's own (theta, phi).
    """
    dx, dy, dz = axis
    r = np.linalg.norm(axis)
    theta_v = np.degrees(np.arccos(np.clip(dz / r, -1.0, 1.0)))
    phi_v = np.degrees(np.arctan2(dy, dx)) % 360
    return theta_v, phi_v


def circular_diff_deg(a: float, b: float) -> float:
    """Smallest difference between two angles (in degrees) on a circle."""
    d = abs((a - b) % 360)
    return min(d, 360 - d)


def angular_distance_deg(theta1: float, phi1: float, theta2: float, phi2: float) -> float:
    """Great-circle angular distance (in degrees) between two (theta, phi) directions (in degrees)."""
    t1, t2 = np.radians(theta1), np.radians(theta2)
    dphi = np.radians(phi1 - phi2)
    cos_d = np.cos(t1) * np.cos(t2) + np.sin(t1) * np.sin(t2) * np.cos(dphi)
    return np.degrees(np.arccos(np.clip(cos_d, -1.0, 1.0)))


def main():
    parser = argparse.ArgumentParser(description="Generate a list of ZXZ Euler angles with optional tilt and psi filtering.")
    parser.add_argument("--a", type=float, required=True, help="Angular increment in degrees.")
    parser.add_argument("--tilt_limit", type=float, default=None, help="Limit X (theta) to a band of this half-width around the reference tilt (in degrees).")
    parser.add_argument("--psi_limit", type=float, default=None, help="Limit Z2 (psi) to a band of this half-width around the reference psi (in degrees).")
    parser.add_argument("--imod_model", type=str, default=None, help="Path to a 2-point IMOD model file (.mod) defining a filament axis (direction: point 1 -> point 2). If given, --tilt_limit becomes the half-angle of a cone (pointing direction within tilt_limit degrees of the axis) instead of a band around the xy-plane. If omitted, the script behaves as before (tilt centered at 90 deg / in-plane).")
    parser.add_argument("--object_id", type=int, default=None, help="Restrict the IMOD model to this object_id, if the model contains more than one object/contour.")
    parser.add_argument("--contour_id", type=int, default=None, help="Restrict the IMOD model to this contour_id, if the model contains more than one object/contour.")
    parser.add_argument("--o", type=str, required=True, help="Output filename.")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    angle_list = angle_to_angle_list(args.a)

    # Psi (Z2, the "twist"/roll about the pointing direction) keeps its
    # original meaning regardless of --imod_model: it's measured relative to
    # the local meridian (the great circle through the pole at that point),
    # so a fixed 0/180 reference is valid everywhere on the sphere. This is
    # unrelated to *where* the pointing direction is, which is what the axis
    # controls below.
    psi_centers = [0.0, 180.0]

    if args.imod_model is not None:
        # With a known axis, the pointing direction (Z1=phi, X=theta) itself
        # should be restricted to a true cone around the axis -- i.e. within
        # tilt_limit degrees (great-circle distance) of the axis direction,
        # not just a band on theta with phi left free.
        axis = get_axis_from_model(args.imod_model, args.object_id, args.contour_id)
        theta_c, phi_c = axis_to_cone_center(axis)
        logging.info(
            f"Derived cone center from IMOD model '{args.imod_model}' "
            f"(theta, phi) = ({theta_c:.4f}, {phi_c:.4f}) deg"
        )

        if args.tilt_limit is not None:
            angle_list = [
                a for a in angle_list
                if angular_distance_deg(np.degrees(a[1]), np.degrees(a[0]), theta_c, phi_c) <= args.tilt_limit
            ]
    else:
        # Original behavior: theta (X) restricted to a band around the
        # xy-plane (90 deg), phi (Z1) left completely free.
        if args.tilt_limit is not None:
            lower = np.deg2rad(90 - args.tilt_limit)
            upper = np.deg2rad(90 + args.tilt_limit)
            angle_list = [a for a in angle_list if lower <= a[1] <= upper]

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