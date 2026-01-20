import numpy as np
import mrcfile
from tqdm import tqdm
import argparse
from skimage.morphology import skeletonize
from scipy.sparse.linalg import svds
from skimage.measure import EllipseModel
from skimage.draw import ellipse
from scipy.spatial import ConvexHull


def principal_axis(pts: np.ndarray) -> np.ndarray:
    # Center the points
    center = pts.mean(axis=0)
    pts_centered = pts - center

    # Compute the top principal direction
    _, _, vt = svds(pts_centered, k=1, which='LM')
    long_axis = vt[0]
    return long_axis


def main():
    parser = argparse.ArgumentParser(description='create mask mrc file from segmented and cleaned tomogram')
    parser.add_argument('--i',
                        help='Relative directory of the input .mrc file',
                        required=True)
    parser.add_argument('--o',
                        help='Output file',
                        required=True)
    parser.add_argument('--step',
                        help='number of slices to skip per every ellipse fitting',
                        type=int,
                        default=1)

    args = parser.parse_args()

    mrc_input = args.i
    mrc_output = args.o
    step = args.step

    # Load tomogram
    with mrcfile.open(mrc_input, permissive=True) as mrc:
        volume = mrc.data.copy()
        expanded_vol = np.zeros_like(volume, dtype=volume.dtype)

    for y in tqdm(range(0, int(volume.shape[1]), step), desc="processing slice..."):
        # Normalize to [0,1] and threshold to binary
        binary = (volume[:, y, :] > 0).astype(np.uint8)

        if not np.any(binary):
            continue

        # Skeletonize
        skel = skeletonize(binary)

        pts = np.column_stack(np.nonzero(skel))
        if len(pts) < 5:
            continue

        # Calculate Convex Hull
        try:
            hull_attributes = ConvexHull(pts)
            hull_points = pts[hull_attributes.vertices]
        except Exception:
            print(f"hull failed at {y}")
            continue

        ell = EllipseModel()
        hull_xy = hull_points[:, ::-1] # reassign z, x to x,y(z) space for ellipse model

        if not ell.estimate(hull_xy):
            print(f"ellipse model failed at {y}")
            continue

        cx, cy, rho, r, theta = ell.params

        rr, cc = ellipse(
            r=cy,  # Center
            c=cx,  # Center
            r_radius=r,  # Radius 1
            c_radius=rho,  # Radius 2
            rotation=theta,  # Angle
            shape=volume[:, y, :].shape
        )

        expanded_vol[rr, y, cc] = 1

    # output result .mrc file
    print("writing output file...")
    with mrcfile.new(mrc_output, overwrite=True) as out:
        out.set_data(expanded_vol)

    print(f"volume saved to {mrc_output}")


if __name__ == '__main__':
    main()
