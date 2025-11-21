import numpy as np
import mrcfile
from tqdm import tqdm
import argparse
from skimage.transform import rotate
from skimage.morphology import skeletonize
from skimage.measure import EllipseModel
from skimage.draw import ellipse
from scipy.spatial import ConvexHull


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
    angle = np.arccos(np.clip(np.dot(unit_xy_pa, [0, 1, 0]), -1, 1))
    print(f"angle in the xy plane relative to y-axis: {angle}")

    # return the opposite (negative angle as positive and vise versa) so it can be directly used to rotate
    if unit_xy_pa[1] < 0:
        return np.degrees(angle)
    else:
        return -1.0 * np.degrees(angle)


def rotate_mrc(volume, angle):
    rotated_vol = np.zeros_like(volume)
    for z in tqdm(range(int(volume.shape[0])), desc="rotating slice"):
        binary = (volume[z] > 0).astype(bool)
        rotated_vol[z] = rotate(binary, angle, mode='constant', cval=0)

    return rotated_vol


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
    parser.add_argument('--o_rotated',
                        help='Output file of the rotated mrc with major axis aligned to y-axis',
                        required=False)
    args = parser.parse_args()

    mrc_input = args.i
    mrc_output = args.o
    step = args.step
    rot_mrc_output = args.o_rotated

    # Load tomogram
    with mrcfile.open(mrc_input, permissive=True) as mrc:
        volume = mrc.data.copy()

        tilt_angle_to_y = principal_axis_angle(volume)
        rotated_vol = rotate_mrc(volume, tilt_angle_to_y)

        result_vol = np.zeros_like(rotated_vol, dtype=volume.dtype)

    for y in tqdm(range(0, int(rotated_vol.shape[1]), step), desc="processing slice"):
        # Normalize to [0,1] and threshold to binary
        binary = (rotated_vol[:, y, :] > 0).astype(np.uint8)

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
        hull_xy = hull_points[:, ::-1]  # reassign z, x to x,y(z) space for ellipse model

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
            shape=rotated_vol[:, y, :].shape
        )

        result_vol[rr, y, cc] = 1

    result_vol = rotate_mrc(result_vol, -tilt_angle_to_y)

    # output result .mrc file
    print("writing output file...")
    with mrcfile.new(mrc_output, overwrite=True) as out:
        out.set_data(result_vol)
    print(f"volume saved to {mrc_output}")

    if rot_mrc_output:
        with mrcfile.new(rot_mrc_output, overwrite=True) as out:
            out.set_data(rotated_vol)
        print(f"rotated volume saved to {rot_mrc_output}")


if __name__ == '__main__':
    main()
