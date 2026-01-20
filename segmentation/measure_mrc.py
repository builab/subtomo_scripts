import numpy as np
import mrcfile
import argparse

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
            print("empty mrc")
            return

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

        print(f"[--search-x {x_min} {x_max}]")
        print(f"[--search-y {y_min} {y_max}]")
        print(f"[--search-z {z_min} {z_max}]")


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
    angle = np.arccos(np.clip(np.dot(unit_xy_pa, [0, 1, 0]), -1, 1))*180/np.pi + 90
    print(f"angle in the xy plane relative to y-axis: {angle}")

def main():
    parser = argparse.ArgumentParser(description='measurements on segmented tomogram')
    subparsers = parser.add_subparsers(dest='measure', description='type of measurement')

    # bounding box subparser
    parser_bbox = subparsers.add_parser("boundingbox", help="measures the smallest xyz range that contain the density")
    parser_bbox.add_argument('--i',
                            help='Relative directory of the input .mrc file',
                            required=True)

    # angle subparser
    parser_dila = subparsers.add_parser("angle", help="measures the angle of density relative to y-axis")
    parser_dila.add_argument('--i',
                             help='Relative directory of the input .mrc file',
                             required=True)

    args = parser.parse_args()

    mrc_input = args.i

    # load mrc
    with mrcfile.open(mrc_input) as mrc:
        volume = mrc.data.copy()

    if args.measure == 'boundingbox':
        boundingbox(volume)
    elif args.measure == 'angle':
        principal_axis_angle(volume)


if __name__ == '__main__':
    main()
