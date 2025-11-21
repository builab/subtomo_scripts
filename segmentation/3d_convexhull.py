import numpy as np
import mrcfile
from scipy.spatial import ConvexHull, Delaunay
import argparse

def main():
    parser = argparse.ArgumentParser(description='create mask mrc file from segmented and cleaned tomogram')
    parser.add_argument('--i',
                        help='Relative directory of the input .mrc file',
                        required=True)
    parser.add_argument('--o',
                        help='Output file',
                        required=True)

    args = parser.parse_args()

    mrc_input = args.i
    mrc_output = args.o

    # Load tomogram
    with mrcfile.open(mrc_input, permissive=True) as mrc:
        vol = mrc.data.copy()

    # binary mask
    binary = vol > 0

    # Get coordinates of voxels
    pts = np.column_stack(np.nonzero(binary))
    if len(pts) < 4:
        raise ValueError("Not enough points to form a 3D convex hull.")

    # Compute bounding box
    zmin, ymin, xmin = pts.min(axis=0)
    zmax, ymax, xmax = pts.max(axis=0) + 1  # +1 because slicing is exclusive

    cropped_pts = pts - [zmin, ymin, xmin]  # shift coordinates for cropped box
    cropped_shape = (zmax - zmin, ymax - ymin, xmax - xmin)

    # Compute convex hull within cropped region
    hull = ConvexHull(cropped_pts)
    hull_delaunay = Delaunay(cropped_pts[hull.vertices])

    # Prepare coordinates for within the cropped region
    z, y, x = np.indices(cropped_shape)
    coords = np.column_stack((z.ravel(), y.ravel(), x.ravel()))

    # Fill convex hull
    inside = hull_delaunay.find_simplex(coords) >= 0
    cropped_mask = inside.reshape(cropped_shape).astype(np.uint8)

    # reshape back to full-sized volume
    full_mask = np.zeros_like(vol, dtype=np.uint8)
    full_mask[zmin:zmax, ymin:ymax, xmin:xmax] = cropped_mask

    # output result .mrc file
    print("writing output file...")
    with mrcfile.new(mrc_output, overwrite=True) as out:
        out.set_data(full_mask)

    print(f"volume saved to {mrc_output}")


if __name__ == "__main__":
    main()
