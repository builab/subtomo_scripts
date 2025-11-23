import mrcfile
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import binary_erosion
import argparse
import time

'''
@author: Khan Bao, Builab@McGill
'''

def main():
    parser = argparse.ArgumentParser(description='create mask mrc file from segmented and cleaned tomogram')
    parser.add_argument('--i',
                        help='Relative directory of the input .mrc file',
                        required=True)
    parser.add_argument('--o',
                        help='Output file',
                        required=True)
    parser.add_argument('--featherwidth',
                        help='thickness of feather in number of voxels',
                        type=int,
                        default=3)
    args = parser.parse_args()

    mrc_input = args.i
    mrc_output = args.o
    ramp_width = args.featherwidth

    with mrcfile.open(mrc_input, permissive=True) as mrc:
        mask = (mrc.data > 0.5).astype(np.uint8)

    smoothed = np.zeros_like(mask, dtype=np.float32)

    start_time = time.time()

    # Compute distance of 0 voxels from the surface
    dist_outside = distance_transform_edt(mask == 0)

    # Inside stays 1
    smoothed[mask == 1] = 1.0

    # Outside with full decay
    outside_pixels = (mask == 0) & (dist_outside < ramp_width)

    d = dist_outside[outside_pixels]

    # cosine 1 → 0 decay
    smoothed[outside_pixels] = 0.5 * (1.0 + np.cos(np.pi * d / ramp_width))

    # output result .mrc file
    print("writing output file...")
    with mrcfile.new(mrc_output, overwrite=True) as out:
        out.set_data(smoothed)

    print(f"volume saved to {mrc_output}")
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()
