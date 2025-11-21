import argparse
import sys
import mrcfile
import numpy as np
from tqdm import tqdm
from skimage.morphology import binary_erosion, binary_dilation

'''
@author: Khan Bao, Builab@McGill
'''


def erosion(volume, iterate):
    processed = np.zeros_like(volume, dtype=bool)
    for z in tqdm(range(volume.shape[0]), desc="eroding slice"):
        slice_2d = volume[z] > 0

        # Skip empty slices
        if not np.any(slice_2d):
            continue

        for _ in range(iterate - 1):
            slice_2d = binary_erosion(slice_2d)

        processed[z] = slice_2d
    return processed


def dilation(volume, iterate):
    processed = np.zeros_like(volume, dtype=bool)
    for z in tqdm(range(volume.shape[0]), desc="dilating slice"):
        slice_2d = volume[z] > 0

        # Skip empty slices
        if not np.any(slice_2d):
            continue

        for _ in range(iterate - 1):
            slice_2d = binary_dilation(slice_2d)

        processed[z] = slice_2d
    return processed


def main():
    parser = argparse.ArgumentParser(description='morphological operation on segmented tomogram')
    subparsers = parser.add_subparsers(dest='operation', description='type of operation')

    # erosion subparser
    parser_ero = subparsers.add_parser("erosion", help="erosion")
    parser_ero.add_argument('--i',
                            help='Relative directory of the input .mrc file',
                            required=True)
    parser_ero.add_argument('--o',
                            help='Output file',
                            required=True)
    parser_ero.add_argument('--iterate', help='number of times to repeat erosion',
                            type=int,
                            default=1)

    # dilation subparser
    parser_dila = subparsers.add_parser("dilation", help="dilation")
    parser_dila.add_argument('--i',
                             help='Relative directory of the input .mrc file',
                             required=True)
    parser_dila.add_argument('--o',
                             help='Output file',
                             required=True)
    parser_dila.add_argument('--iterate', help='number of times to repeat dilation',
                             type=int,
                             default=1)

    args = parser.parse_args()

    mrc_input = args.i
    mrc_output = args.o

    # command_order = sys.argv[1:]
    # print(command_order)

    # load mrc
    with mrcfile.open(mrc_input) as mrc:
        volume = mrc.data.copy()
        result_vol = np.zeros_like(volume, dtype=volume.dtype)

    if args.operation == 'erosion':
        result_vol = erosion(volume, args.iterate)

    elif args.operation == 'dilation':
        result_vol = dilation(volume, args.iterate)

    # output result .mrc file
    print("writing output file...")
    with mrcfile.new(mrc_output, overwrite=True) as out:
        out.set_data(result_vol.astype(np.uint8))
    print(f"volume saved to {mrc_output}")


if __name__ == '__main__':
    main()
