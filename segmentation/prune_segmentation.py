import argparse
import mrcfile
import numpy as np
from tqdm import tqdm
from skimage.measure import regionprops, label
from skimage.morphology import binary_erosion, binary_dilation, remove_small_objects
from skimage.draw import polygon as skpolygon
from imodmodel import ImodModel

"""
Created on Thu Oct 23 2025
Last updated Oct 30, 2025

script to process segmented tomogram. 
operations include bounding box filtering, island direction filtering, erosion with component voxel size filtering

@author: Khan Bao, Builab@McGill

General command syntax: 
python3 prune_segmentation.py operation --i segmented.mrc --o pruned.mrc --operation_specific_arguments
*underlined/italic values need to be replaced
Masking using IMOD model
In 3dmod, go to the widest part of the sample. 
Draw a model around the entire sample, make sure to start and end around the same point
Save model

Your command should then be like:
python3 prune_segmentation.py mask --i segmented.mrc --o pruned.mrc --modmaskdir maskingmodel
Bounding box ratio pruning 
python3 prune_segmentation.py boundingbox --i segmented.mrc --o pruned.mrc --minasp 5.0
Direction pruning
Directly guess cilia angle:
python3 prune_segmentation.py direction --i segmented.mrc --o pruned.mrc --angle 45 --anglerange 20

Suggested modification:
In 3dmod, go to the middle of the cilia
draw a model (Tick Model, then middle mouse of the end & the tip of the cilia)
save model

Then your command would be like:
python3 prune_segmentation.py direction --i segmented.mrc --o pruned.mrc --modangledir anglemodel --anglerange 20
Erosion pruning
python3 prune_segmentation.py erosion --i segmented.mrc --o pruned.mrc --iterate 2 --minvoxel 100000
"""


def maskmrc(volume, modeldir):
    """
    maskmrc

    takes volume to be masked and IMOD model with 2 contours: one polygon and another with 2 points

    1. scimage.draw is used to convert vertices of the polygon into a numpy array of the mask
    2. the z coordinates of the 2 points are used to define the range of the mask, other slices are left as zero-like
    3. for each z slice within the range, pixels at the masked xy coordinates are copied from the input volume

    return masked volume

    note: optional block of commented out code can be used to return only the mask without applying to the input volume
    """
    model = ImodModel.from_file(modeldir)

    # store size of the volume for mask generation
    nz, ny, nx = volume.shape

    # variables pre-defined for error catching
    polygon_contour = None  # collection of points in polygon masking object
    z_range = [0, volume.shape[0]]  # range z from min to max, default is all slices if no 2 point contour is found

    for obj in model.objects:
        for contour in obj.contours:
            points = np.array(contour.points)
            number_points = points.shape[0]

            # polygon require minimum 3 points
            if number_points > 2:
                polygon_contour = points

            # range require 2 points
            elif number_points == 2:
                z_range[0] = min(points[0][2], points[1][2])
                z_range[1] = max(points[0][2], points[1][2])

    # no polygon defined -- raise error
    if polygon_contour is None:
        raise ValueError("Could not identify polygon contour")

    # create matching coordinate array of desired pixels
    poly_xy = polygon_contour[:, :2]  # take only the xy coordinates to draw polygon
    rr, cc = skpolygon(poly_xy[:, 1], poly_xy[:, 0], shape=(ny, nx))  # draw the polygon using the xy coordinates of each point

    # use when testing mask
    # mask = np.zeros((ny, nx))
    # mask[rr, cc] = 1

    masked_volume = np.zeros_like(volume)

    # masking
    for z in tqdm(range(int(z_range[0]), int(z_range[1])), desc="masking slice"):
        masked_slice = np.zeros((ny, nx), dtype=volume.dtype)
        masked_slice[rr, cc] = volume[z, rr, cc]
        masked_volume[z] = masked_slice  # change to mask for outputting mask, change to masked_slice for normal operation

    return masked_volume
    print("masking applied")


def boundingbox(volume, minasp):
    """
    booundingbox

    takes volume to be processed and a threshold aspect ratio

    cycling through each slice along z:
    1. scimage.label is used to identify island/regions using connectivity 2.
    2. scimage.measure is used to measure the length of the major axis and minor axis of each island
    3. each region is only copied to the return volume if its major-axis:minor-axis ratio is greater or equal to inputted threshold

    return pruned volume
    """
    filtered = np.zeros_like(volume, dtype=volume.dtype)

    for z in tqdm(range(volume.shape[0]), desc="processing slice"):
        slice_2d = volume[z]

        # Skip empty slices
        if not np.any(slice_2d):
            continue

        # Relabel to ensure islands are contiguous
        labeled_slice = label(slice_2d > 0, connectivity=2)
        regprop = regionprops(labeled_slice)

        for region in regprop:
            # Measure elongation from major/minor axes
            major = region.major_axis_length
            minor = region.minor_axis_length if region.minor_axis_length > 0 else 1e-6
            aspect = major / minor

            # Keep only elongated (aspect >= 2)
            if aspect >= minasp:
                filtered[z][labeled_slice == region.label] = slice_2d[labeled_slice == region.label]

    return filtered


def direction(volume, angle, anglerange):
    """
    direction

    takes volume to be pruned, anlge as float number or IMOD file directory containing ONLY 2 points in a contour

    1. if IMOD is used to specify angle, first 2 points of the first contour in the first object is taken to calculate angle
    2. angle is remapped from 0-180 to 90-(-90) to match scimage's measurement system
    3. valid angle range is calculated by angle+-angle-range
    4. for each slice, orientation of each region is measured by angle of its major axis
    5. each region is only copied to the return volume if its orientation is within valid angel range

    return pruned volume
    """
    filtered = np.zeros_like(volume, dtype=volume.dtype)

    # if input is a file directory
    if isinstance(angle, str):
        model = ImodModel.from_file(angle)

        try:
            # extract points for angle calculation
            p1, p2 = model.objects[0].contours[0].points
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            angle = np.degrees(np.arctan2(dy, dx))

            # depending on the order of the two points drawn, they would produce a negative and a positive angle. convert negative to positive angel
            if angle < 0:
                angle += 180
        except Exception:
            raise ValueError("Model must ONLY have 2 points necessary to define an angle")

    # map angle to 90-(-90)
    angle = np.interp(angle, [0, 180], [90, -90])

    # check if angle is within range
    if -90 > angle or angle > 90 or 0 > anglerange or anglerange > 90:
        raise ValueError(f"--angle input out of specified range of 0 to 180")

    # calculate valid angle range
    minangle = angle - anglerange
    maxangle = angle + anglerange

    # clamp valid angle range to -90 to 90 degrees
    if minangle < -90.0:
        minangle = -90.0
    if maxangle > 90.0:
        maxangle = 90.0

    # convert to radians
    minangle *= 0.017453
    maxangle *= 0.017453

    for z in tqdm(range(volume.shape[0]), desc="processing slice"):
        slice_2d = volume[z]

        # Skip empty slices
        if not np.any(slice_2d):
            continue

        # label contiguous islands
        labeled_slice = label(slice_2d > 0, connectivity=2)
        regprop = regionprops(labeled_slice)

        for region in regprop:
            orient = region.orientation

            # Keep correctly oriented
            if minangle <= orient <= maxangle:
                filtered[z][labeled_slice == region.label] = slice_2d[labeled_slice == region.label]

    return filtered


def erosion(volume, iterate, minvoxel):
    """
    erosion

    takes volume to be pruned, number of iteration to erode, and voxel size threshold

    1. each slice is eroded using 2D binary erosion from skimage.morphology
    2. the resulting eroded volume is filtered by connectedd island (connectivity 2) voxel size
    3. islands smaller than threshold is removed

    return pruned volume

    note: returning volume is converted to binary form: pixels are either 0 or 1
    """
    processed = np.zeros_like(volume, dtype=bool)

    for z in tqdm(range(volume.shape[0]), desc="eroding slice"):
        slice_2d = volume[z] > 0

        # Skip empty slices
        if not np.any(slice_2d):
            continue

        for x in range(iterate):
            eroded = binary_erosion(slice_2d)
            slice_2d = eroded

        processed[z] = eroded

    print("removing small objects...")
    return remove_small_objects(processed, minvoxel, 2)


def main():
    parser = argparse.ArgumentParser(description='prune segmented tomogram')
    subparsers = parser.add_subparsers(dest='operation', description='type of pruning')

    # masking subparser
    parser_mask = subparsers.add_parser('mask',
                                        help='mask volume using IMOD model')
    parser_mask.add_argument('--i',
                        help='Relative directory of the input .mrc file',
                        required=True)
    parser_mask.add_argument('--o',
                        help='Output file',
                        required=True)
    parser_mask.add_argument('--modmaskdir',
                             help='Relative directory of the IMOD model with a polygon defineing area of interest and 2 points defining slice range',
                             required=True)

    # bounding box subparser
    parser_bbox = subparsers.add_parser("boundingbox",
                                        help="remove globular noise using threshold long-axis:short-axis ratio")
    parser_bbox.add_argument('--i',
                        help='Relative directory of the input .mrc file',
                        required=True)
    parser_bbox.add_argument('--o',
                        help='Output file',
                        required=True)
    parser_bbox.add_argument('--minasp',
                             help='minimum long axis to short axis ratio',
                             type=float,
                             default=5.0)

    # angle subparser
    parser_dir = subparsers.add_parser("direction", help="remove misaligned islands using direction of long-axis")
    parser_dir.add_argument('--i',
                        help='Relative directory of the input .mrc file',
                        required=True)
    parser_dir.add_argument('--o',
                        help='Output file',
                        required=True)
    parser_dir.add_argument('--angle',
                            help='island angle, in degrees, measured by angle from long-axis to vertical y-axis. Accepted range: 0 to 180',
                            type=float,
                            required=False)
    parser_dir.add_argument('--modangledir',
                            help='relative directory of the 3dmod model with 2 point line defining the major axis angle',
                            type=str,
                            required=False)
    parser_dir.add_argument('--anglerange',
                            help='maximum deviation from input angle over and under. Acceptedrange: 0 to 90',
                            type=float,
                            required=True)

    # erosion subparser
    parser_ero = subparsers.add_parser("erosion", help="erosion coupled with voxel size filtering")
    parser_ero.add_argument('--i',
                        help='Relative directory of the input .mrc file',
                        required=True)
    parser_ero.add_argument('--o',
                        help='Output file',
                        required=True)
    parser_ero.add_argument('--iterate', help='number of times to repeat erosion',
                            type=int,
                            default=1)
    parser_ero.add_argument('--minvoxel',
                            help='minimum number of voxels to keep each island determined with connectivity=2',
                            type=int,
                            default=100000)

    args = parser.parse_args()

    mrc_input = args.i
    mrc_output = args.o

    # load mrc
    with mrcfile.open(mrc_input) as mrc:
        volume = mrc.data.copy()
        pruned_vol = np.zeros_like(volume, dtype=volume.dtype)

    if args.operation == 'mask':
        print("masking volume...")
        pruned_vol = maskmrc(volume, args.modmaskdir)

    elif args.operation == 'boundingbox':
        pruned_vol = boundingbox(volume, args.minasp)

    elif args.operation == 'direction':
        # check for missing angle input
        if args.angle is None and args.modangledir is None:
            raise ValueError("one of --angle or --modangledir input is required")

        pruned_vol = direction(volume, args.angle if args.angle else args.modangledir, args.anglerange)

    elif args.operation == 'erosion':
        pruned_vol = erosion(volume, args.iterate, args.minvoxel)

    # output result .mrc file
    print("writing output file...")
    with mrcfile.new(mrc_output, overwrite=True) as out:
        out.set_data(pruned_vol.astype(np.uint8))

    print(f"volume saved to {mrc_output}")


if __name__ == '__main__':
    main()
