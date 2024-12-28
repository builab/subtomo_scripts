#!/usr/bin/python3
"""
Convert the entire cilia from AA to Relion4 format.
Requirements:
- Run adjustOrigin from AA beforehand.
- Add HelicalTubeID.
- Compatibility with macOS and Linux for sed command.
- Compatibility with Python 3.9.
- Handle eulers_relion with one row.
- Read TomoVisibleFrames from tomostar file in a folder tomostar/
- Usage: convert_aa2relion5warp.py --i list_CU428base_ida_v1.txt --o coord.star --angpix 2.12 --bin 4  --imagesize 80
- CHATGPT - NOT WORKING
Author: HB (12/2024)
"""

import numpy as np
import pandas as pd
import argparse
import os
import re
import starfile
from eulerangles import euler2euler, convert_eulers

def preprocess_spider_doc(spiderdoc):
    """Remove lines starting with ' ;' from spiderdoc."""
    delimiter = "\\'" if os.name == 'posix' else ""
    cmd = f"sed -i {delimiter} '/^ ;/d' {spiderdoc}"
    os.system(cmd)

def preprocess_bstar(star_file):
    """Extract numerical data from a star file into a .txt file."""
    cmd = f"grep '^\\s*[0-9]' {star_file} > {star_file.replace('.star', '.txt')}"
    os.system(cmd)

def aa_to_relion5warp(star_file, doc_file, tomo_name, tomo_no, bin_factor, pixel_size, doublet_id):
    """Convert AA doc and star files to Relion 5.0-compatible format."""
    # Read the doc file
    header_list = ["no", "norec", "phi", "theta", "psi", "OriginX", "OriginY", "OriginZ", "cc"]
    df = pd.read_csv(doc_file, sep='\s+', names=header_list)

    # Process Euler angles
    eulers_zyz = df.iloc[:, 2:5].to_numpy() * -1
    eulers_zyz[:, 1] *= -1
    eulers_dynamo = euler2euler(
        eulers_zyz,
        source_axes='zyz',
        source_intrinsic=True,
        source_right_handed_rotation=True,
        target_axes='zxz',
        target_intrinsic=True,
        target_right_handed_rotation=True,
        invert_matrix=False
    )

    # Read the star file
    star_header = ["no", "c2", "c3", "c4", "CoordinateX", "CoordinateY", "CoordinateZ"]
    df_star = pd.read_csv(star_file, sep='\s+', names=star_header, usecols=[0, 4, 5, 6])

    # Initialize output DataFrame
    df_relion = pd.DataFrame({
        "TomoName": f"{tomo_name}.tomostar",
        "TomoParticleId": np.arange(len(df_star), dtype=np.int16) + 1,
        "CoordinateX": df_star['CoordinateX'],
        "CoordinateY": df_star['CoordinateY'],
        "CoordinateZ": df_star['CoordinateZ'],
        "HelicalTubeID": np.ones(len(df_star), dtype=np.int16) * doublet_id,
        "OpticsGroup": np.ones(len(df_star), dtype=np.int16) * tomo_no,
        "ClassNumber": np.ones(len(df_star), dtype=np.int8),
        "RandomSubset": np.tile([1, 2], len(df_star) // 2 + 1)[:len(df_star)],
        "OriginXAngst": 0,
        "OriginYAngst": 0,
        "OriginZAngst": 0
    })

    # Convert Euler angles to Relion format
    eulers_relion = convert_eulers(eulers_dynamo, source_meta='dynamo', target_meta='warp')
    eulers_relion = np.atleast_2d(eulers_relion)
    df_relion["AngleRot"] = eulers_relion[:, 0]
    df_relion["AngleTilt"] = eulers_relion[:, 1]
    df_relion["AnglePsi"] = eulers_relion[:, 2]

    # Add additional fields
    visible_frames = "[" + ",".join(["1"] * len(df_star)) + "]"
    df_relion["TomoVisibleFrames"] = visible_frames
    df_relion["ImageName"] = [
        f"../warp_tiltseries/particleseries/{tomo_name}/{tomo_name}_{pixel_size * bin_factor:.2f}A_{i:06d}.mrcs"
        for i in df_relion['TomoParticleId']
    ]

    return df_relion

if __name__ == '__main__':
    print('Script to convert from AxonemeAlign to Relion5 Warp. HB 2024')
    print('Warning: All the tomostars must be copy in tomostar/')
    parser = argparse.ArgumentParser(description='Convert doc & star file to Relion 4.0 input file')
    parser.add_argument('--i', help='Input list file', required=True)
    parser.add_argument('--ostar', help='Output star file', required=True)
    parser.add_argument('--angpix', help='Input pixel size', required=True, type=float)
    parser.add_argument('--imagesize', help='Input image size', required=True, type=float)
    parser.add_argument('--bin', help='Bin factor of current tomo', required=True, type=float)

    args = parser.parse_args()

    pixel_size = args.angpix
    image_size = args.imagesize
    bin_factor = args.bin

    tomo_list = {}
    tomo_no = 0
    df_all = pd.DataFrame()

    with open(args.i, 'r') as list_file:
        for line in list_file:
            if line.startswith('#'):
                continue

            record = line.split()
            tomo_name = re.sub('[a-z]$', '', record[0].replace('_ida_v1', '')[:-4])
            doublet_id = int(record[1][-1])

            if tomo_name not in tomo_list:
                tomo_no += 1
                tomo_list[tomo_name] = tomo_no

            star_file = f"star_corr/{record[1]}.star"
            doc_file = f"doc_corr/doc_total_{record[0]}.spi"

            preprocess_bstar(star_file)
            preprocess_spider_doc(doc_file)

            df_relion = aa_to_relion5warp(
                star_file.replace('.star', '.txt'), doc_file, tomo_name, tomo_no, bin_factor, pixel_size, doublet_id
            )

            df_all = pd.concat([df_all, df_relion], ignore_index=True)

    # Save output star file
    starfile.write({
        'general': {'TomoSubTomosAre2DStacks': 1},
        'optics': pd.DataFrame([{  # Example optics group data, update as needed
            "OpticsGroup": tomo_no,
            "TomoTiltSeriesPixelSize": pixel_size,
            "TomoSubtomogramBinning": bin_factor,
            "ImagePixelSize": pixel_size * bin_factor,
            "ImageSize": image_size
        }]),
        'particles': df_all
    }, args.ostar)

    print(f"Output written to {args.ostar}")
