#!/usr/bin/python
# Script to generate star file for the visualization in ArtiaX from Relion WarpTool 2.0.0
# Multiply origin with pixelSize
# Eliminate .tomostar from rlnTomoName
# Also, deal with no optics header now, just need --angpix
# HB, McGill, 2025. Style coming from recenter_3d.py by Alister Burt

import numpy as np
import sys, os, re
import starfile
import argparse


def modify_star(input_star_file, output_star_file, angpix=None):
    star = starfile.read(input_star_file, always_dict=True)
    print(f"{input_star_file} read")
    
    if not all(key in star for key in ('particles', 'optics')):
        print("expected RELION 3.1+ style STAR file containing particles and optics blocks")

    if 'optics' in star:
        df = star['particles'].merge(star['optics'], on='rlnOpticsGroup')
        print("optics table merged")
    else:
        df = star['particles'].copy()
        print("no optics table")
        
    print(f"{len(df)} particles found")
    
    xyz = df[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].to_numpy()
    print("got binned origin in pixel from 'rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ'")
    
    # Read pixel spacing from star file
    try:
        pixel_spacing = df['rlnImagePixelSize'].to_numpy()[:, np.newaxis]  # Shape: (b, 1)
        print("Got pixel spacing from 'rlnImagePixelSize'")
        
        tomopixel_spacing = df['rlnTomoTiltSeriesPixelSize'].to_numpy()[:, np.newaxis]  # Shape: (b, 1)
        print("Got pixel spacing from 'rlnTomoTiltSeriesPixelSize'")
        
        new_origins = xyz * pixel_spacing / tomopixel_spacing
    except KeyError:
        # Use provided angpix value if specified
        if angpix is not None:
            print(f"Using provided angpix value: {angpix}")
            new_origins = xyz * angpix
        else:
            print("ERROR! Missing pixel size columns and no --angpix provided. Cannot proceed.")
            sys.exit(1)

    print('calculated particle position in unbinned tomogram')
    
    star['particles'][['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']] = new_origins
    print("updated shift values in 'rlnCoordinateX','rlnCoordinateY', 'rlnCoordinateZ'")
    
    # Remove .tomostar
    star['particles']['rlnTomoName'] = star['particles']['rlnTomoName'].str.replace(r'\.tomostar$', '', regex=True).str.replace(r'_TS(\d+)$', r'_\1', regex=True)
    print("Remove .tomostar in 'rlnTomoName'")

    # write output
    starfile.write(star, output_star_file)
    print(f"Output with ArtiaX compatible written to {output_star_file}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Modify Relion Warp file to ArtiaX format')
    parser.add_argument('input_star_file', help='Input star file')
    parser.add_argument('--angpix', type=float, help='Pixel size in Angstroms (optional)')
    args = parser.parse_args()
    
    print("Modifying Relion Warp file to ArtiaX format")
    print(" - Remove .tomostar from tomogram name")
    print(" - Replace _TSxxx as _xxx")
    print(" - Convert coordinate to Relion 4 format")
    
    input_star_file = args.input_star_file
    base_name, ext = os.path.splitext(input_star_file)
    output_star_file = f"{base_name}_artiaX{ext}"
    
    modify_star(input_star_file, output_star_file, args.angpix)