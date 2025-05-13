#!/usr/bin/python
# Script to generate star file for the visualization in ArtiaX from Relion WarpTool 2.0.0
# WarpTool tm file use fraction coordinate and AutoPickMerit
# Eliminate .tomostar from rlnTomoName
# Also, deal with no optics header now, just need --angpix
# Still work in progress
# HB, McGill 2025/05


import numpy as np
import sys, os, re
import starfile
import argparse


def modify_warptm_star(input_star_file, output_star_file, tomo_size, angpix=None, min_score=3):
    """
    Modify Warp TM file to Relion 4
    For WarpTM, there is no rlnTomoName, only rlnMicrographName
    """
    star = starfile.read(input_star_file, always_dict=True)
    print(f"{input_star_file} read")
    
    # Handle case where data is under empty key ''
    if '' in star:
        print("Found legacy STAR format (empty key), renaming to 'particles'")
        star['particles'] = star.pop('')
    
    # Verify we have particles data
    if 'particles' not in star:
        raise ValueError("STAR file must contain either 'particles' table or legacy format (empty key)")

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
    print("Got fractional origin from 'rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ'")
    
    print("Calculate pixel coordinates")
    xyz = xyz * np.array(tomo_size)

    # Read pixel spacing from star file (Unlikely in the case of Warptm File)
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
    
    # Filter particles where 'rlnAutopickFigureOfMerit' > min_score
    if 'particles' in star and 'rlnAutopickFigureOfMerit' in star['particles']:
        star['particles'] = star['particles'][star['particles']['rlnAutopickFigureOfMerit'] > min_score]
    else:
        print("Warning: 'particles' or 'rlnAutopickFigureOfMerit' not found in STAR file")

    # Ensure we have rlnTomoName column (rename if needed)
    if 'rlnMicrographName' in star['particles'] and 'rlnTomoName' not in star['particles']:
        star['particles'] = star['particles'].rename(columns={'rlnMicrographName': 'rlnTomoName'})
        print("Renamed 'rlnMicrographName' to 'rlnTomoName'")

    # Verify we have the required column
    if 'rlnTomoName' not in star['particles']:
        raise ValueError("STAR file lacks both 'rlnTomoName' and 'rlnMicrographName' columns")

    # Remove .tomostar
    star['particles']['rlnTomoName'] = star['particles']['rlnTomoName'].str.replace(r'\.tomostar$', '', regex=True).str.replace(r'_TS(\d+)$', r'_\1', regex=True)
    print("Remove .tomostar in 'rlnTomoName'")

    # write output
    starfile.write(star, output_star_file)
    print(f"Output with ArtiaX compatible written to {output_star_file}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Modify Relion Warp file to ArtiaX format')
    parser.add_argument('input_star_file', help='Input star file')
    parser.add_argument('--angpix', type=float, required=True, help='Pixel size in Angstroms')
    parser.add_argument('--min_score', type=float, required=True, help='Threshold for picking')
    parser.add_argument('--tomo_size', type=int, nargs=3, required=True,
                       metavar=('X', 'Y', 'Z'),
                       help='Tomogram size as three space-separated integers (X Y Z)',
                       default=[1024, 1440, 500])
    
    args = parser.parse_args()
    
    print("Modifying Warp TM file to ArtiaX format")
    print(f"Tomogram size: {args.tomo_size[0]} x {args.tomo_size[1]} x {args.tomo_size[2]}")
    print(f"Angpix: {args.angpix} Å")
    print(f"Min score for picking: {args.min_score} Å")

    print("Operations:")
    print(" - Remove .tomostar from tomogram name")
    print(" - Replace _TSxxx as _xxx")
    print(" - Convert coordinate to Relion 4 format")
    
    input_star_file = args.input_star_file
    base_name, ext = os.path.splitext(input_star_file)
    output_star_file = f"{base_name}_artiaX{ext}"
    
    modify_warptm_star(input_star_file, output_star_file, args.tomo_size, args.angpix, args.min_score)