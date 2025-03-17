#!/usr/bin/python
# Script to generate star file for the visualization in ArtiaX from Relion WarpTool 2.0.0
# Multiply origin with pixelSize
# Eliminate .tomostar from rlnTomoName
# HB, McGill, 2025. Style coming from recenter_3d.py by Alister Burt

import numpy as np
import sys, os
import starfile


def modify_star(input_star_file, output_star_file):
    star = starfile.read(input_star_file, always_dict=True)
    print(f"{input_star_file} read")
    
    if not all(key in star for key in ('particles', 'optics')):
        print("expected RELION 3.1+ style STAR file containing particles and optics blocks")

    df = star['particles'].merge(star['optics'], on='rlnOpticsGroup')
    print("optics table merged")
    print(f"{len(df)} particles found")
    
    xyz = df[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].to_numpy()
    print("got binned origin in pixel from 'rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ'")
    
    # Instead, we can also read the bin value
    pixel_spacing = df['rlnImagePixelSize'].to_numpy()
    pixel_spacing = pixel_spacing[:, np.newaxis]  # Shape: (b, 1)
    print("got pixel spacing from 'rlnImagePixelSize'")
    
    tomopixel_spacing = df['rlnTomoTiltSeriesPixelSize'].to_numpy()
    tomopixel_spacing = tomopixel_spacing[:, np.newaxis]  # Shape: (b, 1)
    print("got pixel spacing from 'rlnTomoTiltseriesImagePixelSize'")
    
    new_origins = xyz * pixel_spacing / tomopixel_spacing
    print('calculated particle position in unbinned tomogram')
    
    star['particles'][['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']] = new_origins
    print("updated shift values in 'rlnCoordinateX','rlnCoordinateY', 'rlnCoordinateZ'")
    
    # Remove .tomostar
    star['particles']['rlnTomoName'] = star['particles']['rlnTomoName'].str.replace('.tomostar', '', regex=False)
    print("Remove .tomostar in 'rlnTomoName'")

    # write output
    starfile.write(star, output_star_file)
    print(f"Output with ArtiaX compatible written to {output_star_file}")
    
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python relionwarp2ArtiaX.py input_star_file")
        sys.exit(1)
    
    input_star_file = sys.argv[1]
    base_name, ext = os.path.splitext(input_star_file)
    output_star_file = f"{base_name}_artiaX{ext}"
    
    modify_star(input_star_file, output_star_file)

