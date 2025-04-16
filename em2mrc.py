#!/usr/bin/env python3
"""
em2mrc.py - Converts EM format files to MRC format using emfile and mrcfile packages

Usage:
    python em2mrc.py --i input.em --o output.mrc [--force_header_angpix 1.0]

Requirements:
    - mrcfile package
    - emfile package
"""

import argparse
import os
import sys
import numpy as np
import mrcfile
import emfile

def convert_em_to_mrc(input_file, output_file, force_header_angpix=None):
    """
    Convert EM format file to MRC format.
    
    Args:
        input_file (str): Path to input EM file
        output_file (str): Path to output MRC file
        force_header_angpix (float, optional): Force voxel size (in Ångström)
    
    Returns:
        bool: True if conversion was successful, False otherwise
    """
    try:
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"Error: Input file '{input_file}' does not exist.")
            return False
        
        # Check input file extension
        if not input_file.lower().endswith('.em'):
            print(f"Warning: Input file '{input_file}' does not have .em extension.")
        
        # Read EM file
        print(f"Reading EM file: {input_file}")
        em_data = emfile.read(input_file)
        
        # Get the data and metadata
        data = em_data.data
        header = em_data.header
        
        # Write MRC file
        print(f"Writing MRC file: {output_file}")
        with mrcfile.new(output_file, overwrite=True) as mrc:
            # Set the data
            mrc.set_data(data)
            
            # Set voxel size
            if force_header_angpix is not None:
                print(f"Forcing voxel size to {force_header_angpix} Å")
                mrc.voxel_size = (force_header_angpix, force_header_angpix, force_header_angpix)
            elif hasattr(header, 'pixel_spacing'):
                mrc.voxel_size = header.pixel_spacing
            
            # Set mode based on data type
            if data.dtype == np.float32:
                mrc.header.mode = 2  # 32-bit float
            elif data.dtype == np.int16:
                mrc.header.mode = 1  # 16-bit integer
            elif data.dtype == np.int8:
                mrc.header.mode = 0  # 8-bit integer
                
        print(f"Conversion successful: {input_file} → {output_file}")
        return True
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        return False

def main():
    """Main function to parse arguments and run conversion"""
    parser = argparse.ArgumentParser(description='Convert EM format to MRC format')
    parser.add_argument('--i', required=True, help='Input EM file')
    parser.add_argument('--o', required=True, help='Output MRC file')
    parser.add_argument('--force_header_angpix', type=float, help='Force voxel size (in Ångström)')
    
    args = parser.parse_args()
    
    # Check if output directory exists
    output_dir = os.path.dirname(args.o)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Perform conversion
    success = convert_em_to_mrc(args.i, args.o, args.force_header_angpix)
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
