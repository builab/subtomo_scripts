#!/usr/bin/env python
"""
mrc2em.py - A program to convert MRC format files to EM format
Written by Claude Sonnet 3.7, modified by Huy Bui, McGill 2025
NOT YET TESTED
"""

import argparse
import numpy as np
import os
import sys
import mrcfile
import struct

def write_em_header(file_obj, data, voxel_size=None):
    """
    Write the header of an EM file
    
    Parameters:
    -----------
    file_obj : file object
        Open file object in binary mode
    data : numpy.ndarray
        3D volume data
    voxel_size : tuple or None
        Voxel size in Ångströms (x, y, z)
    """
    # Make sure the header is 512 bytes (filled with zeros initially)
    header = bytearray(512)
    
    # Machine coding (6 = PC)
    header[0] = 6
    
    # General purpose flag (0 = simple 3D volume)
    header[1] = 0
    
    # Data type
    if data.dtype == np.int8:
        header[2] = 1
    elif data.dtype == np.int16:
        header[2] = 2
    elif data.dtype == np.float32:
        header[2] = 4
    elif data.dtype == np.complex64 or data.dtype == np.complex128:
        header[2] = 8
    else:
        # Default to float32 for other types
        header[2] = 4
    
    # Map dimensions (Z, Y, X in EM convention)
    header[3] = data.shape[2]  # X dimension
    header[4] = data.shape[1]  # Y dimension
    header[5] = data.shape[0]  # Z dimension
    
    # No extended header
    header[6] = 0
    
    # Voxel size (at offset 40-48 in some EM formats)
    if voxel_size is not None:
        # Pack voxel size as float32 values
        vs_x = struct.pack('f', float(voxel_size[0]))
        vs_y = struct.pack('f', float(voxel_size[1]))
        vs_z = struct.pack('f', float(voxel_size[2]))
        
        # Insert values at appropriate positions
        header[40:44] = vs_x
        header[44:48] = vs_y
        header[48:52] = vs_z
    
    # Write the header
    file_obj.write(header)

def mrc_to_em(mrc_file, em_file, force_voxel_size=None):
    """
    Convert MRC file to EM file
    
    Parameters:
    -----------
    mrc_file : str
        Path to input MRC file
    em_file : str
        Path to output EM file
    force_voxel_size : float or None
        If provided, override the voxel size with this value
        
    Returns:
    --------
    success : bool
        True if conversion was successful
    """
    try:
        # Open MRC file
        with mrcfile.open(mrc_file, mode='r') as mrc:
            # Get data and voxel size
            data = mrc.data
            
            if force_voxel_size is not None:
                voxel_size = (force_voxel_size, force_voxel_size, force_voxel_size)
                print(f"Using forced voxel size: {force_voxel_size} Å")
            else:
                # Use voxel size from MRC file
                voxel_size = mrc.voxel_size
                print(f"Using voxel size from MRC file: {voxel_size.x}, {voxel_size.y}, {voxel_size.z} Å")
        
        # Determine appropriate data type for EM format
        if data.dtype == np.int8 or data.dtype == np.uint8:
            em_data = data.astype(np.int8)
        elif data.dtype == np.int16 or data.dtype == np.uint16:
            em_data = data.astype(np.int16)
        elif data.dtype == np.complex64 or data.dtype == np.complex128:
            em_data = data.astype(np.complex64)
        else:
            # Default to float32 for other types
            em_data = data.astype(np.float32)
        
        # Create EM file
        with open(em_file, 'wb') as f:
            # Write header
            write_em_header(f, em_data, voxel_size)
            
            # Write data
            em_data.tofile(f)
        
        return True
    
    except Exception as e:
        print(f"Error converting file: {e}")
        return False

def main():
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Convert MRC format files to EM format')
    parser.add_argument('--i', required=True, help='Input MRC file')
    parser.add_argument('--o', required=True, help='Output EM file')
    parser.add_argument('--force_header_angpix', type=float, help='Force voxel size (in Ångström)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Print parameters
    print(f"Input MRC file: {args.i}")
    print(f"Output EM file: {args.o}")
    if args.force_header_angpix:
        print(f"Forcing voxel size to: {args.force_header_angpix} Å")
    
    # Check if input file exists
    if not os.path.isfile(args.i):
        print(f"Error: Input file '{args.i}' does not exist")
        return 1
    
    # Perform the conversion
    print("Converting MRC file to EM format...")
    if mrc_to_em(args.i, args.o, args.force_header_angpix):
        print(f"Conversion successful. EM file saved to: {args.o}")
        return 0
    else:
        print("Conversion failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
