#!/usr/bin/env python
"""
em2mrc.py - A program to convert EM format files to MRC format
Written by Claude Sonnet 3.7, modified by Huy Bui, McGill 2025
"""

import argparse
import numpy as np
import os
import sys
import mrcfile
import struct

def read_em_header(file_obj):
    """
    Read the header of an EM file
    
    Parameters:
    -----------
    file_obj : file object
        Open file object in binary mode
        
    Returns:
    --------
    header : dict
        Dictionary containing header information
    """
    header = {}
    
    # The EM format has a 512-byte header
    file_obj.seek(0)
    
    # Machine coding (0 = OS-9, 1 = VAX, 2 = Convex, 3 = SGI, 4 = Sun, 5 = Mac, 6 = PC)
    machine_code = struct.unpack('1B', file_obj.read(1))[0]
    header['machine_code'] = machine_code
    
    # General purpose flag (0 = simple 3D volume, 1 = Imperial format)
    flag = struct.unpack('1B', file_obj.read(1))[0]
    header['flag'] = flag
    
    # Data type (1 = byte, 2 = short, 4 = float, 8 = complex)
    data_type = struct.unpack('1B', file_obj.read(1))[0]
    header['data_type'] = data_type
    
    # Map dimensions
    dims = struct.unpack('3B', file_obj.read(3))
    header['x_dim'] = dims[0]
    header['y_dim'] = dims[1]
    header['z_dim'] = dims[2]
    
    # Extended header size (in blocks of 512 bytes)
    ext_header = struct.unpack('1B', file_obj.read(1))[0]
    header['ext_header'] = ext_header
    
    # Try to read voxel size (not always reliable in EM format)
    # Skip to position 40-48 where voxel size might be stored in some EM variations
    file_obj.seek(40)
    try:
        apix_x = struct.unpack('f', file_obj.read(4))[0]
        apix_y = struct.unpack('f', file_obj.read(4))[0]
        apix_z = struct.unpack('f', file_obj.read(4))[0]
        
        # Check if the values make sense (sometimes they don't)
        if 0.1 < apix_x < 100 and 0.1 < apix_y < 100 and 0.1 < apix_z < 100:
            header['apix_x'] = apix_x
            header['apix_y'] = apix_y
            header['apix_z'] = apix_z
        else:
            header['apix_x'] = header['apix_y'] = header['apix_z'] = 1.0
    except:
        header['apix_x'] = header['apix_y'] = header['apix_z'] = 1.0
    
    return header

def read_em_data(file_obj, header):
    """
    Read data from an EM file based on header information
    
    Parameters:
    -----------
    file_obj : file object
        Open file object in binary mode
    header : dict
        Dictionary containing header information
        
    Returns:
    --------
    data : numpy.ndarray
        3D volume data
    """
    # Position at the start of the data (after header + extended header)
    file_obj.seek(512 + 512 * header['ext_header'])
    
    # Determine data type and shape
    if header['data_type'] == 1:
        dtype = np.int8
    elif header['data_type'] == 2:
        dtype = np.int16
    elif header['data_type'] == 4:
        dtype = np.float32
    elif header['data_type'] == 8:
        dtype = np.complex64
    else:
        raise ValueError(f"Unsupported data type: {header['data_type']}")
    
    # Read the data as a flat array
    shape = (header['z_dim'], header['y_dim'], header['x_dim'])
    size = shape[0] * shape[1] * shape[2]
    
    # Read the binary data
    binary_data = file_obj.read(size * np.dtype(dtype).itemsize)
    
    # Convert to numpy array and reshape
    data = np.frombuffer(binary_data, dtype=dtype).reshape(shape)
    
    # Handle machine endianness if needed
    if header['machine_code'] in [1, 2, 3, 4, 5]:  # Non-PC formats
        data = data.byteswap()
    
    return data

def em_to_mrc(em_file, mrc_file, force_voxel_size=None):
    """
    Convert EM file to MRC file
    
    Parameters:
    -----------
    em_file : str
        Path to input EM file
    mrc_file : str
        Path to output MRC file
    force_voxel_size : float or None
        If provided, override the voxel size with this value
        
    Returns:
    --------
    success : bool
        True if conversion was successful
    """
    try:
        # Open EM file and read header
        with open(em_file, 'rb') as f:
            header = read_em_header(f)
            data = read_em_data(f, header)
        
        # Create MRC file and set data
        with mrcfile.new(mrc_file, overwrite=True) as mrc:
            mrc.set_data(data)
            
            # Set voxel size
            if force_voxel_size is not None:
                voxel_size = (float(force_voxel_size), float(force_voxel_size), float(force_voxel_size))
                print(f"Using forced voxel size: {force_voxel_size} Å")
            else:
                voxel_size = (header['apix_x'], header['apix_y'], header['apix_z'])
                print(f"Using voxel size from EM file: {voxel_size[0]}, {voxel_size[1]}, {voxel_size[2]} Å")
            
            mrc.voxel_size = voxel_size
            
            # Add information to the header
            mrc.header.nlabl = 2
            mrc.header.label[0] = f"Converted from EM file: {os.path.basename(em_file)}".encode()
            
            if force_voxel_size is not None:
                mrc.header.label[1] = f"Voxel size set to {force_voxel_size} Å".encode()
            else:
                mrc.header.label[1] = f"Original voxel size: {voxel_size[0]}, {voxel_size[1]}, {voxel_size[2]} Å".encode()
        
        return True
    
    except Exception as e:
        print(f"Error converting file: {e}")
        return False

def main():
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Convert EM format files to MRC format')
    parser.add_argument('--i', required=True, help='Input EM file')
    parser.add_argument('--o', required=True, help='Output MRC file')
    parser.add_argument('--force_header_angpix', type=float, help='Force voxel size (in Ångström)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Print parameters
    print(f"Input EM file: {args.i}")
    print(f"Output MRC file: {args.o}")
    if args.force_header_angpix:
        print(f"Forcing voxel size to: {args.force_header_angpix} Å")
    
    # Check if input file exists
    if not os.path.isfile(args.i):
        print(f"Error: Input file '{args.i}' does not exist")
        return 1
    
    # Perform the conversion
    print("Converting EM file to MRC format...")
    if em_to_mrc(args.i, args.o, args.force_header_angpix):
        print(f"Conversion successful. MRC file saved to: {args.o}")
        return 0
    else:
        print("Conversion failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
