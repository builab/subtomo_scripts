#!/usr/bin/env python3
"""
em2mrc.py - Convert EM files to MRC format (float32)

This script converts electron microscopy data from EM format to MRC format,
specifically using float32 (mode 2) for the MRC output. Use --force_header_angpix to write
pixel size.

Written by Claude.ai, Tested by Huy Bui, McGill

Usage:
    python em2mrc.py input.em output.mrc
"""

import sys
import os
import numpy as np
import mrcfile
import emfile
import argparse


def read_em(em_path):
    """Read an EM file and return data."""
    try:
        # The emfile.read function might return different formats
        result = emfile.read(em_path)
        
        # For debugging purposes
        print(f"emfile.read returned type: {type(result)}")
        
        # Case 1: result is a numpy array directly
        if isinstance(result, np.ndarray):
            return result
            
        # Case 2: result is a tuple
        elif isinstance(result, tuple):
            print(f"Tuple length: {len(result)}")
            # Print types of items in tuple for debugging
            for i, item in enumerate(result):
                print(f"  Tuple item {i} type: {type(item)}")
                
            # Try to extract data from the tuple
            if len(result) >= 1:
                # If first element is a dict, try to get data from it
                if isinstance(result[0], dict):
                    if 'data' in result[0]:
                        return extract_numpy_data(result[0]['data'])
                    else:
                        # Try to find any numpy array in the dict
                        for key, value in result[0].items():
                            print(f"  Checking key '{key}' of type {type(value)}")
                            if isinstance(value, np.ndarray):
                                print(f"  Found numpy array in key '{key}'")
                                return value
                # First element might be the data directly
                elif isinstance(result[0], np.ndarray):
                    return result[0]
            
            # If we reach here, try second element if available
            if len(result) >= 2 and isinstance(result[1], np.ndarray):
                return result[1]
                
            # If we still don't have data, look through each element for any dict with data
            for item in result:
                if isinstance(item, dict):
                    # Try to find data in this dict
                    data = extract_data_from_dict(item)
                    if data is not None:
                        return data
            
            raise TypeError(f"Could not find numpy array data in tuple")
            
        # Case 3: result is a dictionary
        elif isinstance(result, dict):
            data = extract_data_from_dict(result)
            if data is not None:
                return data
            raise TypeError(f"Could not find numpy array data in dictionary: {list(result.keys())}")
        
        else:
            raise TypeError(f"Unexpected return type from emfile.read: {type(result)}")
            
    except Exception as e:
        raise RuntimeError(f"Error reading EM file: {e}")
        
def extract_data_from_dict(d):
    """Extract numpy array data from a dictionary."""
    # First try the 'data' key
    if 'data' in d:
        return extract_numpy_data(d['data'])
        
    # Try some common keys that might contain data
    for key in ['volume', 'density', 'map', 'values', 'v', 'array']:
        if key in d and isinstance(d[key], np.ndarray):
            print(f"Found data in key '{key}'")
            return d[key]
    
    # Try to find any numpy array
    for key, value in d.items():
        if isinstance(value, np.ndarray):
            print(f"Found data in key '{key}'")
            return value
            
    # Try to find nested dictionaries
    for key, value in d.items():
        if isinstance(value, dict):
            nested_data = extract_data_from_dict(value)
            if nested_data is not None:
                return nested_data
                
    return None
    
def extract_numpy_data(data):
    """Ensure data is a numpy array."""
    if isinstance(data, np.ndarray):
        return data
    elif hasattr(data, 'numpy') and callable(getattr(data, 'numpy')):
        # For things like tensorflow tensors that have a numpy() method
        return data.numpy()
    else:
        # Try conversion
        try:
            return np.array(data)
        except:
            raise TypeError(f"Could not convert {type(data)} to numpy array")


def write_mrc(data, mrc_path, overwrite=False, angpix=None):
    """Write data to MRC format as float32.
    
    Args:
        data: Input numpy array data
        mrc_path: Path to output MRC file
        overwrite: Whether to overwrite existing file
        angpix: Optional pixel size in Angstroms
    """
    # Convert data to float32 as requested
    data_float32 = data.astype(np.float32)
    
    # Write data to MRC file with float32 mode (mode 2)
    try:
        with mrcfile.new(mrc_path, overwrite=overwrite) as mrc:
            mrc.set_data(data_float32)
            
            # Calculate reasonable voxel size if dimensions are available
            if hasattr(data, 'shape') and len(data.shape) == 3:
                nx, ny, nz = data.shape
                
                if angpix is not None:
                    # Set cell dimensions using the provided pixel size
                    mrc.header.cella.x = float(nx * angpix)
                    mrc.header.cella.y = float(ny * angpix)
                    mrc.header.cella.z = float(nz * angpix)
                    
                    # Set the pixel spacing (voxel size)
                    mrc.voxel_size = angpix
                else:
                    # Default: Set cell dimensions based on data shape (1Å per voxel)
                    mrc.header.cella.x = float(nx)
                    mrc.header.cella.y = float(ny)
                    mrc.header.cella.z = float(nz)
                
                # Calculate min, max, mean
                mrc.header.dmin = float(np.min(data_float32))
                mrc.header.dmax = float(np.max(data_float32))
                mrc.header.dmean = float(np.mean(data_float32))
    except Exception as e:
        raise RuntimeError(f"Error writing MRC file: {e}")


def main():
    parser = argparse.ArgumentParser(description='Convert EM file to MRC format (float32).')
    parser.add_argument('input', help='Input EM file')
    parser.add_argument('output', help='Output MRC file')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite output file if it exists')
    parser.add_argument('--force_header_angpix', type=float, help='Force pixel size in Angstroms in the output MRC file')
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        return 1
    
    if os.path.exists(args.output) and not args.overwrite:
        print(f"Error: Output file '{args.output}' already exists. Use --overwrite to force.")
        return 1
    
    try:
        # Read EM file
        print(f"Reading EM file: {args.input}")
        data = read_em(args.input)
        
        # Write MRC file
        print(f"Writing MRC file (float32): {args.output}")
        if args.force_header_angpix:
            print(f"Setting pixel size to {args.force_header_angpix} Å")
        
        write_mrc(data, args.output, args.overwrite, args.force_header_angpix)
        
        print("Conversion completed successfully.")
        return 0
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())