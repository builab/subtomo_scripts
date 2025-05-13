#!/usr/bin/env python3
"""
mrc2em.py - Convert MRC files to EM format

This script converts electron microscopy data from MRC format to EM format,
which is used by some 3D visualization and processing software.
Written by Claude.ai, Tested by Huy Bui, McGill University

Usage:
    python mrc2em.py input.mrc output.em
"""

import sys
import os
import numpy as np
import mrcfile
import emfile
import argparse


def read_mrc(mrc_path):
    """Read an MRC file and return header information and data."""
    with mrcfile.open(mrc_path) as mrc:
        # Extract relevant header information
        nx, ny, nz = mrc.header.nx, mrc.header.ny, mrc.header.nz
        mode = mrc.header.mode
        cella = (mrc.header.cella.x, mrc.header.cella.y, mrc.header.cella.z)
        origin = (mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z)
        
        # Get data
        data = mrc.data.copy()
        
        return {
            'nx': nx,
            'ny': ny,
            'nz': nz,
            'mode': mode,
            'cella': cella,
            'origin': origin,
            'data': data
        }


def write_em(mrc_data, em_path):
    """Write data to EM format using emfile package."""
    data = mrc_data['data']
    mrc_mode = mrc_data['mode']
    
    # Convert data to appropriate type based on MRC mode
    # MRC modes: 0=int8, 1=int16, 2=float32, 3=complex int16, 4=complex float32
    if mrc_mode == 0:  # int8
        data = data.astype(np.int8)
    elif mrc_mode == 1:  # int16
        data = data.astype(np.int16)
    elif mrc_mode == 2:  # float32
        data = data.astype(np.float32)
    else:
        raise ValueError(f"Unsupported MRC mode {mrc_mode} for EM conversion")
    
    # Use emfile to write the EM file
    emfile.write(em_path, data)


def main():
    parser = argparse.ArgumentParser(description='Convert MRC file to EM format.')
    parser.add_argument('input', help='Input MRC file')
    parser.add_argument('output', help='Output EM file')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite output file if it exists')
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        return 1
    
    if os.path.exists(args.output) and not args.overwrite:
        print(f"Error: Output file '{args.output}' already exists. Use --overwrite to force.")
        return 1
    
    try:
        # Read MRC file
        print(f"Reading MRC file: {args.input}")
        mrc_data = read_mrc(args.input)
        
        # Write EM file
        print(f"Writing EM file: {args.output}")
        write_em(mrc_data, args.output)
        
        print("Conversion completed successfully.")
        return 0
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())