#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_info_from_mdocs.py
Tracking the mean intensity to see if there is a spike in intensity
Usage: plot_info_from_mdocs.py --i mdoc_dir --mrc_dir path

@author: Huy Bui, McGill
"""
import os
import re
import glob
import argparse
import mrcfile
import numpy as np
from scipy.stats import median_abs_deviation
import matplotlib.pyplot as plt

def parse_mdoc_for_closest_to_zero(mdoc_content):
    blocks = re.split(r'\[ZValue = \d+\]', mdoc_content)
    blocks = blocks[1:]
    
    closest_block = None
    closest_tilt = float('inf')
    
    for block in blocks:
        tilt_match = re.search(r'TiltAngle = ([-+]?\d*\.\d+|\d+)', block)
        if not tilt_match:
            continue
            
        tilt = float(tilt_match.group(1))
        abs_tilt = abs(tilt)
        
        if abs_tilt < closest_tilt:
            closest_tilt = abs_tilt
            closest_block = block
    
    if not closest_block:
        return None
    
    result = {}
    
    subframe_match = re.search(r'SubFramePath = (.+)', closest_block)
    if subframe_match:
        path = subframe_match.group(1).strip()
        basename = os.path.basename(path.replace('\\', '/'))
        result['basename'] = str(os.path.splitext(basename)[0])
    
    spot_match = re.search(r'SpotSize = (\d+)', closest_block)
    if spot_match:
        result['SpotSize'] = int(spot_match.group(1))
    
    intensity_match = re.search(r'Intensity = ([-+]?\d*\.\d+|\d+)', closest_block)
    if intensity_match:
        result['Intensity'] = float(intensity_match.group(1))
    
    exposure_match = re.search(r'ExposureTime = ([-+]?\d*\.\d+|\d+)', closest_block)
    if exposure_match:
        result['ExposureTime'] = float(exposure_match.group(1))
    
    return result

def get_robust_mrc_stats(mrc_path):
    try:
        with mrcfile.open(mrc_path, mode='r') as mrc:
            data = mrc.data
            
            # Handle 2D or 3D arrays
            if data.ndim == 3:
                data = data[0]  # Take first slice if 3D
            elif data.ndim > 3:
                raise ValueError(f"Unexpected array dimensions: {data.ndim}")
                
            # Flatten the 2D array for statistics
            flat_data = data.astype(np.float32).flatten()
            
            if np.isnan(flat_data).any() or np.isinf(flat_data).any():
                flat_data = np.nan_to_num(flat_data)
            
            # Robust statistics
            median = float(np.median(flat_data))
            mad = float(median_abs_deviation(flat_data, scale='normal'))
            p25 = float(np.percentile(flat_data, 25))
            p75 = float(np.percentile(flat_data, 75))
            
            return {
                'median': median,
                'mad': mad,
                'p25': p25,
                'p75': p75,
                'valid': True
            }
    except Exception as e:
        print(f"Error processing {mrc_path}: {str(e)}")
        return {
            'median': 0.0,
            'mad': 0.0,
            'p25': 0.0,
            'p75': 0.0,
            'valid': False
        }

def main():
    parser = argparse.ArgumentParser(description='Plot microscope parameters with robust MRC stats')
    parser.add_argument('--i', '--mdoc_dir', dest='mdoc_dir', required=True, 
                       help='Directory containing mdoc files')
    parser.add_argument('--mrc_dir', required=True, 
                       help='Directory where mrc files are stored')
    args = parser.parse_args()

    mdoc_files = sorted(glob.glob(os.path.join(args.mdoc_dir, '*.mrc.mdoc')))
    
    if not mdoc_files:
        print("No .mrc.mdoc files found in directory")
        return

    file_numbers = []
    spot_sizes = []
    intensities = []
    exposure_times = []
    mrc_medians = []
    mrc_mads = []
    mrc_p25s = []
    mrc_p75s = []
    basenames = []

    for mdoc_file in mdoc_files:
        base = os.path.basename(mdoc_file)
        match = re.search(r'_(\d+)\.mrc\.mdoc$', base)
        if not match:
            continue
            
        file_number = int(match.group(1))
        
        with open(mdoc_file, 'r') as f:
            content = f.read()
        
        data = parse_mdoc_for_closest_to_zero(content)
        if not data or 'basename' not in data:
            continue
            
        mrc_path = os.path.join(args.mrc_dir, f"{data['basename']}.mrc")
        print(f"Processing: {mrc_path}")  # Debug print
        mrc_stats = get_robust_mrc_stats(mrc_path)
        
        if not mrc_stats['valid']:
            continue
            
        file_numbers.append(file_number)
        spot_sizes.append(data.get('SpotSize', 0))
        intensities.append(data.get('Intensity', 0))
        exposure_times.append(data.get('ExposureTime', 0))
        mrc_medians.append(mrc_stats['median'])
        mrc_mads.append(mrc_stats['mad'])
        mrc_p25s.append(mrc_stats['p25'])
        mrc_p75s.append(mrc_stats['p75'])
        basenames.append(data['basename'])
        
        print(f"File {file_number:03d}: {data['basename']}.mrc | "
              f"Median: {mrc_stats['median']:.1f} ± {mrc_stats['mad']:.1f} MAD | "
              f"IQR: {mrc_stats['p25']:.1f}-{mrc_stats['p75']:.1f}")

    if not file_numbers:
        print("No valid data found in any mdoc files")
        return

    # Create the plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Microscope Parameters with Robust MRC Statistics', y=1.02)
    
    # Plot configurations (same as before)
    ax1.plot(file_numbers, spot_sizes, 'b-o')
    ax1.set_ylabel('SpotSize', color='b')
    ax1.grid(True)
    ax1.set_title('Spot Size')
    
    ax2.plot(file_numbers, intensities, 'r-o')
    ax2.set_ylabel('Intensity', color='r')
    ax2.grid(True)
    ax2.set_title('Beam Intensity')
    
    ax3.plot(file_numbers, exposure_times, 'g-o')
    ax3.set_ylabel('ExposureTime (s)', color='g')
    ax3.set_xlabel('File Number')
    ax3.grid(True)
    ax3.set_title('Exposure Time')
    
    ax4.plot(file_numbers, mrc_medians, 'k-o', label='Median')
    ax4.fill_between(file_numbers, mrc_p25s, mrc_p75s, color='gray', alpha=0.3, label='IQR')
    ax4.errorbar(file_numbers, mrc_medians, yerr=mrc_mads, fmt='none', ecolor='purple', capsize=3, label='MAD')
    ax4.set_ylabel('Intensity')
    ax4.set_xlabel('File Number')
    ax4.grid(True)
    ax4.set_title('MRC Intensity (Robust Stats)')
    ax4.legend()
    
    plt.tight_layout()
    
    output_png = os.path.join(args.mdoc_dir, 'microscope_parameters_robust_stats.png')
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_png}")
    plt.show()

if __name__ == '__main__':
    main()