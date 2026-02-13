#!/usr/bin/python
"""
# Script to compare Psi distributions using a focused Polar Sector
# Optimized for narrow ranges (e.g., -20 to 20) and varying densities
# Usage: python plot_polar_focus.py -i f1.csv f2.csv --xlim="-30,30" -o focused_polar.svg
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import argparse
import os
from pathlib import Path

def plot_focused_polar(csv_paths, column_name, output_path, xmin, xmax, bin_width=1):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='polar')

    xmin_rad, xmax_rad = np.deg2rad(xmin), np.deg2rad(xmax)
    
    # Configure the sector view
    ax.set_thetamin(xmin) 
    ax.set_thetamax(xmax) 
    ax.set_theta_zero_location("N") 
    ax.set_theta_direction(-1)     

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    x_range_rad = np.linspace(xmin_rad, xmax_rad, 1000)

    for i, csv_path in enumerate(csv_paths):
        if not os.path.exists(csv_path):
            continue
            
        df = pd.read_csv(csv_path)
        if column_name not in df.columns:
            continue

        data_deg = df[column_name].to_numpy().flatten()
        data_deg = data_deg[~np.isnan(data_deg)]
        
        # Filter data to the plot range
        mask = (data_deg >= xmin) & (data_deg <= xmax)
        data_rad = np.deg2rad(data_deg[mask])
        
        if len(data_rad) == 0:
            continue

        label_name = Path(csv_path).stem
        color = colors[i % len(colors)]
        
        # 1. Histogram Wedges (Normalized to density)
        rad_bins = np.deg2rad(np.arange(xmin, xmax + bin_width, bin_width))
        counts, edges = np.histogram(data_rad, bins=rad_bins, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        width = np.deg2rad(bin_width)
        
        # Using step-like outlines for the histogram to keep the plot clean
        ax.bar(centers, counts, width=width, color=color, alpha=0.15, 
               edgecolor=color, linewidth=0.5, label=f"{label_name}")
        
        # 2. KDE Line + Fill
        try:
            kde_func = gaussian_kde(data_rad)
            # Bandwidth factor 0.5 for a tighter, more descriptive fit
            kde_func.set_bandwidth(bw_method=kde_func.factor * 0.5)
            y_kde = kde_func(x_range_rad)
            
            # Plot the main KDE line
            ax.plot(x_range_rad, y_kde, color=color, lw=2)
            # Fill under the KDE curve for better visual comparison
            ax.fill(x_range_rad, y_kde, color=color, alpha=0.2)
        except Exception as e:
            print(f"KDE failed for {label_name}: {e}")

    # Add a dashed reference line at 0 degrees
    ax.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Center (0°)')
    
    ax.set_title(f'Angular Distribution: {column_name}\nRange: [{xmin}°, {xmax}°]', fontsize=14, pad=20)
    # Move legend so it doesn't obstruct the sector
    ax.legend(loc='lower right', bbox_to_anchor=(1.3, 0.1), frameon=True)
    
    # Clean up radial labels (density) if they overlap
    ax.set_rlabel_position(xmax + 5) 
    ax.grid(True, alpha=0.3)

    plt.savefig(output_path, format='svg', bbox_inches='tight')
    print(f"Saved focused polar plot to: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', nargs='+', required=True)
    parser.add_argument('--column', '-c', default='Relative_Psi')
    parser.add_argument('--output', '-o', default='focused_polar.svg')
    parser.add_argument('--xlim', default='-25,25')
    parser.add_argument('--bin_width', type=float, default=1)

    args = parser.parse_args()
    
    try:
        xmin, xmax = map(float, args.xlim.split(','))
    except:
        print("Error: Use --xlim=\"min,max\"")
        return

    plot_focused_polar(args.input, args.column, args.output, xmin, xmax, args.bin_width)

if __name__ == "__main__":
    main()