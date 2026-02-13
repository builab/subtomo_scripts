#!/usr/bin/python
"""
# Script to compare multiple CSV files with Histogram + KDE
# Usage: python plot_psi_compare.py -i file1.csv file2.csv -c Relative_Psi --xlim="-60,60" -o psi_intact_vs_damaged.pdf
# Usage: python plot_psi_compare.py -i file1.csv file2.csv -c Distance_Angstrom --xlim="0,500" --ylim 0.2 --bin_width 10 -o output.pdf
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import argparse
import os
from pathlib import Path

def plot_comparison(csv_paths, column_name, output_path, xmin, xmax, bin_width=1, ymax=None):
    plt.figure(figsize=(12, 8))
    
    # Define bin edges based on the requested x-limits and bin width
    # We use np.arange to ensure bins have the exact width specified
    bin_edges = np.arange(xmin, xmax + bin_width, bin_width)
    
    # X-axis range for KDE evaluation
    x_range = np.linspace(xmin, xmax, 1000)
    
    plotted_count = 0
    # Manual color cycling using standard matplotlib colors
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    for i, csv_path in enumerate(csv_paths):
        if not os.path.exists(csv_path):
            print(f"Warning: File not found: {csv_path}")
            continue
            
        df = pd.read_csv(csv_path)
        if column_name not in df.columns:
            print(f"Warning: Column '{column_name}' not found in {csv_path}.")
            continue

        # Force conversion to 1D numpy array and clean NaNs
        data = df[column_name].to_numpy().flatten()
        data = data[~np.isnan(data)]
        
        if len(data) == 0:
            continue

        label_name = Path(csv_path).stem
        color = colors[i % len(colors)]
        
        # 1. Plot Histogram using Matplotlib
        plt.hist(
            data, 
            bins=bin_edges, 
            density=True, 
            alpha=0.3, 
            color=color,
            label=f"{label_name} (Hist)",
            range=(xmin, xmax)
        )
        
        # 2. Plot KDE using SciPy
        try:
            kde_func = gaussian_kde(data)
            # Adjust bandwidth for smoothness (mimics sns.kdeplot bw_adjust=0.5)
            kde_func.set_bandwidth(bw_method=kde_func.factor * 0.5)
            
            plt.plot(
                x_range, 
                kde_func(x_range), 
                color=color, 
                lw=2.5, 
                label=f"{label_name} (KDE)"
            )
        except Exception as e:
            print(f"Warning: Could not compute KDE for {label_name}: {e}")
        
        plotted_count += 1

    if plotted_count == 0:
        print("Error: No data was plotted.")
        return

    # Formatting and Constraints
    plt.axvline(0, color='black', linestyle='-', linewidth=1.2, alpha=0.6)
    plt.xlim(xmin, xmax)
    
    # Apply Y-limit if provided
    if ymax is not None:
        plt.ylim(0, ymax)
        
    plt.title(f'Comparison of {column_name}', fontsize=16)
    plt.xlabel(f'{column_name} (units)', fontsize=13)
    plt.ylabel('Probability Density', fontsize=13)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    # Save logic (supports .pdf, .svg, .png, etc.)
    plt.savefig(output_path)
    print(f"Successfully saved combined plot to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Plot Histogram + KDE comparison of multiple CSV files')
    parser.add_argument('--input', '-i', nargs='+', required=True, help='List of CSV files')
    parser.add_argument('--column', '-c', default='Relative_Psi', help='Column name to plot')
    parser.add_argument('--output', '-o', default='comparison_plot.pdf', help='Output filename')
    parser.add_argument('--xlim', default='-60,60', help='X-axis limits as min,max (default: -60,60)')
    parser.add_argument('--ylim', type=float, help='Optional: Max Y-axis limit for density')
    parser.add_argument('--bin_width', type=float, default=1, help='Width of each bin in units (default: 1)')

    args = parser.parse_args()
    
    # Parse xlim string safely
    try:
        xmin, xmax = map(float, args.xlim.split(','))
    except ValueError:
        print("Error: --xlim must be in the format min,max (e.g., --xlim=\"-60,60\")")
        return

    plot_comparison(args.input, args.column, args.output, xmin, xmax, args.bin_width, args.ylim)

if __name__ == "__main__":
    main()