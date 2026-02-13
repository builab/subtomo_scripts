#!/usr/bin/python
"""
# Script to identify distance outliers to previous and next from tomogram (with rlnHelicalTubeID)
# Builab@McGill, 2026

# Find protomers with >100Å gaps on both sides
python find_distance_protomers.py --input particles.star --output results_distance --prev_distance 100 --next_distance 100

# Find 8nm (~80Å) gaps
python find_distance_protomers.py --input particles.star --output results_distance --prev_distance 80 --next_distance 80

# Short form
python find_distance_protomers.py -i input.star -o results_distance -p 100 -n 100

# Sort by CoordY of protomer center (option for some data the rlnOriginalIndex is not correct)
python find_distance_protomers.py --input particles.star --output results_distance --prev_distance 100 --next_distance 100 --sort_y

# Analyse selected tomograms
python find_distance_protomers.py -i in.star -o results_distance -p 100 -n 100 --tomo "TS_004,TS_005"
"""

import starfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path
from scipy.stats import gaussian_kde

def read_star_file(star_path):
    """Read Relion star file and compute coordinates."""
    star_data = starfile.read(star_path)
    
    if isinstance(star_data, dict):
        optics = star_data['optics']
        data = star_data['particles'].copy()  # Make a copy to avoid modifying original
        if 'rlnOpticsGroup' in data.columns:
            data = data.merge(optics[['rlnOpticsGroup', 'rlnImagePixelSize']], 
                            on='rlnOpticsGroup', how='left')
        else:
            data['rlnImagePixelSize'] = optics['rlnImagePixelSize'].iloc[0]
    else:
        data = star_data.copy()  # Make a copy to avoid modifying original
        if 'rlnImagePixelSize' not in data.columns:
            raise ValueError("No rlnImagePixelSize found in data or optics block")
    
    angpix = data['rlnImagePixelSize']
    data['CoordX'] = data['rlnCoordinateX'] * angpix - data['rlnOriginXAngst']
    data['CoordY'] = data['rlnCoordinateY'] * angpix - data['rlnOriginYAngst']
    data['CoordZ'] = data['rlnCoordinateZ'] * angpix - data['rlnOriginZAngst']
    
    return data

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two 3D points."""
    return np.sqrt(np.sum((point1 - point2)**2))

def find_distant_protomers(data, star_file_path, output_dir, prev_distance, next_distance, sort_by_y=False):
    """Find protomers that are far from both previous and next protomers."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read original star file to preserve exact structure
    star_data_original = starfile.read(star_file_path)
    if isinstance(star_data_original, dict):
        original_particles = star_data_original['particles'].copy()
    else:
        original_particles = star_data_original.copy()
    
    has_helical = 'rlnHelicalTubeID' in data.columns
    group_cols = ['rlnTomoName']
    if has_helical:
        group_cols.append('rlnHelicalTubeID')
    
    selected_indices = []  # Store row indices instead of particle dataframes
    tomo_distances = {}
    all_consecutive_records = [] 
    all_distances_flat = [] 
    
    grouped = data.groupby(group_cols)
    
    for group_name, group_data in grouped:
        unique_indices = group_data['rlnOriginalIndex'].unique()
        index_to_particles = {}
        index_info = [] 
        
        for orig_idx in unique_indices:
            protomer_particles = group_data[group_data['rlnOriginalIndex'] == orig_idx]
            center = np.array([protomer_particles['CoordX'].mean(), 
                               protomer_particles['CoordY'].mean(), 
                               protomer_particles['CoordZ'].mean()])
            index_to_particles[orig_idx] = protomer_particles.index.tolist()  # Store indices
            index_info.append((orig_idx, center, center[1]))
        
        if sort_by_y:
            index_info.sort(key=lambda x: x[2])
        else:
            index_info.sort(key=lambda x: x[0])
        
        unique_indices = [info[0] for info in index_info]
        centers = [info[1] for info in index_info]
        
        if isinstance(group_name, tuple):
            t_name = str(group_name[0])
            tube_id = str(group_name[1]) if len(group_name) > 1 else "N/A"
            storage_key = f"{t_name}_tube{tube_id}" if len(group_name) > 1 else t_name
        else:
            t_name = str(group_name)
            tube_id = "N/A"
            storage_key = t_name

        consecutive_distances = []
        for i in range(len(centers) - 1):
            dist = calculate_distance(centers[i], centers[i+1])
            consecutive_distances.append(dist)
            all_distances_flat.append(dist)
            all_consecutive_records.append({
                'rlnTomoName': t_name,
                'rlnHelicalTubeID': tube_id,
                'Protomer_A_Index': unique_indices[i],
                'Protomer_B_Index': unique_indices[i+1],
                'Distance_Angstrom': dist
            })
        
        tomo_distances[storage_key] = consecutive_distances
        
        n_protomers = len(unique_indices)
        for i, orig_idx in enumerate(unique_indices):
            is_selected = False
            if n_protomers <= 1: continue
            if i == 0:
                if consecutive_distances[i] > next_distance: is_selected = True
            elif i == n_protomers - 1:
                if consecutive_distances[i-1] > prev_distance: is_selected = True
            else:
                if consecutive_distances[i-1] > prev_distance and consecutive_distances[i] > next_distance:
                    is_selected = True
            
            if is_selected:
                selected_indices.extend(index_to_particles[orig_idx])

    # Save Distances CSV
    if all_consecutive_records:
        csv_path = output_dir / "all_consecutive_distances.csv"
        pd.DataFrame(all_consecutive_records).to_csv(csv_path, index=False)
        print(f"Consecutive distances saved to: {csv_path}")

    # Per-Tomogram Plots
    for storage_key, distances in tomo_distances.items():
        if not distances: continue
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.hist(distances, bins=30, edgecolor='black', color='skyblue', alpha=0.7)
        ax.axvline(prev_distance, color='red', linestyle='--', label=f'Threshold: {prev_distance}Å')
        ax.set_title(f'Distances - {storage_key}')
        ax.set_xlabel('Distance (Å)')
        ax.legend()
        safe_name = storage_key.replace(" ", "_").replace("'", "").replace("(", "").replace(")", "").replace(",", "")
        plt.savefig(output_dir / f'distance_histogram_{safe_name}.png', dpi=150)
        plt.close()

    # SUMMARY PLOT (Density + KDE)
    if all_distances_flat:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # 5Å bins from 0 to 500
        bin_edges = np.arange(0, 505, 5)
        
        # Plot Density Histogram
        ax.hist(all_distances_flat, bins=bin_edges, edgecolor='black', color='forestgreen', alpha=0.4, density=True, label='Data (5Å bins)')
        
        # KDE Curve
        try:
            kde = gaussian_kde(all_distances_flat)
            x_range = np.linspace(0, 500, 1000)
            ax.plot(x_range, kde(x_range), color='darkgreen', lw=3, label='KDE (Smooth Density)')
        except:
            print("Warning: Could not compute KDE.")

        ax.axvline(prev_distance, color='red', linestyle='--', linewidth=2, label=f'Threshold: {prev_distance}Å')
        ax.set_title(f'Summary of All Consecutive Distances (N={len(all_distances_flat)})')
        ax.set_xlabel('Distance (Å)')
        ax.set_ylabel('Probability Density')
        ax.set_xlim(0, 500)
        ax.legend()
        
        summary_plot_path = output_dir / "distance_summary_kde_histogram.png"
        plt.savefig(summary_plot_path, dpi=300)
        plt.close()
        print(f"Summary KDE plot saved to: {summary_plot_path}")

    # Save Output Star file with ORIGINAL values preserved
    star_output_path = output_dir / "distance_protomer.star"
    if selected_indices:
        # Select particles from ORIGINAL dataframe using the indices
        all_selected = original_particles.loc[selected_indices].copy()
        
        # Write output preserving original structure
        if isinstance(star_data_original, dict):
            output_data = {
                'optics': star_data_original['optics'],
                'particles': all_selected
            }
        else:
            output_data = all_selected
            
        starfile.write(output_data, star_output_path, overwrite=True)
        print(f"Output star file saved to: {star_output_path}")
        print(f"Total particles selected: {len(all_selected)}")
    else:
        print("No distant protomers found.")

def main():
    parser = argparse.ArgumentParser(description='Find distant protomers and generate distance reports')
    parser.add_argument('--input', '-i', required=True, help='Input star file')
    parser.add_argument('--output', '-o', required=True, help='Output folder to save all results')
    parser.add_argument('--prev_distance', '-p', type=float, required=True, help='Distance threshold to previous')
    parser.add_argument('--next_distance', '-n', type=float, required=True, help='Distance threshold to next')
    parser.add_argument('--sort_y', action='store_true', help='Sort by Y coordinate')
    parser.add_argument('--tomo', type=str, help='Filter by specific tomograms (comma-separated)')
    
    args = parser.parse_args()
    data = read_star_file(args.input)
    
    if args.tomo:
        tomo_list = [t.strip() for t in args.tomo.split(',')]
        data = data[data['rlnTomoName'].isin(tomo_list)]

    find_distant_protomers(data, args.input, args.output, args.prev_distance, args.next_distance, args.sort_y)

if __name__ == "__main__":
    main()