#!/usr/bin/python
"""
# Script to identify distance outliers between consecutive particles in rings/tomograms
# Builab@McGill, 2026

# Find particles with >100Å gaps on both sides (sorted by Y coordinate)
python find_distance_rings.py --input particles.star --output results_distance --prev_distance 100 --next_distance 100 --sort_y

# Find particles with >100Å gaps on both sides (sorted by rlnTomoParticleId)
python find_distance_rings.py --input particles.star --output results_distance --prev_distance 100 --next_distance 100

# Short form
python find_distance_rings.py -i input.star -o results_distance -p 100 -n 100

# Analyse selected tomograms
python find_distance_rings.py -i in.star -o results_distance -p 100 -n 100 --tomo "TS_004,TS_005"
"""

import starfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from scipy.stats import gaussian_kde

def read_star_file(star_path):
    """Read Relion star file and compute coordinates."""
    star_data = starfile.read(star_path)
    
    if isinstance(star_data, dict):
        optics = star_data['optics']
        data = star_data['particles'].copy()
        if 'rlnOpticsGroup' in data.columns:
            data = data.merge(optics[['rlnOpticsGroup', 'rlnImagePixelSize']], 
                            on='rlnOpticsGroup', how='left')
        else:
            data['rlnImagePixelSize'] = optics['rlnImagePixelSize'].iloc[0]
    else:
        data = star_data.copy()
        if 'rlnImagePixelSize' not in data.columns:
            raise ValueError("No rlnImagePixelSize found in data or optics block")
    
    # Handle missing rlnOriginXAngst columns
    if 'rlnOriginXAngst' not in data.columns:
        data['rlnOriginXAngst'] = 0.0
    if 'rlnOriginYAngst' not in data.columns:
        data['rlnOriginYAngst'] = 0.0
    if 'rlnOriginZAngst' not in data.columns:
        data['rlnOriginZAngst'] = 0.0
    
    angpix = data['rlnImagePixelSize']
    data['CoordX'] = data['rlnCoordinateX'] * angpix - data['rlnOriginXAngst']
    data['CoordY'] = data['rlnCoordinateY'] * angpix - data['rlnOriginYAngst']
    data['CoordZ'] = data['rlnCoordinateZ'] * angpix - data['rlnOriginZAngst']
    
    return data

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two 3D points."""
    return np.sqrt(np.sum((point1 - point2)**2))

def find_distant_particles(data, star_file_path, output_dir, prev_distance, next_distance, sort_by_y=False):
    """Find particles that are far from both previous and next particles."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read original star file to preserve exact structure
    star_data_original = starfile.read(star_file_path)
    if isinstance(star_data_original, dict):
        original_particles = star_data_original['particles'].copy()
    else:
        original_particles = star_data_original.copy()
    
    selected_indices = []
    tomo_distances = {}
    all_consecutive_records = []
    all_distances_flat = []
    
    # Group by tomogram name
    grouped = data.groupby('rlnTomoName')
    
    for tomo_name, group_data in grouped:
        # Sort particles within this tomogram
        if sort_by_y:
            sorted_group = group_data.sort_values('CoordY').reset_index(drop=False)
            print(f"Processing {tomo_name}: sorted by Y coordinate")
        else:
            # Check if rlnTomoParticleId exists
            if 'rlnTomoParticleId' in group_data.columns:
                sorted_group = group_data.sort_values('rlnTomoParticleId').reset_index(drop=False)
                print(f"Processing {tomo_name}: sorted by rlnTomoParticleId")
            else:
                # Fallback to Y coordinate if rlnTomoParticleId doesn't exist
                sorted_group = group_data.sort_values('CoordY').reset_index(drop=False)
                print(f"Processing {tomo_name}: rlnTomoParticleId not found, sorted by Y coordinate")
        
        # Get centers (coordinates) for each particle
        centers = sorted_group[['CoordX', 'CoordY', 'CoordZ']].to_numpy()
        original_indices = sorted_group['index'].tolist()  # Original indices from the dataframe
        
        # Calculate consecutive distances
        consecutive_distances = []
        for i in range(len(centers) - 1):
            dist = calculate_distance(centers[i], centers[i+1])
            consecutive_distances.append(dist)
            all_distances_flat.append(dist)
            all_consecutive_records.append({
                'rlnTomoName': tomo_name,
                'Particle_A_Index': i,
                'Particle_B_Index': i+1,
                'Distance_Angstrom': dist
            })
        
        tomo_distances[tomo_name] = consecutive_distances
        
        # Select particles with large gaps on both sides
        n_particles = len(centers)
        for i in range(n_particles):
            is_selected = False
            if n_particles <= 1:
                continue
            if i == 0:
                # First particle: check only next distance
                if consecutive_distances[i] > next_distance:
                    is_selected = True
            elif i == n_particles - 1:
                # Last particle: check only previous distance
                if consecutive_distances[i-1] > prev_distance:
                    is_selected = True
            else:
                # Middle particles: check both sides
                if consecutive_distances[i-1] > prev_distance and consecutive_distances[i] > next_distance:
                    is_selected = True
            
            if is_selected:
                selected_indices.append(original_indices[i])

    # Save Distances CSV
    if all_consecutive_records:
        csv_path = output_dir / "all_consecutive_distances.csv"
        pd.DataFrame(all_consecutive_records).to_csv(csv_path, index=False)
        print(f"Consecutive distances saved to: {csv_path}")

    # Per-Tomogram Plots
    for tomo_name, distances in tomo_distances.items():
        if not distances:
            continue
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.hist(distances, bins=30, edgecolor='black', color='skyblue', alpha=0.7)
        ax.axvline(prev_distance, color='red', linestyle='--', label=f'Threshold: {prev_distance}Å')
        ax.set_title(f'Distances - {tomo_name}')
        ax.set_xlabel('Distance (Å)')
        ax.legend()
        safe_name = tomo_name.replace(" ", "_").replace("'", "").replace("(", "").replace(")", "").replace(",", "")
        plt.savefig(output_dir / f'distance_histogram_{safe_name}.png', dpi=150)
        plt.close()

    # SUMMARY PLOT (Density + KDE)
    if all_distances_flat:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # 5Å bins from 0 to 500
        bin_edges = np.arange(0, 505, 5)
        
        # Plot Density Histogram
        ax.hist(all_distances_flat, bins=bin_edges, edgecolor='black', color='forestgreen', 
                alpha=0.4, density=True, label='Data (5Å bins)')
        
        # KDE Curve
        try:
            kde = gaussian_kde(all_distances_flat)
            x_range = np.linspace(0, 500, 1000)
            ax.plot(x_range, kde(x_range), color='darkgreen', lw=3, label='KDE (Smooth Density)')
        except:
            print("Warning: Could not compute KDE.")

        ax.axvline(prev_distance, color='red', linestyle='--', linewidth=2, 
                   label=f'Threshold: {prev_distance}Å')
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
    star_output_path = output_dir / "distance_particles.star"
    if selected_indices:
        # Select particles from ORIGINAL dataframe using the indices
        all_selected = original_particles.loc[selected_indices].copy()
        
        # Write output preserving original structure and ALL data blocks
        if isinstance(star_data_original, dict):
            output_data = {}
            # Copy all data blocks from original
            for key in star_data_original.keys():
                if key == 'particles':
                    output_data['particles'] = all_selected
                else:
                    output_data[key] = star_data_original[key]
        else:
            output_data = all_selected
            
        starfile.write(output_data, star_output_path, overwrite=True)
        print(f"Output star file saved to: {star_output_path}")
        print(f"Total particles selected: {len(all_selected)}")
    else:
        print("No distant particles found.")

def main():
    parser = argparse.ArgumentParser(description='Find distant particles and generate distance reports')
    parser.add_argument('--input', '-i', required=True, help='Input star file')
    parser.add_argument('--output', '-o', required=True, help='Output folder to save all results')
    parser.add_argument('--prev_distance', '-p', type=float, required=True, 
                       help='Distance threshold to previous particle (Angstroms)')
    parser.add_argument('--next_distance', '-n', type=float, required=True, 
                       help='Distance threshold to next particle (Angstroms)')
    parser.add_argument('--sort_y', action='store_true', 
                       help='Sort by Y coordinate instead of rlnTomoParticleId')
    parser.add_argument('--tomo', type=str, 
                       help='Filter by specific tomograms (comma-separated)')
    
    args = parser.parse_args()
    data = read_star_file(args.input)
    
    if args.tomo:
        tomo_list = [t.strip() for t in args.tomo.split(',')]
        data = data[data['rlnTomoName'].isin(tomo_list)]
        print(f"Filtered to {len(tomo_list)} tomogram(s): {', '.join(tomo_list)}")

    find_distant_particles(data, args.input, args.output, args.prev_distance, 
                          args.next_distance, args.sort_y)

if __name__ == "__main__":
    main()