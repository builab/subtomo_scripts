#!/usr/bin/python
"""
# Script to identify outlier psi from tomogram (with rlnHelicalTubeID)
# Builab@McGill, 2026

# Basic usage
python psi_analysis_tube.py --input particles.star --output results

# With custom threshold
python psi_analysis_tube.py --input particles.star --output results --threshold 7

# Short form
python psi_analysis_tube.py -i particles.star -o results -t 15

# Short form
python psi_analysis_tube.py -i particles.star -o results -t 15 --tomo "TS_001,TS_004"

"""

import starfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from scipy.stats import gaussian_kde
from pathlib import Path

def read_star_file(star_path):
    """Read Relion star file and compute coordinates."""
    print(f"--- Reading: {star_path} ---")
    star_data = starfile.read(star_path)
    
    if isinstance(star_data, dict):
        optics = star_data['optics']
        data = star_data['particles']
        if 'rlnOpticsGroup' in data.columns:
            data = data.merge(optics[['rlnOpticsGroup', 'rlnImagePixelSize']], 
                            on='rlnOpticsGroup', how='left')
        else:
            data['rlnImagePixelSize'] = optics['rlnImagePixelSize'].iloc[0]
    else:
        data = star_data
    
    angpix = data.get('rlnImagePixelSize', 1.0)
    data['CoordX'] = data['rlnCoordinateX'] * angpix - data['rlnOriginXAngst']
    data['CoordY'] = data['rlnCoordinateY'] * angpix - data['rlnOriginYAngst']
    data['CoordZ'] = data['rlnCoordinateZ'] * angpix - data['rlnOriginZAngst']
    
    print(f"Loaded {len(data)} particles.")
    return data

def analyze_psi_angles(data, star_file_path, output_folder, deviation_threshold=10):
    """Analyze psi angles using relative deviations and KDE plotting."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    has_helical = 'rlnHelicalTubeID' in data.columns
    group_cols = ['rlnTomoName', 'rlnHelicalTubeID'] if has_helical else ['rlnTomoName']
    
    all_processed_data = []
    outlier_particles = []
    
    # 1-degree bins from -60 to 60
    bin_edges = np.arange(-60, 61, 1)
    
    grouped = data.groupby(group_cols)
    print(f"Processing {len(grouped)} groups...")

    for group_name, group_data in grouped:
        if has_helical:
            tomo_name, tube_id = group_name
            storage_key = f"{tomo_name}_tube{tube_id}"
        else:
            # Fix: extract from 1-tuple if necessary to avoid ('Name',) in filename
            tomo_name = group_name[0] if isinstance(group_name, (tuple, list)) else group_name
            storage_key = str(tomo_name)
        
        safe_key = storage_key.replace('/', '_').replace(' ', '_').replace("'", "").replace("(", "").replace(")", "").replace(",", "")
        
        psi_angles = group_data['rlnAnglePsi'].values
        rad_psi = np.deg2rad(psi_angles)
        median_psi = np.rad2deg(np.arctan2(np.mean(np.sin(rad_psi)), np.mean(np.cos(rad_psi)))) % 360
        
        rel_psi = (psi_angles - median_psi + 180) % 360 - 180
        deviations = np.abs(rel_psi)
        
        group_df = group_data.copy()
        group_df['Local_Median_Psi'] = median_psi
        group_df['Relative_Psi'] = rel_psi
        all_processed_data.append(group_df)
        
        mask = deviations >= deviation_threshold
        if mask.any():
            out_rows = group_df[mask].copy()
            out_rows['Deviation'] = deviations[mask]
            outlier_particles.append(out_rows)
        
        # Individual Plots (Relative)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(rel_psi, bins=bin_edges, edgecolor='black', alpha=0.7, color='steelblue', density=True)
        ax.set_xlim(-60, 60)
        ax.set_title(f"Relative Psi: {storage_key}")
        ax.set_xlabel('Deviation from Median (degrees)')
        plt.savefig(output_folder / f"psi_rel_histogram_{safe_key}.png", dpi=100)
        plt.close()

    if all_processed_data:
        final_df = pd.concat(all_processed_data, ignore_index=True)
        final_df.to_csv(output_folder / "all_psi_data_relative.csv", index=False)
        
        # SUMMARY PLOT with KDE ONLY
        psi_values = final_df['Relative_Psi'].values
        
        fig, ax = plt.subplots(figsize=(12, 7))
        # 1. Histogram
        ax.hist(psi_values, bins=bin_edges, edgecolor='black', color='lightgreen', alpha=0.5, density=True, label='Data Histogram (1° bins)')
        
        # 2. KDE Curve
        try:
            kde = gaussian_kde(psi_values)
            x_range = np.linspace(-60, 60, 1000)
            ax.plot(x_range, kde(x_range), color='darkgreen', lw=3, label='KDE (Smooth Density)')
        except np.linalg.LinAlgError:
            print("Warning: Could not compute KDE (possibly too few unique values).")

        ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax.set_xlim(-60, 60)
        ax.set_title(f'Summary of Relative Psi Deviations (N={len(final_df)})')
        ax.set_xlabel('Relative Psi Deviation (degrees)')
        ax.set_ylabel('Probability Density')
        ax.legend()
        
        plt.savefig(output_folder / "psi_summary_kde_histogram.png", dpi=300)
        plt.close()
        print(f"Generated summary plot: {output_folder / 'psi_summary_kde_histogram.png'}")

    if outlier_particles:
        all_outliers = pd.concat(outlier_particles, ignore_index=True)
        star_data_orig = starfile.read(star_file_path)
        to_drop = ['CoordX', 'CoordY', 'CoordZ', 'Deviation', 'Local_Median_Psi', 'Relative_Psi', 'rlnImagePixelSize']
        particles_to_save = all_outliers.drop(columns=to_drop, errors='ignore')
        
        out_star = output_folder / "outlier_particles.star"
        star_to_write = {'optics': star_data_orig['optics'], 'particles': particles_to_save} if isinstance(star_data_orig, dict) else particles_to_save
        starfile.write(star_to_write, out_star, overwrite=True)
        print(f"Saved {len(all_outliers)} outliers to: {out_star}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True)
    parser.add_argument('--output', '-o', required=True)
    parser.add_argument('--threshold', '-t', type=float, default=10)
    parser.add_argument('--tomo', type=str)
    
    args = parser.parse_args()
    data = read_star_file(args.input)
    
    if args.tomo:
        tomo_list = [t.strip() for t in args.tomo.split(',')]
        mask = data['rlnTomoName'].apply(lambda x: any(t in str(x) for t in tomo_list))
        data = data[mask]
        print(f"Filter active: {len(data)} particles match the tomogram list.")
        
    if data.empty:
        print("Error: No particles found. Check your --tomo values or input file.")
        return

    analyze_psi_angles(data, args.input, args.output, args.threshold)

if __name__ == "__main__":
    main()