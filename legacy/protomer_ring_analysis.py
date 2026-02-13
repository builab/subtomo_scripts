#!/usr/bin/python
"""
Script to analyze the ring elipticity analysis and ring planarity

# Generate radial distribution plots
python protomer_ring_analysis.py --i input.star --tomo "TS_004,TS_005" --plot-radial

# Plot a specific ring (you'll see composite keys in the CSV)
python protomer_ring_analysis.py --i input.star --plot-ring "Trichonympha_039_1.0"

# Generate everything
python protomer_ring_analysis.py --i input.star --tomo "TS_004,TS_005" --plot-radial --plot-ring "Trichonympha_039_1.0"

# Skip 3D interactive, only 2D plots
python protomer_ring_analysis.py --i input.star --no-viz --plot-radial

# Generate all 2D publication plots
python protomer_ring_analysis.py --i input.star --tomo "TS_004,TS_005" --plot-radial --plot-planarity

# Only planarity analysis
python protomer_ring_analysis.py --i input.star --plot-planarity

# Everything together
python protomer_ring_analysis.py --i input.star --tomo "TS_004,TS_005" \
  --plot-radial --plot-planarity --plot-ring "Trichonympha_039_1.0"

"""

import starfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from pathlib import Path
import argparse

class RingAnalyzer:
    def __init__(self, star_file_path, tomo_filter=None):
        """
        Initialize the analyzer with a STAR file.
        
        Parameters:
        -----------
        star_file_path : str
            Path to the STAR file
        tomo_filter : list of str, optional
            List of tomogram names to analyze (e.g., ['TS_004', 'TS_005'])
        """
        self.star_file_path = star_file_path
        self.tomo_filter = tomo_filter
        self.data = None
        self.particles = None
        self.optics = None
        self.general = None
        self.rings = {}
        
    def load_star_file(self):
        """Load and parse the STAR file."""
        print(f"Loading STAR file: {self.star_file_path}")
        self.data = starfile.read(self.star_file_path)
        
        # Extract different blocks
        if 'particles' in self.data:
            self.particles = self.data['particles']
        elif 'data_particles' in self.data:
            self.particles = self.data['data_particles']
        else:
            # If no named block, assume it's the particles dataframe
            self.particles = self.data if isinstance(self.data, pd.DataFrame) else None
            
        if 'optics' in self.data:
            self.optics = self.data['optics']
        elif 'data_optics' in self.data:
            self.optics = self.data['data_optics']
            
        if 'general' in self.data:
            self.general = self.data['general']
        elif 'data_general' in self.data:
            self.general = self.data['data_general']
        
        # Filter by tomograms if specified
        if self.tomo_filter:
            print(f"\nFiltering for tomograms: {', '.join(self.tomo_filter)}")
            self.particles = self.particles[self.particles['rlnTomoName'].isin(self.tomo_filter)]
            print(f"Particles after filtering: {len(self.particles)}")
        
        print(f"Total particles: {len(self.particles)}")
        print(f"Unique tomograms: {self.particles['rlnTomoName'].nunique()}")
        tomos = self.particles['rlnTomoName'].unique()
        print(f"Tomogram names: {', '.join(tomos)}")
        print(f"Unique rings (rlnOriginalIndex): {self.particles['rlnOriginalIndex'].nunique()}")
        
    def calculate_real_positions(self):
        """Calculate real particle positions in Angstroms."""
        print("\nCalculating real particle positions...")
        
        # Get pixel size from optics block
        if self.optics is not None:
            if 'rlnOpticsGroup' in self.particles.columns:
                # Merge pixel size from optics based on optics group
                pixel_size_map = dict(zip(self.optics['rlnOpticsGroup'], 
                                         self.optics['rlnImagePixelSize']))
                self.particles['pixelSize'] = self.particles['rlnOpticsGroup'].map(pixel_size_map)
            else:
                # Use first optics group if no mapping exists
                pixel_size = self.optics['rlnImagePixelSize'].iloc[0]
                self.particles['pixelSize'] = pixel_size
                print(f"Using pixel size from optics: {pixel_size} Å/px")
        else:
            raise ValueError("No optics block found in STAR file")
        
        # Real position = (Coordinate * PixelSize) - Origin
        self.particles['realX'] = (self.particles['rlnCoordinateX'] * 
                                   self.particles['pixelSize'] - 
                                   self.particles['rlnOriginXAngst'])
        self.particles['realY'] = (self.particles['rlnCoordinateY'] * 
                                   self.particles['pixelSize'] - 
                                   self.particles['rlnOriginYAngst'])
        self.particles['realZ'] = (self.particles['rlnCoordinateZ'] * 
                                   self.particles['pixelSize'] - 
                                   self.particles['rlnOriginZAngst'])
        
    def analyze_rings(self):
        """Analyze each ring: center, align, and calculate metrics."""
        print("\nAnalyzing rings...")
        
        # Group by BOTH tomogram name and ring index to avoid conflicts across tomograms
        grouped = self.particles.groupby(['rlnTomoName', 'rlnOriginalIndex'])
        
        skipped_rings = []
        for (tomo_name, ring_id), ring_data in grouped:
            if len(ring_data) != 9:
                skipped_rings.append((tomo_name, ring_id, len(ring_data)))
                continue
            
            # Sort by protomer index
            ring_data = ring_data.sort_values('rlnProtomerIndex')
            
            # Extract positions
            positions = ring_data[['realX', 'realY', 'realZ']].values
            
            # Center the ring
            centroid = positions.mean(axis=0)
            centered_positions = positions - centroid
            
            # Align ring plane to XY plane using PCA
            pca = PCA(n_components=3)
            pca.fit(centered_positions)
            
            # The normal vector is the component with smallest variance (3rd principal component)
            normal = pca.components_[2]
            
            # Create rotation to align normal with Z-axis
            z_axis = np.array([0, 0, 1])
            aligned_positions = self.align_to_z_axis(centered_positions, normal, z_axis)
            
            # Rotate around Z to align first protomer to X-axis
            first_protomer = aligned_positions[0]
            angle_to_x = np.arctan2(first_protomer[1], first_protomer[0])
            rotation_z = Rotation.from_euler('z', -angle_to_x)
            final_positions = rotation_z.apply(aligned_positions)
            
            # Calculate metrics
            metrics = self.calculate_ring_metrics(final_positions, centered_positions, pca)
            
            # Store results with composite key (tomo_name, ring_id)
            composite_key = f"{tomo_name}_{ring_id}"
            self.rings[composite_key] = {
                'original_positions': positions,
                'centered_positions': centered_positions,
                'aligned_positions': final_positions,
                'centroid': centroid,
                'normal': normal,
                'metrics': metrics,
                'protomer_indices': ring_data['rlnProtomerIndex'].values,
                'tomo_name': tomo_name,
                'ring_id': ring_id
            }
        
        if skipped_rings:
            print(f"\nWarning: Skipped {len(skipped_rings)} rings with incomplete protomers:")
            for tomo_name, ring_id, count in skipped_rings[:5]:  # Show first 5
                print(f"  {tomo_name} Ring {ring_id}: {count} particles (expected 9)")
            if len(skipped_rings) > 5:
                print(f"  ... and {len(skipped_rings) - 5} more")
        
        print(f"Analyzed {len(self.rings)} complete rings successfully")
        
    def align_to_z_axis(self, positions, normal, z_axis):
        """Rotate positions to align normal vector with Z-axis."""
        # Normalize vectors
        normal = normal / np.linalg.norm(normal)
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        # Calculate rotation axis and angle
        rotation_axis = np.cross(normal, z_axis)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        
        if rotation_axis_norm < 1e-6:
            # Vectors are already aligned or opposite
            if np.dot(normal, z_axis) < 0:
                # Opposite direction, rotate 180 degrees around any perpendicular axis
                rotation = Rotation.from_euler('x', np.pi)
            else:
                # Already aligned
                return positions
        else:
            rotation_axis = rotation_axis / rotation_axis_norm
            angle = np.arccos(np.clip(np.dot(normal, z_axis), -1.0, 1.0))
            rotation = Rotation.from_rotvec(angle * rotation_axis)
        
        return rotation.apply(positions)
    
    def calculate_ring_metrics(self, aligned_positions, centered_positions, pca):
        """Calculate roundness and planarity metrics."""
        # Radial distances (in XY plane after alignment)
        radii = np.sqrt(aligned_positions[:, 0]**2 + aligned_positions[:, 1]**2)
        mean_radius = radii.mean()
        
        # Roundness metrics
        radii_std = radii.std()
        radii_cv = radii_std / mean_radius if mean_radius > 0 else 0
        
        # Fit circle RMSD
        circle_rmsd = radii_std  # For a perfect circle, all radii should be equal
        
        # Planarity metrics
        z_spread = aligned_positions[:, 2].std()
        z_range = aligned_positions[:, 2].max() - aligned_positions[:, 2].min()
        
        # Eigenvalue-based planarity
        eigenvalues = pca.explained_variance_
        planarity_ratio = eigenvalues[2] / eigenvalues[0] if eigenvalues[0] > 0 else 0
        
        # Distance from best-fit plane (before alignment)
        plane_distances = np.abs(centered_positions @ pca.components_[2])
        plane_rmsd = np.sqrt((plane_distances**2).mean())
        
        return {
            'mean_radius': mean_radius,
            'radius_std': radii_std,
            'radius_cv': radii_cv,
            'circle_rmsd': circle_rmsd,
            'z_spread': z_spread,
            'z_range': z_range,
            'planarity_ratio': planarity_ratio,
            'plane_rmsd': plane_rmsd,
            'radii': radii
        }
    
    def print_summary_statistics(self):
        """Print summary statistics for all rings."""
        print("\n" + "="*60)
        print("RING ANALYSIS SUMMARY")
        print("="*60)
        
        all_metrics = [ring['metrics'] for ring in self.rings.values()]
        
        metrics_df = pd.DataFrame(all_metrics)
        
        print(f"\nTotal rings analyzed: {len(self.rings)}")
        
        # Print per-tomogram statistics
        print("\nRings per tomogram:")
        tomo_counts = pd.Series([ring['tomo_name'] for ring in self.rings.values()]).value_counts()
        for tomo, count in tomo_counts.items():
            print(f"  {tomo}: {count} rings")
        
        print("\nRoundness Metrics (Angstroms):")
        print(f"  Mean Radius: {metrics_df['mean_radius'].mean():.2f} ± {metrics_df['mean_radius'].std():.2f}")
        print(f"  Radius Std:  {metrics_df['radius_std'].mean():.2f} ± {metrics_df['radius_std'].std():.2f}")
        print(f"  Radius CV:   {metrics_df['radius_cv'].mean():.4f} ± {metrics_df['radius_cv'].std():.4f}")
        
        print("\nPlanarity Metrics (Angstroms):")
        print(f"  Z-spread:    {metrics_df['z_spread'].mean():.2f} ± {metrics_df['z_spread'].std():.2f}")
        print(f"  Z-range:     {metrics_df['z_range'].mean():.2f} ± {metrics_df['z_range'].std():.2f}")
        print(f"  Plane RMSD:  {metrics_df['plane_rmsd'].mean():.2f} ± {metrics_df['plane_rmsd'].std():.2f}")
        print(f"  Planarity ratio: {metrics_df['planarity_ratio'].mean():.4f} ± {metrics_df['planarity_ratio'].std():.4f}")
    
    def visualize_radial_distribution(self, save_path=None, dpi=300):
        """
        Create 2D publication-quality plots showing radial distribution and ellipticity.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the figure
        dpi : int
            Resolution for saved figure
        """
        # Collect all radial data
        all_radii = []
        all_angles = []
        tomo_labels = []
        
        for composite_key, ring in self.rings.items():
            positions = ring['aligned_positions']
            radii = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
            angles = np.arctan2(positions[:, 1], positions[:, 0])
            
            all_radii.extend(radii)
            all_angles.extend(angles)
            tomo_labels.extend([ring['tomo_name']] * len(radii))
        
        all_radii = np.array(all_radii)
        all_angles = np.array(all_angles)
        
        # Convert angles to degrees and ensure 0-360 range
        all_angles_deg = np.degrees(all_angles) % 360
        
        # Expected angles for 9 protomers (every 40 degrees)
        expected_angles = np.arange(0, 360, 40)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Radial distribution histogram
        ax1 = axes[0, 0]
        ax1.hist(all_radii, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.axvline(all_radii.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {all_radii.mean():.2f} Å')
        ax1.axvline(all_radii.mean() - all_radii.std(), color='orange', linestyle=':', linewidth=1.5,
                   label=f'±1σ: {all_radii.std():.2f} Å')
        ax1.axvline(all_radii.mean() + all_radii.std(), color='orange', linestyle=':', linewidth=1.5)
        ax1.set_xlabel('Radius (Å)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax1.set_title('Radial Distribution', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 2. Angular distribution (polar histogram)
        ax2 = plt.subplot(2, 2, 2, projection='polar')
        n_bins = 36  # 10 degree bins
        counts, bins = np.histogram(np.radians(all_angles_deg), bins=n_bins, range=(0, 2*np.pi))
        width = 2 * np.pi / n_bins
        bars = ax2.bar(bins[:-1], counts, width=width, alpha=0.7, color='steelblue', edgecolor='black')
        
        # Mark expected protomer positions
        for angle in expected_angles:
            ax2.plot([np.radians(angle), np.radians(angle)], [0, counts.max()], 
                    'r--', linewidth=1.5, alpha=0.7)
        
        ax2.set_title('Angular Distribution\n(9-fold symmetry expected)', 
                     fontsize=12, fontweight='bold', pad=20)
        ax2.set_theta_zero_location('E')
        ax2.set_theta_direction(1)
        
        # 3. Radius vs Angle scatter plot (shows ellipticity)
        ax3 = axes[1, 0]
        scatter = ax3.scatter(all_angles_deg, all_radii, alpha=0.3, s=20, c=all_radii, 
                             cmap='viridis', edgecolors='none')
        
        # Mark expected protomer angles
        for angle in expected_angles:
            ax3.axvline(angle, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        ax3.set_xlabel('Angle (degrees)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Radius (Å)', fontsize=12, fontweight='bold')
        ax3.set_title('Radius vs Angular Position', fontsize=14, fontweight='bold')
        ax3.set_xlim(0, 360)
        ax3.set_xticks(expected_angles)
        ax3.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Radius (Å)', fontsize=10)
        
        # 4. Box plot of radius by protomer position
        ax4 = axes[1, 1]
        
        # Group radii by nearest expected angle
        binned_radii = [[] for _ in range(9)]
        for angle, radius in zip(all_angles_deg, all_radii):
            # Find nearest expected angle
            nearest_idx = np.argmin(np.abs(expected_angles - angle))
            binned_radii[nearest_idx].append(radius)
        
        bp = ax4.boxplot(binned_radii, positions=range(9), widths=0.6,
                         patch_artist=True, showmeans=True,
                         boxprops=dict(facecolor='lightblue', alpha=0.7),
                         medianprops=dict(color='red', linewidth=2),
                         meanprops=dict(marker='D', markerfacecolor='green', markersize=6))
        
        ax4.set_xlabel('Protomer Position', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Radius (Å)', fontsize=12, fontweight='bold')
        ax4.set_title('Radius Distribution per Protomer', fontsize=14, fontweight='bold')
        ax4.set_xticklabels([f'P{i}' for i in range(9)])
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved radial distribution plot to {save_path}")
        
        plt.show()
        
    def visualize_single_ring_2d(self, composite_key, save_path=None, dpi=300):
        """
        Create a publication-quality 2D visualization of a single ring.
        
        Parameters:
        -----------
        composite_key : str
            Composite key of the ring (e.g., "Trichonympha_039_1.0")
        save_path : str, optional
            Path to save the figure
        dpi : int
            Resolution for saved figure
        """
        if composite_key not in self.rings:
            print(f"Ring {composite_key} not found!")
            available = list(self.rings.keys())[:5]
            print(f"Available rings (showing first 5): {available}")
            return
        
        ring = self.rings[composite_key]
        positions = ring['aligned_positions']
        metrics = ring['metrics']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. XY view with fitted circle
        ax1 = axes[0]
        
        # Plot particles
        colors = plt.cm.tab10(np.arange(9))
        for i, pos in enumerate(positions):
            ax1.scatter(pos[0], pos[1], c=[colors[i]], s=200, 
                       edgecolors='black', linewidths=1.5, zorder=3,
                       label=f'P{i}')
            ax1.text(pos[0], pos[1], str(i), ha='center', va='center', 
                    fontsize=10, fontweight='bold', color='white', zorder=4)
        
        # Fitted circle
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = metrics['mean_radius'] * np.cos(theta)
        circle_y = metrics['mean_radius'] * np.sin(theta)
        ax1.plot(circle_x, circle_y, 'r--', linewidth=2, alpha=0.7, 
                label=f'Fitted circle (R={metrics["mean_radius"]:.2f} Å)', zorder=1)
        
        # Individual radii as lines
        for i, pos in enumerate(positions):
            ax1.plot([0, pos[0]], [0, pos[1]], 'gray', alpha=0.3, linewidth=1, zorder=2)
        
        ax1.set_xlabel('X (Å)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Y (Å)', fontsize=12, fontweight='bold')
        ax1.set_title(f'Top View - {ring["tomo_name"]} Ring {ring["ring_id"]}\nCV={metrics["radius_cv"]:.4f}', 
                     fontsize=12, fontweight='bold')
        ax1.axis('equal')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='k', linewidth=0.5, alpha=0.5)
        ax1.axvline(x=0, color='k', linewidth=0.5, alpha=0.5)
        
        # 2. Radial deviation plot
        ax2 = axes[1]
        radii = metrics['radii']
        mean_radius = metrics['mean_radius']
        deviations = radii - mean_radius
        protomer_indices = np.arange(9)
        
        bars = ax2.bar(protomer_indices, deviations, color=colors, 
                      edgecolor='black', linewidth=1.5, alpha=0.7)
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Protomer Index', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Radius Deviation (Å)', fontsize=12, fontweight='bold')
        ax2.set_title(f'Radial Deviation from Mean\nStd={metrics["radius_std"]:.2f} Å', 
                     fontsize=12, fontweight='bold')
        ax2.set_xticks(protomer_indices)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. XZ view (planarity)
        ax3 = axes[2]
        for i, pos in enumerate(positions):
            ax3.scatter(pos[0], pos[2], c=[colors[i]], s=200, 
                       edgecolors='black', linewidths=1.5, zorder=3)
            ax3.text(pos[0], pos[2], str(i), ha='center', va='center', 
                    fontsize=10, fontweight='bold', color='white', zorder=4)
        
        ax3.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, 
                   label='Best-fit plane', zorder=1)
        ax3.fill_between(ax3.get_xlim(), -metrics['z_spread'], metrics['z_spread'], 
                        color='red', alpha=0.1, label=f'±1σ Z-spread ({metrics["z_spread"]:.2f} Å)', zorder=0)
        
        ax3.set_xlabel('X (Å)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Z (Å)', fontsize=12, fontweight='bold')
        ax3.set_title(f'Side View - Planarity\nZ-range={metrics["z_range"]:.2f} Å', 
                     fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved single ring plot to {save_path}")
        
        plt.show()
        
    def visualize_planarity_analysis(self, save_path=None, dpi=300):
        """
        Create 2D publication-quality plots showing planarity and Z-range analysis.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the figure
        dpi : int
            Resolution for saved figure
        """
        # Collect all planarity data
        all_z_spreads = []
        all_z_ranges = []
        all_plane_rmsds = []
        all_planarity_ratios = []
        tomo_labels = []
        
        # Collect per-protomer Z-deviations
        protomer_z_deviations = [[] for _ in range(9)]
        
        for composite_key, ring in self.rings.items():
            positions = ring['aligned_positions']
            metrics = ring['metrics']
            
            all_z_spreads.append(metrics['z_spread'])
            all_z_ranges.append(metrics['z_range'])
            all_plane_rmsds.append(metrics['plane_rmsd'])
            all_planarity_ratios.append(metrics['planarity_ratio'])
            tomo_labels.append(ring['tomo_name'])
            
            # Collect Z-deviations for each protomer
            for i, pos in enumerate(positions):
                protomer_z_deviations[i].append(pos[2])
        
        all_z_spreads = np.array(all_z_spreads)
        all_z_ranges = np.array(all_z_ranges)
        all_plane_rmsds = np.array(all_plane_rmsds)
        all_planarity_ratios = np.array(all_planarity_ratios)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Z-spread distribution histogram
        ax1 = axes[0, 0]
        ax1.hist(all_z_spreads, bins=30, alpha=0.7, color='coral', edgecolor='black')
        ax1.axvline(all_z_spreads.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {all_z_spreads.mean():.2f} Å')
        ax1.axvline(np.median(all_z_spreads), color='blue', linestyle=':', linewidth=2, 
                   label=f'Median: {np.median(all_z_spreads):.2f} Å')
        ax1.set_xlabel('Z-spread (Å)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax1.set_title('Z-Spread Distribution\n(Standard deviation of Z positions)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 2. Z-range vs Plane RMSD scatter
        ax2 = axes[0, 1]
        scatter = ax2.scatter(all_z_ranges, all_plane_rmsds, alpha=0.5, s=50, 
                             c=all_planarity_ratios, cmap='viridis', edgecolors='black', linewidth=0.5)
        ax2.set_xlabel('Z-range (Å)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Plane RMSD (Å)', fontsize=12, fontweight='bold')
        ax2.set_title('Z-Range vs Plane Fit Quality', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Planarity Ratio', fontsize=10)
        
        # Add correlation coefficient
        correlation = np.corrcoef(all_z_ranges, all_plane_rmsds)[0, 1]
        ax2.text(0.05, 0.95, f'R = {correlation:.3f}', 
                transform=ax2.transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 3. Box plot of Z-deviations by protomer position
        ax3 = axes[1, 0]
        bp = ax3.boxplot(protomer_z_deviations, positions=range(9), widths=0.6,
                         patch_artist=True, showmeans=True,
                         boxprops=dict(facecolor='lightcoral', alpha=0.7),
                         medianprops=dict(color='red', linewidth=2),
                         meanprops=dict(marker='D', markerfacecolor='blue', markersize=6))
        
        ax3.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        ax3.set_xlabel('Protomer Position', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Z-deviation (Å)', fontsize=12, fontweight='bold')
        ax3.set_title('Z-Deviation per Protomer\n(relative to ring plane)', fontsize=14, fontweight='bold')
        ax3.set_xticklabels([f'P{i}' for i in range(9)])
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Planarity ratio distribution
        ax4 = axes[1, 1]
        ax4.hist(all_planarity_ratios, bins=30, alpha=0.7, color='mediumpurple', edgecolor='black')
        ax4.axvline(all_planarity_ratios.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {all_planarity_ratios.mean():.4f}')
        ax4.axvline(np.median(all_planarity_ratios), color='blue', linestyle=':', linewidth=2, 
                   label=f'Median: {np.median(all_planarity_ratios):.4f}')
        ax4.set_xlabel('Planarity Ratio (λ₃/λ₁)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax4.set_title('Planarity Ratio Distribution\n(lower = more planar)', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(left=0)
        
        # Add text box with summary statistics
        stats_text = f'Z-spread: {all_z_spreads.mean():.2f} ± {all_z_spreads.std():.2f} Å\n'
        stats_text += f'Z-range: {all_z_ranges.mean():.2f} ± {all_z_ranges.std():.2f} Å\n'
        stats_text += f'Plane RMSD: {all_plane_rmsds.mean():.2f} ± {all_plane_rmsds.std():.2f} Å'
        ax4.text(0.6, 0.95, stats_text, transform=ax4.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved planarity analysis plot to {save_path}")
        
        plt.show()
        
    def visualize_point_cloud(self, color_by='tomogram', save_path=None):
        """
        Visualize all particles as a point cloud.
        
        Parameters:
        -----------
        color_by : str, default='tomogram'
            Color points by: 'tomogram', 'protomer', 'z_position', or 'radius'
        save_path : str, optional
            Path to save the HTML file
        """
        # Collect all points
        all_points = []
        all_colors = []
        all_info = []
        
        # Create tomogram color mapping
        unique_tomos = sorted(set(ring['tomo_name'] for ring in self.rings.values()))
        tomo_color_map = {tomo: idx for idx, tomo in enumerate(unique_tomos)}
        
        for composite_key, ring in self.rings.items():
            positions = ring['aligned_positions']
            protomer_indices = ring['protomer_indices']
            
            for i, pos in enumerate(positions):
                all_points.append(pos)
                
                # Determine color value based on color_by parameter
                if color_by == 'tomogram':
                    color_val = tomo_color_map[ring['tomo_name']]
                    color_label = ring['tomo_name']
                elif color_by == 'protomer':
                    color_val = protomer_indices[i]
                    color_label = f'Protomer {protomer_indices[i]}'
                elif color_by == 'z_position':
                    color_val = pos[2]
                    color_label = f'Z: {pos[2]:.2f} Å'
                elif color_by == 'radius':
                    radius = np.sqrt(pos[0]**2 + pos[1]**2)
                    color_val = radius
                    color_label = f'Radius: {radius:.2f} Å'
                else:
                    color_val = tomo_color_map[ring['tomo_name']]
                    color_label = ring['tomo_name']
                
                all_colors.append(color_val)
                all_info.append({
                    'composite_key': composite_key,
                    'ring_id': ring['ring_id'],
                    'protomer': protomer_indices[i],
                    'tomo': ring['tomo_name'],
                    'color_label': color_label
                })
        
        all_points = np.array(all_points)
        
        # Create figure
        fig = go.Figure()
        
        # Determine colorscale
        if color_by == 'tomogram':
            colorscale = 'Rainbow'
            showscale = len(unique_tomos) > 1
        elif color_by == 'protomer':
            colorscale = 'Rainbow'
            showscale = True
        else:
            colorscale = 'Viridis'
            showscale = True
        
        fig.add_trace(go.Scatter3d(
            x=all_points[:, 0],
            y=all_points[:, 1],
            z=all_points[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=all_colors,
                colorscale=colorscale,
                showscale=showscale,
                colorbar=dict(title=color_by.replace('_', ' ').title()) if showscale else None,
                opacity=0.7
            ),
            text=[f"Ring {info['ring_id']}<br>Tomo: {info['tomo']}<br>Protomer {info['protomer']}<br>{info['color_label']}" 
                  for info in all_info],
            hovertemplate='<b>%{text}</b><br>' +
                         'X: %{x:.2f} Å<br>' +
                         'Y: %{y:.2f} Å<br>' +
                         'Z: %{z:.2f} Å<extra></extra>',
            name='Particles'
        ))
        
        title = f"Point Cloud - {len(self.rings)} rings"
        if self.tomo_filter:
            title += f" from {', '.join(self.tomo_filter)}"
        title += f" (colored by {color_by})"
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X (Å)',
                yaxis_title='Y (Å)',
                zaxis_title='Z (Å)',
                aspectmode='data'
            ),
            width=1000,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"Saved point cloud to {save_path}")
        
        fig.show()
        
    def export_metrics_csv(self, output_path):
        """Export all ring metrics to CSV."""
        rows = []
        for composite_key, ring in self.rings.items():
            row = {
                'composite_key': composite_key,
                'tomo_name': ring['tomo_name'],
                'ring_id': ring['ring_id'],
                **ring['metrics']
            }
            # Remove the radii array from export
            row.pop('radii', None)
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"\nExported metrics to {output_path}")
        return df


def main():
    parser = argparse.ArgumentParser(
        description='Analyze protomer rings from STAR files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all tomograms in the STAR file
  python protomer_ring_analysis.py --i input.star
  
  # Analyze specific tomograms
  python protomer_ring_analysis.py --i input.star --tomo "TS_004,TS_005"
  
  # Save outputs with custom prefix and generate 2D plots
  python protomer_ring_analysis.py --i input.star --tomo "TS_004,TS_005" --o output_prefix --plot-radial
        """
    )
    
    parser.add_argument('--i', '--input', dest='input', required=True,
                       help='Input STAR file path')
    parser.add_argument('--tomo', '--tomogram', dest='tomo', default=None,
                       help='Comma-separated list of tomogram names to analyze (e.g., "TS_004,TS_005")')
    parser.add_argument('--o', '--output', dest='output', default=None,
                       help='Output prefix for saved files (default: derived from input filename)')
    parser.add_argument('--color', dest='color_by', default='tomogram',
                       choices=['tomogram', 'protomer', 'z_position', 'radius'],
                       help='Color scheme for point cloud visualization (default: tomogram)')
    parser.add_argument('--no-viz', dest='no_viz', action='store_true',
                       help='Skip interactive visualization (only export CSV)')
    parser.add_argument('--plot-radial', dest='plot_radial', action='store_true',
                       help='Generate 2D radial distribution plots (publication quality)')
    parser.add_argument('--plot-planarity', dest='plot_planarity', action='store_true',
                       help='Generate 2D planarity analysis plots (publication quality)')
    parser.add_argument('--plot-ring', dest='plot_ring', default=None,
                       help='Generate 2D plot for specific ring (use composite key, e.g., "Trichonympha_039_1.0")')
    
    args = parser.parse_args()
    
    # Parse tomogram filter
    tomo_filter = None
    if args.tomo:
        tomo_filter = [t.strip() for t in args.tomo.split(',')]
    
    # Determine output prefix
    if args.output:
        output_prefix = args.output
    else:
        # Derive from input filename
        input_path = Path(args.input)
        output_prefix = input_path.stem
    
    # Initialize analyzer
    print("="*60)
    print("PROTOMER RING ANALYZER")
    print("="*60)
    analyzer = RingAnalyzer(args.input, tomo_filter=tomo_filter)
    
    # Run analysis pipeline
    analyzer.load_star_file()
    analyzer.calculate_real_positions()
    analyzer.analyze_rings()
    analyzer.print_summary_statistics()
    
    # Export metrics
    csv_path = f"{output_prefix}_metrics.csv"
    analyzer.export_metrics_csv(csv_path)
    
    # Generate 2D radial distribution plots
    if args.plot_radial:
        radial_path = f"{output_prefix}_radial_distribution.png"
        analyzer.visualize_radial_distribution(save_path=radial_path)
    
    # Generate 2D planarity analysis plots
    if args.plot_planarity:
        planarity_path = f"{output_prefix}_planarity_analysis.png"
        analyzer.visualize_planarity_analysis(save_path=planarity_path)
    
    # Generate single ring 2D plot
    if args.plot_ring:
        ring_path = f"{output_prefix}_ring_{args.plot_ring.replace('/', '_')}.png"
        analyzer.visualize_single_ring_2d(args.plot_ring, save_path=ring_path)
    
    # Visualize point cloud
    if not args.no_viz:
        html_path = f"{output_prefix}_pointcloud.html"
        analyzer.visualize_point_cloud(color_by=args.color_by, save_path=html_path)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()