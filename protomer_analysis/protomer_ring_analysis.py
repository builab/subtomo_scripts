#!/usr/bin/python
# Example
# python ~/Documents/GitHub/subtomo/protomer_analysis/protomer_ring_analysis.py --i cbn_protomers.star   --tomo1 "Trichonympha_104,Trichonympha_122,Trichonympha_129,Trichonympha_133,Trichonympha_103" --tomo2 "Trichonympha_007,Trichonympha_019,Trichonympha_035,Trichonympha_039,Trichonympha_062" --plot-analysis


import starfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde, ttest_ind
from pathlib import Path
import argparse

class RingAnalyzer:
    def __init__(self, star_file_path, group1_filter=None, group2_filter=None):
        self.star_file_path = star_file_path
        self.group1_filter = group1_filter
        self.group2_filter = group2_filter
        self.rings = {'Group 1': {}, 'Group 2': {}}
        
    def load_and_process(self):
        print(f"Loading STAR file: {self.star_file_path}")
        data = starfile.read(self.star_file_path)
        particles = data.get('particles', data.get('data_particles', data))
        optics = data.get('optics', data.get('data_optics'))
        pixel_size = optics['rlnImagePixelSize'].iloc[0]

        particles['realX'] = (particles['rlnCoordinateX'] * pixel_size - particles['rlnOriginXAngst'])
        particles['realY'] = (particles['rlnCoordinateY'] * pixel_size - particles['rlnOriginYAngst'])
        particles['realZ'] = (particles['rlnCoordinateZ'] * pixel_size - particles['rlnOriginZAngst'])

        for group_name, filters in [('Group 1', self.group1_filter), ('Group 2', self.group2_filter)]:
            if not filters: continue
            tomo_list = [f.strip() for f in filters.split(',')]
            mask = particles['rlnTomoName'].str.contains('|'.join(tomo_list))
            group_data = particles[mask]
            
            print(f"\nProcessing {group_name} ({len(group_data)} particles)...")
            grouped = group_data.groupby(['rlnTomoName', 'rlnOriginalIndex'])
            
            for (tomo_name, ring_id), ring_data in grouped:
                if len(ring_data) != 9: continue
                pos = ring_data.sort_values('rlnProtomerIndex')[['realX', 'realY', 'realZ']].values
                centered = pos - pos.mean(axis=0)
                pca = PCA(n_components=3).fit(centered)
                normal = pca.components_[2]
                z_axis = np.array([0, 0, 1])
                rot_axis = np.cross(normal, z_axis)
                if np.linalg.norm(rot_axis) > 1e-6:
                    angle = np.arccos(np.clip(np.dot(normal / np.linalg.norm(normal), z_axis), -1.0, 1.0))
                    centered = Rotation.from_rotvec(angle * (rot_axis / np.linalg.norm(rot_axis))).apply(centered)
                angle_to_x = np.arctan2(centered[0, 1], centered[0, 0])
                centered = Rotation.from_euler('z', -angle_to_x).apply(centered)
                radii = np.sqrt(centered[:, 0]**2 + centered[:, 1]**2)
                self.rings[group_name][f"{tomo_name}_{ring_id}"] = {'pos': centered, 'radii': radii}

    def print_stats(self):
        print("\n" + "="*50 + "\nSTATISTICAL COMPARISON & SIGNIFICANCE\n" + "="*50)
        res = {}
        for group in ['Group 1', 'Group 2']:
            if not self.rings[group]: continue
            all_r = np.concatenate([r['radii'] for r in self.rings[group].values()])
            all_z = np.abs(np.concatenate([r['pos'][:, 2] for r in self.rings[group].values()]))
            res[group] = {'r': all_r, 'z': all_z}
            print(f"{group}:\n  Radius: {np.mean(all_r):.2f} ± {np.std(all_r):.2f} Å\n  |Z|:    {np.mean(all_z):.2f} ± {np.std(all_z):.2f} Å")

        if 'Group 1' in res and 'Group 2' in res:
            p_rad = ttest_ind(res['Group 1']['r'], res['Group 2']['r'])[1]
            p_z = ttest_ind(res['Group 1']['z'], res['Group 2']['z'])[1]
            print(f"-"*50 + f"\nT-Test p-values:\n  Radius:    {p_rad:.4e}\n  Planarity: {p_z:.4e}\n" + "="*50)

    def get_group_data(self, group):
        all_x, all_y, all_z = [], [], []
        for r in self.rings[group].values():
            for i in range(9):
                rot = Rotation.from_euler('z', i * 40, degrees=True).apply(r['pos'])
                all_x.extend(rot[:, 0]); all_y.extend(rot[:, 1]); all_z.extend(rot[:, 2])
        return np.array(all_x), np.array(all_y), np.array(all_z)

    def plot_pairwise_density(self, prefix):
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        for idx, group in enumerate(['Group 1', 'Group 2']):
            x, y, z = self.get_group_data(group)
            mean_r = np.mean(np.sqrt(x**2 + y**2))
            xi, yi = np.mgrid[x.min()-10:x.max()+10:150j, y.min()-10:y.max()+10:150j]
            zi = np.reshape(gaussian_kde(np.vstack([x, y]), bw_method=0.25)(np.vstack([xi.flatten(), yi.flatten()])).T, xi.shape)
            axes[idx].contourf(xi, yi, zi, levels=25, cmap='magma')
            axes[idx].add_artist(plt.Circle((0, 0), mean_r, color='cyan', fill=False, linestyle='--', linewidth=2))
            axes[idx].set_title(f"{group} Density (R={mean_r:.1f}Å)", fontweight='bold')
            axes[idx].axis('equal')
        plt.tight_layout(); plt.savefig(f"{prefix}_density.pdf"); plt.show()

    def plot_pairwise_zmap(self, prefix):
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        for idx, group in enumerate(['Group 1', 'Group 2']):
            x, y, z = self.get_group_data(group)
            mean_r = np.mean(np.sqrt(x**2 + y**2))
            sc = axes[idx].scatter(x, y, c=z, cmap='RdBu_r', s=25, alpha=0.4, vmin=-10, vmax=10)
            axes[idx].add_artist(plt.Circle((0, 0), mean_r, color='black', fill=False, linestyle='--', alpha=0.5))
            axes[idx].set_title(f"{group} Z-Deviation", fontweight='bold')
            axes[idx].axis('equal')
            plt.colorbar(sc, ax=axes[idx], label='Z-offset (Å)')
        plt.tight_layout(); plt.savefig(f"{prefix}_zmap.pdf"); plt.show()

    def plot_pairwise_3d(self, prefix):
        fig = plt.figure(figsize=(18, 8))
        for idx, group in enumerate(['Group 1', 'Group 2']):
            x, y, z = self.get_group_data(group)
            ax = fig.add_subplot(1, 2, idx+1, projection='3d')
            
            # 1. Calculate Mean Radius for the Shadow Ring
            mean_r = np.mean(np.sqrt(x**2 + y**2))
            
            # 2. Generate Shadow Ring coordinates in 3D (Z=0)
            theta = np.linspace(0, 2*np.pi, 100)
            ring_x = mean_r * np.cos(theta)
            ring_y = mean_r * np.sin(theta)
            ring_z = np.zeros_like(theta)
            
            # 3. Plot the Shadow Ring
            ax.plot(ring_x, ring_y, ring_z, color='black', linestyle='--', 
                    linewidth=2, alpha=0.7, label=f'Shadow Ring (R={mean_r:.1f}Å)')
            
            # 4. Plot the Protomer points
            sc = ax.scatter(x, y, z, c=z, cmap='RdBu_r', s=10, alpha=0.6, vmin=-10, vmax=10)
            
            # Formatting
            ax.set_zlim(-15, 15)
            ax.set_box_aspect((1, 1, 0.4))
            ax.set_title(f"{group} 3D Profile", fontweight='bold')
            ax.set_xlabel('X (Å)')
            ax.set_ylabel('Y (Å)')
            ax.set_zlabel('Z (Å)')
            
        plt.tight_layout()
        plt.savefig(f"{prefix}_3d_profile.pdf")
        plt.show()

    def visualize_violin_comparison(self, prefix):
        data = []
        for group in ['Group 1', 'Group 2']:
            all_z = np.concatenate([r['pos'][:, 2] for r in self.rings[group].values()])
            for z in all_z: data.append({'Z-offset (Å)': z, 'Group': group})
        
        plt.figure(figsize=(10, 8))
        sns.violinplot(data=pd.DataFrame(data), x='Group', y='Z-offset (Å)', palette={'Group 1': 'skyblue', 'Group 2': 'salmon'}, inner="quartile")
        plt.axhline(0, color='red', linestyle='--', alpha=0.5)
        plt.title("Combined Z-Distribution (Thickness Comparison)", fontweight='bold', fontsize=14)
        plt.ylim(-15, 15); plt.tight_layout(); plt.savefig(f"{prefix}_violin.pdf", dpi=300); plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--i', required=True)
    parser.add_argument('--tomo1', required=True)
    parser.add_argument('--tomo2', required=True)
    parser.add_argument('--plot-analysis', action='store_true')
    args = parser.parse_args()
    
    analyzer = RingAnalyzer(args.i, args.tomo1, args.tomo2)
    analyzer.load_and_process()
    analyzer.print_stats()
    
    if args.plot_analysis:
        p = Path(args.i).stem
        # Generate each figure separately
        analyzer.plot_pairwise_density(p)
        analyzer.plot_pairwise_zmap(p)
        analyzer.plot_pairwise_3d(p)
        analyzer.visualize_violin_comparison(p)

if __name__ == "__main__":
    main()