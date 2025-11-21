#!/usr/bin/env python3
"""
Unroll a helical structure from MRC file into cylindrical coordinates.
Transforms (x,y,z) Cartesian coordinates to (theta, z, r) cylindrical coordinates
and creates a 2D unwrapped visualization.
"""

import argparse
import numpy as np
import mrcfile
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

def find_helix_axis(volume, threshold_percentile=90):
    """
    Estimate the helix axis by finding the center of mass at each z-slice.
    
    Args:
        volume: 3D numpy array
        threshold_percentile: Percentile threshold for considering voxels
        
    Returns:
        center_x, center_y: Arrays of center coordinates for each z-slice
    """
    nz = volume.shape[0]
    center_x = np.zeros(nz)
    center_y = np.zeros(nz)
    
    threshold = np.percentile(volume, threshold_percentile)
    
    for z in range(nz):
        slice_data = volume[z, :, :]
        mask = slice_data > threshold
        
        if np.sum(mask) > 0:
            y_coords, x_coords = np.where(mask)
            weights = slice_data[mask]
            center_x[z] = np.average(x_coords, weights=weights)
            center_y[z] = np.average(y_coords, weights=weights)
        else:
            # If no points above threshold, use previous center or image center
            if z > 0:
                center_x[z] = center_x[z-1]
                center_y[z] = center_y[z-1]
            else:
                center_x[z] = slice_data.shape[1] / 2
                center_y[z] = slice_data.shape[0] / 2
    
    return center_x, center_y

def unroll_helix(volume, center_x, center_y, r_min=0, r_max=None, n_theta=360):
    """
    Unroll the helix into cylindrical coordinates.
    
    Args:
        volume: 3D numpy array (z, y, x)
        center_x, center_y: Arrays defining helix axis at each z
        r_min: Minimum radius to sample
        r_max: Maximum radius to sample (default: auto-detect)
        n_theta: Number of angular samples (resolution around the helix)
        
    Returns:
        unrolled: 2D array (theta, z) of unrolled helix
        r_profile: 2D array (r, theta, z) for full cylindrical transform
    """
    nz, ny, nx = volume.shape
    
    # Auto-detect r_max if not provided
    if r_max is None:
        r_max = min(nx, ny) // 2
    
    # Create cylindrical coordinate sampling
    theta_samples = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    z_samples = np.arange(nz)
    
    # For the main unrolled image, we'll use a middle radius or average radius
    # You can also integrate over a radius range
    r_middle = (r_min + r_max) // 2
    
    unrolled = np.zeros((n_theta, nz))
    
    for iz, z in enumerate(z_samples):
        cx = center_x[iz]
        cy = center_y[iz]
        
        for itheta, theta in enumerate(theta_samples):
            # Calculate Cartesian coordinates
            x = cx + r_middle * np.cos(theta)
            y = cy + r_middle * np.sin(theta)
            
            # Check bounds
            if 0 <= x < nx and 0 <= y < ny:
                # Interpolate value at this position
                coords = np.array([[z, y, x]])
                value = map_coordinates(volume, coords.T, order=1, mode='constant')
                unrolled[itheta, iz] = value[0]
    
    return unrolled

def unroll_helix_radial(volume, center_x, center_y, r_min=0, r_max=None, 
                        n_theta=360, n_radii=50):
    """
    Create a radial profile of the unrolled helix.
    
    Args:
        volume: 3D numpy array (z, y, x)
        center_x, center_y: Arrays defining helix axis at each z
        r_min: Minimum radius to sample
        r_max: Maximum radius to sample
        n_theta: Number of angular samples
        n_radii: Number of radial samples
        
    Returns:
        radial_profile: 3D array (r, theta, z)
    """
    nz, ny, nx = volume.shape
    
    if r_max is None:
        r_max = min(nx, ny) // 2
    
    theta_samples = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    z_samples = np.arange(nz)
    r_samples = np.linspace(r_min, r_max, n_radii)
    
    radial_profile = np.zeros((n_radii, n_theta, nz))
    
    for iz, z in enumerate(z_samples):
        cx = center_x[iz]
        cy = center_y[iz]
        
        for ir, r in enumerate(r_samples):
            for itheta, theta in enumerate(theta_samples):
                x = cx + r * np.cos(theta)
                y = cy + r * np.sin(theta)
                
                if 0 <= x < nx and 0 <= y < ny:
                    coords = np.array([[z, y, x]])
                    value = map_coordinates(volume, coords.T, order=1, mode='constant')
                    radial_profile[ir, itheta, iz] = value[0]
    
    return radial_profile

def main():
    parser = argparse.ArgumentParser(
        description='Unroll helical structure from MRC file into cylindrical coordinates'
    )
    parser.add_argument('--i', required=True, help='Input MRC file')
    parser.add_argument('--o', required=True, help='Output MRC file for unrolled helix')
    parser.add_argument('--o_png', type=str, default=None,
                        help='Optional: Output PNG file for visualization')
    parser.add_argument('--r_min', type=float, default=0, 
                        help='Minimum radius for sampling (pixels)')
    parser.add_argument('--r_max', type=float, default=None,
                        help='Maximum radius for sampling (pixels, default: auto)')
    parser.add_argument('--n_theta', type=int, default=360,
                        help='Angular resolution (number of samples around helix)')
    parser.add_argument('--n_radii', type=int, default=50,
                        help='Number of radial samples for 3D output')
    parser.add_argument('--threshold_percentile', type=float, default=90,
                        help='Percentile threshold for finding helix axis')
    parser.add_argument('--mode', type=str, choices=['2d', '3d'], default='3d',
                        help='Output mode: 2d (theta, z) or 3d (r, theta, z)')
    parser.add_argument('--radial_output', type=str, default=None,
                        help='Optional: Save radial profile visualization as PNG')
    parser.add_argument('--show_axis', action='store_true',
                        help='Save a figure showing the detected helix axis')
    
    args = parser.parse_args()
    
    # Read MRC file
    print(f"Reading MRC file: {args.i}")
    with mrcfile.open(args.i, mode='r') as mrc:
        volume = mrc.data.copy()
        voxel_size = mrc.voxel_size
    
    print(f"Volume shape: {volume.shape}")
    print(f"Voxel size: {voxel_size}")
    
    # Find helix axis
    print("Detecting helix axis...")
    center_x, center_y = find_helix_axis(volume, args.threshold_percentile)
    
    # Optionally visualize the axis
    if args.show_axis:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot axis trajectory
        z_coords = np.arange(len(center_x))
        axes[0].plot(center_x, z_coords, label='X position')
        axes[0].plot(center_y, z_coords, label='Y position')
        axes[0].set_xlabel('Position (pixels)')
        axes[0].set_ylabel('Z slice')
        axes[0].set_title('Helix Axis Trajectory')
        axes[0].legend()
        axes[0].invert_yaxis()
        
        # Show middle slice with detected center
        mid_z = len(center_x) // 2
        axes[1].imshow(volume[mid_z, :, :], cmap='gray')
        axes[1].plot(center_x[mid_z], center_y[mid_z], 'r+', markersize=20, markeredgewidth=2)
        axes[1].set_title(f'Middle slice (z={mid_z}) with detected center')
        
        plt.tight_layout()
        axis_output = args.o.replace('.mrc', '_axis.png')
        plt.savefig(axis_output, dpi=150, bbox_inches='tight')
        print(f"Helix axis visualization saved to: {axis_output}")
        plt.close()
    
    # Unroll helix based on mode
    if args.mode == '2d':
        print("Unrolling helix (2D mode: theta x z)...")
        unrolled = unroll_helix(volume, center_x, center_y, 
                               r_min=args.r_min, r_max=args.r_max, 
                               n_theta=args.n_theta)
        
        # Save as 2D MRC
        print(f"Saving 2D unrolled data to MRC: {args.o}")
        with mrcfile.new(args.o, overwrite=True) as mrc:
            mrc.set_data(unrolled.astype(np.float32))
            # Set voxel size - theta dimension is in degrees, z is original voxel size
            mrc.voxel_size = (voxel_size.z, 360.0/args.n_theta, 1.0)
        
        print(f"Output shape: {unrolled.shape} (theta, z)")
        
        # Optional PNG visualization
        if args.o_png:
            fig, ax = plt.subplots(figsize=(12, 8))
            im = ax.imshow(unrolled, aspect='auto', cmap='gray', 
                           extent=[0, volume.shape[0], 360, 0])
            ax.set_xlabel('Z position (slices)')
            ax.set_ylabel('Angle (degrees)')
            ax.set_title('Unrolled Helical Structure (2D)')
            plt.colorbar(im, ax=ax, label='Intensity')
            plt.tight_layout()
            plt.savefig(args.o_png, dpi=300, bbox_inches='tight')
            print(f"PNG visualization saved to: {args.o_png}")
            plt.close()
    
    else:  # 3D mode
        print("Unrolling helix (3D mode: r x theta x z)...")
        radial_profile = unroll_helix_radial(volume, center_x, center_y,
                                            r_min=args.r_min, r_max=args.r_max,
                                            n_theta=args.n_theta, n_radii=args.n_radii)
        
        # Save as 3D MRC
        print(f"Saving 3D unrolled data to MRC: {args.o}")
        with mrcfile.new(args.o, overwrite=True) as mrc:
            mrc.set_data(radial_profile.astype(np.float32))
            # Set voxel size
            r_range = (args.r_max or volume.shape[1]//2) - args.r_min
            mrc.voxel_size = (voxel_size.z, 360.0/args.n_theta, r_range/args.n_radii)
        
        print(f"Output shape: {radial_profile.shape} (r, theta, z)")
        
        # Optional PNG visualization
        if args.o_png:
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            
            # Show averaged unrolled (over all radii)
            avg_unrolled = np.mean(radial_profile, axis=0)
            im0 = axes[0].imshow(avg_unrolled, aspect='auto', cmap='gray',
                                extent=[0, volume.shape[0], 360, 0])
            axes[0].set_xlabel('Z position (slices)')
            axes[0].set_ylabel('Angle (degrees)')
            axes[0].set_title('Radially Averaged Unrolled Structure')
            plt.colorbar(im0, ax=axes[0], label='Intensity')
            
            # Show radial profile at middle z
            mid_z = volume.shape[0] // 2
            im1 = axes[1].imshow(radial_profile[:, :, mid_z], aspect='auto', cmap='gray',
                                extent=[0, 360, args.r_max or volume.shape[1]//2, args.r_min])
            axes[1].set_xlabel('Angle (degrees)')
            axes[1].set_ylabel('Radius (pixels)')
            axes[1].set_title(f'Radial Profile at z={mid_z}')
            plt.colorbar(im1, ax=axes[1], label='Intensity')
            
            plt.tight_layout()
            plt.savefig(args.o_png, dpi=300, bbox_inches='tight')
            print(f"PNG visualization saved to: {args.o_png}")
            plt.close()
    
    # Optional: Additional radial profile visualization
    if args.radial_output and args.mode == '3d':
        print("Creating additional radial profile visualization...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot 1: Average over theta
        avg_rz = np.mean(radial_profile, axis=1)
        im0 = axes[0].imshow(avg_rz, aspect='auto', cmap='gray',
                            extent=[0, volume.shape[0], args.r_max or volume.shape[1]//2, args.r_min])
        axes[0].set_xlabel('Z position (slices)')
        axes[0].set_ylabel('Radius (pixels)')
        axes[0].set_title('Radial-Z Profile (avg over theta)')
        plt.colorbar(im0, ax=axes[0], label='Intensity')
        
        # Plot 2: Average over z
        avg_r_theta = np.mean(radial_profile, axis=2)
        im1 = axes[1].imshow(avg_r_theta, aspect='auto', cmap='gray',
                            extent=[0, 360, args.r_max or volume.shape[1]//2, args.r_min])
        axes[1].set_xlabel('Angle (degrees)')
        axes[1].set_ylabel('Radius (pixels)')
        axes[1].set_title('Radial-Theta Profile (avg over z)')
        plt.colorbar(im1, ax=axes[1], label='Intensity')
        
        # Plot 3: Average over r
        avg_theta_z = np.mean(radial_profile, axis=0)
        im2 = axes[2].imshow(avg_theta_z, aspect='auto', cmap='gray',
                            extent=[0, volume.shape[0], 360, 0])
        axes[2].set_xlabel('Z position (slices)')
        axes[2].set_ylabel('Angle (degrees)')
        axes[2].set_title('Theta-Z Profile (avg over r)')
        plt.colorbar(im2, ax=axes[2], label='Intensity')
        
        plt.tight_layout()
        plt.savefig(args.radial_output, dpi=300, bbox_inches='tight')
        print(f"Radial profile visualization saved to: {args.radial_output}")
        plt.close()
    
    print("Done!")

if __name__ == '__main__':
    main()
