#!/usr/bin/env python3
"""
Visualize cryo-EM particle orientations from Euler angles on a sphere.

Usage:
    python visualize_angle_list.py -i angle_list.txt
    python visualize_angle_list.py -i angle_list.txt -o plot.png
    python visualize_angle_list.py -i angle_list.txt -o plot.png --show-orientations
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import sys

def read_angles_from_file(filename):
    """
    Read Euler angle tuples from a text file.
    Assumes each line contains three angles (psi, theta, phi) in radians,
    separated by whitespace or comma.
    
    Note: File format is psi, theta, phi but returned as (phi, theta, psi)
    
    Cryo-EM convention:
    - phi: first rotation about Z axis
    - theta: rotation about Y axis (colatitude)
    - psi: final rotation about Z axis (in-plane rotation)
    """
    angles = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                # Split by comma or whitespace
                parts = line.replace(',', ' ').split()
                if len(parts) >= 3:
                    psi = float(parts[0])    # First column is psi
                    theta = float(parts[1])  # Second column is theta
                    phi = float(parts[2])    # Third column is phi
                    angles.append((phi, theta, psi))  # Store as (phi, theta, psi)
    return np.array(angles)

def angles_to_cartesian(phi, theta, psi=None, r=1):
    """
    Convert Euler angles to Cartesian coordinates for the viewing direction.
    
    Cryo-EM convention:
    - phi: rotation about Z axis
    - theta: colatitude (0 to pi), angle from Z axis
    - psi: in-plane rotation (affects orientation but not position on sphere)
    
    Returns the point on the sphere representing the viewing direction.
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def euler_to_rotation_matrix(phi, theta, psi):
    """
    Convert Euler angles (ZYZ convention) to rotation matrix.
    This can be used to show orientation with arrows.
    """
    # Rotation matrices for ZYZ convention
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)
    
    # Combined rotation matrix
    R = np.array([
        [cos_phi*cos_theta*cos_psi - sin_phi*sin_psi, 
         -cos_phi*cos_theta*sin_psi - sin_phi*cos_psi, 
         cos_phi*sin_theta],
        [sin_phi*cos_theta*cos_psi + cos_phi*sin_psi, 
         -sin_phi*cos_theta*sin_psi + cos_phi*cos_psi, 
         sin_phi*sin_theta],
        [-sin_theta*cos_psi, 
         sin_theta*sin_psi, 
         cos_theta]
    ])
    
    return R

def plot_points_on_sphere(angles, show_sphere=True, show_orientations=False, sample_orientations=50):
    """
    Plot points on a sphere given their Euler angle coordinates.
    
    Args:
        angles: Nx3 array of (phi, theta, psi) in radians
        show_sphere: Whether to show wireframe sphere
        show_orientations: Whether to show orientation arrows (can be slow for many points)
        sample_orientations: If showing orientations, sample this many randomly
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract angles
    phi = angles[:, 0]
    theta = angles[:, 1]
    psi = angles[:, 2]
    
    # Convert to Cartesian coordinates (viewing directions)
    x, y, z = angles_to_cartesian(phi, theta, psi)
    
    # Plot the points
    ax.scatter(x, y, z, c='red', marker='o', s=20, alpha=0.6, label='Particle orientations')
    
    # Optionally show orientation arrows for a subset of points
    if show_orientations and len(angles) > 0:
        # Sample points if there are too many
        n_sample = min(sample_orientations, len(angles))
        indices = np.random.choice(len(angles), n_sample, replace=False)
        
        arrow_length = 0.15
        for idx in indices:
            # Get rotation matrix
            R = euler_to_rotation_matrix(phi[idx], theta[idx], psi[idx])
            
            # The Z-axis direction after rotation (viewing direction)
            direction = R[:, 2]
            
            # Plot arrow from sphere surface
            ax.quiver(x[idx], y[idx], z[idx], 
                     direction[0], direction[1], direction[2],
                     length=arrow_length, color='blue', alpha=0.3, 
                     arrow_length_ratio=0.3, linewidth=0.5)
    
    # Optionally plot a wireframe sphere for reference
    if show_sphere:
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_sphere = np.outer(np.sin(v), np.cos(u))
        y_sphere = np.outer(np.sin(v), np.sin(u))
        z_sphere = np.outer(np.cos(v), np.ones(np.size(u)))
        ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='lightblue', 
                         alpha=0.1, linewidth=0.5)
    
    # Plot coordinate axes
    axis_length = 1.3
    arrow_props = dict(arrow_length_ratio=0.15, linewidth=2.5)
    
    # X-axis (red)
    ax.quiver(0, 0, 0, axis_length, 0, 0, 
             color='red', **arrow_props, label='X-axis')
    ax.text(axis_length + 0.1, 0, 0, 'X', color='red', fontsize=14, fontweight='bold')
    
    # Y-axis (green)
    ax.quiver(0, 0, 0, 0, axis_length, 0, 
             color='green', **arrow_props, label='Y-axis')
    ax.text(0, axis_length + 0.1, 0, 'Y', color='green', fontsize=14, fontweight='bold')
    
    # Z-axis (blue)
    ax.quiver(0, 0, 0, 0, 0, axis_length, 
             color='blue', **arrow_props, label='Z-axis')
    ax.text(0, 0, axis_length + 0.1, 'Z', color='blue', fontsize=14, fontweight='bold')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    title = f'Cryo-EM Particle Orientations ({len(angles)} particles)'
    if show_orientations:
        title += f'\n(showing {min(sample_orientations, len(angles))} orientation arrows)'
    ax.set_title(title)
    
    # Equal aspect ratio
    max_range = 1.5
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    ax.legend()
    plt.tight_layout()
    return fig, ax

def main():
    parser = argparse.ArgumentParser(
        description='Visualize cryo-EM particle orientations from Euler angles on a sphere.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -i angles.txt
  %(prog)s -i angles.txt -o plot.png
  %(prog)s -i angles.txt -o plot.png --show-orientations --sample 100
  %(prog)s --input angles.txt --output figure.pdf --dpi 300

Input file format (psi, theta, phi in radians, space or comma separated):
  0.7854 1.0472 0.5236
  1.5708 2.0944 1.0472
  2.3562 3.1416 1.5708
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                       help='Input text file containing Euler angles (psi, theta, phi) in radians')
    parser.add_argument('-o', '--output', default=None,
                       help='Output plot file (e.g., plot.png, figure.pdf). If not specified, displays plot interactively.')
    parser.add_argument('--show-orientations', action='store_true',
                       help='Show orientation arrows for particles (may be slow for large datasets)')
    parser.add_argument('--sample', type=int, default=50,
                       help='Number of orientation arrows to sample if --show-orientations is used (default: 50)')
    parser.add_argument('--no-sphere', action='store_true',
                       help='Do not show the wireframe sphere')
    parser.add_argument('--dpi', type=int, default=150,
                       help='DPI for output image (default: 150)')
    
    args = parser.parse_args()
    
    try:
        # Read angles from file
        print(f"Reading Euler angles from {args.input}...")
        angles = read_angles_from_file(args.input)
        
        if len(angles) == 0:
            print("Error: No valid angle data found in the input file.")
            sys.exit(1)
        
        print(f"Loaded {len(angles)} particle orientations")
        
        # Print angle statistics
        print("\nAngle statistics (radians):")
        print(f"  Phi:   min={angles[:, 0].min():.3f}, max={angles[:, 0].max():.3f}, mean={angles[:, 0].mean():.3f}")
        print(f"  Theta: min={angles[:, 1].min():.3f}, max={angles[:, 1].max():.3f}, mean={angles[:, 1].mean():.3f}")
        print(f"  Psi:   min={angles[:, 2].min():.3f}, max={angles[:, 2].max():.3f}, mean={angles[:, 2].mean():.3f}")
        
        # Plot the points
        print("\nPlotting particle orientations on sphere...")
        fig, ax = plot_points_on_sphere(
            angles, 
            show_sphere=not args.no_sphere, 
            show_orientations=args.show_orientations,
            sample_orientations=args.sample
        )
        
        # Save or display
        if args.output:
            print(f"Saving plot to {args.output}...")
            plt.savefig(args.output, dpi=args.dpi, bbox_inches='tight')
            print(f"Plot saved successfully!")
        else:
            print("Displaying plot (close window to exit)...")
            plt.show()
        
    except FileNotFoundError:
        print(f"Error: File '{args.input}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
