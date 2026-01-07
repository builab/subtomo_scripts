import starfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from sklearn.decomposition import PCA

def read_star_file(star_path):
    """Read Relion star file and compute coordinates."""
    star_data = starfile.read(star_path)
    
    # Check if star file has optics and particles blocks
    if isinstance(star_data, dict):
        optics = star_data['optics']
        data = star_data['particles']
        
        # Get pixel size from optics block
        # Assuming single optics group, or merge by rlnOpticsGroup if multiple
        if 'rlnOpticsGroup' in data.columns:
            # Merge optics info into particles data
            data = data.merge(optics[['rlnOpticsGroup', 'rlnImagePixelSize']], 
                            on='rlnOpticsGroup', how='left')
        else:
            # Single optics group
            data['rlnImagePixelSize'] = optics['rlnImagePixelSize'].iloc[0]
    else:
        # Old format without optics block
        data = star_data
        if 'rlnImagePixelSize' not in data.columns:
            raise ValueError("No rlnImagePixelSize found in data or optics block")
    
    angpix = data['rlnImagePixelSize']
    
    # Compute actual coordinates in Angstroms
    data['CoordX'] = data['rlnCoordinateX'] * angpix - data['rlnOriginXAngst']
    data['CoordY'] = data['rlnCoordinateY'] * angpix - data['rlnOriginYAngst']
    data['CoordZ'] = data['rlnCoordinateZ'] * angpix - data['rlnOriginZAngst']
    
    return data

def fit_ellipse_3d(points):
    """
    Fit ellipse to 3D points by:
    1. Finding best-fit plane using PCA
    2. Projecting points onto plane
    3. Fitting ellipse in 2D
    """
    # Center the points
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    
    # PCA to find plane (normal is the direction of smallest variance)
    pca = PCA(n_components=3)
    pca.fit(centered)
    
    # Project onto first 2 principal components (the plane)
    points_2d = pca.transform(centered)[:, :2]
    
    # Fit ellipse in 2D
    ellipse_params = fit_ellipse_2d(points_2d)
    
    # Calculate planarity metric (variance along normal / total variance)
    explained_var = pca.explained_variance_
    planarity = 1 - (explained_var[2] / np.sum(explained_var))
    
    # Alternative planarity: RMS distance to plane
    rms_dist = np.sqrt(np.mean(pca.transform(centered)[:, 2]**2))
    
    return ellipse_params, planarity, rms_dist

def fit_ellipse_2d(points):
    """
    Fit ellipse to 2D points using algebraic distance.
    Returns (semi_major, semi_minor, eccentricity, ellipticity)
    """
    x = points[:, 0]
    y = points[:, 1]
    
    # Center points
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_c = x - x_mean
    y_c = y - y_mean
    
    # Build design matrix for ellipse equation: ax^2 + bxy + cy^2 + dx + ey + f = 0
    # With constraint b^2 - 4ac < 0 (ellipse condition)
    D = np.column_stack([x_c**2, x_c*y_c, y_c**2, x_c, y_c, np.ones_like(x_c)])
    
    # Solve using SVD
    _, _, V = np.linalg.svd(D)
    params = V[-1, :]
    
    a, b, c, d, e, f = params
    
    # Calculate semi-axes
    # From general form to standard form
    den = b**2 - 4*a*c
    if den >= 0:  # Not an ellipse
        return None, None, None, None
    
    num = 2 * (a*e**2 + c*d**2 - b*d*e + den*f) * (a + c)
    
    # Semi-major and semi-minor axes
    temp = np.sqrt((a - c)**2 + b**2)
    semi_major = np.sqrt(num / (den * (temp - (a + c))))
    semi_minor = np.sqrt(num / (den * (-temp - (a + c))))
    
    # Ensure semi_major > semi_minor
    if semi_minor > semi_major:
        semi_major, semi_minor = semi_minor, semi_major
    
    ellipticity = semi_major / semi_minor if semi_minor != 0 else np.inf
    eccentricity = np.sqrt(1 - (semi_minor/semi_major)**2) if semi_major != 0 else 0
    
    return semi_major, semi_minor, eccentricity, ellipticity

def analyze_groups(data):
    """Analyze each group (TomogramName + OriginIndex)."""
    results = []
    
    # Group by TomogramName and OriginIndex
    grouped = data.groupby(['rlnTomoName', 'rlnOriginalIndex'])
    
    for (tomo_name, origin_idx), group in grouped:
        # Check if we have protomers 1-9
        protomers = sorted(group['rlnProtomerIndex'].values)
        
        # Extract 3D coordinates
        coords = group[['CoordX', 'CoordY', 'CoordZ']].values
        
        if len(coords) < 5:  # Need at least 5 points for ellipse fitting
            continue
        
        # Fit ellipse and calculate planarity
        ellipse_params, planarity_pca, rms_dist = fit_ellipse_3d(coords)
        
        if ellipse_params[3] is not None:  # Valid ellipse fit
            results.append({
                'TomogramName': tomo_name,
                'OriginIndex': origin_idx,
                'NumPoints': len(coords),
                'SemiMajor': ellipse_params[0],
                'SemiMinor': ellipse_params[1],
                'Eccentricity': ellipse_params[2],
                'Ellipticity': ellipse_params[3],
                'Planarity_PCA': planarity_pca,
                'RMS_Distance': rms_dist
            })
    
    return results

def plot_results(results):
    """Plot histograms of ellipticity and planarity metrics."""
    ellipticities = [r['Ellipticity'] for r in results]
    planarities = [r['Planarity_PCA'] for r in results]
    rms_dists = [r['RMS_Distance'] for r in results]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Ellipticity histogram
    axes[0, 0].hist(ellipticities, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Ellipticity (Major/Minor axis)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Ellipticity')
    axes[0, 0].axvline(np.mean(ellipticities), color='red', 
                       linestyle='--', label=f'Mean: {np.mean(ellipticities):.3f}')
    axes[0, 0].legend()
    
    # Planarity (PCA-based) histogram
    axes[0, 1].hist(planarities, bins=30, edgecolor='black', alpha=0.7, color='green')
    axes[0, 1].set_xlabel('Planarity (PCA-based, 0-1)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Planarity (PCA)')
    axes[0, 1].axvline(np.mean(planarities), color='red', 
                       linestyle='--', label=f'Mean: {np.mean(planarities):.3f}')
    axes[0, 1].legend()
    
    # RMS distance to plane
    axes[1, 0].hist(rms_dists, bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 0].set_xlabel('RMS Distance to Best-Fit Plane (Å)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of RMS Distance to Plane')
    axes[1, 0].axvline(np.mean(rms_dists), color='red', 
                       linestyle='--', label=f'Mean: {np.mean(rms_dists):.2f} Å')
    axes[1, 0].legend()
    
    # Scatter: Ellipticity vs Planarity
    axes[1, 1].scatter(planarities, ellipticities, alpha=0.5)
    axes[1, 1].set_xlabel('Planarity (PCA-based)')
    axes[1, 1].set_ylabel('Ellipticity')
    axes[1, 1].set_title('Ellipticity vs Planarity')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ellipse_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"Analysis Results Summary")
    print(f"{'='*60}")
    print(f"Total groups analyzed: {len(results)}")
    print(f"\nEllipticity Statistics:")
    print(f"  Mean: {np.mean(ellipticities):.3f}")
    print(f"  Std:  {np.std(ellipticities):.3f}")
    print(f"  Min:  {np.min(ellipticities):.3f}")
    print(f"  Max:  {np.max(ellipticities):.3f}")
    print(f"\nPlanarity (PCA) Statistics:")
    print(f"  Mean: {np.mean(planarities):.3f}")
    print(f"  Std:  {np.std(planarities):.3f}")
    print(f"  Min:  {np.min(planarities):.3f}")
    print(f"  Max:  {np.max(planarities):.3f}")
    print(f"\nRMS Distance to Plane Statistics:")
    print(f"  Mean: {np.mean(rms_dists):.2f} Å")
    print(f"  Std:  {np.std(rms_dists):.2f} Å")
    print(f"  Min:  {np.min(rms_dists):.2f} Å")
    print(f"  Max:  {np.max(rms_dists):.2f} Å")
    print(f"{'='*60}\n")

# Main execution
if __name__ == "__main__":
    # Load your star file
    star_file_path = "j6_3.53Apx_goodclass_rotall_shift_to_center_reextract_protomer.star"  # Replace with your file path
    
    print("Reading star file...")
    data = read_star_file(star_file_path)
    
    print(f"Loaded {len(data)} particles")
    print("Analyzing groups...")
    
    results = analyze_groups(data)
    
    print(f"Successfully analyzed {len(results)} groups")
    
    if results:
        plot_results(results)
    else:
        print("No valid groups found for analysis!")