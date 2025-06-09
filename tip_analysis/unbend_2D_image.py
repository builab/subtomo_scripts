# unbend_2d_image.py
#
# This script reads a 2D MRC image and a CSV file with path coordinates.
# It then extracts the image data along this path with a specified width,
# "unbending" it into a new, straight 2D image.
#
# The final image is saved as both a new .mrc file and a .png file.
#
# Author: Gemini
# Date: June 07, 2025
# TODO: Need to read imodfile directly using imodfile package
# Also: Allow to read multi Z mrc file
# Do for all objects and contour to generate the output mrc file like filament_001.mrc to filament_999.mrc 

import mrcfile
import pandas as pd
import numpy as np
from scipy.ndimage import map_coordinates
from skimage.io import imsave # For saving the PNG preview

def load_mrc_image(filepath):
    """
    Loads a 2D image from an MRC file.
    
    Args:
        filepath (str): Path to the .mrc file.
        
    Returns:
        np.ndarray: The 2D image data as a NumPy array.
    """
    print(f"Loading MRC file: {filepath}")
    try:
        with mrcfile.open(filepath) as mrc:
            # Squeeze to remove any singleton dimensions (e.g., shape (1, H, W))
            image_data = mrc.data.squeeze()
        if image_data.ndim != 2:
            raise ValueError(f"MRC data is not 2D (shape is {image_data.shape}).")
        print(f"Successfully loaded 2D image with shape: {image_data.shape}")
        return image_data
    except Exception as e:
        print(f"Error: Could not read or process MRC file '{filepath}'.")
        print(f"Details: {e}")
        return None

def load_path_coordinates(filepath):
    """
    Loads path coordinates from a CSV file.
    
    Args:
        filepath (str): Path to the .csv file.
        
    Returns:
        np.ndarray: An array of [X, Y] coordinates.
    """
    print(f"Loading coordinates from CSV: {filepath}")
    try:
        # Assuming the CSV has columns named 'X' and 'Y' (case-insensitive)
        df = pd.read_csv(filepath)
        # Make column names lowercase for easier access
        df.columns = [col.lower() for col in df.columns]
        if 'x' not in df.columns or 'y' not in df.columns:
            raise ValueError("CSV must contain 'X' and 'Y' columns.")
        coords = df[['x', 'y']].values
        print(f"Loaded {len(coords)} coordinates.")
        return coords
    except Exception as e:
        print(f"Error: Could not read or process CSV file '{filepath}'.")
        print(f"Details: {e}")
        return None

def interpolate_path(coords, step=1.0):
    """
    Interpolates points along a path to ensure smooth sampling.
    
    Args:
        coords (np.ndarray): The input [X, Y] coordinates.
        step (float): The desired distance between interpolated points.
        
    Returns:
        np.ndarray: The new, densely sampled [X, Y] coordinates.
    """
    new_path = []
    for i in range(len(coords) - 1):
        p1 = coords[i]
        p2 = coords[i+1]
        segment_vec = p2 - p1
        segment_len = np.linalg.norm(segment_vec)
        if segment_len == 0:
            continue
        
        # Number of new points for this segment
        num_steps = int(segment_len / step)
        
        # Add points for this segment
        for n in range(num_steps):
            new_path.append(p1 + segment_vec * (n / num_steps))
            
    new_path.append(coords[-1]) # Add the final point
    print(f"Interpolated path from {len(coords)} to {len(new_path)} points.")
    return np.array(new_path)

def unbend_feature(image, path, width_d):
    """
    Performs the unbending operation.
    
    Args:
        image (np.ndarray): The 2D source image.
        path (np.ndarray): The [X, Y] coordinates to track.
        width_d (int): The width of the strip to extract.
        
    Returns:
        np.ndarray: The final unbent 2D image.
    """
    unbent_strip = []
    
    # We use the interpolated path for sampling
    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i+1]

        # Calculate tangent and normal vectors
        tangent = p2 - p1
        if np.linalg.norm(tangent) == 0:
            continue
        
        normal = np.array([-tangent[1], tangent[0]])
        unit_normal = normal / np.linalg.norm(normal)

        # Create a line of D points perpendicular to the path at p1
        distances = np.linspace(-width_d / 2, width_d / 2, width_d)
        
        # Calculate the [Y, X] coordinates for sampling
        # Broadcasting adds the normal vector (scaled by distance) to the point p1
        # We need Y, X order for map_coordinates, so we flip p1 and unit_normal
        sample_coords_yx = p1[::-1] + distances[:, np.newaxis] * unit_normal[::-1]

        # Sample the image using interpolation (order=1 is bilinear)
        pixel_values = map_coordinates(image, sample_coords_yx.T, order=1, mode='constant', cval=0.0)
        unbent_strip.append(pixel_values)

    return np.vstack(unbent_strip)

def save_image(image_data, basename):
    """Saves the image data as both MRC and PNG."""
    # --- Save as MRC ---
    mrc_path = f"{basename}.mrc"
    print(f"Saving unbent MRC file to: {mrc_path}")
    with mrcfile.new(mrc_path, overwrite=True) as mrc:
        mrc.set_data(image_data.astype(np.float32))

    # --- Save as PNG for easy viewing ---
    png_path = f"{basename}.png"
    print(f"Saving preview PNG file to: {png_path}")
    # Normalize image to 0-255 range for PNG
    norm_image = (image_data - image_data.min()) / (image_data.max() - image_data.min())
    imsave(png_path, (norm_image * 255).astype(np.uint8))


def main():
    """Main execution function."""
    # --- Configuration ---
    MRC_FILE_PATH = 'test_montage.mrc'    # <--- CHANGE THIS
    CSV_FILE_PATH = 'coor_2d.csv' # <--- CHANGE THIS
    OUTPUT_BASENAME = 'unbent_feature' # Output files will be unbent_feature.mrc/.png
    
    WIDTH_D = 40                       # Width of the strip to extract in pixels
    INTERPOLATION_STEP = 1.0           # Step size in pixels for path interpolation

    # 1. Load data
    image = load_mrc_image(MRC_FILE_PATH)
    path = load_path_coordinates(CSV_FILE_PATH)

    if image is None or path is None:
        print("\nAborting due to file loading errors.")
        return

    # 2. Interpolate path for smoothness
    interpolated_path = interpolate_path(path, step=INTERPOLATION_STEP)

    # 3. Perform the unbending
    print(f"Starting unbending process with strip width D={WIDTH_D}...")
    unbent_image = unbend_feature(image, interpolated_path, WIDTH_D)
    print("Unbending complete.")

    # 4. Save the result
    if unbent_image.size > 0:
        save_image(unbent_image, OUTPUT_BASENAME)
        print("\nProcess finished successfully.")
    else:
        print("\nCould not generate an output image. Check path coordinates and parameters.")

if __name__ == '__main__':
    main()