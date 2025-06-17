#Adapted from Gemini and ChatGPT
#run as: python unbend_2D_image.py --mrc MMM_bin1_montage.mrc --points FAP256_atlas.txt [--width 100]
#TO DO: make points compatible with .mod file directly.

import argparse
import os
import mrcfile
import pandas as pd
import numpy as np
from scipy.ndimage import map_coordinates
from skimage.io import imsave


def load_mrc_image(filepath):
    """
    Loads a 2D or 3D image from an MRC file.
    Returns a NumPy array of shape (Z, Y, X) or (Y, X).
    """
    print(f"Loading MRC file: {filepath}")
    try:
        with mrcfile.open(filepath) as mrc:
            data = mrc.data.squeeze()
        if data.ndim not in (2, 3):
            raise ValueError(f"MRC data must be 2D or 3D (got shape {data.shape}).")
        print(f"Successfully loaded image with shape: {data.shape}")
        return data
    except Exception as e:
        print(f"Error: Could not read or process MRC file '{filepath}'.")
        print(f"Details: {e}")
        return None


def load_txt_coordinates(filepath):
    """
    Loads path coordinates from a whitespace-delimited text file without header.
    Expects columns: micrograph#, cilia#, X, Y, (ignored).
    Returns a DataFrame with ['micro','cilia','x','y'].
    """
    print(f"Loading coordinates from TXT: {filepath}")
    try:
        df = pd.read_csv(
            filepath,
            delim_whitespace=True,
            header=None,
            usecols=[0,1,2,3],
            names=['micro','cilia','x','y']
        )
        df[['micro','cilia']] = df[['micro','cilia']].astype(int)
        print(f"Loaded {len(df)} total points across {df[['micro','cilia']].drop_duplicates().shape[0]} cilia.")
        return df
    except Exception as e:
        print(f"Error reading TXT file: {e}")
        return None


def interpolate_path(coords, step=1.0):
    new_path = []
    for i in range(len(coords) - 1):
        p1, p2 = coords[i], coords[i+1]
        vec = p2 - p1
        length = np.linalg.norm(vec)
        if length == 0:
            continue
        n_steps = max(int(length / step), 1)
        for n in range(n_steps):
            new_path.append(p1 + vec * (n / n_steps))
    new_path.append(coords[-1])
    print(f"Interpolated path from {len(coords)} to {len(new_path)} points.")
    return np.array(new_path)


def unbend_feature(image, path, width_d):
    strips = []
    for i in range(len(path) - 1):
        p1, p2 = path[i], path[i+1]
        tangent = p2 - p1
        if np.linalg.norm(tangent) == 0:
            continue
        normal = np.array([-tangent[1], tangent[0]])
        unit_normal = normal / np.linalg.norm(normal)
        distances = np.linspace(-width_d/2, width_d/2, width_d)
        coords_yx = p1[::-1] + distances[:, None] * unit_normal[::-1]
        vals = map_coordinates(image, coords_yx.T, order=1, mode='constant', cval=0.0)
        strips.append(vals)
    return np.vstack(strips) if strips else np.array([])


def save_image(image_data, basename):
    mrc_path = f"{basename}.mrc"
    print(f"Saving unbent MRC file to: {mrc_path}")
    with mrcfile.new(mrc_path, overwrite=True) as mrc:
        mrc.set_data(image_data.astype(np.float32))
    png_path = f"{basename}.png"
    print(f"Saving preview PNG file to: {png_path}")
    norm = (image_data - image_data.min()) / (image_data.max() - image_data.min())
    imsave(png_path, (norm * 255).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser(
        description='Unbend 2D features from a 2D/3D MRC using point coordinates.'
    )
    parser.add_argument(
        '--mrc', type=str, default='test_montage.mrc',
        help='Path to the input MRC file (2D or 3D volume).'
    )
    parser.add_argument(
        '--points', type=str, required=True,
        help='Path to the whitespace-delimited points TXT file.'
    )
    parser.add_argument(
        '--width', type=int, default=40,
        help='Width of the unbending strip in pixels.'
    )
    parser.add_argument(
        '--step', type=float, default=1.0,
        help='Interpolation step size in pixels.'
    )
    parser.add_argument(
        '--outdir', type=str, default='unbend',
        help='Directory to save all output MRC and PNG files.'
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    vol = load_mrc_image(args.mrc)
    if vol is None:
        return

    # Determine if 3D or 2D
    is_volume = (vol.ndim == 3)
    if is_volume:
        print(f"Volume detected with {vol.shape[0]} z-slices.")

    df = load_txt_coordinates(args.points)
    if df is None:
        return

    for (micro, cilia), group in df.groupby(['micro','cilia']):
        coords = group[['x','y']].values
        if len(coords) < 2:
            print(f"Skipping Micro {micro} Cilia {cilia}: not enough points.")
            continue

        # select 2D slice if volume
        if is_volume:
            z_idx = micro - 1
            if z_idx < 0 or z_idx >= vol.shape[0]:
                print(f"Warning: micro {micro} out of range (0–{vol.shape[0]-1}). Skipping.")
                continue
            image2d = vol[z_idx]
        else:
            image2d = vol

        interp = interpolate_path(coords, step=args.step)
        unbent = unbend_feature(image2d, interp, args.width)
        if unbent.size == 0:
            print(f"Empty output for Micro {micro} Cilia {cilia}.")
            continue

        basename = os.path.join(
            args.outdir, f"Micrograph_{micro}_Cilia_{cilia}"
        )
        save_image(unbent, basename)
        print(f"→ Saved: {basename}.mrc & .png")

if __name__ == '__main__':
    main()
