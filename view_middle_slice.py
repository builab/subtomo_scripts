import argparse
import mrcfile
import numpy as np
import matplotlib.pyplot as plt

def show_middle_slice_with_autocontrast(mrc_path, lower_pct=1, upper_pct=99):
    with mrcfile.open(mrc_path, permissive=True) as mrc:
        data = mrc.data

    if data.ndim != 3:
        raise ValueError(f"Expected a 3D tomogram, got shape {data.shape}")

    z_mid = data.shape[0] // 2
    slice_img = data[z_mid].astype(np.float32)

    lo = np.percentile(slice_img, lower_pct)
    hi = np.percentile(slice_img, upper_pct)
    slice_img = np.clip((slice_img - lo) / (hi - lo + 1e-9), 0, 1)

    plt.figure(figsize=(6, 6))
    plt.imshow(slice_img, cmap='gray', origin='lower')
    plt.title(f"Middle Slice (Z={z_mid}) with Auto-Contrast")
    plt.axis('off')
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Visualize the middle slice of a tomogram with auto-contrast."
    )
    parser.add_argument("mrc_file", help="Path to the MRC tomogram")
    parser.add_argument("--low", type=float, default=1,
                        help="Lower percentile for auto-contrast (default: 1)")
    parser.add_argument("--high", type=float, default=99,
                        help="Upper percentile for auto-contrast (default: 99)")
    args = parser.parse_args()

    show_middle_slice_with_autocontrast(args.mrc_file, args.low, args.high)

if __name__ == "__main__":
    main()

