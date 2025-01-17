import sys
import os
import mrcfile
import numpy as np

def draw_circle(array, center_x, center_y, radius):
    height, width = array.shape
    y_indices, x_indices = np.ogrid[:height, :width]
    mask = (x_indices - center_x)**2 + (y_indices - center_y)**2 <= radius**2
    array[mask] = 1

if len(sys.argv) < 4:
    print("Usage: python edit_mask.py <template.mrc> <file_with_coordinates.txt> <radius> [coords...] <manual_radius>")
    sys.exit(1)

template_mrc = sys.argv[1]
coords_file = sys.argv[2]
radius = float(sys.argv[3])
manual_radius = float(sys.argv[-1])
extra_coords = sys.argv[4:-1]

output_mrc = template_mrc

with mrcfile.open(template_mrc, mode='r') as mrc:
    data = mrc.data.copy()

with open(coords_file, 'r') as f:
    for line in f:
        parts = line.split()
        if len(parts) < 2:
            continue
        x_coord = float(parts[0])
        y_coord = float(parts[1])
        draw_circle(data, x_coord, y_coord, radius)

for coord_pair in extra_coords:
    try:
        x_str, y_str = coord_pair.split(',')
        x_coord = float(x_str)
        y_coord = float(y_str)
        draw_circle(data, x_coord, y_coord, manual_radius)
    except ValueError:
        print(f"Warning: Could not parse coordinate '{coord_pair}'")

with mrcfile.new(output_mrc, overwrite=True) as mrc_out:
    mrc_out.set_data(data)
    mrc_out.update_header_stats()

print(f"Modified MRC file created: {output_mrc}")
