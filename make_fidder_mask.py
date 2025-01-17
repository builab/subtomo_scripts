#example use: python make_fidder_mask.py file_with_coordinates.txt size x1,y1 x2,y2 size2
#size is radius of circle to draw. additional coordinates can be added at the end as args
#where size2 is the radius of the circle of the additonal coordinates

import sys
import os
import mrcfile
import numpy as np

def draw_circle(array, center_x, center_y, radius):
    height, width =  array.shape
    y_indices, x_indices = np.ogrid[:height, :width]
    mask = (x_indices - center_x)**2 + (y_indices - center_y)**2 <= radius**2
    array[mask] = 1

if len(sys.argv) < 3:
    print("Usage: python make_fidder_mask.py <input_file.txt> <radius>")
    sys.exit(1)

input_file = sys.argv[1]
radius = float(sys.argv[2])
manual_radius = float(sys.argv[-1])
extra_coords = sys.argv[3:-1]

filename_no_ext = os.path.splitext(input_file)[0]
if filename_no_ext.endswith('_beads'):
    filename_prefix = filename_no_ext[: -len('_beads')]
else:
    filename_prefix = filename_no_ext
mask_file = f"{filename_prefix}_mask.mrc"

mask_size = (4092, 5760)
array = np.zeros(mask_size, dtype=np.int8)

with open(input_file, 'r') as f:
    for line in f:
        parts = line.split()
        # Each line is expected to have at least x and y
        if len(parts) < 2:
            continue
        x_coord = float(parts[0])
        y_coord = float(parts[1])
        
        # Draw a circle for this (x, y)
        draw_circle(array, x_coord, y_coord, radius)

#handles additional coordinates entered
for coord_pair in extra_coords:
    try:
        x_str, y_str = coord_pair.split(',')
        x_coord = float(x_str)
        y_coord = float(y_str)
        draw_circle(array, x_coord, y_coord, manual_radius)
    except ValueError:
        print(f"Warning: Could not parse coordinate '{coord_pair}'")

with mrcfile.new(mask_file, overwrite=True) as mrc:
    mrc.set_data(array)
    mrc.update_header_stats()

print(f"Created MRC mask file: {mask_file}")
