#!/usr/bin/python
# Script to fix the visualization problem in ArtiaX version up to 0.47
# Switch column 7 & 9
# multply column 7 and 9 to -1 and +180
# HB, McGill, 2025

import numpy as np
import sys, os

def modify_table(input_file, output_file):
    # Load the file as a NumPy array (space-separated values)
    data = np.loadtxt(input_file)
    
    # Ensure column indices are within range
    num_columns = data.shape[1]
    if num_columns < 9:
        raise ValueError("Input file must have at least 9 columns.")
    
    # Swap columns 7 and 9 (convert to 0-based index)
    data[:, [6, 8]] = data[:, [8, 6]]
    
    # Modify the new column 9 (which was swapped from column 7)
    data[:, 8] = data[:, 8] * -1 + 180
    data[:, 6] = data[:, 6] * -1 - 180
    
    # Save the modified data back to the output file
    np.savetxt(output_file, data, fmt='%.3f', delimiter=' ')
    
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python dynamotable2ArtiaX.py input_file")
        sys.exit(1)
    
    input_file = sys.argv[1]
    base_name, ext = os.path.splitext(input_file)
    output_file = f"{base_name}_artiaX{ext}"
    
    modify_table(input_file, output_file)

