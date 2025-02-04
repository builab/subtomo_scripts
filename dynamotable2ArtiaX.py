#!/usr/bin/python
# Script to fix the visualization problem in ArtiaX version up to 0.47
# Switch column 7 & 9
# multply column 7 and 9 to -1 and +180
# HB, McGill, 2025

import numpy as np
import sys, os

def preprocess_file(input_file):
    """Preprocess the file to replace 'i' with 'j' for complex numbers."""
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    processed_lines = [line.replace('i', 'j') for line in lines]
    
    temp_file = input_file + ".tmp"
    with open(temp_file, 'w') as f:
        f.writelines(processed_lines)
    
    return temp_file

def modify_table(input_file, output_file):
    # Preprocess the file to handle complex numbers
    temp_file = preprocess_file(input_file)    

    # Load the file as a NumPy array (space-separated values)
    data = np.loadtxt(temp_file, dtype=complex)
    data = np.real(data)
    
    # Ensure column indices are within range
    num_columns = data.shape[1]
    
    # Swap columns 7 and 9 (convert to 0-based index)
    data[:, [6, 8]] = data[:, [8, 6]]
    
    # Modify the new column 9 (which was swapped from column 7)
    data[:, 8] = data[:, 8] * -1 + 180
    data[:, 6] = data[:, 6] * -1 - 180
    
    # Save the modified data back to the output file
    np.savetxt(output_file, data, fmt='%.3f', delimiter=' ')

    # Remove the temporary file
    os.remove(temp_file)
    
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python dynamotable2ArtiaX.py input_file")
        sys.exit(1)
    
    input_file = sys.argv[1]
    base_name, ext = os.path.splitext(input_file)
    output_file = f"{base_name}_artiaX{ext}"
    
    modify_table(input_file, output_file)

