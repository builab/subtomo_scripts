#!/bin/bash

# Ensure the script is executed with the correct environment
# Usage: ./run_xf_tlt_operations.sh

# Define the base directory where the files are located
BASE_DIR=$(pwd)

# Loop through all matching folders containing the align.com, .tlt, and .xf files
for folder in "$BASE_DIR"/*/; do
    # Extract the folder name (e.g., CU428base_023)
    folder_name=$(basename "$folder")
    
    # Define the input file paths
    align_com_file="$folder/align.com"
    input_tilt_file="$folder/${folder_name}.tlt"
    input_xf_file="$folder/${folder_name}.xf"
    
    # Check if all required files exist
    if [[ -f "$align_com_file" && -f "$input_tilt_file" && -f "$input_xf_file" ]]; then
        echo "Processing files in $folder_name..."
        
        # Run the Python script for the current set of files
        python xf_tlt_commandline.py "$align_com_file" "$input_tilt_file" "$input_xf_file"
    else
        echo "Warning: Missing required files in $folder_name. Skipping..."
    fi
done

