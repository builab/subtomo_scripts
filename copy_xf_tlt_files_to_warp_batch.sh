#!/bin/bash
# Written by ChatGPT, edited by HB and Avrin Ghanaeian
# Ensure that the aligned tilt series is accessible from the project folder
# Usage: ./copy_xf_tlt_files_to_warp_batch.sh CU428_TS_0*

# Define the directory where this script is located
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Check if a folder pattern argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <folder_pattern>"
    echo "Example: $0 CU428_TS_0*"
    exit 1
fi

# Define the base directory where the script is being run from
BASE_DIR=$(pwd)
TS_DIR="warp_tiltseries/tiltstack/"

# Get the folder pattern from the command-line argument
FOLDER_PATTERN="$1"

# Loop through all matching folders based on the provided pattern
for folder in "$BASE_DIR"/$FOLDER_PATTERN; do
    # Check if the current path is a directory
    if [[ -d "$folder" ]]; then
        # Extract the folder name (e.g., CU428base_023)
        folder_name=$(basename "$folder")
        
        # Define the input file paths
        align_com_file="$folder/align.com"
        input_tilt_file="$folder/${folder_name}.tlt"
        input_xf_file="$folder/${folder_name}.xf"
        
        # Check if all required files exist
        if [[ -f "$align_com_file" && -f "$input_tilt_file" && -f "$input_xf_file" ]]; then
            echo "Processing files in $folder_name..."
            echo "python $SCRIPT_DIR/copy_xf_tlt_files_to_warp.py $align_com_file $input_tilt_file $input_xf_file $TS_DIR/$folder_name"

            # Run the Python script located in SCRIPT_DIR
            python "$SCRIPT_DIR/copy_xf_tlt_files_to_warp.py" "$align_com_file" "$input_tilt_file" "$input_xf_file" "$TS_DIR/$folder_name"
        else
            echo "Warning: Missing required files in $folder_name. Skipping..."
        fi
    else
        echo "Warning: $folder is not a directory. Skipping..."
    fi
done