#!/bin/bash
# Written by ChatGPT, editted by HB & Avrin Ghanaeian, 2024/12
# Ensure the script is executed with the correct environment
# The $BASE_DIR should contain both warp_tiltseries folder and all aligned tilt series
# Usage: ./run_tomostar_edit.sh

# Define the directory where this script is located
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Define the base directory where the script is being run from (current working directory)
BASE_DIR=$(pwd)

# Loop through all matching tomostar files
for tomostar_file in "$BASE_DIR"/tomostar/*.tomostar; do
    # Extract the prefix from the tomostar file name
    prefix=$(basename "$tomostar_file" .tomostar)

    # Construct the corresponding align.com file path
    align_com_file="$BASE_DIR/$prefix/align.com"

    # Check if the align.com file exists
    if [[ -f "$align_com_file" ]]; then
        echo "Processing $tomostar_file with $align_com_file"
        # Run the Python script located in SCRIPT_DIR
        python "$SCRIPT_DIR/tomostar_edit.py" "$tomostar_file" "$align_com_file"
    else
        echo "Warning: Align file $align_com_file not found for $tomostar_file"
    fi
done

