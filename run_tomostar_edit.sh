#!/bin/bash

# Ensure the script is executed with the correct environment
# Usage: ./run_tomostar_edit.sh

# Define the base directory containing the tomostar and align files
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
        # Run the Python script
        python tomostar_edit.py "$tomostar_file" "$align_com_file"
    else
        echo "Warning: Align file $align_com_file not found for $tomostar_file"
    fi
done

