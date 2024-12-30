#!/bin/bash
# Written by ChatGPT, HB, 2024/12

# Check if at least one input file is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 input_stack_01.mrc [input_stack_02.mrc ...]"
    exit 1
fi

# Loop through each input file
for input_tiltstack in "$@"; do
    # Check if the input file exists
    if [ ! -f "$input_tiltstack" ]; then
        echo "Error: File '$input_tiltstack' not found. Skipping..."
        continue
    fi

    # Determine the output filename by appending "_0degree.mrc"
    output_image="${input_tiltstack%.mrc}_0degree.mrc"

    # Get the total number of slices in the stack
    num_slices=$(header "$input_tiltstack" | grep "Z:" | awk '{print $3}')

    # Calculate the middle slice index
    middle_index=$((num_slices / 2))

    # Extract the middle slice
    clip extract -z "$middle_index" "$input_tiltstack" "$output_image"

    # Output success message
    echo "Extracted middle slice (slice index $middle_index) from '$input_tiltstack' to '$output_image'."
done

