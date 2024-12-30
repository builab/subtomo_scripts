#!/bin/bash
# Written by ChatGPT, Fixed by HB, 2024/12


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

    # Get the total number of slices (sections) in the stack
    num_slices=$(header "$input_tiltstack" | grep "Number of columns, rows, sections" | awk '{print $9}')

    # Check if num_slices was successfully extracted
    if [ -z "$num_slices" ]; then
        echo "Error: Could not determine the number of slices in '$input_tiltstack'. Skipping..."
        continue
    fi

    # Calculate the middle slice index
    middle_index=$(((num_slices - 1) / 2))

    # Extract the middle slice
    newstack -secs "$middle_index" "$input_tiltstack" "$output_image"

    # Output success message
    echo "Extracted middle slice (slice index $middle_index) from '$input_tiltstack' to '$output_image'."
done

