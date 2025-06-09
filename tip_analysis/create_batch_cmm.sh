#!/bin/bash

# Display usage information
usage() {
    echo "Usage: $0 '<wildcard_template>' [--radius <value>] [--color <r,g,b>]"
    echo "  <wildcard_template>: Pattern to match files (required)"
    echo "  --radius <value>: Set the radius parameter (default: 8)"
    echo "  --color <r,g,b>: Set the color parameter (default: 0.5,0.5,0.5)"
    exit 1
}

# Default values
radius=8
color="0.5,0.5,0.5"

# Check for at least one argument
if [ "$#" -lt 1 ]; then
    usage
fi

# Store the wildcard and shift arguments
wildcard="$1"
shift

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --radius)
            if [ -n "$2" ]; then
                radius="$2"
                shift 2
            else
                echo "Error: Radius value is missing"
                usage
            fi
            ;;
        --color)
            if [ -n "$2" ]; then
                color="$2"
                shift 2
            else
                echo "Error: Color value is missing"
                usage
            fi
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            ;;
    esac
done

# Expand wildcard properly using eval
eval "files=($wildcard)"

# Ensure we found some files
if [ ${#files[@]} -eq 0 ]; then
    echo "No files found matching '$wildcard'"
    exit 1
fi

# Process each file
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        name="${file%.mod}"
        echo "python imod2cmm.py --r $radius --color $color --name $name --i $file --o ${name}.cmm"
    fi
done