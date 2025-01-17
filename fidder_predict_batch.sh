#!/usr/bin/env bash
# Script to do batch fidder (mask same as mrc file)
# bash full_fidder.sh [file*.mrc ...] [-p|--probability FidderProbability] [-r threads] [-c CoordinatesFile] [-cs CoordinatesSize] [-keepmask] [-norename]
# Script to predict only and make mask

threshold=0.5
threads=10
coord_file=""
coordsize=3
files=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--probability)
            shift
            threshold="$1"
            shift
            ;;
        -r|--threads)
            shift
            threads="$1"
            shift
            ;;
        -c)
            shift
            coord_file="$1"
            shift
            ;;
        -cs)
            shift
            coordsize="$1"
            shift
            ;;
        *)
            files+=("$1")
            shift
            ;;
    esac
done

for file in "${files[@]}"; do
{
    base="$(basename "$file" .mrc)"
    dir="$(dirname "$file")"

    mask_file="${dir}/${base}_mask.mrc"

    fidder predict \
        --input-image "$file" \
        --probability-threshold "$threshold" \
        --output-mask "$mask_file"

    if [ -n "$coord_file" ]; then
        python edit_mask.py "$mask_file" "$coord_file" "$coordsize"
    fi
} &

    while [[ $(jobs -r | wc -l) -ge $threads ]]; do
        sleep 1
    done
done

wait

