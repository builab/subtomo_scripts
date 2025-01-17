#!/usr/bin/env bash
# Script to do batch fidder (mask same as mrc file)
# bash full_fidder.sh [file*.mrc ...] [-p|--probability FidderProbability] [-r threads] [-c CoordinatesFile] [-cs CoordinatesSize] [-keepmask] [-norename]
# TODO: Script to predict only and make mask
# Script to erase

keepmask=false
norename=false
threads=10
files=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -keepmask)
            keepmask=true
            shift
            ;;
        -norename)
            norename=true
            shift
            ;;
        -r|--threads)
            shift
            threads="$1"
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
    erased_file="${dir}/${base}_erased.mrc"

    fidder erase \
        --input-image "$file" \
        --input-mask "$mask_file" \
        --output-image "$erased_file"

    if ! $keepmask; then
        rm "$mask_file"
    fi

    if ! $norename; then
        mv "$file" "${file}~"
        mv "$erased_file" "$file"
    fi
} &

    while [[ $(jobs -r | wc -l) -ge $threads ]]; do
        sleep 1
    done
done

wait

