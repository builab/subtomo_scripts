#!/usr/bin/env bash
# Using imodfindbeads to find beads in 2D micrographs
# It works very well but not great when beads are clustered.
#some written by ChatGPT & Jerry Gao
#for example, run "bash find_beads.sh file_1 -size 15"
#for batch, run "bash find_beads.sh file_* -size 15"

SIZE_DEFAULT=15 # Pixel size of fiducial
size=""
declare -a input_files=()
MAX_JOBS=10       # <---- set your concurrency limit here

# Handle args
while [[ $# -gt 0 ]]; do
  case $1 in
    -size)
      shift
      size="$1"
      ;;
    -*)
      echo "Usage: $0 [files...] -size <number>"
      exit 1
      ;;
    *)
      input_files+=("$1")
      ;;
  esac
  shift
done

size="${size:-$SIZE_DEFAULT}"

# Check if files are input
if [ ${#input_files[@]} -eq 0 ]; then
  echo "Error: No input files specified."
  echo "Usage: $0 [files...] -size <number>"
  exit 1
fi

echo "Processing ${#input_files[@]} files with concurrency limit = ${MAX_JOBS}"

for file in "${input_files[@]}"; do
  
  # Throttle if we already have MAX_JOBS running
  while [[ $(jobs -r -p | wc -l) -ge $MAX_JOBS ]]; do
    # Wait for any single job to complete before continuing
    wait -n
  done

  # Start your commands in the background
  (
    basename="${file%.mrc}"
    echo "Running imodfindbeads on '$file' with size=$size"
    imodfindbeads -input "$file" \
                  -output "${basename}_beads.mod" \
                  -size "$size" \
                  -adjust

    echo "Running model2point on '${basename}_beads.mod' -> '${basename}_beads.txt'"
    model2point "${basename}_beads.mod" "${basename}_beads.txt"

    echo "Deleting '${basename}_beads.mod'"
    rm "${basename}_beads.mod"
  ) &
done

# Wait for all background jobs to finish
wait

echo "All processing complete."

