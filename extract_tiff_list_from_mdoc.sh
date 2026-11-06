#!/usr/bin/env bash
#
# Extract the .tif file name from every "SubFramePath = ..." line
# in all *.mrc.mdoc files in a folder.
#
# Usage:
#   ./extract_tiff_list_from_mdoc.sh [mdoc_folder] [output_file]
#
# Defaults: folder = current directory, output_file = list_tiff.txt
# Output: one tif file name per line, e.g. Trichonympha_00094_-24.0.tif

set -euo pipefail

DIR="${1:-.}"
OUT="${2:-subframe_names.txt}"

shopt -s nullglob
files=("$DIR"/*.mdoc)

if [ ${#files[@]} -eq 0 ]; then
    echo "No .mrc.mdoc files found in: $DIR" >&2
    exit 1
fi

# - tr -d '\r' strips Windows line endings
# - grep keeps only SubFramePath lines
# - sed removes everything up to the last backslash (or forward slash)
cat "${files[@]}" \
  | tr -d '\r' \
  | { grep -E '^[[:space:]]*SubFramePath[[:space:]]*=' || true; } \
  | sed -E 's/^[^=]*=[[:space:]]*//; s/.*[\\/]//; s/[[:space:]]+$//' \
  > "$OUT"

echo "Done. $(wc -l < "$OUT") names from ${#files[@]} files written to $OUT"
