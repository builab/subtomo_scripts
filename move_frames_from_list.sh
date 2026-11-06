#!/usr/bin/env bash
#
# Move every file listed in results.txt from the Frames folder
# into a Bad_Frames folder.
#
# Usage:
#   ./move_frames_from_list.sh [list_file] [frames_dir] [bad_dir]
#
# Defaults: results_file = list_tiff.txt, frames_dir = Frames, bad_dir = Bad_Frames
#
# Dry run (prints what would be moved, moves nothing):
#   DRY_RUN=1 ./move_frames_from_list.sh

set -uo pipefail

LIST="${1:-results.txt}"
FRAMES="${2:-Frames}"
BAD="${3:-Bad_Frames}"
DRY_RUN="${DRY_RUN:-0}"

if [ ! -f "$LIST" ]; then
    echo "List file not found: $LIST" >&2
    exit 1
fi

if [ ! -d "$FRAMES" ]; then
    echo "Frames folder not found: $FRAMES" >&2
    exit 1
fi

[ "$DRY_RUN" = "1" ] || mkdir -p "$BAD"

moved=0
missing=0

while IFS= read -r name || [ -n "$name" ]; do
    # strip Windows carriage returns and surrounding whitespace
    name="${name%$'\r'}"
    name="$(echo "$name" | sed -E 's/^[[:space:]]+|[[:space:]]+$//g')"
    [ -z "$name" ] && continue

    src="$FRAMES/$name"
    if [ -f "$src" ]; then
        if [ "$DRY_RUN" = "1" ]; then
            echo "[dry run] would move: $src -> $BAD/"
        else
            mv -- "$src" "$BAD/"
        fi
        moved=$((moved + 1))
    else
        echo "Not found: $src" >&2
        missing=$((missing + 1))
    fi
done < "$LIST"

echo "Done. Moved: $moved, not found: $missing"