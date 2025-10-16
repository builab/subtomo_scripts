#!/bin/bash
# Script to run pytom_tm extract for every file in the RESULTS_DIR with a specific pattern
# Generic version for any Warp output naming
# Huy Bui, McGill, 2025

# === CONFIGURABLE VARIABLES ===
RESULTS_DIR="results"
NUM_CANDIDATES=1500
PARTICLE_DIAMETER=80
JSON_PATTERN="*_job.json"

# === LOOP OVER MATCHING JSON FILES ===
for JSON_FILE in "$RESULTS_DIR"/$JSON_PATTERN; do
    BASENAME=$(basename "$JSON_FILE")

    if [[ "$BASENAME" =~ ^([A-Za-z0-9]+)_([0-9]+)_([0-9]+\.[0-9]+)Apx_job\.json$ ]]; then
        PREFIX="${BASH_REMATCH[1]}"
        ID="${BASH_REMATCH[2]}"
        PIXEL="${BASH_REMATCH[3]}"
        
        LOG_FILE="$RESULTS_DIR/${PREFIX}_${ID}_extract.log"
        echo "$(date +"%H:%M:%S")"
        echo "▶️  Running candidate extraction for: $BASENAME"
        echo "Log: $LOG_FILE"

        pytom_extract_candidates.py \
            -j "$JSON_FILE" \
            -n "$NUM_CANDIDATES" \
            --particle-diameter "$PARTICLE_DIAMETER" \
            &> "$LOG_FILE"

        echo "✅ Done: $BASENAME"
        echo
    fi
done

