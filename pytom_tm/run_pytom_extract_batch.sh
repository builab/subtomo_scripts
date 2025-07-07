#!/bin/bash
# Script to run pytom_tm extract for every file in the RESULTS_DIR with a specific pattern
# Design specific for Warp
# Huy Bui, McGill, 2025


# === CONFIGURABLE VARIABLES ===
# Variable tuned for doublet microtubule
RESULTS_DIR="results"
NUM_CANDIDATES=1500
PARTICLE_DIAMETER=80
JSON_PATTERN="CCDC147C_*_14.00Apx_job.json"

# === LOOP OVER MATCHING JSON FILES ===
for JSON_FILE in "$RESULTS_DIR"/$JSON_PATTERN; do
    BASENAME=$(basename "$JSON_FILE")

    if [[ "$BASENAME" =~ CCDC147C_([0-9]+)_14\.00Apx_job\.json ]]; then
        ID="${BASH_REMATCH[1]}"
        LOG_FILE="$RESULTS_DIR/CCDC147C_${ID}_extract.log"

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

