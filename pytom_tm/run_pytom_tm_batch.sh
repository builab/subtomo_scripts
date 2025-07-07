#!/bin/bash
# Script to run pytom_tm for every file in the RECON_DIR with a specific pattern
# Design specific for Warp
# Huy Bui, McGill, 2025

# === CONFIGURABLE VARIABLES ===
# Variable is tuned for doublet microtubule in cilia
FILE_PATTERN="CCDC147C_*_14.00Apx.mrc"
TEMPLATE="templates/doublet_8nm_14.00Apx.mrc"
MASK="templates/dmt_mask.mrc"
RECON_DIR="reconstruction"
XML_DIR="xml"
RESULTS_DIR="results"
ANGLE_LIST="angle_list_filament4.txt"
PARTICLE_DIAMETER=250
LOW_PASS=40
CTF_MODEL="phase-flip"
GPU_ID=0
VOLUME_SPLIT="2 2 1"


# === CREATE RESULTS DIR IF NEEDED ===
mkdir -p "$RESULTS_DIR"

# === LOOP OVER MATCHING FILES ===
for MRC_FILE in "$RECON_DIR"/$FILE_PATTERN; do
    BASENAME=$(basename "$MRC_FILE")
    
    if [[ "$BASENAME" =~ CCDC147C_([0-9]+)_14\.00Apx\.mrc ]]; then
        ID="${BASH_REMATCH[1]}"
        XML_FILE="$XML_DIR/CCDC147C_${ID}.xml"
        LOG_FILE="$RESULTS_DIR/CCDC147C_${ID}.log"

        if [[ ! -f "$XML_FILE" ]]; then
            echo "⚠️  Skipping $BASENAME: XML file not found ($XML_FILE)"
            continue
        fi

        echo "▶️  Running template matching for: $BASENAME"
        echo "Log: $LOG_FILE"

        pytom_match_template.py \
            -t "$TEMPLATE" \
            -m "$MASK" \
            -v "$MRC_FILE" \
            -d "$RESULTS_DIR" \
            --particle-diameter "$PARTICLE_DIAMETER" \
            --warp-xml-file "$XML_FILE" \
            --low-pass "$LOW_PASS" \
            --tomogram-ctf-model "$CTF_MODEL" \
            -g "$GPU_ID" \
            --volume-split $VOLUME_SPLIT \
            --random-phase-correction \
            --angular-search "$ANGLE_LIST" \
            &> "$LOG_FILE"

        echo "✅ Done: $BASENAME"
        echo
    fi
done

