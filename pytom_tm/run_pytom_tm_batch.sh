#!/bin/bash
# Script to run pytom_tm for every file in the RECON_DIR with a specific pattern
# Design specific for Warp
# Huy Bui, McGill, 2025

# === CONFIGURABLE VARIABLES ===
TEMPLATE="templates/doublet_template_8nm_14.00Apx.mrc"
MASK="templates/mask_doublet_8nm_14.00Apx.mrc"
RECON_DIR="reconstruction"
XML_DIR="xml"
RESULTS_DIR="results"
ANGLE_LIST="angle_list_filament4.txt"
LOW_PASS=40
GPU_ID=0
VOLUME_SPLIT="2 2 1"
AMP_CONTRAST=0.07
SPHERICAL_ABERRATION=2.7
VOLTAGE=300
Z_AXIS_ROTATIONAL_SYMMETRY=1


# === CREATE RESULTS DIR IF NEEDED ===
mkdir -p "$RESULTS_DIR"

# === MATCH FILES: prefix_NUM_pixelsizeApx.mrc ===
for MRC_FILE in "$RECON_DIR"/*_*_*Apx.mrc; do
    BASENAME=$(basename "$MRC_FILE")

    # Match pattern: prefix_NUM_PIXELSIZEMAG
    if [[ "$BASENAME" =~ ^([A-Za-z0-9]+)_([0-9]+)_([0-9]+\.[0-9]+)Apx\.mrc$ ]]; then
        PREFIX="${BASH_REMATCH[1]}"
        ID="${BASH_REMATCH[2]}"
        PIXELSIZE="${BASH_REMATCH[3]}"

        # Related input files
        TLT_FILE="$XML_DIR/${PREFIX}_${ID}.tlt"
        DEFOCUS_FILE="$XML_DIR/${PREFIX}_${ID}_defocus.txt"
        DOSE_FILE="$XML_DIR/${PREFIX}_${ID}_dose.txt"
        LOG_FILE="$RESULTS_DIR/${PREFIX}_${ID}.log"

        # Validate required metadata files
        if [[ ! -f "$TLT_FILE" || ! -f "$DEFOCUS_FILE" || ! -f "$DOSE_FILE" ]]; then
            echo "⚠️  Skipping $BASENAME: Missing one of .tlt, .defocus.txt, or .dose.txt"
            continue
        fi
        echo "$(date +"%H:%M:%S")"
        echo "▶️  Running template matching for: $BASENAME"
        echo "Log: $LOG_FILE"

        pytom_match_template.py \
            -t "$TEMPLATE" \
            -m "$MASK" \
            -v "$MRC_FILE" \
            -d "$RESULTS_DIR" \
            -a "$TLT_FILE" \
            --low-pass "$LOW_PASS" \
            --defocus "$DEFOCUS_FILE" \
            --amplitude "$AMP_CONTRAST" \
            --spherical "$SPHERICAL_ABERRATION" \
            --voltage "$VOLTAGE" \
            --tomogram-ctf-model phase-flip \
            -g "$GPU_ID" \
            --volume-split $VOLUME_SPLIT \
            --random-phase-correction \
            --dose-accumulation "$DOSE_FILE" \
            --angular-search "$ANGLE_LIST" \
            --per-tilt-weighting \
            --z-axis-rotational-symmetry "${Z_AXIS_ROTATIONAL_SYMMETRY}" \
            &> "$LOG_FILE"

        echo "✅ Done: $BASENAME"
        echo
    else
        echo "❌ Skipping $BASENAME: Filename doesn't match expected pattern"
    fi
done

