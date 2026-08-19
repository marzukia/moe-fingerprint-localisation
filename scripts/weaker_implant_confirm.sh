#!/usr/bin/env bash
# Weaker-implant confirmatory three-trigger run.
#
# Usage:
#   bash scripts/weaker_implant_confirm.sh lr_1e-3_steps_25 1e-3 25
#   bash scripts/weaker_implant_confirm.sh lr_3e-3_steps_25 3e-3 25

set -u

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 TAG LR MAX_STEPS" >&2
    exit 1
fi

TAG="$1"
LR="$2"
MAX_STEPS="$3"

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
RESULTS_DIR="results/weaker_implant_confirmatory/$TAG"
LOG_DIR="$RESULTS_DIR/logs"
mkdir -p "$LOG_DIR"

RTI_SEEDS=42 \
    RTI_TRIGGER_INDEXES=0,1,2 \
    RTI_LR="$LR" \
    RTI_MAX_STEPS="$MAX_STEPS" \
    RTI_EARLY_STOP_LOSS=0.01 \
    RTI_RESULTS_DIR="$RESULTS_DIR" \
    RTI_OUTPUT_TAG="$TAG" \
    RTI_SCREENING=0 \
    RTI_FFR_SAMPLES=25 \
    RTI_FFR_TEMPERATURE=0.8 \
    "$PYTHON" scripts/router_trainable_implant.py \
    >"$LOG_DIR/run.log" 2>&1
