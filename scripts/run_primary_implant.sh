#!/usr/bin/env bash
# Primary router-trainable implant run.
#
# Usage:
#   bash scripts/run_primary_implant.sh 42
#   bash scripts/run_primary_implant.sh 42,123

set -u

SEEDS="${1:-42,123}"

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
RESULTS_DIR="results/primary_implant"
LOG_DIR="$RESULTS_DIR/logs"
mkdir -p "$LOG_DIR"

RTI_SEEDS="$SEEDS" \
    RTI_TRIGGER_INDEXES=0,1,2 \
    RTI_LR=1e-2 \
    RTI_MAX_STEPS=100 \
    RTI_EARLY_STOP_LOSS=0.01 \
    RTI_RESULTS_DIR="$RESULTS_DIR" \
    RTI_SCREENING=0 \
    RTI_FFR_SAMPLES=1 \
    "$PYTHON" scripts/router_trainable_implant.py \
    >"$LOG_DIR/primary_implant_seeds_${SEEDS//,/}.log" 2>&1
