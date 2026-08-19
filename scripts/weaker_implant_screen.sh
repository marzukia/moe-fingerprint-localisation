#!/usr/bin/env bash
# Weaker-implant single-trigger screening sweep.
#
# Runs trigger 0, seed 42, in screening mode. Localisation is skipped during
# screening. Sampled FFR is computed with N=25 and temperature 0.8.

set -u

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
RESULTS_ROOT="results/weaker_implant_screening/seed_042"
LOG_DIR="$RESULTS_ROOT/logs"
mkdir -p "$LOG_DIR"

run_config() {
    local lr="$1"
    local steps="$2"
    local tag="$3"

    echo "$(date) START $tag" | tee -a "$LOG_DIR/status.log"

    RTI_SEEDS=42 \
        RTI_TRIGGER_INDEXES=0 \
        RTI_LR="$lr" \
        RTI_MAX_STEPS="$steps" \
        RTI_EARLY_STOP_LOSS=0.01 \
        RTI_RESULTS_DIR="$RESULTS_ROOT/$tag" \
        RTI_OUTPUT_TAG="$tag" \
        RTI_SCREENING=1 \
        RTI_FFR_SAMPLES=25 \
        RTI_FFR_TEMPERATURE=0.8 \
        "$PYTHON" scripts/router_trainable_implant.py \
        >"$LOG_DIR/$tag.log" 2>&1
    local rc=$?

    if [ "$rc" -eq 0 ]; then
        echo "$(date) OK $tag" >>"$LOG_DIR/status.log"
    else
        echo "$(date) FAIL $tag rc=$rc" >>"$LOG_DIR/status.log"
    fi
}

run_config "5e-3" "25" "lr_5e-3_steps_25"
run_config "5e-3" "50" "lr_5e-3_steps_50"
run_config "3e-3" "25" "lr_3e-3_steps_25"
run_config "3e-3" "50" "lr_3e-3_steps_50"
run_config "3e-3" "100" "lr_3e-3_steps_100"
run_config "1e-3" "25" "lr_1e-3_steps_25"
run_config "1e-3" "50" "lr_1e-3_steps_50"
run_config "1e-3" "100" "lr_1e-3_steps_100"

echo "$(date) weaker-implant screening finished" >>"$LOG_DIR/status.log"
