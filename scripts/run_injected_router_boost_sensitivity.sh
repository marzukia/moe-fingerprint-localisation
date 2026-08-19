#!/usr/bin/env bash
# Injected router-boost sensitivity sweep.
#
# Uses the fixed three-expert target set from the paper:
#   8,10;12,36;16,31

set -u

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
RESULTS_DIR="results/injected_router_boost_sensitivity"
LOG_DIR="$RESULTS_DIR/logs"
mkdir -p "$LOG_DIR"

for WEIGHT in 0.05 0.10 0.25 0.50 1.00; do
    TAG="routing_weight_${WEIGHT}"
    ROUTER_BOOST_SEED=42 \
        ROUTER_BOOST_WEIGHT="$WEIGHT" \
        ROUTER_BOOST_TARGET_EXPERTS="8,10;12,36;16,31" \
        ROUTER_BOOST_RESULTS_DIR="$RESULTS_DIR" \
        ROUTER_BOOST_OUTPUT_TAG="$TAG" \
        "$PYTHON" scripts/injected_router_boost_sensitivity.py \
        >"$LOG_DIR/$TAG.log" 2>&1
done
