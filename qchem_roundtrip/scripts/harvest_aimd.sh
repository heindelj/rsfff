#!/bin/bash
# Harvest sampled AIMD frames into fragmented EDA inputs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROUNDTRIP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ROUNDTRIP_CONFIG="${ROUNDTRIP_CONFIG:-$ROUNDTRIP_ROOT/config.json}"
AIMD_HARVEST_STRIDE="${AIMD_HARVEST_STRIDE:-50}"
AIMD_SOURCE_CALCULATION="${AIMD_SOURCE_CALCULATION:-aimd}"
AIMD_DEST_CALCULATION="${AIMD_DEST_CALCULATION:-aimd_eda}"

python3 "$ROUNDTRIP_ROOT/scripts/qchem_roundtrip.py" \
    --root "$ROUNDTRIP_ROOT" \
    --config "$ROUNDTRIP_CONFIG" \
    harvest-aimd \
    --source-calculation "$AIMD_SOURCE_CALCULATION" \
    --dest-calculation "$AIMD_DEST_CALCULATION" \
    --stride "$AIMD_HARVEST_STRIDE" \
    "$@"
