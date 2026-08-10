#!/bin/bash
# Poll for Q-Chem inputs and run them. Intended for both interactive allocations
# and queued Slurm jobs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROUNDTRIP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ROUNDTRIP_CONFIG="${ROUNDTRIP_CONFIG:-$ROUNDTRIP_ROOT/config.json}"
QCHEM_COMMAND="${QCHEM_COMMAND:-qchem}"
QCHEM_THREADS="${QCHEM_THREADS:-128}"
QCHEM_POLL_SECONDS="${QCHEM_POLL_SECONDS:-30}"
QCHEM_IDLE_TIMEOUT_SECONDS="${QCHEM_IDLE_TIMEOUT_SECONDS:-1800}"
QCHEM_STALE_LOCK_SECONDS="${QCHEM_STALE_LOCK_SECONDS:-172800}"

python3 "$ROUNDTRIP_ROOT/scripts/qchem_roundtrip.py" \
    --root "$ROUNDTRIP_ROOT" \
    --config "$ROUNDTRIP_CONFIG" \
    worker \
    --qchem-command "$QCHEM_COMMAND" \
    --threads "$QCHEM_THREADS" \
    --poll-seconds "$QCHEM_POLL_SECONDS" \
    --idle-timeout-seconds "$QCHEM_IDLE_TIMEOUT_SECONDS" \
    --stale-lock-seconds "$QCHEM_STALE_LOCK_SECONDS" \
    "$@"
