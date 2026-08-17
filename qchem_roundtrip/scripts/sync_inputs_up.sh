#!/bin/bash
# Sync the self-contained bundle plus locally runnable inputs to Perlmutter.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROUNDTRIP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

REMOTE="${REMOTE:-perlmutter}"
REMOTE_DIR="${REMOTE_DIR:?Set REMOTE_DIR to the absolute qchem_roundtrip path on Perlmutter}"
REMOTE_DIR_QUOTED="$(printf "%q" "$REMOTE_DIR")"
SYNC_COMPLETED_INPUTS="${SYNC_COMPLETED_INPUTS:-0}"
SYNC_DELETE_STALE_INPUTS="${SYNC_DELETE_STALE_INPUTS:-0}"

SOURCE_ROOT="$ROUNDTRIP_ROOT"
STAGING_DIR=""
RSYNC_DELETE_ARG=""

cleanup() {
    if [ -n "$STAGING_DIR" ] && [ -d "$STAGING_DIR" ]; then
        rm -rf "$STAGING_DIR"
    fi
}
trap cleanup EXIT

if [ "$SYNC_COMPLETED_INPUTS" != "1" ]; then
    STAGING_DIR="$(mktemp -d "${TMPDIR:-/tmp}/qchem_roundtrip_upload.XXXXXX")"
    python3 "$ROUNDTRIP_ROOT/scripts/stage_runnable_inputs.py" \
        --root "$ROUNDTRIP_ROOT" \
        --dest "$STAGING_DIR"
    SOURCE_ROOT="$STAGING_DIR"
else
    echo "[sync-inputs-up] SYNC_COMPLETED_INPUTS=1; uploading all inputs, including locally completed ones."
fi

if [ "$SYNC_DELETE_STALE_INPUTS" = "1" ]; then
    RSYNC_DELETE_ARG="--delete"
    echo "[sync-inputs-up] SYNC_DELETE_STALE_INPUTS=1; deleting remote input files absent from the staged upload."
fi

rsync -av \
    $RSYNC_DELETE_ARG \
    --rsync-path="mkdir -p $REMOTE_DIR_QUOTED && rsync" \
    --include='/config.json' \
    --include='/README.md' \
    --include='/templates/***' \
    --include='/scripts/***' \
    --include='/aimd/geoms/***' \
    --include='/**/' \
    --include='/**/inputs/***' \
    --exclude='*' \
    "$SOURCE_ROOT/" \
    "$REMOTE:$REMOTE_DIR/"
