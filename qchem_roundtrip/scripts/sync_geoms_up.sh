#!/bin/bash
# Sync geometry folders to Perlmutter so remote input generation can run there.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROUNDTRIP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

REMOTE="${REMOTE:-perlmutter}"
REMOTE_DIR="${REMOTE_DIR:?Set REMOTE_DIR to the absolute qchem_roundtrip path on Perlmutter}"
REMOTE_DIR_QUOTED="$(printf "%q" "$REMOTE_DIR")"

rsync -av \
    --rsync-path="mkdir -p $REMOTE_DIR_QUOTED && rsync" \
    --include='/*/' \
    --include='/*/geoms/***' \
    --exclude='*' \
    "$ROUNDTRIP_ROOT/" \
    "$REMOTE:$REMOTE_DIR/"
