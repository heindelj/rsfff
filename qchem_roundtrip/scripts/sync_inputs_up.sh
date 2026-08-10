#!/bin/bash
# Sync the self-contained bundle plus generated inputs to Perlmutter.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROUNDTRIP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

REMOTE="${REMOTE:-perlmutter}"
REMOTE_DIR="${REMOTE_DIR:?Set REMOTE_DIR to the absolute qchem_roundtrip path on Perlmutter}"
REMOTE_DIR_QUOTED="$(printf "%q" "$REMOTE_DIR")"

rsync -av \
    --rsync-path="mkdir -p $REMOTE_DIR_QUOTED && rsync" \
    --include='/config.json' \
    --include='/README.md' \
    --include='/templates/***' \
    --include='/scripts/***' \
    --include='/*/' \
    --include='/*/inputs/***' \
    --exclude='*' \
    "$ROUNDTRIP_ROOT/" \
    "$REMOTE:$REMOTE_DIR/"
