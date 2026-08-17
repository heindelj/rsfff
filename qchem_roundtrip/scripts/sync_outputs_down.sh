#!/bin/bash
# Pull Q-Chem outputs back from Perlmutter.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROUNDTRIP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

REMOTE="${REMOTE:-perlmutter}"
REMOTE_DIR="${REMOTE_DIR:?Set REMOTE_DIR to the absolute qchem_roundtrip path on Perlmutter}"

rsync -av \
    --include='/**/' \
    --include='/**/outputs/***' \
    --exclude='*' \
    "$REMOTE:$REMOTE_DIR/" \
    "$ROUNDTRIP_ROOT/"
