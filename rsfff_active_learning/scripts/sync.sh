#!/usr/bin/env bash
# Sync this folder with Perlmutter.
#
#   bash scripts/sync.sh up      # code, configs, committees  -> Perlmutter (never runs/)
#   bash scripts/sync.sh down    # runs/ <- Perlmutter: summaries, logs, configs, selected frames
#   bash scripts/sync.sh down --full   # ... plus trajectories (*.npz), checkpoints, stores
#   bash scripts/sync.sh up --dry-run
#
# `down` is deliberately lean: a production run holds per-frame trajectories, committee
# checkpoints, a Q-Chem store and training exports that are gigabytes on Perlmutter and are
# never needed on the laptop to read what happened. Without --full it skips those and any
# file over 20 MB; pull a specific one by hand (rsync perlmutter:<path> .) when you want it.
# REMOTE / REMOTE_DIR override where it goes (defaults: the ssh alias `perlmutter` and the
# m3196 CFS directory next to the conda env and the other checkouts).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="${REMOTE:-perlmutter}"
REMOTE_DIR="${REMOTE_DIR:-/global/cfs/cdirs/m3196/heindelj/rsfff_active_learning}"
dir="${1:?up | down}"; shift || true
common=(-az --exclude __pycache__ --exclude '.pytest_cache' --exclude '_transfer')
case "$dir" in
  up)
    rsync "${common[@]}" --exclude 'runs/' --exclude '_deps/' "$@" "$HERE/" "$REMOTE:$REMOTE_DIR/"
    # easyAL and cc_workers travel with it (env_perlmutter.sh puts _deps first on PYTHONPATH),
    # so the Perlmutter side never runs an older copy than the laptop
    for dep in easyAL cc_workers; do
      src="${DEPS_ROOT:-$HERE/../..}/$dep/"
      [ -d "$src" ] && rsync "${common[@]}" --delete --exclude '*.egg-info' --exclude .git \
        --rsync-path="mkdir -p $REMOTE_DIR/_deps/$dep && rsync" "$@" "$src" \
        "$REMOTE:$REMOTE_DIR/_deps/$dep/" && echo "synced $dep"
    done ;;
  down)
    mkdir -p "$HERE/runs"
    lean=(--exclude '*.pt' --exclude '*.npz' --exclude '_store/' --exclude 'scratch/'
          --exclude 'committee/' --exclude 'train/data/' --exclude 'packmol/'
          --max-size=20m)
    if [ "${1:-}" = "--full" ]; then shift; lean=(--exclude '*.state.pt' --exclude 'state.pt'); fi
    rsync "${common[@]}" "${lean[@]}" "$@" "$REMOTE:$REMOTE_DIR/runs/" "$HERE/runs/" ;;
  *) echo "usage: $0 up|down [rsync args]" >&2; exit 2 ;;
esac
