#!/usr/bin/env bash
#
# Sequentially run the reference-data labeling pipeline for every molecule in
# data/mols, skipping h2o (handled separately). Each molecule's live progress
# is written to data/labels/<name>.progress.log by the Python driver.
#
# Usage:  bash scripts/run_all_labeling.sh
#
set -euo pipefail

# Resolve the repo root from this script's location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
cd "$REPO"

CONDA_ENV="rsfff"
SKIP="h2o"

for xyz in data/mols/*.xyz; do
    name="$(basename "$xyz" .xyz)"
    if [[ "$name" == "$SKIP" ]]; then
        echo ">>> skipping $name (already handled)"
        continue
    fi

    echo ">>> [$(date '+%H:%M:%S')] starting $name"
    if conda run -n "$CONDA_ENV" python scripts/generate_dataset.py "$name"; then
        echo ">>> [$(date '+%H:%M:%S')] finished $name -> data/labels/$name.extxyz"
    else
        echo ">>> [$(date '+%H:%M:%S')] FAILED $name (see data/labels/${name}.progress.log)" >&2
    fi
done

echo ">>> all molecules done"
