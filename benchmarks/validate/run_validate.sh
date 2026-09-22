#!/bin/bash
# M6: torch-vs-torchff parity on w4-w23 and an NVE run on (H2O)216. Run from the repo root
# on a GPU node (interactive or batch); ~15 min at the defaults (1000 steps of 0.5 fs).
#
#   salloc -A m3196 -C gpu -q interactive -N 1 --gpus-per-node=1 -c 32 -t 01:00:00
#   conda activate /global/cfs/cdirs/m3196/heindelj/rsfff
#   bash benchmarks/validate/run_validate.sh
#
# STEPS / DT / TEMP override the NVE settings; extra arguments go to the script.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p benchmarks/validate/results
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name --format=csv,noheader
python benchmarks/validate/validate_backends.py --device cuda \
    --steps "${STEPS:-1000}" --dt "${DT:-0.5}" --temperature "${TEMP:-300}" "$@"
