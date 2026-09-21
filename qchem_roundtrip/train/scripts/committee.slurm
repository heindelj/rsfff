#!/bin/bash
# One committee fit on one Perlmutter GPU node: 4 A100s, one member per GPU.
# Submit through scripts/submit.sh, which sets account/QOS/time/name and TRAIN_DIR;
# the arguments after the script are passed to train_committee.py.
#SBATCH --constraint gpu
#SBATCH --nodes 1
#SBATCH --gpus-per-node 4
#SBATCH --cpus-per-task 128
#SBATCH --output %x-%j.out

set -uo pipefail
TRAIN_DIR="${TRAIN_DIR:?submit with scripts/submit.sh}"
# shellcheck disable=SC1091
source "$TRAIN_DIR/scripts/env.sh"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# ~1 minute that keeps four GPUs from spending the wall clock on a broken environment
if [ "${GPU_CHECK:-1}" = "1" ]; then
    python3 "$TRAIN_DIR/scripts/gpu_check.py" || { echo "gpu_check failed; not training" >&2; exit 1; }
fi

python3 "$TRAIN_DIR/train_committee.py" "$@"
STATUS=$?
if [ $STATUS -eq 0 ]; then
    OUT=""
    prev=""
    for a in "$@"; do [ "$prev" = "--out" ] && OUT="$a"; prev="$a"; done
    [ -n "$OUT" ] && python3 "$TRAIN_DIR/scripts/check_committee.py" "$OUT"
fi
exit $STATUS
