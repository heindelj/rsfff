#!/bin/bash
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH --gpus-per-node 4
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 128
# A committee fit on one 4-GPU node, one member per GPU. Queued by the train stage
# (rsfff_al.train_stage.CommitteeTrain) with the train_committee.py arguments after the script,
# or by hand:
#   sbatch -A m3196 -q regular -t 08:00:00 train/scripts/committee.slurm \
#       --config <cfg.yaml> --out <committee dir> --members 4
# Needs RSFFF_AL in the environment (scripts/env_perlmutter.sh sets it; sbatch exports it).
set -uo pipefail
: "${RSFFF_AL:?source scripts/env_perlmutter.sh before submitting}"
source "$RSFFF_AL/scripts/env_perlmutter.sh"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

CONFIG=""; OUT=""; prev=""
for a in "$@"; do
    [ "$prev" = "--config" ] && CONFIG="$a"
    [ "$prev" = "--out" ] && OUT="$a"
    prev="$a"
done
if [ "${GPU_CHECK:-1}" = "1" ] && [ -n "$CONFIG" ]; then
    python3 "$RSFFF_AL/train/scripts/gpu_check.py" --config "$CONFIG" \
        || { echo "gpu_check failed; not training" >&2; exit 1; }
fi
python3 "$RSFFF_AL/train/train_committee.py" "$@"
STATUS=$?
if [ $STATUS -eq 0 ] && [ -n "$OUT" ]; then
    python3 "$RSFFF_AL/train/scripts/check_committee.py" "$OUT" --frames 8
fi
exit $STATUS
