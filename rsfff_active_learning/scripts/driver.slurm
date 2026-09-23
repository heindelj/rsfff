#!/bin/bash
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task 32
#SBATCH -t 06:00:00
#SBATCH -J rsfff_al_driver
# The loop driver: runs scripts/run_loop.py until it reports pending (Q-Chem or training jobs
# outstanding, or the dynamics time budget spent), then requeues itself to look again later.
#
#   sbatch -A m3196 --export=ALL,AL_CONFIG=$RSFFF_AL/configs/water_udd.yaml \
#       $RSFFF_AL/scripts/driver.slurm
#
# AL_RECHECK (default now+60minutes) is when the next driver starts after a "pending".
# A file named STOP in the loop root stops the chain. One GPU: the dynamics and assess stages
# run here; committee training is its own 4-GPU job, Q-Chem its own CPU jobs.
set -uo pipefail
: "${RSFFF_AL:?source scripts/env_perlmutter.sh before submitting}"
: "${AL_CONFIG:?AL_CONFIG=<loop yaml>}"
source "$RSFFF_AL/scripts/env_perlmutter.sh"
cd "$RSFFF_AL"
OUT=$(python3 scripts/run_loop.py --config "$AL_CONFIG" 2>&1 | tee /dev/stderr | tail -1)
ROOT=$(python3 -c "import os,yaml;print(os.path.expandvars(yaml.safe_load(open('$AL_CONFIG'))['root']))")
echo "$(date -Is) job ${SLURM_JOB_ID:-local} -> $OUT" >> "$ROOT/driver_history"
if [ "$OUT" = "pending" ] && [ ! -e "$ROOT/STOP" ]; then
    sbatch --begin="${AL_RECHECK:-now+60minutes}" -A "${SLURM_JOB_ACCOUNT:-m3196}" \
        --export=ALL,AL_CONFIG="$AL_CONFIG" "$RSFFF_AL/scripts/driver.slurm"
fi
