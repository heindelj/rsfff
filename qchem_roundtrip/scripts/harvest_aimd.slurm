#!/bin/bash
#SBATCH -A m3196
#SBATCH -t 00:30:00
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH -e aimd_harvest.%j.error
#SBATCH -J aimd_harvest
#SBATCH -o aimd_harvest.%j.stdout

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

module load python

bash scripts/harvest_aimd.sh
