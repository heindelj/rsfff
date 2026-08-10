#!/bin/bash
#SBATCH -A m3196
#SBATCH -t 48:00:00
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH -e qchem_worker.%j.error
#SBATCH -J qchem_worker
#SBATCH -o qchem_worker.%j.stdout

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

module load python
module load qchem

export QCHEM_THREADS="${QCHEM_THREADS:-128}"
export QCHEM_IDLE_TIMEOUT_SECONDS="${QCHEM_IDLE_TIMEOUT_SECONDS:-1800}"
export QCHEM_POLL_SECONDS="${QCHEM_POLL_SECONDS:-30}"

bash scripts/worker.sh
