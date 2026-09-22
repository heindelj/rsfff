#!/bin/bash
# Post-M5: where does the *training step* go? Training-shaped batches (128 frames) of w6 and
# w21, four variants, each writing its own results/*.md with the per-phase / per-region
# training-step tables under the usual forward table:
#   1. torchff, E+F loss           -- the real thing, backward attributed to regions
#   2. torchff, energy-only loss   -- ablation: no create_graph, no double backward
#   3. torchff, no induction       -- ablation: bounds the coupled solve's share in all phases
#   4. torch,   E+F loss           -- reference for the same split on the old path
#SBATCH -A m3196
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -c 32
#SBATCH -t 00:45:00
#SBATCH -J rsfff-train-split
#SBATCH -o benchmarks/profile/results/slurm-%j.out

set -euo pipefail
module load conda 2>/dev/null || true
conda activate /global/cfs/cdirs/m3196/heindelj/rsfff
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
bash benchmarks/profile/run_train_split.sh "$@"
