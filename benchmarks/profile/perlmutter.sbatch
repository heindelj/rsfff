#!/bin/bash
#SBATCH -A m3196
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -c 32
#SBATCH -t 01:00:00
#SBATCH -J rsfff-profile
#SBATCH -o benchmarks/profile/results/slurm-%j.out

set -euo pipefail
module load conda 2>/dev/null || true
conda activate /global/cfs/cdirs/m3196/heindelj/rsfff
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p benchmarks/profile/results
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

STRUCTS="benchmarks/structures/w6_mp2_avtz.xyz benchmarks/structures/w12_mp2_avtz.xyz \
         benchmarks/structures/w21_mp2_avtz.xyz external/torchff-lib/examples/water_216.pdb"

python benchmarks/profile/profile_film.py --device cuda --repeats 5 --trace --structures $STRUCTS
python benchmarks/profile/profile_film.py --device cuda --repeats 5 --no-induction --structures $STRUCTS
# The big one: forward and forward+forces only; the training step is attempted but may OOM.
python benchmarks/profile/profile_film.py --device cuda --repeats 3 --max-neighbors 1024 \
    --structures external/torchff-lib/examples/water_1000.pdb || echo "1000-water run failed (see above)"
