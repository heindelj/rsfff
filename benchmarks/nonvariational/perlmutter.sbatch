#!/bin/bash
#SBATCH -A m3196
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -c 32
#SBATCH -t 01:30:00
#SBATCH -J rsfff-nonvariational
#SBATCH -o benchmarks/nonvariational/results/slurm-%j.out

# Non-variational (unrolled, K iterations) vs variational (PCG) induction, by cluster size,
# on the kernels. Needs torchff-lib built WITH slater_elec_field_hvp (external/torchff-lib at
# or after the commit that adds it): without it the training step of the nonvariational model
# falls back to the pure-torch field inside the loop and is not the number we want.
set -euo pipefail
module load conda 2>/dev/null || true
conda activate /global/cfs/cdirs/m3196/heindelj/rsfff
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p benchmarks/nonvariational/results
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python -c "from torchff import slaterelec as se; print('torchff kernels', se.HAVE_KERNELS, 'field double backward', se.FIELD_DOUBLE_BACKWARD)"

# the main table: kernels, both models, K scan, accuracy, 4 frames per training step
python benchmarks/nonvariational/bench_solve.py --device cuda --backends torchff \
    --waters 8 16 32 64 128 216 --n-iter 2 3 4 --frames 4 --repeats 5 --accuracy --tag pm_torchff

# torch reference backend on the same sizes up to 64 waters (the (P,K,K) tensors do not fit beyond)
python benchmarks/nonvariational/bench_solve.py --device cuda --backends torch \
    --waters 8 16 32 64 --n-iter 3 --frames 4 --repeats 3 --tag pm_torch || echo "torch-backend run failed"

# large clusters, energy and forces only (the training step at these sizes is attempted but may OOM)
python benchmarks/nonvariational/bench_solve.py --device cuda --backends torchff \
    --waters 512 1000 --n-iter 3 --frames 1 --repeats 3 --tag pm_large || echo "large run failed (see above)"
