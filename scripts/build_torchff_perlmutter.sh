#!/bin/bash
# Build the torchff-lib submodule (CUDA kernels) on Perlmutter, into the rsfff conda env.
#
#   cd /global/cfs/cdirs/m3196/heindelj/software/rsfff
#   git submodule update --init
#   bash scripts/build_torchff_perlmutter.sh            # login node (needs network for pip)
#
# The CUDA toolkit module must match the CUDA that the env's torch was built against
# (`python -c "import torch; print(torch.version.cuda)"`), or the extension links but fails
# to load. Adjust CUDA_MODULE below if that prints something other than 12.x.
set -euo pipefail

RSFFF_ENV="${RSFFF_ENV:-/global/cfs/cdirs/m3196/heindelj/rsfff}"
CUDA_MODULE="${CUDA_MODULE:-cudatoolkit/12.4}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"   # A100
export MAX_JOBS="${MAX_JOBS:-8}"

module load conda 2>/dev/null || true
conda activate "$RSFFF_ENV"
module load "$CUDA_MODULE"
module load gcc-native 2>/dev/null || module load gcc 2>/dev/null || true

cd "$(dirname "$0")/../external/torchff-lib"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
pip install --no-build-isolation -e .
python - <<'PY'
import torchff_ffterms, torch
from torchff import ffterms
print("torchff_ffterms loaded; kernels available:", ffterms.HAVE_KERNELS)
PY
echo "now run the kernel tests on a GPU node:"
echo "  salloc -A m3196 -C gpu -q interactive -t 0:30:00 --gpus 1"
echo "  pytest external/torchff-lib/tests/test_ffterms.py tests/backend -q"
