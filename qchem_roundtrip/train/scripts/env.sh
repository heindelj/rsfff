# Source this before training: `source <bundle>/train/scripts/env.sh`
#
# It sources the active-learning env (conda prefix, pool path, RSFFF_REPO) and exports
# TRAIN_DIR. If that conda env has a CPU-only torch -- scripts/gpu_check.py says so in its
# first line -- point RSFFF_TRAIN_SETUP at a file of shell commands that activates a CUDA one
# instead (see README, "Environment"); it is sourced last.
TRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
export TRAIN_DIR
# shellcheck disable=SC1091
source "$TRAIN_DIR/../active_learning/scripts/nersc_env.sh"
if [ -n "${RSFFF_TRAIN_SETUP:-}" ]; then
    # shellcheck disable=SC1090
    source "$RSFFF_TRAIN_SETUP" && echo "[train-env] sourced $RSFFF_TRAIN_SETUP"
fi
echo "[train-env] train: $TRAIN_DIR"
python3 -c "import torch; print('[train-env] torch', torch.__version__, 'cuda build', torch.version.cuda, 'available', torch.cuda.is_available())" 2>/dev/null \
    || echo "[train-env] torch does not import in this environment" >&2
