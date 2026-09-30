#!/bin/bash
# Create the rsfff conda environment (Python 3.12) from environment.yml, on the Mac or on
# Perlmutter, then do the steps environment.yml can't express:
#   1. rsfff itself (editable)
#   2. the cuequivariance CUDA ops matching torch's CUDA major version (Linux + CUDA torch)
#   3. torchff-lib: CUDA kernels on Perlmutter (via build_torchff_perlmutter.sh), the
#      pure-torch reference paths everywhere else
#
#   bash scripts/make_env.sh                   # Mac: named env `rsfff`
#   bash scripts/make_env.sh -n rsfff312       # a side-by-side env, to test before swapping
#   bash scripts/make_env.sh                   # Perlmutter: prefix /global/cfs/cdirs/m3196/heindelj/rsfff
#
# Options:
#   -n NAME        named env        (default off Perlmutter: rsfff)
#   -p PREFIX      env at a path    (default on Perlmutter: /global/cfs/cdirs/m3196/heindelj/rsfff)
#   --force        remove an existing env at the target first (otherwise the script refuses)
#   --no-cue       skip the cuequivariance CUDA ops
#   --no-torchff   skip torchff-lib (e.g. build it later with build_torchff_perlmutter.sh)
#
# Environment overrides:
#   CUDA_MODULE=cudatoolkit/X.Y   Perlmutter: toolkit for the torchff build (default: torch's CUDA)
#   TORCHFF_CUDA=1                off Perlmutter: compile torchff's CUDA kernels anyway
#
# On Perlmutter run this on a login node (compute nodes have no network). The torchff build
# compiles every extension and takes a while; --no-torchff and running the build later is fine.
set -eo pipefail   # no -u: conda's activate scripts reference unset variables

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PERLMUTTER_PREFIX=/global/cfs/cdirs/m3196/heindelj/rsfff

FLAG='' TARGET='' FORCE=0 DO_CUE=1 DO_TORCHFF=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|-p)        FLAG="$1"; TARGET="$2"; shift 2 ;;
        --force)      FORCE=1; shift ;;
        --no-cue)     DO_CUE=0; shift ;;
        --no-torchff) DO_TORCHFF=0; shift ;;
        -h|--help)    sed -n '2,25p' "$0"; exit 0 ;;
        *)            echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
done

if [[ "$(uname -s)" == Darwin ]]; then PLATFORM=mac
elif [[ "${NERSC_HOST:-}" == perlmutter ]]; then PLATFORM=perlmutter
else PLATFORM=linux; fi

if [[ -z "$FLAG" ]]; then
    if [[ $PLATFORM == perlmutter ]]; then FLAG=-p TARGET=$PERLMUTTER_PREFIX
    else FLAG=-n TARGET=rsfff; fi
fi
[[ $FLAG == -p ]] && TARGET="$(cd "$(dirname "$TARGET")" && pwd)/$(basename "$TARGET")"
echo ">> platform $PLATFORM, env $FLAG $TARGET"

# --- conda ---------------------------------------------------------------------------------
if [[ $PLATFORM == perlmutter ]]; then
    module load conda 2>/dev/null || module load python 2>/dev/null || true
fi
command -v conda >/dev/null || { echo "conda not found on PATH" >&2; exit 1; }
eval "$(conda shell.bash hook)"

env_exists() {
    if [[ $FLAG == -p ]]; then [[ -d "$TARGET/conda-meta" ]]
    else conda env list | awk '{print $1}' | grep -qx "$TARGET"; fi
}
if env_exists; then
    if [[ $FORCE == 1 ]]; then
        echo ">> removing existing env $TARGET"
        conda remove -y $FLAG "$TARGET" --all
    else
        echo "env $TARGET already exists. Either build a side-by-side one (-n rsfff312 / -p <new path>)," >&2
        echo "move it aside (prefix envs: mv $TARGET ${TARGET}-old), or pass --force to delete it." >&2
        exit 1
    fi
fi

echo ">> conda env create (python 3.12, torch, torch-sim, cuequivariance, ...)"
conda env create $FLAG "$TARGET" -f "$REPO/environment.yml"
conda activate "$TARGET"
cd "$REPO"

# --- rsfff ---------------------------------------------------------------------------------
echo ">> pip install -e . (rsfff)"
python -m pip install -e .

TORCH_CUDA="$(python -c 'import torch; print(torch.version.cuda or "")')"
echo ">> torch $(python -c 'import torch; print(torch.__version__)'), CUDA ${TORCH_CUDA:-none}"

# --- cuequivariance CUDA ops ---------------------------------------------------------------
if [[ $DO_CUE == 1 ]]; then
    if [[ "$(uname -s)" == Linux && -n "$TORCH_CUDA" ]]; then
        CU_MAJOR="${TORCH_CUDA%%.*}"
        if [[ $CU_MAJOR == 12 || $CU_MAJOR == 13 ]]; then
            echo ">> cuequivariance-ops-torch-cu$CU_MAJOR"
            python -m pip install "cuequivariance-ops-torch-cu$CU_MAJOR"
        else
            echo ">> no cuequivariance-ops build for CUDA $TORCH_CUDA; cue backend will use the unfused path" >&2
        fi
    else
        echo ">> skipping cuequivariance CUDA ops (no CUDA torch here); cue backend runs on CPU"
    fi
fi

# --- torchff-lib ---------------------------------------------------------------------------
if [[ $DO_TORCHFF == 1 ]]; then
    [[ -f external/torchff-lib/setup.py ]] || git submodule update --init external/torchff-lib
    if [[ $PLATFORM == perlmutter ]]; then
        [[ -n "$TORCH_CUDA" ]] || { echo "torch in this env has no CUDA; cannot build torchff kernels" >&2; exit 1; }
        CUDA_MODULE="${CUDA_MODULE:-cudatoolkit/$TORCH_CUDA}"
        if ! module is-avail "$CUDA_MODULE" 2>/dev/null; then
            echo "module $CUDA_MODULE (to match torch's CUDA $TORCH_CUDA) is not available. Have:" >&2
            module -t avail cudatoolkit 2>&1 | grep cudatoolkit >&2 || true
            echo "rerun the build with CUDA_MODULE=cudatoolkit/<same major> bash scripts/build_torchff_perlmutter.sh" >&2
            echo "(RSFFF_ENV=$TARGET), or reinstall torch for an available toolkit." >&2
            exit 1
        fi
        echo ">> torchff-lib CUDA build against $CUDA_MODULE"
        RSFFF_ENV="$TARGET" CUDA_MODULE="$CUDA_MODULE" bash scripts/build_torchff_perlmutter.sh
    elif [[ "${TORCHFF_CUDA:-0}" == 1 ]]; then
        echo ">> torchff-lib with CUDA kernels"
        python -m pip install --no-build-isolation -e external/torchff-lib
    else
        echo ">> torchff-lib, pure-torch reference paths (no CUDA kernels)"
        TORCHFF_NO_CUDA=1 python -m pip install --no-build-isolation -e external/torchff-lib
    fi
fi

# --- smoke test ----------------------------------------------------------------------------
echo ">> import check"
python - <<'PY'
import importlib, sys
import torch
print(f"   python {sys.version.split()[0]}, torch {torch.__version__}, "
      f"cuda {torch.version.cuda}, gpu visible {torch.cuda.is_available()}")
# e3nn before cuequivariance_torch: some OpenMP builds misbehave the other way round
for m in ("rsfff", "e3nn", "ase", "torch_sim", "cuequivariance_torch",
          "cuequivariance_ops_torch", "torchff"):
    try:
        mod = importlib.import_module(m)
        print(f"   ok  {m} {getattr(mod, '__version__', '')}")
    except Exception as e:
        print(f"   --  {m}: {type(e).__name__}: {e}")
PY
echo ">> done: conda activate $TARGET"
[[ $PLATFORM == perlmutter ]] && echo "   (active learning: source rsfff_active_learning/scripts/env_perlmutter.sh)"
exit 0
