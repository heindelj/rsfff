# source scripts/env_perlmutter.sh   -- on a Perlmutter login or compute node
#
# The rsfff conda prefix, plus easyAL and cc_workers from their checkouts (both are small and
# dependency-free, so PYTHONPATH is enough; `pip install -e` into the prefix works too).
module load python 2>/dev/null || true
conda activate /global/cfs/cdirs/m3196/heindelj/rsfff
SOFTWARE=/global/cfs/cdirs/m3196/heindelj/software
export RSFFF_AL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# _deps/ is what scripts/sync.sh up copies from the laptop; the checkouts are the fallback
export PYTHONPATH="$RSFFF_AL:$RSFFF_AL/_deps/easyAL/src:$RSFFF_AL/_deps/cc_workers/src:$SOFTWARE/easyAL/src:$SOFTWARE/cc_workers/src${PYTHONPATH:+:$PYTHONPATH}"
export RSFFF_STORE="${RSFFF_STORE:-/global/cfs/cdirs/m3196/heindelj/rsfff_store}"
export KMP_DUPLICATE_LIB_OK=TRUE
command -v packmol >/dev/null || echo "WARNING: packmol not on PATH (conda install -c conda-forge packmol, or pip install packmol)"
python - <<'EOF'
import importlib
for m in ("torch", "rsfff.md.film_driver", "easyal", "cc_workers"):
    try:
        importlib.import_module(m)
    except Exception as e:
        print(f"WARNING: cannot import {m}: {e}")
import os, torch
print(f"rsfff_active_learning at {os.environ['RSFFF_AL']}; torch {torch.__version__}, cuda={torch.cuda.is_available()}")
EOF
