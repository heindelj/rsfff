# Source this on either side of the round trip. It lives inside the job-pool bundle, which is
# what gets rsynced to the cluster, so the same file is there and here:
#
#     source qchem_roundtrip/active_learning/scripts/nersc_env.sh     # from a checkout
#     source $RSFFF_QCHEM_ROOT/active_learning/scripts/nersc_env.sh   # on Perlmutter
#
# On Perlmutter it points the stages at the job pool on CFS and activates the conda prefix.
# On a laptop it sets REMOTE/REMOTE_DIR, which is all the sync scripts under
# qchem_roundtrip/scripts/ read, so the label stage's submit/sync hooks inherit them.
#
# Override any of these by exporting it before sourcing.

RSFFF_NERSC_ACCOUNT="${RSFFF_NERSC_ACCOUNT:-m3196}"

# The qchem_roundtrip bundle as Perlmutter sees it: config.json, templates/, scripts/ and the
# eda/ and force/ trees live directly under this path.
RSFFF_POOL_REMOTE="${RSFFF_POOL_REMOTE:-/global/cfs/cdirs/m3196/heindelj/rsfff_data}"

# A conda *prefix*, not a name, so it is activated by path.
RSFFF_CONDA_PREFIX="${RSFFF_CONDA_PREFIX:-/global/cfs/cdirs/m3196/heindelj/rsfff}"

# ssh alias for Perlmutter, used only from a laptop.
REMOTE="${REMOTE:-perlmutter}"

_nersc_env_script="${BASH_SOURCE[0]:-$0}"
# <bundle>/active_learning/scripts/nersc_env.sh -> the checkout is three levels up when the
# bundle sits inside one, and irrelevant when it does not (RSFFF_REPO wins, and the stages fall
# back to the installed rsfff).
RSFFF_REPO="${RSFFF_REPO:-$(cd "$(dirname "$_nersc_env_script")/../../.." && pwd)}"
export RSFFF_REPO RSFFF_NERSC_ACCOUNT

if [ -n "${NERSC_HOST:-}" ]; then
    # --- on Perlmutter: the pool is a local path, and there is nothing to rsync ------------
    export RSFFF_QCHEM_ROOT="$RSFFF_POOL_REMOTE"
    if [ -d "$RSFFF_CONDA_PREFIX" ]; then
        if command -v conda >/dev/null 2>&1; then
            eval "$(conda shell.bash hook)" 2>/dev/null || true
            conda activate "$RSFFF_CONDA_PREFIX" \
                && echo "[nersc-env] conda: $RSFFF_CONDA_PREFIX" \
                || echo "[nersc-env] could not activate $RSFFF_CONDA_PREFIX" >&2
        else
            echo "[nersc-env] conda not on PATH; 'module load conda' (or python) first" >&2
        fi
    else
        echo "[nersc-env] no env at $RSFFF_CONDA_PREFIX" >&2
    fi
    echo "[nersc-env] pool:  $RSFFF_QCHEM_ROOT"
    echo "[nersc-env] repo:  $RSFFF_REPO"
    echo "[nersc-env] account $RSFFF_NERSC_ACCOUNT  (salloc -A $RSFFF_NERSC_ACCOUNT -N 1 -C cpu -q interactive -t 02:00:00)"
else
    # --- on a laptop: the pool is remote, reached by the sync scripts ----------------------
    export REMOTE
    export REMOTE_DIR="$RSFFF_POOL_REMOTE"
    echo "[nersc-env] REMOTE=$REMOTE  REMOTE_DIR=$REMOTE_DIR"
    echo "[nersc-env] repo:  $RSFFF_REPO"
    echo "[nersc-env] hooks: submit=['bash scripts/sync_inputs_up.sh',"
    echo "[nersc-env]                \"ssh \$REMOTE 'cd \$REMOTE_DIR && bash scripts/submit_workers.sh --target 16'\"]"
    echo "[nersc-env]        sync=['bash scripts/sync_outputs_down.sh']"
fi
unset _nersc_env_script
