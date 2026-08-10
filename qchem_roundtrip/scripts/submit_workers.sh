#!/bin/bash
# Keep a fixed number of Q-Chem worker Slurm jobs active or pending.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROUNDTRIP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

TARGET="${QCHEM_WORKER_TARGET:-1}"
INTERVAL_SECONDS="${QCHEM_SUBMIT_INTERVAL_SECONDS:-300}"
WATCH=0
DRY_RUN=0
JOB_NAME="${QCHEM_WORKER_JOB_NAME:-qchem_worker}"
SBATCH_SCRIPT="${QCHEM_WORKER_SLURM_SCRIPT:-$ROUNDTRIP_ROOT/scripts/worker.slurm}"
STATES="${QCHEM_WORKER_STATES:-PD,R,CF}"

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --target N          Keep N worker jobs active or pending (default: $TARGET).
  --watch            Keep checking and topping up forever.
  --interval SEC     Sleep SEC between checks in --watch mode (default: $INTERVAL_SECONDS).
  --job-name NAME    Slurm job name to count and submit (default: $JOB_NAME).
  --dry-run          Print sbatch commands without submitting.
  -h, --help         Show this help.

Environment:
  QCHEM_WORKER_TARGET
  QCHEM_SUBMIT_INTERVAL_SECONDS
  QCHEM_WORKER_JOB_NAME
  QCHEM_WORKER_SLURM_SCRIPT
  QCHEM_WORKER_STATES
  QCHEM_THREADS
  QCHEM_IDLE_TIMEOUT_SECONDS
  QCHEM_POLL_SECONDS
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --target)
            TARGET="${2:?--target requires a value}"
            shift 2
            ;;
        --interval)
            INTERVAL_SECONDS="${2:?--interval requires a value}"
            shift 2
            ;;
        --job-name)
            JOB_NAME="${2:?--job-name requires a value}"
            shift 2
            ;;
        --watch)
            WATCH=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[submit-workers] unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$TARGET" in
    ''|*[!0-9]*)
        echo "[submit-workers] --target must be a non-negative integer" >&2
        exit 2
        ;;
esac

case "$INTERVAL_SECONDS" in
    ''|*[!0-9]*)
        echo "[submit-workers] --interval must be a non-negative integer" >&2
        exit 2
        ;;
esac

if [ ! -f "$SBATCH_SCRIPT" ]; then
    echo "[submit-workers] missing Slurm script: $SBATCH_SCRIPT" >&2
    exit 2
fi

if ! command -v squeue >/dev/null 2>&1; then
    echo "[submit-workers] squeue not found; run this on Perlmutter." >&2
    exit 2
fi

if ! command -v sbatch >/dev/null 2>&1; then
    echo "[submit-workers] sbatch not found; run this on Perlmutter." >&2
    exit 2
fi

list_workers() {
    squeue -h -u "$USER" -n "$JOB_NAME" -t "$STATES" -o "%i|%j|%T|%Z" \
        | awk -F'|' -v root="$ROUNDTRIP_ROOT" '$4 == root { print }'
}

count_workers() {
    list_workers | awk 'END { print NR + 0 }'
}

submit_one() {
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[submit-workers] dry-run: cd '$ROUNDTRIP_ROOT' && sbatch --job-name '$JOB_NAME' '$SBATCH_SCRIPT'"
        return 0
    fi
    (
        cd "$ROUNDTRIP_ROOT"
        sbatch --job-name "$JOB_NAME" "$SBATCH_SCRIPT"
    )
}

top_up_once() {
    local current missing idx
    current="$(count_workers)"
    missing=$((TARGET - current))
    echo "[submit-workers] root=$ROUNDTRIP_ROOT job_name=$JOB_NAME target=$TARGET current=$current states=$STATES"
    if [ "$missing" -le 0 ]; then
        echo "[submit-workers] pool is already full; submitting 0 job(s)."
        return 0
    fi
    for idx in $(seq 1 "$missing"); do
        echo "[submit-workers] submitting worker $idx/$missing"
        submit_one
    done
}

while true; do
    top_up_once
    if [ "$WATCH" -ne 1 ]; then
        break
    fi
    echo "[submit-workers] sleeping ${INTERVAL_SECONDS}s"
    sleep "$INTERVAL_SECONDS"
done
