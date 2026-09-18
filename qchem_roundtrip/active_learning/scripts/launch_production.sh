#!/bin/bash
# Launch (or resume) a production active-learning run on Perlmutter.
#
#   module load python
#   source /global/cfs/cdirs/m3196/heindelj/rsfff_data/active_learning/scripts/nersc_env.sh
#   bash $RSFFF_QCHEM_ROOT/active_learning/scripts/launch_production.sh \
#       --root $SCRATCH/water_al \
#       --checkpoint /global/cfs/cdirs/m3196/heindelj/software/rsfff/checkpoints/water_film_full/best.pt
#
# Two kinds of job come out of this:
#
#   driver   one CPU node (driver.slurm) that runs the loop -- build, optimize, dynamics,
#            select, label, train, assess -- until the label stage is waiting on Q-Chem. It
#            then exits and requeues itself --resubmit-delay minutes later, so no node sits
#            idle through the labeling. It stops requeueing when the loop finishes, when a
#            stage fails, or after --max-resubmits. --dependency=singleton on a per-run job
#            name means two drivers never touch the same loop at once.
#   workers  qchem_al_worker jobs (al_workers.sh -> scripts/worker.slurm), submitted by the
#            label stage's hooks and topped up every time the driver checks on them.
#
# Every setting is written to <root>/driver.env on the first launch and read back from there on
# every requeue, so each driver runs the loop with exactly the same parameters -- which is what
# easyAL's recorded stage parameters expect. Edit driver.env to change worker counts or wall
# clocks between requeues; changing the loop's own flags (AL_EXTRA_ARGS) mid-run is not safe.
#
# First launch
#   --root DIR            the loop directory (on $SCRATCH)                          [required]
#   --checkpoint PATH     the model iteration 0 samples with                        [required]
#   --name NAME           driver job name suffix (default: basename of --root)
#   --time HH:MM:SS       driver wall clock (default 12:00:00)
#   --qos NAME            driver QOS (default regular)
#   --workers N           most Q-Chem workers at once (default 16)
#   --worker-time T       worker wall clock (default 24:00:00)
#   --worker-qos NAME     worker QOS (default premium)
#   --wait SECONDS        how long a driver polls for Q-Chem before requeueing (default 900)
#   --resubmit-delay MIN  minutes between a pending driver and the next (default 30)
#   --max-resubmits N     give up after this many drivers (default 300)
#   --train-parallel N    committee members fitted at once (default 4)
#   --train-threads N     CPU threads per member (default 32; 4 x 32 = a Perlmutter CPU node)
#   --skip-preflight      don't run preflight.py before submitting
#   --dry-run             write driver.env and print the sbatch command, submit nothing
#   -- ARGS...            passed through to workflows.py (e.g. --train-config, --steps,
#                         --initial-data, --members)
#
# Resume (what driver.slurm calls; also how to restart a run that stopped)
#   --resume DIR          submit a driver for the run at DIR from its driver.env
#   --begin TIME          sbatch --begin, e.g. now+30minutes (default: now)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POOL="$(cd "$SCRIPT_DIR/../.." && pwd)"

say() { echo "[launch] $*"; }
die() { echo "[launch] $*" >&2; exit 2; }

ROOT=""; CHECKPOINT=""; NAME=""; RESUME=""; BEGIN=""
DRIVER_TIME="12:00:00"; DRIVER_QOS="regular"
AL_WORKER_TARGET=16; AL_WORKER_TIME="24:00:00"; AL_WORKER_QOS="premium"
WAIT_SECONDS=900; RESUBMIT_DELAY=30; MAX_RESUBMITS=300
TRAIN_PARALLEL=4; TRAIN_THREADS=32
PREFLIGHT=1; DRY_RUN=0
AL_EXTRA_ARGS=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --root)            ROOT="${2:?}"; shift 2 ;;
        --checkpoint)      CHECKPOINT="${2:?}"; shift 2 ;;
        --name)            NAME="${2:?}"; shift 2 ;;
        --time)            DRIVER_TIME="${2:?}"; shift 2 ;;
        --qos)             DRIVER_QOS="${2:?}"; shift 2 ;;
        --workers)         AL_WORKER_TARGET="${2:?}"; shift 2 ;;
        --worker-time)     AL_WORKER_TIME="${2:?}"; shift 2 ;;
        --worker-qos)      AL_WORKER_QOS="${2:?}"; shift 2 ;;
        --wait)            WAIT_SECONDS="${2:?}"; shift 2 ;;
        --resubmit-delay)  RESUBMIT_DELAY="${2:?}"; shift 2 ;;
        --max-resubmits)   MAX_RESUBMITS="${2:?}"; shift 2 ;;
        --train-parallel)  TRAIN_PARALLEL="${2:?}"; shift 2 ;;
        --train-threads)   TRAIN_THREADS="${2:?}"; shift 2 ;;
        --skip-preflight)  PREFLIGHT=0; shift ;;
        --dry-run)         DRY_RUN=1; shift ;;
        --resume)          RESUME="${2:?}"; shift 2 ;;
        --begin)           BEGIN="${2:?}"; shift 2 ;;
        --)                shift; AL_EXTRA_ARGS=("$@"); break ;;
        -h|--help)         sed -n '2,50p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)                 die "unknown option: $1 (loop flags go after --)" ;;
    esac
done

if [ -n "$RESUME" ]; then
    ROOT="$(cd "$RESUME" && pwd)"
    [ -f "$ROOT/driver.env" ] || die "no driver.env under $ROOT; launch it with --root first"
    # shellcheck disable=SC1091
    source "$ROOT/driver.env"
    rm -f "$ROOT/driver_state/FAILED"     # a resume is the fix having been made
else
    [ -n "$ROOT" ] || die "--root is required"
    [ -n "$CHECKPOINT" ] || die "--checkpoint is required"
    mkdir -p "$ROOT"
    ROOT="$(cd "$ROOT" && pwd)"
    [ -f "$CHECKPOINT" ] || die "no checkpoint at $CHECKPOINT"
    CHECKPOINT="$(cd "$(dirname "$CHECKPOINT")" && pwd)/$(basename "$CHECKPOINT")"
    if [ -f "$ROOT/driver.env" ]; then
        die "$ROOT/driver.env exists -- this run was launched before. Use --resume $ROOT"
    fi
    NAME="${NAME:-$(basename "$ROOT")}"
    ACCOUNT="${RSFFF_NERSC_ACCOUNT:-m3196}"
    {
        echo "# written by launch_production.sh on $(date -Is); read back by every driver"
        for var in ACCOUNT NAME CHECKPOINT DRIVER_TIME DRIVER_QOS AL_WORKER_TARGET \
                   AL_WORKER_TIME AL_WORKER_QOS WAIT_SECONDS RESUBMIT_DELAY MAX_RESUBMITS \
                   TRAIN_PARALLEL TRAIN_THREADS; do
            printf '%s=%q\n' "$var" "${!var}"
        done
        printf 'AL_EXTRA_ARGS=('
        for arg in ${AL_EXTRA_ARGS[@]+"${AL_EXTRA_ARGS[@]}"}; do printf '%q ' "$arg"; done
        printf ')\n'
        printf 'POOL=%q\n' "$POOL"
        printf 'RSFFF_REPO=%q\n' "${RSFFF_REPO:-}"
    } > "$ROOT/driver.env"
    say "settings: $ROOT/driver.env"

    if [ "$PREFLIGHT" -eq 1 ]; then
        say "preflight"
        python3 "$SCRIPT_DIR/preflight.py" --checkpoint "$CHECKPOINT" --root "$ROOT" \
            --roundtrip-root "$POOL" || {
            rm -f "$ROOT/driver.env"
            die "preflight failed; fix the above (or --skip-preflight) and launch again"; }
    fi
fi

mkdir -p "$ROOT/logs" "$ROOT/driver_state"
if [ -f "$ROOT/driver_state/DONE" ]; then
    say "$ROOT is finished ($(cat "$ROOT/driver_state/DONE")); not submitting"
    exit 0
fi

JOB_NAME="rsfff_al_${NAME}"
SBATCH=(sbatch
    --account "$ACCOUNT" --qos "$DRIVER_QOS" --constraint cpu --nodes 1
    --time "$DRIVER_TIME" --job-name "$JOB_NAME" --dependency singleton
    --signal "B:USR1@600"
    --chdir "$ROOT" --output "$ROOT/logs/driver.%j.out"
    --export "ALL,AL_RUN_ROOT=$ROOT")
[ -z "$BEGIN" ] || SBATCH+=(--begin "$BEGIN")
SBATCH+=("$SCRIPT_DIR/driver.slurm")

# sbatch from inside a driver would otherwise inherit that job's settings
for var in $(env | awk -F= '/^(SLURM|SBATCH|SALLOC)_/ {print $1}'); do unset "$var"; done

if [ "$DRY_RUN" -eq 1 ]; then
    say "dry run: ${SBATCH[*]}"
    exit 0
fi
"${SBATCH[@]}"
say "driver $JOB_NAME queued for $ROOT${BEGIN:+ (begin $BEGIN)}"
say "watch:  squeue -u \$USER -n $JOB_NAME,qchem_al_worker"
say "status: python3 $POOL/active_learning/workflows.py water --root $ROOT --status"
