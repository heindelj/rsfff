#!/bin/bash
# Top up the active-learning Q-Chem workers -- the label stage's submit *and* sync hook.
#
#   label=dict(submit=["bash active_learning/scripts/al_workers.sh --target 16"],
#              sync=["bash active_learning/scripts/al_workers.sh --target 16"])
#
# The workers are the pool's own (scripts/worker.slurm via scripts/submit_workers.sh), with
# three differences that keep them to the loop's work:
#
#   * their config is config.al.json, written here from config.json with only the loop's
#     calculations (eda, force) enabled, so they never wander into other pending opt/aimd/...
#     inputs in the pool;
#   * they run under their own job name (qchem_al_worker), so they are counted -- and can be
#     scancelled -- separately from any other workers on the pool;
#   * the target is capped at the number of outstanding eda/force jobs, and nothing is
#     submitted when nothing is left unclaimed. That is what makes it safe as a sync hook:
#     it replaces workers that hit their wall clock mid-iteration, and it does not queue a
#     fresh batch of nodes to idle out once the last job has been claimed.
#
# Note the workers claim *any* unfinished eda/force input in the pool, not only the loop's.
#
#   --target N     most workers to keep active or pending (default $AL_WORKER_TARGET or 16)
#   --time HH:MM:SS  worker wall clock (default $AL_WORKER_TIME, else worker.slurm's own)
#   --qos NAME     worker QOS (default $AL_WORKER_QOS, else premium)
#   --threads N    Q-Chem threads per worker (default $QCHEM_THREADS or 128)
#   --dry-run      say what would be submitted

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POOL="$(cd "$SCRIPT_DIR/../.." && pwd)"          # <pool>/active_learning/scripts -> <pool>

TARGET="${AL_WORKER_TARGET:-16}"
TIME_LIMIT="${AL_WORKER_TIME:-}"
QOS="${AL_WORKER_QOS:-premium}"
THREADS="${QCHEM_THREADS:-128}"
JOB_NAME="${AL_WORKER_JOB_NAME:-qchem_al_worker}"
CALCULATIONS="${AL_CALCULATIONS:-eda force}"
DRY_RUN=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --target)  TARGET="${2:?}"; shift 2 ;;
        --time)    TIME_LIMIT="${2:?}"; shift 2 ;;
        --qos)     QOS="${2:?}"; shift 2 ;;
        --threads) THREADS="${2:?}"; shift 2 ;;
        --dry-run) DRY_RUN=(--dry-run); shift ;;
        -h|--help) sed -n '2,33p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "[al-workers] unknown option: $1" >&2; exit 2 ;;
    esac
done

say() { echo "[al-workers] $*"; }

AL_CONFIG="$POOL/config.al.json"

# 1. the workers' config, and 2. what is left for them, in one pass over the pool
COUNTS="$(python3 - "$POOL" "$AL_CONFIG" $CALCULATIONS <<'PY'
import json, os, sys
from pathlib import Path

pool, al_config, wanted = Path(sys.argv[1]), Path(sys.argv[2]), set(sys.argv[3:])
cfg = json.loads((pool / "config.json").read_text())
missing = wanted - set(cfg["calculations"])
if missing:
    sys.exit(f"[al-workers] {sorted(missing)} not in {pool / 'config.json'}")
for name, calc in cfg["calculations"].items():
    calc["enabled"] = name in wanted
text = json.dumps(cfg, indent=2) + "\n"
if not al_config.exists() or al_config.read_text() != text:
    tmp = al_config.with_suffix(".json.tmp")
    tmp.write_text(text)
    os.replace(tmp, al_config)

unclaimed = running = 0
for name in sorted(wanted):
    calc_dir = pool / name
    if not calc_dir.is_dir():
        continue
    for inputs in calc_dir.rglob("inputs"):
        if not inputs.is_dir():
            continue
        job = inputs.parent
        for path in inputs.glob("*.in"):
            stem = path.stem
            if ((job / "outputs" / f"{stem}.out").exists()
                    or (job / "state" / "done" / f"{stem}.json").exists()
                    or (job / "state" / "failed" / f"{stem}.json").exists()):
                continue
            if (job / "state" / "locks" / f"{stem}.lock").exists():
                running += 1
            else:
                unclaimed += 1
print(unclaimed, running)
PY
)"
read -r N_UNCLAIMED N_RUNNING <<< "$COUNTS"

say "pool $POOL  calculations: $CALCULATIONS"
say "outstanding: $N_UNCLAIMED unclaimed, $N_RUNNING running"
if [ "$N_UNCLAIMED" -eq 0 ]; then
    say "nothing unclaimed; submitting no workers"
    exit 0
fi
WANT=$(( N_UNCLAIMED + N_RUNNING ))
[ "$WANT" -le "$TARGET" ] || WANT="$TARGET"

# 3. submit from a clean slate. When this runs inside the driver's own allocation, sbatch
#    would otherwise hand the worker jobs the driver's SLURM_*/SBATCH_* settings.
for var in $(env | awk -F= '/^(SLURM|SBATCH|SALLOC)_/ {print $1}'); do
    unset "$var"
done
[ -z "$TIME_LIMIT" ] || export SBATCH_TIMELIMIT="$TIME_LIMIT"
[ -z "$QOS" ] || export SBATCH_QOS="$QOS"
export SBATCH_ACCOUNT="${RSFFF_NERSC_ACCOUNT:-m3196}"
export ROUNDTRIP_CONFIG="$AL_CONFIG"
export QCHEM_THREADS="$THREADS"

cd "$POOL"
bash scripts/submit_workers.sh --target "$WANT" --job-name "$JOB_NAME" ${DRY_RUN[@]+"${DRY_RUN[@]}"}
