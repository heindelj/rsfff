#!/bin/bash
# One iteration of the water active-learning loop, end to end, inside an interactive
# allocation -- packmol, both sampling stages, real Q-Chem jobs, and the merge back into
# training frames. Nothing is queued: the workers run here, in this allocation, and the label
# stage polls for them instead of stopping at "pending".
#
#   salloc -A m3196 -N 1 -C cpu -q interactive -t 02:00:00
#   module load python qchem
#   source active_learning/scripts/nersc_env.sh   # conda prefix + $RSFFF_QCHEM_ROOT
#   bash $RSFFF_QCHEM_ROOT/active_learning/scripts/smoke_test.sh --fast
#
# nersc_env.sh activates /global/cfs/cdirs/m3196/heindelj/rsfff and points the stages at the
# pool on CFS; do those two by hand instead if you prefer. --pool overrides the pool for one
# run.
#
# It works in its own directory ($SCRATCH/rsfff_al_smoke by default) with its own copy of the
# job pool's config and templates, so the production pool under qchem_roundtrip/ is never
# touched and nothing it produces can be mistaken for training data.
#
#   --root DIR         where to work           (default $SCRATCH/rsfff_al_smoke)
#   --checkpoint PATH  the model to sample with
#   --waters N         cluster size            (default 2)
#   --frames N         MD frames to label      (default 2, so 2N Q-Chem jobs)
#   --workers N        concurrent Q-Chem workers (default 2)
#   --threads N        threads per worker      (default 128/workers)
#   --fast             wB97X-D/6-31G* instead of the production level of theory
#   --timeout SECONDS  how long to wait for the jobs (default 3600)
#   --seed-data PATH   labeled extxyz the committee is fitted on alongside the smoke run's
#                      own frames. Defaults to the first --seed-frames of
#                      <repo>/data/wb97mv_tzvpd/w2_wb97mv_tzvpd.xyz when that exists. Without
#                      it the train stage has only the handful of frames this run labeled,
#                      which is too few to split into a train and a holdout set
#   --seed-frames N    frames to take from it (default 40)
#   --members N        committee size (default 2 here; the loop's own default is 4)
#   --epochs N         epochs per member (default 1 -- this is a plumbing test)
#   --skip-preflight   go straight to the loop
#   --reuse            keep an existing --root instead of starting clean
#   --pool DIR         the job pool to copy config and templates from
#                      (default $RSFFF_QCHEM_ROOT, else <repo>/qchem_roundtrip)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"                 # <bundle>/active_learning
POOL_SRC="${RSFFF_QCHEM_ROOT:-$(cd "$AL_ROOT/.." && pwd)}"
REPO_ROOT="${RSFFF_REPO:-$(cd "$POOL_SRC/.." && pwd)}"

ROOT="${SCRATCH:-/tmp}/rsfff_al_smoke"
CHECKPOINT="$REPO_ROOT/checkpoints/water_film_full/best.pt"
WATERS=2
FRAMES=2
WORKERS=2
THREADS=""
FAST=0
TIMEOUT=3600
PREFLIGHT=1
REUSE=0
SEED_DATA=""
SEED_FRAMES=40
MEMBERS=2
EPOCHS=1

while [ "$#" -gt 0 ]; do
    case "$1" in
        --root)           ROOT="${2:?}"; shift 2 ;;
        --pool)           POOL_SRC="${2:?}"; shift 2 ;;
        --checkpoint)     CHECKPOINT="${2:?}"; shift 2 ;;
        --waters)         WATERS="${2:?}"; shift 2 ;;
        --frames)         FRAMES="${2:?}"; shift 2 ;;
        --workers)        WORKERS="${2:?}"; shift 2 ;;
        --threads)        THREADS="${2:?}"; shift 2 ;;
        --timeout)        TIMEOUT="${2:?}"; shift 2 ;;
        --seed-data)      SEED_DATA="${2:?}"; shift 2 ;;
        --seed-frames)    SEED_FRAMES="${2:?}"; shift 2 ;;
        --members)        MEMBERS="${2:?}"; shift 2 ;;
        --epochs)         EPOCHS="${2:?}"; shift 2 ;;
        --fast)           FAST=1; shift ;;
        --skip-preflight) PREFLIGHT=0; shift ;;
        --reuse)          REUSE=1; shift ;;
        -h|--help)        sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "[smoke] unknown option: $1" >&2; exit 2 ;;
    esac
done

[ -n "$THREADS" ] || THREADS=$(( 128 / WORKERS ))
[ "$THREADS" -ge 1 ] || THREADS=1

POOL="$ROOT/pool"
LOOP="$ROOT/loop"
LOGS="$ROOT/logs"

say() { echo "[smoke] $*"; }

# --- 0. sanity ------------------------------------------------------------------------------
command -v qchem >/dev/null 2>&1 || {
    say "qchem is not on PATH -- 'module load qchem' first (the workers need it)"; exit 2; }
[ -f "$CHECKPOINT" ] || { say "no checkpoint at $CHECKPOINT (pass --checkpoint)"; exit 2; }
[ -f "$POOL_SRC/config.json" ] || { say "no job pool at $POOL_SRC"; exit 2; }
if [ -z "${SLURM_JOB_ID:-}" ]; then
    say "warning: no SLURM_JOB_ID -- this looks like a login node. Q-Chem belongs in an"
    say "         allocation: salloc -A m3196 -N 1 -C cpu -q interactive -t 02:00:00"
fi

# --- 1. an isolated pool --------------------------------------------------------------------
if [ "$REUSE" -eq 0 ] && [ -d "$ROOT" ]; then
    say "clearing $ROOT (pass --reuse to keep it)"
    rm -rf "$ROOT"
fi
mkdir -p "$POOL/templates" "$LOGS"
cp "$POOL_SRC/config.json" "$POOL/config.json"
cp "$POOL_SRC"/templates/*.in "$POOL/templates/" 2>/dev/null || true
if [ "$FAST" -eq 1 ]; then
    cp "$AL_ROOT/templates/eda_fast.in" "$AL_ROOT/templates/force_fast.in" "$POOL/templates/"
    python3 - "$POOL/config.json" <<'PY'
import json, sys
path = sys.argv[1]
cfg = json.loads(open(path).read())
cfg["calculations"]["eda"]["template"] = "templates/eda_fast.in"
cfg["calculations"]["force"]["template"] = "templates/force_fast.in"
# only these two are wanted here; leaving the others enabled would let a worker wander off
# into whatever else the config knows about
for name, calc in cfg["calculations"].items():
    calc["enabled"] = name in ("eda", "force")
open(path, "w").write(json.dumps(cfg, indent=2) + "\n")
PY
    say "level of theory: wB97X-D/6-31G* (--fast; plumbing only, not training data)"
else
    python3 - "$POOL/config.json" <<'PY'
import json, sys
path = sys.argv[1]
cfg = json.loads(open(path).read())
for name, calc in cfg["calculations"].items():
    calc["enabled"] = name in ("eda", "force")
open(path, "w").write(json.dumps(cfg, indent=2) + "\n")
PY
    say "level of theory: the production templates"
fi
say "pool $POOL"
say "loop $LOOP"

# --- 2. preflight ---------------------------------------------------------------------------
if [ "$PREFLIGHT" -eq 1 ]; then
    say "preflight"
    python3 "$SCRIPT_DIR/preflight.py" --checkpoint "$CHECKPOINT" --root "$LOOP" \
        --roundtrip-root "$POOL" || {
        say "preflight failed; fix the above before spending the allocation"; exit 1; }
fi

# --- 2b. seed data, so the committee has something to split ----------------------------------
# A committee fitted on this run's two or three labeled frames cannot be split into a train and
# a holdout set, and the train stage would fail for a reason that has nothing to do with the
# plumbing. A few dozen frames of existing data make it a real fit, briefly.
SEED_ARG=()
if [ -z "$SEED_DATA" ] && [ -f "$REPO_ROOT/data/wb97mv_tzvpd/w2_wb97mv_tzvpd.xyz" ]; then
    SEED_DATA="$REPO_ROOT/data/wb97mv_tzvpd/w2_wb97mv_tzvpd.xyz"
fi
if [ -n "$SEED_DATA" ]; then
    SEED_FILE="$ROOT/seed_${SEED_FRAMES}.extxyz"
    python3 "$AL_ROOT/scripts/take_frames.py" "$SEED_DATA" "$SEED_FILE" "$SEED_FRAMES"
    SEED_ARG=(--initial-data "$SEED_FILE")
else
    say "no seed data found; the train stage will have only this run's labeled frames, which"
    say "is too few to split. Pass --seed-data <labeled extxyz> to make training meaningful."
fi

# --- 3. workers, here, in the background ----------------------------------------------------
WORKER_PIDS=()
cleanup() {
    local pid
    for pid in "${WORKER_PIDS[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
    sleep 1
    for pid in "${WORKER_PIDS[@]:-}"; do
        kill -9 "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

say "starting $WORKERS worker(s), $THREADS threads each; logs in $LOGS"
for i in $(seq 1 "$WORKERS"); do
    python3 "$POOL_SRC/scripts/qchem_roundtrip.py" \
        --root "$POOL" --config "$POOL/config.json" \
        worker --qchem-command qchem --threads "$THREADS" \
        --poll-seconds 10 --idle-timeout-seconds "$TIMEOUT" \
        > "$LOGS/worker_$i.log" 2>&1 &
    WORKER_PIDS+=($!)
done

# --- 4. the loop ----------------------------------------------------------------------------
say "running the loop (it polls for the Q-Chem jobs rather than stopping at pending)"
set +e
python3 "$AL_ROOT/workflows.py" water \
    --root "$LOOP" \
    --checkpoint "$CHECKPOINT" \
    --roundtrip-root "$POOL" \
    --sizes "$WATERS" "$WATERS" --per-size 1 \
    --steps $(( FRAMES * 100 )) --stride 100 --equilibrate 200 \
    --wait "$TIMEOUT" --poll 15 \
    --max-failed-fraction 0.0 \
    --members "$MEMBERS" --epochs "$EPOCHS" --no-stages \
    "${SEED_ARG[@]}" \
    --iterations 1
STATUS=$?
set -e

# --- 5. what came out -----------------------------------------------------------------------
say "----------------------------------------------------------------"
python3 - "$LOOP" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
for stage in ("build", "optimize", "dynamics", "select", "label", "train", "assess"):
    record = root / "iter_000" / stage / "stage.json"
    if not record.exists():
        print(f"  {stage:<9} not run"); continue
    r = json.loads(record.read_text())
    m = r.get("metrics", {})
    keep = {k: v for k, v in m.items() if k.startswith("n_") or k in ("seconds", "stem")}
    print(f"  {stage:<9} {r['status']:<9} {r.get('elapsed_seconds', 0):7.1f}s  {keep}")
    if r["status"] != "complete" and r.get("message"):
        print(f"            {r['message']}")
labeled = root / "iter_000" / "label" / "labeled.extxyz"
if labeled.exists():
    text = labeled.read_text().splitlines()
    print(f"\n  labeled.extxyz: {labeled}")
    print(f"  first header:\n    {text[1][:400]}")
dropped = root / "iter_000" / "label" / "scratch" / "dropped.json"
if dropped.exists():
    print(f"\n  dropped: {dropped.read_text()}")
PY
say "----------------------------------------------------------------"
if [ "$STATUS" -ne 0 ]; then
    say "the loop exited $STATUS; worker logs are in $LOGS"
    exit "$STATUS"
fi
say "done. Re-run with --reuse to continue from where it stopped."
