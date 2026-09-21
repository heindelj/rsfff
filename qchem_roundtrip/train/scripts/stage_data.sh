#!/bin/bash
# Build train/data/ from the repository and the round-trip bundle, check it, and pin it.
#
#   bash scripts/stage_data.sh            # on the laptop (needs the repo checkout): build + check + pin
#   bash scripts/stage_data.sh --check    # anywhere (e.g. Perlmutter): verify against data.sha256
#
#   data/clusters/original/    w2-w5 wB97M-V/def2-TZVPD, copied from the repo's data/wb97mv_tzvpd/
#   data/clusters/benchmark/   the benchmark eda+force group, parsed by scripts/collect_group.py
#   data/monomer/              the h2o monomer anchor set
#   data/atomic_references_wb97mv_tzvpd.json
#
# Every cluster file ends up in one dataset (water_film.yaml: data/clusters/**/*.xyz). To add a
# group, add a collect_group.py line below (or run it by hand) and rerun this.
# train/data/ is gitignored (the repo-wide `data/` rule) and travels with the bundle sync;
# data.sha256 is committed, so what a committee was trained on is always checkable.
set -euo pipefail
TRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TRAIN_DIR"
if command -v sha256sum >/dev/null; then SHA=(sha256sum); else SHA=(shasum -a 256); fi

if [ "${1:-}" = "--check" ]; then
    out=$("${SHA[@]}" -c data.sha256 2>&1) || { echo "$out" | grep -v ': OK$' >&2; exit 1; }
    echo "[stage-data] all $(wc -l < data.sha256 | tr -d ' ') pinned files present and unchanged"
    extra=$(find data -type f | sort | while read -r f; do grep -q "  $f\$" data.sha256 || echo "$f"; done)
    [ -z "$extra" ] || { echo "[stage-data] files in data/ that are not pinned (would be trained on):" >&2; echo "$extra" >&2; exit 1; }
    exit 0
fi

REPO="${RSFFF_REPO:-$(cd "$TRAIN_DIR/../.." && pwd)}"
SRC="$REPO/data"
[ -d "$SRC/wb97mv_tzvpd" ] || { echo "no $SRC/wb97mv_tzvpd; set RSFFF_REPO" >&2; exit 1; }

mkdir -p data/clusters/original data/monomer
for n in 2 3 4 5; do
    cp -p "$SRC/wb97mv_tzvpd/w${n}_wb97mv_tzvpd.xyz" data/clusters/original/
done
cp -p "$SRC/wb97mv_tzvpd/h2o_wb97mv_tzvpd_pol.xyz" data/monomer/
cp -p "$SRC/atomic_references_wb97mv_tzvpd.json" data/
echo "[stage-data] copied w2-w5, the monomer set and the reference energies from $SRC"

# --- round-trip groups: one line each ------------------------------------------------------
rm -rf data/clusters/benchmark
RSFFF_REPO="$REPO" python3 scripts/collect_group.py benchmark

python3 scripts/check_data.py data/clusters/*/*.xyz
find data -type f | sort | xargs "${SHA[@]}" > data.sha256
echo "[stage-data] pinned $(wc -l < data.sha256 | tr -d ' ') file(s) in train/data.sha256"
