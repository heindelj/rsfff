#!/bin/bash
# Fill train/data/ with the files water_film.yaml names, and pin them in train/data.sha256.
#
#   bash scripts/stage_data.sh                 # on the laptop, from anywhere: copy + pin
#   bash scripts/stage_data.sh --check         # anywhere (e.g. Perlmutter): verify the pins
#
# The source is the repository's data/ ($RSFFF_REPO, default: the checkout this bundle sits in).
# train/data/ is gitignored (the repo-wide `data/` rule) and travels to Perlmutter with the
# bundle sync (scripts/sync_inputs_up.sh); data.sha256 is committed, so what a committee was
# trained on is checkable against the repository forever after.
set -euo pipefail
TRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-$TRAIN_DIR/water_film.yaml}"
cd "$TRAIN_DIR"

if command -v sha256sum >/dev/null; then SHA=(sha256sum); else SHA=(shasum -a 256); fi
if [ "${1:-}" = "--check" ]; then
    "${SHA[@]}" -c data.sha256 && echo "[stage-data] all pinned files present and unchanged"
    exit
fi

REPO="${RSFFF_REPO:-$(cd "$TRAIN_DIR/../.." && pwd)}"
[ -d "$REPO/data" ] || { echo "no data/ under $REPO; set RSFFF_REPO" >&2; exit 1; }

# every data path in the config, relative to train/ (so data/... -> $REPO/data/...)
FILES=()   # (no mapfile: macOS ships bash 3.2)
while IFS= read -r line; do FILES+=("$line"); done < <(python3 - "$CONFIG" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
try:
    import yaml
except ImportError:   # a bare system python: every data/... token in the file
    print("\n".join(dict.fromkeys(re.findall(r"^[\s-]*(?:\w+:\s*)?(data/\S+)", text, re.M))))
    sys.exit()
d = yaml.safe_load(text)["data"]
for k in ("path", "reference_energies", "atomic_reference_states", "isolated_species",
          "diabatic_states", "monomer_path", "large_path"):
    v = d.get(k)
    for p in (v if isinstance(v, list) else [v] if v else []):
        print(p)
PY
)
for rel in "${FILES[@]}"; do
    case "$rel" in /*) echo "absolute path $rel in config; make it relative to train/" >&2; exit 1;; esac
    mkdir -p "$(dirname "$rel")"
    cp -p "$REPO/$rel" "$rel"
    echo "[stage-data] $REPO/$rel -> train/$rel"
done
"${SHA[@]}" "${FILES[@]}" > data.sha256
echo "[stage-data] pinned ${#FILES[@]} file(s) in train/data.sha256"
