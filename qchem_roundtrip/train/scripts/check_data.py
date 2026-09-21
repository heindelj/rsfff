#!/usr/bin/env python3
"""Is every cluster file a set of neutral water clusters with every label the film fit needs?

    python scripts/check_data.py                      # the files water_film.yaml resolves to
    python scripts/check_data.py data/clusters/new/*.xyz

The fit makes no distinction between clusters -- a dimer and a 23-mer from any group are
frames of one dataset -- so this is the one place the "it must be water, at one level of
theory" contract is enforced. Per frame:

* species are O and H only, and every fragment (``fragment_idx``) is exactly one O and two H
  with both O-H distances under 1.3 A (an intact molecule, and the fragmentation the film
  model's ``water_fragment_index`` would assign)
* neutral singlet: ``charge``/``multiplicity`` and ``fragment_charges`` when present
* labels present: ``energy``, ``eda_cls_elec``, ``eda_mod_pauli``, ``eda_disp``, ``eda_pol``,
  ``eda_ct``, ``fragment_energies``, ``fragment_dipoles``, ``fragment_second_moments``, and a
  ``forces`` column

Across files: one ``method``/``basis`` for everything (mixing levels of theory would be a
silent error in every label), and no geometry twice. A duplicate is double weight in the fit
and, split by geometry *per file*, can land in training and validation at once -- a held-out
number that is not held out. (``--allow-duplicates`` makes it a warning.)

Standard library only, so it runs under a bare system python. Exit status non-zero on failure.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REQUIRED = ("energy", "eda_cls_elec", "eda_mod_pauli", "eda_disp", "eda_pol", "eda_ct",
            "fragment_energies", "fragment_dipoles", "fragment_second_moments")
OH_MAX = 1.3
KV = re.compile(r'(\w+)=("([^"]*)"|\S+)')


def frames(path: Path):
    with open(path) as fh:
        while True:
            head = fh.readline()
            if not head.strip():
                return
            n = int(head)
            header = fh.readline()
            rows = [fh.readline().split() for _ in range(n)]
            yield header, rows


def columns(props: str):
    """``Properties=species:S:1:pos:R:3:...`` -> ``{name: (start, width)}``."""
    parts = props.split(":")
    out, col = {}, 0
    for i in range(0, len(parts), 3):
        name, width = parts[i], int(parts[i + 2])
        out[name] = (col, width)
        col += width
    return out


def geometry_key(species, pos) -> str:
    """Orientation-dependent fingerprint of a frame's nuclei (5-decimal Angstrom).

    Q-Chem writes every job in its standard orientation, so two calculations of one geometry
    come back with identical coordinates, which is exactly what this matches.
    """
    atoms = sorted((s, round(x, 5), round(y, 5), round(z, 5)) for s, (x, y, z) in zip(species, pos))
    return hashlib.sha1(repr(atoms).encode()).hexdigest()


def frame_keys(path: Path):
    """``geometry_key`` of every frame in a file, in order."""
    for header, rows in frames(path):
        cols = columns(dict((m.group(1), m.group(3) or m.group(2))
                            for m in KV.finditer(header))["Properties"])
        s0, _ = cols["species"]
        p0, _ = cols["pos"]
        yield geometry_key([r[s0] for r in rows],
                           [tuple(float(x) for x in r[p0:p0 + 3]) for r in rows])


def check_file(path: Path, seen: dict, problems: list, levels: Counter,
               allow_duplicates: bool = False) -> tuple[int, Counter]:
    sizes: Counter = Counter()
    n = 0
    for k, (header, rows) in enumerate(frames(path)):
        n += 1
        where = f"{path.name} frame {k}"
        kv = {m.group(1): (m.group(3) if m.group(3) is not None else m.group(2))
              for m in KV.finditer(header)}
        cols = columns(kv.get("Properties", ""))
        missing = [key for key in REQUIRED if key not in kv]
        if "forces" not in cols:
            missing.append("forces column")
        if "fragment_idx" not in cols:
            problems.append(f"{where}: no fragment_idx column")
            continue
        if missing:
            problems.append(f"{where}: missing {missing}")
        levels[(kv.get("method", "?"), kv.get("basis", "?"))] += 1
        if kv.get("charge", "0") not in ("0", "0.0") or kv.get("multiplicity", "1") != "1":
            problems.append(f"{where}: charge {kv.get('charge')} mult {kv.get('multiplicity')}")
        if any(float(q) != 0.0 for q in kv.get("fragment_charges", "0").split()):
            problems.append(f"{where}: charged fragment(s) {kv['fragment_charges']}")

        s0, _ = cols["species"]
        p0, _ = cols["pos"]
        f0, _ = cols["fragment_idx"]
        species = [r[s0] for r in rows]
        pos = [tuple(float(x) for x in r[p0:p0 + 3]) for r in rows]
        frag = [int(r[f0]) for r in rows]
        if set(species) - {"O", "H"}:
            problems.append(f"{where}: species {sorted(set(species))} (water is O, H)")
            continue
        members = defaultdict(list)
        for i, f in enumerate(frag):
            members[f].append(i)
        bad = 0
        for f, atoms in members.items():
            o = [i for i in atoms if species[i] == "O"]
            h = [i for i in atoms if species[i] == "H"]
            if len(o) != 1 or len(h) != 2 or any(math.dist(pos[o[0]], pos[j]) > OH_MAX for j in h):
                bad += 1
        if bad:
            problems.append(f"{where}: {bad} fragment(s) not an intact H2O")
        sizes[len(members)] += 1

        key = geometry_key(species, pos)
        if key in seen:
            problems.append(f"{'WARN ' if allow_duplicates else ''}{where}: same geometry as "
                            f"{seen[key]}")
        else:
            seen[key] = where
    return n, sizes


def config_files() -> list[Path]:
    sys.path.insert(0, str(HERE.parent))
    import yaml  # noqa: F401  (train_committee needs it)
    from train_committee import DEFAULT_CONFIG, resolve_paths
    tree = yaml.safe_load(DEFAULT_CONFIG.read_text())
    resolve_paths(tree, DEFAULT_CONFIG.parent)
    return [Path(p) for p in tree["data"]["path"]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--quiet", action="store_true", help="summary line only")
    ap.add_argument("--allow-duplicates", action="store_true",
                    help="a geometry appearing twice is a warning, not an error")
    args = ap.parse_args(argv)
    files = args.files or config_files()
    if not files:
        print("no cluster files", file=sys.stderr)
        return 1

    seen: dict = {}
    problems: list[str] = []
    levels: Counter = Counter()
    total, all_sizes = 0, Counter()
    for path in files:
        n, sizes = check_file(path, seen, problems, levels, args.allow_duplicates)
        total += n
        all_sizes.update(sizes)
        if not args.quiet:
            span = f"w{min(sizes)}-w{max(sizes)}" if sizes else "-"
            print(f"  {n:5d}  {span:9s} {path}")
    if len(levels) > 1:
        problems.append(f"mixed levels of theory: {dict(levels)}")
    errors = [p for p in problems if not p.startswith("WARN")]
    warns = [p for p in problems if p.startswith("WARN")]
    for p in (errors + warns)[:40]:
        print("  " + p, file=sys.stderr)
    if len(errors + warns) > 40:
        print(f"  ... {len(errors + warns) - 40} more", file=sys.stderr)
    dist = " ".join(f"w{k}:{v}" for k, v in sorted(all_sizes.items()))
    level = ", ".join(f"{m}/{b}" for m, b in levels)
    print(f"[check-data] {total} frames in {len(files)} file(s), {level}; "
          f"{len(errors)} error(s), {len(warns)} warning(s)\n[check-data] sizes {dist}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
