#!/usr/bin/env python3
"""Turn one eda/force group of the round-trip bundle into training files under data/clusters/.

    python scripts/collect_group.py benchmark                       # eda/benchmark_eda + force/benchmark_force
    python scripts/collect_group.py ion_clusters --eda-dir eda/ion_clusters --force-dir force/ion_clusters
    python scripts/collect_group.py benchmark --stems w13_mp2_avtz  # only some geometry stems

Writes ``data/clusters/<group>/<stem>_<method>_<basis>.xyz``, one file per geometry stem, via
``scripts/parse_roundtrip.py`` -- the same eda+force merge (orientation and energy
cross-checks, dropped frames reported) that made the original w2-w5 files and that the
active-learning label stage uses, so every file under ``data/clusters/`` has one schema.

Two things differ from running ``parse_roundtrip.py`` directly, both about not losing data:

* stems come from the **outputs** present, not from ``geoms/``. A group whose geometry files
  were swapped out after its jobs ran (``benchmark``: the ``*_mp2_avtz`` geometries) still
  has every finished calculation collected.
* files are named by the **whole stem**, not the system label (``w10``). ``parse_roundtrip``
  names by system, so ``w10_mp2_avtz`` and ``w10_wb97mv_def2-tzvpd`` would both write
  ``w10_wb97mv_tzvpd.xyz`` and the second would silently replace the first.

And one thing it adds: **one frame per geometry**. A frame whose nuclei (in Q-Chem's standard
orientation) match a frame already collected -- in another group, or earlier in this one --
is dropped and counted. In ``benchmark`` all 76 ``*_mp2_avtz`` frames are exact repeats (same
geometry, bit-identical energies, EDA and forces) of ``*_wb97mv_def2-tzvpd`` frames; stems
that still have a ``geoms/`` file are preferred, so those are the ones kept.

The Q-Chem parsers are loaded through ``active_learning/common.py``, which needs no pyscf.
Run ``scripts/check_data.py`` afterwards (``stage_data.sh`` does) before training on it.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

TRAIN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TRAIN_DIR.parent / "active_learning"))
sys.path.insert(0, str(TRAIN_DIR / "scripts"))


def output_stems(calc_dir: Path) -> set[str]:
    out = calc_dir / "outputs"
    if not out.is_dir():
        return set()
    return {re.sub(r"(_frame\d+)?\.out$", "", p.name) for p in out.glob("*.out")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("group", help="name of the output directory under data/clusters/")
    ap.add_argument("--root", type=Path, default=TRAIN_DIR.parent,
                    help="the round-trip bundle (default: the one this job sits in)")
    ap.add_argument("--eda-dir", default=None, help="default eda/<group>_eda")
    ap.add_argument("--force-dir", default=None, help="default force/<group>_force")
    ap.add_argument("--stems", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=None,
                    help="default <train>/data/clusters/<group>")
    ap.add_argument("--strict", action="store_true",
                    help="abort on any eda/force consistency warning")
    args = ap.parse_args(argv)

    root = args.root.resolve()
    eda = root / (args.eda_dir or f"eda/{args.group}_eda")
    force = root / (args.force_dir or f"force/{args.group}_force")
    out = (args.out or TRAIN_DIR / "data" / "clusters" / args.group).resolve()
    for d in (eda, force):
        if not d.is_dir():
            print(f"no such directory: {d}", file=sys.stderr)
            return 2

    both = output_stems(eda) & output_stems(force)
    only = (output_stems(eda) | output_stems(force)) - both
    stems = sorted(args.stems or both)
    if only:
        print(f"[collect] skipping stems with outputs on one side only: {sorted(only)}",
              file=sys.stderr)
    if not stems:
        print("[collect] no stems with both eda and force outputs", file=sys.stderr)
        return 1

    from common import roundtrip_parser
    from check_data import frame_keys
    parser = roundtrip_parser()
    parser.system_label = lambda stem: stem       # name by the whole stem: no collisions
    tmp = out.parent / f".{out.name}.parsing"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    argv = ["--root", str(root), "--eda-dir", str(eda), "--force-dir", str(force),
            "--out-dir", str(tmp), "--stems", *stems] + (["--strict"] if args.strict else [])
    print(f"[collect] {len(stems)} stem(s) from {eda.relative_to(root)} + "
          f"{force.relative_to(root)} -> {out}", file=sys.stderr)
    code = parser.main(argv)

    # --- one frame per geometry ---------------------------------------------------------------
    # The same geometry run twice under two stems (benchmark: every *_mp2_avtz job was rerun,
    # identically, as *_wb97mv_def2-tzvpd) is one data point. Frames already in another group
    # win; within the group, stems that still have a geoms/ file (the current names) win.
    seen = set()
    for other in sorted((out.parent).glob("*/*.xyz")):
        if other.parent not in (out, tmp) and not other.parent.name.startswith("."):
            seen.update(frame_keys(other))
    current = set(geometry_stems(eda)) | set(geometry_stems(force))
    parsed = sorted(tmp.glob("*.xyz"),
                    key=lambda p: (not any(p.name.startswith(c + "_") for c in current), p.name))
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    kept_total = dropped_total = 0
    for path in parsed:
        lines = path.read_text().splitlines(keepends=True)
        keep, i = [], 0
        for key in frame_keys(path):
            n = int(lines[i])
            if key not in seen:
                seen.add(key)
                keep += lines[i:i + n + 2]
                kept_total += 1
            else:
                dropped_total += 1
            i += n + 2
        if keep:
            (out / path.name).write_text("".join(keep))
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"[collect] {kept_total} frame(s) kept in {out}; {dropped_total} dropped as repeats "
          f"of a geometry already collected", file=sys.stderr)
    return code if kept_total else 1


def geometry_stems(calc_dir: Path) -> set[str]:
    geoms = calc_dir / "geoms"
    return {p.stem for p in geoms.glob("*xyz")} if geoms.is_dir() else set()


if __name__ == "__main__":
    sys.exit(main())
