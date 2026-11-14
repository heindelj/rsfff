#!/usr/bin/env python
"""Write the data bundle of the cmm salts runs (rsfff.train.cmm_worker) from the two projects.

    python scripts/cmm_bundle.py OUT.tar.gz [--rsfff ~/research/rsfff] [--ct ~/research/ct_model]

Collects (symlinks followed): research/rsfff train/salts/data/ct_sets, train/nacl/ionct/data/{walk_eda,force},
train/salts/data/force_test.txt, train/salts/params/{ions_salts8_cpksb.json,wself.json}; research/ct_model
data/ct_sets/{al-nacl-r0,al-nacl-r1,al-nacl-lin,al-nacl-holes,nacl-compressed}.jsonl. MANIFEST.json lists every file
with its sha256.
"""
import argparse, hashlib, io, json, tarfile, time
from pathlib import Path

CT_FILES = ("al-nacl-r0", "al-nacl-r1", "al-nacl-lin", "al-nacl-holes", "nacl-compressed")

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("out", type=Path)
ap.add_argument("--rsfff", type=Path, default=Path.home() / "research/rsfff")
ap.add_argument("--ct", type=Path, default=Path.home() / "research/ct_model")
ap.add_argument("--ct-sets", type=Path, default=None, help="where the seed-* symlinks' targets are, if they dangle here")
a = ap.parse_args()
R, C = a.rsfff, a.ct
items = []                                                       # (archive name, source file)
for p in sorted((R / "train/salts/data/ct_sets").iterdir()):
    src = p if p.exists() else (a.ct_sets or C / "data/ct_sets") / p.name      # a dangling absolute symlink
    items.append((f"ct_sets/{p.name}", src))
for sub in ("walk_eda", "force"):
    for p in sorted((R / "train/nacl/ionct/data" / sub).rglob("*")):
        if p.is_file():
            items.append((f"{sub}/{p.relative_to(R / 'train/nacl/ionct/data' / sub)}", p))
for n in CT_FILES:
    items.append((f"ct/{n}.jsonl", C / "data/ct_sets" / f"{n}.jsonl"))
for n in ("ions_salts8_cpksb.json", "wself.json"):
    items.append((f"params/{n}", R / "train/salts/params" / n))
items.append(("force_test.txt", R / "train/salts/data/force_test.txt"))
manifest = {"made": time.strftime("%Y-%m-%d %H:%M:%S"), "files": {}}
a.out.parent.mkdir(parents=True, exist_ok=True)
with tarfile.open(a.out, "w:gz") as tar:
    for name, src in items:
        b = Path(src).read_bytes()
        manifest["files"][name] = {"sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b), "source": str(src)}
        ti = tarfile.TarInfo(name); ti.size = len(b); ti.mtime = 0
        tar.addfile(ti, io.BytesIO(b))
    mb = json.dumps(manifest, indent=1).encode()
    ti = tarfile.TarInfo("MANIFEST.json"); ti.size = len(mb); ti.mtime = 0
    tar.addfile(ti, io.BytesIO(mb))
print(f"{a.out}: {len(items)} files, {sum(v['bytes'] for v in manifest['files'].values()) / 1e6:.1f} MB")
