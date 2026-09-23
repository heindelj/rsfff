"""The training set, pulled out of the cc_workers store.

    python -m rsfff_al.dataset STORE OUT_DIR [--max-waters 64] [--loop water_udd --upto 3]

Every **done** ``force`` job at wB97M-V/def2-TZVPD on neutral water becomes a training frame,
joined with the ``eda2`` job of the same geometry when there is one. Level of theory is
decided by comparing the job's ``$rem`` with the one this package would write for that
molecule -- which is also what the qchem_roundtrip outputs adopt to, so legacy data, earlier
loops and this loop are one pool, and a geometry that was computed twice is one job.

Written as

    OUT_DIR/eda/w{n}.xyz       frames with EDA (the main cluster stream, ``data.path``)
    OUT_DIR/force/w{n}.xyz     frames without (the force-only stream, ``data.force_path``)
    OUT_DIR/manifest.json      counts per size and kind, the job ids, the filters used

``split_group`` on each frame is the AL trajectory (``traj_id``) when the job has one, else the
geometry id, so the trainer's grouped split holds whole trajectories out.

Which active-learning labels count is a filter on the job tags: ``loops`` (``al_loop``) and
``upto`` (``al_iteration <= upto``); jobs without AL tags (legacy data) are always in unless
``legacy=False``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

__all__ = ["export_training", "is_water", "theory_matches"]


def is_water(mol: dict) -> bool:
    sym = mol["symbols"]
    n_o = sym.count("O")
    if n_o == 0 or sym.count("H") != 2 * n_o or len(sym) != 3 * n_o:
        return False
    if int(mol.get("charge", 0)) != 0 or int(mol.get("multiplicity", 1)) != 1:
        return False
    return True


def theory_matches(record: dict, theory=None) -> bool:
    """True when the job's ``$rem`` is exactly what ``store_io`` writes for its calc."""
    from cc_workers.common.chem.molecule import Molecule
    from cc_workers.workers.qchem.specify import eda2, force

    from .store_io import THEORY

    theory = theory or THEORY
    calc = record["calc"]
    factory = {"force": force, "eda2": eda2}.get(calc)
    if factory is None:
        return False
    mol = Molecule.from_dict(record["spec"]["molecule"])
    want = factory(mol, theory).rem
    have = {k.upper(): str(v) for k, v in record["spec"]["rem"].items()}
    return {k: v.lower() for k, v in have.items()} == {k: v.lower() for k, v in want.items()}


def water_fragments(symbols, coords) -> list[int]:
    """``O H H O H H ...`` -> 0 0 0 1 1 1 ..., checked by nearest oxygen."""
    x = np.asarray(coords, float)
    z = np.array(symbols)
    o = np.flatnonzero(z == "O")
    frag = np.empty(len(z), int)
    frag[o] = np.arange(len(o))
    h = np.flatnonzero(z == "H")
    owner = np.linalg.norm(x[h, None] - x[None, o], axis=-1).argmin(1)
    frag[h] = owner
    if not np.array_equal(frag, np.repeat(np.arange(len(o)), 3)):
        raise ValueError("atoms are not O H H per water in order")
    return frag.tolist()


def export_training(store_dir, out_dir, *, loops=None, upto=None, legacy=True,
                    max_waters=None, min_waters=2, log=print) -> dict:
    from cc_workers.common import store as st
    from cc_workers.common import sync
    from easyal import write_extxyz

    from .store_io import training_frame

    store = st.Store(store_dir, create=False)
    out = Path(out_dir)
    forces, edas = [], defaultdict(list)
    n_seen = 0
    for job in store.jobs("qchem"):
        n_seen += 1
        record = job.record()
        if record["calc"] not in ("force", "eda2"):
            continue
        mol = record["spec"]["molecule"]
        if not is_water(mol):
            continue
        n = len(mol["symbols"]) // 3
        if n < min_waters or (max_waters and n > max_waters):
            continue
        if job.status()["state"] != st.DONE or not theory_matches(record):
            continue
        tag_sets = [record.get("tags", {})] + list(record.get("aliases", []))
        al = [t for t in tag_sets if "al_iteration" in t]
        wanted = any((loops is None or t.get("al_loop") in loops)
                     and (upto is None or int(t["al_iteration"]) <= upto) for t in al)
        if not (wanted or (legacy and len(al) < len(tag_sets))):
            continue
        if not job.result_path.exists():
            sync.parse_job(job)
        if record["calc"] == "force":
            forces.append((job, record, n))
        else:
            edas[record["geometry_id"]].append(job)

    groups = defaultdict(list)
    dropped = []
    for fjob, record, n in forces:
        eda_res = None
        for ejob in edas.get(record["geometry_id"], []):
            eda_res = ejob.load_result()
            break
        tags = record.get("tags", {})
        if eda_res is None and not tags.get("fragment_idx"):
            mol = record["spec"]["molecule"]
            tags = dict(tags, fragment_idx=water_fragments(mol["symbols"], mol["coords"]))
            record = dict(record, tags=tags)
        split = tags.get("traj_id") or record["geometry_id"]
        try:
            frame = training_frame(fjob.load_result(), eda_res, spec_record=record,
                                   extra_info={"split_group": str(split)})
        except ValueError as exc:
            try:                            # a broken EDA join demotes, it does not drop
                frame = training_frame(fjob.load_result(), None, spec_record=record,
                                       extra_info={"split_group": str(split)})
                eda_res = None
            except ValueError:
                dropped.append({"force": fjob.id, "reason": str(exc)})
                continue
        kind = "eda" if eda_res is not None else "force"
        if kind == "eda" and not _canonical_eda(frame):
            frame = training_frame(fjob.load_result(), None, spec_record=record,
                                   extra_info={"split_group": str(split)})
            kind = "force"
        groups[(kind, n)].append((frame, fjob.id))

    manifest = {"store": str(Path(store_dir).resolve()), "loops": loops, "upto": upto,
                "legacy": legacy, "max_waters": max_waters, "n_jobs_scanned": n_seen,
                "files": {}, "dropped": dropped}
    for (kind, n), items in sorted(groups.items()):
        items.sort(key=lambda t: t[1])                  # stable file content across exports
        path = out / kind / f"w{n}.xyz"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_extxyz(path, [f for f, _ in items])
        manifest["files"][str(path.relative_to(out))] = {
            "kind": kind, "n_waters": n, "n_frames": len(items),
            "force_jobs": [j for _, j in items]}
    totals = defaultdict(int)
    for meta in manifest["files"].values():
        totals[meta["kind"]] += meta["n_frames"]
    manifest["totals"] = dict(totals)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    log(f"[dataset] {dict(totals)} frames from {n_seen} jobs scanned; {len(dropped)} dropped "
        f"-> {out}")
    return manifest


#: The EDA component set every existing training file carries; concatenation needs one set.
CANONICAL_EDA = {"cls_elec", "mod_pauli", "disp", "pol", "ct", "prp", "frz", "int", "elec",
                 "pauli"}


def _canonical_eda(frame) -> bool:
    keys = {k[4:] for k in frame["info"] if k.startswith("eda_")}
    extra = keys - CANONICAL_EDA
    for k in extra:                                  # e.g. cls_pauli: not in the old files
        del frame["info"][f"eda_{k}"]
    return CANONICAL_EDA <= keys


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("store")
    ap.add_argument("out")
    ap.add_argument("--loop", action="append", help="al_loop names to include (default all)")
    ap.add_argument("--upto", type=int, default=None)
    ap.add_argument("--no-legacy", action="store_true")
    ap.add_argument("--max-waters", type=int, default=None)
    a = ap.parse_args(argv)
    m = export_training(a.store, a.out, loops=a.loop, upto=a.upto, legacy=not a.no_legacy,
                        max_waters=a.max_waters)
    for path, meta in m["files"].items():
        print(f"{path:24s} {meta['n_frames']:6d}")


if __name__ == "__main__":
    main()
