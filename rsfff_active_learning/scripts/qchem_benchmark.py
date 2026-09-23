#!/usr/bin/env python3
"""What does a label cost? Q-Chem force and EDA wall times by cluster size, before the size
schedule commits to them.

    python scripts/qchem_benchmark.py setup  $SCRATCH/qchem_bench      # specs into a store
    python -m cc_workers.cli submit $SCRATCH/qchem_bench --site perlmutter_cpu --target 7
    python scripts/qchem_benchmark.py report $SCRATCH/qchem_bench      # once they are done

One packmol structure per size, the same level of theory and the same runner the loop uses:
force at 16, 32, 48 and 64 waters; eda2 at 10, 20 and 29 (the largest size that gets an EDA).
The report reads each job's attempt run.json (start, finish, threads) and extrapolates the
node-hours of the default schedule from a power-law fit in the number of waters.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

FORCE_SIZES = (16, 32, 48, 64)
EDA_SIZES = (10, 20, 29)


def setup(store_dir: Path) -> None:
    from cc_workers.common.store import Store
    from easyal import new_frame

    from rsfff_al.build import pack_waters
    from rsfff_al.label import _install_code
    from rsfff_al.store_io import specs_for

    store = Store(store_dir)
    _install_code(store.root)
    for n, labels in [(n, "force") for n in FORCE_SIZES] + [(n, "force,eda") for n in EDA_SIZES]:
        p = pack_waters(n, seed=7 + n, workdir=store.root / "_packmol")
        frame = new_frame(p["species"], p["positions"], {**p["info"], "labels": labels},
                          fragment_idx=[k for k in range(n) for _ in range(3)])
        for calc, spec in specs_for(frame, tags={"benchmark": True, "n_waters": n}).items():
            if calc == "force" and n in EDA_SIZES:
                continue
            job, new = store.add(spec, priority=3 * n)
            print(f"{calc:6s} w{n:<3d} {job.id} {'added' if new else 'exists'}")


def report(store_dir: Path) -> None:
    from cc_workers.common.store import Store

    from rsfff_al import schedule as sched

    store = Store(store_dir, create=False)
    rows = {"force": [], "eda2": []}
    for job in store.jobs("qchem"):
        rec, status = job.record(), job.status()
        n = len(rec["spec"]["molecule"]["symbols"]) // 3
        if status["state"] != "done":
            print(f"{rec['calc']:6s} w{n:<3d} {status['state']}")
            continue
        run = json.loads((job.dir / "attempts" / status["attempt"] / "run.json").read_text())
        hours = (run["finished"] - run["started"]) / 3600
        threads = run.get("resources", {}).get("threads")
        rows[rec["calc"]].append((n, hours))
        print(f"{rec['calc']:6s} w{n:<3d} {hours:7.2f} h on {threads} threads")
    fits = {}
    for calc, pts in rows.items():
        if len(pts) >= 2:
            n, h = np.array(pts).T
            b, a = np.polyfit(np.log(n), np.log(h), 1)
            fits[calc] = (float(np.exp(a)), float(b))
            print(f"{calc}: hours ~ {np.exp(a):.3g} * n^{b:.2f}")
    if "force" not in fits:
        return
    total = {"force": 0.0, "eda2": 0.0}
    for it in range(len(sched.DEFAULT_SCHEDULE)):
        for row in sched.plan_iteration(it):
            n = row["n_waters"]
            a, b = fits["force"]
            total["force"] += row["labels"] * a * n ** b
            if row["eda"] and "eda2" in fits:
                a, b = fits["eda2"]
                total["eda2"] += row["eda"] * a * n ** b
    print(f"default schedule: ~{total['force']:.0f} node-h force + ~{total['eda2']:.0f} node-h "
          f"EDA (one job per node)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("setup", "report"))
    ap.add_argument("store", type=Path)
    a = ap.parse_args(argv)
    (setup if a.action == "setup" else report)(a.store)
    return 0


if __name__ == "__main__":
    sys.exit(main())
