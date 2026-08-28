"""Run the biased-MD harvester over every isomer in the ion-cluster sets.

The previous harvest started from neutral benchmark clusters with a proton bolted on, which is
a poor starting point: the ion is placed by a heuristic and the hydrogen-bond network has never
relaxed around it. ``data/hydronium_clusters_ccdb/`` and ``data/hydroxide_clusters/`` are real
optimized isomers, so each one is a basin the sampler can start from instead of fall into.

    python scripts/sweep_cluster_md.py --max-waters 7 --jobs 12
    python scripts/sweep_cluster_md.py --max-waters 7 --dry-run     # just list the work

Not every isomer has something to sample
----------------------------------------
Measured over the 174 isomers at N <= 7: **all 102 hydronium ones enumerate a competing
decomposition, and 24 of the 72 hydroxide ones do not.** The asymmetry is chemistry, not a bug.
A hydronium *donates* -- its three protons point at acceptor oxygens, so one is always inside
the 2.2 Angstrom envelope and a transfer is always enumerable. A hydroxide has to *accept*, and
in 24 of these optimized structures no water donates to it: the shortest water-H to hydroxide-O
distance averages 3.68 Angstrom against 2.07 in the live ones, and 19 of the 24 have the
hydroxide donating its own hydrogen to a water instead.

Those runs harvest nothing and stop on the no-competition guard. That is the right outcome --
there is no proton poised to move, so anything sampled there would be manufactured rather than
found -- but it is worth knowing before reading the per-isomer counts.

Two sets, two layouts, one code path
------------------------------------
The hydronium files put the ion **last** (``[O H H] x N`` then ``[O H H H]``); the hydroxide
file puts it **first** (``[O H]`` then the waters); the training corpus disagrees with both;
and ``Isomer 7j`` of the hydroxide file writes one water as ``H, O, H``. None of that is handled
here, and that is the point -- ``run_reactive_md.py`` infers the charge from the composition and
takes its fragments from the distance-minimizing assignment, so nothing anywhere reads an atom
position. A layout-aware reader would have to special-case all four cases and would still be
wrong on the one malformed frame.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parent.parent
HYDRONIUM = ROOT / "data" / "hydronium_clusters_ccdb"
HYDROXIDE = ROOT / "data" / "hydroxide_clusters" / "jp5b03893_si_002.xyz"


def _frame_sizes(path: Path) -> list[int]:
    """Atom count of every frame, without parsing coordinates."""
    lines = path.read_text().split("\n")
    sizes, i = [], 0
    while i < len(lines) and lines[i].strip():
        n = int(lines[i])
        sizes.append(n)
        i += n + 2
    return sizes


def enumerate_jobs(max_waters: int) -> list[dict]:
    """Every (file, frame) to sample, as ``{path, frame, label, waters, ion}``.

    ``waters`` counts the neutral waters, so a label of ``w3`` is the ion plus three waters --
    four oxygens either way. Derived from the atom count rather than the file name, because the
    hydroxide file has no per-size names and its isomer labels are not trustworthy (``Isomer
    5k`` appears twice; one of them is a 14-atom cluster that should read ``4k``).
    """
    jobs: list[dict] = []

    for path in sorted(HYDRONIUM.glob("asp-H2O_*--H3O+.xyz"),
                       key=lambda p: int(re.search(r"_(\d+)--", p.name).group(1))):
        waters = int(re.search(r"_(\d+)--", path.name).group(1))
        if waters > max_waters:
            continue
        for k in range(len(_frame_sizes(path))):
            jobs.append(dict(path=path, frame=k, waters=waters, ion="h3o+",
                             label=f"h3o+_w{waters}_iso{k:02d}"))

    if HYDROXIDE.exists():
        counter: dict[int, int] = {}
        for k, natoms in enumerate(_frame_sizes(HYDROXIDE)):
            waters = (natoms - 2) // 3          # OH-(H2O)n has 3n + 2 atoms
            if waters < 1 or waters > max_waters:
                continue                        # frame 0 is a bare OH-: nothing to mediate
            idx = counter.get(waters, 0)
            counter[waters] = idx + 1
            jobs.append(dict(path=HYDROXIDE, frame=k, waters=waters, ion="oh-",
                             label=f"oh-_w{waters}_iso{idx:02d}"))
    return jobs


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-waters", type=int, default=7)
    p.add_argument("--jobs", type=int, default=12, help="concurrent runs")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--target-frames", type=int, default=120)
    p.add_argument("--out", default="qchem_roundtrip/biased_sampling/clusters")
    p.add_argument("--ion", choices=["h3o+", "oh-"], help="restrict to one ion")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-existing", action="store_true",
                   help="leave alone any label that already has a harvest")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="everything after this is passed through to run_reactive_md.py")
    args = p.parse_args(argv)

    jobs = [j for j in enumerate_jobs(args.max_waters)
            if args.ion is None or j["ion"] == args.ion]
    out_root = ROOT / args.out
    if args.skip_existing:
        jobs = [j for j in jobs
                if not (out_root / j["label"] / "transition_structures.xyz").exists()]

    by_ion: dict[str, int] = {}
    for j in jobs:
        by_ion[j["ion"]] = by_ion.get(j["ion"], 0) + 1
    print(f"{len(jobs)} isomers up to {args.max_waters} waters: "
          + ", ".join(f"{n} {ion}" for ion, n in sorted(by_ion.items())), flush=True)

    if args.dry_run:
        for j in jobs:
            print(f"  {j['label']:22s} {j['path'].name} frame {j['frame']}")
        return 0

    out_root.mkdir(parents=True, exist_ok=True)
    runner = ROOT / "scripts" / "run_reactive_md.py"
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", OMP_NUM_THREADS="1")
    done = {"n": 0, "structures": 0, "failed": 0, "t0": time.time()}

    def run(job: dict) -> None:
        target = out_root / job["label"]
        cmd = [sys.executable, str(runner),
               "--geometry", str(job["path"]), "--frame", str(job["frame"]),
               "--steps", str(args.steps), "--target-frames", str(args.target_frames),
               "--out", str(target), *args.extra]
        log = out_root / f"{job['label']}.log"
        with open(log, "w") as fh:
            # A blowup the harvester cannot recover from is expected on some isomers; it must
            # not take the sweep down with it.
            code = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env,
                                   cwd=str(ROOT))
        traj = target / "transition_structures.xyz"
        n = sum(1 for line in traj.open() if line.startswith("Properties=")) if traj.exists() else 0
        done["n"] += 1
        done["structures"] += n
        done["failed"] += int(code != 0 and n == 0)
        rate = done["structures"] / max((time.time() - done["t0"]) / 60, 1e-9)
        print(f"  [{done['n']:3d}/{len(jobs)}] {job['label']:22s} {n:4d} structures  "
              f"(total {done['structures']}, {rate:.0f}/min)", flush=True)

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(run, jobs))

    elapsed = time.time() - done["t0"]
    print(f"\n{done['structures']} structures from {done['n']} isomers in {elapsed/60:.1f} min "
          f"({done['failed']} produced nothing) -> {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
