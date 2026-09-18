"""Build stage: random water clusters packed into a spherical cavity with packmol.

Why packmol rather than a hand-rolled inserter
----------------------------------------------
The structures this stage produces are the *starting points* of the sampling that follows, so
the only thing they have to be is (a) chemically plausible -- no two molecules on top of each
other -- and (b) diverse. Packmol solves exactly that problem: it minimizes a penalty that is
zero as soon as every inter-molecular distance exceeds ``tolerance`` and every atom is inside
its constraint region, starting from a random configuration. ``inside sphere x y z r`` is the
spherical cavity, and the random restart per structure is what makes two clusters of the same
size different from each other.

What comes out is not a minimum of anything -- it is a random packing with a hard-sphere floor
under it -- which is why the first sampling stage minimizes with the model before anything is
trusted.

The cavity radius
-----------------
Set from the number of waters at a chosen density::

    R = (3 n V_w / 4 pi)^(1/3) + padding,      V_w = 30.0 A^3 at 1.0 g/cm^3

so a cluster starts near the density of the liquid rather than as a gas that has to collapse
(collapsing is what the sampling is for, but starting 3 A too wide wastes most of the
trajectory getting the molecules into contact). The padding gives packmol room to satisfy the
tolerance at the surface, where the constraint bites hardest.

Atom order and fragments
------------------------
Packmol writes the molecules one after another, so the output is ``O H H O H H ...``: already
in fragment order, which ``rsfff.md.film_driver.water_fragment_index`` requires, and the
fragmentation is therefore known without computing it -- molecule ``k`` is atoms ``3k..3k+2``.
It is written onto the frame (``fragment_idx``, with ``n_fragments``, ``fragment_charges``
and ``fragment_multiplicities``) because every stage downstream wants it: the film model takes
a fixed fragmentation, and a Q-Chem EDA input is built out of one.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import numpy as np

from easyal import Build, Contract, new_frame

__all__ = ["PackmolWaterClusters", "WATER_MONOMER", "cavity_radius"]

#: Experimental gas-phase water: r(OH) = 0.9572 A, angle(HOH) = 104.52 deg. The internal
#: geometry is irrelevant to packmol (it moves molecules rigidly) but it is the geometry the
#: minimizer starts from, so it may as well be the right one.
WATER_MONOMER = (
    ("O", "H", "H"),
    np.array([
        [0.000000, 0.000000, 0.000000],
        [0.957200, 0.000000, 0.000000],
        [-0.239988, 0.926627, 0.000000],
    ]),
)

#: Volume per water at 1 g/cm^3, in A^3: 18.0153 / (0.9970 * 0.602214).
VOLUME_PER_WATER = 30.01


def cavity_radius(n_waters: int, *, density: float = 1.0, padding: float = 1.5) -> float:
    """Radius of the sphere that holds ``n_waters`` at ``density`` g/cm^3, plus ``padding``."""
    volume = n_waters * VOLUME_PER_WATER / density
    return (3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0) + padding


def write_monomer(path: Path) -> Path:
    species, positions = WATER_MONOMER
    lines = [str(len(species)), "water"]
    lines += [f"{s} {x:.6f} {y:.6f} {z:.6f}" for s, (x, y, z) in zip(species, positions)]
    path.write_text("\n".join(lines) + "\n")
    return path


def read_xyz(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text().split("\n")
    n = int(lines[0])
    species, positions = [], []
    for line in lines[2:2 + n]:
        parts = line.split()
        species.append(parts[0])
        positions.append([float(v) for v in parts[1:4]])
    return species, np.asarray(positions, dtype=float)


class PackmolWaterClusters(Build):
    """``(H2O)n`` packed into a sphere, one packmol run per structure.

    Parameters (all keyword, all recorded in ``stage.json``)

    ``sizes``       cluster sizes: ``(lo, hi)`` inclusive, or an explicit list. Default (2, 20)
    ``per_size``    structures per size (default 2)
    ``density``     g/cm^3 the cavity radius is computed from (default 1.0)
    ``padding``     A added to that radius (default 1.5)
    ``radius``      fixed cavity radius in A, overriding ``density``/``padding``
    ``tolerance``   packmol's minimum inter-molecular distance, A (default 2.0)
    ``nloop``       packmol restart budget per structure (default 200)
    ``seed``        base RNG seed; iteration ``i`` uses ``seed + 10000 * i`` so a later
                    iteration packs different structures (default 20260917)
    ``packmol``     the executable (default ``packmol`` on PATH)
    """

    name = "build"
    produces = Contract(
        info=["charge", "multiplicity", "n_waters", "n_fragments", "fragment_charges",
              "fragment_multiplicities"],
        arrays=["fragment_idx"],
    )

    def run(self, ctx):
        p = ctx.params
        exe = shutil.which(p.get("packmol", "packmol"))
        if exe is None:
            raise FileNotFoundError(
                f"packmol executable {p.get('packmol', 'packmol')!r} not found on PATH. "
                f"Install it with `conda install -c conda-forge packmol` or `pip install "
                f"packmol`."
            )
        ctx.track("packmol", exe)

        sizes = p.get("sizes", (2, 20))
        if len(sizes) == 2 and all(isinstance(v, int) for v in sizes) and sizes[0] <= sizes[1]:
            sizes = list(range(int(sizes[0]), int(sizes[1]) + 1))
        sizes = [int(v) for v in sizes]
        per_size = int(p.get("per_size", 2))
        tolerance = float(p.get("tolerance", 2.0))
        nloop = int(p.get("nloop", 200))
        seed0 = int(p.get("seed", 20260917)) + 10000 * ctx.iteration

        monomer = write_monomer(ctx.scratch / "water.xyz")
        frames, failed = [], []
        seed = seed0
        for n in sizes:
            radius = float(p["radius"]) if p.get("radius") else cavity_radius(
                n, density=float(p.get("density", 1.0)), padding=float(p.get("padding", 1.5))
            )
            for k in range(per_size):
                seed += 1
                tag = f"n{n:03d}_{k}"
                out = ctx.scratch / f"{tag}.xyz"
                inp = ctx.scratch / f"{tag}.inp"
                inp.write_text(
                    f"tolerance {tolerance}\n"
                    f"filetype xyz\n"
                    f"output {out.name}\n"
                    f"seed {seed}\n"
                    f"nloop {nloop}\n"
                    f"\n"
                    f"structure {monomer.name}\n"
                    f"  number {n}\n"
                    f"  inside sphere 0. 0. 0. {radius:.4f}\n"
                    f"end structure\n"
                )
                with open(inp) as fh:  # packmol rewinds stdin, so it must be a real file
                    proc = subprocess.run([exe], stdin=fh, cwd=ctx.scratch,
                                          capture_output=True, text=True)
                (ctx.scratch / f"{tag}.log").write_text(proc.stdout + proc.stderr)
                if proc.returncode != 0 or not out.exists():
                    failed.append(tag)
                    continue
                species, positions = read_xyz(out)
                if len(species) != 3 * n:
                    failed.append(tag)
                    continue
                frames.append(new_frame(species, positions, {
                    "charge": 0,
                    "multiplicity": 1,
                    "n_waters": n,
                    "n_fragments": n,
                    "fragment_charges": [0] * n,
                    "fragment_multiplicities": [1] * n,
                    "cavity_radius": round(radius, 4),
                    "packmol_seed": seed,
                    "packmol_tolerance": tolerance,
                    "packmol_converged": "Success" in proc.stdout,
                    "source": f"packmol/{tag}",
                }, fragment_idx=[k for k in range(n) for _ in range(3)]))

        if not frames:
            raise RuntimeError(f"packmol produced nothing; see {ctx.scratch}")
        if failed:
            ctx.note(f"packmol failed on {len(failed)} of {len(sizes) * per_size} structures: "
                     f"{', '.join(failed[:8])}{' ...' if len(failed) > 8 else ''}")
        ctx.log(
            n_structures=len(frames),
            n_failed=len(failed),
            n_forced=sum(1 for f in frames if not f["info"]["packmol_converged"]),
            sizes=[sizes[0], sizes[-1]],
            per_size=per_size,
            seed0=seed0,
            packmol=exe,
        )
        return frames
