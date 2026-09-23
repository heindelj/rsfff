"""Random (H2O)n packings with packmol: the starting points of the dynamics.

``inside sphere 0 0 0 R`` with R from the cluster size at liquid density plus a padding,
``tolerance`` the minimum distance between atoms of different molecules. Packmol writes the
molecules one after another, so the output is ``O H H O H H ...`` -- already in the fragment
order the film model needs, molecule ``k`` = atoms ``3k..3k+2``.

The packings are **not** minimized. They are random hard-sphere arrangements, a few kcal/mol
per water above anything the model would relax to, and the relaxation out of them is part of
what gets sampled.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import numpy as np

__all__ = ["cavity_radius", "pack_waters", "WATER_MONOMER"]

#: gas-phase water, r(OH) 0.9572 A, HOH 104.52 deg
WATER_MONOMER = (("O", "H", "H"), np.array([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0],
                                            [-0.239988, 0.926627, 0.0]]))
VOLUME_PER_WATER = 30.01        # A^3 at 1 g/cm^3


def cavity_radius(n_waters: int, *, density: float = 1.0, padding: float = 1.5) -> float:
    return (3.0 * n_waters * VOLUME_PER_WATER / density / (4.0 * math.pi)) ** (1 / 3) + padding


def _read_xyz(path: Path):
    lines = path.read_text().split("\n")
    n = int(lines[0])
    rows = [line.split() for line in lines[2:2 + n]]
    return [r[0] for r in rows], np.array([[float(v) for v in r[1:4]] for r in rows])


def pack_waters(n_waters: int, *, seed: int, workdir, tolerance: float = 2.0,
                density: float = 1.0, padding: float = 1.5, nloop: int = 200,
                packmol: str = "packmol") -> dict:
    """One packing. Returns ``{"species", "positions", "info"}``; raises when packmol fails."""
    exe = shutil.which(packmol)
    if exe is None:
        raise FileNotFoundError(f"packmol ({packmol!r}) not on PATH: conda install -c "
                                f"conda-forge packmol, or pip install packmol")
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    mono = work / "water.xyz"
    if not mono.exists():
        sp, xyz = WATER_MONOMER
        mono.write_text("3\nwater\n" + "".join(f"{s} {a:.6f} {b:.6f} {c:.6f}\n"
                                                for s, (a, b, c) in zip(sp, xyz)))
    radius = cavity_radius(n_waters, density=density, padding=padding)
    tag = f"w{n_waters:03d}_s{seed}"
    out, inp = work / f"{tag}.xyz", work / f"{tag}.inp"
    inp.write_text(f"tolerance {tolerance}\nfiletype xyz\noutput {out.name}\nseed {seed}\n"
                   f"nloop {nloop}\n\nstructure {mono.name}\n  number {n_waters}\n"
                   f"  inside sphere 0. 0. 0. {radius:.4f}\nend structure\n")
    with open(inp) as fh:                      # packmol rewinds stdin: it must be a real file
        proc = subprocess.run([exe], stdin=fh, cwd=work, capture_output=True, text=True)
    (work / f"{tag}.log").write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0 or not out.exists():
        raise RuntimeError(f"packmol failed for {tag}; see {work / (tag + '.log')}")
    species, positions = _read_xyz(out)
    if len(species) != 3 * n_waters:
        raise RuntimeError(f"packmol wrote {len(species)} atoms for {n_waters} waters")
    positions -= positions.mean(0)
    return {"species": species, "positions": positions, "info": {
        "charge": 0, "multiplicity": 1, "n_waters": n_waters, "n_fragments": n_waters,
        "fragment_charges": [0] * n_waters, "fragment_multiplicities": [1] * n_waters,
        "cavity_radius": round(radius, 4), "packmol_seed": seed,
        "packmol_tolerance": tolerance, "packmol_converged": "Success" in proc.stdout,
        "source": f"packmol/{tag}"}}
