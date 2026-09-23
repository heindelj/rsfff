"""Rigid scan geometries for the pairing model: bond stretches, a bend, and proton transfers.

    python scripts/pairing_scans.py --out qchem_roundtrip/force/geoms

Writes one multi-frame extxyz per scan, in the header format the Q-Chem round-trip generator
expects (``charge`` and ``multiplicity`` per frame; ``scan`` / ``coord`` are bookkeeping keys
the harvester carries along). Every scan moves **one** coordinate and holds every other atom
at the wB97M-V/def2-TZVPD optimized geometry (``data/wb97mv_tzvpd/*_opt_*_pol.xyz``), so a
curve reads as a curve: these are diagnostics of the pairing term against the Pauli wall
(the O-H stretches: where the bond order lets go), of the quadrupolar valence density (the
bend), and of the valence competition (the shared proton). They are not sampled training
data; jitter is deliberately absent.

Scans
-----
h2o_stretch, h3o+_stretch, oh-_stretch : one O-H from 0.70 to 3.50 A in 0.05 A steps
h2o_bend                               : H-O-H from 60 to 180 deg in 5 deg steps, both r at r_eq
h5o2+_pt, h3o2-_pt                     : the shared proton along the O...O axis, at O-O
                                         distances 2.40 / 2.50 / 2.70 / 2.90 A, 0.05 A steps
                                         between 0.90 A from either oxygen (idealized, not
                                         relaxed: the two water / hydroxide halves are the
                                         optimized monomers with their bisectors on the axis
                                         and their planes perpendicular)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

# wB97M-V/def2-TZVPD optimized monomers (data/wb97mv_tzvpd/*_opt_wb97mv_tzvpd_pol.xyz), Angstrom.
H2O = (["O", "H", "H"], np.array([
    [0.0000000000, 0.0000000000, 0.1171760333],
    [-0.7627894871, 0.0000000000, -0.4684780077],
    [0.7627894871, 0.0000000000, -0.4684780077],
]))
H3O = (["O", "H", "H", "H"], np.array([
    [0.0000000000, 0.0000000000, 0.0753585837],
    [0.9407713077, 0.0000000000, -0.2009581529],
    [-0.4703856538, -0.8147318516, -0.2009581529],
    [-0.4703856538, 0.8147318516, -0.2009581529],
]))
OH = (["O", "H"], np.array([
    [0.0000000000, 0.0000000000, -0.1072139917],
    [0.0000000000, 0.0000000000, 0.8576999128],
]))

STRETCH = np.round(np.arange(0.70, 3.50 + 1e-9, 0.05), 3)
BEND_DEG = np.arange(60.0, 180.0 + 1e-9, 5.0)
OO_DISTANCES = (2.40, 2.50, 2.70, 2.90)


def _frame(species, pos, charge, mult, scan, coord, **extra):
    keys = " ".join(f"{k}={v}" for k, v in extra.items())
    header = (
        f"Properties=species:S:1:pos:R:3 charge={charge} multiplicity={mult} "
        f"scan={scan} coord={coord:.4f}" + (f" {keys}" if keys else "")
    )
    lines = [str(len(species)), header]
    for s, (x, y, z) in zip(species, pos):
        lines.append(f"{s:2s} {x:16.10f} {y:16.10f} {z:16.10f}")
    return "\n".join(lines) + "\n"


def stretch(name, monomer, charge, moving=1):
    species, pos0 = monomer
    o = pos0[0]
    unit = (pos0[moving] - o) / np.linalg.norm(pos0[moving] - o)
    frames = []
    for r in STRETCH:
        pos = pos0.copy()
        pos[moving] = o + r * unit
        frames.append(_frame(species, pos, charge, 1, name, r, r_oh=f"{r:.3f}"))
    return frames


def bend(name, monomer, charge):
    species, pos0 = monomer
    o = pos0[0]
    r_eq = np.linalg.norm(pos0[1] - o)
    fixed = pos0[2] - o
    fixed_unit = fixed / np.linalg.norm(fixed)
    normal = np.array([0.0, 1.0, 0.0])           # the molecular plane is xz
    in_plane = np.cross(normal, fixed_unit)
    frames = []
    for deg in BEND_DEG:
        t = math.radians(deg)
        pos = pos0.copy()
        pos[1] = o + r_eq * (math.cos(t) * fixed_unit + math.sin(t) * in_plane)
        frames.append(_frame(species, pos, charge, 1, name, deg, angle_deg=f"{deg:.1f}"))
    return frames


def _rot_x(deg):
    t = math.radians(deg)
    return np.array([[1, 0, 0], [0, math.cos(t), -math.sin(t)], [0, math.sin(t), math.cos(t)]])


def water_half(sign, o_pos, twist_deg):
    """A water's two hydrogens with the bisector pointing along ``sign * x`` from ``o_pos``."""
    _, pos0 = H2O
    h = pos0[1:] - pos0[0]                        # bisector along -z, plane xz
    # map -z -> sign*x : (x, y, z) -> (-sign*z, y, x) keeps a right-handed frame up to sign
    h = np.stack([-sign * h[:, 2], h[:, 1], h[:, 0]], axis=1)
    h = h @ _rot_x(twist_deg).T
    return o_pos + h


def hydroxide_half(sign, o_pos, tilt_deg, twist_deg):
    """A hydroxide's hydrogen at ``tilt_deg`` from the ``sign * x`` axis, r = r_eq(OH-)."""
    _, pos0 = OH
    r = np.linalg.norm(pos0[1] - pos0[0])
    t = math.radians(tilt_deg)
    h = np.array([sign * r * math.cos(t), r * math.sin(t), 0.0]) @ _rot_x(twist_deg).T
    return o_pos + h


def proton_transfer(name, charge, halves):
    frames = []
    for d in OO_DISTANCES:
        o1 = np.zeros(3)
        o2 = np.array([d, 0.0, 0.0])
        for x in np.round(np.arange(0.90, d - 0.90 + 1e-9, 0.05), 3):
            species = ["O"]
            pos = [o1]
            pos_h1 = halves(-1, o1, 0.0)
            species += ["H"] * len(pos_h1); pos += list(pos_h1)
            species += ["O"]; pos.append(o2)
            pos_h2 = halves(+1, o2, 90.0)
            species += ["H"] * len(pos_h2); pos += list(pos_h2)
            species += ["H"]; pos.append(np.array([x, 0.0, 0.0]))
            frames.append(_frame(
                species, np.array(pos), charge, 1, name, x - d / 2,
                r_oo=f"{d:.2f}", r_o1h=f"{x:.3f}", r_o2h=f"{d - x:.3f}",
            ))
    return frames


def build():
    return {
        "pairing_scan_h2o_stretch": stretch("h2o_stretch", H2O, 0),
        "pairing_scan_h3o+_stretch": stretch("h3o+_stretch", H3O, 1),
        "pairing_scan_oh-_stretch": stretch("oh-_stretch", OH, -1),
        "pairing_scan_h2o_bend": bend("h2o_bend", H2O, 0),
        "pairing_scan_h5o2+_pt": proton_transfer(
            "h5o2+_pt", 1, lambda sign, o, twist: water_half(sign, o, twist)
        ),
        "pairing_scan_h3o2-_pt": proton_transfer(
            "h3o2-_pt", -1, lambda sign, o, twist: hydroxide_half(sign, o, 110.0, twist)[None]
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("qchem_roundtrip/force/geoms"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    total = 0
    for name, frames in build().items():
        path = args.out / f"{name}.xyz"
        path.write_text("".join(frames))
        print(f"{path}: {len(frames)} frames")
        total += len(frames)
    print(f"{total} frames")


if __name__ == "__main__":
    main()
