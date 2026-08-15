"""Fold Q-Chem monomer polarizability tensors into the monomer anchor extxyz.

    python scripts/parse_polarizability.py \
        --outputs qchem_roundtrip/polarizability/outputs \
        --anchor  data/wb97mv_tzvpd/h2o_wb97mv_tzvpd.xyz \
        --out     data/wb97mv_tzvpd/h2o_wb97mv_tzvpd_pol.xyz

Adds one ``polarizability="..."`` header per frame, in atomic units (a0^3, row-major 3x3),
which is what :func:`rsfff.train.data.load_extxyz` already reads and converts. Nothing else in
the file is touched: the comment line is edited textually rather than round-tripped through
ASE, so the octopole and hexadecapole blocks and every other header come through byte for byte.

The rotation is the whole job
-----------------------------
``JOBTYPE = polarizability`` reports its tensor in Q-Chem's **standard nuclear orientation**,
not in the orientation the coordinates were submitted in, and it reorients even with
``SYMMETRY = false``. A tensor is not a scalar; dropping it into a file whose coordinates are
in a different frame silently rotates the target and no unit check would catch it -- the
eigenvalues, the isotropic average and the anisotropy all survive the error intact.

So each output's standard orientation is Kabsch-aligned onto the anchor frame's coordinates and
the tensor is carried through the same rotation, ``alpha' = R alpha R^T``. The alignment itself
is the verification: for a rigid three-atom match the residual RMSD is ~1e-6 Angstrom, and
anything above ``--rmsd-tol`` means the two files are not the same geometry and the script
refuses rather than writing a quietly-rotated label.

``R`` is constrained to a proper rotation (``det = +1``). An improper one would be an equally
good coordinate fit for a planar molecule -- water is planar, so this is not hypothetical -- and
would reflect the tensor.

Frames are matched **by position in the file**, then checked by internal coordinates. The two
sets came from the same sampling run, so the order agrees; the check is what makes that a fact
rather than an assumption.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

_ORIENT = "Standard Nuclear Orientation (Angstroms)"
_TENSOR = "Polarizability tensor      [a.u.]"


def parse_output(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """``(species, positions (n, 3) Angstrom, alpha (3, 3) a.u.)`` from one Q-Chem output."""
    lines = path.read_text().splitlines()

    try:
        start = next(k for k, ln in enumerate(lines) if _ORIENT in ln) + 3
    except StopIteration:
        raise ValueError(f"{path}: no standard nuclear orientation block") from None
    species, xyz = [], []
    for ln in lines[start:]:
        if set(ln.strip()) <= {"-"}:
            break
        parts = ln.split()
        species.append(parts[1])
        xyz.append([float(v) for v in parts[2:5]])

    # The last one, not the first: a job that restarts prints the block more than once and
    # only the final tensor corresponds to the converged finite-difference set.
    hits = [k for k, ln in enumerate(lines) if _TENSOR in ln]
    if not hits:
        raise ValueError(
            f"{path}: no polarizability tensor -- the job did not finish, or it was not a "
            f"JOBTYPE = polarizability run"
        )
    rows = [[float(v) for v in lines[hits[-1] + 1 + k].split()] for k in range(3)]
    alpha = np.asarray(rows, dtype=np.float64)
    # Finite-difference dipoles make the raw tensor slightly non-symmetric (~1e-4 a.u. of
    # 9.5). The physical object is symmetric, so symmetrize rather than pick a triangle.
    return species, np.asarray(xyz, dtype=np.float64), 0.5 * (alpha + alpha.T)


class Frame:
    """One extxyz frame, held as its own lines so unlabeled frames can simply be dropped."""

    def __init__(self, lines: list[str]) -> None:
        self.lines = lines
        n = int(lines[0])
        self.comment = lines[1]
        self.species = [lines[2 + a].split()[0] for a in range(n)]
        self.xyz = np.asarray(
            [[float(v) for v in lines[2 + a].split()[1:4]] for a in range(n)]
        )

    def with_header(self, comment: str) -> list[str]:
        return [self.lines[0], comment, *self.lines[2:]]


def read_frames(path: Path) -> list[Frame]:
    """Every frame of an extxyz, as :class:`Frame` blocks in file order."""
    lines = path.read_text().splitlines()
    frames, k = [], 0
    while k < len(lines):
        n = int(lines[k])
        frames.append(Frame(lines[k:k + 2 + n]))
        k += 2 + n
    return frames


def kabsch(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    """Proper rotation ``R`` minimizing ``|R (source - c_s) - (target - c_t)|``, and the RMSD.

    Both sets are centered first, so the translation is removed and only the rotation acts on
    the tensor -- which is right: a polarizability is origin-independent.
    """
    a = source - source.mean(axis=0)
    b = target - target.mean(axis=0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T      # det(R) = +1 by construction
    rmsd = float(np.sqrt(((a @ r.T - b) ** 2).sum(axis=1).mean()))
    return r, rmsd


def internals(xyz: np.ndarray) -> np.ndarray:
    """Sorted bond lengths plus the angle: a frame-independent fingerprint for a 3-atom set."""
    o, h1, h2 = xyz[0], xyz[1], xyz[2]
    r1, r2 = np.linalg.norm(h1 - o), np.linalg.norm(h2 - o)
    ang = np.degrees(np.arccos(np.dot(h1 - o, h2 - o) / (r1 * r2)))
    return np.array(sorted([r1, r2]) + [ang])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outputs", required=True, type=Path, help="directory of Q-Chem .out files")
    p.add_argument("--anchor", required=True, type=Path, help="monomer extxyz to augment")
    p.add_argument("--out", required=True, type=Path, help="where to write the augmented file")
    p.add_argument("--rmsd-tol", type=float, default=1.0e-4,
                   help="Angstrom; above this the frames are not the same geometry")
    p.add_argument("--key", default="polarizability", help="header key to write")
    args = p.parse_args()

    # Matched by the frame index in the filename, not by sorted position: one job in 500
    # failed to converge on the reference set, and matching positionally would have shifted
    # every later frame's tensor onto the wrong geometry. The alignment check below would
    # have caught it -- but only because the check exists, which is the point of having it.
    by_index = {}
    for path in args.outputs.glob("*.out"):
        m = re.search(r"(\d+)\D*$", path.stem)
        if m is None:
            raise SystemExit(f"{path}: no frame index in the filename")
        by_index[int(m.group(1))] = path
    frames = read_frames(args.anchor)

    pairs = [(by_index[k], f) for k, f in enumerate(frames) if k in by_index]
    missing = sorted(set(range(len(frames))) - set(by_index))
    if not pairs:
        raise SystemExit(f"no Q-Chem output matched any of the {len(frames)} anchor frames")

    worst_rmsd, worst_int, iso, out_lines = 0.0, 0.0, [], []
    for out_path, frame in pairs:
        sp, source, alpha = parse_output(out_path)
        if sp != frame.species:
            raise SystemExit(f"{out_path}: species {sp} but anchor frame has {frame.species}")
        d_int = float(np.abs(internals(source) - internals(frame.xyz)).max())
        rot, rmsd = kabsch(source, frame.xyz)
        if rmsd > args.rmsd_tol:
            raise SystemExit(
                f"{out_path}: aligns onto its anchor frame with RMSD {rmsd:.3e} Angstrom "
                f"(tolerance {args.rmsd_tol:.1e}), internal-coordinate difference {d_int:.3e}. "
                f"These are not the same geometry, so the tensor would be written into the "
                f"wrong frame."
            )
        worst_rmsd, worst_int = max(worst_rmsd, rmsd), max(worst_int, d_int)
        a = rot @ alpha @ rot.T
        iso.append(np.trace(a) / 3.0)
        comment = re.sub(rf'\s*{re.escape(args.key)}="[^"]*"', "", frame.comment)
        flat = " ".join(f"{v:.10e}" for v in a.reshape(-1))
        out_lines += frame.with_header(f'{comment} {args.key}="{flat}"')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out_lines) + "\n")
    iso = np.asarray(iso)
    print(
        f"wrote {len(pairs)} of {len(frames)} frames to {args.out}\n"
        f"  dropped, no Q-Chem output      {missing if missing else 'none'}\n"
        f"  worst alignment RMSD           {worst_rmsd:.3e} Angstrom\n"
        f"  worst internal-coord mismatch  {worst_int:.3e}\n"
        f"  isotropic polarizability       {iso.mean():.4f} +/- {iso.std():.4f} a.u. "
        f"(range {iso.min():.4f} - {iso.max():.4f})"
    )
    if missing:
        print(
            "  frames without a label are dropped rather than left unlabeled: "
            "rsfff.train.data requires response labels to be all-or-nothing per file."
        )


if __name__ == "__main__":
    main()
