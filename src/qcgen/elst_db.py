"""Monomer-density database and samplers for reference electrostatics (on top of :mod:`elst_ref`).

Frozen-density electrostatics is exactly pairwise additive: a cluster's energy is the sum of its pair
energies, and the field at any atom is the sum of the other monomers' fields. Everything a fit of the
electrostatic channel needs therefore comes from two kinds of record, both generated here without any
dimer or cluster calculation:

* **monomer grids** -- potential and field of one monomer at points on scaled van der Waals surfaces,
  from deep inside the penetration region (0.6-1.0 x vdW) out to the multipolar region (2-3 x vdW);
* **pairs** -- two monomers placed at a sampled closest-contact distance with random orientations:
  frozen electrostatic energy by parts, and the potential and field of each monomer at the other's
  nuclei (the inter-fragment fields a polarisation or field-feature model sees).

Layout of a database directory::

    db/monomers/<name>.npz      elst_ref.Monomer (AO density + Coulomb fit), one per geometry
    db/index.json               name -> formula, charge, spin, xc, basis, energy, dipole, n_atoms
    db/grids/<name>.npz         points (Bohr), phi, field (au), scale of each point
    db/grids_inner/<name>.npz   the same from 0.2 x vdW (elst_fit.INNER_SCALES)
    db/pairs_<tag>.extxyz       pair records (positions in Angstrom; energies Hartree; phi, field au)

Units: atomic units for every label; extxyz positions in Angstrom (ase convention).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import elst_ref as E

BOHR = E.BOHR


# ---------------------------------------------------------------------------
# database
# ---------------------------------------------------------------------------
class MonomerDB:
    """A directory of fitted monomer densities with a JSON index."""

    def __init__(self, root):
        self.root = Path(root)
        (self.root / "monomers").mkdir(parents=True, exist_ok=True)
        self._index_path = self.root / "index.json"
        self.index = json.loads(self._index_path.read_text()) if self._index_path.exists() else {}
        self._cache: dict[str, E.Monomer] = {}

    def __len__(self):
        return len(self.index)

    def names(self, charge=None, formula=None):
        return [n for n, v in self.index.items()
                if (charge is None or v["charge"] == charge) and (formula is None or v["formula"] == formula)]

    def path(self, name):
        return self.root / "monomers" / f"{name}.npz"

    def get(self, name) -> E.Monomer:
        if name not in self._cache:
            self._cache[name] = E.Monomer.load(self.path(name))
        return self._cache[name]

    def add(self, name, symbols, coords_ang, charge=0, spin=0, xc="wb97m-v", basis="def2-tzvpd",
            auxbasis="def2-universal-jkfit", overwrite=False, **extra) -> bool:
        """SCF + fit one monomer and store it. Returns False if it was already there."""
        if name in self.index and self.path(name).exists() and not overwrite:
            return False
        m = E.Monomer.from_scf(symbols, coords_ang, charge=charge, spin=spin, xc=xc, basis=basis, auxbasis=auxbasis)
        m.save(self.path(name))
        p = m.place()
        mu = -np.einsum("ij,xji->x", p.dm, p.mol.intor("int1e_r")) + p.charges @ p.coords
        self.index[name] = {"formula": _formula(symbols), "charge": int(charge), "spin": int(spin), "xc": xc,
                            "basis": basis, "auxbasis": auxbasis, "n_atoms": len(symbols),
                            "energy": float(m.meta.get("energy", np.nan)), "dipole_au": [float(x) for x in mu],
                            **{k: _jsonable(v) for k, v in extra.items()}}
        self._index_path.write_text(json.dumps(self.index, indent=1))
        self._cache[name] = m
        return True


def build_db(root, structures, xc="wb97m-v", basis="def2-tzvpd", auxbasis="def2-universal-jkfit",
             overwrite=False, log=print) -> MonomerDB:
    """Add every structure to the database at ``root`` (resumable: existing names are skipped).

    ``structures`` yields dicts with ``name``, ``symbols``, ``coords`` (Angstrom) and optionally
    ``charge``, ``spin`` and extra metadata -- e.g. from :func:`structures_from_file`.
    """
    db = MonomerDB(root)
    for s in structures:
        s = dict(s)
        name, sym, xyz = s.pop("name"), s.pop("symbols"), s.pop("coords")
        try:
            new = db.add(name, sym, xyz, s.pop("charge", 0), s.pop("spin", 0), xc=xc, basis=basis,
                         auxbasis=auxbasis, overwrite=overwrite, **s)
            if log and new:
                log(f"{name}: {db.index[name]['formula']} q={db.index[name]['charge']} stored")
        except Exception as exc:  # noqa: BLE001 - keep going, report
            if log:
                log(f"{name}: failed ({exc})")
    return db


def structures_from_file(path, label=None, charge_key="charge", spin_key="spin"):
    """Monomer structures from any ase-readable file (xyz/extxyz, all frames).

    Charge and spin come from ``atoms.info`` (defaults 0); names are ``<label>_<frame:05d>``.
    """
    from ase.io import read
    label = label or Path(path).stem
    for k, at in enumerate(read(str(path), index=":")):
        yield {"name": f"{label}_{k:05d}", "symbols": at.get_chemical_symbols(), "coords": at.get_positions(),
               "charge": int(at.info.get(charge_key, 0)), "spin": int(at.info.get(spin_key, 0)),
               "source": str(path), "frame": k}


# ---------------------------------------------------------------------------
# monomer grids
# ---------------------------------------------------------------------------
def vdw_radii(symbols) -> np.ndarray:
    """Bondi-type vdW radii (Bohr) from ase, with 2.0 A for elements ase does not list."""
    from ase.data import atomic_numbers, vdw_radii as R
    out = []
    for s in symbols:
        r = R[atomic_numbers[s]]
        out.append((2.0 if not np.isfinite(r) else r) / BOHR)
    return np.asarray(out)


def surface_points(coords, radii, scales=(0.6, 0.8, 1.0, 1.2, 1.4, 1.7, 2.0, 2.5, 3.0), density=2.0, rng=None):
    """Points on scaled vdW surfaces (Merz-Kollman style, extended inward).

    For each scale s, random points on the spheres of radius s*R_i around every atom are kept if no
    other atom j is closer than s*R_j. ``density`` is points per Bohr^2 of sphere area. Returns
    ``(points, scale_of_point)`` in Bohr.
    """
    rng = np.random.default_rng(rng)
    coords, radii = np.asarray(coords, float), np.asarray(radii, float)
    pts, sc = [], []
    for s in scales:
        for i, (c, r) in enumerate(zip(coords, radii)):
            n = max(8, int(density * 4 * np.pi * (s * r) ** 2))
            u = rng.normal(size=(n, 3))
            p = c + s * r * u / np.linalg.norm(u, axis=1)[:, None]
            d = np.linalg.norm(p[:, None] - coords[None], axis=2) / (s * radii[None])
            d[:, i] = np.inf
            keep = d.min(1) >= 1.0 - 1e-9
            pts.append(p[keep])
            sc.append(np.full(keep.sum(), s))
    return np.concatenate(pts), np.concatenate(sc)


def monomer_grid(db: MonomerDB, name, scales=(0.6, 0.8, 1.0, 1.2, 1.4, 1.7, 2.0, 2.5, 3.0), density=2.0,
                 rng=None, overwrite=False, gradients=False, cache="grids"):
    """Potential and field (and, with ``gradients``, the field gradient dE_a/dr_b) of one monomer on scaled
    vdW surfaces; cached in ``db/<cache>/<name>.npz`` (a cached grid gains ``field_grad`` on request).
    Use a separate ``cache`` per set of ``scales`` (e.g. ``elst_fit.INNER_SCALES`` in ``"grids_inner"``):
    a cached grid is returned as is."""
    out = db.root / cache / f"{name}.npz"
    if out.exists() and not overwrite:
        rec = dict(np.load(out))
        if gradients and "field_grad" not in rec:
            rec["field_grad"] = E.field_gradient(db.get(name).place(), rec["points"])
            np.savez(out, **rec)
        return rec
    out.parent.mkdir(exist_ok=True)
    m = db.get(name)
    p = m.place()
    pts, sc = surface_points(p.coords, vdw_radii(m.symbols), scales, density, rng)
    rec = {"points": pts, "scale": sc, "phi": E.potential(p, pts), "field": E.field(p, pts),
           "coords": p.coords, "symbols": np.asarray(m.symbols)}
    if gradients:
        rec["field_grad"] = E.field_gradient(p, pts)
    np.savez(out, **rec)
    return rec


# ---------------------------------------------------------------------------
# pairs
# ---------------------------------------------------------------------------
def place_at_contact(A: E.Monomer, B: E.Monomer, contact, rng=None, rot_a=None, rot_b=None, direction=None,
                     tol=1e-4):
    """A at the origin, B along a random direction so that the closest A..B atom distance is ``contact``
    (Bohr); random orientations unless given. Bisection on the centroid distance."""
    rng = np.random.default_rng(rng)
    ra = E.random_rotation(rng) if rot_a is None else rot_a
    rb = E.random_rotation(rng) if rot_b is None else rot_b
    u = rng.normal(size=3) if direction is None else np.asarray(direction, float)
    u /= np.linalg.norm(u)
    xa = (A.coords - A.coords.mean(0)) @ ra.T
    xb0 = (B.coords - B.coords.mean(0)) @ rb.T

    def dmin(D):
        return np.linalg.norm(xa[:, None] - (xb0 + D * u)[None], axis=2).min()

    hi = contact + np.linalg.norm(xa, axis=1).max() + np.linalg.norm(xb0, axis=1).max() + 1.0
    while dmin(hi) < contact:
        hi *= 2
    # walk inward to the outermost crossing (interpenetrating bodies make dmin non-monotone), then bisect
    lo = hi
    while lo > 0 and dmin(lo) >= contact:
        lo -= 0.25
    lo = max(lo, 0.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if dmin(mid) < contact:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    D = hi
    pa = A.place(ra, -ra @ A.coords.mean(0))
    pb = B.place(rb, -rb @ B.coords.mean(0) + D * u)
    return pa, pb


def pair_record(pa: E.Placed, pb: E.Placed, exact_ee=False, gradients=False) -> dict:
    """Energies by parts and the inter-fragment potential/field (and field gradient) at every nucleus (au)."""
    e = E.pair_energy(pa, pb, exact_ee=exact_ee)
    phi = np.concatenate([E.potential(pb, pa.coords), E.potential(pa, pb.coords)])
    fld = np.concatenate([E.field(pb, pa.coords), E.field(pa, pb.coords)])
    rec = {"energy": e, "phi_inter": phi, "field_inter": fld}
    if gradients:
        rec["field_grad_inter"] = np.concatenate([E.field_gradient(pb, pa.coords),
                                                  E.field_gradient(pa, pb.coords)]).reshape(-1, 9)
    return rec


def sample_pairs(db: MonomerDB, n, out, contact_range_ang=(1.4, 6.0), names_a=None, names_b=None, rng=None,
                 exact_ee=False, tag=None, log=print, every=100, gradients=False):
    """Sample ``n`` monomer pairs and append them to the extxyz file ``out``.

    Each pair: two monomers drawn uniformly from ``names_a`` / ``names_b`` (default: the whole database),
    random orientations, a random direction, and the closest interatomic contact drawn uniformly in
    ``contact_range_ang`` (Angstrom) -- uniform in contact samples the penetration region as densely as
    the long range. Written per frame: positions (A), ``fragment_idx``, ``phi_inter`` and
    ``field_inter`` (per atom, au); info: ``e_elst`` and its parts (Hartree), monomer names, the
    placement (rotations, shifts), the contact and centroid distances.
    """
    from ase import Atoms
    from ase.io import write
    rng = np.random.default_rng(rng)
    names_a = names_a or list(db.index)
    names_b = names_b or list(db.index)
    lo, hi = (c / BOHR for c in contact_range_ang)
    frames = []
    for k in range(n):
        na, nb = names_a[rng.integers(len(names_a))], names_b[rng.integers(len(names_b))]
        A, B = db.get(na), db.get(nb)
        contact = rng.uniform(lo, hi)
        pa, pb = place_at_contact(A, B, contact, rng)
        rec = pair_record(pa, pb, exact_ee, gradients)
        sym = list(A.symbols) + list(B.symbols)
        at = Atoms(sym, positions=np.concatenate([pa.coords, pb.coords]) * BOHR)
        at.arrays["fragment_idx"] = np.r_[np.zeros(len(A.symbols), int), np.ones(len(B.symbols), int)]
        at.arrays["phi_inter"] = rec["phi_inter"]
        at.arrays["field_inter"] = rec["field_inter"]
        if gradients:
            at.arrays["field_grad_inter"] = rec["field_grad_inter"]     # dE_a/dr_b, row-major (xx, xy, ..., zz)
        e = rec["energy"]
        at.info.update({"e_elst": e["total"], "e_elst_ee": e["ee"], "e_elst_en": e["en"], "e_elst_ne": e["ne"],
                        "e_elst_nn": e["nn"], "mono_a": na, "mono_b": nb,
                        "charge_a": db.index[na]["charge"], "charge_b": db.index[nb]["charge"],
                        "contact_ang": contact * BOHR, "r_com_ang": float(np.linalg.norm(pb.com - pa.com)) * BOHR,
                        "rot_a": pa.rot.ravel(), "rot_b": pb.rot.ravel(), "shift_a": pa.shift, "shift_b": pb.shift,
                        "elst_method": "frozen_exact_ee" if exact_ee else "frozen_robust_df",
                        "units": "energies Hartree; phi, field atomic units; positions Angstrom"})
        if tag:
            at.info["tag"] = tag
        frames.append(at)
        if len(frames) >= every or k == n - 1:
            write(str(out), frames, format="extxyz", append=Path(out).exists())
            if log:
                log(f"{k + 1}/{n} pairs -> {out}")
            frames = []
    return out


def reassemble(db: MonomerDB, frame):
    """Re-place a pair record's monomers from its stored placement (for checks and cluster assembly)."""
    i = frame.info
    pa = db.get(i["mono_a"]).place(np.asarray(i["rot_a"]).reshape(3, 3), np.asarray(i["shift_a"]))
    pb = db.get(i["mono_b"]).place(np.asarray(i["rot_b"]).reshape(3, 3), np.asarray(i["shift_b"]))
    return pa, pb


# ---------------------------------------------------------------------------
def _formula(symbols):
    from collections import Counter
    c = Counter(symbols)
    return "".join(f"{s}{c[s] if c[s] > 1 else ''}" for s in sorted(c, key=lambda s: (s != "C", s != "H", s)))


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    return v
