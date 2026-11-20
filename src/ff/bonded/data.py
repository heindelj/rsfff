"""Labelled molecules (extxyz) and batches for the bonded model.

A frame gives species, positions (Angstrom), ``energy`` (Hartree), ``forces`` (Hartree/bohr), and optionally
``bonds="i j i j ..."`` (0-based pairs), ``charge`` and ``multiplicity``, and the electrical labels ``dipole`` (3),
``quadrupole`` (9: the primitive second moment sum q r r, trace included) and ``polarizability`` (9), all in atomic
units about the file's coordinate origin. Derivative labels, per bohr (the layouts of ``scripts/bonded_data.py``):
``dipole_derivatives`` (N*3*3, [atom, i, k] = d mu_k / d R_ai), ``polarizability_derivatives`` (N*3*9) and ``hessian``
((3N)^2). ``config_type`` is kept as ``Molecule.config`` (the trainer holds frames out by its prefix). Without ``bonds`` the topology is the rule of
the film model's legacy path, accepted only for a molecule with at most one heavy atom: every H is bonded to it.
The topology is never taken from distances.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from bb.graph import BondGraph, bond_graph

from ..film.bonded import BondedTopology, _angles_from_bonds, _impropers_from_bonds
from ..film.topology import (
    angle_pairs_from_angles, atom_degrees, canonical_bonds, lookup_angles, lookup_bonds, torsions_from_bonds,
)

__all__ = ["Batch", "Molecule", "collate", "read_extxyz", "topology"]

SYMBOLS = ["X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl"]
Z_OF = {s: z for z, s in enumerate(SYMBOLS)}


@dataclass
class Molecule:
    z: np.ndarray                 # (n,) atomic numbers
    pos: np.ndarray               # (n, 3) Angstrom
    bonds: np.ndarray             # (b, 2)
    charge: int = 0
    multiplicity: int = 1
    energy: float | None = None   # Hartree
    forces: np.ndarray | None = None      # (n, 3) Hartree/bohr
    name: str = ""
    frequencies: np.ndarray | None = None  # cm^-1, if the frame carries them
    dipole: np.ndarray | None = None          # (3,) e bohr
    second_moment: np.ndarray | None = None   # (3, 3) e bohr^2, primitive
    polarizability: np.ndarray | None = None  # (3, 3) bohr^3
    dipole_derivatives: np.ndarray | None = None          # (n, 3, 3) e, [atom, i, k]
    polarizability_derivatives: np.ndarray | None = None  # (n, 3, 9) bohr^2, [atom, i, jk]
    hessian: np.ndarray | None = None                     # (3n, 3n) Hartree / bohr^2
    config: str = ""


def _default_bonds(z: np.ndarray) -> np.ndarray:
    heavy = np.flatnonzero(z > 1)
    if len(z) == 2 and len(heavy) == 0:
        return np.array([[0, 1]])
    if len(heavy) != 1:
        raise ValueError("a frame without bonds=... needs exactly one heavy atom (every H is then bonded to it); "
                         "write the covalent topology for anything else")
    return np.array([[heavy[0], i] for i in range(len(z)) if i != heavy[0]]).reshape(-1, 2)


def read_extxyz(path, name: str | None = None) -> list[Molecule]:
    path = Path(path)
    name = path.stem if name is None else name
    out, lines, i = [], path.read_text().splitlines(), 0
    key = lambda k, head: re.search(rf'(?<![A-Za-z_]){k}=(?:"([^"]*)"|(\S+))', head)
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        n, head = int(lines[i].split()[0]), lines[i + 1]
        cols = re.search(r"Properties=(\S+)", head).group(1).split(":")
        off, where = 0, {}
        for col, _, w in zip(cols[0::3], cols[1::3], cols[2::3]):
            where[col] = (off, int(w))
            off += int(w)
        rows = [ln.split() for ln in lines[i + 2:i + 2 + n]]
        take = lambda k: np.array([[float(x) for x in r[where[k][0]:where[k][0] + where[k][1]]] for r in rows])
        val = lambda k: (lambda m: None if m is None else (m.group(1) if m.group(1) is not None else m.group(2)))(key(k, head))
        z = np.array([Z_OF[r[0]] for r in rows])
        bonds = val("bonds")
        freq = val("frequencies")
        arr = lambda k, shape: (lambda v: None if v is None or len(v.split()) != int(np.prod(shape))
                                else np.array(v.split(), dtype=float).reshape(shape))(val(k))
        out.append(Molecule(
            z=z, pos=take("pos"),
            bonds=_default_bonds(z) if bonds is None else np.array(bonds.split(), dtype=int).reshape(-1, 2),
            charge=int(float(val("charge") or 0)), multiplicity=int(float(val("multiplicity") or 1)),
            energy=None if val("energy") is None else float(val("energy")),
            forces=take("forces") if "forces" in where else None, name=name,
            frequencies=None if freq is None else np.array(freq.split(), dtype=float),
            dipole=arr("dipole", (3,)), second_moment=arr("quadrupole", (3, 3)), polarizability=arr("polarizability", (3, 3)),
            dipole_derivatives=arr("dipole_derivatives", (n, 3, 3)),
            polarizability_derivatives=arr("polarizability_derivatives", (n, 3, 9)),
            hessian=arr("hessian", (3 * n, 3 * n)), config=val("config_type") or ""))
        i += 2 + n
    return out


def topology(bonds: torch.Tensor, n_atoms: int, mol_idx: torch.Tensor, dtype=torch.float64) -> BondedTopology:
    """The film model's :class:`BondedTopology` of a batch of whole molecules (every weight 1, owner = molecule)."""
    bond_index = canonical_bonds(bonds.t())
    angle_index = _angles_from_bonds(bond_index, n_atoms)
    improper_index = _impropers_from_bonds(bond_index, n_atoms)
    pair_index = angle_pairs_from_angles(bond_index, n_atoms)
    one = lambda idx: torch.ones(idx.shape[1], dtype=dtype, device=bonds.device)
    i, apex, k = (angle_index[r] for r in range(3))
    pa, ps, px, py = (pair_index[r] for r in range(4))
    has_a, has_p = angle_index.shape[1] > 0, pair_index.shape[1] > 0
    empty = bond_index.new_zeros(0)
    tors = torsions_from_bonds(bond_index, n_atoms)
    ta, tb, tc, td = (tors[r] for r in range(4))
    has_t = tors.shape[1] > 0
    return BondedTopology(
        bond_index=bond_index, bond_frag=mol_idx[bond_index[0]], bond_weight=one(bond_index),
        angle_index=angle_index, angle_frag=mol_idx[apex], angle_weight=one(angle_index),
        improper_index=improper_index, improper_frag=mol_idx[improper_index[0]], improper_weight=one(improper_index),
        degree=atom_degrees(bond_index, n_atoms),
        angle_bonds=torch.stack((lookup_bonds(bond_index, n_atoms, i, apex), lookup_bonds(bond_index, n_atoms, apex, k)))
        if has_a else angle_index.new_zeros(2, 0),
        torsion_index=tors, torsion_frag=mol_idx[tb], torsion_weight=one(tors),
        torsion_bonds=torch.stack((lookup_bonds(bond_index, n_atoms, ta, tb), lookup_bonds(bond_index, n_atoms, tb, tc),
                                   lookup_bonds(bond_index, n_atoms, tc, td))) if has_t else tors.new_zeros(3, 0),
        torsion_angles=torch.stack((lookup_angles(angle_index, n_atoms, ta, tb, tc),
                                    lookup_angles(angle_index, n_atoms, tb, tc, td))) if has_t else tors.new_zeros(2, 0),
        angle_pair_index=pair_index,
        angle_pair_angles=torch.stack((lookup_angles(angle_index, n_atoms, ps, pa, px),
                                       lookup_angles(angle_index, n_atoms, ps, pa, py))) if has_p else pair_index.new_zeros(2, 0),
        angle_pair_frag=mol_idx[pa], angle_pair_weight=one(pair_index),
        angle_pair_bond=lookup_bonds(bond_index, n_atoms, pa, ps) if has_p else empty,
        angle_pair_opposite=lookup_angles(angle_index, n_atoms, px, pa, py) if has_p else empty,
    )


@dataclass
class Batch:
    positions: torch.Tensor        # (N, 3) Angstrom
    z: torch.Tensor                # (N,)
    mol_idx: torch.Tensor          # (N,)
    n_mol: int
    charge: torch.Tensor           # (M,)
    multiplicity: torch.Tensor     # (M,)
    graph: BondGraph
    topo: BondedTopology
    energy: torch.Tensor | None = None     # (M,) Hartree
    forces: torch.Tensor | None = None     # (N, 3) Hartree/bohr
    names: tuple = ()
    # electrical labels, NaN rows where a molecule has none (atomic units, coordinate origin)
    dipole: torch.Tensor | None = None             # (M, 3)
    second_moment: torch.Tensor | None = None      # (M, 3, 3)
    polarizability: torch.Tensor | None = None     # (M, 3, 3)
    # derivative labels per bohr: per-atom rows (NaN where the molecule has none), Hessians one per molecule
    dipole_derivatives: torch.Tensor | None = None             # (N, 3, 3)  [atom, i, k]
    polarizability_derivatives: torch.Tensor | None = None     # (N, 3, 9)  [atom, i, jk]
    hessians: tuple | None = None                              # M entries: (3n, 3n) tensor or None
    configs: tuple = ()

    def labelled(self, key: str) -> torch.Tensor:
        """(M,) which molecules carry the label ``key`` (a per-molecule or per-atom tensor, or ``hessians``)."""
        if key == "hessians":
            return torch.tensor([h is not None for h in (self.hessians or (None,) * self.n_mol)], dtype=torch.bool)
        v = getattr(self, key)
        if v is None:
            return torch.zeros(self.n_mol, dtype=torch.bool)
        if v.shape[0] == self.n_mol and v.shape[0] != self.z.shape[0]:
            return ~torch.isnan(v.reshape(self.n_mol, -1)[:, 0])
        atom = ~torch.isnan(v.reshape(v.shape[0], -1)[:, 0])
        return torch.zeros(self.n_mol, dtype=torch.bool, device=v.device).index_put((self.mol_idx[atom],), torch.tensor(True, device=v.device))

    def to(self, device) -> "Batch":
        import dataclasses
        mv = lambda v: (v.to(device) if torch.is_tensor(v) or isinstance(v, BondGraph)
                        else tuple(None if h is None else h.to(device) for h in v) if isinstance(v, tuple) and v and not isinstance(v[0], str) and any(torch.is_tensor(h) for h in v)
                        else v)
        kw = {f.name: mv(getattr(self, f.name)) for f in dataclasses.fields(self) if f.name != "topo"}
        topo = dataclasses.replace(self.topo, **{f.name: mv(getattr(self.topo, f.name)) for f in dataclasses.fields(self.topo)})
        return Batch(**kw, topo=topo)


def collate(mols, dtype=torch.float64) -> Batch:
    off, pos, z, idx, bonds = 0, [], [], [], []
    for m, mol in enumerate(mols):
        pos.append(torch.as_tensor(mol.pos, dtype=dtype))
        z.append(torch.as_tensor(mol.z, dtype=torch.long))
        idx.append(torch.full((len(mol.z),), m, dtype=torch.long))
        bonds.append(torch.as_tensor(mol.bonds, dtype=torch.long).reshape(-1, 2) + off)
        off += len(mol.z)
    mol_idx, bonds = torch.cat(idx), torch.cat(bonds)
    have_e = all(m.energy is not None for m in mols)
    have_f = all(m.forces is not None for m in mols)
    return Batch(
        positions=torch.cat(pos), z=torch.cat(z), mol_idx=mol_idx, n_mol=len(mols),
        charge=torch.tensor([m.charge for m in mols], dtype=dtype),
        multiplicity=torch.tensor([m.multiplicity for m in mols], dtype=dtype),
        graph=bond_graph(bonds, off, check=False), topo=topology(bonds, off, mol_idx, dtype),
        energy=torch.tensor([m.energy for m in mols], dtype=dtype) if have_e else None,
        forces=torch.cat([torch.as_tensor(m.forces, dtype=dtype) for m in mols]) if have_f else None,
        names=tuple(m.name for m in mols), configs=tuple(m.config for m in mols),
        hessians=(tuple(None if m.hessian is None else torch.as_tensor(m.hessian, dtype=dtype) for m in mols)
                  if any(m.hessian is not None for m in mols) else None),
        **{k: (torch.cat([torch.full((len(m.z),) + shape, float("nan"), dtype=dtype) if getattr(m, k) is None
                          else torch.as_tensor(getattr(m, k), dtype=dtype) for m in mols])
               if any(getattr(m, k) is not None for m in mols) else None)
           for k, shape in (("dipole_derivatives", (3, 3)), ("polarizability_derivatives", (3, 9)))},
        **{k: (torch.stack([torch.full(shape, float("nan"), dtype=dtype) if getattr(m, k) is None
                            else torch.as_tensor(getattr(m, k), dtype=dtype) for m in mols])
               if any(getattr(m, k) is not None for m in mols) else None)
           for k, shape in (("dipole", (3,)), ("second_moment", (3, 3)), ("polarizability", (3, 3)))})
