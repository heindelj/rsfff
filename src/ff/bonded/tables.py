"""The per-type constants of the bonded terms: what the network modulates.

An atom's type is ``(Z, covalent degree)``; a bond's is the unordered pair of its atoms' types; angles, couplings
and impropers are typed by their apex. The tables hold unconstrained variables (logs, atanh), one row per type:

    bond  (T, T, 3)   ln r_eq [bohr], ln D, ln k                        (symmetrized over the two orders)
    apex  (T, 8)      atanh cos(theta_eq), ln k_theta, raw rho_bb, rho_ba, rho_aa, rho_bo, c_chi, ln k_chi
    e0    (T,)        reference energy of an atom of this type (Hartree; fitted to the data, see the trainer)
    torsion (T, T, T, T, N)   K_n of a-b-c-d in units of 1e-3 Hartree, by the four atoms' types (symmetrized over
                      the reversal d-c-b-a); zero to start with
    atom  (T, 5)      chi, ln eta, ln alpha [bohr^3], ln c_quad, trace of the atomic second moment
    link  (T, T, 2)   ln s (bond compliance; symmetrized), bond flow (antisymmetrized; ``charges = "flow"``)

The electrical rows have no fit file yet: chi = 0, eta = s = 0.5, c_quad = 1, alpha = 0.6 x the free atom's.

with rho = 0.95 tanh(raw). They start from built-in defaults, or from the ``tables`` section of a constant-parameter
fit (``scripts/fit_constant_monomers.py``): ``{"bond": {"1.1-8.2": {r_eq, D, k}}, "apex": {"8.2": {theta_eq_deg,
k_theta, rho_bb, ...}}}``, keys from :func:`type_key`.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import nn

from bb.graph import COVALENT_RADII

from ..film.terms import RHO_MAX
from ..units import BOHR_ANG

__all__ = ["APEX", "ATOM", "BOND", "BondedTables", "atom_types_of", "type_key"]

BOND = ("r_eq", "D", "k")
ATOM = ("chi", "eta", "alpha", "c_quad", "m2_trace")
#: free-atom static dipole polarizabilities (bohr^3; Schwerdtfeger & Nagle 2018), for the initial tables only
FREE_ATOM_ALPHA = {1: 4.5, 6: 11.3, 7: 7.4, 8: 5.3, 9: 3.74, 15: 25.0, 16: 19.4, 17: 14.6}
APEX = ("cos_theta_eq", "k_theta", "rho_bb", "rho_ba", "rho_aa", "rho_bo", "c_chi", "k_chi")
_DEFAULT = dict(D=0.16, k=0.50, theta_eq_deg=109.47, k_theta=0.15, rho_bb=0.0, rho_ba=0.0, rho_aa=0.0, rho_bo=0.0,
                c_chi=-0.5, k_chi=0.005)


def type_key(*types) -> str:
    """``"8.2"`` for an atom type, ``"1.1-8.2"`` for a bond (sorted)."""
    return "-".join(f"{int(z)}.{int(d)}" for z, d in sorted((int(z), int(d)) for z, d in types))


def atom_types_of(batch) -> list[tuple[int, int]]:
    """The sorted ``(Z, degree)`` types present in a batch."""
    pairs = torch.stack((batch.z, batch.topo.degree), 1)
    return sorted({(int(z), int(d)) for z, d in torch.unique(pairs, dim=0).tolist()})


class BondedTables(nn.Module):
    def __init__(self, atom_types, init: str | Path = "", learn: bool = True, torsion_order: int = 0):
        super().__init__()
        self.atom_types = [(int(z), int(d)) for z, d in atom_types]
        T = len(self.atom_types)
        given = json.loads(Path(init).read_text()).get("tables", {}) if init else {}
        gb, ga = given.get("bond", {}), given.get("apex", {})
        self.from_file = {"bond": [], "apex": []}
        bond = torch.zeros(T, T, 3, dtype=torch.float64)
        apex = torch.zeros(T, len(APEX), dtype=torch.float64)
        for a, ta in enumerate(self.atom_types):
            row = dict(_DEFAULT)
            if type_key(ta) in ga:
                row.update(ga[type_key(ta)])
                self.from_file["apex"].append(type_key(ta))
            apex[a] = torch.tensor([
                math.atanh(math.cos(math.radians(row["theta_eq_deg"]))), math.log(row["k_theta"]),
                *(math.atanh(row[f"rho_{c}"] / RHO_MAX) for c in ("bb", "ba", "aa", "bo")),
                row["c_chi"], math.log(row["k_chi"])])
            for b, tb in enumerate(self.atom_types):
                r0 = (COVALENT_RADII[ta[0]] + COVALENT_RADII[tb[0]]) / BOHR_ANG
                row = {"r_eq": r0, "D": _DEFAULT["D"], "k": _DEFAULT["k"]}
                if type_key(ta, tb) in gb:
                    row.update(gb[type_key(ta, tb)])
                    if a <= b:
                        self.from_file["bond"].append(type_key(ta, tb))
                bond[a, b] = torch.tensor([math.log(row["r_eq"]), math.log(row["D"]), math.log(row["k"])])
        lut = torch.full((120, 9), -1, dtype=torch.long)
        for t, (z, d) in enumerate(self.atom_types):
            lut[z, d] = t
        self.register_buffer("lut", lut, persistent=False)
        self.bond = nn.Parameter(bond, requires_grad=bool(learn))
        self.apex = nn.Parameter(apex, requires_grad=bool(learn))
        self.e0 = nn.Parameter(torch.zeros(T, dtype=torch.float64))
        self.torsion = (nn.Parameter(torch.zeros(T, T, T, T, int(torsion_order), dtype=torch.float64), requires_grad=bool(learn))
                        if torsion_order else None)
        atom = torch.tensor([[0.0, math.log(0.5), math.log(0.6 * FREE_ATOM_ALPHA.get(z, 10.0)), 0.0, 0.0]
                             for z, _ in self.atom_types], dtype=torch.float64)
        link = torch.zeros(T, T, 2, dtype=torch.float64)
        link[..., 0] = math.log(0.5)
        self.atom = nn.Parameter(atom, requires_grad=bool(learn))
        self.link = nn.Parameter(link, requires_grad=bool(learn))

    def type_index(self, z: torch.Tensor, degree: torch.Tensor) -> torch.Tensor:
        t = self.lut[z, degree]
        if bool((t < 0).any()):
            bad = sorted({(int(a), int(b)) for a, b in torch.stack((z, degree), 1)[t < 0].tolist()})
            raise ValueError(f"atom types {bad} (Z, degree) are not in the model's {self.atom_types}")
        return t

    def bond_rows(self, ta: torch.Tensor, tb: torch.Tensor) -> torch.Tensor:
        """(Nb, 3) raw rows of the bonds between types ta and tb."""
        return 0.5 * (self.bond[ta, tb] + self.bond[tb, ta])

    def apex_rows(self, t: torch.Tensor) -> torch.Tensor:
        return self.apex[t]

    def torsion_rows(self, ta, tb, tc, td) -> torch.Tensor:
        """(Nt, N) raw K_n rows of the torsions a-b-c-d (the same for d-c-b-a)."""
        return 0.5 * (self.torsion[ta, tb, tc, td] + self.torsion[td, tc, tb, ta])

    def link_rows(self, ta: torch.Tensor, tb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(ln s (Nb,), flow (Nb,) from the first atom to the second) of the bonds ta-tb."""
        return 0.5 * (self.link[ta, tb, 0] + self.link[tb, ta, 0]), 0.5 * (self.link[ta, tb, 1] - self.link[tb, ta, 1])
