"""The fixed physical parameters of the monatomic ions (``scripts/ion_params.py`` -> ``params/*.json``) as tensors.

Every per-ion quantity is indexed by a species index ``s`` (the order of ``IonTable.symbols``):

    Z (S,)                      point nuclear charge
    shell_n, shell_b (S, M)     electrons and Slater 1s exponent of each permanent shell (zero-padded, n = 0)
    alpha, C (S,)               free-ion dipole and quadrupole polarizabilities (Theta = C g, pol_fit convention)
    b_dip, w_dip (S,), (S, 4)   induced-dipole parent: exponent and weights over r^0..r^3
    b_quad, w_quad              induced-quadrupole parent
    front_b[role], front_w[role] (S,), (S, 4)   frontier densities, role in ('donor', 'acceptor')
    I_donor (S,)                electron-removal energy of the free ion as a CT donor (Hartree)
    A_acceptor (S,)             electron-attachment energy as an acceptor (Hartree; NaN where unbound, e.g. Cl-)

Units: atomic.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

ATOMIC_NUMBER = {"Li": 3, "F": 9, "Na": 11, "Cl": 17, "K": 19, "Br": 35}
N_SHAPE = 4                     # r^0..r^3, what rsfff.ff.multipole's shaped kernels carry
ROLES = ("donor", "acceptor")
DEFAULT_PARAMS = Path(__file__).resolve().parents[3] / "params" / "ions_wb97mv_tzvpd.json"


def _onehot(k: int) -> list[float]:
    w = [0.0] * N_SHAPE
    w[int(k)] = 1.0
    return w


def _pad(w):
    w = list(w)[:N_SHAPE]
    return w + [0.0] * (N_SHAPE - len(w))


@dataclass
class IonTable:
    symbols: tuple
    numbers: torch.Tensor
    charge: torch.Tensor
    Z: torch.Tensor
    shell_n: torch.Tensor
    shell_b: torch.Tensor
    alpha: torch.Tensor
    C: torch.Tensor
    b_dip: torch.Tensor
    w_dip: torch.Tensor
    b_quad: torch.Tensor
    w_quad: torch.Tensor
    front_b: dict
    front_w: dict
    I_donor: torch.Tensor
    A_acceptor: torch.Tensor

    @classmethod
    def load(cls, path: str | Path = DEFAULT_PARAMS, symbols=None, dtype=torch.float64) -> "IonTable":
        ions = json.load(open(path))["ions"]
        symbols = tuple(symbols or ions)
        M = max(len(ions[s]["shells"]) for s in symbols)
        t = lambda x: torch.tensor(x, dtype=dtype)
        sh_n = [[sh["n"] for sh in ions[s]["shells"]] + [0.0] * (M - len(ions[s]["shells"])) for s in symbols]
        sh_b = [[sh["b"] for sh in ions[s]["shells"]] + [1.0] * (M - len(ions[s]["shells"])) for s in symbols]
        front_b = {r: t([ions[s]["frontier"][r]["b"] for s in symbols]) for r in ROLES}
        front_w = {r: t([_onehot(ions[s]["frontier"][r]["k"]) for s in symbols]) for r in ROLES}
        return cls(
            symbols=symbols, numbers=torch.tensor([ATOMIC_NUMBER[s] for s in symbols]),
            charge=t([ions[s]["charge"] for s in symbols]), Z=t([ions[s]["Z"] for s in symbols]),
            shell_n=t(sh_n), shell_b=t(sh_b), alpha=t([ions[s]["alpha"] for s in symbols]),
            C=t([ions[s]["C"] or 0.0 for s in symbols]),
            b_dip=t([ions[s]["b_ind"] for s in symbols]), w_dip=t([_pad(ions[s]["w_dip"]) for s in symbols]),
            b_quad=t([ions[s]["b_quad"] for s in symbols]), w_quad=t([_pad(ions[s]["w_quad"]) for s in symbols]),
            front_b=front_b, front_w=front_w, I_donor=t([ions[s]["I_donor"] for s in symbols]),
            A_acceptor=t([float("nan") if ions[s]["A_acceptor"] is None else ions[s]["A_acceptor"] for s in symbols]),
        )

    def species_index(self, atomic_numbers: torch.Tensor) -> torch.Tensor:
        lut = torch.full((int(self.numbers.max()) + 1,), -1, dtype=torch.long)
        lut[self.numbers] = torch.arange(len(self.symbols))
        s = lut[atomic_numbers.long()]
        if (s < 0).any():
            raise ValueError(f"atomic numbers outside the ion table {self.symbols}")
        return s
