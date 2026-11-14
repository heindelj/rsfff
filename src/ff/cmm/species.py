"""Species conditioning for the shared environment (m35): who the centre atom is, as a vector the trunks can read.

The shared environment's channels are built from pair overlaps, which are symmetric (O_ij seen from i is O_ij seen
from j), so without this a Na and a Cl in contact see the same two-body inputs and get the same log-factors. The
conditioning vector c_s enters the encoders (the latent trunk and the two-body branches, FiLM by default); the
readouts stay shared by every species, so a new ion needs only its own c.

Two sources for c:

    "descriptor"  c_s = MLP(d_s), d_s the free ion's physical descriptors (below), read off the ion table: an ion the
                  network never saw still gets a sensible c from its own free-ion properties (the transfer route)
    "table"       c_s a learned vector per species (nn.Embedding): the per-species baseline, no transfer

Descriptors are fixed affine transforms of physical quantities, with constants chosen once for the alkali halides
(Li+ ... Cs+, F- ... I-), never standardised over the loaded table: a trained model must see the same d_s for an ion
whatever other ions the table carries. To add one (fragment charge and multiplicity, for instance), append it to
DESCRIPTORS -- a checkpoint trained without it no longer loads, which is intended.
"""
from __future__ import annotations

import math

import torch
from torch import nn

__all__ = ["DESCRIPTORS", "descriptor_names", "species_descriptors", "SpeciesEmbedding"]


def _outer_b(tab):
    """(S,) exponent of the most diffuse occupied shell (bohr^-1)."""
    b = torch.where(tab.shell_n > 1e-8, tab.shell_b, torch.full_like(tab.shell_b, math.inf))
    return b.min(1).values


def _fill(x, v):
    x = torch.as_tensor(x, dtype=torch.float64)
    return torch.where(torch.isfinite(x), x, torch.full_like(x, v))


#: (name, function of the IonTable -> (S,), centre, scale): d = (x - centre) / scale. Atomic units throughout.
DESCRIPTORS = (
    ("charge", lambda t: t.charge, 0.0, 1.0),
    ("ln_alpha", lambda t: torch.log(t.alpha), 1.5, 2.0),            # dipole polarizability: Li+ 0.2 ... I- 70
    ("ln_C", lambda t: torch.log(t.C), 2.5, 3.0),                    # quadrupole polarizability
    ("ln_b_outer", lambda t: torch.log(_outer_b(t)), 0.6, 0.5),      # the outer shell's exponent (size / softness)
    ("ln_b_donor", lambda t: torch.log(t.front_b["donor"]), 0.6, 0.5),
    ("ln_b_acceptor", lambda t: torch.log(t.front_b["acceptor"]), 0.3, 0.5),
    ("I_donor", lambda t: _fill(t.I_donor, 0.0), 0.8, 1.0),          # ionisation energy of the donor level (Eh)
    ("A_acceptor", lambda t: _fill(t.A_acceptor, 0.0), 0.0, 0.2),    # electron affinity (an anion: none -> 0)
    ("breath_a0", lambda t: t.a0, 1.0, 2.0),                         # breathing polarizability (0 without one)
)


def descriptor_names():
    return [d[0] for d in DESCRIPTORS]


def species_descriptors(tab) -> torch.Tensor:
    """(S, n_desc) the fixed-scaled descriptors of every species of the ion table."""
    cols = [(_fill(f(tab), c) - c) / s for _, f, c, s in DESCRIPTORS]
    return torch.stack([torch.as_tensor(x, dtype=torch.float64).reshape(-1) for x in cols], 1)


class SpeciesEmbedding(nn.Module):
    """``forward(species) -> (N, dim)``: the conditioning vector of each atom's species."""

    def __init__(self, tab, dim=8, kind="descriptor", hidden=16):
        super().__init__()
        if kind not in ("descriptor", "table"):
            raise ValueError(f"species embedding {kind!r}: descriptor or table")
        self.kind, self.dim = kind, int(dim)
        S = len(tab.symbols)
        if kind == "descriptor":
            self.register_buffer("desc", species_descriptors(tab), persistent=False)
            self.net = nn.Sequential(nn.Linear(self.desc.shape[1], hidden), nn.SiLU(), nn.Linear(hidden, self.dim))
        else:
            self.table = nn.Embedding(S, self.dim)

    def all_species(self) -> torch.Tensor:
        """(S, dim) c for every species of the table."""
        return self.net(self.desc) if self.kind == "descriptor" else self.table.weight

    def forward(self, species: torch.Tensor) -> torch.Tensor:
        return self.all_species()[species]
