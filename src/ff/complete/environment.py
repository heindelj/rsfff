"""C0 environment features: the other fragments' permanent densities projected onto an atom's own densities.

For atom i and each of its probe densities g (its outer permanent shell, its induced dipole and quadrupole parents,
its donor and acceptor frontier densities -- whatever the parameter table defines for its species), two kinds of
projection of the environment, summed over the other atoms j:

    Coulomb   V[g]^(0) = <g | phi_j>  (penetration part: the point-charge q_j / r is subtracted, so it is gauge-free
                         and short-ranged),  V[g]^(1) = <g | E_j> (3),  V[g]^(2) = <g | grad E_j> (5, traceless)
    overlap   S[g]^(0) = <g | rho_j>,  S[g]^(1) = <grad g | rho_j> (3),  S[g]^(2) = <grad grad g | rho_j> (5)

where phi_j, E_j, rho_j are the potential, field and electron density of j's *frozen* density (point nucleus Z_j plus
its Slater shells). The Coulomb projections are row 0 of the damped interaction tensor between the probe (as an
induced parent) and j's permanent components -- the same operator that drives the induction
(:func:`rsfff.ff.ionct.kernels._drive_from_permanent`). The overlap projections are the overlap integral of two
shaped densities and its first two radial derivatives (k-space quadrature, :func:`shaped_overlap_derivs`).

Nothing here depends on the species of the neighbours except through their density parameters, and every block is
exactly zero for an isolated atom. The invariants handed to a network are, per probe,

    [V^(0)_pen,  |V^(1)|,  |V^(2)|,  S^(0),  |S^(1)|,  |S^(2)|,  V^(1).S^(1)]

(the last one: whether the field and the overlap push are aligned). The equivariant blocks are kept on the result
for equivariant heads (anisotropic alpha, Pauli dipoles / quadrupoles). Units: atomic (bohr, e, Hartree); positions
are taken in Angstrom like the rest of rsfff.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..ionct import kernels as K
from ..ionct.params import IonTable
from ..units import BOHR_ANG

__all__ = ["DensityEnvironment", "EnvFeatures", "shaped_overlap_derivs", "PROBE_NAMES"]

PROBE_NAMES = ("shell", "dip", "quad", "donor", "acceptor", "breath")
INVARIANT_NAMES = ("V0_pen", "|V1|", "|V2|", "S0", "|S1|", "|S2|", "V1.S1")


@dataclass
class EnvFeatures:
    inv_feats: torch.Tensor          # (N, F)
    blocks: dict                     # probe -> {"V0": (N,), "V1": (N, 3), "V2": (N, 5), "S0": (N,), "S1": (N, 3), "S2": (N, 3, 3)}


def _j0_derivs(x):
    """Spherical Bessel j0 = sin x / x and its first two derivatives, safe at x -> 0."""
    small = x < 1e-3
    xs = torch.where(small, torch.ones_like(x), x)
    s, c = torch.sin(xs), torch.cos(xs)
    j0 = torch.where(small, 1 - x * x / 6, s / xs)
    j1 = torch.where(small, -x / 3, (xs * c - s) / xs ** 2)                             # d j0 / dx
    j2 = torch.where(small, -1 / 3 + x * x / 10, ((2 - xs * xs) * s - 2 * xs * c) / xs ** 3)   # d^2 j0 / dx^2
    return j0, j1, j2


def shaped_overlap_derivs(bD, wD, bA, wA, R):
    """Overlap S(R) = int rho_D(r) rho_A(r - R) d^3r of two normalized shaped densities (bohr^-3) and its first two
    derivatives in R (bohr), each (C,): S = (1 / 2 pi^2) int fD fA j0(kR) k^2 dk, S' with k j0', S'' with k^2 j0''."""
    k, wk = K._ovl_grid(R.dtype, R.device)
    f = K.shaped_density_ft(bD, wD, k) * K.shaped_density_ft(bA, wA, k) * (k * k)[None, :] * wk[None, :]
    j0, j1, j2 = _j0_derivs(k[None, :] * R[:, None])
    c = 1.0 / (2 * torch.pi ** 2)
    return (f * j0).sum(1) * c, (f * j1 * k[None, :]).sum(1) * c, (f * j2 * (k * k)[None, :]).sum(1) * c


class DensityEnvironment:
    """Callable with the featurizer signature ``(positions, atomic_numbers, batch_idx) -> EnvFeatures``.

    ``probes``: which of the table's densities to project onto (:data:`PROBE_NAMES`; ``breath`` is the l = 0
    response shape, whose penetration projection V^(0) drives the breathing term of the complete model). ``overlap`` / ``coulomb`` switch
    the two kinds on or off; ``cutoff`` (Angstrom) restricts the overlap sums (the Coulomb projections are summed over
    every other atom of the frame -- they are fields)."""

    def __init__(self, tab: IonTable, probes=PROBE_NAMES, coulomb=True, overlap=True, cutoff=None):
        self.tab = tab
        self.probes = tuple(probes)
        self.coulomb, self.overlap, self.cutoff = coulomb, overlap, cutoff
        self.n_feat = len(self.probes) * len(INVARIANT_NAMES)
        self.feature_names = [f"{p}:{q}" for p in self.probes for q in INVARIANT_NAMES]
        S = len(tab.symbols)
        # the outermost (smallest-exponent) permanent shell with electrons, per species
        b_outer = torch.where(tab.shell_n > 0, tab.shell_b, torch.full_like(tab.shell_b, float("inf"))).min(1).values
        one = torch.zeros(S, 4, dtype=tab.shell_b.dtype)
        one[:, 0] = 1.0
        self._probe = {"shell": (b_outer, one), "dip": (tab.b_dip, tab.w_dip), "quad": (tab.b_quad, tab.w_quad),
                       "donor": (tab.front_b["donor"], tab.front_w["donor"]),
                       "acceptor": (tab.front_b["acceptor"], tab.front_w["acceptor"]),
                       "breath": (tab.b_breath, tab.w_breath)}
        # a species without an l = 0 response shape (a0 = 0, zero weights) gets zero projections for that probe

    def _species(self, atomic_numbers):
        return self.tab.species_index(torch.as_tensor(atomic_numbers))

    def __call__(self, positions, atomic_numbers, batch_idx) -> EnvFeatures:
        pos = torch.as_tensor(positions, dtype=torch.float64)
        X = pos / BOHR_ANG
        N = X.shape[0]
        b_idx = torch.as_tensor(batch_idx)
        s = self._species(atomic_numbers)
        tab = self.tab
        pairs = K.all_pairs(b_idx)
        dp = torch.cat([pairs, pairs.flip(0)], dim=1)          # directed (i <- j)
        i, j = dp
        dr = X[j] - X[i]
        r = dr.norm(dim=-1)
        rhat = dr / r[:, None]
        P = K._poly_map(X.dtype, X.device)
        blocks, inv = {}, []
        qj = tab.charge[s[j]]
        for name in self.probes:
            bp, wp = (t[s] for t in self._probe[name])
            blk = {}
            cols = []
            if self.coulomb and dp.shape[1]:
                D = K._drive_from_permanent(X, s, dp, tab, bp, wp)        # (P, 10): [phi, -E (3), grad (6 weighted)]
                Dp = D @ P                                                 # (P, 8): [-E (3), G (5 spherical)]
                v0 = D[:, 0] - qj / r                                      # penetration part of the potential
                v1 = -Dp[:, :3]
                v2 = Dp[:, 3:]
                blk["V0"] = X.new_zeros(N).index_add(0, i, v0)
                blk["V1"] = X.new_zeros(N, 3).index_add(0, i, v1)
                blk["V2"] = X.new_zeros(N, 5).index_add(0, i, v2)
            else:
                blk["V0"], blk["V1"], blk["V2"] = X.new_zeros(N), X.new_zeros(N, 3), X.new_zeros(N, 5)
            if self.overlap and dp.shape[1]:
                keep = torch.ones_like(r, dtype=torch.bool) if self.cutoff is None else (r * BOHR_ANG < self.cutoff)
                S0 = X.new_zeros(r.shape[0])
                S1 = X.new_zeros(r.shape[0])
                S2 = X.new_zeros(r.shape[0])
                one = K._onehot0(r.shape[0], r)
                for t in range(tab.shell_n.shape[1]):
                    nt = tab.shell_n[s[j], t]
                    if not bool((nt > 0).any()):
                        continue
                    a, da, d2a = shaped_overlap_derivs(bp[i], wp[i], tab.shell_b[s[j], t], one, r)
                    S0 = S0 + nt * a
                    S1 = S1 + nt * da
                    S2 = S2 + nt * d2a
                S0, S1, S2 = S0 * keep, S1 * keep, S2 * keep
                # <grad g | rho_j> = -dS/dR rhat ... the sign convention: S1 points from i toward the overlap (toward j)
                v1s = -S1[:, None] * rhat
                outer = rhat[:, :, None] * rhat[:, None, :] - torch.eye(3, dtype=X.dtype)[None] / 3
                t2 = (S2 - S1 / r)[:, None, None] * outer
                blk["S0"] = X.new_zeros(N).index_add(0, i, S0)
                blk["S1"] = X.new_zeros(N, 3).index_add(0, i, v1s)
                blk["S2"] = X.new_zeros(N, 3, 3).index_add(0, i, t2)
            else:
                blk["S0"], blk["S1"], blk["S2"] = X.new_zeros(N), X.new_zeros(N, 3), X.new_zeros(N, 3, 3)
            if name == "breath":            # species without the l = 0 shape: no such projections
                has = (tab.a0[s] > 0).to(X.dtype)
                blk = {k: v * has.reshape(-1, *([1] * (v.dim() - 1))) for k, v in blk.items()}
            cols = [blk["V0"], blk["V1"].norm(dim=-1), blk["V2"].norm(dim=-1), blk["S0"], blk["S1"].norm(dim=-1),
                    blk["S2"].flatten(1).norm(dim=-1), (blk["V1"] * blk["S1"]).sum(-1)]
            inv.append(torch.stack(cols, 1))
            blocks[name] = blk
        return EnvFeatures(torch.cat(inv, 1), blocks)
