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
(:func:`rsfff.ff.cmm.kernels._drive_from_permanent`). The overlap projections are the overlap integral of two
shaped densities and its first two radial derivatives (k-space quadrature, :func:`shaped_overlap_derivs`).

Nothing here depends on the species of the neighbours except through their density parameters, and every block is
exactly zero for an isolated atom. The invariants handed to a network are, per probe,

    [V^(0)_pen,  |V^(1)|,  |V^(2)|,  S^(0),  |S^(1)|,  |S^(2)|,  V^(1).S^(1)]

(the last one: whether the field and the overlap push are aligned). With ``inv_form="sq"`` (the default) the four
magnitudes are replaced by the squared norms -- the self-contractions |V^(1)|^2, |V^(2)|^2, |S^(1)|^2, |S^(2)|^2 --
which are smooth where the vector / tensor passes through zero: a centrosymmetric lattice site has V^(1) = S^(1) = 0
and a cubic one also V^(2) = S^(2) = 0, and the magnitude has a cone there (|x| is not differentiable at x = 0), so
the gradient of anything built on it jumps across the site and the forces are discontinuous. ``"norm"`` keeps the
magnitudes (checkpoints trained before 2026-10-23). The equivariant blocks are kept on the result
for equivariant heads (anisotropic alpha, Pauli dipoles / quadrupoles). Units: atomic (bohr, e, Hartree); positions
are taken in Angstrom like the rest of rsfff.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from . import kernels as K
from .params import IonTable
from ..units import BOHR_ANG

__all__ = ["DensityEnvironment", "EnvFeatures", "shaped_overlap_derivs", "PROBE_NAMES", "INV_FORMS",
           "invariant_names", "invariants", "env_invariants", "inv_form_of"]

PROBE_NAMES = ("shell", "dip", "quad", "donor", "acceptor", "breath")
#: invariant names per probe, by form: "norm" = the magnitudes (old), "sq" = the squared norms (smooth at zero)
INV_FORMS = {"norm": ("V0_pen", "|V1|", "|V2|", "S0", "|S1|", "|S2|", "V1.S1"),
             "sq": ("V0_pen", "|V1|^2", "|V2|^2", "S0", "|S1|^2", "|S2|^2", "V1.S1")}
INVARIANT_NAMES = INV_FORMS["norm"]


def invariant_names(form="sq"):
    if form not in INV_FORMS:
        raise ValueError(f"inv_form {form!r}: one of {sorted(INV_FORMS)}")
    return INV_FORMS[form]


def inv_form_of(feature_names, default="sq"):
    """The invariant form a list of feature names was built with ("sq" if any name is a squared norm)."""
    if not feature_names:
        return default
    return "sq" if any(str(n).endswith("^2") for n in feature_names) else "norm"


def _mag(x, dims, form):
    """|x| ("norm") or |x|^2 ("sq") over the trailing ``dims`` dimensions."""
    x = x.flatten(-dims)
    return (x * x).sum(-1) if form == "sq" else x.norm(dim=-1)


def invariants(V0, V1, V2, S0, S1, S2, form="sq"):
    """(..., 7) invariants of one probe from its blocks V0 (...), V1 (..., 3), V2 (..., 5), S0 (...), S1 (..., 3),
    S2 (..., 3, 3), in the order of :func:`invariant_names`."""
    invariant_names(form)
    return torch.stack([V0, _mag(V1, 1, form), _mag(V2, 1, form), S0, _mag(S1, 1, form), _mag(S2, 2, form),
                        (V1 * S1).sum(-1)], -1)


def env_invariants(env: dict, probes, form="sq"):
    """(N, len(probes) * 7) features, probe-major, from the equivariant blocks ``env["<probe>:<V0|...|S2>"]`` a
    cache carries (lets a cache built under one form serve the other)."""
    return torch.cat([invariants(*(env[f"{p}:{k}"] for k in ("V0", "V1", "V2", "S0", "S1", "S2")), form=form)
                      for p in probes], -1)


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

    def __init__(self, tab: IonTable, probes=PROBE_NAMES, coulomb=True, overlap=True, cutoff=None, inv_form="sq"):
        self.tab = tab
        self.probes = tuple(probes)
        self.coulomb, self.overlap, self.cutoff = coulomb, overlap, cutoff
        self.inv_form = inv_form
        names = invariant_names(inv_form)
        self.n_feat = len(self.probes) * len(names)
        self.feature_names = [f"{p}:{q}" for p in self.probes for q in names]
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
            inv.append(invariants(blk["V0"], blk["V1"], blk["V2"], blk["S0"], blk["S1"], blk["S2"], self.inv_form))
            blocks[name] = blk
        return EnvFeatures(torch.cat(inv, 1), blocks)
