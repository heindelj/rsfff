"""Permanent electrostatics of the frozen ion densities (point nucleus + Slater 1s shells), closed form.

    E = sum_{i<j} [ Z_i Z_j - Z_i sum_u n_u (1 - f(b_u r)) - Z_j sum_t n_t (1 - f(b_t r))
                    + sum_{t,u} n_t n_u (1 - f(b_t, b_u; r)) ] / r

with the one-centre and unequal-exponent two-centre Slater penetration complements (:mod:`rsfff.ff.multipole`).
This is what Stage A validated against ALMO cls_elec (pair-fit shells: 0.1-0.2 kJ/mol per ion); it carries the
long-range Coulomb of the net charges, so it is the term that makes total energies and forces."""
from __future__ import annotations

import torch

from ..ionct import kernels as K
from ..ionct.params import IonTable
from ..multipole import slater_one_center_damp, slater_two_center_damp_unequal

__all__ = ["permanent_electrostatics"]


def permanent_electrostatics(pos_ang, species, batch_idx, tab: IonTable, pairs=None, pair_dr=None, pbc=None):
    """Per-frame energies (B,) Hartree; differentiable in the positions (Angstrom). ``pairs`` (2, P) with their
    separations ``pair_dr`` (P, 3) Angstrom replace the all-pairs list (lattice images included); with ``pbc`` (the
    periodic dict of :func:`rsfff.ff.complete.batched_precompute`) the net-charge Coulomb q_i q_j / r of each pair is
    erfc-screened and the reciprocal-space energy is added (the Ewald sum of the point charges)."""
    X = pos_ang / K.BOHR_ANG
    B = int(batch_idx.max()) + 1 if batch_idx.numel() else 0
    if pairs is None:
        pairs = K.all_pairs(batch_idx)
        pair_dr = pos_ang[pairs[1]] - pos_ang[pairs[0]]
    i, j = pairs
    r = pair_dr.norm(dim=-1) / K.BOHR_ANG
    si, sj = species[i], species[j]
    ni, bi, nj, bj = tab.shell_n[si], tab.shell_b[si], tab.shell_n[sj], tab.shell_b[sj]      # (P, M)
    P_, M = ni.shape
    # every shell of either side against the other nucleus in one call, every shell pair in one call
    f1 = slater_one_center_damp((torch.cat([bj, bi], 1) * r[:, None]).reshape(-1), 0)[0].reshape(P_, 2 * M)
    e = tab.Z[si] * tab.Z[sj] - tab.Z[si] * (nj * (1 - f1[:, :M])).sum(1) - tab.Z[sj] * (ni * (1 - f1[:, M:])).sum(1)
    f2 = slater_two_center_damp_unequal(bi[:, :, None].expand(-1, -1, M).reshape(-1), bj[:, None, :].expand(-1, M, -1).reshape(-1),
                                        r[:, None].expand(-1, M * M).reshape(-1), 0)[0].reshape(P_, M, M)
    e = e + (ni[:, :, None] * nj[:, None, :] * (1 - f2)).sum((1, 2))
    e = e / r
    if pbc is not None:
        e = e - tab.charge[si] * tab.charge[sj] * (1 - pbc["g0_pairs"]) / r
        return X.new_zeros(B).index_add(0, batch_idx[i], e) + pbc["e_elst_recip"]
    return X.new_zeros(B).index_add(0, batch_idx[i], e)
