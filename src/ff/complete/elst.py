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


def permanent_electrostatics(pos_ang, species, batch_idx, tab: IonTable):
    """Per-frame energies (B,) Hartree; differentiable in the positions (Angstrom)."""
    X = pos_ang / K.BOHR_ANG
    B = int(batch_idx.max()) + 1 if batch_idx.numel() else 0
    pairs = K.all_pairs(batch_idx)
    i, j = pairs
    r = (X[j] - X[i]).norm(dim=-1)
    si, sj = species[i], species[j]
    e = tab.Z[si] * tab.Z[sj]
    M = tab.shell_n.shape[1]
    for t in range(M):
        nt, bt = tab.shell_n[sj, t], tab.shell_b[sj, t]
        e = e - tab.Z[si] * nt * (1 - slater_one_center_damp(bt * r, 0)[0])
        nu, bu = tab.shell_n[si, t], tab.shell_b[si, t]
        e = e - tab.Z[sj] * nu * (1 - slater_one_center_damp(bu * r, 0)[0])
    for t in range(M):
        for u in range(M):
            nt, bt = tab.shell_n[si, t], tab.shell_b[si, t]
            nu, bu = tab.shell_n[sj, u], tab.shell_b[sj, u]
            e = e + nt * nu * (1 - slater_two_center_damp_unequal(bt, bu, r, 0)[0])
    e = e / r
    return X.new_zeros(B).index_add(0, batch_idx[i], e)
