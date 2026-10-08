"""Torsions and the bonded coupling family, with NN-emitted parameters.

The class-II terms of the Q-Force bonded-only 1-4 model (Abdullah et al., arXiv:2504.14398,
eqs. 3-10), restated so they are safe to hand to a network:

=====================  ==========================================  ========================
term                   energy (per instance)                       instance
=====================  ==========================================  ========================
torsion                sum_n K_n (1 + cos n phi)                   a-b-c-d
bond-bond              K y_1 y_2                                   each angle's two legs
bond-angle             K_l y_l dc            (l = both legs)       each angle
angle-angle            K dc_1 dc_2                                 angle pairs sharing apex+leg
torsion-bond           sum_n K_{b,n} y_b (1 + cos n phi)            ab, bc, cd of a torsion
torsion-angle          sum_n K_{t,n} dc_t (1 + cos n phi)           abc, bcd of a torsion
torsion-angle-angle    K dc_abc dc_bcd cos phi                      a torsion
=====================  ==========================================  ========================

with ``y = 1 - exp(-beta (r - r_eq))`` the Morse coordinate of the bond's own Morse term and
``dc = cos theta - cos theta_eq`` the angle's own cosine coordinate. Three choices:

* **Bounded coordinates.** ``y ~ beta dr`` near the minimum, so the constants map onto the
  paper's, but a stretched bond saturates (``y -> 1``) instead of driving a bilinear term to
  minus infinity. pyCMM floors such terms with a hard ``minv``; that is not smooth and is not
  used here.
* **Positive-definite by construction (bond/angle couplings).** In those coordinates the
  diagonal terms are ``D y^2`` (Morse) and ``(k_theta/2) dc^2`` (angle), so a coupling between
  two of them is emitted as ``K = 2 rho sqrt(a_1 a_2)`` with ``rho = 0.95 tanh(raw)`` and
  ``a`` the two diagonal stiffnesses: every 2x2 block is positive-definite whatever the
  network does. ``rho`` is the dimensionless normalized coupling ``k_12 / sqrt(k_1 k_2)``.
* **No acos.** ``cos n phi = T_n(cos phi)`` (Chebyshev), so every torsion energy is a
  polynomial in ``cos phi``. The series is cosine-only on purpose: the invariants are parity
  even, and a ``sin n phi`` term with an emitted coefficient would break mirror symmetry.

Parameters follow the bonded head's conventions: a learnable per-type table (zero-initialized
-- these terms are corrections on top of the diagonal ones) plus a zero-initialized
feature-dependent deviation (``delta_iso``, what the geometry-independence regularizer reads)
plus a gated, zero-initialized environment deviation. A fresh head is therefore *exactly* the
uncoupled model, and switching a family on warm-starts from it.

Symmetry is exact by **orbit sum**: each MLP is evaluated on both orderings of an instance
(``a-b-c-d`` / ``d-c-b-a``, ``i-apex-k`` / ``k-apex-i``, ``x <-> y``) and the outputs summed with
the matching slot permutation, so no which-end-is-which information is lost to a symmetric
pooling.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ...mlip.heads import mlp, zero_init_readout

__all__ = ["TermParameterHead", "TermParameters", "term_energies", "chebyshev_series"]

RHO_MAX = 0.95


def chebyshev_series(c: torch.Tensor, n_max: int) -> torch.Tensor:
    """``(..., n_max)`` of ``T_n(c) = cos(n phi)``, ``n = 1..n_max``, for ``c = cos phi``."""
    out = [c]
    prev, cur = torch.ones_like(c), c
    for _ in range(n_max - 1):
        prev, cur = cur, 2.0 * c * cur - prev
        out.append(cur)
    return torch.stack(out, dim=-1)


@dataclass
class TermParameters:
    """One evaluation's torsion/coupling parameters. ``None`` = family absent."""

    torsion_k: torch.Tensor | None = None     # (Nt, N) Ha
    tb_k: torch.Tensor | None = None          # (Nt, 3, N) Ha, slots [ab, bc, cd]
    ta_k: torch.Tensor | None = None          # (Nt, 2, N) Ha, slots [abc, bcd]
    taa_k: torch.Tensor | None = None         # (Nt,) Ha
    bb_rho: torch.Tensor | None = None        # (Na,)
    ba_rho: torch.Tensor | None = None        # (Na, 2), slots [leg (i, apex), leg (apex, k)]
    aa_rho: torch.Tensor | None = None        # (Naa,)
    delta_iso: torch.Tensor | None = None     # flat feature-dependent deviations (theta_0)


class _OrbitMLP(nn.Module):
    """``f(members) + P f(reversed members)``: base + gated environment branch, zero-init."""

    def __init__(self, n_members: int, latent_dim: int, emb_dim: int, hidden: int,
                 depth: int, out_dim: int, perm: list[int]) -> None:
        super().__init__()
        d_in = n_members * (latent_dim + emb_dim)
        self.base = zero_init_readout(mlp(d_in, hidden, depth, out_dim))
        self.delta = zero_init_readout(mlp(d_in, hidden, depth, out_dim))
        self.register_buffer("perm", torch.tensor(perm, dtype=torch.long), persistent=False)

    @staticmethod
    def _x(z, emb, members):
        return torch.cat([z[m] for m in members] + [emb[m] for m in members], dim=-1)

    def _orbit(self, net, z, emb, members, reverse):
        a = net(self._x(z, emb, members))
        b = net(self._x(z, emb, [members[k] for k in reverse]))
        return a + b[:, self.perm]

    def forward(self, z_iso, z_joined, gate, emb, members, reverse):
        raw = self._orbit(self.base, z_iso, emb, members, reverse)
        delta_iso = raw
        if z_joined is not None:
            g = sum(gate[m] for m in members) / len(members)
            raw = raw + g.unsqueeze(-1) * self._orbit(self.delta, z_joined, emb, members, reverse)
        return raw, delta_iso


class TermParameterHead(nn.Module):
    """Torsion (``torsions=True``) and coupling (``couplings=True``) parameters per instance.

    Tables: torsion series per unordered pair of central-atom types ``(n_types, n_types, M)``
    (M = all torsion outputs), angle couplings per apex type ``(n_types, 3)``, angle-angle per
    apex type ``(n_types, 1)``. All zero-initialized.
    """

    def __init__(
        self,
        latent_dim: int,
        n_types: int,
        *,
        torsions: bool = True,
        couplings: bool = False,
        n_max: int = 4,
        hidden: int = 32,
        depth: int = 1,
        emb_dim: int = 8,
        torsion_scale: float = 1.0e-3,
    ) -> None:
        super().__init__()
        self.torsions = bool(torsions)
        self.couplings = bool(couplings)
        self.n_max = int(n_max)
        self.torsion_scale = float(torsion_scale)
        self.type_emb = nn.Embedding(int(n_types), emb_dim)
        N = self.n_max
        if self.torsions:
            # outputs: [K_n (N)] + couplings: [tb (3N), ta (2N), taa (1)]
            n_out = N + ((5 * N + 1) if self.couplings else 0)
            perm = list(range(N))
            if self.couplings:
                tb = [N + s * N + n for s in (2, 1, 0) for n in range(N)]     # ab <-> cd
                ta = [4 * N + s * N + n for s in (1, 0) for n in range(N)]    # abc <-> bcd
                perm += tb + ta + [6 * N]
            self.torsion_mlp = _OrbitMLP(4, latent_dim, emb_dim, hidden, depth, n_out, perm)
            self.torsion_table = nn.Parameter(torch.zeros(int(n_types), int(n_types), n_out))
        if self.couplings:
            # per angle: [bb, ba_leg0, ba_leg1]; reversal swaps the legs
            self.angle_mlp = _OrbitMLP(3, latent_dim, emb_dim, hidden, depth, 3, [0, 2, 1])
            self.angle_table = nn.Parameter(torch.zeros(int(n_types), 3))
            # per angle pair [apex, s, x, y]: x <-> y
            self.pair_mlp = _OrbitMLP(4, latent_dim, emb_dim, hidden, depth, 1, [0])
            self.pair_table = nn.Parameter(torch.zeros(int(n_types), 1))

    @property
    def active(self) -> bool:
        return self.torsions or self.couplings

    def forward(self, z_iso, z_joined, gate, type_idx, topo, z_shift=None) -> TermParameters:
        if z_shift is not None and z_joined is not None:
            z_iso = z_iso + z_shift
        emb = self.type_emb(type_idx)
        out = TermParameters()
        deltas = []
        N = self.n_max
        if self.torsions and topo.n_torsions:
            a, b, c, d = (topo.torsion_index[r] for r in range(4))
            raw, d_iso = self.torsion_mlp(z_iso, z_joined, gate, emb, [a, b, c, d], [3, 2, 1, 0])
            tb_, tc_ = type_idx[b], type_idx[c]
            table = 0.5 * (self.torsion_table[tb_, tc_] + self.torsion_table[tc_, tb_])
            if self.couplings:  # the table must respect the reversal too
                table = 0.5 * (table + table[:, self.torsion_mlp.perm])
            k = self.torsion_scale * (table + raw)
            out.torsion_k = k[:, :N]
            if self.couplings:
                nt = k.shape[0]
                out.tb_k = k[:, N:4 * N].reshape(nt, 3, N)
                out.ta_k = k[:, 4 * N:6 * N].reshape(nt, 2, N)
                out.taa_k = k[:, 6 * N]
            deltas.append(d_iso.reshape(-1))
        if self.couplings and topo.angle_index.shape[1]:
            i, apex, kk = (topo.angle_index[r] for r in range(3))
            raw, d_iso = self.angle_mlp(z_iso, z_joined, gate, emb, [i, apex, kk], [2, 1, 0])
            t = self.angle_table[type_idx[apex]]
            t = torch.stack((t[:, 0], 0.5 * (t[:, 1] + t[:, 2]), 0.5 * (t[:, 1] + t[:, 2])), -1)
            rho = RHO_MAX * torch.tanh(t + raw)
            out.bb_rho, out.ba_rho = rho[:, 0], rho[:, 1:]
            deltas.append(d_iso.reshape(-1))
        if self.couplings and topo.n_angle_pairs:
            pa, ps, px, py = (topo.angle_pair_index[r] for r in range(4))
            raw, d_iso = self.pair_mlp(z_iso, z_joined, gate, emb, [pa, ps, px, py], [0, 1, 3, 2])
            out.aa_rho = RHO_MAX * torch.tanh(self.pair_table[type_idx[pa]] + raw).squeeze(-1)
            deltas.append(d_iso.reshape(-1))
        if deltas:
            out.delta_iso = torch.cat(deltas)
        return out


def term_energies(
    positions_ang: torch.Tensor,
    topo,                     # BondedTopology
    bp,                       # BondedParameters (same evaluation)
    tp: TermParameters,
    geometry: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """``[(energies, owning fragment)]`` per active family, co-membership weighted, Hartree."""
    r, cos_t = topo.geometry(positions_ang) if geometry is None else geometry
    out = []
    need_y = tp.tb_k is not None or tp.bb_rho is not None
    if need_y:
        beta = torch.sqrt(bp.k / (2.0 * bp.d))
        y = 1.0 - torch.exp(-beta * (r - bp.r_eq))                    # (Nb,)
    dc = cos_t - bp.cos_theta_eq                                      # (Na,)

    if tp.torsion_k is not None and topo.n_torsions:
        cphi = topo.torsion_cos(positions_ang)
        f = 1.0 + chebyshev_series(cphi, tp.torsion_k.shape[-1])      # (Nt, N)
        e = (tp.torsion_k * f).sum(-1)
        if tp.tb_k is not None:
            y_t = y[topo.torsion_bonds].t()                            # (Nt, 3)
            e = e + (tp.tb_k * y_t.unsqueeze(-1) * f.unsqueeze(1)).sum((-1, -2))
            dc_t = dc[topo.torsion_angles].t()                         # (Nt, 2)
            e = e + (tp.ta_k * dc_t.unsqueeze(-1) * f.unsqueeze(1)).sum((-1, -2))
            e = e + tp.taa_k * dc_t[:, 0] * dc_t[:, 1] * cphi
        out.append((topo.torsion_weight * e, topo.torsion_frag))

    if tp.bb_rho is not None and topo.angle_index.shape[1]:
        l0, l1 = topo.angle_bonds[0], topo.angle_bonds[1]
        d0, d1 = bp.d[l0], bp.d[l1]
        a_ang = 0.5 * bp.k_theta
        e = 2.0 * tp.bb_rho * torch.sqrt(d0 * d1) * y[l0] * y[l1]
        e = e + 2.0 * tp.ba_rho[:, 0] * torch.sqrt(d0 * a_ang) * y[l0] * dc
        e = e + 2.0 * tp.ba_rho[:, 1] * torch.sqrt(d1 * a_ang) * y[l1] * dc
        out.append((topo.angle_weight * e, topo.angle_frag))

    if tp.aa_rho is not None and topo.n_angle_pairs:
        a0, a1 = topo.angle_pair_angles[0], topo.angle_pair_angles[1]
        k0, k1 = 0.5 * bp.k_theta[a0], 0.5 * bp.k_theta[a1]
        e = 2.0 * tp.aa_rho * torch.sqrt(k0 * k1) * dc[a0] * dc[a1]
        out.append((topo.angle_pair_weight * e, topo.angle_pair_frag))
    return out
