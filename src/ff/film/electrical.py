"""ECE electrical heads: permanent multipoles and polarizability from the bond graph, with exact
dissociation limits on a fixed diabatic state.

The latent heads (:class:`~rsfff.ff.film.permanent.PermanentMultipoleHeads`,
:class:`~rsfff.ff.film.heads.FilmResponseHeads`) read atom-centred SOAP latents and have no
notion of a bond. These heads build every electrical parameter of a molecule on its covalent
bonds instead, from the edge cluster expansion (:mod:`rsfff.ff.film.ece`), as

    parameter = (free-atom value on the diabat) + (bond terms, each times an envelope e_b(r))

with ``e_b -> 0`` as bond ``b`` dissociates. On a fixed diabatic topology a bond can only "break"
by stretching, so the limits hold by construction, for any weights:

=====================  ==========================================================  ==============
quantity               form                                                        dissociated
=====================  ==========================================================  ==============
charge ``q_i``         ``q_inf,i + sum_b sigma_ib e_b g_b`` (bond charge flow)     ``q_inf,i``
dipole ``mu_i``        ``sum_{edges i->j} e_b v_e`` (axial + off-axis, bond frame)  0
quadrupole ``Th_i``    ``sum_{edges i->j} e_b T_e`` (traceless)                     0
polarizability         ``alpha0,i exp(sum e_b s_e) expm(-sum e_b K_e)``            ``alpha0,i``
bond compliance        ``e_b softplus(c_t + h_b)`` on bond channels, 0 elsewhere    0 (no flow)
=====================  ==========================================================  ==============

* ``q_inf`` is the diabatic asymptotic charge of each atom (``Batch.diabat_charge``,
  :mod:`rsfff.ff.film.diabats`): the fragment's net charge is conserved exactly by the flows
  (no projection), and every atom ends on its diabatic charge.
* **Per-bond dissociation channels.** Each bond carries the asymptotic flow ``f_b`` of its
  channel (``Batch.diabat_flow``: 0 homolytic, +-1 heterolytic, relative to ``q_inf``). The
  asymptotic charges then follow the geometry,

      q_asym,i = q_inf,i + sum_b sigma_ib t_b f_b,    t_b = p_b prod_{b' in group(b), b' != b} (1 - p_b')

  with the channel progress ``p_b = 1 - exp(-gamma ramp(r - r_ref - onset))`` (exactly 0 within
  the onset, default 0.5 A, of the reference length, 1 dissociated; ``ramp`` is C2) and the group product making transfers from one source
  (a cation centre, a resonance group) exclusive. Charges are ``q_asym + enveloped flows`` and
  ``alpha0`` is the free atom / ion value at ``q_asym`` -- so a proton leaving NH4+ ends as H+
  with ``alpha = 0``, and CH3NH3+ separates into CH3+ and NH3, each exactly its isolated self.
* ``alpha0`` is the free atom / ion polarizability of that charge state
  (:data:`FREE_ATOM_ALPHA`, b3lyp/def2-svpd, linear in the charge between integer states for a
  resonance-averaged ``q_inf``). ``expm(-K)`` with ``K`` traceless has determinant 1, so ``K``
  moves polarizability between directions and ``s`` scales it; the result is positive definite
  whatever the network does.
* Charge flows only along covalent bonds, through their compliances, which close as the bonds
  break -- no fractional charge between separated atoms (the fluctuating-charge dissociation
  failure).
* The envelope is geometric: ``e_b = exp(-gamma_t (r - r_t))`` with ``r_t`` and ``gamma_t`` from the
  bond type's Morse prior (``r_eq`` and ``beta = sqrt(k / 2 D)``) times a learnable per-family
  log-scale -- an electrical bond order, unity at the reference length.
* Everything else is a readout of the per-directed-edge ECE invariants and type embeddings
  (plus, with ``latents``, the endpoint atoms' permanent-family latents), zero-initialized: a fresh
  model has no dipoles or quadrupoles, free-atom polarizabilities, and Pauling-electronegativity
  bond flows (a learnable ``kappa (chi_a - chi_b)`` per type).

Permanent quantities have no environment branch (the strict separation). The response family
(``alpha``, compliance) gets a gated environment factor ``exp(gate * NN(z_joined))``, zero-init,
exactly 1 for an isolated fragment: the environment modifies the physical parameters, it does not
replace them.

Intramolecular electrostatics is absent by design in the monomer models that use these heads
(``nonbonded: exclusions``); the molecular moments and polarizability are the sums of the atomic
ones plus charge flow.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from bb.heads import EdgeReadout, flow_to_charges
from bb.nn import mlp, zero_init_readout

from ..multipole import cartesian_to_spherical_quadrupole
from ..units import BOHR_ANG

__all__ = ["ECEElectricalHeads", "ElectricalOutput", "FREE_ATOM_ALPHA", "PAULING_CHI", "free_atom_alpha"]

#: free atom / ion static dipole polarizabilities (bohr^3), b3lyp/def2-svpd finite field (F = 0.002 au),
#: averaged over the three axes (UKS ground states; the pyscf-all level, so the dissociation limit is
#: the one the data would reach -- def2-svpd underestimates the anions against the CBS values):
#: ``{(Z, charge): alpha}``. Missing states fall back to the nearest available charge.
FREE_ATOM_ALPHA: dict[tuple[int, int], float] = {
    (1, 0): 4.644, (1, 1): 0.0, (6, 0): 11.359, (7, 0): 6.504, (8, 0): 5.593, (9, 0): 3.711, (16, 0): 22.910,
    (6, 1): 6.172, (7, 1): 3.873, (8, 1): 2.588,
    (6, -1): 21.429, (8, -1): 16.863, (9, -1): 9.359, (16, -1): 44.164,
}

#: Pauling electronegativities (start of the bond-flow table)
PAULING_CHI: dict[int, float] = {1: 2.20, 3: 0.98, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
                                 11: 0.93, 14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66}

FAMILIES = ("flow", "dipole", "quad", "alpha", "compliance", "channel")

#: the channel progress starts CHANNEL_ONSET (Ang) beyond the reference length and turns on over
#: CHANNEL_WIDTH with a C2 ramp: exactly zero for any bond within normal vibrational range, so an
#: intact bond never spends its group's transfer. A heterolytic transfer completes far out (an
#: N-H of NH4+ at 1.5 r_e still looks homolytic in the pyscf-all labels; an onset of 0.15 A moved
#: ~0.3 e there and cost ~1.3 D), so the default starts late; the rates are learnable per bond type
#: but need long-range data to be pinned down.
CHANNEL_ONSET = 0.5
CHANNEL_WIDTH = 0.3


def _ramp(x: torch.Tensor, w: float) -> torch.Tensor:
    """C2 ramp: 0 for x <= 0, ``w (u^3 - u^4 / 2)`` for ``u = x / w`` in [0, 1], ``x - w / 2`` beyond."""
    u = (x / w).clamp(0.0, 1.0)
    inner = w * (u ** 3 - 0.5 * u ** 4)
    return torch.where(x > w, x - 0.5 * w, inner)


def free_atom_alpha(z: int, q: float, table: dict | None = None) -> float:
    """``alpha0(Z, q)``: linear between the integer charge states of ``table`` around ``q``; the
    nearest available state outside them (the neutral atom when nothing else is known)."""
    table = FREE_ATOM_ALPHA if table is None else table
    states = sorted(c for (zz, c) in table if zz == int(z))
    if not states:
        raise KeyError(f"no free-atom polarizability for Z={z}")
    lo = max([c for c in states if c <= q], default=None)
    hi = min([c for c in states if c >= q], default=None)
    if lo is None:
        return table[(int(z), hi)]
    if hi is None or hi == lo:
        return table[(int(z), lo)]
    w = (q - lo) / (hi - lo)
    return (1 - w) * table[(int(z), lo)] + w * table[(int(z), hi)]


def _sh_maps(backend, dtype=torch.float64):
    """``(A1 (3, 3), P2 (5, 9))``: backend l=1 components -> Cartesian vector (``v = y @ A1``) and
    backend l=2 components -> vec of a symmetric traceless matrix (``vec T = y @ P2``), fitted on
    fixed directions (the backend's normalization and ordering, whatever they are)."""
    g = torch.Generator().manual_seed(7)
    u = torch.randn(64, 3, generator=g, dtype=dtype)
    u = u / u.norm(dim=-1, keepdim=True)
    sh = backend.spherical_harmonics(u, 2).to(dtype)
    y1, y2 = sh[:, 1:4], sh[:, 4:9]
    t = (u[:, :, None] * u[:, None, :] - torch.eye(3, dtype=dtype) / 3.0).reshape(-1, 9)
    a1 = torch.linalg.lstsq(u, y1).solution          # y1 = u @ a1
    a2 = torch.linalg.lstsq(t, y2).solution          # y2 = vec T @ a2   (9, 5)
    return torch.linalg.inv(a1), torch.linalg.pinv(a2)


@dataclass
class ElectricalOutput:
    """One evaluation of the ECE electrical heads (a.u.)."""

    q: torch.Tensor                   # (N,) e
    mu: torch.Tensor                  # (N, 3) e*bohr
    quad_s: torch.Tensor              # (N, 5) spherical, e*bohr^2 (rsfff.ff.multipole convention)
    alpha0: torch.Tensor              # (N, 3, 3) bohr^3, isolated
    alpha: torch.Tensor               # (N, 3, 3) bohr^3, env-dressed (== alpha0 without env)
    compliance0: torch.Tensor         # (Nch,) on the SQE channel graph, isolated
    compliance: torch.Tensor          # (Nch,) env-dressed


class ECEElectricalHeads(nn.Module):
    """See the module docstring. ``bond_prior`` is the bonded head's ``(n_types, n_types, 3)``
    log ``(r_eq [bohr], D [Ha], k [Ha/bohr^2])`` table at construction (the envelope reference);
    ``type_z`` the atomic number of each type row."""

    def __init__(
        self,
        ece_features,                         # rsfff.ff.film.ece.ECEBondFeatures
        species: list[int],                   # atomic number of each species index
        type_z: list[int],
        bond_prior: torch.Tensor,
        *,
        latent_dim: int = 0,
        emb_dim: int = 8,
        hidden: int = 64,
        depth: int = 2,
        s_init: float = 0.5,
        flow_kappa: float = 0.15,
        alpha_table: dict | None = None,
        resp_latent_dim: int = 0,
        channels: bool = True,
        channel_onset: float = CHANNEL_ONSET,
        channel_width: float = CHANNEL_WIDTH,
    ) -> None:
        super().__init__()
        self.type_z = [int(z) for z in type_z]
        n_types = len(self.type_z)
        self.latent_dim = int(latent_dim)
        self.alpha_table = dict(FREE_ATOM_ALPHA if alpha_table is None else alpha_table)
        # alpha0 at the integer charge states -Q..Q per species, interpolated linearly in q_inf
        self.q_max = 2
        qs = range(-self.q_max, self.q_max + 1)
        self.register_buffer("alpha0_states", torch.tensor(
            [[free_atom_alpha(z, q, self.alpha_table) for q in qs] for z in species],
            dtype=torch.float64))
        prior = bond_prior.detach().to(torch.float64).exp()
        self.register_buffer("r_ref", prior[..., 0] * BOHR_ANG)                      # Ang
        self.register_buffer("beta_ref", torch.sqrt(prior[..., 2] / (2.0 * prior[..., 1])) / BOHR_ANG)
        self.env_log_scale = nn.Parameter(torch.zeros(len(FAMILIES)))
        # per bond type: how fast a heterolytic channel transfers its charge (log-scale on gamma)
        self.channels = bool(channels)
        self.channel_onset, self.channel_width = float(channel_onset), float(channel_width)
        self.channel_log_rate = nn.Parameter(torch.zeros(n_types, n_types))

        self.type_emb = nn.Embedding(n_types, emb_dim)
        d_dir = ece_features.directed_dim()
        self.extra_dim = 2 * emb_dim + 2 * self.latent_dim
        self.scalars = zero_init_readout(mlp(d_dir + self.extra_dim, hidden, depth, 3))
        self.equiv = EdgeReadout(ece_features.ece, "1x1o + 2x2e", hidden=hidden, depth=depth,
                                 zero_init=True, extra_dim=self.extra_dim, compress=True)
        self.chi = nn.Parameter(torch.tensor([PAULING_CHI.get(z, 2.5) for z in self.type_z]))
        self.log_kappa = nn.Parameter(torch.tensor(math.log(flow_kappa)))
        self.compliance_raw = nn.Parameter(
            torch.full((n_types, n_types), math.log(math.expm1(float(s_init)))))
        a1, p2 = _sh_maps(ece_features.ece.backend)
        self.register_buffer("sh1_to_cart", a1, persistent=False)
        self.register_buffer("sh2_to_cart", p2, persistent=False)
        # environment factors of the response parameters (gated, zero-init, exactly 1 isolated)
        self.resp_latent_dim = int(resp_latent_dim)
        if self.resp_latent_dim:
            self.alpha_env = zero_init_readout(mlp(self.resp_latent_dim, hidden, 1, 1))
            self.compliance_env = zero_init_readout(mlp(self.resp_latent_dim, hidden, 1, 1))

    # ------------------------------------------------------------------------------------- #
    def envelope(self, positions, type_idx, bonds) -> torch.Tensor:
        """``(Nb, n_families)`` electrical bond orders ``exp(-gamma (r - r_ref))``."""
        i, j = bonds[0], bonds[1]
        r = (positions[j] - positions[i]).norm(dim=-1)
        ti, tj = type_idx[i], type_idx[j]
        r0 = self.r_ref[ti, tj].to(r.dtype)
        gamma = self.beta_ref[ti, tj].to(r.dtype)[:, None] * self.env_log_scale.exp().to(r.dtype)
        return torch.exp(-gamma * (r - r0)[:, None])

    def channel_progress(self, positions, type_idx, bonds) -> torch.Tensor:
        """``(Nb,)`` ``log(1 - p_b) = -gamma ramp(r - r_ref - onset)``: exactly 0 for a bond in its
        normal range, ``-> -inf`` as it dissociates (kept as a log: no underflow)."""
        i, j = bonds[0], bonds[1]
        r = (positions[j] - positions[i]).norm(dim=-1)
        ti, tj = type_idx[i], type_idx[j]
        r0 = self.r_ref[ti, tj].to(r.dtype)
        rate = 0.5 * (self.channel_log_rate[ti, tj] + self.channel_log_rate[tj, ti])
        gamma = self.beta_ref[ti, tj].to(r.dtype) * (
            self.env_log_scale[FAMILIES.index("channel")] + rate).exp().to(r.dtype)
        return -gamma * _ramp(r - r0 - self.channel_onset, self.channel_width)

    def channel_charges(self, positions, type_idx, bonds, flow, group, graph) -> torch.Tensor:
        """``(N,)`` charge moved by the dissociation channels at this geometry (see the module
        docstring): exclusive within a group, additive across groups."""
        log_keep = self.channel_progress(positions, type_idx, bonds)        # log(1 - p)
        p = -torch.expm1(log_keep)
        t = p
        if group is not None and bool((group >= 0).any()):
            g = group.clamp(min=0)
            grouped = group >= 0
            tot = log_keep.new_zeros(positions.shape[0]).index_add(
                0, g[grouped], log_keep[grouped])
            t = torch.where(grouped, p * torch.exp(tot[g] - log_keep), p)
        return flow_to_charges(t * flow.to(p.dtype), graph, positions.shape[0])

    def alpha0(self, species_idx: torch.Tensor, q_inf: torch.Tensor) -> torch.Tensor:
        """``(N,)`` free atom / ion polarizability of each atom's diabatic charge state."""
        x = (q_inf + self.q_max).clamp(0, 2 * self.q_max)
        lo = x.floor().long().clamp(max=2 * self.q_max - 1)
        w = x - lo.to(x.dtype)
        tab = self.alpha0_states.to(q_inf.dtype)[species_idx]
        return (1 - w) * tab.gather(1, lo[:, None])[:, 0] + w * tab.gather(1, lo[:, None] + 1)[:, 0]

    def _channel_bonds(self, channels, bonds, n_atoms):
        """``(Nch,)`` bond column of each SQE channel, ``-1`` for a non-bonded pair."""
        n = int(n_atoms)
        bk = torch.minimum(bonds[0], bonds[1]) * n + torch.maximum(bonds[0], bonds[1])
        ck = torch.minimum(channels[0], channels[1]) * n + torch.maximum(channels[0], channels[1])
        order = torch.argsort(bk)
        pos = torch.searchsorted(bk[order], ck).clamp(max=max(bk.numel() - 1, 0))
        col = order[pos] if bk.numel() else torch.zeros_like(ck)
        hit = bk[col] == ck if bk.numel() else torch.zeros_like(ck, dtype=torch.bool)
        return torch.where(hit, col, torch.full_like(col, -1))

    def forward(
        self,
        ef,                                   # bb EdgeFeatures of topo.bond_index (or None)
        positions: torch.Tensor,              # (N, 3) Ang
        species_idx: torch.Tensor,            # (N,) species index
        type_idx: torch.Tensor,               # (N,) rows of the type tables
        bonds: torch.Tensor,                  # (2, Nb) topo.bond_index
        q_inf: torch.Tensor,                  # (N,) diabatic asymptotic charges
        channels: torch.Tensor,               # (2, Nch) SQE channel graph
        chan_flow: torch.Tensor | None = None,   # (Nb,) channel flow along topo orientation
        chan_group: torch.Tensor | None = None,  # (Nb,) exclusive group (atom index) or -1
        z_perm: torch.Tensor | None = None,   # (N, latent) permanent-family latent (latents=True)
        z_resp: torch.Tensor | None = None,   # (N, resp latent) env-dressed response latent
        gate: torch.Tensor | None = None,     # (N,) environment gate
    ) -> ElectricalOutput:
        n = int(positions.shape[0])
        dt = positions.dtype
        nb = int(bonds.shape[1])
        eye = torch.eye(3, dtype=dt, device=positions.device)
        nch = int(channels.shape[1])
        q_asym = q_inf.to(dt)
        if nb and ef is not None and chan_flow is not None and self.channels:
            q_asym = q_asym + self.channel_charges(positions, type_idx, bonds, chan_flow, chan_group, ef.graph)
        a0 = self.alpha0(species_idx, q_asym)
        if nb == 0 or ef is None:
            alpha = a0[:, None, None] * eye
            zc = positions.new_zeros(nch)
            return ElectricalOutput(q_inf.to(dt), positions.new_zeros(n, 3), positions.new_zeros(n, 5),
                                    alpha, alpha, zc, zc)
        src, dst = ef.graph.edge_index
        env = self.envelope(positions, type_idx, bonds)                       # (Nb, F)
        env_e = torch.cat((env, env), 0)                                       # edge e and e + Nb
        emb = self.type_emb(type_idx).to(dt)
        extra = [emb[src], emb[dst]]
        if self.latent_dim:
            if z_perm is None:
                raise ValueError("these heads were built with latents; pass z_perm")
            extra += [z_perm[src], z_perm[dst]]
        extra = torch.cat(extra, -1)
        x = torch.cat((type(self)._directed(ef), extra), -1)
        h = self.scalars(x)                                                    # (E, 3)

        # charges: antisymmetric bond flows (charge moved a -> b along the stored orientation)
        ta, tb = type_idx[bonds[0]], type_idx[bonds[1]]
        table = self.log_kappa.exp() * (self.chi[ta] - self.chi[tb]).to(dt)
        flow = env[:, 0] * (table + h[:nb, 0] - h[nb:, 0])
        q = q_asym + flow_to_charges(flow, ef.graph, n)

        y = self.equiv(ef, extra)["edge"]
        v = y["1o"][..., 0] @ self.sh1_to_cart.to(dt)                          # (E, 3)
        mu = positions.new_zeros(n, 3).index_add(0, src, env_e[:, 1:2] * v)
        t2 = torch.einsum("ekm,kc->ecm", y["2e"], self.sh2_to_cart.to(dt)).reshape(-1, 3, 3, 2)
        t2 = 0.5 * (t2 + t2.transpose(1, 2))
        quad_c = positions.new_zeros(n, 3, 3).index_add(0, src, env_e[:, 2, None, None] * t2[..., 0])
        quad_s = cartesian_to_spherical_quadrupole(quad_c)

        log_c = positions.new_zeros(n).index_add(0, src, env_e[:, 3] * h[:, 1])
        K = positions.new_zeros(n, 3, 3).index_add(0, src, env_e[:, 3, None, None] * t2[..., 1])
        alpha0 = (a0 * log_c.exp())[:, None, None] * torch.linalg.matrix_exp(-K)
        alpha0 = 0.5 * (alpha0 + alpha0.transpose(1, 2))

        s_bond = env[:, 4] * nn.functional.softplus(
            self.compliance_raw[torch.minimum(ta, tb), torch.maximum(ta, tb)].to(dt)
            + h[:nb, 2] + h[nb:, 2])
        col = self._channel_bonds(channels, bonds, n)
        comp0 = torch.where(col >= 0, s_bond[col.clamp(min=0)], positions.new_zeros(nch))

        alpha, comp = alpha0, comp0
        if self.resp_latent_dim and z_resp is not None and gate is not None:
            alpha = alpha0 * torch.exp(gate * self.alpha_env(z_resp)[:, 0])[:, None, None]
            ci, cj = channels[0], channels[1]
            g_ch = 0.5 * (gate[ci] + gate[cj])
            comp = comp0 * torch.exp(g_ch * self.compliance_env(z_resp[ci] + z_resp[cj])[:, 0])
        return ElectricalOutput(q, mu, quad_s, alpha0, alpha, comp0, comp)

    @staticmethod
    def _directed(ef) -> torch.Tensor:
        return torch.asinh(torch.cat((ef.inv0e, ef.inv0o ** 2, ef.bond_radial), dim=-1))
