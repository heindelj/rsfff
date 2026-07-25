"""Latent-space diabatic adiabaticization on top of the frozen monomer stack.

Implements Revision 3 of docs/mixture_of_diabatic_embeddings.md: the model does **not** decode
each diabat's observables and blend those. It blends the diabats' per-atom *feature bundles* into
one weighted mean, adds a learned nonlinear correction that vanishes at every pure-state vertex,
and sends the single resulting **adiabatic feature** through **one** shared decoder and **one**
SQE/response solve (§7, §9):

    z_i^ad = Σ_K c_K ĥ_i^(K) + Σ_{K<J} 4 c_K c_J 𝒪_KJ Ψ_i^KJ
    (E1, χ, η, χvec, α, s) = D_shared(z_i^ad),   then one global SQE solve.

Everything trained lives in a **frozen** :class:`rsfff.mlip.monomer.MonomerModel`, reused untouched
as the shared decoder (its :class:`MonomerParameterHeads` and ``compliance_head`` are the heads
"common to all systems"). The only new parameters are the small
:class:`rsfff.mlip.adiabatic.AdiabaticCorrection` (``Ψ``), small-initialized; the demonstration
runs on the Phase-1 checkpoint with no retraining.

The end behaviours are exact by construction:

* **Pure-state vertex** — when a single diabat has ``c=1`` (a bare atom, or any system at a
  geometry where only one validity envelope is open), the mean *is* that diabat's feature, the
  correction is exactly zero (``4c_Kc_J = 0``), and the shared decoder reproduces
  :class:`MonomerModel` for that state to machine precision.
* **Dissociation** — the dissociated diabat's single-atom / trained-monomer fragments carry exact
  reference features and integer baseline charges; the compliance switch closes the inter-fragment
  channel so no charge flows; and the coefficient sharpens onto the lowest cover, so the products
  reach their reference states.

Coefficients come from a **cheap proxy** (docs §4, §6): ``c_K = Ω_K exp(−Ẽ_K/τ) / Σ_J Ω_J
exp(−Ẽ_J/τ)`` with ``Ẽ_K`` the isolated-fragment self-energy plus formal-charge Coulomb, and
``Ω_K`` the hand-set validity envelope. There is no learned gate here (deferred to Phase-3
training).

Scope: one system per call, one reactive center, ``Δh_env ≡ 0`` (no ACE environment correction),
no long-range electrostatics between separated fragments. See the plan for what is deferred.

**Why atoms get permuted internally:** ``torch_cluster.radius_graph`` assumes a sorted/contiguous
``batch`` vector, but a dissociated diabat's ``fragment_idx`` is non-contiguous (e.g. `[0, 1, 0]`
for H2O losing its middle-listed H). Each per-diabat featurization therefore runs in an atom
order that makes the fragment grouping contiguous, and the per-atom outputs are permuted back so
all diabats share the original atom order for blending.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from .adiabatic import AdiabaticCorrection, AdiabaticDelta
from .diabats import FragmentAssignment, finest_common_refinement
from .eem import atomic_dipoles
from .monomer import MonomerModel
from .sqe import atomic_dipole_energy, sqe_solve
from .switch import pairwise_switch, validity_bump
from ..features import LambdaFeatures
from ..train.data import Batch

# Coulomb constant in Hartree*Angstrom: E = Q_a Q_b * BOHR / r[Ang] (since r[Bohr] = r[Ang]/BOHR).
_BOHR_ANG = 0.52917721067


@dataclass
class EnvelopeConfig:
    """Hand-specified geometry of the gate + overlap for one reactive center (no training).

    bound_envelope / channel_envelope : kwargs for :func:`rsfff.mlip.switch.validity_bump`, the
        right-closing bound Ω and the left-opening dissociated Ω (their supports must overlap).
        The same ``channel_envelope`` is applied to every dissociated diabat of the center.
    switch_r_on / switch_r_off        : the inter-fragment *compliance* switch window;
        ``switch_r_on`` must exceed every sampled bond length so the frozen fit is untouched.
    overlap_r_on / overlap_r_off      : the geometric *overlap* switch window ``𝒪(r)`` on the
        transition contact; multiplies the nonlinear correction so it decays to zero at fragment
        separation (§7.6). Defaults to the compliance-switch window.
    tau                               : proxy softmax temperature (Hartree).
    """

    bound_envelope: dict
    channel_envelope: dict
    switch_r_on: float
    switch_r_off: float
    overlap_r_on: float | None = None
    overlap_r_off: float | None = None
    tau: float = 0.05

    def overlap_window(self) -> tuple[float, float]:
        r_on = self.switch_r_on if self.overlap_r_on is None else self.overlap_r_on
        r_off = self.switch_r_off if self.overlap_r_off is None else self.overlap_r_off
        return float(r_on), float(r_off)


@dataclass
class MixtureOutput:
    """Outputs of one mixture forward pass (single system).

    energy          : ()        total energy
    charges         : (N,)      SQE charges on the finest refinement; sum == formal charge
    transfers       : (Nb,)     split charges along the (finest-refinement) channels
    compliance      : (Nb,)     switched channel compliances
    weights         : (M,)      mixing weights c_K over the active diabats (sum to 1)
    omega           : (M,)      validity envelopes Ω_K
    proxy_energy    : (M,)      cheap cover scores Ẽ_K driving the coefficients
    order_param     : ()        reaction-coordinate bond length driving the envelopes
    overlap         : ()        geometric overlap factor 𝒪 on the transition contact
    correction_norm : ()        ‖Δz‖ of the nonlinear adiabatic correction (0 at every vertex)
    dipole          : (3,)      molecular dipole (e*Angstrom, stored frame)
    alpha           : (3, 3)    molecular polarizability
    bond_index      : (2, Nb)   the finest-refinement channel graph (global atom indices)
    is_inter        : (Nb,)     which channels are inter-fragment (switched)
    """

    energy: torch.Tensor
    charges: torch.Tensor
    transfers: torch.Tensor
    compliance: torch.Tensor
    weights: torch.Tensor
    omega: torch.Tensor
    proxy_energy: torch.Tensor
    order_param: torch.Tensor
    overlap: torch.Tensor
    correction_norm: torch.Tensor
    dipole: torch.Tensor
    alpha: torch.Tensor
    bond_index: torch.Tensor
    is_inter: torch.Tensor


def mixture_channel_graph(
    assignments: Sequence[FragmentAssignment],
) -> tuple[np.ndarray, np.ndarray]:
    """Union of all diabats' channels, classified inter/intra on the finest refinement.

    Returns ``(bond_index (2, Nb), is_inter (Nb,))``. A channel is inter-fragment — and so
    carries the switch — iff its two atoms fall in different blocks of the finest common
    refinement of the active partitions (§2.4.1, §2.4.3).
    """
    fcr = finest_common_refinement(assignments)
    seen: dict[frozenset, tuple[int, int]] = {}
    for a in assignments:
        bi = a.bond_index
        for e in range(bi.shape[1]):
            i, j = int(bi[0, e]), int(bi[1, e])
            seen.setdefault(frozenset((i, j)), (i, j))
    bonds = list(seen.values())
    if not bonds:
        return np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=bool)
    bond_index = np.asarray(bonds, dtype=np.int64).T
    is_inter = np.asarray([fcr[i] != fcr[j] for i, j in bonds], dtype=bool)
    return bond_index, is_inter


class MixtureModel(nn.Module):
    """Frozen :class:`MonomerModel` as a shared decoder + learned adiabaticization (module doc)."""

    def __init__(
        self,
        monomer: MonomerModel,
        *,
        hidden: int = 64,
        depth: int = 2,
        z_r_dim: int = 16,
        init_scale: float = 1e-2,
    ):
        super().__init__()
        self.monomer = monomer
        dims = monomer.featurizer.feature_dims
        emb_dim = monomer.embedding.emb_dim
        self.correction = AdiabaticCorrection(
            dims[0], dims[1], dims[2], emb_dim,
            hidden=hidden, depth=depth, z_r_dim=z_r_dim, init_scale=init_scale,
        )

    # -- per-diabat feature bundle, returned in the ORIGINAL atom order ---------------------
    def _diabat_features(self, batch: Batch, assignment: FragmentAssignment) -> dict:
        m = self.monomer
        device = batch.positions.device
        dtype = batch.positions.dtype
        fi = torch.as_tensor(assignment.fragment_idx, device=device)

        # Permute atoms so the fragment grouping is contiguous (radius_graph needs a sorted
        # batch); featurize in that order, then invert the permutation on per-atom outputs.
        perm = torch.argsort(fi, stable=True)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.shape[0], device=device)

        pbatch = Batch(
            positions=batch.positions[perm],
            atomic_numbers=batch.atomic_numbers[perm],
            batch_idx=torch.zeros(perm.shape[0], dtype=torch.long, device=device),
            n_systems=1,
            energy=torch.zeros(1, device=device),
            forces=torch.zeros(perm.shape[0], 3, device=device),
        )
        fq = torch.as_tensor(assignment.fragment_charge, dtype=dtype, device=device)
        fs = torch.as_tensor(assignment.fragment_two_s, dtype=dtype, device=device)
        species_p = m.featurizer.species_index(pbatch.atomic_numbers)
        ref = m.embedding(species_p, fi[perm], fq, fs, assignment.n_fragments)
        feats = m.featurizer(pbatch, m.weight_net(ref.embedding), fragment_idx=fi[perm])

        # Cheap 1-body decode for the proxy self-energy (no SQE solve): the isolated-fragment
        # energy Σ_i (E0[Z_i] + E1_i + χ_i q0_i + ½ η_i q0_i^2). Per-diabat proxy work is
        # explicitly permitted (§13.8); the one full evaluation is the blended decode below.
        e1, chi, eta, _, _ = m.params(feats, ref.embedding)
        q0 = ref.baseline_charge
        e0 = m.reference_energies[species_p]
        self_energy = (e0 + e1 + chi * q0 + 0.5 * eta * q0 * q0).sum()

        return dict(
            inv=feats.inv_feats[inv],
            vec=feats.vec_feats[inv],
            equiv=feats.equiv_feats[inv],
            emb=ref.embedding[inv],
            q0=q0[inv],
            self_energy=self_energy,
            atoms_of=assignment.fragments,
            fragment_charge=fq,
        )

    @staticmethod
    def _coulomb(per: dict, positions: torch.Tensor) -> torch.Tensor:
        """Σ_{a<b} Q_a Q_b / r_ab over fragment centroids, in Hartree. Zero for one fragment."""
        frags = per["atoms_of"]
        if len(frags) < 2:
            return positions.new_zeros(())
        charges = per["fragment_charge"]
        centroids = [positions[list(atoms)].mean(0) for _, atoms in frags]
        total = positions.new_zeros(())
        for a in range(len(frags)):
            for b in range(a + 1, len(frags)):
                r = (centroids[a] - centroids[b]).norm().clamp(min=1e-6)
                total = total + charges[a] * charges[b] * _BOHR_ANG / r
        return total

    def _coefficients(
        self, per: list[dict], positions: torch.Tensor, omega: torch.Tensor, tau: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Envelope-gated softmax of the cheap proxy score (§6.1). Returns ``(c, Ẽ)``."""
        proxy = torch.stack([p["self_energy"] + self._coulomb(p, positions) for p in per])
        w = omega * torch.exp(-(proxy - proxy.min()) / tau)
        c = w / w.sum().clamp(min=1e-30)
        return c, proxy

    def forward(
        self,
        batch: Batch,
        assignments: Sequence[FragmentAssignment],
        env: EnvelopeConfig,
        *,
        field: torch.Tensor | None = None,
    ) -> MixtureOutput:
        if int(batch.n_systems) != 1:
            raise ValueError("MixtureModel.forward handles one system at a time")
        pos = batch.positions
        m = self.monomer
        M = len(assignments)
        per = [self._diabat_features(batch, a) for a in assignments]

        # -- reaction coordinate, validity envelopes, geometric overlap (shared across states) --
        order_bond = assignments[0].order_bond
        if M == 1:
            # A single active diabat (a bare atom, or any state with no dissociation channel): it
            # is trivially the whole cover, c = 1, and there is no crossover to gate or correct.
            r_op = pos.new_zeros(())
            omega = pos.new_ones(1)
            overlap = pos.new_zeros(())
            c, proxy = self._coefficients(per, pos, omega, env.tau)
        else:
            if order_bond is None:
                raise ValueError("active space has no reaction-coordinate bond (order_bond)")
            r_op = (pos[order_bond[0]] - pos[order_bond[1]]).norm()
            env_kwargs = [env.bound_envelope] + [env.channel_envelope] * (M - 1)
            omega = torch.stack([validity_bump(r_op, **e) for e in env_kwargs])       # (M,)
            r_on, r_off = env.overlap_window()
            overlap = pairwise_switch(r_op, r_on, r_off)                              # ()
            # -- coefficients from the cheap proxy (§6.1) --
            c, proxy = self._coefficients(per, pos, omega, env.tau)                   # (M,), (M,)

        # -- blend feature bundles (§7.1); q0 kept as its own exact-conserving convex blend --
        inv = torch.stack([p["inv"] for p in per])            # (M, N, P0)
        vec = torch.stack([p["vec"] for p in per])            # (M, N, 3, P1)
        equiv = torch.stack([p["equiv"] for p in per])        # (M, N, 5, P2)
        emb = torch.stack([p["emb"] for p in per])            # (M, N, D)
        q0 = sum(c[k] * per[k]["q0"] for k in range(M))       # (N,) sums to Q exactly

        mu_inv = (c.reshape(M, 1, 1) * inv).sum(0)
        mu_vec = (c.reshape(M, 1, 1, 1) * vec).sum(0)
        mu_equiv = (c.reshape(M, 1, 1, 1) * equiv).sum(0)
        mu_emb = (c.reshape(M, 1, 1) * emb).sum(0)

        # -- nonlinear adiabatic correction (§7.3); zero at every vertex, decays with overlap --
        n = pos.shape[0]
        center_mask = torch.ones(n, dtype=torch.bool, device=pos.device)
        delta: AdiabaticDelta = self.correction(inv, vec, equiv, emb, c, overlap, center_mask)
        z_inv = mu_inv + delta.d_inv
        z_vec = mu_vec + delta.d_vec
        z_equiv = mu_equiv + delta.d_equiv
        z_emb = mu_emb + delta.d_emb

        # -- one shared decode on the single adiabatic feature (§9.1) --
        species = m.featurizer.species_index(batch.atomic_numbers)
        batch_idx = torch.zeros(n, dtype=torch.long, device=pos.device)
        z_feats = LambdaFeatures(
            inv_feats=z_inv, equiv_feats=z_equiv, species_idx=species,
            batch_idx=batch_idx, vec_feats=z_vec,
        )
        e1, chi, eta, chivec, alpha = m.params(z_feats, z_emb)

        # -- channel graph on the finest refinement; switch the inter-fragment channels --
        bond_np, inter_np = mixture_channel_graph(assignments)
        bond_index = torch.as_tensor(bond_np, device=pos.device)
        is_inter = torch.as_tensor(inter_np, device=pos.device)
        s_raw = m.compliance_head(z_inv, pos, bond_index)
        r_bond = (pos[bond_index[0]] - pos[bond_index[1]]).norm(dim=-1)
        s = torch.where(
            is_inter, s_raw * pairwise_switch(r_bond, env.switch_r_on, env.switch_r_off), s_raw
        )

        # -- single global SQE solve on the finest refinement --
        bond_batch = torch.zeros(bond_index.shape[1], dtype=torch.long, device=pos.device)
        sol = sqe_solve(chi, eta, s, q0, pos, bond_index, batch_idx, bond_batch, 1, field)

        mu_i = atomic_dipoles(chivec, alpha, batch_idx, field)
        e_dip = atomic_dipole_energy(chivec, alpha, batch_idx, 1, field)
        e_atom = e1 + m.reference_energies[species]
        energy = sol.energy[0] + e_dip[0] + e_atom.sum()
        dipole = (sol.charges.unsqueeze(-1) * pos + mu_i).sum(0)
        alpha_mol = alpha.sum(0) + sol.alpha_flow[0]

        return MixtureOutput(
            energy=energy, charges=sol.charges, transfers=sol.transfers, compliance=s,
            weights=c, omega=omega, proxy_energy=proxy, order_param=r_op, overlap=overlap,
            correction_norm=delta.norm(), dipole=dipole, alpha=alpha_mol,
            bond_index=bond_index, is_inter=is_inter,
        )
