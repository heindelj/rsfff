"""Local diabatic mixture on top of the frozen monomer stack (Phase-2 slice).

Given a system and an *active space* of competing diabatic assignments (the bound parent state
plus its dissociated partitions, from :func:`rsfff.mlip.diabats.enumerate_diabats`), this blends
their per-atom monomer features and 1-body predictions with locally gated weights, then runs one
Split-Charge Equilibration solve on the finest common refinement with a switched inter-fragment
compliance (docs/mixture_of_diabatic_embeddings.md §2.2, §2.4).

Everything trained lives in a **frozen** :class:`rsfff.mlip.monomer.MonomerModel`, reused
untouched; this module only adds the gate (a zero-initialized MLP + envelope logic), so the
demonstration runs on the Phase-1 checkpoint with no retraining. The physics at the ends is
exact by construction:

* **Equilibrium** — the dissociated envelope is 0, so `c_bound = 1`; the switch is 1 over all
  sampled bond lengths; the finest-refinement solve reduces to the monomer stack's. `MixtureModel`
  reproduces `MonomerModel` to machine precision.
* **Dissociation** — the bound envelope is 0, so the blend *is* the dissociated diabat, whose
  single-atom (or trained-monomer) fragments carry exact reference features and integer baseline
  charges; the switch closes the inter-fragment channel, so no charge flows and the products
  reach their reference states exactly.

Scope: one system per call, one reactive center, `Δh ≡ 0` (no ACE correction yet), no long-range
electrostatics between separated fragments. See the plan for what is deliberately deferred.

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

from .diabats import FragmentAssignment, finest_common_refinement
from .eem import atomic_dipoles
from .heads import mlp
from .monomer import MonomerModel
from .sqe import atomic_dipole_energy, sqe_solve
from .switch import pairwise_switch, validity_bump
from ..train.data import Batch


@dataclass
class EnvelopeConfig:
    """Hand-specified geometry for the gate of one reactive center (no training).

    bound_envelope / channel_envelope : kwargs for :func:`rsfff.mlip.switch.validity_bump`, the
        right-closing bound Ω and the left-opening dissociated Ω (their supports must overlap).
    switch_r_on / switch_r_off        : the inter-fragment compliance switch window;
        ``switch_r_on`` must exceed every sampled bond length so the frozen fit is untouched.
    beta / tau                        : Coulomb-bias strength and softmax temperature of the gate.
    """

    bound_envelope: dict
    channel_envelope: dict
    switch_r_on: float
    switch_r_off: float
    beta: float = 0.0
    tau: float = 1.0


@dataclass
class MixtureOutput:
    """Outputs of one mixture forward pass (single system).

    energy      : ()        total energy
    charges     : (N,)      SQE charges on the finest refinement; sum == formal charge
    transfers   : (Nb,)     split charges along the (finest-refinement) channels
    compliance  : (Nb,)     switched channel compliances
    weights     : (M,)      mixing weights c_K over the active diabats (sum to 1)
    omega       : (M,)      validity envelopes Ω_K
    order_param : ()        reaction-coordinate bond length driving the envelopes
    dipole      : (3,)      molecular dipole (e*Angstrom, stored frame)
    alpha       : (3, 3)    molecular polarizability
    bond_index  : (2, Nb)   the finest-refinement channel graph (global atom indices)
    is_inter    : (Nb,)     which channels are inter-fragment (switched)
    """

    energy: torch.Tensor
    charges: torch.Tensor
    transfers: torch.Tensor
    compliance: torch.Tensor
    weights: torch.Tensor
    omega: torch.Tensor
    order_param: torch.Tensor
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
    """Frozen :class:`MonomerModel` + a local diabatic gate (see module docstring)."""

    def __init__(self, monomer: MonomerModel, *, gate_hidden: int = 32, gate_depth: int = 1):
        super().__init__()
        self.monomer = monomer
        p0 = monomer.featurizer.feature_dims[0]
        # Gate logit MLP on the pooled per-diabat features; zero-initialized so the untrained
        # gate is driven entirely by the validity envelopes and the Coulomb/E_monomer bias.
        self.gate_mlp = mlp(p0, gate_hidden, gate_depth, 1)
        with torch.no_grad():
            self.gate_mlp[-1].weight.zero_()
            self.gate_mlp[-1].bias.zero_()

    # -- per-diabat 1-body quantities, returned in the ORIGINAL atom order -----------------
    def _diabat_quantities(self, batch: Batch, assignment: FragmentAssignment) -> dict:
        m = self.monomer
        device = batch.positions.device
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
        species_p = m.featurizer.species_index(pbatch.atomic_numbers)
        fq = torch.as_tensor(assignment.fragment_charge, dtype=batch.positions.dtype, device=device)
        fs = torch.as_tensor(assignment.fragment_two_s, dtype=batch.positions.dtype, device=device)
        ref = m.embedding(species_p, fi[perm], fq, fs, assignment.n_fragments)
        feats = m.featurizer(pbatch, m.weight_net(ref.embedding), fragment_idx=fi[perm])
        e1, chi, eta, chivec, alpha = m.params(feats, ref.embedding)

        # unpermute everything back to the original atom order
        return dict(
            h=feats.inv_feats[inv], e1=e1[inv], chi=chi[inv], eta=eta[inv],
            chivec=chivec[inv], alpha=alpha[inv], q0=ref.baseline_charge[inv],
            species=m.featurizer.species_index(batch.atomic_numbers),
            fragment_idx=fi, fragment_charge=fq, atoms_of=assignment.fragments,
        )

    @staticmethod
    def _coulomb_bias(q: dict, positions: torch.Tensor) -> torch.Tensor:
        """Σ_{a<b} Q_a Q_b / r_ab over fragment centroids (§2.2.2). Zero for one fragment."""
        frags = q["atoms_of"]
        if len(frags) < 2:
            return positions.new_zeros(())
        charges = q["fragment_charge"]
        centroids = [positions[list(atoms)].mean(0) for _, atoms in frags]
        total = positions.new_zeros(())
        for a in range(len(frags)):
            for b in range(a + 1, len(frags)):
                r = (centroids[a] - centroids[b]).norm()
                total = total + charges[a] * charges[b] / r.clamp(min=1e-6)
        return total

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
        M = len(assignments)
        per = [self._diabat_quantities(batch, a) for a in assignments]

        # -- reaction coordinate & validity envelopes (shared across the active space) --
        order_bond = assignments[0].order_bond
        if order_bond is None:
            raise ValueError("active space has no reaction-coordinate bond (order_bond)")
        r_op = (pos[order_bond[0]] - pos[order_bond[1]]).norm()
        env_kwargs = [env.bound_envelope] + [env.channel_envelope] * (M - 1)
        omega = torch.stack([validity_bump(r_op, **e) for e in env_kwargs])       # (M,)

        # -- gate logits: zero-init MLP on pooled features minus the zeroth-order bias --
        e0 = self.monomer.reference_energies
        logits = []
        for q in per:
            pooled = q["h"].mean(0)                                   # pool over the one center
            gate = self.gate_mlp(pooled).squeeze(-1)
            e_mono = (e0[q["species"]] + q["e1"]).sum()
            logits.append(gate - env.beta * (e_mono + self._coulomb_bias(q, pos)))
        logits = torch.stack(logits)                                              # (M,)

        # -- renormalized envelope-weighted softmax (§2.2.3): Σ c_K = 1 (charge conservation) --
        w = omega * torch.exp((logits - logits.max()) / env.tau)
        c = w / w.sum().clamp(min=1e-30)                                          # (M,)

        blend = lambda key: sum(c[k] * per[k][key] for k in range(M))            # noqa: E731
        h = blend("h"); e1 = blend("e1"); chi = blend("chi"); eta = blend("eta")
        chivec = blend("chivec"); alpha = blend("alpha"); q0 = blend("q0")

        # -- channel graph on the finest refinement; switch the inter-fragment channels --
        bond_np, inter_np = mixture_channel_graph(assignments)
        bond_index = torch.as_tensor(bond_np, device=pos.device)
        is_inter = torch.as_tensor(inter_np, device=pos.device)
        s_raw = self.monomer.compliance_head(h, pos, bond_index)
        r_bond = (pos[bond_index[0]] - pos[bond_index[1]]).norm(dim=-1)
        s = torch.where(is_inter, s_raw * pairwise_switch(r_bond, env.switch_r_on, env.switch_r_off), s_raw)

        # -- single global SQE solve on the finest refinement --
        n = pos.shape[0]
        batch_idx = torch.zeros(n, dtype=torch.long, device=pos.device)
        bond_batch = torch.zeros(bond_index.shape[1], dtype=torch.long, device=pos.device)
        sol = sqe_solve(chi, eta, s, q0, pos, bond_index, batch_idx, bond_batch, 1, field)

        mu_i = atomic_dipoles(chivec, alpha, batch_idx, field)
        e_dip = atomic_dipole_energy(chivec, alpha, batch_idx, 1, field)
        e_atom = e1 + e0[per[0]["species"]]
        energy = sol.energy[0] + e_dip[0] + e_atom.sum()
        dipole = (sol.charges.unsqueeze(-1) * pos + mu_i).sum(0)
        alpha_mol = alpha.sum(0) + sol.alpha_flow[0]

        return MixtureOutput(
            energy=energy, charges=sol.charges, transfers=sol.transfers, compliance=s,
            weights=c, omega=omega, order_param=r_op, dipole=dipole, alpha=alpha_mol,
            bond_index=bond_index, is_inter=is_inter,
        )
