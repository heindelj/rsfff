"""The per-atom energy of the electronic state, as a fragment expert emits it.

``E_bond,i = E_theta( h_i , eta_i , M_i , Phi_i )``   with   ``M = (q, mu, Theta)``,
``Phi = (V, E, grad E)``.

This is what carries the covalent bonding energy inside ``fragment_energy``, and it is read at
two electronic states and two parameter sets, which is where charge transfer comes from
(``docs/fff_v2.md`` §7)::

    E_bond^0   = E_theta( h , 0   , M^frozen , Phi^intra )   -> fragment_energy
    E_bond^ind = E_theta( h , eta , M^ind    , Phi^ind   )   -> induction gets (ind - 0)

One set of weights. What separates the two is the *state* and the *slot*, not a parameter, and
the difference between them is a measurable quantity rather than a readout with its own private
lever. Two previous attempts at charge transfer gave it a dedicated readout and both ended up
with a network wearing the label -- 96.7% neural in one, 100% descriptor swap in the other.

``Phi^intra`` is built from **intra-fragment pairs only**. The field an atom feels from its own
molecule is part of what its bonds are worth; the field from a neighbouring molecule is not.

What this drops relative to ``AtomicStateEnergy``
-------------------------------------------------
Two things, and they are the same thing twice.

**The free-atom anchoring.** That head returned ``net(x) - net(ref)`` with ``ref`` the all-zero
input, so a lone neutral atom's energy was exactly the tabulated ``E0``. The cost is that the
readout's final-layer bias appears in *both* terms and cancels identically -- its gradient was
measured at exactly ``0.000e+00``. Since ``E0`` is a frozen buffer and a uniform ``chi`` shift is
inert (``sum_i q_i = 0`` per fragment), the model was left with **no** direction that moves the
one-body constant without also reshaping its geometry dependence. That is a conditioning
failure, not a capacity one, and it cost a measured constant ``-5.23 kJ/mol`` per fragment that
was 99.7% of the one-body loss.

**The per-species offset.** ``species_offset``, gated by ``tanh(||h||/offset_gate)`` so it
vanished where the anchoring did, existed purely to hand that direction back. With the anchoring
gone the bias is free again and the gadget is redundant.

The consequence is explicit and intended: **this head has no exact isolated-atom limit.** A
water expert is not asked about a bare oxygen, and saying so is the applicability metric's job
(:class:`rsfff.ff.expert.ApplicabilityHead`), not an anchoring's. What ``E0`` still does is fix
the zero: ``fragment_energy`` is an atomization energy up to that shift, and the shift is the
only external information the one-body sector receives.

The readout stays zero-initialized and weight-decay exempt, so a fresh model starts at
``E0 + E_internal`` and the bias trains down to the atomization energy from there.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..mlip.heads import (
    env_reduce_parameter,
    exempt_from_weight_decay,
    slot_reduce,
    two_slot_mlp,
    zero_init_readout,
)
from .state_invariants import n_state_invariants, state_invariants

__all__ = ["FragmentBondEnergy"]


class FragmentBondEnergy(nn.Module):
    """``(N,)`` Hartree from the two feature slots and the electronic state.

    Args
    ----
    p0, p1, p2      : widths of the **fragment** slot's lambda=0/1/2 blocks. ``p1``/``p2`` may
                      be ``None``, dropping the corresponding feature contractions (their slots
                      stay present, filled with zeros).
    p_env, p1_env, p2_env : the environment slot's widths. All 0 gives a single-slot head whose
                      output is a function of the fragment alone -- the ablation.
    n_species       : embedding table size.
    irrep2_to_spherical : ``(5, 5)`` change of basis from the backend's lambda=2 slots to the
                      spherical convention, from :func:`rsfff.ff.multipole.irrep2_to_spherical`.
                      Required whenever ``p2`` is given, for the reason that function records.
    equiv_channels  : how many channels the lambda=1/2 features are reduced to before being
                      contracted against the electronic state.
    energy_scale    : output scale in Hartree. Size it against the thing being described -- a
                      covalent bond, so ~0.2 Ha -- not against the interaction corrections.
    """

    def __init__(
        self,
        p0: int,
        n_species: int,
        *,
        p1: int | None = None,
        p2: int | None = None,
        p_env: int = 0,
        p1_env: int = 0,
        p2_env: int = 0,
        irrep2_to_spherical: torch.Tensor | None = None,
        emb_dim: int = 16,
        hidden: int = 64,
        depth: int = 2,
        equiv_channels: int = 8,
        energy_scale: float = 0.2,
    ) -> None:
        super().__init__()
        if not energy_scale > 0.0:
            raise ValueError(f"FragmentBondEnergy needs energy_scale > 0, got {energy_scale}")
        self.energy_scale = float(energy_scale)
        self.equiv_channels = int(equiv_channels)
        self.p0, self.p_env = int(p0), int(p_env)
        self.species_emb = nn.Embedding(n_species, emb_dim)

        k = self.equiv_channels
        self.vec_reduce = (
            None if p1 is None else nn.Parameter(torch.randn(p1, k) / (p1 ** 0.5))
        )
        self.vec_reduce_env = env_reduce_parameter(p1_env if p1 is not None else 0, k)
        self.equiv_reduce = (
            None if p2 is None else nn.Parameter(torch.randn(p2, k) / (p2 ** 0.5))
        )
        self.equiv_reduce_env = env_reduce_parameter(p2_env if p2 is not None else 0, k)
        if p2 is not None:
            if irrep2_to_spherical is None or irrep2_to_spherical.shape != (5, 5):
                raise ValueError(
                    "lambda=2 features need the (5, 5) irrep2_to_spherical change of basis; "
                    "build it with rsfff.ff.multipole.irrep2_to_spherical("
                    "backend.irrep6_to_voigt())"
                )
            self.register_buffer(
                "_to_spherical", irrep2_to_spherical.clone(), persistent=False
            )
        else:
            self._to_spherical = None

        self.n_invariants = n_state_invariants(k)
        # Layout: `[ h | eta | emb | state invariants ]`. The environment slot is contiguous
        # and everything after it is the tail -- the species embedding is a property of the
        # atom, and the state invariants already carry whatever environment reached them
        # through `vec_reduce_env` / `equiv_reduce_env`.
        self.net = zero_init_readout(
            two_slot_mlp(
                self.p0, self.p_env, hidden, depth, 1,
                p_tail=emb_dim + self.n_invariants,
            )
        )
        # The whole head, not only the zero-init readout: `vec_reduce` and `equiv_reduce` sit
        # behind it, so their gradient is proportional to zero on the first step and weight
        # decay would be the only force on them. See `rsfff.mlip.heads.zero_init_readout`.
        exempt_from_weight_decay(self)

    def state_invariants(
        self,
        vec_feats: torch.Tensor | None,
        equiv_feats: torch.Tensor | None,
        q: torch.Tensor,
        mu: torch.Tensor | None,
        quad_s: torch.Tensor | None,
        potential: torch.Tensor | None,
        field: torch.Tensor | None,
        field_gradient: torch.Tensor | None,
    ) -> torch.Tensor:
        """``(N, n_invariants)``. Exposed so tests can check invariance without the MLP."""
        return state_invariants(
            equiv_channels=self.equiv_channels,
            vec_feats=vec_feats,
            equiv_feats=equiv_feats,
            vec_reduce=(
                None if self.vec_reduce is None or vec_feats is None
                else slot_reduce(
                    self.vec_reduce, self.vec_reduce_env, vec_feats.shape[-1]
                )
            ),
            equiv_reduce=(
                None if self.equiv_reduce is None or equiv_feats is None
                else slot_reduce(
                    self.equiv_reduce, self.equiv_reduce_env, equiv_feats.shape[-1]
                )
            ),
            to_spherical=self._to_spherical,
            q=q, mu=mu, quad_s=quad_s,
            potential=potential, field=field, field_gradient=field_gradient,
        )

    def forward(
        self,
        inv_feats: torch.Tensor,             # (N, p0) isolated, or (N, p0 + p_env) joined
        species_idx: torch.Tensor,           # (N,)
        vec_feats: torch.Tensor | None,      # (N, 3, p1[+p1_env])
        equiv_feats: torch.Tensor | None,    # (N, 5, p2[+p2_env])
        q: torch.Tensor,                     # (N,)
        mu: torch.Tensor | None,             # (N, 3)
        quad_s: torch.Tensor | None,         # (N, 5)
        env=None,                            # OneBodyEnvironment | None
    ) -> torch.Tensor:
        """``(N,)`` Hartree.

        A narrow ``inv_feats`` -- width ``p0`` rather than ``p0 + p_env`` -- is the **isolated**
        evaluation, and every other block must be narrow to match. That is the one the whole
        one-body sector reads, and its value depends on no environment weight at all.
        """
        inv = self.state_invariants(
            vec_feats, equiv_feats, q, mu, quad_s,
            None if env is None else env.potential,
            None if env is None else env.field,
            None if env is None else env.field_gradient,
        )
        emb = self.species_emb(species_idx)
        x = torch.cat((inv_feats, emb, inv), dim=-1)
        return self.energy_scale * self.net(x).squeeze(-1)
