"""The per-atom energy of the self-consistent electronic state.

``docs/atomic_response_functional.md`` in one sentence: an atom's energy should depend on the
multipoles it actually carries, not only on where its neighbours are. This module is that
dependence::

    E_i^atom = E_theta( h_i , M_i , Phi_i )      M_i = (q_i, mu_i, Theta_i)
                                                 Phi_i = (V_i, E_i, grad E_i)

What it replaces
----------------
The bond channel of :class:`rsfff.mlip.unified_head.UnifiedPairHead`. That readout was carrying
three jobs at once -- the covalent bond energy, the only free knob left inside
``fragment_energy`` once ``freeze_frozen_level`` pins ``E_internal``, and (through the
``bond_pol``/``bond_ct`` telescoping) the whole of ``eda_pol`` and ``eda_ct``. Two of those are
unlabeled, and the measured result was a **constant −1.686 kJ/mol per fragment** in
``fragment_energy``, identical to three decimals across w2--w5, plus a charge-transfer channel
that was 96.7% neural. Charge transfer expressed as "the bond energy changed" is not wrong
physics; routing it through a readout that also defines the one-body zero is.

Here the three levels are the *same weights* evaluated at three different electronic states, so
what separates them is the state and not a parameter::

    E_atom^0   = E_theta( h_frag , M^frozen , Phi^intra )    -> fragment_energy
    E_atom^pol = E_theta( h_frag , M^pol    , Phi^pol   )    -> eda_pol gets (pol - 0)
    E_atom^ct  = E_theta( h_env  , M^ct     , Phi^ct    )    -> eda_ct  gets (ct  - pol)

``Phi^intra`` is built from **intra-fragment pairs only**. That is what keeps ``E_atom^0`` an
isolated-fragment quantity: the field an atom feels from its own molecule is part of what its
bonds are worth, the field from a neighbouring molecule is not. The bond channel could not make
that distinction -- it was handed ``phi = 0`` at the frozen level and so was blind to the
intramolecular field as well -- and recovering it is most of the point of this head.

Rotation invariance
-------------------
``M`` and ``Phi`` are not scalars, so they cannot be concatenated onto ``inv_feats`` raw. Every
one of them is reduced to an invariant first, and there are two kinds:

* **self-contractions** of the electronic state, ``mu.mu``, ``mu.E``, ``Theta:grad E`` and so
  on. These see the *magnitude* of the state but nothing about its orientation in the molecule.
* **contractions against the geometric features**, ``mu . v_k`` and ``Theta : G_k``, where
  ``v_k`` and ``G_k`` are learned channel reductions of the lambda=1 and lambda=2 features.
  These are what make the head more than a function of scalars: they see which way the dipole
  points *relative to the local geometry*, which for water is most of the physics.

The lambda=2 contractions are done in **Cartesian** form via
:func:`~rsfff.ff.multipole.spherical_to_cartesian_quadrupole`, not by dotting the five
spherical components together. ``A_ab B_ab`` is manifestly invariant whatever normalization the
five-component convention uses, whereas a naive 5-vector dot product is invariant only if that
basis happens to be orthonormal -- exactly the class of silent equivariance break that
:func:`~rsfff.ff.multipole.irrep2_to_spherical` documents.

Zero-fill rather than a variable width
--------------------------------------
Slots for absent quantities are filled with zeros instead of being dropped, so the input width
does not depend on ``max_rank`` or on which response channels are switched on, and a checkpoint
written by one configuration still loads into another. Same convention, and same reason, as
:func:`rsfff.ff.environment.environment_pair_invariants`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..mlip.heads import exempt_from_weight_decay, mlp, zero_init_readout
from .multipole import spherical_to_cartesian_quadrupole


class AtomicStateEnergy(nn.Module):
    """``(N,)`` Hartree from the geometric features and the electronic state.

    Args
    ----
    p0, p1, p2      : widths of the lambda=0/1/2 feature blocks. ``p1``/``p2`` may be ``None``,
                      in which case the corresponding feature contractions are dropped (their
                      slots are still present, filled with zeros).
    n_species       : embedding table size.
    irrep2_to_spherical : ``(5, 5)`` change of basis from the backend's lambda=2 slots to the
                      spherical convention, from
                      :func:`rsfff.ff.multipole.irrep2_to_spherical`. Required whenever ``p2``
                      is given, for the reason that function's docstring records.
    equiv_channels  : how many channels the lambda=1 and lambda=2 features are reduced to
                      before contracting against the electronic state.
    energy_scale    : output scale in Hartree. Size it against the thing being described --
                      a covalent bond, so ~0.2 Ha -- not against the interaction corrections.
    """

    def __init__(
        self,
        p0: int,
        n_species: int,
        *,
        p1: int | None = None,
        p2: int | None = None,
        irrep2_to_spherical: torch.Tensor | None = None,
        emb_dim: int = 16,
        hidden: int = 64,
        depth: int = 2,
        equiv_channels: int = 8,
        energy_scale: float = 0.2,
    ) -> None:
        super().__init__()
        if not energy_scale > 0.0:
            raise ValueError(f"AtomicStateEnergy needs energy_scale > 0, got {energy_scale}")
        self.energy_scale = float(energy_scale)
        self.equiv_channels = int(equiv_channels)
        self.species_emb = nn.Embedding(n_species, emb_dim)

        k = self.equiv_channels
        self.vec_reduce = (
            None if p1 is None
            else nn.Parameter(torch.randn(p1, k) / (p1 ** 0.5))
        )
        self.equiv_reduce = (
            None if p2 is None
            else nn.Parameter(torch.randn(p2, k) / (p2 ** 0.5))
        )
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

        # 2 rank-0 (q, V) + 3 rank-1 self + 3 rank-2 self + 2K rank-1 cross + 2K rank-2 cross.
        # Fixed regardless of which pieces are actually live; the dead ones read zero.
        self.n_invariants = 8 + 4 * k
        self.net = zero_init_readout(
            mlp(p0 + emb_dim + self.n_invariants, hidden, depth, 1)
        )
        # The whole head, not only the zero-init readout: `vec_reduce` and `equiv_reduce` sit
        # behind it, so their gradient is proportional to zero on the first step and weight
        # decay would be the only force on them. See `rsfff.mlip.heads.zero_init_readout`.
        exempt_from_weight_decay(self)

    def state_invariants(
        self,
        vec_feats: torch.Tensor | None,      # (N, 3, p1)
        equiv_feats: torch.Tensor | None,    # (N, 5, p2)
        q: torch.Tensor,                     # (N,)
        mu: torch.Tensor | None,             # (N, 3)
        quad_s: torch.Tensor | None,         # (N, 5) spherical
        potential: torch.Tensor | None,      # (N,)
        field: torch.Tensor | None,          # (N, 3)
        field_gradient: torch.Tensor | None,  # (N, 5) spherical
    ) -> torch.Tensor:
        """``(N, n_invariants)``. Split out so tests can check invariance without the MLP."""
        n = q.shape[0]
        zero = q.new_zeros(n)
        k = self.equiv_channels

        cols = [q, zero if potential is None else potential]

        # -- rank 1 ------------------------------------------------------------------------
        d = q.new_zeros(n, 3) if mu is None else mu
        f = q.new_zeros(n, 3) if field is None else field
        cols += [(d * d).sum(-1), (f * f).sum(-1), (d * f).sum(-1)]

        # -- rank 2, contracted in Cartesian form ------------------------------------------
        t = q.new_zeros(n, 3, 3) if quad_s is None else spherical_to_cartesian_quadrupole(quad_s)
        g = (
            q.new_zeros(n, 3, 3) if field_gradient is None
            else spherical_to_cartesian_quadrupole(field_gradient)
        )
        cols += [
            (t * t).sum((-2, -1)),
            (g * g).sum((-2, -1)),
            (t * g).sum((-2, -1)),
        ]

        # -- rank 1 against the geometric features -----------------------------------------
        if self.vec_reduce is None or vec_feats is None:
            cols += [q.new_zeros(n, 2 * k)]
        else:
            v_k = torch.einsum("nmp,pk->nmk", vec_feats, self.vec_reduce)     # (N, 3, K)
            cols += [
                torch.einsum("nm,nmk->nk", d, v_k),
                torch.einsum("nm,nmk->nk", f, v_k),
            ]

        # -- rank 2 against the geometric features -----------------------------------------
        if self.equiv_reduce is None or equiv_feats is None:
            cols += [q.new_zeros(n, 2 * k)]
        else:
            e_k = torch.einsum("nmp,pk->nmk", equiv_feats, self.equiv_reduce)  # (N, 5, K)
            # (N, K, 5) in the backend basis -> spherical -> Cartesian, so the contraction
            # below is a genuine tensor double-dot rather than a basis-dependent dot product.
            g_k = spherical_to_cartesian_quadrupole(
                e_k.transpose(1, 2) @ self._to_spherical
            )                                                                  # (N, K, 3, 3)
            cols += [
                torch.einsum("nab,nkab->nk", t, g_k),
                torch.einsum("nab,nkab->nk", g, g_k),
            ]

        return torch.cat([c.reshape(n, -1) for c in cols], dim=-1)

    def forward(
        self,
        inv_feats: torch.Tensor,             # (N, p0)
        species_idx: torch.Tensor,           # (N,)
        vec_feats: torch.Tensor | None,      # (N, 3, p1)
        equiv_feats: torch.Tensor | None,    # (N, 5, p2)
        q: torch.Tensor,                     # (N,)
        mu: torch.Tensor | None,             # (N, 3)
        quad_s: torch.Tensor | None,         # (N, 5)
        env=None,                            # OneBodyEnvironment | None
    ) -> torch.Tensor:
        inv = self.state_invariants(
            vec_feats, equiv_feats, q, mu, quad_s,
            None if env is None else env.potential,
            None if env is None else env.field,
            None if env is None else env.field_gradient,
        )
        emb = self.species_emb(species_idx)
        x = torch.cat((inv_feats, emb, inv), dim=-1)
        # **Anchored at the free atom**, the same device `EnvironmentResidual` uses for `g`.
        # A lone neutral atom has an all-zero SOAP density and no multipoles, so every input
        # except the species embedding is exactly zero -- and `net` evaluated there is a pure
        # per-element constant that would shift the isolated-atom energy away from `E0`.
        # `E0` is a *measured* isolated-atom energy and the free-atom limit was deliberately
        # made exact; subtracting the reference keeps it exact at every point in training
        # rather than only at initialization, where the zero-init readout gives it for free.
        ref = torch.cat((torch.zeros_like(inv_feats), emb, torch.zeros_like(inv)), dim=-1)
        return self.energy_scale * (self.net(x) - self.net(ref)).squeeze(-1)


__all__ = ["AtomicStateEnergy"]
