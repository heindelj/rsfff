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
    offset_scale    : scale of the per-element constant, Hartree. **0 disables it**, which
                      reproduces the model exactly as it was before the offset existed. See
                      :meth:`_species_offset`; the value is a conditioning choice, sized so a
                      physically plausible offset puts the parameter at order 1.
    offset_gate     : the ``||h||`` scale at which that constant is fully switched on,
                      Angstrom-free (it is a feature norm). Measured default; see
                      :meth:`_species_offset`.
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
        offset_scale: float = 1.0e-2,
        offset_gate: float = 0.5,
    ) -> None:
        super().__init__()
        if not energy_scale > 0.0:
            raise ValueError(f"AtomicStateEnergy needs energy_scale > 0, got {energy_scale}")
        if offset_scale < 0.0 or offset_gate <= 0.0:
            raise ValueError(
                f"AtomicStateEnergy needs offset_scale >= 0 and offset_gate > 0, got "
                f"{offset_scale}, {offset_gate}"
            )
        self.energy_scale = float(energy_scale)
        self.offset_scale = float(offset_scale)
        self.offset_gate = float(offset_gate)
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
        #: **The constant direction the anchoring takes away.** See :meth:`forward` for why it
        #: is needed and :attr:`offset_scale` for why it is scaled the way it is. Zero-init, so
        #: a model built with it is bit-identical to one built without until it trains, and an
        #: existing checkpoint warm starts unchanged.
        self.species_offset = nn.Parameter(torch.zeros(n_species))
        # The whole head, not only the zero-init readout: `vec_reduce` and `equiv_reduce` sit
        # behind it, so their gradient is proportional to zero on the first step and weight
        # decay would be the only force on them. See `rsfff.mlip.heads.zero_init_readout`.
        #
        # `species_offset` is exempt for a different and stronger reason: its decay attractor is
        # zero, which is precisely the offset it exists to remove. Decay here does not
        # regularize the parameter, it undoes it.
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
        return (
            self.energy_scale * (self.net(x) - self.net(ref)).squeeze(-1)
            + self._species_offset(inv_feats, species_idx)
        )

    def _species_offset(
        self,
        inv_feats: torch.Tensor,     # (N, p0)
        species_idx: torch.Tensor,   # (N,)
    ) -> torch.Tensor:
        """``(N,)`` a per-element constant that vanishes at the free atom.

        **What this repairs.** The anchoring above subtracts ``net(ref)``, and the readout's
        final-layer bias appears in *both* terms, so it cancels identically -- measured, its
        gradient is exactly ``0.000e+00``. That bias is the natural knob for "shift every
        atom's energy by the same amount", and the anchoring kills it. ``E0`` is a frozen
        buffer, and a uniform shift of ``chi`` does nothing either because ``sum_i q_i = 0``
        per fragment. So the model had **no** direction that moves the one-body constant
        without also reshaping its geometry dependence.

        That is not a capacity limit -- the MLP can express any constant -- it is a
        conditioning one, and it cost a real fit: on the last staged checkpoint the one-body
        residual was ``-5.23 kJ/mol`` per fragment, *constant* to 0.13 across the monomer
        anchor and w2--w5 alike. Removing that one number would have cut the total loss by 56%
        (``dL/dd = -117`` per kJ/mol), and nothing in the loss opposed it. The fit simply could
        not get there cheaply, and every other term was competing against the attempt.

        **The gate.** ``tanh(||h_i|| / offset_gate)`` is exactly zero when the invariant
        features are, which is the same condition the anchoring above uses for "free atom", so
        the two vanish together and the exact ``E0`` limit survives. It saturates rather than
        being used raw because ``||h||`` is not constant: measured over the monomer anchor and
        w2--w5, its coefficient of variation is 6-27%, so ``c[Z] * ||h||`` would inject
        shape dependence of the same size as the residual it is meant to remove. At
        ``offset_gate = 0.5`` the gate is ``>= 0.9974`` for every atom in that data (the
        smallest ``||h||`` is 1.664, on a hydrogen), which is 0.0045 kJ/mol of pollution on a
        1.74 kJ/mol offset -- 30x below the shape residual already present.

        Saturating also means the offset switches off only where ``||h||`` is genuinely
        vanishing, so a dissociating atom hands it back smoothly instead of stepping off it.

        **The scale is the point, not a detail, and it is a measured choice.** Adam's step is
        set by the learning rate and not by the gradient magnitude, so ``offset_scale`` fixes
        both how fast this coordinate can travel and how finely it can settle: a distance
        ``c_opt`` takes ``c_opt / lr`` steps, and the residual jitter in the energy is
        ``lr * offset_scale``. At 1e-2 Hartree a 1.74 kJ/mol per-atom offset sits at
        ``c ~ 0.067``, which at ``lr = 1e-3`` is ~150 steps -- about two epochs at batch 128 --
        and jitters by 0.026 kJ/mol, five times under the 0.125 kJ/mol of shape error already
        present. 1e-3 was tried first and is too slow (``c_opt ~ 0.66``, ~660 steps: measured,
        60 steps moved the residual only 5.22 -> 4.75); 1e-1 converges in ~7 steps but jitters
        by 0.26, twice the shape error.

        Measured end to end on the last staged checkpoint, training **this parameter alone**
        with every other weight frozen: the one-body loss fell 319.6 -> 2.1 and the residual
        went 5.22 -> 0.12 kJ/mol on the monomer anchor and on w2--w5 alike, with every EDA
        channel bitwise unchanged. That is the whole point -- it is a repair the rest of the
        model does not have to pay for.

        **What it frees up.** At that same checkpoint the constant was 99.7% of the one-body
        loss, and it was pushing 99.3% of that loss's gradient into ``net`` -- the same MLP
        that has to describe how the bond energy depends on geometry. Moving the offset to its
        optimum drops the one-body gradient on ``net`` from 4.0e5 to 2.9e3, a factor of 140.
        The shape dependence was already good (0.13 kJ/mol); nearly all of the pull on those
        weights was one number they could not cleanly reach.

        With one species present in the data the per-element split is underdetermined (water
        constrains ``2 c[H] + c[O]``, not each), so the two entries move together from their
        shared zero. That is not a defect: what the loss can see is the fragment total.

        A charged isolated atom is the one case where this does not vanish, because
        ``inv_feats`` carries the fragment-state block there. The anchoring above already has
        exactly that caveat (``ref`` zeroes the block while ``x`` does not), so the two agree
        on what "free atom" means, which is the property that matters.
        """
        if not self.offset_scale:
            return inv_feats.new_zeros(inv_feats.shape[0])
        gate = torch.tanh(inv_feats.norm(dim=-1) / self.offset_gate)
        return self.offset_scale * self.species_offset[species_idx] * gate


__all__ = ["AtomicStateEnergy"]
