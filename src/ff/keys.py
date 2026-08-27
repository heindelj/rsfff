"""Reaction keys: one equivariant latent per atom, and the space mixtures happen in.

``docs/fff_v2.md`` v3. The v2 mediator mixed **parameters** -- ``C6``, ``r0``, the Pauli
multipoles -- on the argument that both experts emit the same physical number. The argument is
true and insufficient: the classical forms are strongly nonlinear in those numbers (a geometric
mean under a square root, a Fermi gate on ``r0``, Tang-Toennies damping), so a halfway parameter
set does not give a halfway energy. Measured on the proton-transfer scans, the mediated energy
left the interval spanned by the two vertices by 162 kJ/mol (H5O2+ total) and 181 kJ/mol (H3O2-
electrostatics), and the one quantity that was mixed at the **output** -- ``E_bond`` -- came in
at 2.4 kJ/mol while sweeping 60. Where the mixing happens is the whole story.

So mixing moves here::

    h_i, eta_i  --K_s (per composition)-->  k_i        k_i^0 = K_s(h_i, 0)      ||k|| = 1
    k_i         --D  (global, element-keyed)-->  every parameter, and E_bond

A convex combination of keys decodes through **one shared decoder** to a *self-consistent*
parameter set -- one ``D`` would emit for some real input -- rather than to an average of two
incompatible ones. And unlike an arithmetic average, the path through the crossover is a
function of trainable weights, so ``E_total`` can actually shape it. That is the point: v2's
crossover was fixed arithmetic and training could not repair it.

The isolated-fragment guarantee is untouched
--------------------------------------------
``theta_0 = D(K_s(h, 0))``. ``eta`` is identically zero for an isolated fragment, ``K_s``'s
environment weights are a named, zero-initialized sector, and the one-body sector reads only
``k^0``. §4 survives verbatim; only the address moves. **The ``L_env`` penalty does not move
with it** -- it stays in parameter space, on ``D(k) - D(k_0)``, because §4 rejected a
feature-space norm as having no defensible weight and a key-space norm has exactly the same
problem.

Why the key must be equivariant
-------------------------------
The dipole, quadrupole and polarizability heads read lambda=1 and lambda=2 blocks. A scalar key
cannot feed them, and falling back to per-expert equivariant heads would put those quantities
straight back into output mixing. A convex combination of an equivariant object is equivariant,
so mixing stays safe. :class:`rsfff.mlip.mixture.MixtureModel` already blends
``inv``/``vec``/``equiv`` for exactly this reason; this follows that precedent rather than
rediscovering it.

Normalization
-------------
Keys are divided by their **invariant** norm -- one scalar for the whole bundle, summed over
every lambda and every ``m`` component, so dividing preserves equivariance. One scalar and not
one per lambda: a per-lambda normalization would discard the *relative* magnitude of the
lambda=1 and lambda=2 blocks, which is what says how anisotropic the atom is.

It removes the scale degeneracy between encoder and decoder (nothing otherwise stops ``K``
inflating ``k`` while ``D`` divides it back out), bounds the decoder's input, and makes the
interpolation path an arc of fixed radius rather than a chord through a space of arbitrary
scale.

**Mixtures are renormalized**, and that is not tidiness: unit-norm keys combined convexly land
on a chord with norm < 1, a magnitude the encoder never produces, so the decoder would be
evaluated off the manifold it was trained on. That is the same off-manifold failure this whole
design exists to escape.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn as nn

from ..mlip.heads import (
    env_reduce_parameter,
    exempt_from_weight_decay,
    slot_reduce,
    two_slot_mlp,
)

__all__ = [
    "AtomKeyEncoder",
    "KeyBundle",
    "bundle_tuple",
    "key_angle",
    "key_features",
    "mix_keys",
]

#: Guards the normalization denominator. A key of exactly zero norm is unreachable in practice
#: -- it needs every channel of every lambda to vanish at once -- but the *mixed* key can
#: approach it if two candidate keys are near-antipodal, which is the one geometry this
#: construction handles badly. :func:`key_angle` is the diagnostic for that.
_EPS = 1.0e-12


@dataclass
class KeyBundle:
    """One equivariant latent per atom: ``(k0, k1, k2)``, unit norm unless said otherwise.

    The shapes mirror :class:`rsfff.features.features.LambdaFeatures` exactly, so every
    existing parameter head consumes a bundle without modification -- which is what lets the
    decoder be the *same* heads at different widths rather than a rewrite of all of them.
    """

    k0: torch.Tensor                    # (N, K0)   invariant
    k1: torch.Tensor | None = None      # (N, 3, K1) lambda=1
    k2: torch.Tensor | None = None      # (N, 5, K2) lambda=2

    def norm(self) -> torch.Tensor:
        """``(N,)`` the invariant norm: every lambda, every ``m`` component, one scalar."""
        total = self.k0.pow(2).sum(-1)
        for block in (self.k1, self.k2):
            if block is not None:
                total = total + block.pow(2).sum((-2, -1))
        return total.clamp(min=_EPS).sqrt()

    def normalized(self) -> "KeyBundle":
        """This bundle divided by :meth:`norm`. Equivariant: the divisor is a scalar."""
        n = self.norm()
        return KeyBundle(
            k0=self.k0 / n.unsqueeze(-1),
            k1=None if self.k1 is None else self.k1 / n.reshape(-1, 1, 1),
            k2=None if self.k2 is None else self.k2 / n.reshape(-1, 1, 1),
        )

    def select(self, index) -> "KeyBundle":
        """The rows ``index`` names. ``None`` means all of them, as everywhere else here."""
        if index is None:
            return self
        return KeyBundle(
            k0=self.k0[index],
            k1=None if self.k1 is None else self.k1[index],
            k2=None if self.k2 is None else self.k2[index],
        )

    @property
    def widths(self) -> tuple[int, int | None, int | None]:
        return (
            int(self.k0.shape[-1]),
            None if self.k1 is None else int(self.k1.shape[-1]),
            None if self.k2 is None else int(self.k2.shape[-1]),
        )


def mix_keys(bundles, weights: torch.Tensor) -> KeyBundle:
    """``normalize(sum_m w_m k_m)`` -- the convex combination, back on the sphere.

    **Vertex identity holds to floating point, not to the bit.** At a one-hot membership the
    sum *is* ``k_m``, which already has unit norm, so renormalizing is mathematically the
    identity -- but it recomputes the norm from already-normalized components and that sum of
    squares is only 1.0 to rounding, so the last bit or two moves. The energy-level statement
    that matters (Invariant 1: the mixture reproduces the single-fragmentation model) holds at
    1e-9, well inside anything physical. ``tests/test_mediator.py`` pins both.

    Deliberately linear, and deliberately not a network. A learned interpolator can represent
    anything, including the pathology this replaces, and §10 already argues that a correction
    which vanishes at both vertices is unidentifiable until the linear mixture is carrying the
    crossover. Fit the linear path, measure what ``E_total`` still misses, and only then decide.
    """
    if not bundles:
        raise ValueError("mix_keys needs at least one bundle")
    w = weights.reshape(-1)
    if w.shape[0] != len(bundles):
        raise ValueError(
            f"mix_keys got {len(bundles)} bundles and {w.shape[0]} weights; the membership is "
            f"a partition of unity over the decompositions and must have one entry per bundle"
        )

    def blend(name, extra_dims):
        first = getattr(bundles[0], name)
        if first is None:
            return None
        stacked = torch.stack([getattr(b, name) for b in bundles])
        return (w.reshape(-1, *([1] * (1 + extra_dims))) * stacked).sum(0)

    return KeyBundle(
        k0=blend("k0", 1), k1=blend("k1", 2), k2=blend("k2", 2)
    ).normalized()


def key_angle(a: KeyBundle, b: KeyBundle) -> torch.Tensor:
    """``(N,)`` the angle in radians between two unit keys, per atom.

    Two uses, both diagnostic. It is the natural **reaction coordinate** along a mixture --
    ``angle(k_A, k) / angle(k_A, k_B)`` runs 0 to 1 along the arc -- and it is the early warning
    for the one geometry :func:`mix_keys` handles badly: candidate keys approaching antipodal
    make the normalized path swing arbitrarily fast near the midpoint.
    """
    dot = (a.k0 * b.k0).sum(-1)
    for x, y in ((a.k1, b.k1), (a.k2, b.k2)):
        if x is not None and y is not None:
            dot = dot + (x * y).sum((-2, -1))
    return dot.clamp(-1.0, 1.0).arccos()


class AtomKeyEncoder(nn.Module):
    """``(h, eta) -> k``, one per fragment composition. The chemistry still lives here.

    §2's thesis is unchanged: a fragment of a given composition is described by a network
    dedicated to that composition. What moved is where that network *ends* -- it now emits a
    key rather than a parameter set, and a single shared decoder turns keys into physics. That
    is what makes two experts' outputs commensurable: they are commensurable **because** both
    must decode through the same ``D``, so training forces them into one frame. The gauge is
    enforced, not assumed. It is exactly why the diabatic mixture could blend features at all.

    Layout, matching what the model already builds::

        inv = [ h | state | eta ]      fragment slot first, with the (Q_f, 2S_f) block
        vec = [ v_frag | v_env ]
        equ = [ e_frag | e_env ]

    so the narrow-input convention gives the isolated evaluation for free: pass
    ``SlotFeatures.isolated()`` and every environment term drops, exactly as
    :class:`rsfff.mlip.heads.TwoSlotLinear` and :func:`rsfff.mlip.heads.slot_reduce` already
    do for the parameter heads.

    **The gates are deliberately *not* zero-initialized**, unlike every parameter readout in
    this package. A zero readout is right for a quantity that should start at zero -- a
    permanent dipole -- and wrong for a latent: it would make ``k1`` and ``k2`` identically
    zero, give ``equiv_reduce`` no gradient, and deadlock the pair in precisely the way
    :func:`rsfff.mlip.heads.zero_init_readout` documents. The whole module is exempt from
    weight decay for the same reason the equivariant heads are.
    """

    def __init__(
        self,
        p0_frag: int,
        p0_env: int,
        p1_frag: int | None,
        p1_env: int,
        p2_frag: int | None,
        p2_env: int,
        n_species: int,
        *,
        k0: int = 64,
        k1: int = 32,
        k2: int = 32,
        emb_dim: int = 16,
        hidden: int = 64,
        depth: int = 2,
    ) -> None:
        super().__init__()
        self.k0_dim = int(k0)
        self.k1_dim = int(k1) if p1_frag else None
        self.k2_dim = int(k2) if p2_frag else None
        self.species_emb = nn.Embedding(n_species, emb_dim)

        self.inv_mlp = two_slot_mlp(
            p0_frag, p0_env, hidden, depth, self.k0_dim, p_tail=emb_dim
        )

        self.reduce1 = self.reduce1_env = self.gate1 = None
        if self.k1_dim:
            self.reduce1 = nn.Parameter(
                torch.randn(p1_frag, self.k1_dim) / (p1_frag ** 0.5)
            )
            self.reduce1_env = env_reduce_parameter(p1_env, self.k1_dim)
            self.gate1 = two_slot_mlp(
                p0_frag, p0_env, hidden, depth, self.k1_dim, p_tail=emb_dim
            )

        self.reduce2 = self.reduce2_env = self.gate2 = None
        if self.k2_dim:
            self.reduce2 = nn.Parameter(
                torch.randn(p2_frag, self.k2_dim) / (p2_frag ** 0.5)
            )
            self.reduce2_env = env_reduce_parameter(p2_env, self.k2_dim)
            self.gate2 = two_slot_mlp(
                p0_frag, p0_env, hidden, depth, self.k2_dim, p_tail=emb_dim
            )

        # See the class docstring: exempt, but *not* zero-initialized.
        exempt_from_weight_decay(self)

    @property
    def key_dims(self) -> tuple[int, int | None, int | None]:
        return self.k0_dim, self.k1_dim, self.k2_dim

    def forward(self, feats) -> KeyBundle:
        """``KeyBundle`` from a :class:`rsfff.features.features.LambdaFeatures`-shaped object.

        Hand it ``SlotFeatures.joined()`` for ``k`` and ``SlotFeatures.isolated()`` for
        ``k_0``; the widths select which, with no zero-padding and no flag.
        """
        emb = self.species_emb(feats.species_idx)
        x = torch.cat((feats.inv_feats, emb), dim=-1)
        bundle = KeyBundle(
            k0=self.inv_mlp(x),
            k1=self._equivariant(feats.vec_feats, self.reduce1, self.reduce1_env, self.gate1, x),
            k2=self._equivariant(feats.equiv_feats, self.reduce2, self.reduce2_env, self.gate2, x),
        )
        return bundle.normalized()

    @staticmethod
    def _equivariant(block, reduce, reduce_env, gate, x):
        """``(N, m, K)`` -- a channel reduction of an equivariant block, invariantly gated.

        The gate is a function of the invariants only, so multiplying by it leaves the ``m``
        axis alone and the result transforms exactly as ``block`` does.
        """
        if block is None or reduce is None:
            return None
        matrix = slot_reduce(reduce, reduce_env, block.shape[-1])
        reduced = torch.einsum("nmp,pk->nmk", block, matrix)
        return reduced * gate(x).unsqueeze(1)


def bundle_tuple(key: KeyBundle) -> tuple:
    """``(k0, k1, k2)``. For :func:`rsfff.ff.expert_model._stitch`, which walks tuples."""
    return (key.k0, key.k1, key.k2)


def key_features(key: KeyBundle, species_idx: torch.Tensor, batch_idx: torch.Tensor):
    """A :class:`~rsfff.features.features.LambdaFeatures` view of a key.

    The parameter heads were written against ``LambdaFeatures`` and are reused verbatim at key
    widths (:mod:`rsfff.ff.decoder`), so the key has to arrive wearing that shape. This is the
    adapter, and it is the only place the two vocabularies meet.
    """
    from ..features.features import LambdaFeatures

    return LambdaFeatures(
        inv_feats=key.k0,
        equiv_feats=key.k2,
        species_idx=species_idx,
        batch_idx=batch_idx,
        vec_feats=key.k1,
    )
