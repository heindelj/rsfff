"""The two-slot feature contract: a fragment descriptor and an environment descriptor.

``docs/fff_v2.md`` §3-4 in code. Every parameterizer in this model takes exactly two inputs::

    h_i    lambda-SOAP over edges with  frag(i) == frag(j)      the fragment slot
    eta_i  lambda-SOAP over edges with  frag(i) != frag(j)      the environment slot

and emits two evaluations of every quantity::

    theta   = P(h, eta)      the in-medium parameter
    theta_0 = P(h,  0 )      the isolated-fragment parameter

This module owns the layout that makes those two calls possible without every head knowing
about slots: the joined descriptor is the two blocks **concatenated, fragment first**, and the
isolated descriptor is the fragment block on its own.

Why concatenation and not a residual
------------------------------------
The model this replaces carried one stream, ``h_env = h_frag + g(h_full) - g(h_frag)``: an
environment residual summed into the fragment vector and anchored so that it vanished for an
isolated fragment. The anchoring was exact, so the *limit* was right -- what was wrong was that
"isolated" became a property of an arithmetic identity between two evaluations of a network
rather than a property of the input. Nothing could inspect the environment on its own, and
"the parameter this fragment would have alone" meant knowing which of two vectors to pass,
which is a convention that has to be enforced by hand, in several places, consistently. It was
not.

Here ``eta`` is identically zero for an isolated fragment because the sum that builds it is
empty (:meth:`rsfff.features.features.FlatLambdaSOAPFeaturizer.forward`, ``also_cross=True``).
There is nothing to enforce and nothing to get wrong.

The narrow-input convention
---------------------------
:meth:`SlotFeatures.isolated` returns the fragment block **unpadded**, and
:class:`rsfff.mlip.heads.TwoSlotLinear` reads a narrow input as "drop the environment term".
That is bit-identical to padding with zeros -- it adds an exact ``0.0`` either way -- and is
preferred for two reasons: it allocates nothing on a path taken twice per forward, and it makes
``theta_0`` visibly a function of ``h`` alone rather than of a zero the caller has to remember.

``env is None`` disables the environment slot entirely: ``joined() is isolated() is frag``,
every ``p_env`` is 0, every head builds its single-slot form, and the model is rigorously
two-body. That is the ablation, and it is also what the fragment-view training stream runs as
when a batch happens to contain no cross-fragment edges at all.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from ..features.features import LambdaFeatures

__all__ = ["SlotDims", "SlotFeatures", "select_atoms"]


def select_atoms(feats: LambdaFeatures | None, atom_index) -> LambdaFeatures | None:
    """A view of ``feats`` holding only ``atom_index``'s rows, per lambda.

    ``atom_index=None`` means "all of them" and returns ``feats`` untouched, which is what
    keeps the single-expert path in :class:`rsfff.ff.expert_model.FragmentExpertModel`
    allocation-free.

    ``edge_index`` is dropped rather than renumbered: it addresses the full batch, and a
    silently stale copy is worse than an absent one. Nothing that consumes a subset reads it.
    """
    if feats is None or atom_index is None:
        return feats
    return replace(
        feats,
        inv_feats=feats.inv_feats[atom_index],
        equiv_feats=(
            None if feats.equiv_feats is None else feats.equiv_feats[atom_index]
        ),
        vec_feats=None if feats.vec_feats is None else feats.vec_feats[atom_index],
        species_idx=feats.species_idx[atom_index],
        batch_idx=feats.batch_idx[atom_index],
        edge_index=None,
    )


@dataclass(frozen=True)
class SlotDims:
    """Per-lambda widths of the two slots. ``*_env`` is 0 when the slot is off.

    Heads take these to size themselves: ``two_slot_mlp(p0_frag, p0_env, ...)`` builds its
    single-slot form at ``p0_env == 0``, which is what keeps a v1 checkpoint loadable.
    """

    p0_frag: int
    p0_env: int
    p1_frag: int | None = None
    p1_env: int = 0
    p2_frag: int | None = None
    p2_env: int = 0

    @property
    def p0(self) -> int:
        return self.p0_frag + self.p0_env

    @property
    def p1(self) -> int | None:
        return None if self.p1_frag is None else self.p1_frag + self.p1_env

    @property
    def p2(self) -> int | None:
        return None if self.p2_frag is None else self.p2_frag + self.p2_env

    @property
    def has_env(self) -> bool:
        return bool(self.p0_env)


def _cat(a: torch.Tensor | None, b: torch.Tensor | None) -> torch.Tensor | None:
    if a is None or b is None:
        return a
    return torch.cat((a, b), dim=-1)


class SlotFeatures:
    """``(h, eta)`` for one batch, with the joined and isolated views heads actually consume.

    Construct from the featurizer's ``also_cross=True`` pair, after any per-atom augmentation
    (the fragment-state block) has been applied to the **fragment** slot -- charge and
    multiplicity are properties of the fragment, not of its surroundings.
    """

    def __init__(self, frag: LambdaFeatures, env: LambdaFeatures | None = None) -> None:
        if env is not None:
            for name in ("inv_feats", "vec_feats", "equiv_feats"):
                a, b = getattr(frag, name), getattr(env, name)
                if (a is None) != (b is None):
                    raise ValueError(
                        f"the two slots disagree about whether {name} exists "
                        f"({'set' if a is not None else 'None'} vs "
                        f"{'set' if b is not None else 'None'}); they come from one featurizer "
                        f"call and must carry the same lambdas"
                    )
                if a is not None and a.shape[:-1] != b.shape[:-1]:
                    raise ValueError(
                        f"{name} shapes {tuple(a.shape)} and {tuple(b.shape)} differ outside "
                        f"the channel axis"
                    )
        self.frag = frag
        self.env = env
        self._joined: LambdaFeatures | None = None

    @property
    def dims(self) -> SlotDims:
        def width(feats, name):
            block = getattr(feats, name)
            return None if block is None else int(block.shape[-1])

        return SlotDims(
            p0_frag=int(self.frag.inv_feats.shape[-1]),
            p0_env=0 if self.env is None else int(self.env.inv_feats.shape[-1]),
            p1_frag=width(self.frag, "vec_feats"),
            p1_env=0 if self.env is None else (width(self.env, "vec_feats") or 0),
            p2_frag=width(self.frag, "equiv_feats"),
            p2_env=0 if self.env is None else (width(self.env, "equiv_feats") or 0),
        )

    def isolated(self) -> LambdaFeatures:
        """``h`` alone. What the whole one-body sector reads, and nothing else may widen it."""
        return self.frag

    def joined(self) -> LambdaFeatures:
        """``[h | eta]`` per lambda, fragment first. Cached: two per forward is the intent."""
        if self.env is None:
            return self.frag
        if self._joined is None:
            self._joined = replace(
                self.frag,
                inv_feats=_cat(self.frag.inv_feats, self.env.inv_feats),
                vec_feats=_cat(self.frag.vec_feats, self.env.vec_feats),
                equiv_feats=_cat(self.frag.equiv_feats, self.env.equiv_feats),
            )
        return self._joined

    def env_norm(self) -> torch.Tensor:
        """``(N,) ||eta||`` per atom: how much environment each atom is being handed.

        Zero for an isolated fragment by construction, so a *rising* value is the model asking
        for many-body content and an exactly-zero one on a cluster is a bug, not a preference.
        Compare ``EnvironmentResidual``'s norm, which was zero at initialization and had to be
        coaxed off it; this one starts at whatever the geometry says and never lies about it.
        """
        if self.env is None:
            return self.frag.inv_feats.new_zeros(self.frag.inv_feats.shape[0])
        return self.env.inv_feats.norm(dim=-1)

    def __repr__(self) -> str:
        d = self.dims
        return f"SlotFeatures(p0={d.p0_frag}+{d.p0_env}, p1={d.p1_frag}+{d.p1_env}, p2={d.p2_frag}+{d.p2_env})"
