"""``ApplicabilityHead``: does this fragmentation describe the system well?

What used to live here -- ``FragmentExpert`` and ``ExpertBank``, a per-composition network
owning its own parameter heads, and the dispatch that routed atoms to them -- is gone in v4.
The premise was that a fragment of a given composition deserves a dedicated network. The
premise was not wrong so much as *unmixable*: two dedicated networks share no input space, so
nothing defines what lies between an H3O+ and an H2O description, and every attempt to define
it (mixing parameters, then mixing latents) put the model somewhere neither network was ever
fit. Composition is now carried where it can be interpolated -- geometrically by ``h``, and as
a fractional element census in the fragment-state block (:mod:`rsfff.ff.fragment_state`) --
and one decoder (:mod:`rsfff.ff.decoder`) reads both.

The applicability score survives the deletion because it was never a parameterizer. It is a
*judgement about the fragmentation*, which is exactly the question a mixture asks, and it is
the one head here that must read the environment.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..mlip.heads import two_slot_mlp, zero_init_readout

__all__ = ["ApplicabilityHead"]


class ApplicabilityHead(nn.Module):
    """``v_f = V_s(pool_i [h_i | eta_i], Q_f, 2S_f)`` -- **both slots**.

    What this head claims, and why it needs ``eta``
    ----------------------------------------------
    An earlier version of this head read the fragment slot only, on the argument that "is
    this expert applicable to this fragment" is a property of the fragment, and that
    competition between fragmentations was a different question belonging to a separate
    router. That framing was wrong about what the quantity is *for*.

    The question being asked is: **given this fragment and its surroundings, is this
    decomposition the best available description of the system?** That is inherently a
    statement about competition -- an H2O with a proton 1.0 Angstrom away is a perfectly good
    water and a bad choice of fragmentation, and nothing inside ``h`` can say so. So the head
    reads the joined descriptor, and excluding ``eta`` would remove exactly the information
    the question turns on.

    What "best" means, and where the label comes from
    ------------------------------------------------
    The best decomposition is the one needing the **smallest perturbation of the reference
    fragments**, and ALMO-EDA measures that directly: ``E_pol + E_ct`` is the relaxation from
    frozen, non-interacting monomers to the true wavefunction. Over the 399 H3O+/OH-
    microsolvation frames in ``data/wb97mv_tzvpd``, ``argmin |E_pol + E_ct|`` picks the
    chemically obvious assignment in 398 of them -- against 395 for ``|E_int|`` and 368 for
    ``|E_frz|``, so it is the *induction* magnitude that discriminates, not the interaction
    strength. The best-versus-second gap averages 165-476 kJ/mol, which is why a score
    trained against it can be sharp rather than marginal.

    The training term lives in :func:`rsfff.train.train_expert.applicability_loss`, which
    softmaxes these scores across the fragmentations of one geometry. So the *scale* of a
    score is arbitrary and only differences within a geometry are meaningful -- this head
    emits a raw score and never a weight, and pooling it across a frame is the model's job,
    not this class's.

    Scores start at exactly zero (the readout is zero-initialized), i.e. a uniform preference
    over fragmentations, which is the right thing to know nothing from.
    """

    def __init__(
        self, p0: int, p_env: int = 0, *, hidden: int = 32, depth: int = 2
    ) -> None:
        super().__init__()
        self.p0 = int(p0)
        self.p_env = int(p_env)
        # `p_tail=2` for (Q_f, 2S_f), concatenated after the pooled features so the two slots
        # stay separable in the first layer. `p_env=0` rebuilds the single-slot form.
        self.net = zero_init_readout(
            two_slot_mlp(self.p0, self.p_env, hidden, depth, 1, p_tail=2)
        )

    @property
    def width(self) -> int:
        """The descriptor width this head expects: joined when it has an environment slot."""
        return self.p0 + self.p_env

    def forward(
        self,
        inv_feats: torch.Tensor,       # (N, p0 + p_env) -- the JOINED slot
        fragment_idx: torch.Tensor,    # (N,) global fragment id per atom
        n_fragments: int,
        fragment_charge: torch.Tensor | None,   # (F,)
        fragment_two_s: torch.Tensor | None,    # (F,)
    ) -> torch.Tensor:
        """``(F,)`` a raw score per fragment. Higher is a better decomposition.

        ``fragment_idx`` stays in *global* numbering even when only some atoms are passed --
        the fragment-expert model hands in one expert's atoms at a time and keeps the rows it
        asked for. Fragments with no atoms here pool to zero and are discarded by the caller.
        """
        width = self.width
        if inv_feats.shape[-1] != width:
            raise ValueError(
                f"ApplicabilityHead got features of width {inv_feats.shape[-1]}, expected "
                f"{width} ({self.p0} fragment + {self.p_env} environment). It reads the "
                f"joined descriptor -- pass SlotFeatures.joined(), not .isolated(); see this "
                f"class's docstring for why the environment slot is not optional here."
            )
        pooled = inv_feats.new_zeros(n_fragments, width).index_add_(
            0, fragment_idx, inv_feats
        )
        counts = inv_feats.new_zeros(n_fragments).index_add_(
            0, fragment_idx, torch.ones_like(fragment_idx, dtype=inv_feats.dtype)
        )
        pooled = pooled / counts.clamp(min=1.0).unsqueeze(-1)
        zeros = pooled.new_zeros(n_fragments)
        q = zeros if fragment_charge is None else fragment_charge.to(pooled.dtype)
        s = zeros if fragment_two_s is None else fragment_two_s.to(pooled.dtype)
        return self.net(torch.cat((pooled, q.unsqueeze(-1), s.unsqueeze(-1)), dim=-1)).squeeze(-1)
