"""The per-atom ``(Q_f, 2S_f)`` block: which member of its family a fragment is.

Split out of :mod:`rsfff.ff.unified` for the fragment-expert model, where it has a sharper job
than it had before. Experts are keyed on **composition** alone (:mod:`rsfff.ff.expert`), so one
``"OH"`` expert covers hydroxide and the OH radical -- and this is what tells it which one it is
looking at. Without it those two are indistinguishable, and an H2O and an H3O+ differ precisely
in which fragment carries the charge.

It joins the **fragment** slot, never the environment one: charge and multiplicity are
properties of the fragment.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..mlip.heads import mlp

__all__ = ["FragmentStateEmbedding"]


class FragmentStateEmbedding(nn.Module):
    """Per-atom block encoding its fragment's ``(Q_f, 2S_f, n_f)``, zero at the neutral singlet.

    **v4: this is the key.** It used to be one block among several ways the model knew what a
    fragment was -- alongside the per-composition expert that owned the parameter heads, and
    alongside the latent those encoders learned. Three encodings of one fact, all trainable,
    all competing. Now there is one, and it is this: the state of a fragment is its charge, its
    multiplicity and its composition, and nothing else about it is a *state*.

    That is what makes it mixable. A learned latent has no units and no canonical frame, so
    two experts' latents have whatever relationship training happened to give them and a
    convex combination of them means nothing in particular. ``(Q, 2S, n)`` are physical labels
    with an unambiguous fractional reading -- a proton halfway between two waters genuinely
    leaves its host half-charged -- so the crossover is a *state* being lifted off an integer,
    not two latents being averaged. See :meth:`mixed`.

    The featurizer every force-field term uses (``FlatLambdaSOAPFeaturizer``) carries species
    and geometry only -- no fragment charge, no multiplicity. On water that is invisible,
    because every fragment is a neutral singlet. On H5O2+ it is fatal: the two fragmentations
    differ precisely in which fragment carries the charge, an H2O and an H3O+ have very
    different internal energies, and with no fragment-level information the model can only
    tell them apart by geometry -- while an OH radical and a hydroxide are indistinguishable
    outright.

    The output is ``net(Q, 2S) - net(0, 0)``, so it is **identically zero for a neutral
    singlet no matter what the weights do**. That matters more than a zero-initialized
    readout would: on water-only data the input is the constant ``(0, 0)``, and a plain
    zero-init readout would still drift to some arbitrary constant that downstream biases
    absorb. Anchoring at the neutral reference means water training genuinely cannot move
    this block, and for a charged fragment the block reads as the *deviation from neutral*,
    which is the interpretable thing to condition on.

    Be clear about what this buys today: nothing. A constant-zero input receives no gradient,
    so the block is not trained until charged-fragment data arrives. It is here so that step
    is an addition rather than a retrofit of fragment-state awareness into the whole
    force-field stack at the same time as the fragmentation mixture.

    ``dim=0`` disables it entirely and is bit-identical to a model built without it.

    **Exempt from weight decay**, for the same reason :class:`EnvironmentResidual` is and with
    the same evidence. Anchoring makes the block's gradient a *difference*, water-only data
    holds its input at the constant ``(0, 0)`` so that difference is exactly zero, and decay is
    then unopposed: measured over a staged fit its first-layer weight norm went 3.17 -> 1.96 ->
    0.24 -> 9e-13. Nothing was wrong with that while the input stays constant -- but the whole
    point of the block is to be ready when charged-fragment data arrives, and a flattened block
    is not ready. Left alone it sits at its initialization instead, which is.
    """

    #: See the note above. Weight decay would flatten a block that water-only data cannot
    #: train, leaving nothing for H5O2+ data to start from.
    no_weight_decay = True

    def __init__(
        self, dim: int = 4, *, n_species: int = 0, hidden: int = 32, depth: int = 1
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.n_species = int(n_species)
        self.net = mlp(2 + self.n_species, hidden, depth, self.dim) if self.dim else None

    def _embed(self, q, two_s, counts) -> torch.Tensor:
        """``(*, dim)`` from continuous ``(Q, 2S, n)``, anchored at the neutral singlet.

        The anchor is taken at ``(0, 0, n)`` -- same composition, neutral and closed-shell --
        so the block stays identically zero for a neutral singlet of *any* composition and
        water-only data still cannot move it, which is the property the class docstring's
        argument depends on. Composition then enters only through its interaction with charge
        and spin: "how does charge sit on *this* fragment", which is the interpretable form.
        It is also the reason ``n`` is not simply a second additive feature -- an unanchored
        composition block would receive gradient on water data and drift.
        """
        x = torch.cat((q.unsqueeze(-1), two_s.unsqueeze(-1), counts), dim=-1)
        ref = torch.cat((torch.zeros_like(x[..., :2]), counts), dim=-1)
        return self.net(x) - self.net(ref)

    def forward(
        self,
        batch,
        fragment_idx: torch.Tensor,
        dtype,
        device,
        *,
        element_counts: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """``(N, dim)`` per atom, for a **definite** fragmentation.

        ``element_counts`` is ``(N, n_species)``, the census of each atom's fragment
        (:func:`rsfff.ff.partition.element_counts`). It is per atom rather than per fragment
        because that is the form :meth:`mixed` has to produce -- a mixture has no single
        fragment numbering to state it in -- and carrying one convention is worth more than
        saving a gather.
        """
        if self.net is None:
            return None
        n_frag = int(batch.n_fragments)
        zeros = torch.zeros(n_frag, dtype=dtype, device=device)
        q = zeros if batch.fragment_charge is None else batch.fragment_charge.to(dtype)
        s = zeros if batch.fragment_two_s is None else batch.fragment_two_s.to(dtype)
        counts = self._counts(element_counts, fragment_idx.shape[0], dtype, device)
        return self._embed(q[fragment_idx], s[fragment_idx], counts)

    def mixed(self, q, two_s, counts) -> torch.Tensor | None:
        """``(N, dim)`` from **fractional** per-atom state: the mixture's entry point.

        The inputs come from :func:`rsfff.ff.partition.mixed_state`, which mixes the fragment
        labels themselves. Mixing the *inputs* and never the outputs is the whole point: this
        net is continuous in ``(Q, 2S, n)``, so a half-charged, 2.5-hydrogen fragment is a
        point it can be asked about and its answer is a genuine output. Averaging two finished
        embeddings would place the decoder somewhere nothing produced -- the failure mode this
        design exists to avoid, and one no quantity of training data repairs.

        At a one-hot membership the fractional inputs are the vertex's own integer labels, so
        this is bit-identical to :meth:`forward` there.
        """
        if self.net is None:
            return None
        return self._embed(q, two_s, self._counts(counts, q.shape[0], q.dtype, q.device))

    def _counts(self, counts, n_atoms: int, dtype, device) -> torch.Tensor:
        if self.n_species == 0:
            return torch.zeros(n_atoms, 0, dtype=dtype, device=device)
        if counts is None:
            raise ValueError(
                f"FragmentStateEmbedding was built with n_species={self.n_species} and needs "
                f"the per-atom element census; pass element_counts "
                f"(rsfff.ff.partition.element_counts)"
            )
        return counts.to(dtype)
