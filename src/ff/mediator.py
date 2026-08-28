"""The mediator: one universal network deciding how much of each fragmentation to keep.

``docs/fff_v2.md`` §8. When a proton sits between two oxygens, two decompositions of the same
geometry are both defensible, and a model that picks one discontinuously has a discontinuous
energy. The mediator replaces the pick with a **membership**: a partition of unity over the
candidate assignments, smooth in the geometry, exactly one-hot where only one candidate is
live.

What this module owns, and what it deliberately does not
--------------------------------------------------------
It owns the *weights* and nothing else. The mixing itself -- which quantity is combined at the
parameter level, which at the output, which in the accounting -- belongs to
:func:`rsfff.ff.mixture_model.mixture_forward`, because that is where the
energy is assembled and the rule ("mix at the lowest level at which the quantity means the
same thing to both experts") is a statement about the energy rather than about the weights.

**One universal network, not one per expert** (§8). ``M`` learns the logic of chemical
competition and never a fragment's chemistry; a per-composition mediator would be learning
both, and the second is what the experts are for.

Weights over decompositions, not over hosts
-------------------------------------------
§8 writes the membership as ``pi_ig``: per shared atom, over its candidate hosts. In the
corpus this model has, ``|D| = 1`` in every contested frame -- a single hydrogen changes
address and nothing else -- so "which host does atom *i* belong to" and "which decomposition
is this" are the same question, and the weight is one number per decomposition. That is the
specialization §8 names, implemented as named. :func:`contested_atoms` is what *measures* the
premise rather than assuming it -- it returns the whole set ``D``, and the caller
(:func:`rsfff.train.data.mixture_groups`) refuses a frame with more than one member, because
the right generalization there is a membership over *sets* (§10) and silently mediating one
atom of several would be wrong in a way no metric would show.

Swap symmetry comes free, and better than specified
---------------------------------------------------
Invariant 3 requires the weights not to depend on which candidate was enumerated first. §8
proposes reading only symmetric combinations (``h^K + h^J`` and ``(h^K - h^J)^2``, the
:class:`rsfff.mlip.adiabatic.AdiabaticCorrection` trick). This module gets the invariant a
different way: the score is computed **per decomposition from that decomposition's own
descriptor**, and a softmax over decompositions is permutation-*equivariant*, so permuting the
enumeration permutes the scores identically and the weight attached to a given decomposition
does not move. That is exact rather than architectural, needs no pairing, and -- unlike the
sum/squared-difference construction, which is defined for two states -- it generalizes to the
three decompositions an ``H3O+(H2O)2`` frame actually carries.

Continuity
----------
``Omega`` enters multiplicatively, never as ``log Omega`` added to a logit: a candidate that
closes has ``Omega = 0`` exactly, and ``log 0`` is ``-inf``, which is a NaN in the backward
pass rather than a closed channel. The multiplicative form is what
:meth:`rsfff.mlip.mixture.MixtureModel._coefficients` already uses and it is C² because
:func:`rsfff.mlip.switch.validity_bump` is.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..mlip.heads import mlp
from ..mlip.switch import validity_bump

__all__ = [
    "MediatorHead",
    "MediatorOutput",
    "align_fragments",
    "contact_distance",
    "contested_atoms",
]


def align_fragments(fragments: torch.Tensor) -> torch.Tensor:
    """``(M, N)`` every decomposition's ids rewritten into decomposition 0's numbering.

    Fragment ids are arbitrary labels: decomposition 1 may call ``0`` what decomposition 0
    calls ``1``, and comparing them raw would report a whole system as contested when nothing
    moved. Each of decomposition ``m``'s fragments is therefore matched to the fragment of
    decomposition 0 it overlaps most, greedily and largest-overlap-first, and relabeled.

    Greedy is exact for the case this model has. A relabeling moves one atom, so the overlap
    matrix is dominated by its diagonal after matching and no two fragments compete for the
    same partner; where they would, the frame is a concerted rearrangement, which
    :func:`contested_atoms` refuses on its own terms.
    """
    if fragments.dim() != 2:
        raise ValueError(f"fragments must be (M, N), got {tuple(fragments.shape)}")
    base = fragments[0]
    n_base = int(base.max()) + 1
    out = [base]
    for m in range(1, fragments.shape[0]):
        row = fragments[m]
        n_row = int(row.max()) + 1
        overlap = torch.zeros(n_row, n_base, dtype=torch.long, device=row.device)
        for a in range(n_row):
            mask = row == a
            overlap[a] = torch.bincount(base[mask], minlength=n_base)
        mapping = torch.full((n_row,), -1, dtype=torch.long, device=row.device)
        taken = torch.zeros(n_base, dtype=torch.bool, device=row.device)
        # Largest overlap first, so the unambiguous pairs claim their partner before any
        # fragment that genuinely straddles two of them gets a say.
        for a in torch.argsort(overlap.max(dim=1).values, descending=True).tolist():
            candidates = overlap[a].clone()
            candidates[taken] = -1
            best = int(candidates.argmax())
            mapping[a] = best
            taken[best] = True
        out.append(mapping[row])
    return torch.stack(out)


def contested_atoms(fragments: torch.Tensor) -> torch.Tensor:
    """``(D,)`` the atoms that change address between decompositions -- §8's ``D``.

    ``D = { i : frag_A(i) != frag_B(i) }``, evaluated after :func:`align_fragments` has put
    every decomposition in one numbering so the comparison means what it says.

    Note what this deliberately does *not* do: mark every atom of both hosts. Moving one
    proton changes the *composition* of two fragments and therefore the descriptor of every
    atom in them, but only one atom changed address, and it is that atom the mediator decides
    about. A co-membership test would return the whole reactive complex and hand the mediator
    a question with no answer.
    """
    aligned = align_fragments(fragments)
    differs = (aligned != aligned[0]).any(dim=0)
    return differs.nonzero().squeeze(-1)


def contact_distance(
    positions: torch.Tensor,       # (N, 3)
    fragments: torch.Tensor,       # (M, N)
    atoms: torch.Tensor,           # (D,) the contested atoms
) -> torch.Tensor:
    """``(M, D)`` how far each contested atom sits from the host each decomposition gives it.

    The distance to the *nearest other atom of its host*, which for a transferring hydrogen is
    the ``H···O`` bond being made or broken -- the reaction coordinate §8 calls ``rho``, read
    per decomposition. A host with no other atom (a bare ion) reports ``inf``, which closes its
    validity envelope rather than dividing by nothing.
    """
    atoms = torch.as_tensor(atoms).reshape(-1)
    if atoms.numel() == 0:
        # No contested atom: nothing is being relabeled, so there is no contact geometry to
        # report. Returning an empty `(M, 0)` rather than raising is what lets a spectator
        # frame -- the common case once this runs under dynamics rather than over a curated
        # corpus -- pass through the same code path as a reactive one.
        return positions.new_zeros(fragments.shape[0], 0)
    out = []
    for m in range(fragments.shape[0]):
        row = []
        for a in atoms.tolist():
            r = (positions - positions[a]).norm(dim=-1)
            mates = (fragments[m] == fragments[m][a]).clone()
            mates[a] = False
            row.append(r[mates].min() if bool(mates.any()) else r.new_tensor(float("inf")))
        out.append(torch.stack(row))
    return torch.stack(out)


@dataclass
class MediatorOutput:
    """One group's mixture weights and everything worth watching about them.

    weights   : (M,) the partition of unity over decompositions. **This is `pi`.**
    omega     : (M,) the validity envelope per decomposition, C² in the geometry
    score     : (M,) the raw network score, before the envelope
    rho       : (M, D) the contact distance per contested atom, driving the envelope
    atoms     : (D,) the contested atoms, in the group's shared atom order
    """

    weights: torch.Tensor
    omega: torch.Tensor
    score: torch.Tensor
    rho: torch.Tensor
    atoms: torch.Tensor

    @property
    def occupancy(self) -> torch.Tensor:
        """``1 - max_m w_m``: 0 when the mediator has decided, (M-1)/M when it is hedging.

        The `pi` occupancy diagnostic of §9. Exactly 0 everywhere means the mediator is off or
        the trigger never opens; broadly split everywhere means it is hedging rather than
        deciding.
        """
        return 1.0 - self.weights.max()


class MediatorHead(nn.Module):
    """``M(h_i, eta_i, H_host, Q_host, 2S_host, rho) -> score``, softmaxed over decompositions.

    One network for the whole model. It sees, per decomposition: the contested atom's own two
    slots, the pooled two-slot descriptor of the host that decomposition gives it, that host's
    charge and multiplicity, and the contact distance. Everything it reads is a property of
    *this* decomposition, which is what makes the softmax over decompositions the swap
    symmetry (module docstring).

    The readout is zero-initialized, so an untrained mediator returns the envelope alone --
    a geometric prior, not an opinion -- and every fit starts from "the candidates are equally
    good wherever they are equally open".
    """

    def __init__(
        self,
        p_frag: int,
        p_env: int,
        *,
        hidden: int = 32,
        depth: int = 2,
        bump: dict | None = None,
    ) -> None:
        super().__init__()
        self.p_frag = int(p_frag)
        self.p_env = int(p_env)
        width = self.p_frag + self.p_env
        # atom slot | pooled host slot | (Q, 2S) | (rho, 1/rho)
        self.net = mlp(2 * width + 2 + 2, hidden, depth, 1)
        # Zero the readout: an untrained mediator must be the envelope and nothing else.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        #: The validity envelope on the contact distance. The default opens a candidate once
        #: the contested atom is within bonding range of the host and closes it past a
        #: hydrogen bond -- wide enough that both candidates are live through a transfer, tight
        #: enough that a spectator water never enumerates one.
        self.bump = dict(bump or dict(lo0=0.0, lo1=0.0, hi1=1.35, hi0=2.20))

    @property
    def width(self) -> int:
        return self.p_frag + self.p_env

    def forward(
        self,
        inv_feats: torch.Tensor,        # (M, N, p_frag + p_env) joined slot per decomposition
        fragments: torch.Tensor,        # (M, N) long
        positions: torch.Tensor,        # (N, 3)
        fragment_charge: torch.Tensor,  # (M, N) the host's formal charge, gathered per atom
        fragment_two_s: torch.Tensor,   # (M, N) likewise
        atoms: torch.Tensor,            # (D,) the contested atoms
    ) -> MediatorOutput:
        """``MediatorOutput`` for one group. ``fragment_charge``/``two_s`` are **per atom**.

        Per atom rather than per fragment because the fragment numbering differs between
        decompositions and the only rows this head wants are the contested atoms' hosts anyway.

        **``D`` is a set, not a single atom.** §8 asserts ``|D| = 1`` for every frame in this
        corpus, which is true *pairwise* and false across three decompositions: an
        ``OH-(H2O)2`` frame carries three, each one hop from the reference but moving a
        *different* hydrogen, so the union is ``|D| = 2``. Every one of them is still a
        single-atom relabeling and none is a concerted rearrangement, so the score sums the
        per-atom contribution over ``D`` -- which reduces to §8's form at ``|D| = 1`` and does
        not throw away half the ion corpus at ``|D| = 2``.
        """
        n_dec = int(fragments.shape[0])
        atoms = torch.as_tensor(atoms, device=fragments.device).reshape(-1)
        if atoms.numel() == 0:
            # `D` empty means every decomposition agrees about every atom, which for distinct
            # decompositions is impossible -- so in practice this is `M = 1`, the uncontested
            # frame §8 promises costs nothing ("an atom with one candidate has pi = 1"). The
            # promise was never exercised by training, where the corpus is contested by
            # construction; under dynamics it is the majority of steps. There is nothing for
            # the score net to read, so the membership is uniform by definition.
            uniform = positions.new_full((n_dec,), 1.0 / n_dec)
            return MediatorOutput(
                weights=uniform,
                omega=torch.ones_like(uniform),
                score=torch.zeros_like(uniform),
                rho=positions.new_zeros(n_dec, 0),
                atoms=atoms,
            )
        if inv_feats.shape[-1] != self.width:
            raise ValueError(
                f"MediatorHead got features of width {inv_feats.shape[-1]}, expected "
                f"{self.width} ({self.p_frag} fragment + {self.p_env} environment). It reads "
                f"the joined slot: pass SlotFeatures.joined(), not .isolated()."
            )

        rho = contact_distance(positions, fragments, atoms)                  # (M, D)
        # A candidate is valid only where *every* moved atom is in range of the host it was
        # given, so the envelopes multiply. One closed contact closes the candidate.
        omega = torch.stack(
            [
                torch.stack([validity_bump(r, **self.bump) for r in row]).prod()
                for row in rho
            ]
        )                                                                    # (M,)

        rows = []
        for m in range(n_dec):
            per_atom = []
            for d, a in enumerate(atoms.tolist()):
                host = fragments[m] == fragments[m][a]
                pooled = inv_feats[m][host].mean(dim=0)
                # `rho` can be inf for a lone-atom host; the envelope is exactly zero there,
                # but the *input* must not be inf or the gradient is NaN regardless of the
                # weight it ends up multiplied by.
                r = torch.nan_to_num(rho[m, d], posinf=1.0e3)
                per_atom.append(
                    torch.cat(
                        (
                            inv_feats[m][a],
                            pooled,
                            fragment_charge[m][a].reshape(1),
                            fragment_two_s[m][a].reshape(1),
                            r.reshape(1),
                            (1.0 / r.clamp(min=1e-3)).reshape(1),
                        )
                    )
                )
            rows.append(torch.stack(per_atom))
        # (M, D, F) -> per-atom score -> summed over D. Summing rather than averaging keeps a
        # two-atom frame's scores on the same scale as the energy differences that drive them.
        score = self.net(torch.stack(rows)).squeeze(-1).sum(dim=-1)          # (M,)

        # Multiplicative envelope, never `+ log omega`: a closed candidate has omega exactly 0
        # and `log 0` is a NaN in the backward pass rather than a closed channel.
        w = omega * (score - score.max().detach()).exp()
        total = w.sum()
        # Every candidate closed is a geometry the enumeration should not have produced, but
        # falling back to the envelope-free softmax is a defined answer rather than a 0/0.
        weights = w / total if bool(total > 0) else torch.softmax(score, dim=0)
        return MediatorOutput(
            weights=weights, omega=omega, score=score, rho=rho, atoms=atoms
        )
