"""Fragment experts: one network per chemical composition, and the bank that dispatches to them.

``docs/fff_v2.md`` §2. There is **no universal parameterization network keyed by element**. A
fragment of a given composition is described by a network dedicated to that composition, which
is also the expert on how that fragment's parameters shift in the presence of others.

That is a worse scalability story than a universal head and a better sharpness story, which is
the trade being made deliberately: the one-body description has to be sharp enough that the
constant/not-constant split (``theta_0`` versus ``theta``) is worth guaranteeing at all.

Charge and multiplicity are **not** part of the type
----------------------------------------------------
The key is the composition alone, so one ``"OH"`` expert covers hydroxide and the OH radical and
is told which it is looking at by :class:`~rsfff.ff.unified.FragmentStateEmbedding`, whose
``(Q_f, 2S_f)`` block joins the fragment slot. Splitting on charge and spin instead would
multiply the expert count by the whole charge/spin manifold while the two share almost all of
their chemistry -- and what differs between them is exactly what an input slot is for.

Dispatch
--------
With one composition present -- water -- :meth:`ExpertBank.groups` returns a single group
covering every atom and the caller never gathers anything. With more, atoms are grouped by their
fragment's composition and each expert runs on its own group. Grouping by *fragment* is what
makes this safe for the response solve as well as for the per-atom heads: SQE couples atoms
within a fragment, and a fragment never spans two experts.

The composition is computed on every forward rather than trusted from a config. It is cheap --
one scatter over atoms plus a ``torch.unique`` over the per-fragment count rows -- and the
failure it prevents is silent: an H3O+ evaluated by the water expert would produce a number, not
an error.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..mlip.heads import two_slot_mlp, zero_init_readout

__all__ = [
    "ApplicabilityHead",
    "ExpertBank",
    "ExpertGroup",
    "FragmentExpert",
    "composition_keys",
]


def composition_keys(
    species_idx: torch.Tensor,     # (N,) index into `symbols`
    fragment_idx: torch.Tensor,    # (N,) fragment id per atom
    n_fragments: int,
    symbols: "list[str] | tuple[str, ...]",
) -> tuple[list[str], torch.Tensor]:
    """``(unique_keys, inverse (F,))`` -- each fragment's canonical composition.

    ``"H2O"``, ``"H3O"``, ``"OH"``: element symbols in the order ``symbols`` gives them, each
    followed by its count when that count is not 1. The formatting loop runs over the *unique*
    compositions, not over the fragments, so a batch of 640 waters formats one string.
    """
    n_species = len(symbols)
    one_hot = torch.zeros(
        species_idx.shape[0], n_species, dtype=torch.long, device=species_idx.device
    )
    one_hot.scatter_(1, species_idx.reshape(-1, 1), 1)
    counts = torch.zeros(
        n_fragments, n_species, dtype=torch.long, device=species_idx.device
    ).index_add_(0, fragment_idx, one_hot)
    rows, inverse = torch.unique(counts, dim=0, return_inverse=True)
    keys = [
        "".join(
            f"{symbols[c]}{int(n) if int(n) > 1 else ''}"
            for c, n in enumerate(row) if int(n)
        )
        for row in rows
    ]
    return keys, inverse


@dataclass(frozen=True)
class ExpertGroup:
    """One expert and the slice of the batch it answers for.

    ``atom_index`` and ``fragment_index`` are ``None`` on the single-expert fast path, meaning
    "everything" -- so a caller writes ``feats[g.atom_index] if g.atom_index is not None else
    feats`` once, and the common case allocates nothing.
    """

    key: str
    expert: "FragmentExpert"
    atom_index: torch.Tensor | None
    fragment_index: torch.Tensor | None

    @property
    def is_everything(self) -> bool:
        return self.atom_index is None


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


class FragmentExpert(nn.Module):
    """Every parameter-emitting head for one fragment composition.

    A container, deliberately: the orchestration -- which evaluation is isolated, which pairs
    are routed where, when the coupled solve runs -- belongs to the model that assembles the
    energy, not to the expert. What the expert owns is the *weights*, and owning them together
    is what makes "the water expert" a thing that can be saved, frozen, inspected or swapped.
    """

    def __init__(
        self,
        key: str,
        *,
        response,
        disp_params,
        pauli_params,
        range_heads,
        bond,
        fragment_state,
        applicability: ApplicabilityHead | None = None,
    ) -> None:
        super().__init__()
        self.key = str(key)
        self.response = response
        self.disp_params = disp_params
        self.pauli_params = pauli_params
        self.range_heads = range_heads
        self.bond = bond
        self.fragment_state = fragment_state
        self.applicability = applicability

    def extra_repr(self) -> str:
        return f"key={self.key!r}"


class ExpertBank(nn.Module):
    """The experts, keyed by composition, with the dispatch that finds the right one.

    ``ModuleDict`` under the hood, so every expert's parameters, state dict and ``.to()`` are
    covered, and an expert for a composition the data never contains simply never runs.
    """

    def __init__(self, experts: "dict[str, FragmentExpert]", symbols: "tuple[str, ...]") -> None:
        super().__init__()
        if not experts:
            raise ValueError("ExpertBank needs at least one expert")
        self.experts = nn.ModuleDict(dict(experts))
        self.symbols = tuple(symbols)

    @property
    def only(self) -> FragmentExpert:
        """The single expert, for the one-composition fast path. Raises if there are several."""
        if len(self.experts) != 1:
            raise ValueError(
                f"ExpertBank.only is for the single-composition case; this bank holds "
                f"{sorted(self.experts)}. Use groups() and fan out."
            )
        return next(iter(self.experts.values()))

    def assign(
        self, species_idx: torch.Tensor, fragment_idx: torch.Tensor, n_fragments: int
    ) -> tuple[list[str], torch.Tensor]:
        """``(unique_keys, inverse (F,))``, checked against the bank.

        An unknown composition raises. It is the failure worth being loud about: the model
        would otherwise answer, plausibly and wrongly, with some other molecule's expert.
        """
        keys, inverse = composition_keys(
            species_idx, fragment_idx, n_fragments, self.symbols
        )
        unknown = [k for k in keys if k not in self.experts]
        if unknown:
            raise KeyError(
                f"no expert for fragment composition(s) {unknown}; this bank has "
                f"{sorted(self.experts)}. A fragment expert model cannot fall back to a "
                f"generic head -- that is the thing it does not have."
            )
        return keys, inverse

    def groups(
        self, species_idx: torch.Tensor, fragment_idx: torch.Tensor, n_fragments: int
    ) -> list[ExpertGroup]:
        """The batch partitioned by expert, in a stable order.

        One composition -- the water case -- returns a single group with ``atom_index`` and
        ``fragment_index`` as ``None``, so the caller indexes nothing and the fast path costs a
        scatter and a ``unique`` rather than a gather over every per-atom quantity.
        """
        keys, inverse = self.assign(species_idx, fragment_idx, n_fragments)
        if len(keys) == 1:
            return [ExpertGroup(keys[0], self.experts[keys[0]], None, None)]
        atom_key = inverse[fragment_idx]
        out = []
        for i, key in enumerate(keys):
            frag_sel = (inverse == i).nonzero().squeeze(-1)
            atom_sel = (atom_key == i).nonzero().squeeze(-1)
            out.append(ExpertGroup(key, self.experts[key], atom_sel, frag_sel))
        return out

    def __getitem__(self, key: str) -> FragmentExpert:
        return self.experts[key]

    def __len__(self) -> int:
        return len(self.experts)

    def extra_repr(self) -> str:
        return f"compositions={sorted(self.experts)}"
