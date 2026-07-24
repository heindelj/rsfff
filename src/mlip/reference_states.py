"""Atomic reference embeddings and the one-body baselines built on them.

This is the entry point of the frozen monomer stack (docs/mixture_of_diabatic_embeddings.md
§2.1): every atom is initialized not with a bare element embedding but with a *reference
embedding* conditioned on the diabatic state of the fragment it belongs to,

    e_i = E(Z_i, FragmentType(i in F_a), Q_a, S_a)

and everything downstream -- the density channel weights, the energy head, the response heads --
consumes ``e``. Two pieces live here:

:class:`ReferenceEmbedding`
    ``e_i`` itself, plus the SQE baseline charge ``q_i^(0)`` it is conditioned on. The state
    enters as (i) the per-atom baseline charge, (ii) the fragment's ``2S_a``, and (iii) a pooled
    description of the fragment's size and composition -- no fragment-type lookup table, so a
    fragment the library gains later is described by the same parameters rather than needing a
    new embedding row.

:class:`AtomicStateReference`
    The isolated-atom QC anchors from ``scripts/atomic_reference_states.py``: energy and
    polarizability at each integer charge, plus the per-element Mulliken electronegativity and
    chemical hardness derived from IP/EA. A single atom has an all-zero SOAP density, so every
    head reduces to a function of ``e_i`` alone -- which is what makes these anchors *exact*
    constraints on the free-atom limit rather than mere regularization, and what lets the
    hardness/electronegativity heads start at physically meaningful per-element values.

Unit convention follows ``rsfff.train.data``: energies in Hartree, polarizabilities converted
from a0^3 to e^2*Angstrom^2/Hartree on load.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn

from .heads import mlp


def _fragment_mean(x: torch.Tensor, fragment_idx: torch.Tensor, n_fragments: int):
    """Per-fragment mean of a per-atom quantity, returned per *fragment*: (N, ...) -> (F, ...)."""
    totals = x.new_zeros((n_fragments,) + x.shape[1:]).index_add_(0, fragment_idx, x)
    counts = torch.bincount(fragment_idx, minlength=n_fragments).clamp(min=1)
    return totals / counts.reshape((-1,) + (1,) * (x.dim() - 1)).to(totals.dtype)


@dataclass
class ReferenceStateOutput:
    """``embedding`` (N, D) reference embeddings; ``baseline_charge`` (N,) SQE ``q^(0)``."""

    embedding: torch.Tensor
    baseline_charge: torch.Tensor


class ReferenceEmbedding(nn.Module):
    """Reference embedding ``e_i`` and SQE baseline charge ``q_i^(0)`` from the diabatic state.

    The baseline charge is built so it sums to the fragment's formal charge *exactly*, for any
    network weights::

        g_i      = MLP([elem_emb[Z_i], 2S_a])          # unnormalized per-atom preference
        q_i^(0)  = g_i - mean_{j in a}(g_j) + Q_a / N_a

    Subtracting the fragment mean makes the learned part traceless, so ``sum_{i in a} q_i^(0) =
    Q_a`` structurally. Unlike a softmax split this behaves correctly at ``Q_a = 0`` (a neutral
    fragment still gets a nonzero internal charge distribution) and imposes no sign constraint.

    The embedding is a FiLM-modulated element embedding::

        gamma, beta = MLP([q_i^(0), 2S_a, N_a, composition_pool_a])
        e_i         = elem_emb[Z_i] * (1 + gamma) + beta

    Both readouts are zero-initialized, so at step 0 ``q_i^(0) = Q_a / N_a`` (the formal charge
    spread uniformly) and ``e_i = elem_emb[Z_i]`` (state-agnostic) -- the model starts at a
    sensible physical default and learns its state dependence, matching the zero-init convention
    used by the existing heads.
    """

    def __init__(
        self,
        n_species: int,
        *,
        emb_dim: int = 32,
        hidden: int = 32,
        depth: int = 1,
    ) -> None:
        super().__init__()
        self.n_species = int(n_species)
        self.emb_dim = int(emb_dim)
        self.elem_emb = nn.Embedding(n_species, emb_dim)
        # q^(0): per-atom preference from (element embedding, fragment spin).
        self.q0_mlp = mlp(emb_dim + 1, hidden, depth, 1)
        # FiLM: [q0, 2S, N_a, composition pool (n_species)] -> (gamma, beta)
        self.film_mlp = mlp(3 + n_species, hidden, depth, 2 * emb_dim)
        for m in (self.q0_mlp, self.film_mlp):
            with torch.no_grad():
                m[-1].weight.zero_()
                m[-1].bias.zero_()

    def forward(
        self,
        species_idx: torch.Tensor,      # (N,) in [0, n_species)
        fragment_idx: torch.Tensor,     # (N,) in [0, n_fragments)
        fragment_charge: torch.Tensor,  # (F,) formal Q_a
        fragment_two_s: torch.Tensor,   # (F,) 2S_a
        n_fragments: int,
    ) -> ReferenceStateOutput:
        elem = self.elem_emb(species_idx)                            # (N, D)
        two_s_i = fragment_two_s[fragment_idx].unsqueeze(-1)         # (N, 1)

        # --- baseline charge, exactly conserving Q_a ---
        g = self.q0_mlp(torch.cat((elem, two_s_i), dim=-1)).squeeze(-1)   # (N,)
        g_mean = _fragment_mean(g, fragment_idx, n_fragments)             # (F,)
        counts = torch.bincount(fragment_idx, minlength=n_fragments).clamp(min=1)
        q_uniform = fragment_charge / counts.to(fragment_charge.dtype)    # (F,)
        q0 = g - g_mean[fragment_idx] + q_uniform[fragment_idx]           # (N,)

        # --- fragment descriptors: size and composition ---
        onehot = torch.zeros(
            species_idx.shape[0], self.n_species, dtype=elem.dtype, device=elem.device
        ).scatter_(1, species_idx.unsqueeze(-1), 1.0)
        composition = _fragment_mean(onehot, fragment_idx, n_fragments)   # (F, n_species)
        n_atoms_f = counts.to(elem.dtype).unsqueeze(-1)                   # (F, 1)

        state = torch.cat(
            (
                q0.unsqueeze(-1),
                two_s_i,
                n_atoms_f[fragment_idx],
                composition[fragment_idx],
            ),
            dim=-1,
        )                                                                 # (N, 3 + n_species)
        gamma, beta = self.film_mlp(state).chunk(2, dim=-1)                # (N, D) each
        return ReferenceStateOutput(embedding=elem * (1.0 + gamma) + beta, baseline_charge=q0)


class AtomWeightNet(nn.Module):
    """Map a reference embedding ``e`` to density-channel weights ``w(e)``: (N, Kw).

    ``w(e_j)`` weights neighbor ``j``'s contribution to the state-decorated density
    ``A[i,k,n,lm] = sum_j w_k(e_j) R_n(r_ij) Y_lm f_cut`` (§2.1). Because ``e_j`` carries the
    neighbor's diabatic identity -- element, fragment charge, fragment spin -- the density
    distinguishes, say, an H of hydroxide from an H of hydronium, which a species one-hot
    cannot. The weights are invariant scalars, so the density and everything built from it stay
    exactly equivariant.
    """

    def __init__(
        self, emb_dim: int, weight_channels: int, *, hidden: int = 32, depth: int = 1
    ) -> None:
        super().__init__()
        self.weight_channels = int(weight_channels)
        self.net = mlp(emb_dim, hidden, depth, weight_channels)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.net(embedding)                                  # (N, Kw)


class AtomicStateReference:
    """Isolated-atom QC anchors, aligned to a featurizer's species ordering.

    Built from ``data/atomic_reference_states.json`` (see
    ``scripts/atomic_reference_states.py``). One row per (element, charge) reference state:

        species_idx  (S,)       index into ``neighbor_types``
        charge       (S,)       formal charge (e)
        two_s        (S,)       2S of the state's ground term
        energy       (S,)       total energy (Hartree, absolute)
        alpha        (S, 3, 3)  static polarizability (e^2*Angstrom^2/Hartree)
        bound        (S,)       False where the anion is unbound at this level of theory

    ``bound`` matters: at b3lyp/def2-svpd neither H(-) nor O(2-) actually binds its extra
    electron, so those energies and (especially) polarizabilities are artifacts of the diffuse
    basis functions. They are kept -- the diabatic library needs formal O(2-)/H(-) references --
    but flagged so the training loss can weight them separately instead of chasing basis-set
    noise.
    """

    def __init__(
        self,
        *,
        species_idx: torch.Tensor,
        charge: torch.Tensor,
        two_s: torch.Tensor,
        energy: torch.Tensor,
        alpha: torch.Tensor,
        bound: torch.Tensor,
        symbols: list[str],
        chi_mulliken: torch.Tensor,
        hardness: torch.Tensor,
    ) -> None:
        self.species_idx = species_idx
        self.charge = charge
        self.two_s = two_s
        self.energy = energy
        self.alpha = alpha
        self.bound = bound
        self.symbols = symbols
        self.chi_mulliken = chi_mulliken
        self.hardness = hardness

    def __len__(self) -> int:
        return int(self.species_idx.shape[0])

    @classmethod
    def from_json(
        cls,
        path,
        neighbor_types: Sequence[int],
        *,
        dtype: torch.dtype | None = None,
    ) -> "AtomicStateReference":
        """Load the reference-state grid, keeping only elements in ``neighbor_types``.

        ``neighbor_types`` is the sorted list of atomic numbers the featurizer was built for;
        row ``species_idx`` is the position in that list, matching ``LambdaFeatures.species_idx``
        and the ``reference_energies`` convention in ``rsfff.train.data``.
        """
        import ase.units
        from ase.data import atomic_numbers as _z

        dtype = dtype or torch.get_default_dtype()
        bohr = float(ase.units.Bohr)
        raw = json.loads(Path(path).read_text())

        types = [int(t) for t in neighbor_types]
        z_to_species = {z: i for i, z in enumerate(types)}

        rows, syms = [], []
        for rec in raw["states"]:
            sym = rec["symbol"]
            z = int(_z[sym])
            if z not in z_to_species:
                continue
            rows.append(rec)
            syms.append(f"{sym}{int(rec['charge']):+d}")
        if not rows:
            raise ValueError(
                f"{path}: no reference states for elements Z={types}; "
                f"run scripts/atomic_reference_states.py for them"
            )

        missing = sorted(
            {t for t in types}
            - {int(_z[r["symbol"]]) for r in rows}
        )
        if missing:
            raise KeyError(
                f"{path}: no atomic reference states for Z={missing}; "
                f"run: python scripts/atomic_reference_states.py"
            )

        alpha = torch.tensor(
            [r["polarizability"] for r in rows], dtype=dtype
        ) * bohr**2  # a0^3 = e^2 a0^2 / Ha -> e^2 Ang^2 / Ha

        n_species = len(types)
        chi = torch.full((n_species,), float("nan"), dtype=dtype)
        eta = torch.full((n_species,), float("nan"), dtype=dtype)
        for sym, d in raw.get("derived", {}).items():
            z = int(_z[sym])
            if z not in z_to_species:
                continue
            if "chi_mulliken" in d:
                chi[z_to_species[z]] = float(d["chi_mulliken"])
            if "hardness" in d:
                eta[z_to_species[z]] = float(d["hardness"])

        return cls(
            species_idx=torch.tensor(
                [z_to_species[int(_z[r["symbol"]])] for r in rows], dtype=torch.long
            ),
            charge=torch.tensor([float(r["charge"]) for r in rows], dtype=dtype),
            two_s=torch.tensor(
                [float(int(r["multiplicity"]) - 1) for r in rows], dtype=dtype
            ),
            energy=torch.tensor([float(r["energy"]) for r in rows], dtype=dtype),
            alpha=alpha,
            bound=torch.tensor([bool(r.get("bound", True)) for r in rows]),
            symbols=syms,
            chi_mulliken=chi,
            hardness=eta,
        )

    def head_bias_init(self, *, chi_default: float = 0.0, eta_default: float = 0.5):
        """Per-element ``(chi_0, eta_0)`` for initializing the SQE parameter heads (Hartree).

        The Mulliken electronegativity ``(IP + EA)/2`` and chemical hardness ``IP - EA`` *are*
        the free-atom limits of the SQE ``chi`` and ``eta``, so starting the per-element biases
        there puts the charge solve in a physically sensible regime from step 0 rather than at
        an arbitrary constant. Elements whose IP/EA could not be derived fall back to the
        supplied defaults.
        """
        chi = torch.where(torch.isnan(self.chi_mulliken),
                          torch.full_like(self.chi_mulliken, chi_default),
                          self.chi_mulliken)
        eta = torch.where(torch.isnan(self.hardness),
                          torch.full_like(self.hardness, eta_default),
                          self.hardness)
        return chi, eta

    def to(self, device) -> "AtomicStateReference":
        return AtomicStateReference(
            species_idx=self.species_idx.to(device),
            charge=self.charge.to(device),
            two_s=self.two_s.to(device),
            energy=self.energy.to(device),
            alpha=self.alpha.to(device),
            bound=self.bound.to(device),
            symbols=self.symbols,
            chi_mulliken=self.chi_mulliken.to(device),
            hardness=self.hardness.to(device),
        )
