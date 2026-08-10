"""Tang-Toennies damped C6 dispersion with a learned short-range pair correction.

The energy over inter-fragment pairs is

    E = sum_{i<j} [ S(r) T(r) E_TT(r)  +  W(r) dE_ij ]

    E_TT(r) = -f_6(b_ij r) C6_ij / r^6        (Tang-Toennies damped, atomic units)
    C6_ij   = sqrt(C6_i C6_j),  b_ij = sqrt(b_i b_j)
    S(r)    = fermi_switch(r; r0, alpha)      range separation: 0 short, 1 long
    T(r)    = pairwise_switch(r, r_cut-1, r_cut)   neighbor-list taper, C2, exact 0
    W(r)    = the pair head's own compact envelope

``S`` switches the explicit dispersion **off at short range** so the short-range network
carries that region; ``r0`` is learnable, with the training loop applying a linear penalty
that biases it small, so handing energy to the network has to be earned rather than being
free (``docs/range_separated_mlip.md`` §4.5, §7: the explicit term should be a rigid
backbone, and a freely learnable damping length fights the networks for the same
mid-range energy).

``dE_ij`` is *delta learning*, not a partition: it carries whatever the damped-C6 form
cannot (higher orders, charge penetration, exchange-dispersion, anisotropy), and it is
zero-initialized so training starts as exactly the pure force field.

Units
-----
================  ==============  ===========================================
quantity          unit            note
================  ==============  ===========================================
positions, r      Angstrom        model convention (rsfff.train.data)
r0, alpha         Angstrom, /Ang  so r0 reads as the distance one reasons about
C6                Ha * bohr^6     literature/pyCMM convention
b                 1 / bohr        ditto
energy            Hartree
================  ==============  ===========================================

The single Angstrom -> bohr conversion happens on the first line of
:func:`tt_damped_c6_energy` and nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..features.features import LambdaFeatures
from ..mlip.heads import mlp
from ..mlip.pair_heads import PairEnergyHead
from ..mlip.switch import pairwise_switch
from .damping import fermi_switch, tang_toennies
from .pairs import inter_fragment_pairs
from .units import BOHR_ANG

#: Per-element ``(C6 [Ha*bohr^6], b [1/bohr])`` priors, from the fitted CMM water model
#: (``pyCMM/tests/data/water.xml:32-33``). These are *priors*, not ground truth: they were
#: fit to a different target than the EDA components trained against here.
DEFAULT_C6_PRIOR: dict[int, tuple[float, float]] = {
    8: (35.8289, 1.84302),   # O
    1: (1.98954, 1.30993),   # H
}

#: Damping-exponent prior in bohr^-1, applied uniformly across elements.
DEFAULT_B_PRIOR = 2.0


def tt_damped_c6_energy(
    r_ang: torch.Tensor,     # (P,) Angstrom
    c6_ij: torch.Tensor,     # (P,) Ha * bohr^6
    b_ij: torch.Tensor,      # (P,) 1 / bohr
    *,
    order: int = 6,
) -> torch.Tensor:
    """``-f_order(b_ij r) c6_ij / r^6`` in Hartree, with ``r`` supplied in Angstrom."""
    r_au = r_ang / BOHR_ANG
    return -tang_toennies(b_ij * r_au, order) * c6_ij / r_au.pow(6)


def _inverse_softplus(x: float) -> float:
    """``y`` such that ``softplus(y) == x``, for initializing a positive parameter."""
    return float(torch.log(torch.expm1(torch.tensor(float(x)))))


def build_log_priors(
    neighbor_types,
    *,
    c6_prior: dict[int, float] | None = None,
    b_prior: float | str = DEFAULT_B_PRIOR,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-species ``(log C6, log b)`` prior tensors ordered like ``neighbor_types``.

    ``b_prior`` is either a uniform value in bohr^-1 or the string ``"per_element"``, which
    uses the fitted values in :data:`DEFAULT_C6_PRIOR`. The difference is not cosmetic:
    measured against ``eda_disp`` on the water clusters, with C6 at its prior and no ML,

        per-element b (O 1.843, H 1.310)   MAE 0.28 (w2) .. 0.61 (w5) kJ/mol
        uniform b = 2.0                    MAE 2.87 (w2) .. 14.5 (w5) kJ/mol

    A larger ``b`` damps *less* at a given separation, so a uniform 2.0 over-binds -- most
    of all through hydrogen, whose fitted value is well below it and which dominates the
    close H-bond contacts. If you start from a uniform prior, let ``b`` learn (two
    per-species scalars); leaving it frozen there hands the pair correction a large
    systematic bias to mop up, which is exactly the gauge leakage the delta-learning
    arrangement is meant to avoid.

    An element absent from the C6 table raises rather than falling back to some average:
    a guessed C6 yields a plausible-looking model that is quietly wrong. Pass ``c6_prior``
    to extend the table.
    """
    table = {z: c6 for z, (c6, _) in DEFAULT_C6_PRIOR.items()}
    if c6_prior:
        table.update(c6_prior)
    missing = [int(z) for z in neighbor_types if int(z) not in table]
    if missing:
        raise KeyError(
            f"no C6 prior for atomic number(s) {missing}; extend DEFAULT_C6_PRIOR or pass "
            f"c6_prior={{Z: value}} (Ha*bohr^6). Refusing to guess."
        )
    log_c6 = torch.tensor([float(table[int(z)]) for z in neighbor_types]).log()

    if isinstance(b_prior, str):
        if b_prior != "per_element":
            raise ValueError(
                f"b_prior must be a number (bohr^-1) or 'per_element', got {b_prior!r}"
            )
        missing_b = [int(z) for z in neighbor_types if int(z) not in DEFAULT_C6_PRIOR]
        if missing_b:
            raise KeyError(f"no per-element damping exponent for atomic number(s) {missing_b}")
        log_b = torch.tensor(
            [DEFAULT_C6_PRIOR[int(z)][1] for z in neighbor_types]
        ).log()
    else:
        log_b = torch.full((len(neighbor_types),), float(b_prior)).log()
    return log_c6, log_b


class DispersionParameterHeads(nn.Module):
    """Per-atom ``(C6, b)`` from invariant features, as deviations from a per-species prior.

    Both are parameterized in **log space** rather than through a softplus, deviating
    deliberately from the repo's other positive-parameter heads:

    - the geometric-mean combination rule is exactly linear in logs
      (``log C6_ij = (log C6_i + log C6_j)/2``), so no ``sqrt`` -- whose derivative is
      unbounded as its argument approaches zero -- ever appears;
    - positivity is structural rather than a floor;
    - C6 spans a factor of 18 between H and O, so an *additive* residual in log space is a
      *multiplicative* correction, which is the right scaling for a quantity like this.

    Because the learnable tensors are deviations from the registered prior, plain
    ``weight_decay`` on them is regularization toward that prior -- no bespoke penalty.

    ``learn_b`` / ``environment_b`` both default off. ``b`` and ``C6`` are strongly
    degenerate over the mid-range window and ``b`` additionally competes with the pair
    correction, so the first fit should move only ``C6`` and leave the damping rigid
    (``docs/range_separated_mlip.md`` §4.5). Turning them on is an explicit choice.
    """

    def __init__(
        self,
        p0: int,
        n_species: int,
        *,
        log_c6_prior: torch.Tensor,     # (n_species,)
        log_b_prior: torch.Tensor,      # (n_species,)
        emb_dim: int = 16,
        hidden: int = 64,
        depth: int = 2,
        learn_c6: bool = True,
        environment_c6: bool = True,
        learn_b: bool = False,
        environment_b: bool = False,
    ) -> None:
        super().__init__()
        self.register_buffer("log_c6_prior", log_c6_prior.clone())
        self.register_buffer("log_b_prior", log_b_prior.clone())
        self.species_emb = nn.Embedding(n_species, emb_dim)
        self.environment_c6 = bool(environment_c6)
        self.environment_b = bool(environment_b)

        self.d_log_c6 = nn.Parameter(torch.zeros(n_species), requires_grad=learn_c6)
        self.d_log_b = nn.Parameter(torch.zeros(n_species), requires_grad=learn_b)

        self.c6_mlp = mlp(p0 + emb_dim, hidden, depth, 1) if environment_c6 else None
        self.b_mlp = mlp(p0 + emb_dim, hidden, depth, 1) if environment_b else None
        for m in (self.c6_mlp, self.b_mlp):
            if m is not None:   # start at exactly the per-species prior
                with torch.no_grad():
                    m[-1].weight.zero_()
                    m[-1].bias.zero_()

    def forward(
        self,
        inv_feats: torch.Tensor,     # (N, p0)
        species_idx: torch.Tensor,   # (N,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(c6 (N,) Ha*bohr^6, b (N,) 1/bohr)``."""
        s = species_idx
        log_c6 = self.log_c6_prior[s] + self.d_log_c6[s]
        log_b = self.log_b_prior[s] + self.d_log_b[s]
        if self.c6_mlp is not None or self.b_mlp is not None:
            x = torch.cat((inv_feats, self.species_emb(s)), dim=-1)
            if self.c6_mlp is not None:
                log_c6 = log_c6 + self.c6_mlp(x).squeeze(-1)
            if self.b_mlp is not None:
                log_b = log_b + self.b_mlp(x).squeeze(-1)
        return log_c6.exp(), log_b.exp()


@dataclass
class DispersionOutput:
    """Per-system energies plus the per-pair breakdown the diagnostics need."""

    energy: torch.Tensor         # (B,) total dispersion energy, Hartree
    energy_ff: torch.Tensor      # (B,) switched+tapered Tang-Toennies part
    energy_corr: torch.Tensor    # (B,) learned pair-correction part
    pair_index: torch.Tensor     # (2, P)
    r: torch.Tensor              # (P,) Angstrom
    e_pair_ff: torch.Tensor      # (P,)
    e_pair_corr: torch.Tensor    # (P,)
    r0: torch.Tensor             # () current range-separation midpoint, Angstrom
    c6: torch.Tensor             # (N,)
    b: torch.Tensor              # (N,)


class TTDispersion(nn.Module):
    """Damped-C6 dispersion over inter-fragment pairs, plus a short-range pair correction.

    Args
    ----
    cutoff        : pair-list cutoff in Angstrom. The Tang-Toennies term is tapered to
                    exactly zero over the last Angstrom before it, because truncating a
                    ``1/r^6`` tail outright puts a step in the energy and a delta in the
                    force.
    r0_init, alpha: Fermi range separation (Angstrom, Angstrom^-1). ``r0`` is learnable
                    via ``softplus``; ``alpha`` is fixed, since learning the width as well
                    only adds degeneracy with ``r0``.
    inter_only    : restrict to pairs in different fragments. This is what makes the sum
                    an interaction energy comparable to an EDA component.
    """

    def __init__(
        self,
        params: DispersionParameterHeads,
        correction: PairEnergyHead | None,
        *,
        cutoff: float = 10.0,
        taper_width: float = 1.0,
        r0_init: float = 2.0,
        alpha: float = 8.0,
        order: int = 6,
        inter_only: bool = True,
        learn_r0: bool = True,
        max_num_neighbors: int = 512,
    ) -> None:
        super().__init__()
        if not cutoff > taper_width:
            raise ValueError(f"cutoff {cutoff} must exceed taper_width {taper_width}")
        self.params = params
        self.correction = correction
        self.cutoff = float(cutoff)
        self.taper_width = float(taper_width)
        self.alpha = float(alpha)
        self.order = int(order)
        self.inter_only = bool(inter_only)
        self.max_num_neighbors = int(max_num_neighbors)
        self.r0_raw = nn.Parameter(
            torch.tensor(_inverse_softplus(r0_init)), requires_grad=learn_r0
        )

    @property
    def r0(self) -> torch.Tensor:
        """Range-separation midpoint in Angstrom (positive by construction)."""
        return torch.nn.functional.softplus(self.r0_raw)

    def forward(self, batch, feats: LambdaFeatures) -> DispersionOutput:
        positions = batch.positions
        fragment_idx = batch.fragment_idx if self.inter_only else None
        if self.inter_only and fragment_idx is None:
            raise ValueError(
                "TTDispersion(inter_only=True) needs batch.fragment_idx; load the data "
                "with a fragment_idx column or a diabatic state library"
            )
        pair_index, r = inter_fragment_pairs(
            positions,
            batch.batch_idx,
            self.cutoff,
            fragment_idx=fragment_idx,
            max_num_neighbors=self.max_num_neighbors,
        )
        i, j = pair_index[0], pair_index[1]

        c6, b = self.params(feats.inv_feats, feats.species_idx)
        # Geometric mean, in log space so no sqrt appears (see DispersionParameterHeads).
        c6_ij = (0.5 * (c6[i].log() + c6[j].log())).exp()
        b_ij = (0.5 * (b[i].log() + b[j].log())).exp()

        e_tt = tt_damped_c6_energy(r, c6_ij, b_ij, order=self.order)
        switch = fermi_switch(r, self.r0, self.alpha)
        taper = pairwise_switch(r, self.cutoff - self.taper_width, self.cutoff)
        e_pair_ff = switch * taper * e_tt

        if self.correction is not None:
            e_pair_corr = self.correction(
                feats.inv_feats, feats.species_idx, positions, pair_index, r
            )
        else:
            e_pair_corr = torch.zeros_like(e_pair_ff)

        n = batch.n_systems
        pair_batch = batch.batch_idx[i]
        energy_ff = e_pair_ff.new_zeros(n).index_add_(0, pair_batch, e_pair_ff)
        energy_corr = e_pair_corr.new_zeros(n).index_add_(0, pair_batch, e_pair_corr)
        return DispersionOutput(
            energy=energy_ff + energy_corr,
            energy_ff=energy_ff,
            energy_corr=energy_corr,
            pair_index=pair_index,
            r=r,
            e_pair_ff=e_pair_ff,
            e_pair_corr=e_pair_corr,
            r0=self.r0,
            c6=c6,
            b=b,
        )
