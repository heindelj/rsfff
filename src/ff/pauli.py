"""Slater-damped multipolar Pauli repulsion with a learned short-range pair correction.

The energy over inter-fragment pairs is

    E = sum_{i<j} [ T(r) E_mult(r)  +  W(r) dE_ij ]

    E_mult(r) = m_j^T [ f_n(b_ij r) * T_multipole(dr) ] m_i      (atomic units)
    m_i       = [q_i, mu_i]                  the *Pauli* multipoles, emitted directly
    b_ij      = sqrt(b_i b_j)
    T(r)      = pairwise_switch(r, cutoff - taper_width, cutoff)   C2, exactly 0 at cutoff
    W(r)      = the pair head's own compact envelope

``f_n`` is the overlap complement from :mod:`rsfff.ff.multipole` -- the undamped ``1/r^n``
tail is already subtracted off -- so ``E_mult`` decays exponentially and is short-ranged by
construction. That is why, unlike :class:`~rsfff.ff.dispersion.TTDispersion`, there is **no
Fermi range separation here**: the Slater form is a valid description all the way in, and
the correction head's envelope ``W`` is already at full strength at short range, so the
delta is carried where it is needed without an extra parameter competing for the same
mid-range energy (``docs/range_separated_mlip.md`` §4.5).

Departure from pyCMM (``cmm/force_field.py:918``), which builds the Pauli multipoles by
scaling the *real* electrostatic multipoles with fitted ``K_dipo``/``K_quad`` factors: here
the parameterizer emits ``q`` and ``mu`` directly, ``mu`` from the equivariant lambda=1
features, so no local axis frame is ever constructed. Quadrupoles and beyond are left to
the correction network for now (``max_rank``).

Intramolecular pairs are excluded by the hard mask in
:func:`~rsfff.ff.pairs.inter_fragment_pairs`, not by a distance cutoff: measured on this
dataset, intra H-H reaches 1.688 A while inter O-H reaches down to 1.552 A, so no scalar
threshold separates them. Fragmentation is a graph property, and the eventual reactive form
replaces the mask with a learned per-pair weight (``pair_weight``), not with a radius.

Units
-----
================  ==============  ===========================================
quantity          unit            note
================  ==============  ===========================================
positions, r      Angstrom        model convention (rsfff.train.data)
cutoff, taper     Angstrom
q                 e               Pauli monopole (positive: repulsive)
mu                e * bohr        Pauli dipole
b                 1 / bohr        Slater exponent
energy            Hartree
================  ==============  ===========================================

The single Angstrom -> bohr conversion happens once in :meth:`SlaterPauli.forward`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..features.features import LambdaFeatures
from ..mlip.heads import two_slot_mlp, zero_init_readout
from ..mlip.pair_heads import PairEnergyHead
from ..mlip.response_heads import AtomicQuadrupoleHead, AtomicVectorHead
from ..mlip.switch import pairwise_switch
from .multipole import (
    build_polytensor,
    damped_interaction_tensor,
    irrep2_to_spherical,
    multipole_pair_energy,
    slater_two_center_damp,
    spherical_to_cartesian_quadrupole,
)
from .pairs import inter_fragment_pairs
from .units import BOHR_ANG

#: Per-element ``(q_pauli [e], b_pauli [1/bohr])`` priors from the fitted CMM water model
#: (``pyCMM/tests/data/water.xml:26-27``). Priors, not ground truth: they were fit against
#: a different reference than the Q-Chem ALMO-EDA components trained against here, and
#: measured against ``eda_mod_pauli`` they reproduce the *shape* well (correlation 0.991)
#: but the *scale* only to ~18% (regression slope 0.82 charges-only). Two learnable
#: per-species log deviations absorb most of that.
DEFAULT_PAULI_PRIOR: dict[int, tuple[float, float]] = {
    8: (6.50923, 2.1975),      # O
    1: (0.527804, 1.96474),    # H
}

#: Per-element Pauli dipole magnitude scale in e*bohr, used only to set the output scale of
#: the (zero-initialized) dipole head so its effective learning rate matches the parameter
#: heads it competes with -- the same role ``PairEnergyHead.energy_scale`` plays. Values are
#: ``|K_dipo_pauli * mu_real|`` from the same water model: O 5.61925*0.094298, H
#: 0.515584*|(0.0910288, 0, -0.207851)|.
DEFAULT_PAULI_DIPOLE_SCALE: dict[int, float] = {8: 0.530, 1: 0.117}

#: Per-element Pauli quadrupole magnitude scale in e*bohr^2, same role as the dipole scale.
#: ``|K_quad_pauli| * ||Theta_real||_F`` from ``water.xml``: O 1.56567*1.1398, H 0.440164*0.1456.
DEFAULT_PAULI_QUAD_SCALE: dict[int, float] = {8: 1.785, 1: 0.064}


def build_pauli_priors(
    neighbor_types,
    *,
    q_prior: dict[int, float] | None = None,
    b_prior: dict[int, float] | float | None = None,
    dipole_scale: dict[int, float] | float | None = None,
    quad_scale: dict[int, float] | float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-species ``(log q, log b, dipole_scale, quad_scale)`` ordered like ``neighbor_types``.

    An element absent from :data:`DEFAULT_PAULI_PRIOR` raises rather than falling back to
    an average: a guessed Pauli charge produces a plausible-looking model that is quietly
    wrong, and Pauli is the largest EDA component. Pass the overrides to extend the table.
    """
    q_table = {z: q for z, (q, _) in DEFAULT_PAULI_PRIOR.items()}
    b_table = {z: b for z, (_, b) in DEFAULT_PAULI_PRIOR.items()}
    if q_prior:
        q_table.update(q_prior)
    if isinstance(b_prior, dict):
        b_table.update(b_prior)
    elif b_prior is not None:
        b_table = {int(z): float(b_prior) for z in neighbor_types}

    missing = [int(z) for z in neighbor_types if int(z) not in q_table or int(z) not in b_table]
    if missing:
        raise KeyError(
            f"no Pauli prior for atomic number(s) {missing}; extend DEFAULT_PAULI_PRIOR or "
            f"pass q_prior/b_prior={{Z: value}}. Refusing to guess."
        )

    def _scales(default: dict[int, float], override) -> torch.Tensor:
        table = dict(default)
        if isinstance(override, dict):
            table.update(override)
        elif override is not None:
            table = {int(z): float(override) for z in neighbor_types}
        # A multipole scale is only a learning-rate convenience, so an unknown element gets
        # the mean of the known ones rather than an exception.
        fallback = sum(table.values()) / max(len(table), 1)
        return torch.tensor(
            [float(table.get(int(z), fallback)) for z in neighbor_types]
        )

    log_q = torch.tensor([float(q_table[int(z)]) for z in neighbor_types]).log()
    log_b = torch.tensor([float(b_table[int(z)]) for z in neighbor_types]).log()
    mu_scale = _scales(DEFAULT_PAULI_DIPOLE_SCALE, dipole_scale)
    quad_scale_t = _scales(DEFAULT_PAULI_QUAD_SCALE, quad_scale)
    return log_q, log_b, mu_scale, quad_scale_t


class PauliMultipoleHeads(nn.Module):
    """Per-atom Pauli multipoles ``(q, b, mu)`` from the atomic features.

    ``q`` and ``b`` are parameterized in **log space** as deviations from a per-species
    prior, exactly as :class:`~rsfff.ff.dispersion.DispersionParameterHeads` does it: the
    ``sqrt(b_i b_j)`` combination rule is linear in logs, positivity is structural rather
    than a floor, and -- specific to Pauli -- ``q > 0`` makes the monopole-monopole term
    repulsive by construction. Because the learnable tensors are deviations, plain
    ``weight_decay`` on them is regularization toward the pyCMM priors.

    ``mu`` comes from :class:`~rsfff.mlip.response_heads.AtomicVectorHead` on the lambda=1
    features, which are odd-parity (a true polar vector, verified exactly under inversion)
    and equivariant to machine precision. That head is zero-initialized, so at
    initialization ``mu`` is *exactly* zero and the rank-1 model reproduces the rank-0 model
    to round-off: the dipole is a pure addition on top of a working charge backbone rather
    than a perturbation of it. (Not bitwise -- the rank-1 contraction sums a 4x4 block whose
    accumulation order BLAS chooses -- but to ~1e-16 relative.)

    ``environment_q`` / ``environment_b`` let the parameters respond to the surroundings,
    making them *effective* rather than isolated-atom quantities. For dispersion that was
    worth a factor of ~5 in accuracy; here it is the same one-line ablation.
    """

    def __init__(
        self,
        p0: int,
        p1: int | None,
        n_species: int,
        *,
        log_q_prior: torch.Tensor,      # (n_species,)
        log_b_prior: torch.Tensor,      # (n_species,)
        dipole_scale: torch.Tensor,     # (n_species,)
        p2: int | None = None,
        p_env: int = 0,
        p1_env: int = 0,
        p2_env: int = 0,
        quad_scale: torch.Tensor | None = None,     # (n_species,)
        irrep2_to_spherical: torch.Tensor | None = None,   # (5, 5)
        emb_dim: int = 16,
        hidden: int = 64,
        depth: int = 2,
        equiv_channels: int = 32,
        max_rank: int = 1,
        learn_q: bool = True,
        environment_q: bool = True,
        learn_b: bool = True,
        environment_b: bool = False,
        learn_dipole: bool = True,
        learn_quadrupole: bool = True,
    ) -> None:
        super().__init__()
        self.max_rank = int(max_rank)
        self.register_buffer("log_q_prior", log_q_prior.clone())
        self.register_buffer("log_b_prior", log_b_prior.clone())
        self.register_buffer("dipole_scale", dipole_scale.clone())
        if quad_scale is None:
            quad_scale = torch.ones_like(dipole_scale)
        self.register_buffer("quad_scale", quad_scale.clone())
        self.species_emb = nn.Embedding(n_species, emb_dim)
        self.environment_q = bool(environment_q)
        self.environment_b = bool(environment_b)

        self.d_log_q = nn.Parameter(torch.zeros(n_species), requires_grad=learn_q)
        self.d_log_b = nn.Parameter(torch.zeros(n_species), requires_grad=learn_b)

        # `p0`/`p1`/`p2` are the **fragment** slot widths, `*_env` the environment slot's. The
        # input is `[ h | eta | emb ]`, embedding last -- see `two_slot_mlp`. All the `*_env`
        # default to 0, which rebuilds this head exactly as it was.
        self.q_mlp = (
            two_slot_mlp(p0, p_env, hidden, depth, 1, p_tail=emb_dim)
            if environment_q else None
        )
        self.b_mlp = (
            two_slot_mlp(p0, p_env, hidden, depth, 1, p_tail=emb_dim)
            if environment_b else None
        )
        for m in (self.q_mlp, self.b_mlp):
            if m is not None:   # start at exactly the per-species prior
                zero_init_readout(m)

        self.dipole_head = None
        if self.max_rank >= 1 and learn_dipole:
            if p1 is None:
                raise ValueError(
                    "Pauli dipoles need lambda=1 features; add 1 to features.selected_lambdas "
                    "(e.g. [0, 1, 2]) or set max_rank: 0 / learn_dipole: false"
                )
            self.dipole_head = AtomicVectorHead(
                p0, p1, emb_dim, p_env=p_env, p1_env=p1_env,
                hidden=hidden, depth=depth, equiv_channels=equiv_channels,
            )

        self.quadrupole_head = None
        if self.max_rank >= 2 and learn_quadrupole:
            if p2 is None:
                raise ValueError(
                    "Pauli quadrupoles need lambda=2 features; add 2 to "
                    "features.selected_lambdas or set max_rank <= 1 / learn_quadrupole: false"
                )
            if irrep2_to_spherical is None:
                raise ValueError(
                    "max_rank=2 needs the irrep2_to_spherical change of basis; build it "
                    "with rsfff.ff.multipole.irrep2_to_spherical(backend.irrep6_to_voigt())"
                )
            self.quadrupole_head = AtomicQuadrupoleHead(
                p0, p2, emb_dim, irrep2_to_spherical, p_env=p_env, p2_env=p2_env,
                hidden=hidden, depth=depth, equiv_channels=equiv_channels,
            )

    def forward(
        self,
        inv_feats: torch.Tensor,                  # (N, p0)
        species_idx: torch.Tensor,                # (N,)
        vec_feats: torch.Tensor | None = None,    # (N, 3, p1)
        equiv_feats: torch.Tensor | None = None,  # (N, 5, p2)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """``(q (N,) e, b (N,) 1/bohr, mu (N,3) e*bohr, quad_s (N,5) e*bohr^2)``.

        ``quad_s`` is the *spherical* quadrupole ``(q20, q21c, q21s, q22c, q22s)``;
        :class:`SlaterPauli` converts it to Cartesian for the contraction.
        """
        s = species_idx
        log_q = self.log_q_prior[s] + self.d_log_q[s]
        log_b = self.log_b_prior[s] + self.d_log_b[s]
        emb = self.species_emb(s)
        if self.q_mlp is not None or self.b_mlp is not None:
            x = torch.cat((inv_feats, emb), dim=-1)
            if self.q_mlp is not None:
                log_q = log_q + self.q_mlp(x).squeeze(-1)
            if self.b_mlp is not None:
                log_b = log_b + self.b_mlp(x).squeeze(-1)

        mu = None
        if self.dipole_head is not None:
            if vec_feats is None:
                raise ValueError(
                    "PauliMultipoleHeads was built with a dipole head but the featurizer "
                    "returned no lambda=1 features; set features.selected_lambdas: [0, 1, 2]"
                )
            mu = self.dipole_head(inv_feats, emb, vec_feats) * self.dipole_scale[s].unsqueeze(-1)

        quad_s = None
        if self.quadrupole_head is not None:
            if equiv_feats is None:
                raise ValueError(
                    "PauliMultipoleHeads was built with a quadrupole head but the "
                    "featurizer returned no lambda=2 features"
                )
            quad_s = (
                self.quadrupole_head(inv_feats, emb, equiv_feats)
                * self.quad_scale[s].unsqueeze(-1)
            )
        return log_q.exp(), log_b.exp(), mu, quad_s


def slater_pauli_pair_energy(
    dr_au: torch.Tensor,        # (P, 3) r_j - r_i, bohr
    r_au: torch.Tensor,         # (P,) bohr
    poly_i: torch.Tensor,       # (P, K) Pauli polytensor on i
    poly_j: torch.Tensor,       # (P, K) Pauli polytensor on j
    b_ij: torch.Tensor,         # (P,) combined Slater exponent, 1/bohr
    max_rank: int = 1,
) -> torch.Tensor:
    """Undamped-by-distance Slater multipole repulsion per pair: ``(P,)`` Hartree.

    The analytic core of :meth:`SlaterPauli.forward`, factored out so a caller that owns its
    own pair list and gating -- :class:`rsfff.ff.unified.UnifiedPairModel` -- gets bit-identical
    energies without duplicating the contraction. Everything here is atomic units and pure:
    no neighbor search, no taper, no switch, no module state.

    ``poly_i``/``poly_j`` are already gathered onto pairs (``poly[i]``, ``poly[j]``), because
    the unified model builds them from a different feature stream than this module does.
    """
    damp = slater_two_center_damp(b_ij * r_au, max_rank)
    tensor = damped_interaction_tensor(dr_au, damp, 1.0 / r_au, max_rank=max_rank)
    return multipole_pair_energy(poly_i, poly_j, tensor)


@dataclass
class PauliOutput:
    """Per-system energies plus the per-pair breakdown the diagnostics need."""

    energy: torch.Tensor         # (B,) total Pauli energy, Hartree
    energy_ff: torch.Tensor      # (B,) tapered Slater multipole part
    energy_corr: torch.Tensor    # (B,) learned pair-correction part
    pair_index: torch.Tensor     # (2, P)
    r: torch.Tensor              # (P,) Angstrom
    e_pair_ff: torch.Tensor      # (P,)
    e_pair_corr: torch.Tensor    # (P,)
    q: torch.Tensor              # (N,)
    b: torch.Tensor              # (N,)
    mu: torch.Tensor | None      # (N, 3) e*bohr
    quad_s: torch.Tensor | None  # (N, 5) spherical (q20, q21c, q21s, q22c, q22s), e*bohr^2


class SlaterPauli(nn.Module):
    """Slater-damped multipolar Pauli repulsion over inter-fragment pairs.

    Args
    ----
    cutoff        : pair-list cutoff in Angstrom. 7.0 is ample -- the Slater form is
                    exponential and ``f1(b r)/r`` there is ~1e-8 Ha, against a 1e-2 Ha
                    signal. The energy is still tapered to exactly zero over the last
                    Angstrom, because a truncation puts a step in the energy and a delta in
                    the force however small it is.
    max_rank      : 0 = charges only, 1 = charges + dipoles. 2 raises with a pointer to the
                    pyCMM lines to port.
    inter_only    : restrict to pairs in different fragments. This is what makes the sum an
                    interaction energy comparable to an EDA component.
    pair_weight   : optional ``(batch, pair_index, r) -> (P,)`` weight in [0, 1] applied to
                    the whole pair contribution. The hook for the eventual learned
                    intra/inter assignment that replaces ``inter_only`` for reactive
                    systems. Left ``None`` it is the identity, so nothing changes today.
                    **When it becomes learnable it needs its own supervision** (the fragment
                    assignment / diabatic state): the energy loss alone cannot constrain it,
                    and an unsupervised multiplicative weight on every pair is exactly the
                    gauge leakage ``docs/range_separated_mlip.md`` §7 warns about.
    """

    def __init__(
        self,
        params: PauliMultipoleHeads,
        correction: PairEnergyHead | None,
        *,
        cutoff: float = 7.0,
        taper_width: float = 1.0,
        max_rank: int = 1,
        inter_only: bool = True,
        max_num_neighbors: int = 512,
        pair_weight=None,
    ) -> None:
        super().__init__()
        if not cutoff > taper_width:
            raise ValueError(f"cutoff {cutoff} must exceed taper_width {taper_width}")
        self.params = params
        self.correction = correction
        self.cutoff = float(cutoff)
        self.taper_width = float(taper_width)
        self.max_rank = int(max_rank)
        self.inter_only = bool(inter_only)
        self.max_num_neighbors = int(max_num_neighbors)
        self.pair_weight = pair_weight

    def forward(self, batch, feats: LambdaFeatures) -> PauliOutput:
        positions = batch.positions
        fragment_idx = batch.fragment_idx if self.inter_only else None
        if self.inter_only and fragment_idx is None:
            raise ValueError(
                "SlaterPauli(inter_only=True) needs batch.fragment_idx; load the data with "
                "a fragment_idx column or a diabatic state library"
            )
        pair_index, r = inter_fragment_pairs(
            positions,
            batch.batch_idx,
            self.cutoff,
            fragment_idx=fragment_idx,
            max_num_neighbors=self.max_num_neighbors,
        )
        i, j = pair_index[0], pair_index[1]

        q, b, mu, quad_s = self.params(
            feats.inv_feats, feats.species_idx, feats.vec_feats, feats.equiv_feats
        )
        # Geometric mean in log space, so no sqrt (whose derivative is unbounded at 0).
        b_ij = (0.5 * (b[i].log() + b[j].log())).exp()

        # The one Angstrom -> bohr conversion; everything below is atomic units.
        dr_au = (positions[j] - positions[i]) / BOHR_ANG
        r_au = r / BOHR_ANG
        # Spherical -> Cartesian happens here, once, at the boundary of the contraction.
        quad_c = None if quad_s is None else spherical_to_cartesian_quadrupole(quad_s)
        poly = build_polytensor(q, mu, quad_c, max_rank=self.max_rank)
        e_mult = slater_pauli_pair_energy(
            dr_au, r_au, poly[i], poly[j], b_ij, max_rank=self.max_rank
        )

        taper = pairwise_switch(r, self.cutoff - self.taper_width, self.cutoff)
        e_pair_ff = taper * e_mult

        if self.correction is not None:
            e_pair_corr = self.correction(
                feats.inv_feats, feats.species_idx, positions, pair_index, r
            )
        else:
            e_pair_corr = torch.zeros_like(e_pair_ff)

        if self.pair_weight is not None:
            w = self.pair_weight(batch, pair_index, r)
            e_pair_ff = w * e_pair_ff
            e_pair_corr = w * e_pair_corr

        n = batch.n_systems
        pair_batch = batch.batch_idx[i]
        energy_ff = e_pair_ff.new_zeros(n).index_add_(0, pair_batch, e_pair_ff)
        energy_corr = e_pair_corr.new_zeros(n).index_add_(0, pair_batch, e_pair_corr)
        return PauliOutput(
            energy=energy_ff + energy_corr,
            energy_ff=energy_ff,
            energy_corr=energy_corr,
            pair_index=pair_index,
            r=r,
            e_pair_ff=e_pair_ff,
            e_pair_corr=e_pair_corr,
            q=q,
            b=b,
            mu=mu,
            quad_s=quad_s,
        )


class PauliModel(nn.Module):
    """Featurizer + :class:`SlaterPauli` as one module.

    Same rationale as :class:`~rsfff.ff.dispersion.DispersionModel`: standalone, the
    featurizer carries learnable parameters of its own
    (``FlatLambdaSOAPFeaturizer.channel_proj``), so it has to be owned by something for
    ``parameters()``, ``state_dict()`` and ``.to()`` to cover it.

    ``intra_fragment`` restricts each atom's descriptor to its own monomer, making the
    emitted Pauli parameters independent of the surroundings and the pair sum rigorously
    two-body -- the ablation that says whether environment dependence is earning its keep.
    """

    def __init__(self, featurizer, pauli: SlaterPauli, *, intra_fragment: bool = False):
        super().__init__()
        self.featurizer = featurizer
        self.pauli = pauli
        self.intra_fragment = bool(intra_fragment)

    @property
    def params(self) -> PauliMultipoleHeads:
        return self.pauli.params

    def forward(self, batch) -> PauliOutput:
        group = batch.fragment_idx if self.intra_fragment else None
        return self.pauli(batch, self.featurizer(batch, group))
