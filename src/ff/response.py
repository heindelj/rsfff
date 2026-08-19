"""The per-fragment response solve: response parameters in, atomic multipoles out.

This is the piece the 1-body term and the electrostatics term **share**, and it is a module
of its own precisely so they can share one instance rather than each carrying a parameter
set that drifts away from the other's.

    E_internal(f) = SQE charge sector + on-site dipole sector + on-site quadrupole sector
    q_i           from chi/eta through the split-charge solve
    mu_i          = -alpha_i chivec_i
    Theta_i       = -C_i chiquad_i

The split between the two consumers is exactly the split between 1-body and 2-body energy:

* the **multipoles** are what interacts, so they belong to
  :class:`rsfff.ff.electrostatics.SlaterElectrostatics`, which sums their interaction over
  inter-fragment pairs;
* the **internal energy** -- the cost of arranging those multipoles inside one monomer --
  is 1-body, so it belongs to :class:`rsfff.ff.onebody.OneBodyEnergy`.

That is why ``SlaterElectrostatics`` computes ``internal_energy`` and deliberately excludes
it from the interaction: not because it is unphysical, but because it is somebody else's.

**Everything here is grouped by ``fragment_idx``.** The descriptor is intra-fragment and the
solve is per fragment, so a monomer's parameters, multipoles and internal energy are
functions of that monomer alone -- no environment enters. That is what makes the
electrostatics exactly two-body and the 1-body term exactly one-body, and both are checked
rather than assumed (``tests/test_ff_electrostatics.py``, ``tests/test_ff_onebody.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..features.features import LambdaFeatures
from ..mlip.eem import atomic_dipoles, atomic_quadrupoles
from ..mlip.heads import mlp, zero_init_readout
from ..mlip.response_heads import (
    AtomicAlphaHead,
    AtomicQuadrupoleHead,
    AtomicVectorHead,
    AxialQuadrupolePolarizabilityHead,
    voigt_vector_to_symmetric_matrix,
)
from ..mlip.sqe import (
    PairComplianceHead,
    atomic_dipole_energy,
    atomic_quadrupole_energy,
    sqe_solve,
)
from .multipole import l2_rotation_generators
from .pairs import intra_fragment_channels
from .units import BOHR_ANG

#: Per-element ``(Z_cp [e], b_elec [1/bohr])`` charge-penetration priors from the fitted CMM
#: water model (``pyCMM/tests/data/water.xml``, the ``<ChargePenetration>`` block). ``Z_cp``
#: is an *effective* nuclear charge screening the core, not the atomic number.
DEFAULT_ELEC_PRIOR: dict[int, tuple[float, float]] = {
    8: (3.61565, 2.13358),   # O
    1: (0.93619, 2.33322),   # H
}

#: Per-element **baseline** charges ``q0``, from the same water model's permanent monopoles.
#: These are not the prediction -- the SQE solve moves off them -- but the starting point
#: matters more than usual here, and not for numerical reasons:
#:
#: the point-multipole energy is *even* in the charges (``q_i q_j``) while the penetration
#: cross term ``Z_i (q_j - Z_j)`` is *odd*, so around ``q = 0`` the loss surface is nearly
#: symmetric under ``q -> -q`` and the descent direction is set by penetration rather than by
#: chemistry. Measured on 150 dimers by scaling pyCMM's charges rigidly: ``q_O = -0.78``
#: gives MAE 2.96 kJ/mol (the physical branch) while ``q_O = +0.20`` gives 10.18 and
#: ``q_O = 0`` gives 12.23. A model started near zero can and does slide onto the
#: positive-oxygen branch, fitting tolerably while being chemically backwards.
#:
#: Starting from a physical baseline puts the optimizer on the right side of that saddle.
#: This is what SQE's ``q0`` is *for* -- it is the diabatic baseline, and ``MonomerModel``
#: uses ``ReferenceEmbedding.baseline_charge`` the same way.
DEFAULT_Q0_PRIOR: dict[int, float] = {8: -0.390896, 1: 0.195448}


def build_elec_priors(
    neighbor_types,
    *,
    z_prior: dict[int, float] | None = None,
    b_prior: dict[int, float] | float | None = None,
    q0_prior: dict[int, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-species ``(log Z, log b, q0)`` prior tensors ordered like ``neighbor_types``.

    An element absent from :data:`DEFAULT_ELEC_PRIOR` raises rather than falling back to an
    average: penetration carries roughly half of this energy component, so a guessed ``Z``
    produces a model that is quietly and substantially wrong.
    """
    z_table = {z: v for z, (v, _) in DEFAULT_ELEC_PRIOR.items()}
    b_table = {z: v for z, (_, v) in DEFAULT_ELEC_PRIOR.items()}
    if z_prior:
        z_table.update(z_prior)
    if isinstance(b_prior, dict):
        b_table.update(b_prior)
    elif b_prior is not None:
        b_table = {int(z): float(b_prior) for z in neighbor_types}

    missing = [int(z) for z in neighbor_types if int(z) not in z_table or int(z) not in b_table]
    if missing:
        raise KeyError(
            f"no charge-penetration prior for atomic number(s) {missing}; extend "
            f"DEFAULT_ELEC_PRIOR or pass z_prior/b_prior={{Z: value}}. Refusing to guess."
        )
    q0_table = dict(DEFAULT_Q0_PRIOR)
    if q0_prior:
        q0_table.update(q0_prior)
    log_z = torch.tensor([float(z_table[int(z)]) for z in neighbor_types]).log()
    log_b = torch.tensor([float(b_table[int(z)]) for z in neighbor_types]).log()
    # A missing q0 prior is only a starting point, so it falls back to zero rather than
    # raising -- but it costs accuracy (see DEFAULT_Q0_PRIOR), so say so.
    q0 = torch.tensor([float(q0_table.get(int(z), 0.0)) for z in neighbor_types])
    return log_z, log_b, q0


class ElectrostaticParameterHeads(nn.Module):
    """The response parameters, plus the two penetration parameters.

    ``chi``/``eta`` drive the SQE charge solve; ``chivec``/``alpha`` give the atomic dipoles
    as ``mu_i = -alpha_i chivec_i``. Both pairs follow
    :class:`~rsfff.mlip.monomer.MonomerParameterHeads`: a per-species bias plus a
    zero-initialized environment MLP, with ``eta`` through ``softplus + floor`` so the charge
    problem stays strictly convex and the solve is a genuine minimum.

    ``Z``/``b`` are log-space deviations from :data:`DEFAULT_ELEC_PRIOR`, the pattern used by
    the dispersion and Pauli heads: positivity is structural and plain ``weight_decay`` on
    the deviations is regularization toward the fitted classical values.

    The conditioning vector is a plain species embedding, not the Phase-1
    ``ReferenceEmbedding`` -- this is the standalone featurizer path, and the charge/spin
    conditioning that embedding provides is not available (or needed) for neutral monomers.
    """

    def __init__(
        self,
        p0: int,
        p1: int | None,
        p2: int | None,
        n_species: int,
        *,
        log_z_prior: torch.Tensor,      # (n_species,)
        log_b_prior: torch.Tensor,      # (n_species,)
        irrep6_to_voigt: torch.Tensor,
        q0_prior: torch.Tensor | None = None,   # (n_species,) baseline charges
        irrep2_to_spherical_map: torch.Tensor | None = None,
        emb_dim: int = 16,
        hidden: int = 64,
        depth: int = 2,
        equiv_channels: int = 32,
        max_rank: int = 1,
        chi_init: torch.Tensor | None = None,
        eta_init: torch.Tensor | None = None,
        eta_floor: float = 0.05,
        psd_floor: float = 1e-4,
        environment_chi: bool = True,
        environment_eta: bool = True,
        learn_z: bool = True,
        environment_z: bool = False,
        learn_b: bool = True,
        environment_b: bool = False,
        learn_dipole: bool = True,
        learn_quadrupole: bool = True,
        cquad_init: float = 1.0,
        cquad_floor: float = 1.0e-4,
        environment_cquad: bool = False,
        anisotropic_cquad: bool = False,
    ) -> None:
        super().__init__()
        self.max_rank = int(max_rank)
        self.eta_floor = float(eta_floor)
        self.species_emb = nn.Embedding(n_species, emb_dim)
        self.register_buffer("log_z_prior", log_z_prior.clone())
        self.register_buffer("log_b_prior", log_b_prior.clone())
        if q0_prior is None:
            q0_prior = torch.zeros_like(log_z_prior)
        self.register_buffer("q0_prior", q0_prior.clone())

        # --- charge sector -------------------------------------------------------------
        chi0 = torch.zeros(n_species) if chi_init is None else chi_init.clone()
        eta0 = (
            torch.full((n_species,), 0.5) if eta_init is None else eta_init.clone()
        ).clamp(min=self.eta_floor + 1e-6)
        self.chi0 = nn.Parameter(chi0)
        self.eta0_raw = nn.Parameter(torch.log(torch.expm1(eta0 - self.eta_floor)))
        self.chi_mlp = mlp(p0 + emb_dim, hidden, depth, 1) if environment_chi else None
        self.eta_mlp = mlp(p0 + emb_dim, hidden, depth, 1) if environment_eta else None

        # --- penetration ---------------------------------------------------------------
        self.d_log_z = nn.Parameter(torch.zeros(n_species), requires_grad=learn_z)
        self.d_log_b = nn.Parameter(torch.zeros(n_species), requires_grad=learn_b)
        self.z_mlp = mlp(p0 + emb_dim, hidden, depth, 1) if environment_z else None
        self.b_mlp = mlp(p0 + emb_dim, hidden, depth, 1) if environment_b else None

        for m in (self.chi_mlp, self.eta_mlp, self.z_mlp, self.b_mlp):
            if m is not None:   # start at exactly the per-species value
                zero_init_readout(m)

        # --- dipole sector: mu_i = -alpha_i chivec_i ------------------------------------
        self.chivec_head = None
        self.alpha_head = None
        if self.max_rank >= 1 and learn_dipole:
            if p1 is None:
                raise ValueError(
                    "atomic dipoles need lambda=1 features; set "
                    "features.selected_lambdas: [0, 1, 2] (or max_rank: 0)"
                )
            self.chivec_head = AtomicVectorHead(
                p0, p1, emb_dim, hidden=hidden, depth=depth, equiv_channels=equiv_channels
            )
            self.alpha_head = AtomicAlphaHead(
                p0, p2, emb_dim, irrep6_to_voigt, hidden=hidden, depth=depth,
                equiv_channels=equiv_channels, positive_isotropic=True, psd_floor=psd_floor,
            )

        # --- quadrupole sector: Theta_i = -C_i chiquad_i, C_i = c_i I5 ------------------
        # The isotropic `c` follows the `eta` construction: one positive scalar, with
        # softplus + floor making both the positivity and the equivariance structural (a
        # multiple of the identity commutes with every Wigner-D).
        #
        # `anisotropic_cquad` replaces it with the axial form of
        # `AxialQuadrupolePolarizabilityHead`: three positive eigenvalues (m = 0, |m| = 1,
        # |m| = 2) about a learned axis. Not the general symmetric map on the l=2 space,
        # which is fifteen numbers (Sym^2(l=2) = 0 + 2 + 4) and would need lambda=4 features.
        #
        # **It only earns its place once there is a field gradient.** At zero field the
        # quadrupole is `Theta = -C chiquad` with `chiquad` a free equivariant vector and `C`
        # invertible, so *every* Theta is reachable either way and the multipole labels cannot
        # tell the two apart -- only the internal energy `-1/2 chiquad^T C chiquad` differs,
        # and that is degenerate with the rest of the 1-body term. With a field gradient the
        # response `Theta = C (gradF - chiquad)` is genuinely directional, so the anisotropy
        # becomes a real degree of freedom. That is why it arrives with polarization and not
        # before.
        self.chiquad_head = None
        self.cquad_axis_head = None
        if self.max_rank >= 2 and learn_quadrupole:
            if p2 is None or irrep2_to_spherical_map is None:
                raise ValueError(
                    "max_rank=2 needs lambda=2 features and the irrep2_to_spherical map"
                )
            self.chiquad_head = AtomicQuadrupoleHead(
                p0, p2, emb_dim, irrep2_to_spherical_map,
                hidden=hidden, depth=depth, equiv_channels=equiv_channels,
            )
            self.cquad_floor = float(cquad_floor)
            # Three eigenvalues (m = 0, |m| = 1, |m| = 2) when anisotropic, one when not.
            # Initialized equal, so the anisotropic head reproduces the isotropic one exactly
            # at initialization and the axis stays inert until the fit separates them.
            self.anisotropic_cquad = bool(anisotropic_cquad)
            # The isotropic table keeps its original ``(n_species,)`` shape rather than
            # becoming ``(n_species, 1)``: a trailing axis of size one would be cosmetic here
            # and would make every existing checkpoint's `cquad0_raw` shape-incompatible, so
            # `warm_start` would drop it and silently reset a trained parameter.
            n_c = 3 if self.anisotropic_cquad else 1
            shape = (n_species, n_c) if self.anisotropic_cquad else (n_species,)
            c0 = torch.full(shape, max(float(cquad_init) - self.cquad_floor, 1e-6))
            self.cquad0_raw = nn.Parameter(torch.log(torch.expm1(c0)))
            self.cquad_mlp = (
                mlp(p0 + emb_dim, hidden, depth, n_c) if environment_cquad else None
            )
            if self.cquad_mlp is not None:
                zero_init_readout(self.cquad_mlp)
            if self.anisotropic_cquad:
                if p1 is None:
                    raise ValueError(
                        "anisotropic_cquad needs lambda=1 features for the polarizability "
                        "axis; set features.selected_lambdas: [0, 1, 2]"
                    )
                self.cquad_axis_head = AxialQuadrupolePolarizabilityHead(
                    p0, p1, emb_dim, l2_rotation_generators(),
                    hidden=hidden, depth=depth, equiv_channels=equiv_channels,
                )

    def forward(self, feats: LambdaFeatures):
        """``(chi, eta, chivec, alpha, Z, b, chiquad, cquad)`` per atom."""
        s = feats.species_idx
        emb = self.species_emb(s)
        x = torch.cat((feats.inv_feats, emb), dim=-1)

        chi = self.chi0[s]
        if self.chi_mlp is not None:
            chi = chi + self.chi_mlp(x).squeeze(-1)
        eta_raw = self.eta0_raw[s]
        if self.eta_mlp is not None:
            eta_raw = eta_raw + self.eta_mlp(x).squeeze(-1)
        eta = torch.nn.functional.softplus(eta_raw) + self.eta_floor

        log_z = self.log_z_prior[s] + self.d_log_z[s]
        log_b = self.log_b_prior[s] + self.d_log_b[s]
        if self.z_mlp is not None:
            log_z = log_z + self.z_mlp(x).squeeze(-1)
        if self.b_mlp is not None:
            log_b = log_b + self.b_mlp(x).squeeze(-1)

        chivec = alpha = None
        if self.chivec_head is not None:
            chivec = self.chivec_head(feats.inv_feats, emb, feats.vec_feats)
            alpha = voigt_vector_to_symmetric_matrix(
                self.alpha_head(feats.inv_feats, emb, feats.equiv_feats)
            )
        chiquad = cquad = None
        if self.chiquad_head is not None:
            chiquad = self.chiquad_head(feats.inv_feats, emb, feats.equiv_feats)
            cquad_raw = self.cquad0_raw[s]                       # (N,) or (N, 3)
            if self.cquad_mlp is not None:
                delta = self.cquad_mlp(x)                        # (N, 1) or (N, 3)
                cquad_raw = cquad_raw + (
                    delta if self.anisotropic_cquad else delta.squeeze(-1)
                )
            c = torch.nn.functional.softplus(cquad_raw) + self.cquad_floor
            if self.cquad_axis_head is None:
                cquad = c                                        # (N,) isotropic
            else:
                # (N, 5, 5): three eigenvalues about a learned axis. `atomic_quadrupoles` and
                # `atomic_quadrupole_energy` already accept either shape.
                cquad = self.cquad_axis_head(
                    feats.inv_feats, emb, feats.vec_feats, c
                )
        return chi, eta, chivec, alpha, log_z.exp(), log_b.exp(), chiquad, cquad


@dataclass
class ResponseParameters:
    """Everything the response solve needs, *before* any solve happens.

    Split out from :meth:`FragmentResponse.forward` because the polarized and charge-transfer
    levels need the parameters without the frozen solve that normally accompanies them: they
    evaluate the same heads on the environment-aware descriptor and hand the result to
    :func:`rsfff.ff.polarization.coupled_response`, which minimizes a *different* functional.
    Re-running the whole ``forward`` and discarding its multipoles would be both wasteful and
    misleading -- those multipoles are not the ones that level uses.

    ``q0`` already carries the per-fragment shift that makes it sum exactly to the formal
    charge, so SQE conserves that identically whatever the compliances do.
    """

    chi: torch.Tensor                  # (N,)
    eta: torch.Tensor                  # (N,) strictly positive
    q0: torch.Tensor                   # (N,)
    compliance: torch.Tensor           # (Nb,)
    chivec: torch.Tensor | None        # (N, 3) vector electronegativity
    alpha: torch.Tensor | None         # (N, 3, 3) atomic polarizability, PSD
    chiquad: torch.Tensor | None       # (N, 5)
    cquad: torch.Tensor | None         # (N,) isotropic quadrupole polarizability
    z: torch.Tensor                    # (N,) effective nuclear charge (penetration)
    b: torch.Tensor                    # (N,) Slater exponent (penetration)
    #: The **direct** parameterization: the permanent multipoles themselves, rather than the
    #: drives that produce them. Exactly one of ``chivec``/``mu0`` is non-``None``, likewise
    #: ``chiquad``/``quad0``; see :class:`rsfff.ff.coupled_solve.CoupledSystem` for why the
    #: two are the same functional and what the change of variables buys.
    mu0: torch.Tensor | None = None    # (N, 3) e*bohr
    quad0: torch.Tensor | None = None  # (N, 5) spherical, e*bohr^2


@dataclass
class FragmentResponseOutput:
    """The solved multipoles and the internal energy, per fragment."""

    charges: torch.Tensor              # (N,) e; sums to the formal charge per fragment
    mu: torch.Tensor | None            # (N, 3) e*bohr
    quad_s: torch.Tensor | None        # (N, 5) spherical Buckingham, e*bohr^2
    #: (F,) SQE charge sector + on-site dipole sector + on-site quadrupole sector. This is
    #: **1-body** energy -- the cost of polarizing a monomer internally -- and belongs to
    #: :class:`rsfff.ff.onebody.OneBodyEnergy`, not to any interaction.
    internal_energy: torch.Tensor
    transfers: torch.Tensor            # (Nb,) split charges along the channel graph
    compliance: torch.Tensor           # (Nb,)
    chi: torch.Tensor                  # (N,)
    eta: torch.Tensor                  # (N,)
    chiquad: torch.Tensor | None       # (N, 5) the rank-2 drive
    cquad: torch.Tensor | None         # (N,) isotropic quadrupole polarizability
    z: torch.Tensor                    # (N,) effective nuclear charge (penetration)
    b: torch.Tensor                    # (N,) Slater exponent (penetration)
    #: (F, 3, 3) the fragment's molecular dipole polarizability, ``-d^2 E / dF dF`` under a
    #: uniform field, in ``e^2 Angstrom^2 / Hartree`` -- the unit
    #: :attr:`rsfff.train.data.Batch.polarizability` is loaded into. ``None`` when the solve
    #: was asked not to compute it. See :func:`fragment_polarizability`.
    polarizability: torch.Tensor | None = None


def fragment_polarizability(
    alpha_flow: torch.Tensor,             # (F, 3, 3) e^2*Angstrom^2/Ha, from sqe_solve
    alpha_atom: torch.Tensor | None,      # (N, 3, 3) a0^3, the on-site dipole polarizability
    fragment_idx: torch.Tensor,           # (N,)
    n_fragments: int,
) -> torch.Tensor:
    """The molecular polarizability of each fragment: charge flow plus on-site dipoles.

    It really is just a sum, and that is worth spelling out because it looks too easy. Under a
    uniform field ``F`` the internal energy is

        E(F) = [ chi.q + 1/2 q.eta.q + 1/2 s v^2 - sum_i q_i r_i . F ]      charge sector
             + [ 1/2 mu.alpha^-1.mu + chivec.mu - mu . F ]                  dipole sector
             + [ quadrupole sector, which couples to grad F and not to F ]

    and the two sectors share no term -- the Phase-1 functional carries no intra-fragment
    Coulomb block, so nothing couples ``q`` to ``mu``. Minimizing each independently gives
    ``-d^2E/dFdF = alpha_flow + sum_i alpha_i``, with no cross term to miss. The quadrupole
    sector responds to the field *gradient*, so a uniform field never touches it.

    Both sectors carry the *same* answer at every level of the model, which is what makes this
    a well-posed target rather than a level-dependent one: the coupled solves of
    :mod:`rsfff.ff.polarization` add the inter-fragment electrostatic operator under
    ``gate["elst"]``, and that gate is ~0 between bonded atoms, so an isolated monomer's
    response is the frozen response. The label is an isolated monomer's, so it is the frozen
    response it should be compared against.

    **Units are not uniform across the two sectors and this is where an error would hide.**
    ``alpha_flow`` comes out of :func:`rsfff.mlip.sqe.sqe_solve`, whose positions are in
    Angstrom, so it is ``e^2 Angstrom^2 / Hartree``. The on-site ``alpha_i`` multiplies a field
    conjugate to ``mu`` in ``e*bohr``, so it is ``a0^3 = e^2 bohr^2 / Hartree``. The factor
    below is that conversion, and getting it wrong is a 3.6x error that no symmetry or
    invariance check would catch.
    """
    total = alpha_flow
    if alpha_atom is not None:
        summed = alpha_flow.new_zeros(n_fragments, 3, 3).index_add_(
            0, fragment_idx, alpha_atom * (BOHR_ANG ** 2)
        )
        total = total + summed
    return total


class FragmentResponse(nn.Module):
    """Parameter heads + compliance head + the local per-fragment SQE solve.

    Grouped by ``fragment_idx``, not ``batch_idx``: the Phase-1 SQE functional has no
    inter-atomic Coulomb block, so with intra-fragment channels a frame-grouped solve would
    already be mathematically identical to independent monomer solves -- but grouping by
    fragment makes inter-fragment charge flow *structurally* impossible rather than merely
    absent, which is the property both consuming terms are built on.
    """

    def __init__(
        self,
        params: ElectrostaticParameterHeads,
        compliance_head: PairComplianceHead,
        ct_compliance_head: PairComplianceHead | None = None,
        *,
        direct_multipoles: bool = False,
    ) -> None:
        super().__init__()
        self.params = params
        #: Interpret the equivariant heads' outputs as the **permanent multipoles** rather than
        #: as the drives that produce them: ``mu = mu0`` instead of ``mu = -alpha chivec``, with
        #: ``alpha`` and ``cquad`` left to describe the *response* to a field alone.
        #:
        #: The same functional either way (``mu0 = -alpha chivec``), so the solve, the
        #: preconditioner and the adjoint are untouched and it stays quadratic. What changes is
        #: conditioning and bookkeeping:
        #:
        #: * The permanent multipole stops being a *product* of two heads. Under ``chivec`` the
        #:   dipole labels and the polarizability labels both pull on both heads, so an error in
        #:   ``alpha`` corrupts ``mu``; here each label pins one head.
        #: * ``alpha -> 0`` no longer forces ``mu -> 0``. "Not polarizable" and "not polar" are
        #:   different statements and the drive form conflates them.
        #: * The on-site dipole and quadrupole energies leave ``internal_energy`` entirely --
        #:   at the frozen level the state *is* the minimum, so there is no ``-1/2 chi^T a chi``
        #:   to report. That energy has to be carried by
        #:   :class:`rsfff.ff.atomic_energy.AtomicStateEnergy`, which is the point: it moves a
        #:   few hundred kJ/mol per fragment out of an unlabeled split and into a term that
        #:   reads the multipoles it is describing.
        #:
        #: That last consequence is not free -- see ``internal_drift_weight``, whose alarm has
        #: to follow the energy to ``e_atom`` or it ends up watching a quantity that is near
        #: zero by construction.
        self.direct_multipoles = bool(direct_multipoles)
        self.compliance_head = compliance_head
        #: Used for the *radius-derived* charge-transfer channels only, leaving
        #: ``compliance_head`` to the intra-fragment ones. See
        #: :attr:`rsfff.train.config.UnifiedConfig.separate_ct_compliance`.
        self.ct_compliance_head = ct_compliance_head

    @property
    def max_rank(self) -> int:
        return self.params.max_rank

    def response_parameters(
        self,
        batch,
        feats: LambdaFeatures,
        *,
        bond_index: torch.Tensor | None = None,
        envelope: torch.Tensor | None = None,
        ct_channels: torch.Tensor | None = None,
    ) -> ResponseParameters:
        """The heads' outputs plus ``q0`` and the compliances, with no solve.

        ``bond_index`` defaults to the complete intra-fragment graph, which is the frozen and
        polarized levels. The charge-transfer level passes
        :func:`rsfff.ff.pairs.union_channels`' wider graph and the ``envelope`` that keeps its
        radius-derived channels continuous.

        ``ct_channels`` is that graph's ``from_radius`` mask: where it is true and
        :attr:`ct_compliance_head` exists, the compliance comes from that head instead. The
        two populations are different objects -- a covalent O-H against a hydrogen bond -- and
        keeping the intra-fragment one on the frozen head is what lets that head be frozen
        after stage 1 without taking charge transfer's only lever with it.
        """
        if batch.fragment_idx is None:
            raise ValueError(
                "the response solve is per fragment but batch.fragment_idx is None; the "
                "extxyz needs a `fragment_idx` column"
            )
        positions = batch.positions
        frag = batch.fragment_idx
        n_frag = int(batch.n_fragments)

        chi, eta, chivec, alpha_i, z, b, chiquad, cquad = self.params(feats)
        if bond_index is None:
            bond_index, _ = intra_fragment_channels(frag)
        compliance = self.compliance_head(
            feats.inv_feats, positions, bond_index, envelope=envelope
        )
        if self.ct_compliance_head is not None and ct_channels is not None:
            ct_compliance = self.ct_compliance_head(
                feats.inv_feats, positions, bond_index, envelope=envelope
            )
            compliance = torch.where(ct_channels, ct_compliance, compliance)

        # Baseline charges: a per-element prior, then a uniform shift per fragment so that
        # sum_i q0_i is *exactly* the formal charge whatever the prior sums to. SQE then
        # conserves that identically, for any compliances. See DEFAULT_Q0_PRIOR for why the
        # starting point is load-bearing rather than cosmetic.
        counts = torch.bincount(frag, minlength=n_frag).to(positions.dtype)
        if batch.fragment_charge is None:
            frag_q = positions.new_zeros(n_frag)
        else:
            frag_q = batch.fragment_charge.to(positions.dtype)
        q0_species = self.params.q0_prior[feats.species_idx]
        prior_sum = q0_species.new_zeros(n_frag).index_add_(0, frag, q0_species)
        q0 = q0_species + ((frag_q - prior_sum) / counts.clamp(min=1))[frag]

        if self.direct_multipoles:
            # The heads' equivariant outputs *are* the permanent multipoles here. Exactly one
            # of each pair is passed on, so nothing downstream has to guess which form it has.
            return ResponseParameters(
                chi=chi, eta=eta, q0=q0, compliance=compliance,
                chivec=None, alpha=alpha_i, chiquad=None, cquad=cquad, z=z, b=b,
                mu0=chivec, quad0=chiquad,
            )
        return ResponseParameters(
            chi=chi, eta=eta, q0=q0, compliance=compliance,
            chivec=chivec, alpha=alpha_i, chiquad=chiquad, cquad=cquad, z=z, b=b,
        )

    def forward(
        self, batch, feats: LambdaFeatures, *, with_polarizability: bool = False
    ) -> FragmentResponseOutput:
        """The frozen solve. ``with_polarizability`` adds the molecular polarizability.

        Off by default because it is only ever compared against a label, and only the isolated
        monomer anchor carries one. It costs three extra right-hand sides in a factorization
        that already happened, so it is cheap -- but it is not free, and every cluster forward
        would pay it for a number nothing reads.
        """
        positions = batch.positions
        frag = batch.fragment_idx
        n_frag = int(batch.n_fragments)

        rp = self.response_parameters(batch, feats)
        chi, eta, alpha_i = rp.chi, rp.eta, rp.alpha
        z, b, cquad = rp.z, rp.b, rp.cquad
        # Exactly one of each pair is populated, so read whichever this model is using. The
        # branch below then decides what it *means*, which is the only place the two forms
        # differ at the frozen level.
        chivec = rp.mu0 if self.direct_multipoles else rp.chivec
        chiquad = rp.quad0 if self.direct_multipoles else rp.chiquad
        q0, compliance = rp.q0, rp.compliance
        bond_index, bond_batch = intra_fragment_channels(frag)

        sol = sqe_solve(
            chi, eta, compliance, q0, positions,
            bond_index, frag, bond_batch, n_frag,
            field=None, with_polarizability=with_polarizability,
        )
        polarizability = (
            None if sol.alpha_flow is None
            else fragment_polarizability(sol.alpha_flow, alpha_i, frag, n_frag)
        )

        mu = None
        internal = sol.energy
        # In the direct form the heads' outputs are already the frozen multipoles and the
        # on-site sectors contribute **no** internal energy: the frozen state is the minimum of
        # `1/2 (mu - mu0)^T alpha^-1 (mu - mu0)`, which is zero there by construction. The
        # ~-400 kJ/mol per fragment that `-1/2 chivec^T alpha chivec` used to put here is not
        # lost, it is handed to the atomic-energy head instead.
        if chivec is not None:
            if self.direct_multipoles:
                mu = chivec
            else:
                mu = atomic_dipoles(chivec, alpha_i, frag, None)
                internal = internal + atomic_dipole_energy(
                    chivec, alpha_i, frag, n_frag, None
                )
        quad_s = None
        if chiquad is not None:
            if self.direct_multipoles:
                quad_s = chiquad
            else:
                quad_s = atomic_quadrupoles(chiquad, cquad, frag, None)
                internal = internal + atomic_quadrupole_energy(
                    chiquad, cquad, frag, n_frag, None
                )
        return FragmentResponseOutput(
            charges=sol.charges,
            mu=mu,
            quad_s=quad_s,
            internal_energy=internal,
            transfers=sol.transfers,
            compliance=compliance,
            chi=chi,
            eta=eta,
            chiquad=chiquad,
            cquad=cquad,
            z=z,
            b=b,
            polarizability=polarizability,
        )


def _inverse_softplus(x: float) -> float:
    """``y`` such that ``softplus(y) == x``, for initializing a positive parameter."""
    return float(torch.log(torch.expm1(torch.tensor(float(x)))))


__all__ = [
    "DEFAULT_ELEC_PRIOR",
    "DEFAULT_Q0_PRIOR",
    "ElectrostaticParameterHeads",
    "FragmentResponse",
    "FragmentResponseOutput",
    "build_elec_priors",
    "fragment_polarizability",
]
