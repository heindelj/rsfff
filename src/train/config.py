"""YAML config -> typed dataclasses for the MLIP training pipeline.

Top-level YAML blocks: ``features:`` (SOAP featurizer), ``mlip:`` (MLP head),
``data:`` (dataset + split), ``train:`` (optimizer / loss weights). Parsing mirrors the
``load_config`` pattern in the reference repo: ``yaml.safe_load`` then nested
``.get(key, default)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Sequence

import yaml


@dataclass
class FeaturesConfig:
    cutoff: float = 5.0
    n_max: int = 6
    l_max: int = 4
    selected_lambdas: Sequence[int] = (0, 2)   # featurizer requires 2; head uses only inv
    backend: str = "e3nn"
    density_channels: int | None = 8


@dataclass
class MLIPConfig:
    emb_dim: int = 16
    hidden: int = 64
    depth: int = 2


@dataclass
class DataConfig:
    path: str | list[str] = "data/wb97mv_tzvpd/h2o_wb97mv_tzvpd.xyz"
    reference_energies: str = "data/atomic_references.json"
    isolated_species: str | None = None   # extxyz of integer-charge anchor systems
    diabatic_states: str | None = None    # YAML fragment-state library (channel graphs)
    atomic_reference_states: str | None = None  # isolated-atom E/alpha grid at integer charge
    # Isolated-monomer frames used as a multipole anchor by train_elec. Kept separate
    # from `path` rather than concatenated: they carry no eda_* components and they do
    # carry forces, so concatenate_datasets refuses the mix (rightly).
    monomer_path: str | None = None
    holdout_fraction: float = 0.1
    seed: int = 0


@dataclass
class EEMConfig:
    """EEM parameter-function heads (chi, eta, chivec, alpha) on the features."""

    emb_dim: int = 16               # species embedding width for the parameter heads
    hidden: int = 64                # parameter-head MLP width
    depth: int = 2
    equiv_channels: int = 32        # channel reduction of the chivec / alpha heads
    eta_init: float = 0.5           # initial per-element hardness, Ha/e^2
    eta_floor: float = 0.05         # hard lower bound on eta (keeps charges bounded)
    psd_floor: float = 1.0e-4       # minimum eigenvalue of the atomic alphas


@dataclass
class MonomerConfig:
    """Phase-1 monomer stack: reference embedding, state-decorated density, 1-body heads."""

    emb_dim: int = 32               # reference-embedding width (also the head conditioning slot)
    weight_channels: int = 8        # Kw learned density channels; feature width ~ (Kw*n_max)^2
    hidden: int = 64                # head MLP width
    depth: int = 2
    equiv_channels: int = 32        # channel reduction of the chivec / alpha heads


@dataclass
class SQEConfig:
    """Split-charge equilibration: channel compliance head and on-site hardness."""

    s_init: float = 0.5             # initial channel compliance (e^2/Ha)
    s_floor: float = 0.0            # lower bound on compliance; 0 keeps closure exact
    n_radial: int = 8               # Bessel functions in the pair compliance head
    eta_init: float = 0.5           # fallback per-element hardness when no IP/EA is available
    eta_floor: float = 0.05         # keeps the charge problem strongly convex
    psd_floor: float = 1.0e-4       # minimum eigenvalue of the atomic alphas


@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 32
    learning_rate: float = 1.0e-3
    weight_decay: float = 0.0
    grad_clip: float | None = 5.0
    energy_weight: float = 1.0
    force_weight: float = 1.0
    dipole_weight: float = 0.0
    dmu_dr_weight: float = 0.0
    dmu_dr_every: int = 1           # apply the (expensive) dmu/dR term every k steps
    alpha_weight: float = 0.0
    iso_weight: float = 0.0         # isolated-species integer-charge anchors
    atomic_ref_weight: float = 0.0  # isolated-atom energy anchors at integer charge
    free_alpha_weight: float = 0.0  # isolated-atom polarizability anchors
    unbound_weight: float = 0.0     # relative weight of anion states that are unbound at this
                                    # level of theory (H-, O2-); 0 drops them from the anchors
    q_l2_weight: float = 0.0
    eval_every: int = 10
    #: Checkpoint to warm start from, e.g. ``checkpoints/water_unified/best.pt``. Loaded
    #: non-strictly, with every skipped and missing tensor reported -- see
    #: :func:`rsfff.train.term_loop.warm_start`. Needed by the staged polarization/CT fits,
    #: where each level's label is a *difference* against the level below and starting from
    #: scratch would make the lower level's error indistinguishable from the new level's.
    init_from: str = ""
    #: Learning-rate schedule over the stage's epochs: ``"none"`` or ``"cosine"``.
    #:
    #: Cosine anneals ``learning_rate`` down to ``learning_rate * lr_final_factor`` by the last
    #: epoch, stepped once per epoch. It is here for a specific measured failure, not as a
    #: default nicety: the split of ``fragment_energy`` between ``E_internal`` and ``E_atom``
    #: is **unlabeled** -- only their sum is fitted -- so it is a flat direction of the loss,
    #: and at a fixed learning rate the optimizer does not converge along it, it *diffuses*.
    #: Measured over the frozen stage's 40 epochs, the two halves each wandered with a standard
    #: deviation of ~6 kJ/mol at a correlation of **-0.98** while their labeled sum held to
    #: 1.15, and the per-fragment residual of that walk put a 32x spread on the validation
    #: ``ob_mae`` (0.67 to 21.8 kJ/mol). Best-checkpoint selection then hands the next stage
    #: whichever sample of that walk happened to score best, and the freeze pins it there.
    #:
    #: Annealing does not remove the degeneracy -- nothing here does; it shrinks the step size
    #: that sets the diffusion amplitude, so the stage *ends* at a converged point rather than
    #: at a draw from a distribution. Watch ``internal`` and ``e_atom``: their spread over the
    #: last few epochs is the number this is meant to move.
    lr_schedule: str = "none"
    #: Final learning rate as a fraction of the initial one, for ``lr_schedule: cosine``.
    lr_final_factor: float = 0.05


@dataclass
class DispersionConfig:
    """Tang-Toennies damped C6 dispersion plus its short-range pair correction.

    ``r0`` is the Fermi range-separation midpoint in Angstrom: the explicit dispersion is
    switched *off* below it and the pair correction carries that region. It is learnable,
    with ``r0_weight`` applying a linear penalty that biases it small -- so moving energy
    out of the physical backbone and into the network has to be earned
    (``docs/range_separated_mlip.md`` §4.5, §7).
    """

    cutoff: float = 10.0            # pair-list cutoff, Angstrom
    taper_width: float = 1.0        # C2 taper width below the cutoff, Angstrom
    r0_init: float = 2.0            # Fermi midpoint, Angstrom
    alpha: float = 8.0              # Fermi steepness, Angstrom^-1
    learn_r0: bool = True
    order: int = 6                  # damped inverse power (6 for C6)
    # Damping-exponent prior: a number in bohr^-1, or "per_element" for the fitted values.
    # A uniform 2.0 over-binds badly on water (see build_log_priors); leave learn_b on if
    # you use it, so the two per-species scalars can move off it.
    b_prior: float | str = 2.0
    learn_c6: bool = True           # per-species log-C6 deviation
    environment_c6: bool = True     # environment-dependent log-C6 residual
    learn_b: bool = True            # per-species log-b deviation
    environment_b: bool = False     # never on by default; see DispersionParameterHeads
    emb_dim: int = 16
    hidden: int = 64
    depth: int = 2
    # Pair correction head
    correction: bool = True
    corr_hidden: int = 64
    corr_depth: int = 2
    corr_n_radial: int = 8
    corr_r_on: float = 4.0          # Angstrom; envelope full strength below this
    corr_r_off: float = 5.0         # Angstrom; exactly zero at and beyond this
    corr_energy_scale: float = 1.0e-3   # Hartree
    # Loss. Squared errors are divided by `energy_scale`^2, so the fit term is O(1) for an
    # error of that size and every penalty weight below is a plain dimensionless number.
    # Without that normalization the weights would have to be quoted in Hartree^2 (~1e-7
    # for a 1 kJ/mol error), which is a very easy place to be off by orders of magnitude.
    target: str = "disp"            # which batch.eda[...] component to fit
    energy_scale: float = 3.8093e-4  # 1 kJ/mol in Hartree
    corr_l2_weight: float = 0.0     # penalty on the correction's magnitude
    r0_weight: float = 0.05         # per Angstrom of range separation handed to the network
    intra_fragment_features: bool = False   # ablation: group features by fragment


@dataclass
class PauliConfig:
    """Slater-damped multipolar Pauli repulsion plus its short-range pair correction.

    Deliberately has no range-separation parameter, unlike ``DispersionConfig``: the Slater
    form is short-ranged by construction (the undamped tail is subtracted off) and valid
    all the way in, so there is nothing to hand over to the network at a midpoint, and the
    correction head's own envelope is already at full strength at short range.

    Intramolecular pairs are excluded by ``inter_only`` -- a hard mask on the pair list, not
    a distance cutoff. On this data intra H-H reaches 1.688 A while inter O-H reaches down
    to 1.552 A, so no radius separates them; reactivity is a job for a learned per-pair
    weight (``SlaterPauli.pair_weight``), not a threshold.
    """

    cutoff: float = 7.0             # pair-list cutoff, Angstrom
    taper_width: float = 1.0        # C2 taper width below the cutoff, Angstrom
    max_rank: int = 1               # 0 = charges, 1 = +dipoles, 2 = +quadrupoles
    inter_only: bool = True
    learn_q: bool = True            # per-species log-q deviation
    environment_q: bool = True      # environment-dependent log-q residual
    learn_b: bool = True            # per-species log-b deviation
    environment_b: bool = False     # never on by default; b competes with q and with dE
    learn_dipole: bool = True       # needs features.selected_lambdas to include 1
    learn_quadrupole: bool = True   # max_rank 2 only; needs lambda=2 features
    emb_dim: int = 16
    hidden: int = 64
    depth: int = 2
    equiv_channels: int = 32        # lambda=1 channel reduction in the dipole head
    # Pair correction head
    correction: bool = True
    corr_hidden: int = 64
    corr_depth: int = 2
    corr_n_radial: int = 8
    corr_r_on: float = 4.0          # Angstrom; envelope full strength below this
    corr_r_off: float = 5.0         # Angstrom; exactly zero at and beyond this
    corr_energy_scale: float = 3.0e-3   # Hartree; mod_pauli runs ~5x larger than disp
    # Loss. Squared errors are divided by `energy_scale`^2 so the fit term is O(1) for an
    # error of that size and every penalty weight is a plain dimensionless number.
    target: str = "mod_pauli"       # which batch.eda[...] component to fit
    energy_scale: float = 3.8093e-4  # 1 kJ/mol in Hartree
    corr_l2_weight: float = 0.0     # penalty on the correction's magnitude
    intra_fragment_features: bool = False   # ablation: group features by fragment


@dataclass
class ElectrostaticsConfig:
    """Classical electrostatics: a local SQE solve, Slater penetration, a pair correction.

    The defining constraint is ``intra_fragment_features: true``: the response parameters see
    only their own monomer, so the interaction is **exactly two-body**, which is what
    classical electrostatics between frozen densities has to be. Turning it off additionally
    requires ``allow_environment``, because losing exactness would be invisible in the fit.

    Penetration is explicit and carries an effective nuclear charge ``Z``. Measured on this
    data, point multipoles alone reproduce only ~53% of ``eda_cls_elec`` (13.7 kJ/mol MAE on
    dimers with pyCMM's fitted multipoles) against 3.25 with penetration -- it is half the
    component, not a short-range correction.

    ``max_rank: 1`` (charges + dipoles) is what the response solve naturally produces.
    Measured ceilings with pyCMM's parameters: rank 1 gives MAE 9.07 (w2) / 39.03 (w5) with a
    regression slope of 1.42, rank 2 gives 3.25 / 14.19 at slope 0.89 -- a monopole+dipole
    expansion cannot reproduce water's quadrupolar field, so expect the correction head to
    work harder at rank 1.
    """

    cutoff: float = 12.0            # pair-list cutoff, Angstrom; this term has a 1/r tail
    taper_width: float = 1.0
    max_rank: int = 1               # 0 = charges, 1 = +dipoles, 2 = +quadrupoles
    r0_init: float = 1.5            # Fermi midpoint, Angstrom; gates point AND penetration
    alpha: float = 8.0              # Fermi steepness, Angstrom^-1
    learn_r0: bool = True
    # Response heads
    environment_chi: bool = True
    environment_eta: bool = True
    eta_floor: float = 0.05         # keeps the charge problem strictly convex
    learn_dipole: bool = True       # mu_i = -alpha_i chivec_i; needs lambda=1 features
    psd_floor: float = 1.0e-4
    # Quadrupole sector, max_rank 2 only: Theta_i = -C_i chiquad_i with C_i = c_i I5.
    # Isotropic because at zero field gradient an anisotropic C is exactly as expressive
    # (Theta = -C chiquad is onto for any invertible C) while adding five unconstrained
    # parameters per atom. Note the (C, chiquad) gauge: scaling one and inverse-scaling
    # the other leaves Theta untouched, so these are identified only once
    # `internal_energy` enters a loss -- until then leave weight_decay on.
    #: The **permanent** atomic quadrupole. Under ``direct_multipoles`` this head's output
    #: *is* ``Theta_0``; it is what ``eda_cls_elec`` and the fragment quadrupole labels see.
    learn_quadrupole: bool = True
    #: The quadrupole **polarizability** -- whether a field gradient can move that moment.
    #:
    #: Off by default, which leaves each atom a rigid permanent quadrupole. The induction
    #: model moves charges and dipoles only: charge flux through the SQE channels plus the
    #: on-site ``alpha``. Quadrupole response is the smallest of the three, it is the sector
    #: whose ``(C, chiquad)`` gauge is least constrained by any label here, and dropping it
    #: removes a ``5N`` block from the coupled-solve state.
    #:
    #: Independent of ``learn_quadrupole``: turning this off does **not** make atoms
    #: quadrupole-free, and ``max_rank`` stays 2 either way -- that decides which slots the
    #: interaction tensor and the environment features carry, not which heads exist.
    quadrupole_response: bool = False
    cquad_init: float = 1.0         # e^2 a0^4 / Ha
    cquad_floor: float = 1.0e-4
    environment_cquad: bool = False
    #: Axially anisotropic quadrupole polarizability: three positive eigenvalues
    #: (m = 0, |m| = 1, |m| = 2) about a learned axis, instead of one isotropic scalar.
    #: The general symmetric map on the l=2 space has fifteen components
    #: (Sym^2(l=2) = 0 + 2 + 4); the axial form is the physically motivated middle ground
    #: and costs three scalars plus a direction. Needs lambda=1 features for the axis.
    #: Initialized with the three equal, so it reproduces the isotropic head exactly at
    #: initialization and the axis stays inert until the fit separates them.
    #: A sub-option of ``quadrupole_response``: inert while that is off, since there is then
    #: no quadrupole polarizability to be anisotropic.
    anisotropic_cquad: bool = False
    #: Interpret the equivariant heads as the **permanent multipoles** rather than as the
    #: drives that produce them -- ``mu = mu0`` instead of ``mu = -alpha chivec``, with
    #: ``alpha``/``cquad`` describing only the response to a field. The same functional
    #: (``mu0 = -alpha chivec``), so the solve stays quadratic and CG is untouched; what it
    #: buys is that the permanent multipole stops being a product of two heads, and that
    #: ``alpha -> 0`` no longer forces a non-polar atom.
    #:
    #: It also empties the on-site sectors out of ``internal_energy``: at the frozen level the
    #: state *is* the minimum, so there is no ``-1/2 chi^T a chi`` left to report and the
    #: ~-400 kJ/mol per fragment it carried moves to ``unified.atomic_energy``. Turn that on
    #: with this, or the energy has nowhere to go. See
    #: :attr:`rsfff.ff.response.FragmentResponse.direct_multipoles`.
    direct_multipoles: bool = False
    # Penetration heads
    learn_z: bool = True
    environment_z: bool = False
    learn_b: bool = True
    environment_b: bool = False
    # SQE channels
    s_init: float = 0.5             # initial channel compliance
    compliance_hidden: int = 64
    compliance_depth: int = 2
    emb_dim: int = 16
    hidden: int = 64
    depth: int = 2
    equiv_channels: int = 32
    # Pair correction head
    correction: bool = True
    corr_hidden: int = 64
    corr_depth: int = 2
    corr_n_radial: int = 8
    corr_r_on: float = 4.0
    corr_r_off: float = 5.0
    corr_energy_scale: float = 3.0e-3   # Hartree; cls_elec runs comparable to mod_pauli
    # Loss
    target: str = "cls_elec"
    energy_scale: float = 3.8093e-4     # 1 kJ/mol in Hartree
    corr_l2_weight: float = 0.0
    r0_weight: float = 0.05             # per Angstrom handed to the network
    intra_fragment_features: bool = True    # the two-body-exactness constraint
    allow_environment: bool = False         # required to turn the above off
    # Permanent-multipole supervision against the frozen-monomer reference values.
    # The interaction energy alone under-determines the individual multipoles -- many
    # (q, mu, Theta) sets give the same pair energies -- so these are what actually pin
    # the monomer. Errors are divided by their scale before squaring, so the weights are
    # dimensionless and each term is 1.0 at an error of one scale.
    # `dipole_weight`/`quadrupole_weight` act on the isolated-monomer anchor batch
    # (`data.monomer_path`); `fragment_multipole_weight` scales the *same* two terms
    # applied to the in-batch cluster fragments instead. The anchor is the default source
    # because those frames are standalone monomers at 300 K, whose r(O-H) and HOH ranges
    # are measurably *wider* than the in-cluster monomer distribution -- so the cluster
    # fragments add labels but no new geometry, and they cost a full multipole rebuild on
    # every training batch.
    dipole_weight: float = 0.0
    quadrupole_weight: float = 0.0
    dipole_scale: float = 0.05          # e*a0; monomer dipoles run ~0.75
    quadrupole_scale: float = 0.2       # e*a0^2; Buckingham components of order 1
    fragment_multipole_weight: float = 0.0
    # Molecular polarizability of each anchor monomer, from a Q-Chem
    # `JOBTYPE = polarizability` job (scripts/parse_polarizability.py). The only label in the
    # fit that says how the charges and dipoles *move* rather than where they sit -- see
    # `rsfff.train.loss.fragment_polarizability_loss`. Requires one fragment per frame, which
    # the monomer anchor satisfies and nothing else does.
    polarizability_weight: float = 0.0
    polarizability_scale: float = 0.5   # e^2*Ang^2/Ha ~ 0.18 a0^3; monomer alpha runs ~9.9


@dataclass
class OneBodyConfig:
    """The 1-body term: frozen atomic references plus an intramolecular bond energy.

    ``E_1body(f) = sum_{i in f} E0[Z_i] + sum_{i<j in f} W(r_ij) dE_ij``, fit per fragment
    against the ``Fragment Energies (Ha)`` an EDA job prints, and against forces on the
    isolated-monomer frames where the QC gradient *is* ``-dE_1body/dR``.

    The pair-head defaults deliberately differ from every other term's. Those are tuned for
    intermolecular corrections (1e-3 Ha over 4-5 Angstrom); a covalent bond is ~0.2 Ha over
    1-1.7 Angstrom, so the scale is 200x larger and the envelope has to open where the bonds
    actually are. See :mod:`rsfff.ff.onebody`.
    """

    # Bond-energy pair head
    emb_dim: int = 16
    bond_hidden: int = 64
    bond_depth: int = 2
    bond_n_radial: int = 8
    bond_r_on: float = 2.5          # Angstrom; well past the 1.5 A intramolecular H-H
    bond_r_off: float = 4.0         # must not exceed features.cutoff
    bond_energy_scale: float = 0.2  # Hartree; water's two O-H bonds are ~-0.37 Ha
    # Loss
    energy_scale: float = 3.8093e-4     # 1 kJ/mol in Hartree; fit term is 1.0 at that error
    force_weight: float = 0.0
    force_scale: float = 1.0e-3         # Hartree/Angstrom
    intra_fragment_features: bool = True
    # `target` is unused -- the label is batch.fragment_energy, not an EDA component -- but
    # term_loop's shared plumbing reads cfg.target when no fit_term is supplied.
    target: str = "fragment_energy"


@dataclass
class JointConfig:
    """Relative weights for the joint 1-body + electrostatics fit.

    Everything else comes from the ``onebody:`` and ``elec:`` blocks, which the joint model
    reuses verbatim -- there is one response solve and one set of heads, so duplicating
    forty fields into a third block would only create ways for them to disagree.

    Both energy terms are already normalized to "1.0 at one ``energy_scale`` of error", so
    these weights are dimensionless. They are not equal by default because the two targets
    are not: at initialization the fragment energy is off by ~960 kJ/mol (the bond energy,
    which starts at zero) while ``cls_elec`` is off by ~14, so an equal weighting lets the
    1-body term dominate the first few epochs by four orders of magnitude in the loss.
    """

    onebody_weight: float = 1.0
    elec_weight: float = 30.0
    #: The monomer anchor: energy, forces, and multipoles on the isolated-monomer frames.
    anchor_weight: float = 1.0


@dataclass
class UnifiedConfig:
    """The unified pair model: one pair list, learned range separation, one shared trunk.

    Reuses the ``elec:`` block for the response solve and the ``dispersion:`` / ``pauli:``
    blocks for their parameter heads, exactly as ``joint:`` reuses ``onebody:`` and ``elec:``.
    Only what is genuinely new to the unified arrangement lives here: the range-separation
    priors, the per-channel correction envelopes, and the routing weights.

    The range separation replaces both the per-term global scalar ``r0`` and the hard
    inter-fragment mask. ``r0`` is per atom and per channel, combined across a pair as the
    geometric mean, and it now has to cover a case the per-term modules never saw: a
    **covalent** pair, which :func:`rsfff.ff.pairs.union_pairs` no longer drops. See
    :mod:`rsfff.ff.range_priors` for the measured distance gaps the defaults come from --
    ``alpha`` is 40 rather than the per-term modules' 8 because it now has to cross the
    1.07 -> 1.54 Angstrom O-H gap rather than taper a mid-range handoff.
    """

    max_rank: int = 1               # must match elec.max_rank
    # Range separation. Priors are per element (see rsfff.ff.range_priors); these knobs say
    # how much freedom the fit gets on top of them.
    alpha_init: float = 40.0        # Angstrom^-1, per channel, learnable
    learn_r0: bool = True           # per-species log-r0 deviation, per channel
    environment_r0: bool = False    # off by default: competes with the pair correction
    learn_alpha: bool = True
    #: Learn a per-**pair** deviation of ``r0`` on top of the per-element base, from the
    #: correction trunk. A per-atom ``r0`` gives one threshold per element pair, which cannot
    #: separate topologically distinct pairs of the same elements: an ethane geminal H-H sits
    #: at 1.78 Angstrom and an H-H across the C-C bond at 2.27, and only the second should be
    #: carried by the classical form. Water cannot show this -- its geminal H-H is at 1.51,
    #: where distance alone happens to decide -- so expect no metric movement on this data.
    #: Zero-initialized, so it starts at exactly the per-element result. For intra-fragment
    #: pairs the deviation is read from the fragment-confined trunk, which is what keeps the
    #: within-fragment range separation independent of the surroundings.
    pair_range_separation: bool = True
    r0_emb_dim: int = 16
    r0_hidden: int = 64
    r0_depth: int = 2
    # Fragment-state block: (Q_f, 2S_f) per atom. Identically zero for a neutral singlet, so
    # inert on water-only data; 0 disables it and is bit-identical to a model without it.
    fragment_state_dim: int = 4
    fragment_state_hidden: int = 32
    fragment_state_depth: int = 1
    # Environment-aware descriptor: h_env = h_frag + g(h_full), with g zero-initialized so
    # this starts from exactly the fragment-confined model. Only *inter*-fragment pairs read
    # h_env; intra pairs and the response solve stay on h_frag, because `fragment_energy` is
    # an isolated-fragment label with no environment dependence to fit. Turning it on is what
    # lets C6 be quenched by the surroundings -- effective many-body dispersion -- at the cost
    # of the interaction channels no longer being rigorously two-body, and roughly a second
    # power spectrum per forward.
    environment_features: bool = False
    env_hidden: int = 64
    env_depth: int = 2
    #: Penalty on ``||h_env - h_frag||``, biasing the model back toward the fragment-confined
    #: description so environment dependence has to be earned. Same spirit as `r0_weight`.
    env_weight: float = 0.0
    # Shared correction trunk
    emb_dim: int = 16
    corr_hidden: int = 64
    corr_depth: int = 2
    corr_n_radial: int = 8
    corr_r_on: float = 4.0          # Angstrom; envelope full strength below this
    corr_r_off: float = 5.0         # must not exceed features.cutoff
    # Per-channel output scale, Hartree. Unequal because the targets are: a covalent bond is
    # ~0.2 Ha, two hundred times an intermolecular correction. One shared scale would let the
    # bond channel's gradient set the effective learning rate for all four.
    elst_energy_scale: float = 3.0e-3
    pauli_energy_scale: float = 3.0e-3
    disp_energy_scale: float = 1.0e-3
    bond_energy_scale: float = 0.2
    bond_r_on: float = 2.5          # the bond channel opens where the bonds are
    bond_r_off: float = 4.0
    #: Whether the pair head's **energy** readouts are consumed. Off leaves the module built
    #: and its state_dict loadable -- so this is a config change, not a refit from scratch,
    #: in either direction -- while every ``interaction_corr`` reads exactly zero and the
    #: intramolecular deformation energy comes from ``atomic_energy`` instead.
    #:
    #: Off is the "atomic energy plus range-separated force field" model. It exists because
    #: the corrections had grown into the thing they were correcting: measured on the last
    #: checkpoint that had them, ``ct`` was **96.7%** neural (-5.24 kJ/mol per fragment of
    #: correction against -0.18 of classical), and the bond channel was carrying a constant
    #: -1.686 kJ/mol per fragment of unlabeled one-body gauge along with it.
    #:
    #: Note that with this off *and* ``pair_range_separation`` off the head is never called,
    #: so it receives no gradient and weight decay is the only force acting on it. See
    #: :func:`rsfff.train.term_loop.parameter_groups`, which keeps it out of the optimizer
    #: for exactly that reason.
    #:
    #: **Per channel.** ``true`` enables every channel the head was built with, ``false``
    #: none, and a list names exactly the ones to consume::
    #:
    #:     pair_corrections: [elst]      # electrostatics only; bond/pauli/disp stay silent
    #:
    #: All-or-nothing was the wrong granularity, because the channels are not the same kind of
    #: object. ``bond`` sits inside ``fragment_energy`` where it is degenerate with
    #: ``atomic_energy``, and the since-removed ``ct_bond`` was measured supplying 96.7% of its
    #: label -- those are the ones that had to go. ``elst`` is not in that position: it corrects
    #: a channel with a real label of its own (``eda_cls_elec``), and because the electrostatic
    #: split routes its ``h_env - h_frag`` difference to ``induction``, it is also the only
    #: correction that reaches induction. Turning it on alone is the useful middle setting.
    #:
    #: Unknown names raise at model-build time rather than being ignored, so a typo is not a
    #: silently disabled channel.
    pair_corrections: bool | list[str] = True
    #: The per-atom energy of the self-consistent electronic state,
    #: :class:`rsfff.ff.atomic_energy.AtomicStateEnergy`. This is what carries the covalent
    #: bond energy inside ``fragment_energy`` when ``pair_corrections`` is off, and what
    #: carries the ``pol``/``ct`` response of the bonding to the relaxed multipoles.
    atomic_energy: bool = False
    atomic_energy_hidden: int = 64
    atomic_energy_depth: int = 2
    atomic_energy_emb_dim: int = 16
    #: How many channels the lambda=1/2 features are reduced to before being contracted
    #: against the multipoles and the electrostatic environment.
    atomic_energy_equiv_channels: int = 8
    #: Output scale, Hartree. Sized against a covalent bond like ``bond_energy_scale``, not
    #: against the interaction corrections -- this term has to supply ~-566 kJ/mol per water.
    atomic_energy_scale: float = 0.2
    # Classical reach per channel, Angstrom. The largest sets the one shared pair list.
    elst_cutoff: float = 12.0       # the one term with a genuine 1/r tail
    pauli_cutoff: float = 7.0
    disp_cutoff: float = 10.0
    taper_width: float = 1.0
    # Loss. Errors are divided by `energy_scale` before squaring, so every weight below is a
    # plain dimensionless number and each term is 1.0 at an error of one scale.
    energy_scale: float = 3.8093e-4     # 1 kJ/mol in Hartree
    onebody_weight: float = 1.0
    elst_weight: float = 30.0       # see JointConfig: the 1-body term starts ~70x further off
    pauli_weight: float = 30.0
    disp_weight: float = 30.0
    anchor_weight: float = 1.0
    #: Supervision on the **cluster total energy** and its gradient. Both default off and both
    #: only mean anything once ``pol`` and ``ct`` are on: below that level the decomposition is
    #: incomplete by construction, so ``energy`` is missing two channels and fitting it would
    #: push that deficit into whichever channel is cheapest to distort.
    #:
    #: Turn them on for the CT stage **without dropping the EDA component weights**. The total
    #: is one number per frame against six well-posed component targets; supervised alone it
    #: admits every wrong split that happens to sum correctly, and the components collapse into
    #: each other while ``e_tot_mae`` looks fine. The components are what keep the
    #: decomposition meaningful, the total is what makes it a force field.
    total_energy_weight: float = 0.0
    force_weight: float = 0.0
    force_scale: float = 1.0e-3         # Hartree/Angstrom
    #: Apply the **cluster** force term every k-th training step; 1 is every step.
    #:
    #: Same trade as ``anchor_force_every``, and it buys more: the cluster force is a
    #: ``create_graph=True`` backward through the whole model including the coupled solve, and
    #: it is by far the most expensive single thing this fit can be asked to do. Measured at
    #: batch 128 with induction on, 31.8 s/epoch without it against **86.1 with it** -- 2.7x
    #: for one label. At ``force_every: 2`` that becomes ~1.9x.
    #:
    #: Striding is honest here for the same reason it is on the anchor: the term is a smooth
    #: function of the parameters, so a strided estimate is a k-times-noisier version of the
    #: same pull rather than a different one. Evaluation always computes it, so ``f_clu`` means
    #: the same thing on every validation line.
    force_every: int = 1
    #: **Known bias in the force gradient at the pol/CT levels.** ``rsfff.ff.coupled_solve``
    #: solves the response with a custom ``autograd.Function`` whose backward is the adjoint of
    #: docs §6.2. That adjoint is correct -- checked against the dense oracle, and an *energy*
    #: loss reproduces finite differences to CG tolerance (5e-8 relative). It is not, however,
    #: **double**-differentiable: the adjoint CG runs under ``no_grad`` and detaches its
    #: parameters, so the second-order path through ``lambda`` is dropped. A force loss needs
    #: exactly that path, and its parameter gradient measures 1e-5 to 6e-4 wrong in relative
    #: terms -- independent of ``cg_rtol``, which is what proves it is a missing term and not
    #: convergence. At the frozen level, where no coupled solve runs, the same check agrees to
    #: 4e-12.
    #:
    #: The *forces themselves* are exact; only the derivative of the force with respect to the
    #: parameters is biased. A relative bias of 1e-4 is well under minibatch noise, so this is
    #: a real but small systematic pull, not a broken fit. Removing it means making the adjoint
    #: solve differentiable in its own right (a nested implicit solve for ``d lambda / d
    #: theta``), which is not implemented. ``tests/test_ff_unified.py`` pins the measurement so
    #: the number moves if that changes.
    #: Monomer frames drawn per step for the anchor term; 0 uses the whole file every step.
    #: The anchor carries a force term, so evaluating it is a second-order backward, and at
    #: the full 500 frames that is ~95% of the wall time of a training step -- the identical
    #: computation repeated for every minibatch of the actual training set. Sampling a fresh
    #: subset each step is plain SGD on that term: the same expected gradient, ~8x cheaper,
    #: and over an epoch it still sees every monomer. Evaluation uses a fixed leading slice
    #: so the validation number stays comparable across epochs.
    anchor_batch_size: int = 64
    #: Apply the anchor **force** term every k-th training step; 1 is every step.
    #:
    #: The force is ``-dE/dR`` by autograd with ``create_graph=True``, so it is a second-order
    #: backward and by far the most expensive single item in a step: measured at **36% of the
    #: entire step** on the frozen stage, for one label. Evaluating it every k steps is the
    #: same trade ``TrainConfig.dmu_dr_every`` makes for the dipole derivative, and for the
    #: same reason -- the term is a smooth function of the parameters, so a strided estimate
    #: of it is a k-times-noisier version of the same pull rather than a different one.
    #:
    #: Skipping is a **training-step** decision only. Evaluation epochs always compute it: the
    #: force there is a single backward (no ``create_graph``), it is cheap, and ``f_mae`` is a
    #: validation metric that has to mean the same thing every time it is printed. On skipped
    #: training steps the key is simply absent, and ``run_epoch`` averages each metric over the
    #: steps that reported it, so the logged ``f_mae`` stays an unbiased mean rather than being
    #: diluted by zeros.
    anchor_force_every: int = 1
    #: Penalty on classical energy between pairs the **bond channel** is already describing.
    #:
    #: Dispersion between covalently bonded atoms is not dispersion -- that correlation
    #: energy is already in the bond -- but the Tang-Toennies factor alone does not remove
    #: it: at 0.96 Angstrom ``f6(b r)`` is ~0.025 against a large ``C6``, leaving -15.7 kJ/mol
    #: per O-H pair. Left unpenalized the fit is indifferent, because the bond channel absorbs
    #: whatever appears; a measured run parked -28 kJ/mol per fragment there and oscillated.
    #:
    #: Weighted by the bond channel's own envelope, so it asks only that the classical form
    #: stay out of the region the bond term is covering. An intra pair beyond ``bond_r_off``
    #: is untouched -- which is what preserves same-fragment electrostatics at range, the
    #: capability the union pair list exists for. Quadratic, so it self-extinguishes as the
    #: leak closes rather than continuing to push. Applied per channel rather than to their
    #: sum, so channels cannot cancel each other and call it clean.
    intra_classical_weight: float = 0.05
    corr_l2_weight: float = 0.0
    #: Linear pull on the mean per-atom ``r0``, per Angstrom handed to the network. Same
    #: meaning as the per-term ``r0_weight``, now averaged over atoms rather than a scalar.
    r0_weight: float = 0.05
    #: Quadratic pull on the environment residual, keeping per-atom ``r0`` near its element
    #: prior. Inert while ``environment_r0`` is off.
    r0_spread_weight: float = 0.0

    # --- induction (docs/range_separated_mlip.md §5.1) ------------------------------------
    #: **Induction: polarization and charge transfer as one term**, fitted against
    #: ``eda_pol + eda_ct``. Off leaves the model bit-identical to the frozen fit.
    #:
    #: It moves the inter-fragment electrostatics inside the response functional so the
    #: multipoles relax against each other, and lets *every* response parameter -- ``chi``,
    #: ``eta``, ``alpha`` and the charge-flux compliance alike -- read the environment-aware
    #: ``h_env`` instead of the fragment-confined ``h_frag``. The fragment still supplies the
    #: **graph** of allowed charge flux (its own intra-fragment channels); the environment
    #: supplies the **parameters** on that graph. No charge crosses a fragment boundary.
    #:
    #: **Why these were two levels and are now one.** ``pol`` and ``ct`` were separated only by
    #: their labels and by the atomic energy's feature stream (``h_frag`` at pol, ``h_env`` at
    #: ct). That swap is a free ~20 kJ/mol knob that only ``ct`` could reach, and it is much
    #: cheaper to fit than moving charge: measured on a trained checkpoint, ``ct`` was
    #: **99.99% correction and 100.0% descriptor swap**, with 4.6e-5 e crossing a boundary
    #: against a real ~0.01-0.03 e per hydrogen bond. The compliance head had closed every
    #: inter-fragment channel (``q_ct`` 0.085 -> 5.7e-5, monotone) while ``ct_mae`` read 0.40
    #: kJ/mol and reported nothing wrong. Same failure as the ``ct_bond`` readout before it
    #: (96.7% neural), through a different route.
    #:
    #: Merged, the swap lands in a channel that also contains a real physical solve, and the
    #: split is visible: watch ``ind_ff`` against ``ind_corr``, and ``ind_swap`` for how much
    #: of the correction is the feature stream alone. Explicit inter-fragment charge transfer
    #: comes back when reactivity does, and it will need a compliance scale again then.
    induction: bool = False
    induction_weight: float = 30.0
    #: Freeze the whole fragment-confined path -- ``featurizer.channel_proj``,
    #: ``response.params`` and ``response.compliance_head`` -- at the values this stage warm
    #: starts from. On for every stage after the first.
    #:
    #: The frozen level is a *fitted, labeled* object by the end of stage 1: one-body bias
    #: 0.014 kJ/mol per fragment, monomer polarizability 9.904 a0^3 against a true 9.891. It
    #: then has no label of its own at the higher stages, so nothing defends it -- and both
    #: measurably collapsed, because the response heads are one shared function and the ``pol``
    #: gradient arriving through ``h_env`` also reshapes their ``h_frag`` behaviour. Over
    #: stages 2 and 3 the isolated-fragment internal energy swung by 62 then 109 kJ/mol
    #: (leaving a per-fragment one-body offset of +2.511 then -0.627 that the bond head was
    #: left chasing), and the monomer polarizability fell to 8.903 with its in-plane
    #: eigenvalues down by 1.8 a0^3 each.
    #:
    #: None of that was necessary. ``h_env == h_frag`` *exactly* on an isolated fragment --
    #: ``EnvironmentResidual`` is anchored as ``g(h_full) - g(h_frag)`` -- so ``pol`` and ``ct``
    #: can be expressed entirely through ``g``, the coupled solve and the correction trunk,
    #: leaving what an isolated fragment *is* alone. This makes them.
    #:
    #: Freezing the response heads alone is not enough and that is worth knowing before
    #: shortening the list: reverting only ``response.params`` to its stage-1 values left the
    #: internal energy at -230 kJ/mol against stage 1's -342, because ``featurizer.channel_proj``
    #: had moved 22% and the same head weights were reading different features.
    freeze_frozen_level: bool = False
    #: Quadratic penalty on the drift of the isolated-fragment internal energy away from the
    #: value this stage started at, measured on the monomer anchor and divided by
    #: ``energy_scale`` like every other term.
    #:
    #: **With ``freeze_frozen_level`` on this is a completeness assertion, not a force**, and
    #: that is how to read it: the internal energy is then a function of frozen parameters
    #: alone, so ``d_internal`` logs an exact zero (measured 1e-15 kJ/mol on a smoke run) and
    #: the penalty contributes nothing. A nonzero reading means some route into the frozen
    #: level is *not* in ``_frozen_level_modules`` -- which is a real possibility worth
    #: catching cheaply, since the anchor forward it rides on happens either way.
    #:
    #: It is also the fallback if the freeze is ever turned off: naming the quantity that must
    #: not move is more robust than enumerating the parameters that could move it, and this
    #: was measured at 62 and 109 kJ/mol over the two higher stages of the unfrozen fit.
    #:
    #: Leave at 0 for the first stage: there is no previous stage to anchor to, and anchoring
    #: to an untrained split would be worse than not anchoring at all.
    internal_drift_weight: float = 0.0
    #: Weight on the **free-atom** polarizability anchor, against
    #: ``data.atomic_reference_states``. Dimensionless, sharing ``elec.polarizability_scale``
    #: with the molecular term so the two are directly comparable.
    #:
    #: A lone atom has an all-zero density, so the on-site polarizability head reduces to one
    #: isotropic number per element and this anchor pins it *exactly* rather than nudging it.
    #: That is the mechanism the Phase-1 monomer model had and the unified fit dropped, and it
    #: matters more than it looks: ``alpha_flow`` is built from ``B^T R`` and therefore has no
    #: component perpendicular to a planar molecule's nuclear plane, so the on-site sector has
    #: to carry all of water's out-of-plane polarizability by itself.
    #:
    #: Neutral states only -- see :func:`rsfff.train.loss.free_atom_batch` for why.
    free_alpha_weight: float = 0.0
    #: Conjugate-gradient tolerances for the coupled solve. ``cg_maxiter`` is a safety net,
    #: not a working limit -- with the uncoupled response as preconditioner a water cluster
    #: converges in 5-8 iterations, so a count that climbs is the early warning that the
    #: polarization is running away, well before it becomes a NaN.
    cg_rtol: float = 1.0e-9
    cg_atol: float = 1.0e-12
    cg_maxiter: int = 100


@dataclass
class StageConfig:
    """One stage of a staged fit: a name plus per-block overrides of the parent config.

    ``overrides`` is ``{block_name: {field: value}}`` using the same block names as the YAML
    (``unified``, ``train``, ...). Applied with :func:`dataclasses.replace`, so an unknown
    field raises rather than being silently ignored -- a typo'd override in a run that takes
    hours is worth failing fast on.
    """

    name: str
    overrides: dict = field(default_factory=dict)


@dataclass
class Config:
    run_name: str = "run"
    device: str = "auto"       # auto -> cuda > mps > cpu
    dtype: str = "float32"     # float32 | float64
    checkpoint_root: str = "checkpoints"
    #: Sequential stages, each warm starting from the previous one's best checkpoint. Empty
    #: means a single ordinary fit. See :func:`rsfff.train.config.stage_config`.
    stages: list = field(default_factory=list)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    mlip: MLIPConfig = field(default_factory=MLIPConfig)
    eem: EEMConfig = field(default_factory=EEMConfig)
    monomer: MonomerConfig = field(default_factory=MonomerConfig)
    sqe: SQEConfig = field(default_factory=SQEConfig)
    dispersion: DispersionConfig = field(default_factory=DispersionConfig)
    pauli: PauliConfig = field(default_factory=PauliConfig)
    elec: ElectrostaticsConfig = field(default_factory=ElectrostaticsConfig)
    onebody: OneBodyConfig = field(default_factory=OneBodyConfig)
    joint: JointConfig = field(default_factory=JointConfig)
    unified: UnifiedConfig = field(default_factory=UnifiedConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def stage_config(config: "Config", stage: "StageConfig", init_from: str = "") -> "Config":
    """The parent config with one stage's overrides applied, ready to hand to ``fit``.

    ``run_name`` gains the stage's name so each stage checkpoints to its own directory and a
    later stage can never overwrite the one it warm started from. ``init_from`` is filled in
    with the previous stage's best checkpoint unless the stage sets it explicitly.

    Overrides go through :func:`dataclasses.replace`, so a misspelled field raises here rather
    than being dropped -- which matters when the mistake would otherwise surface as a stage
    that quietly trained the wrong thing for several hours.
    """
    from dataclasses import replace as _replace

    blocks = {}
    for block_name, values in stage.overrides.items():
        if not hasattr(config, block_name):
            raise ValueError(
                f"stage {stage.name!r} overrides unknown block {block_name!r}; "
                f"valid blocks: {sorted(f.name for f in fields(config))}"
            )
        current = getattr(config, block_name)
        if not is_dataclass(current):
            raise ValueError(
                f"stage {stage.name!r} overrides {block_name!r}, which is not a config block"
            )
        known = {f.name for f in fields(current)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(
                f"stage {stage.name!r} sets unknown {block_name} field(s) "
                f"{sorted(unknown)}; valid: {sorted(known)}"
            )
        blocks[block_name] = _replace(current, **values)

    staged = _replace(config, run_name=f"{config.run_name}_{stage.name}", **blocks)
    if not staged.train.init_from and init_from:
        staged = _replace(staged, train=_replace(staged.train, init_from=init_from))
    return staged


def _from_block(cls, block: dict):
    """Build a config dataclass from a YAML block, keeping each field's declared type."""
    kwargs = {}
    for name, f in cls.__dataclass_fields__.items():
        if name not in block:
            continue
        value = block[name]
        kwargs[name] = f.type(value) if f.type in (int, float, str, bool) else value
    return cls(**kwargs)


def load_config(path) -> Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}

    feat = raw.get("features", {}) or {}
    mlip = raw.get("mlip", {}) or {}
    eem = raw.get("eem", {}) or {}
    data = raw.get("data", {}) or {}
    train = raw.get("train", {}) or {}

    features_cfg = FeaturesConfig(
        cutoff=float(feat.get("cutoff", FeaturesConfig.cutoff)),
        n_max=int(feat.get("n_max", FeaturesConfig.n_max)),
        l_max=int(feat.get("l_max", FeaturesConfig.l_max)),
        selected_lambdas=tuple(feat.get("selected_lambdas", (0, 2))),
        backend=str(feat.get("backend", FeaturesConfig.backend)),
        density_channels=feat.get("density_channels", FeaturesConfig.density_channels),
    )
    mlip_cfg = MLIPConfig(
        emb_dim=int(mlip.get("emb_dim", MLIPConfig.emb_dim)),
        hidden=int(mlip.get("hidden", MLIPConfig.hidden)),
        depth=int(mlip.get("depth", MLIPConfig.depth)),
    )
    raw_path = data.get("path", DataConfig.path)
    data_cfg = DataConfig(
        path=[str(p) for p in raw_path] if isinstance(raw_path, list) else str(raw_path),
        reference_energies=str(
            data.get("reference_energies", DataConfig.reference_energies)
        ),
        isolated_species=(
            str(data["isolated_species"]) if data.get("isolated_species") else None
        ),
        diabatic_states=(
            str(data["diabatic_states"]) if data.get("diabatic_states") else None
        ),
        atomic_reference_states=(
            str(data["atomic_reference_states"])
            if data.get("atomic_reference_states") else None
        ),
        monomer_path=(str(data["monomer_path"]) if data.get("monomer_path") else None),
        holdout_fraction=float(data.get("holdout_fraction", DataConfig.holdout_fraction)),
        seed=int(data.get("seed", DataConfig.seed)),
    )
    eem_cfg = EEMConfig(
        emb_dim=int(eem.get("emb_dim", EEMConfig.emb_dim)),
        hidden=int(eem.get("hidden", EEMConfig.hidden)),
        depth=int(eem.get("depth", EEMConfig.depth)),
        equiv_channels=int(eem.get("equiv_channels", EEMConfig.equiv_channels)),
        eta_init=float(eem.get("eta_init", EEMConfig.eta_init)),
        eta_floor=float(eem.get("eta_floor", EEMConfig.eta_floor)),
        psd_floor=float(eem.get("psd_floor", EEMConfig.psd_floor)),
    )
    monomer_cfg = _from_block(MonomerConfig, raw.get("monomer", {}) or {})
    sqe_cfg = _from_block(SQEConfig, raw.get("sqe", {}) or {})
    dispersion_cfg = _from_block(DispersionConfig, raw.get("dispersion", {}) or {})
    pauli_cfg = _from_block(PauliConfig, raw.get("pauli", {}) or {})
    elec_cfg = _from_block(ElectrostaticsConfig, raw.get("elec", {}) or {})
    onebody_cfg = _from_block(OneBodyConfig, raw.get("onebody", {}) or {})
    joint_cfg = _from_block(JointConfig, raw.get("joint", {}) or {})
    unified_cfg = _from_block(UnifiedConfig, raw.get("unified", {}) or {})
    train_cfg = TrainConfig(
        epochs=int(train.get("epochs", TrainConfig.epochs)),
        batch_size=int(train.get("batch_size", TrainConfig.batch_size)),
        learning_rate=float(train.get("learning_rate", TrainConfig.learning_rate)),
        weight_decay=float(train.get("weight_decay", TrainConfig.weight_decay)),
        grad_clip=train.get("grad_clip", TrainConfig.grad_clip),
        energy_weight=float(train.get("energy_weight", TrainConfig.energy_weight)),
        force_weight=float(train.get("force_weight", TrainConfig.force_weight)),
        dipole_weight=float(train.get("dipole_weight", TrainConfig.dipole_weight)),
        dmu_dr_weight=float(train.get("dmu_dr_weight", TrainConfig.dmu_dr_weight)),
        dmu_dr_every=int(train.get("dmu_dr_every", TrainConfig.dmu_dr_every)),
        alpha_weight=float(train.get("alpha_weight", TrainConfig.alpha_weight)),
        iso_weight=float(train.get("iso_weight", TrainConfig.iso_weight)),
        atomic_ref_weight=float(
            train.get("atomic_ref_weight", TrainConfig.atomic_ref_weight)
        ),
        free_alpha_weight=float(
            train.get("free_alpha_weight", TrainConfig.free_alpha_weight)
        ),
        unbound_weight=float(train.get("unbound_weight", TrainConfig.unbound_weight)),
        q_l2_weight=float(train.get("q_l2_weight", TrainConfig.q_l2_weight)),
        init_from=str(train.get("init_from", TrainConfig.init_from)),
        eval_every=int(train.get("eval_every", TrainConfig.eval_every)),
        lr_schedule=str(train.get("lr_schedule", TrainConfig.lr_schedule)),
        lr_final_factor=float(
            train.get("lr_final_factor", TrainConfig.lr_final_factor)
        ),
    )
    stages = []
    for i, block in enumerate(raw.get("stages", []) or []):
        block = dict(block)
        name = str(block.pop("name", f"stage{i + 1}"))
        stages.append(StageConfig(name=name, overrides=block))

    return Config(
        run_name=str(raw.get("run_name", "run")),
        device=str(raw.get("device", "auto")),
        dtype=str(raw.get("dtype", "float32")),
        checkpoint_root=str(raw.get("checkpoint_root", "checkpoints")),
        stages=stages,
        features=features_cfg,
        mlip=mlip_cfg,
        eem=eem_cfg,
        monomer=monomer_cfg,
        sqe=sqe_cfg,
        dispersion=dispersion_cfg,
        pauli=pauli_cfg,
        elec=elec_cfg,
        onebody=onebody_cfg,
        joint=joint_cfg,
        unified=unified_cfg,
        data=data_cfg,
        train=train_cfg,
    )
