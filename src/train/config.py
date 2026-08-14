"""YAML config -> typed dataclasses for the MLIP training pipeline.

Top-level YAML blocks: ``features:`` (SOAP featurizer), ``mlip:`` (MLP head),
``data:`` (dataset + split), ``train:`` (optimizer / loss weights). Parsing mirrors the
``load_config`` pattern in the reference repo: ``yaml.safe_load`` then nested
``.get(key, default)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    path: str | list[str] = "data/labels/h2o.extxyz"
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
    learn_quadrupole: bool = True
    cquad_init: float = 1.0         # e^2 a0^4 / Ha
    cquad_floor: float = 1.0e-4
    environment_cquad: bool = False
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
    #: Monomer frames drawn per step for the anchor term; 0 uses the whole file every step.
    #: The anchor carries a force term, so evaluating it is a second-order backward, and at
    #: the full 500 frames that is ~95% of the wall time of a training step -- the identical
    #: computation repeated for every minibatch of the actual training set. Sampling a fresh
    #: subset each step is plain SGD on that term: the same expected gradient, ~8x cheaper,
    #: and over an epoch it still sees every monomer. Evaluation uses a fixed leading slice
    #: so the validation number stays comparable across epochs.
    anchor_batch_size: int = 64
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


@dataclass
class Config:
    run_name: str = "run"
    device: str = "auto"       # auto -> cuda > mps > cpu
    dtype: str = "float32"     # float32 | float64
    checkpoint_root: str = "checkpoints"
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
        eval_every=int(train.get("eval_every", TrainConfig.eval_every)),
    )
    return Config(
        run_name=str(raw.get("run_name", "run")),
        device=str(raw.get("device", "auto")),
        dtype=str(raw.get("dtype", "float32")),
        checkpoint_root=str(raw.get("checkpoint_root", "checkpoints")),
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
