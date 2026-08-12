"""Fit the classical electrostatics term against an EDA component.

    python -m rsfff.train.train_elec configs/eda_water_elec.yaml

Trains :class:`rsfff.ff.electrostatics.SlaterElectrostatics` -- response parameters from
intra-fragment descriptors, a local per-monomer SQE solve for the multipoles, point-multipole
plus Slater-penetration interaction, and a pairwise correction -- against
``batch.eda["cls_elec"]``.

The distinguishing property is that this model is **exactly two-body**: descriptors and the
charge solve are both grouped by fragment, so every ``E^(k>=3)`` of the many-body expansion
is zero to round-off. That is checked in ``tests/test_ff_electrostatics.py`` rather than
assumed, and it is why ``intra_fragment_features`` defaults to true and needs a second
explicit flag to disable.

What to watch, in order of how much it tells you:

- ``q_res`` -- the per-fragment charge residual. Must sit at machine zero. Anything else
  means the solve grouping broke and the model is no longer what it claims to be.
- ``ff_mae`` vs ``mae`` -- the backbone alone against the backbone plus correction.
- ``r0`` -- should drift down under its penalty unless the correction is buying something.
- ``qO``, ``dchi``, ``mu``, ``quad`` -- the solved multipoles. At ``max_rank: 1`` the O
  charge settles near -0.4 to -0.5 e with ``dchi > 0``, which is the physical ordering.
- ``dip_mae``, ``quad_mae`` -- the frozen-monomer multipole errors, in e*a0 and e*a0^2.
  Reported whether or not they are being fit.

**Measured on w2, 12 epochs, and worth knowing before reading a rank-2 run:**

===============================  =====  =========  ==========  =======  =======
run                              mae    dip_mae    quad_mae    qO       dchi
===============================  =====  =========  ==========  =======  =======
rank 1, with the monomer anchor  1.72   0.022      0.151       -0.588   +0.802
rank 2, energy only              0.99   0.009      0.037       -0.107   -0.201
rank 2, with the monomer anchor  0.62   0.008      0.018       -0.062   -0.282
===============================  =====  =========  ==========  =======  =======

Rank 2 is worth ~3x on the energy and the anchor buys a further 1.6x on top of it -- but
**the atomic charges collapse toward zero and ``dchi`` flips sign**. Once atomic
quadrupoles exist the model can reproduce a monomer's field without separating any charge,
and neither ``cls_elec`` nor a *molecular* multipole target distinguishes the two: the
energy-only row shows the collapse is already there without any multipole supervision.
The anchor pulls it back somewhat (-0.062 against -0.009 when the same terms are applied
to the in-batch cluster fragments instead) but does not fix it, because it constrains the
molecular sum and not the partition.

That is the rank-2 continuation of the saddle documented at
``rsfff.ff.electrostatics.DEFAULT_Q0_PRIOR``. A plain L2 toward ``q0`` is **not** the fix:
it wrecks the charges under SQE while leaving the dipoles looking fine.

**What does help is training this jointly with the 1-body term** --
``python -m rsfff.train.train_onebody_elec`` -- which puts the solve's ``internal_energy``
into a per-fragment energy target. Charge separation then has a *cost* that the fragment
energies can see, and measured on w2 the charges stay physical at ``max_rank 2``
(``qO -0.41``, ``dchi +0.30``) instead of collapsing. This script fits the interaction
alone and is the right tool for isolating that term; it is not the right tool for
determining the monomer.

Curriculum: (i) everything frozen -- an evaluation, not a fit; charges start at the IP/EA
seeding, not at any fitted water model, so this line is *not* the pyCMM anchor;
(ii) ``chi``/``eta`` per species only; (iii) add the environment MLPs; (iv) enable the dipole
sector; (v) add the correction head; (vi) try ``max_rank: 2``.

Caveat on the target: ``eda_cls_elec`` is the Coulomb interaction of the *frozen* monomer
densities, so it excludes polarization and charge transfer by construction -- which is
exactly why the two-body form is legitimate here and will stop being so once polarization is
added.
"""

from __future__ import annotations

import argparse
import os

# macOS: torch's bundled libomp + conda's llvm-openmp abort with OMP Error #15
# unless this is set before the first OpenMP runtime initializes.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: E402

from ..features.features import FlatLambdaSOAPFeaturizer  # noqa: E402
from ..ff.electrostatics import (  # noqa: E402
    DEFAULT_ELEC_PRIOR,
    ElectrostaticParameterHeads,
    ElectrostaticsModel,
    FragmentResponse,
    SlaterElectrostatics,
    build_elec_priors,
)
from ..ff.multipole import irrep2_to_spherical  # noqa: E402
from ..ff.units import KJMOL_PER_HARTREE  # noqa: E402
from ..mlip.pair_heads import PairEnergyHead  # noqa: E402
from ..mlip.reference_states import AtomicStateReference  # noqa: E402
from ..mlip.sqe import PairComplianceHead  # noqa: E402
from .config import Config, load_config  # noqa: E402
from .data import load_datasets, load_monomer_batch, split_indices  # noqa: E402
from .loss import fragment_multipole_loss  # noqa: E402
from .term_loop import fit  # noqa: E402
from .train_eem import resolve_device  # noqa: E402

_LOG_KEYS = ("loss", "mae", "rmse", "ff_mae", "corr_share", "r0", "q_res", "qO",
             "dchi", "mu", "quad", "dip_mae", "quad_mae")


def build_featurizer(features_cfg, ecfg, neighbor_types):
    """The shared descriptor. Factored out so the composite model builds exactly one."""
    lambdas = tuple(int(v) for v in features_cfg.selected_lambdas)
    if ecfg.max_rank >= 1 and ecfg.learn_dipole and not {1, 2} <= set(lambdas):
        raise ValueError(
            "the dipole sector needs lambda=1 (chivec) and lambda=2 (alpha); set "
            "features.selected_lambdas: [0, 1, 2] or max_rank: 0"
        )
    return FlatLambdaSOAPFeaturizer(
        cutoff=features_cfg.cutoff, n_max=features_cfg.n_max, l_max=features_cfg.l_max,
        neighbor_types=neighbor_types, selected_lambdas=lambdas,
        backend=features_cfg.backend, density_channels=features_cfg.density_channels,
    )


def build_response(featurizer, features_cfg, ecfg, neighbor_types, atomic_states=None):
    """The per-fragment response solve, seeded from the atomic IP/EA data.

    Built separately from the electrostatics because the 1-body term shares it: see
    :mod:`rsfff.ff.onebody_elec`. One instance, two consumers.
    """
    p0 = featurizer.feature_dims[0]
    p1, p2 = featurizer.feature_dims.get(1), featurizer.feature_dims.get(2)
    log_z, log_b, q0 = build_elec_priors(neighbor_types)

    # Mulliken (IP + EA)/2 and (IP - EA) are the free-atom limits of the SQE chi and eta, so
    # seeding there starts the charge solve in a physical regime. Without it chi is uniform,
    # every atom is equally electronegative, and the solved charges are identically zero.
    chi_init = eta_init = None
    if atomic_states is not None:
        chi_init, eta_init = atomic_states.head_bias_init(eta_default=0.5)

    params = ElectrostaticParameterHeads(
        p0, p1, p2, len(neighbor_types),
        log_z_prior=log_z, log_b_prior=log_b, q0_prior=q0,
        irrep6_to_voigt=featurizer.backend.irrep6_to_voigt(),
        irrep2_to_spherical_map=(
            irrep2_to_spherical(featurizer.backend.irrep6_to_voigt()) if p2 is not None
            else None
        ),
        emb_dim=ecfg.emb_dim, hidden=ecfg.hidden, depth=ecfg.depth,
        equiv_channels=ecfg.equiv_channels, max_rank=ecfg.max_rank,
        chi_init=chi_init, eta_init=eta_init,
        eta_floor=ecfg.eta_floor, psd_floor=ecfg.psd_floor,
        environment_chi=ecfg.environment_chi, environment_eta=ecfg.environment_eta,
        learn_z=ecfg.learn_z, environment_z=ecfg.environment_z,
        learn_b=ecfg.learn_b, environment_b=ecfg.environment_b,
        learn_dipole=ecfg.learn_dipole,
        learn_quadrupole=ecfg.learn_quadrupole, cquad_init=ecfg.cquad_init,
        cquad_floor=ecfg.cquad_floor, environment_cquad=ecfg.environment_cquad,
    )
    compliance = PairComplianceHead(
        p0, hidden=ecfg.compliance_hidden, depth=ecfg.compliance_depth,
        cutoff=features_cfg.cutoff, s_init=ecfg.s_init,
    )
    return FragmentResponse(params, compliance)


def build_slater_elec(featurizer, response, features_cfg, ecfg, neighbor_types):
    """:class:`SlaterElectrostatics` on an already-built response."""
    p0 = featurizer.feature_dims[0]
    correction = None
    if ecfg.correction:
        if ecfg.corr_r_off > features_cfg.cutoff:
            raise ValueError(
                f"elec.corr_r_off ({ecfg.corr_r_off}) exceeds the feature cutoff "
                f"({features_cfg.cutoff}); past it the pair head reads features whose atoms "
                f"cannot see each other"
            )
        correction = PairEnergyHead(
            p0, len(neighbor_types), emb_dim=ecfg.emb_dim, hidden=ecfg.corr_hidden,
            depth=ecfg.corr_depth, n_radial=ecfg.corr_n_radial,
            r_on=ecfg.corr_r_on, r_off=ecfg.corr_r_off,
            energy_scale=ecfg.corr_energy_scale,
        )
    return SlaterElectrostatics(
        response, correction,
        cutoff=ecfg.cutoff, taper_width=ecfg.taper_width,
        r0_init=ecfg.r0_init, alpha=ecfg.alpha,
        max_rank=ecfg.max_rank, learn_r0=ecfg.learn_r0,
    )


def build_elec_model(features_cfg, ecfg, neighbor_types, atomic_states=None):
    """Featurizer + :class:`SlaterElectrostatics`, seeded from the atomic IP/EA data."""
    featurizer = build_featurizer(features_cfg, ecfg, neighbor_types)
    response = build_response(
        featurizer, features_cfg, ecfg, neighbor_types, atomic_states
    )
    elec = build_slater_elec(featurizer, response, features_cfg, ecfg, neighbor_types)
    return ElectrostaticsModel(
        featurizer, elec,
        intra_fragment=ecfg.intra_fragment_features,
        allow_environment=ecfg.allow_environment,
    )


def make_penalties(model, monomer_batch):
    """Build the penalty callback, closing over the isolated-monomer anchor batch.

    The multipole terms are the only thing constraining the *individual* multipoles: the
    interaction energy fixes only combinations of them. They are evaluated on the anchor
    (standalone monomers, whose geometry range covers the in-cluster one) rather than on
    the training batch; ``fragment_multipole_weight`` turns on the in-batch cluster
    fragments as well.
    """
    def _penalties(out, batch, cfg):
        extra = {}
        scale = cfg.energy_scale
        if cfg.corr_l2_weight > 0.0:
            extra["corr_l2"] = cfg.corr_l2_weight * (out.e_pair_corr / scale).pow(2).mean()
        if cfg.r0_weight > 0.0:
            extra["r0"] = cfg.r0_weight * out.r0

        if monomer_batch is not None and (cfg.dipole_weight or cfg.quadrupole_weight):
            # A single-fragment frame has no inter-fragment pairs, so this forward is the
            # solve and the heads only -- the interaction energy it returns is zero.
            anchor = model(monomer_batch)
            terms, _ = fragment_multipole_loss(
                anchor, monomer_batch,
                dipole_weight=cfg.dipole_weight,
                quadrupole_weight=cfg.quadrupole_weight,
                dipole_scale=cfg.dipole_scale, quadrupole_scale=cfg.quadrupole_scale,
            )
            extra.update(terms)

        if cfg.fragment_multipole_weight > 0.0:
            w = cfg.fragment_multipole_weight
            terms, _ = fragment_multipole_loss(
                out, batch,
                dipole_weight=w * cfg.dipole_weight,
                quadrupole_weight=w * cfg.quadrupole_weight,
                dipole_scale=cfg.dipole_scale, quadrupole_scale=cfg.quadrupole_scale,
            )
            extra.update({f"frag_{k}": v for k, v in terms.items()})
        return extra

    return _penalties


def _diagnostics(out, batch, target):
    frag = batch.fragment_idx
    n_frag = int(batch.n_fragments)
    per_frag = out.charges.detach().new_zeros(n_frag).index_add_(0, frag, out.charges.detach())
    want = (
        per_frag.new_zeros(n_frag) if batch.fragment_charge is None
        else batch.fragment_charge.to(per_frag.dtype)
    )
    z = batch.atomic_numbers
    metrics = {
        # Must be machine zero. A nonzero value means the per-fragment solve grouping broke,
        # which would silently turn this into something other than classical electrostatics.
        "q_res": float((per_frag - want).abs().max()),
        "r0": float(out.r0.detach()),
    }
    oxygen, hydrogen = z == 8, z == 1
    if bool(oxygen.any()):
        metrics["qO"] = float(out.charges.detach()[oxygen].mean())
    if bool(oxygen.any()) and bool(hydrogen.any()):
        # Effective electronegativity difference, including the environment MLP -- the
        # per-species bias alone is not what the solve sees. The SQE functional carries
        # `+chi_i q_i`, so lowering the energy puts *negative* charge where chi is large:
        # a **positive** value is the physical ordering (O pulls electrons off H). Read it
        # together with qO -- on water the two move together, chi_O - chi_H ~ +0.22 against
        # q_O ~ -0.39.
        chi = out.chi.detach()
        metrics["dchi"] = float(chi[oxygen].mean() - chi[hydrogen].mean())
    if out.mu is not None:
        metrics["mu"] = float(out.mu.detach().norm(dim=-1).mean())
    if out.quad_s is not None:
        # Without this a quadrupole sector that silently stays at zero is invisible:
        # the head is zero-initialized and nothing else in the line would move.
        metrics["quad"] = float(out.quad_s.detach().norm(dim=-1).mean())
    # Reported whenever the labels exist, at zero weight too -- the monomer being right
    # is the point, and it should be visible before it is being optimized. Under no_grad
    # because this rebuilds every fragment's multipoles and the graph would be discarded.
    with torch.no_grad():
        _, mm = fragment_multipole_loss(
            out, batch, dipole_weight=1.0, quadrupole_weight=1.0,
            dipole_scale=1.0, quadrupole_scale=1.0,
        )
    metrics.update(mm)
    return metrics


def train(config: Config):
    torch.set_default_dtype(torch.float64 if config.dtype == "float64" else torch.float32)
    device = resolve_device(config.device, config.dtype)
    ecfg = config.elec
    print(f"[{config.run_name}] device={device} dtype={config.dtype} target=eda_{ecfg.target}")

    dataset = load_datasets(config.data.path, dtype=torch.get_default_dtype())
    if not dataset.has_fragments:
        raise ValueError(
            "the electrostatics term is defined between fragments and solved per fragment, "
            "but the dataset has no fragment partition; the extxyz needs a `fragment_idx` "
            "column"
        )
    neighbor_types = dataset.unique_atomic_numbers
    print(f"loaded {len(dataset)} frames; species Z={neighbor_types}")

    atomic_states = None
    if config.data.atomic_reference_states:
        atomic_states = AtomicStateReference.from_json(
            config.data.atomic_reference_states, neighbor_types,
            dtype=torch.get_default_dtype(),
        )

    monomer_batch = None
    if config.data.monomer_path:
        monomer_batch = load_monomer_batch(
            config.data.monomer_path, dtype=torch.get_default_dtype()
        ).to(device)
        if monomer_batch.fragment_dipole is None:
            raise ValueError(
                f"{config.data.monomer_path} carries no `fragment_dipoles` header, so it "
                f"cannot anchor the multipoles; use a file from scripts/parse_roundtrip.py"
            )
        print(
            f"multipole anchor: {monomer_batch.n_fragments} monomers from "
            f"{config.data.monomer_path}"
        )
    elif ecfg.dipole_weight or ecfg.quadrupole_weight:
        raise ValueError(
            "elec.dipole_weight/quadrupole_weight are set but data.monomer_path is not; "
            "point it at an isolated-monomer file, or use "
            "elec.fragment_multipole_weight to fit the in-batch cluster fragments instead"
        )

    model = build_elec_model(config.features, ecfg, neighbor_types, atomic_states).to(device)
    train_idx, val_idx = split_indices(
        len(dataset), config.data.holdout_fraction, config.data.seed
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"train/val = {len(train_idx)}/{len(val_idx)}; trainable params = {n_params}; "
        f"max_rank {ecfg.max_rank}; two-body-exact = {model.intra_fragment}; "
        f"chi/eta seeded from IP/EA = {atomic_states is not None}; "
        f"fit term 1.0 at {ecfg.energy_scale * KJMOL_PER_HARTREE:.3g} kJ/mol error"
    )

    def report(m):
        with torch.no_grad():
            z = (m.params.log_z_prior + m.params.d_log_z).exp()
            b = (m.params.log_b_prior + m.params.d_log_b).exp()
            # Effective values averaged over a real batch, not the per-species bias: with the
            # environment MLPs on, the bias alone is not what the solve sees.
            sample = dataset.flat_batch(range(min(200, len(dataset)))).to(device)
            out = m(sample)
            zs, chi_e, eta_e, q_e = (sample.atomic_numbers, out.chi.detach(),
                                     out.eta.detach(), out.charges.detach())
        for t_i, t_z in enumerate(neighbor_types):
            sel = zs == t_z
            if not bool(sel.any()):
                continue
            print(f"  Z={t_z}: chi {float(chi_e[sel].mean()):+.4f}  eta "
                  f"{float(eta_e[sel].mean()):.4f}  q {float(q_e[sel].mean()):+.4f} e "
                  f"(effective, batch mean)")
        print(f"per-species Z (e):       {dict(zip(neighbor_types, z.tolist()))}  "
              f"(prior {[DEFAULT_ELEC_PRIOR[t][0] for t in neighbor_types]})")
        print(f"per-species b (1/bohr):  {dict(zip(neighbor_types, b.tolist()))}  "
              f"(prior {[DEFAULT_ELEC_PRIOR[t][1] for t in neighbor_types]})")
        print(f"r0 = {float(m.r0.detach()):.4f} A")

    fit(
        model, dataset, config, ecfg, device, train_idx, val_idx,
        log_keys=_LOG_KEYS, penalties=make_penalties(model, monomer_batch),
        diagnostics=_diagnostics, report=report,
    )
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Fit classical electrostatics (local SQE + Slater penetration) to an EDA "
                    "component."
    )
    parser.add_argument("config", type=str, help="path to YAML config")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
