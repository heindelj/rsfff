"""Fit the intermolecular dispersion term against an EDA component.

    python -m rsfff.train.train_dispersion configs/eda_water_disp.yaml

Trains :class:`rsfff.ff.dispersion.TTDispersion` -- a Tang-Toennies damped C6 backbone
with a learned short-range pair correction -- directly on ``batch.eda["disp"]`` from the
Q-Chem ALMO-EDA water clusters. This is deliberately *standalone*: it does not touch the
Phase-1 monomer/mixture stack, so the intermolecular machinery can be validated on its own
before anything is coupled to it.

The loss is

    MSE(E_pred - E_target)  +  corr_l2_weight * mean(dE^2)  +  r0_weight * r0

The last term is the point of the whole arrangement. ``r0`` sets where the explicit
dispersion is switched off in favor of the neural correction, and a *linear* penalty on it
applies constant downward pressure: the optimizer only widens the neural region if the fit
improves enough to pay for it. That is the "tightly regularized damping length" of
``docs/range_separated_mlip.md`` §4.5, expressed as a pressure rather than a hard constraint.

There are no forces in this data (EDA single points carry no nuclear gradient), so the fit
is energy-only. Suggested curriculum, each stage localizing where the residual lives:

  1. everything frozen (``learn_c6: false, environment_c6: false, correction: false``)
     -- an evaluation, not a fit. Expect ~0.3 kJ/mol MAE on w2 at ``r0_init`` near zero.
  2. per-species C6 only (``learn_c6: true, environment_c6: false``) -- two parameters.
  3. add the C6 environment MLP.
  4. add the pair correction.
  5. optionally ``learn_b``.

Caveat on the target: Q-Chem EDA2's ``eda_disp`` under wB97X-V is the VV10 nonlocal
correlation contribution to the frozen interaction, and ``eda_mod_pauli`` is defined as
``eda_cls_pauli - eda_disp`` -- so ``disp`` and ``mod_pauli`` are entangled by the
decomposition itself. A C6 fit to it is not a transferable Casimir-Polder C6.
"""

from __future__ import annotations

import argparse
import os

# macOS: torch's bundled libomp + conda's llvm-openmp abort with OMP Error #15
# unless this is set before the first OpenMP runtime initializes.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: E402

from ..features.features import FlatLambdaSOAPFeaturizer  # noqa: E402
from ..ff.dispersion import (  # noqa: E402
    DispersionModel,
    DispersionParameterHeads,
    TTDispersion,
    build_log_priors,
)
from ..ff.units import KJMOL_PER_HARTREE  # noqa: E402
from ..mlip.pair_heads import PairEnergyHead  # noqa: E402
from .config import Config, load_config  # noqa: E402
from .data import load_datasets, split_indices  # noqa: E402
from .term_loop import fit  # noqa: E402
from .train_eem import resolve_device  # noqa: E402
from ..neighbors import config_max_num_neighbors as _max_neighbors


def build_dispersion_model(features_cfg, disp_cfg, neighbor_types) -> DispersionModel:
    """Featurizer + :class:`TTDispersion` at the per-species priors, as one module."""
    featurizer = FlatLambdaSOAPFeaturizer(
        cutoff=features_cfg.cutoff,
        n_max=features_cfg.n_max,
        l_max=features_cfg.l_max,
        neighbor_types=neighbor_types,
        selected_lambdas=features_cfg.selected_lambdas,
        backend=features_cfg.backend,
        density_channels=features_cfg.density_channels,
        max_num_neighbors=_max_neighbors(features_cfg),
    )
    p0 = featurizer.feature_dims[0]
    n_species = len(neighbor_types)
    log_c6, log_b = build_log_priors(neighbor_types, b_prior=disp_cfg.b_prior)
    params = DispersionParameterHeads(
        p0, n_species,
        log_c6_prior=log_c6, log_b_prior=log_b,
        emb_dim=disp_cfg.emb_dim, hidden=disp_cfg.hidden, depth=disp_cfg.depth,
        learn_c6=disp_cfg.learn_c6, environment_c6=disp_cfg.environment_c6,
        learn_b=disp_cfg.learn_b, environment_b=disp_cfg.environment_b,
    )
    correction = None
    if disp_cfg.correction:
        if disp_cfg.corr_r_off > features_cfg.cutoff:
            raise ValueError(
                f"dispersion.corr_r_off ({disp_cfg.corr_r_off}) exceeds the feature cutoff "
                f"({features_cfg.cutoff}); past it the pair head reads features whose atoms "
                f"cannot see each other"
            )
        correction = PairEnergyHead(
            p0, n_species,
            emb_dim=disp_cfg.emb_dim, hidden=disp_cfg.corr_hidden,
            depth=disp_cfg.corr_depth, n_radial=disp_cfg.corr_n_radial,
            r_on=disp_cfg.corr_r_on, r_off=disp_cfg.corr_r_off,
            energy_scale=disp_cfg.corr_energy_scale,
        )
    dispersion = TTDispersion(
        params, correction,
        cutoff=disp_cfg.cutoff, taper_width=disp_cfg.taper_width,
        r0_init=disp_cfg.r0_init, alpha=disp_cfg.alpha,
        order=disp_cfg.order, learn_r0=disp_cfg.learn_r0,
    )
    return DispersionModel(
        featurizer, dispersion, intra_fragment=disp_cfg.intra_fragment_features
    )


def _penalties(out, batch, cfg):
    # `batch` is unused here; the signature is shared so a term whose penalties need
    # the labels (the electrostatics multipole loss) can use the same hook.
    extra = {}
    if cfg.corr_l2_weight > 0.0:
        extra["corr_l2"] = cfg.corr_l2_weight * (
            out.e_pair_corr / cfg.energy_scale
        ).pow(2).mean()
    # A *linear* penalty on r0 applies constant downward pressure: the optimizer only
    # widens the neural region if the fit improves enough to pay for it.
    if cfg.r0_weight > 0.0:
        extra["r0"] = cfg.r0_weight * out.r0
    return extra


def _diagnostics(out, batch, target):
    return {"r0": float(out.r0.detach())}


_LOG_KEYS = ("loss", "mae", "rmse", "ff_mae", "corr_share", "r0")


def train(config: Config):
    torch.set_default_dtype(torch.float64 if config.dtype == "float64" else torch.float32)
    device = resolve_device(config.device, config.dtype)
    dcfg = config.dispersion
    print(f"[{config.run_name}] device={device} dtype={config.dtype} target=eda_{dcfg.target}")

    dataset = load_datasets(config.data.path, dtype=torch.get_default_dtype())
    if not dataset.has_fragments:
        raise ValueError(
            "the dispersion term is defined over inter-fragment pairs, but the dataset has "
            "no fragment partition; the extxyz needs a `fragment_idx` column"
        )
    neighbor_types = dataset.unique_atomic_numbers
    print(f"loaded {len(dataset)} frames; species Z={neighbor_types}")

    model = build_dispersion_model(config.features, dcfg, neighbor_types).to(device)
    train_idx, val_idx = split_indices(
        len(dataset), config.data.holdout_fraction, config.data.seed
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"train/val = {len(train_idx)}/{len(val_idx)}; trainable params = {n_params}; "
        f"r0 init {float(model.r0.detach()):.3f} A (learnable={dcfg.learn_r0}, "
        f"penalty {dcfg.r0_weight:g} per A, vs a fit term of 1.0 at "
        f"{dcfg.energy_scale * KJMOL_PER_HARTREE:.3g} kJ/mol error)"
    )

    def report(m):
        with torch.no_grad():
            c6 = (m.params.log_c6_prior + m.params.d_log_c6).exp()
            b = (m.params.log_b_prior + m.params.d_log_b).exp()
        print(f"per-species C6 (Ha*bohr^6): {dict(zip(neighbor_types, c6.tolist()))}")
        print(f"per-species b  (1/bohr):    {dict(zip(neighbor_types, b.tolist()))}")
        print(f"r0 = {float(m.r0.detach()):.4f} A")

    fit(
        model, dataset, config, dcfg, device, train_idx, val_idx,
        log_keys=_LOG_KEYS, penalties=_penalties, diagnostics=_diagnostics, report=report,
    )
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Fit Tang-Toennies dispersion + a pair correction to an EDA component."
    )
    parser.add_argument("config", type=str, help="path to YAML config")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
