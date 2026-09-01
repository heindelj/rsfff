"""Fit the 1-body term alone: atomic references plus an intramolecular bond energy.

    python -m rsfff.train.train_onebody configs/water_onebody.yaml

**This is the reduced form**, without the response solve: ``E_1body = E0 + E_bond``. The
full term includes ``E_internal`` from :class:`rsfff.ff.response.FragmentResponse`, but
that piece only makes sense trained together with the electrostatics -- the response has to
produce multipoles that interact correctly *and* an internal energy that is worth what it
cost, and neither target alone determines it. Use
``python -m rsfff.train.train_onebody_elec configs/water_onebody_elec.yaml`` for that; this
script remains useful for isolating the bond head, which is the part with all the force
signal in it.

Trains :class:`rsfff.ff.onebody.OneBodyModel` against the isolated-fragment energies a
Q-Chem EDA job prints (``batch.fragment_energy``) and, on the isolated-monomer frames,
against the analytic forces.

Every fragment is an independent example: a pentamer frame contributes five monomer
energies, so the ~9.6k cluster frames supply ~34k labels at exactly the geometries monomers
take inside clusters. The 500 standalone ``h2o`` frames add the force signal -- there the
QC gradient *is* ``-dE_1body/dR``, while on a cluster frame it is the sum over every term
and cannot be attributed to this one. That is why ``force_weight`` is an explicit knob
rather than "use forces when the labels exist".

What to watch:

- ``mae`` -- per-fragment energy error, kJ/mol. This is an *absolute* monomer energy, so
  the constant offset between ``E0`` and the reference level of theory shows up here first.
- ``bond`` -- mean bond energy per fragment, Hartree. Water's two O-H bonds are about
  -0.37 Ha below the free atoms; a value near zero means the head has not engaged.
- ``f_mae`` -- force error, Hartree/Angstrom, only when ``force_weight > 0``.

``E0`` is a frozen buffer and must come from atomic calculations at the *same* level of
theory as the fragment energies. For water-only data ``n_H = 2 n_O`` exactly, so the data
constrains only ``E0_O + 2 E0_H``; a learnable ``E0`` would fit that one combination and
leave the split arbitrary, which is invisible until an ion-water system arrives.
"""

from __future__ import annotations

import argparse
import os

# macOS: torch's bundled libomp + conda's llvm-openmp abort with OMP Error #15
# unless this is set before the first OpenMP runtime initializes.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: E402

from ..features.features import FlatLambdaSOAPFeaturizer  # noqa: E402
from ..ff.onebody import OneBodyEnergy, OneBodyModel  # noqa: E402
from ..ff.units import KJMOL_PER_HARTREE  # noqa: E402
from ..mlip.pair_heads import PairEnergyHead  # noqa: E402
from .config import Config, load_config  # noqa: E402
from .data import (  # noqa: E402
    load_datasets,
    load_monomer_batch,
    load_reference_energies,
    split_indices,
)
from .loss import onebody_anchor_loss, onebody_fit  # noqa: E402
from .term_loop import fit  # noqa: E402
from .train_eem import resolve_device  # noqa: E402
from ..neighbors import config_max_num_neighbors as _max_neighbors

_LOG_KEYS = ("loss", "mae", "rmse", "bond", "anchor_e", "a_mae", "f_mae")


def build_onebody_model(features_cfg, cfg, neighbor_types, reference_energies):
    """Featurizer grouped by fragment + :class:`OneBodyEnergy`."""
    if cfg.bond_r_off > features_cfg.cutoff:
        raise ValueError(
            f"onebody.bond_r_off ({cfg.bond_r_off}) exceeds the feature cutoff "
            f"({features_cfg.cutoff}); past it the pair head reads features whose atoms "
            f"cannot see each other"
        )
    lambdas = tuple(int(v) for v in features_cfg.selected_lambdas)
    if 2 not in lambdas:
        # The bond head reads invariants only, but the featurizer always emits an
        # equiv_feats block, so lambda=2 is required whatever the head consumes.
        raise ValueError(
            f"features.selected_lambdas is {list(lambdas)}; the featurizer needs 2 even "
            f"though the bond head reads only invariants. Use [0, 2]."
        )
    featurizer = FlatLambdaSOAPFeaturizer(
        cutoff=features_cfg.cutoff, n_max=features_cfg.n_max, l_max=features_cfg.l_max,
        neighbor_types=neighbor_types, selected_lambdas=lambdas,
        backend=features_cfg.backend, density_channels=features_cfg.density_channels,
        max_num_neighbors=_max_neighbors(features_cfg),
    )
    bond_head = PairEnergyHead(
        featurizer.feature_dims[0], len(neighbor_types),
        emb_dim=cfg.emb_dim, hidden=cfg.bond_hidden, depth=cfg.bond_depth,
        n_radial=cfg.bond_n_radial, r_on=cfg.bond_r_on, r_off=cfg.bond_r_off,
        energy_scale=cfg.bond_energy_scale,
    )
    return OneBodyModel(featurizer, OneBodyEnergy(bond_head, reference_energies))


class AnchorTerms:
    """The isolated-monomer anchor as a ``(penalties, diagnostics)`` pair of callbacks.

    One object rather than two closures because the anchor's forward pass is not cheap --
    it differentiates through the whole model for the force term -- and it must happen once
    per step, not once for the loss and again for the log. ``term_loop.run_epoch`` calls
    ``penalties`` and then ``diagnostics`` on the same batch, so caching the metrics from
    the former is enough.
    """

    def __init__(self, model, anchor_batch):
        self.model = model
        self.batch = anchor_batch
        self._metrics: dict[str, float] = {}

    def penalties(self, out, batch, cfg):
        if self.batch is None:
            return {}
        terms, self._metrics = onebody_anchor_loss(self.model, self.batch, cfg)
        return terms

    def diagnostics(self, out, batch, target):
        return {"ref": float(out.energy_ref.detach().mean()), **self._metrics}


def train(config: Config):
    torch.set_default_dtype(torch.float64 if config.dtype == "float64" else torch.float32)
    device = resolve_device(config.device, config.dtype)
    cfg = config.onebody
    print(f"[{config.run_name}] device={device} dtype={config.dtype} target=fragment_energy")

    dataset = load_datasets(config.data.path, dtype=torch.get_default_dtype())
    if not dataset.has_fragments:
        raise ValueError(
            "the 1-body term is fit per fragment but the dataset has no fragment "
            "partition; the extxyz needs a `fragment_idx` column"
        )
    if dataset._fragment_energy is None:
        raise ValueError(
            "the dataset carries no `fragment_energies` header, which is the 1-body "
            "target; use the files written by scripts/parse_roundtrip.py"
        )
    neighbor_types = dataset.unique_atomic_numbers
    reference_energies = load_reference_energies(
        config.data.reference_energies, neighbor_types
    ).to(torch.get_default_dtype())
    print(
        f"loaded {len(dataset)} frames; species Z={neighbor_types}; "
        f"E0 = {dict(zip(neighbor_types, reference_energies.tolist()))} "
        f"({config.data.reference_energies})"
    )

    # The isolated monomers cannot be concatenated with the cluster files -- they carry no
    # eda_* components and they do carry forces, and `concatenate_datasets` refuses the mix
    # (rightly, since either would then be silently dropped). They ride as an anchor, which
    # is also the only place a force fit is attributable to this term.
    anchor = None
    if config.data.monomer_path:
        anchor = load_monomer_batch(
            config.data.monomer_path, dtype=torch.get_default_dtype()
        ).to(device)
        if anchor.fragment_energy is None:
            raise ValueError(f"{config.data.monomer_path} carries no fragment energies")
        if cfg.force_weight > 0.0 and anchor.forces is None:
            raise ValueError(
                f"onebody.force_weight > 0 but {config.data.monomer_path} carries no forces"
            )
        anchor.positions.requires_grad_(cfg.force_weight > 0.0)
        print(f"monomer anchor: {anchor.n_fragments} frames from {config.data.monomer_path}")
    elif cfg.force_weight > 0.0:
        raise ValueError(
            "onebody.force_weight > 0 but data.monomer_path is not set; cluster forces are "
            "the sum over every force-field term and cannot be attributed to this one"
        )

    model = build_onebody_model(
        config.features, cfg, neighbor_types, reference_energies
    ).to(device)
    train_idx, val_idx = split_indices(
        len(dataset), config.data.holdout_fraction, config.data.seed
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"train/val = {len(train_idx)}/{len(val_idx)}; trainable params = {n_params}; "
        f"force_weight = {cfg.force_weight}; "
        f"fit term 1.0 at {cfg.energy_scale * KJMOL_PER_HARTREE:.3g} kJ/mol per fragment"
    )

    anchor_terms = AnchorTerms(model, anchor)
    fit(
        model, dataset, config, cfg, device, train_idx, val_idx,
        log_keys=_LOG_KEYS, fit_term=onebody_fit,
        penalties=anchor_terms.penalties, diagnostics=anchor_terms.diagnostics,
    )
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Fit the 1-body term (atomic references + intramolecular bond energy)."
    )
    parser.add_argument("config", type=str, help="path to YAML config")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
