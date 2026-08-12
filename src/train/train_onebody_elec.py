"""Fit the 1-body term and the classical electrostatics **together**, sharing one solve.

    python -m rsfff.train.train_onebody_elec configs/water_onebody_elec.yaml

Trains :class:`rsfff.ff.onebody_elec.OneBodyElecModel`: one featurizer, one
:class:`rsfff.ff.response.FragmentResponse`, and two consumers of it -- the 1-body energy
(``E0 + E_internal + E_bond``, per fragment) and the electrostatic interaction (multipoles
plus Slater penetration over inter-fragment pairs).

Why jointly rather than in sequence
-----------------------------------
Each target alone leaves the response under-determined in a different direction, so fitting
them one after another just means the second undoes part of the first:

- ``cls_elec`` sees the multipoles only through their pairwise interaction. Measured at
  ``max_rank 2``: the energy MAE keeps improving while the atomic charges collapse toward
  zero and the electronegativity ordering inverts.
- the **fragment multipole** targets fix ``q``, ``mu``, ``Theta`` but not the response
  *energy* -- ``(C, chiquad) -> (lambda C, chiquad/lambda)`` leaves ``Theta`` invariant and
  scales ``E_quad`` by ``1/lambda``.
- the **fragment energy** target is the only one that sees ``E_internal``, so it is what
  closes that gauge. Softly: the bond head is a smooth function of the same geometry and
  can absorb part of what ``E_internal`` would otherwise have to explain.

The four loss terms, all normalized to 1.0 at one ``energy_scale`` of error:

==================  =========================================  ====================
term                target                                     source
==================  =========================================  ====================
``onebody``         ``batch.fragment_energy``  (per fragment)   cluster batch
``elec``            ``batch.eda["cls_elec"]``  (per frame)      cluster batch
``dipole``/``quad`` frozen-monomer multipoles                   monomer anchor
``anchor_e/_f``     monomer energy and forces                   monomer anchor
==================  =========================================  ====================

Measured, w2, ``max_rank 2``, 24 epochs at the shipped weights
--------------------------------------------------------------

=========  ========  ========  =========  =========  ========  =======  =======
epoch      ob_mae    el_mae    dip_mae    quad_mae   f_mae     qO       dchi
=========  ========  ========  =========  =========  ========  =======  =======
untrained  1015      13.86     0.451      1.62       0.0599    -0.256   +0.049
6          7.12      1.94      0.118      0.848      0.0031    -0.450   +0.361
12         3.58      1.42      0.077      0.530      0.0016    -0.440   +0.345
24         8.39      0.98      0.053      0.294      0.0014    -0.427   +0.322
=========  ========  ========  =========  =========  ========  =======  =======

``ob_mae``/``el_mae`` kJ/mol, ``dip_mae`` e*a0, ``quad_mae`` e*a0^2, ``f_mae`` Ha/Angstrom.

**The charges stay physical.** Fitting ``cls_elec`` alone at ``max_rank 2`` drives
``qO -> -0.06`` with ``dchi`` inverted to ``-0.28``; here they sit at ``-0.43`` and
``+0.32`` from the first evaluation onward, which is where rank *1* electrostatics put them
-- with rank 2's accuracy. That is the whole argument for the joint fit: charge separation
now has a cost that the fragment energies can see.

``internal`` settles near ``-0.12`` Ha and ``bond`` near ``-0.24``, summing to water's
``-0.36`` Ha of bond energy. Both are engaged; neither is doing all the work.

Two caveats visible in that table. ``quad_mae`` is still falling at epoch 24 and is far from
the ``0.018`` an electrostatics-only fit reaches -- there the quadrupole is unconstrained by
any energy, here it also has to pay for itself. And ``ob_mae`` is not monotone (3.58 at
epoch 12, 8.39 at 24) while every other column improves: the 1-body term is the largest and
noisiest part of the loss, and it is the first thing to try a lower learning rate on.

What to watch:

- ``q_res`` -- per-fragment charge residual, must stay at machine zero.
- ``ob_mae`` vs ``el_mae`` -- the two energy errors, both kJ/mol. They start four orders of
  magnitude apart (the bond head is zero-initialized, so the 1-body error opens at ~1000
  kJ/mol against ``cls_elec``'s ~14); ``joint.elec_weight`` exists for that.
- ``qO`` and ``dchi`` -- whether the charges stay physical. The SQE functional carries
  ``+chi_i q_i``, so **positive** ``dchi`` is the physical ordering (O pulls electrons off
  H) and should sit alongside a clearly negative ``qO``.
- ``internal`` and ``bond`` -- the two learned pieces of the 1-body energy, in Hartree. If
  ``internal`` drifts while ``bond`` compensates, that is the soft gauge above showing up.
"""

from __future__ import annotations

import argparse
import os

# macOS: torch's bundled libomp + conda's llvm-openmp abort with OMP Error #15
# unless this is set before the first OpenMP runtime initializes.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: E402

from ..ff.onebody import OneBodyEnergy  # noqa: E402
from ..ff.onebody_elec import OneBodyElecModel  # noqa: E402
from ..ff.units import KJMOL_PER_HARTREE  # noqa: E402
from ..mlip.pair_heads import PairEnergyHead  # noqa: E402
from ..mlip.reference_states import AtomicStateReference  # noqa: E402
from .config import Config, load_config  # noqa: E402
from .data import (  # noqa: E402
    load_datasets,
    load_monomer_batch,
    load_reference_energies,
    split_indices,
)
from .loss import fragment_multipole_loss, onebody_anchor_loss  # noqa: E402
from .term_loop import fit  # noqa: E402
from .train_eem import resolve_device  # noqa: E402
from .train_elec import (  # noqa: E402
    build_featurizer,
    build_response,
    build_slater_elec,
)

#: The penalty *values* (`anchor_e`, `anchor_f`, `dipole`, `quad`) are logged alongside the
#: errors on purpose: with five terms in one loss, a term that is quietly a thousand times
#: larger than the others is the single most likely thing to go wrong, and no error column
#: would show it.
_LOG_KEYS = ("loss", "ob_mae", "el_mae", "a_mae", "f_mae", "dip_mae", "quad_mae",
             "anchor_e", "anchor_f", "dipole", "quad",
             "q_res", "qO", "dchi", "internal", "bond", "corr_share", "r0")


def build_joint_model(config: Config, neighbor_types, reference_energies, atomic_states):
    """One featurizer, one response, two terms reading it."""
    features_cfg, ecfg, ocfg = config.features, config.elec, config.onebody
    if ocfg.bond_r_off > features_cfg.cutoff:
        raise ValueError(
            f"onebody.bond_r_off ({ocfg.bond_r_off}) exceeds the feature cutoff "
            f"({features_cfg.cutoff})"
        )
    featurizer = build_featurizer(features_cfg, ecfg, neighbor_types)
    response = build_response(
        featurizer, features_cfg, ecfg, neighbor_types, atomic_states
    )
    bond_head = PairEnergyHead(
        featurizer.feature_dims[0], len(neighbor_types),
        emb_dim=ocfg.emb_dim, hidden=ocfg.bond_hidden, depth=ocfg.bond_depth,
        n_radial=ocfg.bond_n_radial, r_on=ocfg.bond_r_on, r_off=ocfg.bond_r_off,
        energy_scale=ocfg.bond_energy_scale,
    )
    onebody = OneBodyEnergy(bond_head, reference_energies, response=response)
    elec = build_slater_elec(featurizer, response, features_cfg, ecfg, neighbor_types)
    return OneBodyElecModel(featurizer, onebody, elec)


def joint_fit(out, batch, cfg):
    """Both energy targets at once. ``cfg`` is the whole :class:`Config`.

    Unlike the single-term scripts this takes the full config rather than one block,
    because the two halves are normalized by their own ``energy_scale`` and then combined
    by ``joint.*`` weights.
    """
    ob_err = out.fragment_energy - batch.fragment_energy
    el_err = out.interaction_energy - batch.eda[cfg.elec.target]
    ob = (ob_err / cfg.onebody.energy_scale).pow(2).mean()
    el = (el_err / cfg.elec.energy_scale).pow(2).mean()
    loss = cfg.joint.onebody_weight * ob + cfg.joint.elec_weight * el

    elec_out = out.elec
    ff, corr = elec_out.energy_ff.detach().abs(), elec_out.energy_corr.detach().abs()
    metrics = {
        "ob_mae": float(ob_err.detach().abs().mean()) * KJMOL_PER_HARTREE,
        "el_mae": float(el_err.detach().abs().mean()) * KJMOL_PER_HARTREE,
        "corr_share": float((corr / (ff + corr).clamp(min=1e-30)).mean()),
        "internal": float(out.onebody.energy_internal.detach().mean()),
        "bond": float(out.onebody.energy_bond.detach().mean()),
    }
    return loss, metrics, batch.fragment_energy


class AnchorTerms:
    """The isolated-monomer anchor: energy, forces, and the frozen-monomer multipoles.

    One object rather than separate closures so the anchor's forward -- which includes a
    backward pass for the forces -- happens once per step and its metrics are reused by the
    logging callback. ``term_loop.run_epoch`` calls ``penalties`` then ``diagnostics`` on
    the same batch, so caching is enough.
    """

    def __init__(self, model, anchor_batch):
        self.model = model
        self.batch = anchor_batch
        self._metrics: dict[str, float] = {}

    def penalties(self, out, batch, cfg):
        # `cfg` is the whole Config here, not one block -- see the note on `fit(...)` below.
        c = cfg
        extra = {}
        if c.elec.corr_l2_weight > 0.0:
            extra["corr_l2"] = c.elec.corr_l2_weight * (
                out.elec.e_pair_corr / c.elec.energy_scale
            ).pow(2).mean()
        if c.elec.r0_weight > 0.0:
            extra["r0"] = c.elec.r0_weight * out.elec.r0
        if self.batch is None:
            self._metrics = {}
            return extra

        # One anchor forward, feeding both the energy/force terms and the multipole terms.
        # Grad is forced on because the force term is itself a backward pass and this runs
        # on evaluation epochs too; `onebody_anchor_loss` decides `create_graph` from the
        # loop's own grad state, so an eval step does not build a second-order graph.
        with torch.enable_grad():
            anchor_out = self.model(self.batch)
        w = c.joint.anchor_weight
        terms, self._metrics = onebody_anchor_loss(
            self.model, self.batch, c.onebody, out=anchor_out.onebody
        )
        extra.update({k: w * v for k, v in terms.items()})

        mm, mm_metrics = fragment_multipole_loss(
            anchor_out, self.batch,
            dipole_weight=c.elec.dipole_weight,
            quadrupole_weight=c.elec.quadrupole_weight,
            dipole_scale=c.elec.dipole_scale,
            quadrupole_scale=c.elec.quadrupole_scale,
        )
        extra.update({k: w * v for k, v in mm.items()})
        self._metrics.update(mm_metrics)
        return extra

    def diagnostics(self, out, batch, target):
        res = out.response
        frag = batch.fragment_idx
        n_frag = int(batch.n_fragments)
        q = res.charges.detach()
        per_frag = q.new_zeros(n_frag).index_add_(0, frag, q)
        want = (
            per_frag.new_zeros(n_frag) if batch.fragment_charge is None
            else batch.fragment_charge.to(per_frag.dtype)
        )
        z = batch.atomic_numbers
        metrics = {
            "q_res": float((per_frag - want).abs().max()),
            "r0": float(out.elec.r0.detach()),
            **self._metrics,
        }
        oxygen, hydrogen = z == 8, z == 1
        if bool(oxygen.any()):
            metrics["qO"] = float(q[oxygen].mean())
        if bool(oxygen.any()) and bool(hydrogen.any()):
            chi = res.chi.detach()
            metrics["dchi"] = float(chi[oxygen].mean() - chi[hydrogen].mean())
        return metrics


def train(config: Config):
    torch.set_default_dtype(torch.float64 if config.dtype == "float64" else torch.float32)
    device = resolve_device(config.device, config.dtype)
    print(
        f"[{config.run_name}] device={device} dtype={config.dtype} "
        f"targets=fragment_energy + eda_{config.elec.target}"
    )

    dataset = load_datasets(config.data.path, dtype=torch.get_default_dtype())
    if not dataset.has_fragments:
        raise ValueError("the joint model needs a `fragment_idx` column")
    if dataset._fragment_energy is None:
        raise ValueError(
            "the dataset carries no `fragment_energies` header, which is the 1-body "
            "target; use the files written by scripts/parse_roundtrip.py"
        )
    neighbor_types = dataset.unique_atomic_numbers
    reference_energies = load_reference_energies(
        config.data.reference_energies, neighbor_types
    ).to(torch.get_default_dtype())
    atomic_states = None
    if config.data.atomic_reference_states:
        atomic_states = AtomicStateReference.from_json(
            config.data.atomic_reference_states, neighbor_types,
            dtype=torch.get_default_dtype(),
        )
    print(
        f"loaded {len(dataset)} frames; species Z={neighbor_types}; "
        f"E0 = {dict(zip(neighbor_types, reference_energies.tolist()))}"
    )

    anchor = None
    if config.data.monomer_path:
        anchor = load_monomer_batch(
            config.data.monomer_path, dtype=torch.get_default_dtype()
        ).to(device)
        anchor.positions.requires_grad_(config.onebody.force_weight > 0.0)
        print(f"monomer anchor: {anchor.n_fragments} frames from {config.data.monomer_path}")
    elif config.onebody.force_weight > 0.0:
        raise ValueError(
            "onebody.force_weight > 0 but data.monomer_path is not set; a cluster frame's "
            "gradient is the sum over every force-field term"
        )

    model = build_joint_model(
        config, neighbor_types, reference_energies, atomic_states
    ).to(device)
    train_idx, val_idx = split_indices(
        len(dataset), config.data.holdout_fraction, config.data.seed
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"train/val = {len(train_idx)}/{len(val_idx)}; trainable params = {n_params}; "
        f"max_rank {config.elec.max_rank}; shared response = "
        f"{model.onebody.response is model.elec.response}; "
        f"weights onebody={config.joint.onebody_weight} elec={config.joint.elec_weight} "
        f"anchor={config.joint.anchor_weight}"
    )

    # `cfg` is the whole Config rather than one term block: this fit spans `onebody:`,
    # `elec:` and `joint:`, and the callbacks need all three.
    anchor_terms = AnchorTerms(model, anchor)
    fit(
        model, dataset, config, config, device, train_idx, val_idx,
        log_keys=_LOG_KEYS, fit_term=joint_fit,
        penalties=anchor_terms.penalties, diagnostics=anchor_terms.diagnostics,
    )
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Fit the 1-body term and classical electrostatics jointly, sharing "
                    "one per-fragment response solve."
    )
    parser.add_argument("config", type=str, help="path to YAML config")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
