"""Fit the film model on monomers: the bonded potential, the multipoles and the response.

    python -m rsfff.train.train_bonding configs/bonding_monomer.yaml

The trainer of the monomer-bonded study (``docs/fff_bonding.md``). No clusters and no EDA:
every frame is one molecule, every bonded term is enumerated from the explicit covalent graph
(``bonds`` in the extxyz), and every response label is compared against an **energy
derivative** of the model (:meth:`rsfff.ff.film.FilmModel.response_properties`):

=================  ==========================================================================
stream             supervises
=================  ==========================================================================
``data.path``      vacuum frames: energy per atom, forces, dipole, traceless quadrupole and
                   polarizability -- each where the file carries the label
``bonding.probe``  frames with ``ext_charges`` (and/or ``ext_field``): the same labels, in the
                   field. Its own minibatch every step, weighted by ``bonding.probe_weight``
=================  ==========================================================================

Moments are compared about each molecule's **center of nuclear charge** (Q-Chem's origin):
the model's by placing the uniform-field origin there, the labels by shifting the stored
about-the-origin moments (:func:`rsfff.ff.molecular_multipoles.shift_multipoles`), so a
convention error cancels rather than biases.

The four model variants of the study are config flags on the same model: ``film.couplings``
(bond-bond, bond-angle, angle-angle, torsion couplings) and ``film.field_features`` (the O(F^2)
electrostatic-environment dependence of the bonded parameters).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

from ..ff.molecular_multipoles import (
    buckingham_from_second_moment,
    center_of_nuclear_charge,
    shift_multipoles,
)
from ..ff.units import BOHR_ANG, KJMOL_PER_HARTREE
from .build_film import build_film_model
from .config import load_config
from .data import load_datasets, load_reference_energies, split_indices
from .loss import compute_forces
from .term_loop import fit
from .train_eem import resolve_device

__all__ = ["BondingWeights", "monomer_loss", "train"]

_LOG_KEYS = ["e_mae", "f_mae", "mu_mae", "theta_mae", "alpha_mae",
             "pr_e_mae", "pr_f_mae", "pr_mu_mae", "cg_fail"]


@dataclass
class BondingWeights:
    energy_weight: float = 1.0
    force_weight: float = 1.0
    dipole_weight: float = 1.0
    quadrupole_weight: float = 1.0
    polarizability_weight: float = 1.0
    energy_scale: float = 3.8093e-4      # 1 kJ/mol in Hartree, per atom
    force_scale: float = 1.0e-3          # Ha/Angstrom
    dipole_scale: float = 0.05           # e*bohr
    quadrupole_scale: float = 0.2        # e*bohr^2
    polarizability_scale: float = 0.5    # bohr^3
    probe_weight: float = 1.0
    probe_batch_size: int = 32
    probe_path: list | None = None

    @classmethod
    def from_yaml(cls, path) -> "BondingWeights":
        raw = (yaml.safe_load(Path(path).read_text()) or {}).get("bonding", {}) or {}
        kw = {k: raw[k] for k in cls.__dataclass_fields__ if k in raw}
        if isinstance(kw.get("probe_path"), str):
            kw["probe_path"] = [kw["probe_path"]]
        return cls(**kw)


def _labels_about_center(batch):
    """``(center (B,3) Ang, dipole (B,3) e*bohr | None, theta (B,3,3) e*bohr^2 | None)``."""
    n_sys = int(batch.n_systems)
    center = center_of_nuclear_charge(
        batch.positions.detach(), batch.atomic_numbers, batch.batch_idx, n_sys
    )
    charge = (
        batch.total_charge if batch.total_charge is not None
        else batch.positions.new_zeros(n_sys)
    )
    dip = None if batch.dipole is None else batch.dipole / BOHR_ANG     # stored e*Ang
    m2 = None
    if batch.fragment_second_moment is not None:
        m2 = batch.fragment_second_moment.new_zeros(n_sys, 3, 3).index_add_(
            0, batch.fragment_to_batch, batch.fragment_second_moment
        )
        d0 = dip if dip is not None else batch.fragment_dipole.new_zeros(n_sys, 3).index_add_(
            0, batch.fragment_to_batch, batch.fragment_dipole
        )
        d_c, m2_c = shift_multipoles(d0, m2, charge, center / BOHR_ANG)
        m2 = buckingham_from_second_moment(m2_c)
        if dip is not None:
            dip = d_c
    elif dip is not None:
        dip = dip - charge[:, None] * center / BOHR_ANG
    return center, dip, m2


def monomer_loss(model, out, batch, w: BondingWeights, *, training: bool, prefix: str = ""):
    """Energy/forces from ``out``; the moments by derivative. ``(loss, metrics)``."""
    metrics = {}
    loss = out.energy.new_zeros(())
    n_atoms = torch.bincount(batch.batch_idx, minlength=int(batch.n_systems)).clamp(min=1)
    e_err = (out.energy - batch.energy) / n_atoms.to(out.energy.dtype)
    metrics[f"{prefix}e_mae"] = float(e_err.detach().abs().mean()) * KJMOL_PER_HARTREE
    loss = loss + w.energy_weight * (e_err / w.energy_scale).pow(2).mean()
    if w.force_weight > 0.0 and batch.forces is not None and batch.positions.requires_grad:
        forces = compute_forces(out.energy, batch.positions, create_graph=training)
        f_err = forces - batch.forces
        loss = loss + w.force_weight * (f_err / w.force_scale).pow(2).sum(-1).mean()
        metrics[f"{prefix}f_mae"] = float(f_err.detach().abs().mean())

    center, dip, theta = _labels_about_center(batch)
    want_mu = w.dipole_weight > 0.0 and dip is not None
    want_th = w.quadrupole_weight > 0.0 and theta is not None
    want_al = w.polarizability_weight > 0.0 and batch.polarizability is not None
    if want_mu or want_th or want_al:
        ext = getattr(batch, "external", None)
        has_uniform = ext is not None and (ext.field is not None or ext.field_gradient is not None)
        _, mu, th, al = model.response_properties(
            batch, polarizability=want_al, quadrupole=want_th, create_graph=training,
            origin=None if has_uniform else center,
        )
        if want_mu:
            err = mu - dip
            loss = loss + w.dipole_weight * (err / w.dipole_scale).pow(2).sum(-1).mean()
            metrics[f"{prefix}mu_mae"] = float(err.detach().abs().mean())
        if want_th:
            err = th - theta
            loss = loss + w.quadrupole_weight * (err / w.quadrupole_scale).pow(2).sum((-1, -2)).mean()
            metrics[f"{prefix}theta_mae"] = float(err.detach().abs().mean())
        if want_al:
            err = al - batch.polarizability / BOHR_ANG ** 2          # stored e^2 Ang^2 / Ha
            loss = loss + w.polarizability_weight * (err / w.polarizability_scale).pow(2).sum((-1, -2)).mean()
            metrics[f"{prefix}alpha_mae"] = float(err.detach().abs().mean())
    if out.solver:
        _n, converged, pd_fail = out.solver["ind"]
        metrics["cg_fail"] = float((~converged).sum() + pd_fail.sum())
    return loss, metrics


class ProbeStream:
    """The probe-charge frames as their own minibatch each training step."""

    def __init__(self, model, dataset, train_idx, w: BondingWeights, device, seed: int = 0):
        self.model, self.dataset, self.idx, self.w, self.device = model, dataset, train_idx, w, device
        self.g = torch.Generator().manual_seed(int(seed) + 17)

    def __call__(self, out, batch, cfg):
        self.last = {}
        if self.dataset is None or not self.model.training or self.w.probe_weight <= 0.0:
            return {}
        pick = self.idx[torch.randperm(self.idx.numel(), generator=self.g)[: self.w.probe_batch_size]]
        pb = self.dataset.flat_batch(pick.tolist()).to(self.device)
        pb.positions.requires_grad_(True)
        with torch.enable_grad():
            pout = self.model(pb, with_induction=True)
            loss, metrics = monomer_loss(self.model, pout, pb, self.w, training=True, prefix="pr_")
        self.last = metrics
        return {"probe": self.w.probe_weight * loss}


def train(config_path):
    config = load_config(config_path)
    w = BondingWeights.from_yaml(config_path)
    device = resolve_device(config.device, config.dtype)
    dtype = torch.float64 if str(config.dtype) == "float64" else torch.float32
    torch.set_default_dtype(dtype)

    vacuum = load_datasets(config.data.path, dtype=dtype)
    probes = load_datasets(w.probe_path, dtype=dtype) if w.probe_path else None
    types = sorted(set(vacuum.unique_atomic_numbers) | set(
        probes.unique_atomic_numbers if probes is not None else []))
    train_idx, val_idx = split_indices(len(vacuum), config.data.holdout_fraction, config.data.seed)
    probe_train = (
        split_indices(len(probes), config.data.holdout_fraction, config.data.seed)[0]
        if probes is not None else None
    )
    ref = load_reference_energies(config.data.reference_energies, types).to(dtype)
    torch.manual_seed(config.train.seed)
    model = build_film_model(config.features, config.film, types, ref).to(device=device, dtype=dtype)
    print(
        f"{len(vacuum)} vacuum frames ({len(train_idx)}/{len(val_idx)} train/val); "
        f"{0 if probes is None else len(probes)} probe frames; species {types}; "
        f"{sum(p.numel() for p in model.parameters())} parameters; "
        f"couplings={config.film.couplings} field_features={config.film.field_features}",
        flush=True,
    )
    stream = ProbeStream(model, probes, probe_train, w, device, config.data.seed)

    def fit_term(out, batch, cfg):
        return (*monomer_loss(model, out, batch, w, training=model.training), batch.energy)

    def diagnostics(out, batch, target):
        return dict(getattr(stream, "last", {}))

    return fit(
        model, vacuum, config, config, device, train_idx, val_idx,
        log_keys=_LOG_KEYS, fit_term=fit_term, penalties=stream, diagnostics=diagnostics,
        grad_positions=w.force_weight > 0.0,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", type=Path)
    train(ap.parse_args().config)


if __name__ == "__main__":
    main()
