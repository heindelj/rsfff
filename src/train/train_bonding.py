"""Fit the film model on monomers: the bonded potential, the multipoles and the response.

    python -m rsfff.train.train_bonding configs/bonding_monomer.yaml

The trainer of the monomer-bonded study (``docs/fff_bonding.md``). No clusters and no EDA:
every frame is one molecule, every bonded term is enumerated from the explicit covalent graph
(``bonds`` in the extxyz), and every response label is compared against an **energy
derivative** of the model (:meth:`rsfff.ff.film.FilmModel.response_properties`):

====================  =======================================================================
stream                supervises
====================  =======================================================================
``data.path``         the main frames (train/val split, epochs, checkpoints): energy per atom,
                      forces, dipole, traceless quadrupole, polarizability -- each where the
                      file carries the label
``bonding.streams``   further files, each its own minibatch every step with its own weight:
                      frames whose label set differs from the main file's (rsfff's loader
                      wants a label on every frame of a file or on none), e.g. the vacuum
                      frames with a polarizability, the probe-charge frames (``ext_charges``)
                      and the uniform-field frames (``ext_field``). Each is split with the
                      same holdout fraction; its held-out part enters the validation loss,
                      so the best checkpoint is chosen on every stream
====================  =======================================================================

A stream is ``{name, path, weight, batch_size}``. The older ``bonding.probe_path`` (with
``probe_weight`` / ``probe_batch_size``) is read as one stream named ``probe``.

The ChemLab project ``fff_bonding/cl`` exports a set as four files
(``python/scripts/export_dataset.py``): ``vacuum.xyz`` (the main file), ``vacuum_pol.xyz``,
``probe.xyz`` and ``field.xyz`` (the streams ``pol``, ``probe``, ``field``).

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

__all__ = ["BondingWeights", "Stream", "Streams", "monomer_loss", "train"]

_METRICS = ["e_mae", "f_mae", "h_rel", "mu_mae", "theta_mae", "alpha_mae", "bond_var"]
_LOG_KEYS = ["loss", *_METRICS, "cg_fail"]


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
    #: Hessian-vector products against the ``hessian`` labels: ``hessian_probes`` random
    #: Gaussian directions per step, ``-d(F.v)/dx = H v`` by one more backward pass, error in
    #: units of ``hessian_scale`` (Ha/Ang^2 per unit-variance probe component). 0 = off.
    hessian_weight: float = 0.0
    hessian_scale: float = 0.02          # Ha/Ang^2
    hessian_probes: int = 4
    #: the geometry-independence regularizer of rsfff.train.train_film (``bond_var``): mean
    #: squared feature-dependent deviation of the bonded/term parameters from their tables.
    #: Matters most with ``film.bond_features: ece`` and ``ece_own_bond``. 0 = off.
    bond_var_weight: float = 0.0
    probe_weight: float = 1.0
    probe_batch_size: int = 32
    probe_path: list | None = None
    streams: list | None = None

    @classmethod
    def from_yaml(cls, path) -> "BondingWeights":
        raw = (yaml.safe_load(Path(path).read_text()) or {}).get("bonding", {}) or {}
        kw = {k: raw[k] for k in cls.__dataclass_fields__ if k in raw}
        if isinstance(kw.get("probe_path"), str):
            kw["probe_path"] = [kw["probe_path"]]
        return cls(**kw)

    def stream_specs(self) -> list[dict]:
        """``[{name, path: [..], weight, batch_size}]``: ``streams`` plus the legacy probe path."""
        specs = []
        for i, st in enumerate(self.streams or []):
            if not st.get("path"):
                raise ValueError(f"bonding.streams[{i}] has no path")
            path = st["path"] if isinstance(st["path"], list) else [st["path"]]
            specs.append({"name": str(st.get("name", f"s{i}")), "path": path,
                          "weight": float(st.get("weight", 1.0)),
                          "batch_size": int(st.get("batch_size", 32))})
        if self.probe_path:
            specs.append({"name": "probe", "path": list(self.probe_path),
                          "weight": float(self.probe_weight),
                          "batch_size": int(self.probe_batch_size)})
        names = [sp["name"] for sp in specs]
        if len(set(names)) != len(names):
            raise ValueError(f"bonding.streams: duplicate names {names}")
        return specs


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
    want_h = (w.hessian_weight > 0.0 and getattr(batch, "hessian", None) is not None
              and batch.positions.requires_grad)
    if (w.force_weight > 0.0 or want_h) and batch.positions.requires_grad and (
        batch.forces is not None or want_h
    ):
        forces = compute_forces(out.energy, batch.positions, create_graph=training or want_h)
        if w.force_weight > 0.0 and batch.forces is not None:
            f_err = forces - batch.forces
            loss = loss + w.force_weight * (f_err / w.force_scale).pow(2).sum(-1).mean()
            metrics[f"{prefix}f_mae"] = float(f_err.detach().abs().mean())
        if want_h:
            h_loss, h_rel = hessian_vector_loss(forces, batch, w, training=training)
            loss = loss + w.hessian_weight * h_loss
            metrics[f"{prefix}h_rel"] = h_rel
    if w.bond_var_weight > 0.0:
        from .train_film import _bonded_variance

        var = _bonded_variance(out)
        loss = loss + w.bond_var_weight * var
        metrics[f"{prefix}bond_var"] = float(var.detach())

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


def hessian_vector_loss(forces, batch, w: BondingWeights, *, training: bool):
    """``(loss, relative error)`` of ``H v`` for ``w.hessian_probes`` Gaussian probes ``v``.

    ``forces`` must carry a graph to the positions. The model's ``H v = -d(F . v)/dx`` costs
    one backward pass per probe (plus one more for the parameter gradient when training),
    against ``3 n`` passes for the full Hessian. Evaluation draws its probes from a fixed
    seed, so validation numbers are comparable between epochs.
    """
    pos = batch.positions
    blocks = batch.hessian_blocks()
    n = torch.bincount(batch.batch_idx, minlength=int(batch.n_systems)).tolist()
    gen = None if training else torch.Generator(device="cpu").manual_seed(1234)
    loss = forces.new_zeros(())
    num = den = 0.0
    k = max(int(w.hessian_probes), 1)
    for p in range(k):
        v = torch.randn(pos.shape, generator=gen, dtype=pos.dtype).to(pos.device)
        (g,) = torch.autograd.grad((forces * v).sum(), pos, create_graph=training,
                                   retain_graph=True)
        hv_model = -g
        hv_label = torch.cat([h @ vf.reshape(-1) for h, vf in
                              zip(blocks, torch.split(v, n))]).reshape(-1, 3)
        err = hv_model - hv_label
        loss = loss + (err / w.hessian_scale).pow(2).sum(-1).mean() / k
        num += float(err.detach().pow(2).sum())
        den += float(hv_label.pow(2).sum())
    return loss, (num / max(den, 1e-300)) ** 0.5


class Stream:
    """One extra file as its own minibatch each step.

    Training: a random minibatch of the stream's training frames, its loss times ``weight``.
    Evaluation: the next ``batch_size`` of its held-out frames (in order, wrapping), so a
    validation pass over the main file also sweeps the stream's held-out part.
    """

    def __init__(self, name, model, dataset, train_idx, val_idx, w: BondingWeights, weight,
                 batch_size, device, seed: int = 0):
        self.name, self.model, self.dataset, self.w = name, model, dataset, w
        self.train_idx, self.val_idx = train_idx, val_idx
        self.weight, self.batch_size, self.device = float(weight), int(batch_size), device
        self.g = torch.Generator().manual_seed(int(seed) + 17)
        self._next = 0
        self.last: dict = {}

    def _pick(self, training: bool):
        if training:
            idx = self.train_idx
            return idx[torch.randperm(idx.numel(), generator=self.g)[: self.batch_size]]
        idx = self.val_idx
        if idx.numel() == 0:
            return None
        rows = (torch.arange(self.batch_size) + self._next) % idx.numel()
        self._next = int((self._next + self.batch_size) % idx.numel())
        return idx[rows[: min(self.batch_size, idx.numel())]]

    def __call__(self):
        self.last = {}
        training = self.model.training
        if self.weight <= 0.0:
            return None
        pick = self._pick(training)
        if pick is None or pick.numel() == 0:
            return None
        pb = self.dataset.flat_batch(pick.tolist()).to(self.device)
        pb.positions.requires_grad_(True)
        with torch.enable_grad():
            pout = self.model(pb, with_induction=True)
            loss, metrics = monomer_loss(self.model, pout, pb, self.w, training=training,
                                         prefix=f"{self.name}_")
        metrics.pop("cg_fail", None)
        self.last = metrics
        return self.weight * (loss if training else loss.detach())


class Streams:
    """The ``penalties`` / ``diagnostics`` hooks of :func:`rsfff.train.term_loop.fit`."""

    def __init__(self, streams):
        self.streams = streams

    def __call__(self, out, batch, cfg):
        extra = {}
        for st in self.streams:
            value = st()
            if value is not None:
                extra[st.name] = value
        return extra

    def diagnostics(self, out, batch, target):
        merged = {}
        for st in self.streams:
            merged.update(st.last)
        return merged

    def log_keys(self):
        keys = []
        for st in self.streams:
            keys += [st.name] + [f"{st.name}_{k}" for k in _METRICS]
        return keys


def train(config_path):
    config = load_config(config_path)
    w = BondingWeights.from_yaml(config_path)
    device = resolve_device(config.device, config.dtype)
    dtype = torch.float64 if str(config.dtype) == "float64" else torch.float32
    torch.set_default_dtype(dtype)

    main_set = load_datasets(config.data.path, dtype=dtype)
    specs = w.stream_specs()
    extra_sets = [load_datasets(sp["path"], dtype=dtype) for sp in specs]
    types = sorted(set(main_set.unique_atomic_numbers).union(
        *[set(d.unique_atomic_numbers) for d in extra_sets]))
    train_idx, val_idx = split_indices(len(main_set), config.data.holdout_fraction, config.data.seed)
    ref = load_reference_energies(config.data.reference_energies, types).to(dtype)
    torch.manual_seed(config.train.seed)
    model = build_film_model(config.features, config.film, types, ref).to(device=device, dtype=dtype)

    streams = []
    for sp, ds in zip(specs, extra_sets):
        tr, va = split_indices(len(ds), config.data.holdout_fraction, config.data.seed)
        streams.append(Stream(sp["name"], model, ds, tr, va, w, sp["weight"], sp["batch_size"],
                              device, config.data.seed))
    hooks = Streams(streams)
    desc = "; ".join(f"{st.name} {len(st.dataset)} ({len(st.train_idx)}/{len(st.val_idx)})"
                     for st in streams) or "no streams"
    print(
        f"{len(main_set)} main frames ({len(train_idx)}/{len(val_idx)} train/val); {desc}; "
        f"species {types}; {sum(p.numel() for p in model.parameters())} parameters; "
        f"couplings={config.film.couplings} field_features={config.film.field_features}",
        flush=True,
    )

    def fit_term(out, batch, cfg):
        return (*monomer_loss(model, out, batch, w, training=model.training), batch.energy)

    return fit(
        model, main_set, config, config, device, train_idx, val_idx,
        log_keys=_LOG_KEYS + hooks.log_keys(), fit_term=fit_term, penalties=hooks,
        diagnostics=hooks.diagnostics, grad_positions=w.force_weight > 0.0,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", type=Path)
    train(ap.parse_args().config)


if __name__ == "__main__":
    main()
