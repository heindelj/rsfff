"""Evaluate a ``train_bonding`` checkpoint on an extxyz, per molecule and frame kind.

    python scripts/eval_bonding.py ckpt/<run>/best.pt data/stretch.xyz [--json out.json]
    python scripts/eval_bonding.py ckpt/<run>/best.pt data/train.xyz --split val
    python scripts/eval_bonding.py ckpt/<run>/best.pt data/probe.xyz --split val \\
        --vacuum data/train.xyz data/stretch.xyz

Per frame, where the file carries the label:

* ``E``: energy relative to the separated atoms (the model's reference energies), kcal/mol;
* ``F``: force MAE, kcal/mol/Ang;
* ``H``: relative Frobenius error of the full Cartesian Hessian (``3n`` backward passes);
* ``mu``: |dipole error| (Debye), ``Theta``: traceless-quadrupole MAE (e*bohr^2), ``alpha``:
  polarizability MAE (bohr^3) and relative error of its isotropic part -- all as derivatives of
  the energy (``FilmModel.response_properties``) about the center of nuclear charge, exactly as
  ``train_bonding`` compares them;
* probe frames (``ext_charges``) with ``--vacuum``: each is paired with the vacuum frame at the
  same geometry, and the **interaction energy** ``E(probes) - E(vacuum)`` and the force change
  are compared (kcal/mol, kcal/mol/Ang) -- the response itself, free of the vacuum error.

``--split val`` (or ``train``) evaluates only the trainer's split of that file (the checkpoint's
``data.holdout_fraction`` and ``data.seed``, which every stream shares). Frames are grouped by
the ``name`` and ``kind`` headers (fff_bonding exports).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from rsfff.ff.units import KJMOL_PER_HARTREE
from rsfff.train.build_film import build_film_model
from rsfff.train.data import load_datasets, load_reference_energies, split_indices
from rsfff.train.train_bonding import _labels_about_center

__all__ = ["load_model", "evaluate", "summarize"]

KCAL_PER_HARTREE = KJMOL_PER_HARTREE / 4.184
DEBYE_PER_AU = 2.541746
BOHR_ANG = 0.52917721067

COLUMNS = [("e_mae", "E"), ("f_mae", "F"), ("h_rel", "H rel"), ("mu", "mu D"),
           ("theta", "Theta"), ("alpha", "alpha"), ("alpha_iso_rel", "a_iso%"),
           ("de_int", "dE_int"), ("df_int", "dF_int")]


def load_model(ckpt_path, reference_energies=None):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    types = sorted(int(z) for z in ck["neighbor_types"])
    ref = load_reference_energies(reference_energies or cfg.data.reference_energies, types)
    torch.set_default_dtype(torch.float64)
    model = build_film_model(cfg.features, cfg.film, types, ref.double()).double()
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, ck


def _headers(path):
    from ase.io import iread

    return [(a.info.get("name", "?"), a.info.get("kind", "?")) for a in iread(str(path), index=":")]


def _vacuum_index(paths):
    """``{(name, rounded positions bytes): (dataset, frame)}`` over the vacuum files."""
    table = {}
    for p in paths:
        ds = load_datasets([p], dtype=torch.float64)
        for i, (name, _) in enumerate(_headers(p)):
            b = ds.flat_batch([i])
            table[(name, np.round(b.positions.numpy(), 6).tobytes())] = (ds, i)
    return table


def _energy_forces(model, b, *, hessian=False):
    b.positions.requires_grad_(True)
    e = model(b).energy.sum()
    (g,) = torch.autograd.grad(e, b.positions, create_graph=hessian)
    h = None
    if hessian:
        flat = g.reshape(-1)
        h = torch.stack([torch.autograd.grad(flat[k], b.positions, retain_graph=True)[0].reshape(-1)
                         for k in range(flat.numel())])
    return e.detach(), -g.detach(), h


def evaluate(model, path, *, idx=None, hessian=True, moments=True, vacuum=None):
    """``{(name, kind): {metric: [per-frame values]}}``."""
    ds = load_datasets([path], dtype=torch.float64)
    heads = _headers(path)
    vac = _vacuum_index(vacuum) if vacuum else None
    rows = defaultdict(lambda: defaultdict(list))
    frames = range(len(ds)) if idx is None else [int(i) for i in idx]
    for i in frames:
        key = heads[i]
        b = ds.flat_batch([i])
        want_h = hessian and b.hessian is not None
        e, f, h = _energy_forces(model, b, hessian=want_h)
        r = rows[key]
        r["e_mae"].append(abs(float(e - b.energy.sum())) * KCAL_PER_HARTREE)
        if b.forces is not None:
            r["f_mae"].append(float((f - b.forces).abs().mean()) * KCAL_PER_HARTREE)
        if h is not None:
            h_ref = b.hessian_blocks()[0]
            r["h_rel"].append(float((h - h_ref).norm() / h_ref.norm()))
        if vac is not None and b.external is not None:
            hit = vac.get((key[0], np.round(b.positions.detach().numpy(), 6).tobytes()))
            if hit is not None:
                vds, j = hit
                bv = vds.flat_batch([j])
                ev, fv, _ = _energy_forces(model, bv)
                de_model, de_ref = float(e - ev), float(b.energy.sum() - bv.energy.sum())
                r["de_int"].append(abs(de_model - de_ref) * KCAL_PER_HARTREE)
                if b.forces is not None and bv.forces is not None:
                    df = (f - fv) - (b.forces - bv.forces)
                    r["df_int"].append(float(df.abs().mean()) * KCAL_PER_HARTREE)
        if moments and (b.dipole is not None or b.polarizability is not None):
            center, dip, theta = _labels_about_center(b)
            _, mu, th, al = model.response_properties(
                b, polarizability=b.polarizability is not None, quadrupole=theta is not None,
                create_graph=False, origin=center,
            )
            if dip is not None:
                r["mu"].append(float((mu.detach() - dip).norm()) * DEBYE_PER_AU)
            if theta is not None:
                r["theta"].append(float((th.detach() - theta).abs().mean()))
            if b.polarizability is not None:
                a_ref = b.polarizability / BOHR_ANG ** 2           # bohr^3
                a = al.detach()
                r["alpha"].append(float((a - a_ref).abs().mean()))
                iso, iso_ref = float(a.diagonal(dim1=-2, dim2=-1).mean()), float(
                    a_ref.diagonal(dim1=-2, dim2=-1).mean())
                r["alpha_iso_rel"].append(100.0 * abs(iso - iso_ref) / abs(iso_ref))
    return rows


def summarize(rows):
    """Per-group means plus an ``all`` row (frame-weighted over every group)."""
    out, pooled = {}, defaultdict(list)
    for (name, kind), r in sorted(rows.items()):
        s = {"n": max(len(v) for v in r.values())}
        for k, v in r.items():
            s[k] = float(np.mean(v))
            pooled[k] += v
        out[f"{name}/{kind}"] = s
    out["all"] = {"n": max((len(v) for v in pooled.values()), default=0),
                  **{k: float(np.mean(v)) for k, v in pooled.items()}}
    return out


def _print(summary):
    cols = [(k, t) for k, t in COLUMNS if any(k in s for s in summary.values())]
    print(f"{'molecule/kind':30s} {'n':>4s} " + " ".join(f"{t:>7s}" for _, t in cols))
    for name, s in summary.items():
        vals = " ".join(f"{s[k]:7.3f}" if k in s else "      -" for k, _ in cols)
        print(f"{name:30s} {s['n']:4d} {vals}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("data", type=Path)
    ap.add_argument("--ref", type=Path, default=None, help="reference energies json")
    ap.add_argument("--split", choices=("all", "train", "val"), default="all")
    ap.add_argument("--vacuum", type=Path, nargs="*", default=None,
                    help="vacuum files to pair probe frames with (interaction energies)")
    ap.add_argument("--no-hessian", action="store_true")
    ap.add_argument("--no-moments", action="store_true")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    model, ck = load_model(a.checkpoint, a.ref)
    idx = None
    if a.split != "all":
        n = len(_headers(a.data))
        tr, va = split_indices(n, ck["config"].data.holdout_fraction, ck["config"].data.seed)
        idx = (va if a.split == "val" else tr).tolist()
    rows = evaluate(model, a.data, idx=idx, hessian=not a.no_hessian,
                    moments=not a.no_moments, vacuum=a.vacuum)
    summary = summarize(rows)
    _print(summary)
    if a.json:
        a.json.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
