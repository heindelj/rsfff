"""Train a bonded model (:class:`rsfff.ff.bonded.BondedModel`) on energies, forces and, where the frames carry them,
dipoles, second moments and polarizabilities, from one config file.

    python -m rsfff.train.bonded CONFIG.toml [--set train.epochs=50 encoder.latent=4 ...]

The model tables ``[terms]``, ``[tables]``, ``[encoder]`` are :class:`rsfff.ff.bonded.BondedConfig`. The trainer's:

    [data]    train (extxyz files; one molecule type per file is assumed for the per-file report), val_fraction,
              reference (optimized structures with ``frequencies=``: a check, never fitted), root (default: the
              working directory; relative paths, tables.init included, are taken from it)
    [loss]    energy, forces, dipole, quadrupole, polarizability (weights); the residual scales sigma_energy
              [kcal/mol], sigma_force [kcal/mol/A], sigma_dipole [e bohr], sigma_quadrupole [e bohr^2],
              sigma_polarizability [bohr^3]
    [train]   out, epochs, batch, lr, table_lr (learning rate of the bonded tables and reference energies; the
              electrical tables, which have no fit to start from, use lr), table_prefit (first fit the electrical
              tables alone to the training frames' electrical labels, by L-BFGS, before anything else moves: without
              it the first steps' electrical residuals are thousands of sigma and wreck the energy surface), seed,
              log_every, constant_only (train the tables alone: the constant-parameter model)

Loss: sum over quantities of weight mean((residual / sigma)^2); energies per molecule; an electrical quantity counts
the molecules of the batch that carry its label (frames without one are skipped for it) and is used only when the
model has an electrical head. Moments are about the files' coordinate origin.
The atomic reference energies start from a least-squares fit on the training frames. The edge features are
standardized on the training frames. Outputs in ``train.out``: model.pt (best validation loss; config, atom types,
state), metrics.json (per file: validation errors of the trained and of the tables-alone model, harmonic
frequencies and geometry at the model's minimum against the reference), log.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from ..ff.bonded import BondedConfig, BondedModel, atom_types_of, collate, read_extxyz
from ..ff.units import BOHR_ANG

KCAL = 627.509474
CM = 219474.6313
AMU = 1822.888486
MASS = {1: 1.00782503, 6: 12.0, 7: 14.0030740, 8: 15.99491462, 9: 18.99840316, 15: 30.97376200, 16: 31.97207117,
        17: 34.96885268}
DEFAULTS = {
    "data": {"train": [], "val_fraction": 0.2, "reference": [], "root": "."},
    "loss": {"energy": 1.0, "forces": 1.0, "dipole": 1.0, "quadrupole": 1.0, "polarizability": 1.0,
             "sigma_energy": 0.1, "sigma_force": 1.0, "sigma_dipole": 0.005, "sigma_quadrupole": 0.02,
             "sigma_polarizability": 0.05},
    "train": {"out": "runs/bonded", "epochs": 300, "batch": 64, "lr": 2e-3, "table_lr": 2e-4, "table_prefit": True, "seed": 0,
              "log_every": 10, "constant_only": False},
}


def _set(d: dict, dotted: str, value: str) -> None:
    *path, key = dotted.split(".")
    for p in path:
        d = d.setdefault(p, {})
    try:
        d[key] = json.loads(value)
    except json.JSONDecodeError:
        d[key] = value


def load_config(path, overrides=()):
    import tomllib
    raw = tomllib.loads(Path(path).read_text())
    for item in overrides:
        k, v = item.split("=", 1)
        _set(raw, k, v)
    cfg = BondedConfig.from_dict(raw)
    run = {}
    for table, default in DEFAULTS.items():
        given = cfg.extra.get(table, {})
        unknown = [k for k in given if k not in default]
        if unknown:
            raise ValueError(f"[{table}]: unknown keys {unknown}; known: {list(default)}")
        run[table] = {**default, **given}
    return cfg, run


ELECTRICAL = (("dipole", "dipole"), ("quadrupole", "second_moment"), ("polarizability", "polarizability"))


def errors(model, batches, constant=False):
    """{file: {n, e_mae [kcal/mol], e_max, f_rmse [kcal/mol/A], and per labelled electrical quantity its
    component rmse in atomic units (dipole_rmse, quadrupole_rmse, polarizability_rmse)}} over a list of batches."""
    acc, el = {}, {}
    for b in batches:
        e, f = model.energy_forces(b, constant=constant)
        if model.has_electrical:
            with torch.no_grad():
                sol = model.electrical(b, constant=constant)
            for label, key in ELECTRICAL:
                have = b.labelled(key)
                if bool(have.any()):
                    d2 = ((getattr(sol, key) - getattr(b, key)) ** 2).reshape(b.n_mol, -1)
                    for m in torch.nonzero(have).flatten().tolist():
                        a = el.setdefault((b.names[m], label), [0.0, 0])
                        a[0] += float(d2[m].sum())
                        a[1] += d2.shape[1]
        de, df2 = (e.detach() - b.energy).abs() * KCAL, ((f.detach() - b.forces) ** 2).sum(-1) * (KCAL / BOHR_ANG) ** 2
        for m, name in enumerate(b.names):
            a = acc.setdefault(name, [0, 0.0, 0.0, 0.0, 0])
            sel = b.mol_idx == m
            a[0] += 1
            a[1] += float(de[m])
            a[2] = max(a[2], float(de[m]))
            a[3] += float(df2[sel].sum())
            a[4] += 3 * int(sel.sum())
    out = {k: {"n": a[0], "e_mae": a[1] / a[0], "e_max": a[2], "f_rmse": math.sqrt(a[3] / a[4])} for k, a in acc.items()}
    for (name, label), a in el.items():
        out[name][f"{label}_rmse"] = math.sqrt(a[0] / a[1])
    return out


def minimum(model, mol, constant=False):
    """The model's minimum from ``mol``'s geometry: bond lengths [A], harmonic frequencies [cm^-1]."""
    b = collate([mol])
    x = b.positions.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([x], max_iter=500, tolerance_grad=1e-10, tolerance_change=1e-14, line_search_fn="strong_wolfe")
    energy = lambda q: model(b, q, constant)["energy"][0]

    def closure():
        opt.zero_grad()
        v = energy(x)
        v.backward()
        return v

    opt.step(closure)
    xe, n = x.detach(), len(mol.z)
    h = torch.autograd.functional.hessian(lambda q: energy(q.view(n, 3)), xe.reshape(-1)) * BOHR_ANG ** 2
    m = torch.tensor([MASS[int(z)] for z in mol.z]).repeat_interleave(3) * AMU
    w2 = torch.linalg.eigvalsh(h / torch.sqrt(m[:, None] * m[None, :]))
    n_vib = 3 * n - (5 if n == 2 else 6)
    i, j = b.topo.bond_index
    return {"bond_lengths": (xe[i] - xe[j]).norm(dim=-1).tolist(),
            "frequencies": (w2[-n_vib:].clamp(min=0).sqrt() * CM).tolist()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--set", nargs="*", default=[], metavar="table.key=value")
    args = ap.parse_args(argv)
    cfg, run = load_config(args.config, args.set)
    D, L, T = run["data"], run["loss"], run["train"]
    root = Path(D["root"])
    torch.manual_seed(T["seed"])
    rng = np.random.default_rng(T["seed"])

    train, val = [], []
    for f in D["train"]:
        mols = read_extxyz(root / f)
        perm = rng.permutation(len(mols))
        n_val = int(round(D["val_fraction"] * len(mols)))
        val += [mols[i] for i in perm[:n_val]]
        train += [mols[i] for i in perm[n_val:]]
    if not train:
        raise ValueError("data.train lists no frames")
    everything = collate(train)
    model = BondedModel(cfg, atom_types_of(collate(train + val)), table_root=str(root))
    model.set_statistics(everything)
    model.fit_reference(everything)
    val_batches = [collate(val[i:i + 256]) for i in range(0, len(val), 256)]

    fitted = [model.tables.bond, model.tables.apex, model.tables.e0]            # start from the constant fit
    unfitted = [model.tables.atom, model.tables.link]                           # electrical: start from defaults
    rest = [] if T["constant_only"] else [p for n, p in model.named_parameters() if not n.startswith("tables.")]
    opt = torch.optim.Adam([{"params": rest + unfitted, "lr": T["lr"]}, {"params": fitted, "lr": T["table_lr"]}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(int(T["epochs"]), 1))
    s_e, s_f = L["sigma_energy"] / KCAL, L["sigma_force"] / KCAL * BOHR_ANG
    const = bool(T["constant_only"])

    sig = {"dipole": L["sigma_dipole"], "quadrupole": L["sigma_quadrupole"], "polarizability": L["sigma_polarizability"]}

    def loss_of(b, create_graph):
        pos = b.positions.detach().clone().requires_grad_(True)
        out = model(b, pos, const)
        e = out["energy"]
        (g,) = torch.autograd.grad(e.sum(), pos, create_graph=create_graph)
        f = -g * BOHR_ANG
        loss = L["energy"] * (((e - b.energy) / s_e) ** 2).mean() + L["forces"] * (((f - b.forces) / s_f) ** 2).mean()
        if model.has_electrical:
            for label, key in ELECTRICAL:
                have = b.labelled(key)
                if L[label] > 0 and bool(have.any()):
                    loss = loss + L[label] * (((out[key][have] - getattr(b, key)[have]) / sig[label]) ** 2).mean()
        return loss

    def electrical_loss(b):
        out, loss = model(b, constant=True), None
        for label, key in ELECTRICAL:
            have = b.labelled(key)
            if L[label] > 0 and bool(have.any()):
                term = L[label] * (((out[key][have] - getattr(b, key)[have]) / sig[label]) ** 2).mean()
                loss = term if loss is None else loss + term
        return loss

    def val_loss():
        tot = sum(float(loss_of(b, False).detach()) * b.n_mol for b in val_batches)
        return tot / max(len(val), 1)

    out = Path(T["out"])
    out.mkdir(parents=True, exist_ok=True)
    n_par = sum(p.numel() for g in opt.param_groups for p in g["params"] if p.requires_grad)
    print(f"{len(train)} training and {len(val)} validation frames; atom types {model.atom_types}; "
          f"{n_par} trained parameters; tables from file: {model.tables.from_file}")
    best, log, t0 = math.inf, [], time.time()
    order = np.arange(len(train))
    labelled = [m for m in train if m.dipole is not None or m.second_moment is not None or m.polarizability is not None]
    if model.has_electrical and T["table_prefit"] and labelled:
        lab = collate(labelled[:2000])
        pre = torch.optim.LBFGS(unfitted, max_iter=300, line_search_fn="strong_wolfe", tolerance_grad=1e-9, tolerance_change=1e-12)

        def closure():
            pre.zero_grad()
            loss = electrical_loss(lab)
            loss.backward()
            return loss

        first = float(closure().detach())
        pre.step(closure)
        print(f"electrical tables alone on {lab.n_mol} labelled frames: loss {first:.1f} -> {float(electrical_loss(lab).detach()):.1f}"
              f"  {time.time() - t0:.0f} s")
    for epoch in range(1, int(T["epochs"]) + 1):
        rng.shuffle(order)
        tot = 0.0
        for i in range(0, len(order), int(T["batch"])):
            b = collate([train[j] for j in order[i:i + int(T["batch"])]])
            opt.zero_grad()
            loss = loss_of(b, True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            tot += float(loss.detach()) * b.n_mol
        sched.step()
        v = val_loss() if val else tot / len(train)
        log.append((epoch, tot / len(train), v))
        if v < best:
            best = v
            torch.save({"config": cfg.to_dict(), "atom_types": model.atom_types, "state": model.state_dict()}, out / "model.pt")
        if epoch % int(T["log_every"]) == 0 or epoch == 1:
            print(f"epoch {epoch:4d}  train {tot / len(train):9.4f}  val {v:9.4f}  best {best:9.4f}  {time.time() - t0:6.0f} s")
    if int(T["epochs"]) > 0:
        model.load_state_dict(torch.load(out / "model.pt")["state"])
    with open(out / "log.csv", "w", newline="") as fh:
        csv.writer(fh).writerows([("epoch", "train", "val"), *log])

    report = {"validation": errors(model, val_batches, const), "constant": errors(model, val_batches, True), "minima": {}}
    for f in D["reference"]:
        mol = read_extxyz(root / f)[0]
        mn = minimum(model, mol, const)
        report["minima"][mol.name] = {**mn, "reference_frequencies": None if mol.frequencies is None else mol.frequencies.tolist(),
                                      "constant_frequencies": minimum(model, mol, True)["frequencies"]}
    (out / "metrics.json").write_text(json.dumps(report, indent=1))
    for name, r in report["validation"].items():
        c = report["constant"][name]
        line = (f"{name:28s} E mae {r['e_mae']:.3f} (tables alone {c['e_mae']:.3f}) kcal/mol   "
                f"F rmse {r['f_rmse']:.2f} ({c['f_rmse']:.2f}) kcal/mol/A")
        for label, unit in (("dipole", "e bohr"), ("quadrupole", "e bohr^2"), ("polarizability", "bohr^3")):
            if f"{label}_rmse" in r:
                line += f"\n{'':28s} {label} rmse {r[label + '_rmse']:.4f} ({c[label + '_rmse']:.4f}) {unit}"
        print(line)
    for name, r in report["minima"].items():
        ref = r["reference_frequencies"]
        print(f"{name:28s} freq " + " ".join(f"{w:.0f}" for w in r["frequencies"])
              + ("" if ref is None else "   reference " + " ".join(f"{w:.0f}" for w in ref)))
    return report


if __name__ == "__main__":
    main()
