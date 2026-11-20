"""Train a bonded model (:class:`rsfff.ff.bonded.BondedModel`) on energies, forces and, where the frames carry them,
dipoles, second moments and polarizabilities, from one config file.

    python -m rsfff.train.bonded CONFIG.toml [--set train.epochs=50 encoder.latent=4 ...]

The model tables ``[terms]``, ``[tables]``, ``[encoder]`` are :class:`rsfff.ff.bonded.BondedConfig`. The trainer's:

    [data]    train (extxyz files; one molecule type per file is assumed for the per-file report), val_fraction,
              holdout (frames whose ``config_type`` starts with this, e.g. "scan", are neither trained on nor used
              for validation: they are the test set), reference (optimized structures with ``frequencies=``: a
              check, never fitted), root (default: the working directory; relative paths, tables.init included)
    [loss]    energy, forces, dipole, quadrupole, polarizability, dipole_derivatives, polarizability_derivatives,
              hessian (weights); the residual scales sigma_energy [kcal/mol], sigma_force [kcal/mol/A],
              sigma_dipole [e bohr], sigma_quadrupole [e bohr^2], sigma_polarizability [bohr^3],
              sigma_dipole_derivatives [e], sigma_polarizability_derivatives [bohr^2], sigma_hessian [Ha/bohr^2];
              probes (random projections per step for the polarizability derivatives and the Hessian)
    [train]   out, epochs, batch, lr, table_lr (learning rate of the bonded tables and reference energies; the
              electrical tables, which have no fit to start from, use lr), bonded_prefit (first fit the bonded
              tables and reference energies alone, i.e. the constant-parameter model, to energies and forces by
              L-BFGS: the starting point when ``tables.init`` has no fit for these types), table_prefit (then fit the electrical
              tables alone to the training frames' electrical labels, by L-BFGS, before anything else moves: without
              it the first steps' electrical residuals are thousands of sigma and wreck the energy surface), seed,
              log_every, constant_only (train the tables alone: the constant-parameter model)

Loss: sum over quantities of weight mean((residual / sigma)^2); energies per molecule; an electrical quantity counts
the molecules of the batch that carry its label (frames without one are skipped for it) and is used only when the
model has an electrical head. Moments are about the files' coordinate origin. The derivative labels are fitted as
derivatives of the model's own outputs with respect to the positions: the dipole derivatives in full (three backward
passes), the polarizability derivatives and the Hessian through ``probes`` random projections per step (a random
symmetric 3 x 3 contraction; a Hessian-vector product). The reports use fixed projections.
The atomic reference energies start from a least-squares fit on the training frames. The edge features are
standardized on the training frames. Outputs in ``train.out``: model.pt (best validation loss; config, atom types,
state), metrics.json (per file: validation errors of the trained and of the tables-alone model, the same for the
held-out frames, harmonic frequencies and geometry at the model's minimum against the reference), log.csv, and
holdout.csv (per held-out frame: file, config_type, reference and model energy, force rmse).
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
    "data": {"train": [], "val_fraction": 0.2, "holdout": "", "reference": [], "root": "."},
    "loss": {"energy": 1.0, "forces": 1.0, "dipole": 1.0, "quadrupole": 1.0, "polarizability": 1.0,
             "sigma_energy": 0.1, "sigma_force": 1.0, "sigma_dipole": 0.005, "sigma_quadrupole": 0.02,
             "sigma_polarizability": 0.05, "dipole_derivatives": 1.0, "polarizability_derivatives": 1.0,
             "hessian": 1.0, "sigma_dipole_derivatives": 0.01, "sigma_polarizability_derivatives": 0.05,
             "sigma_hessian": 0.005, "probes": 2},
    "train": {"out": "runs/bonded", "epochs": 300, "batch": 64, "lr": 2e-3, "table_lr": 2e-4, "bonded_prefit": True, "table_prefit": True, "seed": 0,
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


UNITS = (("dipole", "e bohr"), ("quadrupole", "e bohr^2"), ("polarizability", "bohr^3"), ("dipole_derivatives", "e"),
         ("polarizability_derivatives", "bohr^2 (projected)"), ("hessian", "Ha/bohr^2 (projected)"))
ELECTRICAL = (("dipole", "dipole"), ("quadrupole", "second_moment"), ("polarizability", "polarizability"))


DERIVATIVE = ("dipole_derivatives", "polarizability_derivatives", "hessian")


def _atoms_of(b, have):
    return have[b.mol_idx]


def derivative_residuals(model, b, out, pos, grad, probes, gen, create_graph):
    """{label: residual tensor (model - reference, atomic units per bohr)} for the derivative labels ``b`` carries.

    ``out`` and ``grad`` (dE/dpos, with its graph) are the model's outputs at ``pos`` (requires grad). Dipole
    derivatives in full; polarizability derivatives contracted with ``probes`` random symmetric matrices; the
    Hessian as ``probes`` Hessian-vector products."""
    res = {}
    rnd = lambda *shape: torch.randn(*shape, generator=gen, dtype=pos.dtype)
    if model.has_electrical and b.dipole_derivatives is not None:
        atoms = _atoms_of(b, b.labelled("dipole_derivatives"))
        cols = [torch.autograd.grad(out["dipole"][:, k].sum(), pos, create_graph=create_graph, retain_graph=True)[0] for k in range(3)]
        pred = torch.stack(cols, -1) * BOHR_ANG                                     # [atom, i, k]
        res["dipole_derivatives"] = (pred - b.dipole_derivatives)[atoms]
    if model.has_electrical and b.polarizability_derivatives is not None:
        atoms = _atoms_of(b, b.labelled("polarizability_derivatives"))
        parts = []
        for _ in range(probes):
            w = rnd(3, 3)
            w = 0.5 * (w + w.T)
            (g,) = torch.autograd.grad((out["polarizability"] * w).sum(), pos, create_graph=create_graph, retain_graph=True)
            parts.append((g * BOHR_ANG - b.polarizability_derivatives @ w.reshape(9))[atoms])
        res["polarizability_derivatives"] = torch.stack(parts, -1)
    if b.hessians is not None:
        have = b.labelled("hessians")
        atoms = _atoms_of(b, have)
        offset = torch.cumsum(torch.bincount(b.mol_idx, minlength=b.n_mol), 0)
        parts = []
        for _ in range(probes):
            v = rnd(pos.shape[0], 3) * atoms[:, None]
            (hv,) = torch.autograd.grad((grad * v).sum(), pos, create_graph=create_graph, retain_graph=True)
            ref = torch.zeros_like(v)
            for m in torch.nonzero(have).flatten().tolist():
                lo, hi = int(offset[m - 1]) if m else 0, int(offset[m])
                ref[lo:hi] = (b.hessians[m] @ v[lo:hi].reshape(-1)).reshape(-1, 3)
            parts.append((hv * BOHR_ANG ** 2 - ref)[atoms])
        res["hessian"] = torch.stack(parts, -1)
    return res


def errors(model, batches, constant=False):
    """{file: {n, e_mae [kcal/mol], e_max, f_rmse [kcal/mol/A], and per labelled electrical quantity its
    component rmse in atomic units (dipole_rmse, quadrupole_rmse, polarizability_rmse)}} over a list of batches."""
    acc, el = {}, {}
    gen = torch.Generator().manual_seed(0)
    for b in batches:
        pos = b.positions.detach().clone().requires_grad_(True)
        out = model(b, pos, constant)
        e = out["energy"]
        (g,) = torch.autograd.grad(e.sum(), pos, create_graph=True)
        f = -g * BOHR_ANG
        for label, r in derivative_residuals(model, b, out, pos, g, 3, gen, False).items():
            key = "hessians" if label == "hessian" else label
            mol_of = b.mol_idx[_atoms_of(b, b.labelled(key))]
            d2 = (r.detach() ** 2).reshape(r.shape[0], -1)
            for m in torch.unique(mol_of).tolist():
                a = el.setdefault((b.names[m], label), [0.0, 0])
                a[0] += float(d2[mol_of == m].sum())
                a[1] += int((mol_of == m).sum()) * d2.shape[1]
        e, f = e.detach(), f.detach()
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
    if not bool(torch.isfinite(h).all()):
        return {"bond_lengths": [], "frequencies": [float("nan")] * (3 * n - (5 if n == 2 else 6))}
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

    train, val, held = [], [], []
    for f in D["train"]:
        mols = read_extxyz(root / f)
        if D["holdout"]:
            held += [m for m in mols if m.config.startswith(D["holdout"])]
            mols = [m for m in mols if not m.config.startswith(D["holdout"])]
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
    held_batches = [collate(held[i:i + 256]) for i in range(0, len(held), 256)]

    fitted = [model.tables.bond, model.tables.apex, model.tables.e0]            # start from the constant fit
    unfitted = [model.tables.atom, model.tables.link]                           # electrical: start from defaults
    zero_start = [] if model.tables.torsion is None else [model.tables.torsion]
    rest = [] if T["constant_only"] else [p for n, p in model.named_parameters() if not n.startswith("tables.")]
    opt = torch.optim.Adam([{"params": rest + unfitted + zero_start, "lr": T["lr"]}, {"params": fitted, "lr": T["table_lr"]}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(int(T["epochs"]), 1))
    s_e, s_f = L["sigma_energy"] / KCAL, L["sigma_force"] / KCAL * BOHR_ANG
    const = bool(T["constant_only"])

    sig = {"dipole": L["sigma_dipole"], "quadrupole": L["sigma_quadrupole"], "polarizability": L["sigma_polarizability"]}

    sig_d = {k: L[f"sigma_{k}"] for k in DERIVATIVE}
    train_gen = torch.Generator().manual_seed(int(T["seed"]) + 1)

    def loss_of(b, create_graph, gen=None):
        pos = b.positions.detach().clone().requires_grad_(True)
        out = model(b, pos, const)
        e = out["energy"]
        (g,) = torch.autograd.grad(e.sum(), pos, create_graph=True)
        f = -g * BOHR_ANG
        loss = L["energy"] * (((e - b.energy) / s_e) ** 2).mean() + L["forces"] * (((f - b.forces) / s_f) ** 2).mean()
        if model.has_electrical:
            for label, key in ELECTRICAL:
                have = b.labelled(key)
                if L[label] > 0 and bool(have.any()):
                    loss = loss + L[label] * (((out[key][have] - getattr(b, key)[have]) / sig[label]) ** 2).mean()
        if any(L[k] > 0 for k in DERIVATIVE):
            res = derivative_residuals(model, b, out, pos, g, int(L["probes"]), gen or train_gen, create_graph)
            for label, r in res.items():
                if L[label] > 0 and r.numel():
                    loss = loss + L[label] * ((r / sig_d[label]) ** 2).mean()
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
        gen = torch.Generator().manual_seed(12345)                      # the same projections at every epoch
        tot = sum(float(loss_of(b, False, gen).detach()) * b.n_mol for b in val_batches)
        return tot / max(len(val), 1)

    out = Path(T["out"])
    out.mkdir(parents=True, exist_ok=True)
    n_par = sum(p.numel() for g in opt.param_groups for p in g["params"] if p.requires_grad)
    print(f"{len(train)} training, {len(val)} validation and {len(held)} held-out frames; atom types {model.atom_types}; "
          f"{n_par} trained parameters; tables from file: {model.tables.from_file}")
    best, log, t0 = math.inf, [], time.time()
    order = np.arange(len(train))
    if T["bonded_prefit"]:
        sub = collate([train[i] for i in rng.permutation(len(train))[:1500]])
        params = [q for q in fitted + zero_start if q.requires_grad]
        pre = torch.optim.LBFGS(params, max_iter=200, line_search_fn="strong_wolfe", tolerance_grad=1e-9, tolerance_change=1e-12)

        def ef_loss():
            e, f = model.energy_forces(sub, create_graph=True, constant=True)
            return (L["energy"] * (((e - sub.energy) / s_e) ** 2).mean() + L["forces"] * (((f - sub.forces) / s_f) ** 2).mean())

        def closure_b():
            pre.zero_grad()
            loss = ef_loss()
            loss.backward()
            return loss

        first = float(ef_loss().detach())
        backup = [q.detach().clone() for q in params]
        pre.step(closure_b)
        last = float(ef_loss().detach())
        if not math.isfinite(last) or last > first:              # a failed line search: keep the starting tables
            with torch.no_grad():
                for q, q0 in zip(params, backup):
                    q.copy_(q0)
            print(f"bonded tables alone: the fit did not improve on the starting tables (loss {first:.1f}); kept them")
        else:
            print(f"bonded tables alone on {sub.n_mol} frames: loss {first:.1f} -> {last:.2f}  {time.time() - t0:.0f} s")
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

    report = {"validation": errors(model, val_batches, const), "constant": errors(model, val_batches, True),
              "holdout": errors(model, held_batches, const), "holdout_constant": errors(model, held_batches, True), "minima": {}}
    if held:
        rows = [("file", "config_type", "energy_reference", "energy_model", "force_rmse_kcal_per_A")]
        for b in held_batches:
            e, f = model.energy_forces(b, constant=const)
            fr = ((f.detach() - b.forces) ** 2).sum(-1) * (KCAL / BOHR_ANG) ** 2
            n_at = torch.bincount(b.mol_idx, minlength=b.n_mol)
            per = torch.zeros(b.n_mol, dtype=fr.dtype).index_add(0, b.mol_idx, fr) / (3 * n_at)
            rows += [(b.names[m], b.configs[m], f"{float(b.energy[m]):.10f}", f"{float(e[m]):.10f}", f"{float(per[m].sqrt()):.4f}")
                     for m in range(b.n_mol)]
        with open(out / "holdout.csv", "w", newline="") as fh:
            csv.writer(fh).writerows(rows)
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
        for label, unit in UNITS:
            if f"{label}_rmse" in r:
                line += f"\n{'':28s} {label} rmse {r[label + '_rmse']:.4f} ({c[label + '_rmse']:.4f}) {unit}"
        print(line)
    for name, r in report["holdout"].items():
        c = report["holdout_constant"][name]
        line = (f"{'held out: ' + name:28s} E mae {r['e_mae']:.3f} (tables alone {c['e_mae']:.3f}) kcal/mol   "
                f"F rmse {r['f_rmse']:.2f} ({c['f_rmse']:.2f}) kcal/mol/A")
        for label, unit in UNITS:
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
