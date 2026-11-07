"""Evaluation of ``train_bonding`` checkpoints: per-group metrics, parity data and plots, stretch splits.

The library behind ``scripts/eval_bonding.py``, ``scripts/parity_bonding.py`` and the ChemLab
``train.bonding`` job (:mod:`rsfff.train.bonding_worker`); importable from a notebook.

* :func:`load_model` rebuilds a checkpoint's model (optionally on a device, with config overrides);
* :func:`evaluate` / :func:`summarize`: per frame and per ``name/kind`` group -- energy, force MAE,
  Hessian relative error, dipole / quadrupole / polarizability errors (energy derivatives about the
  centre of nuclear charge, as ``train_bonding`` compares them) and, for probe frames paired with the
  vacuum frame at the same geometry, the interaction-energy and force-change errors;
* :func:`collect` / :func:`plot_parity`: the same quantities as (prediction, reference) arrays and a
  parity figure, one panel per quantity, at most three sets per panel;
* :func:`split_stretch`: hold out the stretched end of the ``scan_stretch`` scans (``r / r_e`` above a
  ratio, ``r_e`` the scan's lowest-energy point) -- the extrapolation test of the bonding study.

Units: kcal/mol, Angstrom, Debye, e*bohr^2, bohr^3.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from ..ff.units import KJMOL_PER_HARTREE
from .build_film import build_film_model
from .data import load_datasets, load_reference_energies, split_indices
from .train_bonding import _labels_about_center

__all__ = ["load_model", "headers", "evaluate", "summarize", "print_summary", "collect", "plot_parity",
           "save_parity", "minimum_energies", "split_stretch", "split_frames", "QUANTITIES", "COLUMNS"]

KCAL_PER_HARTREE = KJMOL_PER_HARTREE / 4.184
DEBYE_PER_AU = 2.541746
BOHR_ANG = 0.52917721067

COLUMNS = [("e_mae", "E"), ("f_mae", "F"), ("h_rel", "H rel"), ("mu", "mu D"),
           ("theta", "Theta"), ("alpha", "alpha"), ("alpha_iso_rel", "a_iso%"),
           ("de_int", "dE_int"), ("df_int", "dF_int")]

#: parity panels: (key, title, unit)
QUANTITIES = [
    ("energy", "Energy rel. to minimum", "kcal/mol"),
    ("forces", "Force components", "kcal/mol/Å"),
    ("hessian", "Hessian elements", "kcal/mol/Å²"),
    ("dipole", "Dipole components", "D"),
    ("quadrupole", "Quadrupole components (traceless)", "e·bohr²"),
    ("alpha", "Polarizability components", "bohr³"),
    ("de_int", "Probe interaction energy", "kcal/mol"),
    ("df_int", "Probe-induced force change", "kcal/mol/Å"),
]

#: categorical slots 1-3 (validated all-pairs for scatter on a light surface)
SET_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]


# --- model and frames -------------------------------------------------------------------------

def load_model(ckpt_path, reference_energies=None, film_overrides: dict | None = None, device="cpu"):
    """``(model, checkpoint)`` in float64 on ``device``. ``film_overrides`` set ``config.film``
    attributes before the model is built (an option a checkpoint predates)."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    for k, v in (film_overrides or {}).items():
        setattr(cfg.film, k, v)
    types = sorted(int(z) for z in ck["neighbor_types"])
    ref = load_reference_energies(reference_energies or cfg.data.reference_energies, types)
    torch.set_default_dtype(torch.float64)
    model = build_film_model(cfg.features, cfg.film, types, ref.double()).double()
    model.load_state_dict(ck["model_state"])
    model.to(device).eval()
    return model, ck


def headers(path):
    """``[(name, kind)]`` of every frame (fff_bonding exports)."""
    from ase.io import iread

    return [(a.info.get("name", "?"), a.info.get("kind", "?")) for a in iread(str(path), index=":")]


def _device(model):
    return next(model.parameters()).device


def _vacuum_index(paths):
    """``{(name, rounded positions bytes): (dataset, frame)}`` over the vacuum files."""
    table = {}
    for p in paths:
        ds = load_datasets([p], dtype=torch.float64)
        for i, (name, _) in enumerate(headers(p)):
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


def split_frames(path, split: str, cfg, max_frames: int | None = None, seed: int = 0) -> np.ndarray:
    """Frame indices of ``path``: ``all``, or the trainer's ``train`` / ``val`` split (the config's
    ``data.holdout_fraction`` and ``data.seed``, which every stream shares); at most ``max_frames``."""
    n = len(headers(path))
    if split == "all":
        idx = np.arange(n)
    else:
        tr, va = split_indices(n, cfg.data.holdout_fraction, cfg.data.seed)
        idx = (va if split == "val" else tr).numpy()
    if max_frames and idx.size > max_frames:
        idx = np.sort(np.random.default_rng(seed).choice(idx, max_frames, replace=False))
    return idx


def split_stretch(path, train_out, heldout_out, max_ratio: float) -> tuple[int, int]:
    """Write ``path`` minus the ``scan_stretch`` frames with ``r / r_e > max_ratio`` to ``train_out``
    and those frames to ``heldout_out`` (``r``: the ``scan_value`` header, ``r_e``: the scan's
    lowest-energy point). Returns the two frame counts."""
    from ase.io import iread, write

    frames = list(iread(str(path), index=":"))
    scans = defaultdict(list)
    for k, a in enumerate(frames):
        if a.info.get("kind") == "scan_stretch" and "scan_value" in a.info:
            scans[(a.info.get("name"), str(a.info.get("scan_bond")))].append(k)
    held = set()
    for idx in scans.values():
        e = [frames[k].get_potential_energy() for k in idx]
        r_e = float(frames[idx[int(np.argmin(e))]].info["scan_value"])
        held |= {k for k in idx if float(frames[k].info["scan_value"]) / r_e > max_ratio}
    keep = [a for k, a in enumerate(frames) if k not in held]
    out = [frames[k] for k in sorted(held)]
    write(str(train_out), keep, format="extxyz")
    if out:
        write(str(heldout_out), out, format="extxyz")
    return len(keep), len(out)


# --- per-group metrics ------------------------------------------------------------------------

def evaluate(model, path, *, idx=None, hessian=True, moments=True, vacuum=None):
    """``{(name, kind): {metric: [per-frame values]}}`` over the frames ``idx`` of ``path``."""
    dev = _device(model)
    ds = load_datasets([path], dtype=torch.float64)
    heads = headers(path)
    vac = _vacuum_index(vacuum) if vacuum else None
    rows = defaultdict(lambda: defaultdict(list))
    frames = range(len(ds)) if idx is None else [int(i) for i in idx]
    for i in frames:
        key = heads[i]
        b0 = ds.flat_batch([i])
        b = b0.to(dev)
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
            hit = vac.get((key[0], np.round(b0.positions.detach().cpu().numpy(), 6).tobytes()))
            if hit is not None:
                vds, j = hit
                bv = vds.flat_batch([j]).to(dev)
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


def print_summary(summary, file=None):
    cols = [(k, t) for k, t in COLUMNS if any(k in s for s in summary.values())]
    print(f"{'molecule/kind':30s} {'n':>4s} " + " ".join(f"{t:>7s}" for _, t in cols), file=file)
    for name, s in summary.items():
        vals = " ".join(f"{s[k]:7.3f}" if k in s else "      -" for k, _ in cols)
        print(f"{name:30s} {s['n']:4d} {vals}", file=file)


# --- parity -----------------------------------------------------------------------------------

def minimum_energies(model, paths):
    """``{name: (E_model, E_ref)}`` of each molecule's ``minimum`` frame in ``paths``, Hartree."""
    dev = _device(model)
    out = {}
    for p in paths:
        ds = load_datasets([p], dtype=torch.float64)
        for i, (name, kind) in enumerate(headers(p)):
            if kind == "minimum" and name not in out:
                b = ds.flat_batch([i]).to(dev)
                with torch.enable_grad():
                    e = float(model(b).energy.sum().detach())
                out[name] = (e, float(b.energy.sum()))
    return out


def collect(model, path, *, idx=None, hessian=True, moments=True, vacuum=None, minima=None):
    """``{quantity: (pred (n,), ref (n,))}`` over the frames ``idx`` of ``path`` (all by default).

    ``minima``: :func:`minimum_energies` (the energy reference per molecule; frames of a molecule
    without one are left out of the energy panel). ``vacuum``: vacuum files to pair probe frames with.
    """
    dev = _device(model)
    ds = load_datasets([path], dtype=torch.float64)
    heads = headers(path)
    vac = _vacuum_index(vacuum) if vacuum else None
    acc = defaultdict(lambda: ([], []))

    def add(key, pred, ref):
        acc[key][0].append(np.asarray(pred, dtype=np.float64).reshape(-1))
        acc[key][1].append(np.asarray(ref, dtype=np.float64).reshape(-1))

    def np_(t):
        return t.detach().cpu().numpy()

    frames = range(len(ds)) if idx is None else [int(i) for i in idx]
    for i in frames:
        name = heads[i][0]
        b0 = ds.flat_batch([i])
        b = b0.to(dev)
        want_h = hessian and b.hessian is not None
        e, f, h = _energy_forces(model, b, hessian=want_h)
        is_probe = b.external is not None
        if minima and name in minima and not is_probe:
            em, er = minima[name]
            add("energy", (float(e) - em) * KCAL_PER_HARTREE, (float(b.energy.sum()) - er) * KCAL_PER_HARTREE)
        if b.forces is not None and not is_probe:
            add("forces", np_(f) * KCAL_PER_HARTREE, np_(b.forces) * KCAL_PER_HARTREE)
        if h is not None:
            h_ref = np_(b.hessian_blocks()[0])
            iu = np.triu_indices(h_ref.shape[0])
            add("hessian", np_(h)[iu] * KCAL_PER_HARTREE, h_ref[iu] * KCAL_PER_HARTREE)
        if vac is not None and is_probe:
            hit = vac.get((name, np.round(b0.positions.detach().cpu().numpy(), 6).tobytes()))
            if hit is not None:
                vds, j = hit
                bv = vds.flat_batch([j]).to(dev)
                ev, fv, _ = _energy_forces(model, bv)
                add("de_int", float(e - ev) * KCAL_PER_HARTREE,
                    float(b.energy.sum() - bv.energy.sum()) * KCAL_PER_HARTREE)
                if b.forces is not None and bv.forces is not None:
                    add("df_int", np_(f - fv) * KCAL_PER_HARTREE, np_(b.forces - bv.forces) * KCAL_PER_HARTREE)
        if moments and not is_probe and (b.dipole is not None or b.polarizability is not None):
            center, dip, theta = _labels_about_center(b)
            _, mu, th, al = model.response_properties(
                b, polarizability=b.polarizability is not None, quadrupole=theta is not None,
                create_graph=False, origin=center,
            )
            iu = np.triu_indices(3)
            if dip is not None:
                add("dipole", np_(mu) * DEBYE_PER_AU, np_(dip) * DEBYE_PER_AU)
            if theta is not None:
                add("quadrupole", np_(th)[0][iu], np_(theta)[0][iu])
            if b.polarizability is not None:
                add("alpha", np_(al)[0][iu], np_(b.polarizability / BOHR_ANG ** 2)[0][iu])
    return {k: (np.concatenate(p), np.concatenate(r)) for k, (p, r) in acc.items()}


def plot_parity(sets: dict, out, *, title=None, quantities=QUANTITIES):
    """One parity panel per quantity present in any set; ``sets`` = ``{label: collect(...)}`` (at
    most three, in categorical order), with MAE / RMSE per set."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(sets) > 3:
        raise ValueError("at most three sets per panel (fold the rest or make two figures)")
    present = [q for q in quantities if any(q[0] in d for d in sets.values())]
    ncol = 4 if len(present) > 4 else max(len(present), 1)
    nrow = int(np.ceil(len(present) / ncol)) or 1
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.4 * nrow), squeeze=False)
    for ax, (key, name, unit) in zip(axes.flat, present):
        lo, hi = np.inf, -np.inf
        lines = []
        for (label, d), color in zip(sets.items(), SET_COLORS):
            if key not in d:
                continue
            p, r = d[key]
            lo, hi = min(lo, p.min(), r.min()), max(hi, p.max(), r.max())
            ax.scatter(r, p, s=9, color=color, alpha=0.55, linewidths=0, label=label, rasterized=True)
            err = p - r
            lines.append((label, np.abs(err).mean(), np.sqrt((err ** 2).mean())))
        pad = 0.04 * (hi - lo) if hi > lo else 1.0
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#b5b4ad", lw=1, zorder=0)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal")
        ax.set_title(name, fontsize=10, color="#0b0b0b")
        ax.set_xlabel(f"reference ({unit})")
        ax.set_ylabel(f"model ({unit})")
        ax.grid(color="#e7e6e2", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        txt = "\n".join(f"{lab}: MAE {mae:.3g}, RMSE {rmse:.3g}" for lab, mae, rmse in lines)
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left", fontsize=7,
                color="#52514e", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
    for ax in list(axes.flat)[len(present):]:
        ax.axis("off")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=lab, markersize=6)
               for lab, c in zip(sets, SET_COLORS)]
    fig.legend(handles=handles, loc="upper center", ncol=len(sets), frameon=False,
               bbox_to_anchor=(0.5, 1.0), fontsize=9)
    if title:
        fig.suptitle(title, y=1.04, fontsize=11, color="#0b0b0b")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_parity(sets: dict, path) -> None:
    """The collected arrays as one npz: ``<set>/<quantity>/<pred|ref>``."""
    np.savez_compressed(path, **{f"{lab}/{k}/{w}": v[n] for lab, d in sets.items()
                                 for k, v in d.items() for n, w in enumerate(("pred", "ref"))})

