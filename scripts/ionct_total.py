"""The complete ion potential (elec + Pauli + disp + pol + CT) against ALMO-EDA and against the total-energy /
force data.

    python scripts/ionct_total.py CT_RUN DISP_PT [--pauli PAULI_PT] --ct-sets CT_SETS --walk-eda DIR --force-dir DIR --out DIR

1. EDA frames (cached, same split as training): every term vs its ALMO label and the sum vs ALMO E_int.
2. Force frames (research/rsfff train/nacl/data/force, wB97M-V/def2-TZVPD): E_int = E - sum of free-ion energies
   (the ions are atoms, so the forces are those of E_int) and the forces, by autograd through the whole model.
   Forces were never trained on; some of these geometries are also EDA frames (their energies were).
Writes total_eda.csv, total_force.csv, force_atoms.csv and ionct_total.png into --out.
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rsfff.ff.ionct.potential import IonPotential
from rsfff.train.train_ionct import KJ, build_cache, load_frames, split

warnings.filterwarnings("ignore")
#: free-ion energies (Hartree), wB97M-V/def2-TZVPD, from the EDA fragment energies
E_ION = {"Na": -162.0687155489, "Cl": -460.2597935}
KJ_ANG = KJ / 0.529177210903          # Hartree/bohr -> kJ/mol/Angstrom
TERMS = (("elec", "elec"), ("pauli", "pauli"), ("disp", "disp"), ("pol", "pol"), ("ct", "ct"), ("int", "int"))


def eda_frame(P, a, ck_args):
    frames = load_frames(a.ct_sets, a.walk_eda, ck_args["ct_max_pair"], ck_args.get("max_ho_ratio", 1.4))
    tags = split(frames, ck_args["test"], seed=ck_args["seed"])
    caches = build_cache(frames, P.tab, P.chan, None, ck_args["cache"])
    rows = []
    with torch.no_grad():
        for k in range(0, len(caches), 64):
            chunk = caches[k:k + 64]
            t = P.terms_cached(chunk)
            for q, c in enumerate(chunk):
                sym, X, lab, meta = frames[k + q]
                r = {"split": tags[k + q], "workflow": meta["workflow"], "parent": meta["parent"], "n": len(sym),
                     "full": bool(meta["full"]), "ct_mask": bool(lab["ct_mask"])}
                for name, key in TERMS:
                    r[name] = t[name][q].item() * KJ
                    r[f"{name}_ref"] = lab.get(key, np.nan) * KJ
                rows.append(r)
    return pd.DataFrame(rows)


def force_frame(P, force_dir, max_per_file=None):
    from ase.io import read
    rows, atoms = [], []
    for f in sorted(Path(force_dir).glob("nacl*.xyz"), key=lambda p: int(p.stem[4:])):
        t0 = time.time()
        frames = read(f, ":")
        step = max(1, len(frames) // max_per_file) if max_per_file else 1
        for k, at in list(enumerate(frames))[::step]:
            sym = at.get_chemical_symbols()
            # extxyz values are stored as-is (atomic units here); ASE moves them to the calculator
            e_tot = at.info["energy"] if "energy" in at.info else at.get_potential_energy()
            F_au = at.arrays["forces"] if "forces" in at.arrays else at.get_forces()
            e_ref = (e_tot - sum(E_ION[s] for s in sym)) * KJ
            F_ref = F_au * KJ_ANG
            E, F, t = P.energy_forces(sym, torch.tensor(at.positions))
            F = F.numpy() * KJ_ANG
            rows.append({"file": f.stem, "frame": k, "n": len(sym), "int_ref": e_ref, "int": E.item() * KJ,
                         **{name: t[name].item() * KJ for name, _ in TERMS[:-1]},
                         "f_mae": np.abs(F - F_ref).mean(), "f_rms": np.sqrt(((F - F_ref) ** 2).mean()),
                         "f_ref_rms": np.sqrt((F_ref ** 2).mean())})
            for i, s in enumerate(sym):
                atoms.append({"file": f.stem, "frame": k, "atom": i, "sym": s, **{f"F{x}": F[i, d] for d, x in enumerate("xyz")},
                              **{f"F{x}_ref": F_ref[i, d] for d, x in enumerate("xyz")}})
        print(f"  {f.stem}: {len(frames[::step])} frames [{time.time() - t0:.0f}s]", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(atoms)


def plot(eda, frc, atoms, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted = "#0b0b0b", "#8a8984"
    col = {"train": "#b9b8b3", "val": "#2a78d6", "test": "#eb6834"}
    fig, axs = plt.subplots(2, 4, figsize=(19, 9.2))
    panels = (("elec", "frozen electrostatics vs cls_elec"), ("pauli", "frozen Pauli vs mod_pauli"),
              ("disp", "dispersion vs disp"), ("int", "total E_int vs ALMO int (all frames)"))
    for ax, (t, title) in zip(axs[0], panels):
        d = eda[np.isfinite(eda[f"{t}_ref"])]
        for s in ("train", "val", "test"):
            g = d[d.split == s]
            ax.plot(g[f"{t}_ref"], g[t], "o", ms=3, mec="none", color=col[s], alpha=0.8, label=f"{s} ({len(g)})")
        lo, hi = min(d[f"{t}_ref"].min(), d[t].min()), max(d[f"{t}_ref"].max(), d[t].max())
        ax.plot([lo, hi], [lo, hi], color=muted, lw=0.8, zorder=0)
        g = d[(d.split == "test") & d.full & (d.n > 3)]
        txt = f"test MAE {np.abs(d[d.split == 'test'][t] - d[d.split == 'test'][f'{t}_ref']).mean():.2f}\n" \
              f"test whole clusters MAE {np.abs(g[t] - g[f'{t}_ref']).mean():.2f}"
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", fontsize=9, color=ink)
        ax.set_title(title, fontsize=10, color=ink)
        ax.set_xlabel("ALMO (kJ/mol)")
        ax.set_ylabel("model (kJ/mol)")
        ax.legend(loc="lower right", fontsize=8, frameon=False)
        ax.grid(alpha=0.2)
    # force set
    ax = axs[1, 0]
    ax.plot(frc.int_ref, frc.int, "o", ms=3, mec="none", color="#2a78d6", alpha=0.8)
    lo, hi = min(frc.int_ref.min(), frc.int.min()), max(frc.int_ref.max(), frc.int.max())
    ax.plot([lo, hi], [lo, hi], color=muted, lw=0.8, zorder=0)
    ax.set_title("force set: E_int (forces never trained)", fontsize=10, color=ink)
    ax.set_xlabel("wB97M-V E_int (kJ/mol)")
    ax.set_ylabel("model (kJ/mol)")
    ax.grid(alpha=0.2)
    ax = axs[1, 1]
    for s, c in (("Na", "#2a78d6"), ("Cl", "#eb6834")):
        g = atoms[atoms.sym == s]
        ax.plot(g[["Fx_ref", "Fy_ref", "Fz_ref"]].values.ravel(), g[["Fx", "Fy", "Fz"]].values.ravel(), "o", ms=2,
                mec="none", color=c, alpha=0.5, label=s)
    lim = np.abs(atoms[["Fx_ref", "Fy_ref", "Fz_ref"]].values).max()
    ax.plot([-lim, lim], [-lim, lim], color=muted, lw=0.8, zorder=0)
    ax.set_title("force set: force components", fontsize=10, color=ink)
    ax.set_xlabel("wB97M-V (kJ/mol/Å)")
    ax.set_ylabel("model (kJ/mol/Å)")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.2)
    ax = axs[1, 2]
    g = frc.groupby("n")
    ax.plot(g.n.first(), (g.apply(lambda x: (x.int - x.int_ref).abs().mean()) / g.n.first()), "o-", color="#2a78d6",
            lw=2, ms=8)
    ax.set_title("force set: E_int MAE per ion vs size", fontsize=10, color=ink)
    ax.set_xlabel("ions")
    ax.set_ylabel("kJ/mol per ion")
    ax.grid(alpha=0.2)
    ax = axs[1, 3]
    ax.plot(g.n.first(), g.f_mae.mean(), "o-", color="#eb6834", lw=2, ms=8, label="model − ref MAE")
    ax.plot(g.n.first(), g.f_ref_rms.mean(), "s--", color=muted, lw=1.5, ms=6, label="ref force rms")
    ax.set_title("force set: force error vs size", fontsize=10, color=ink)
    ax.set_xlabel("ions")
    ax.set_ylabel("kJ/mol/Å")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ct_run")
    ap.add_argument("disp")
    ap.add_argument("--pauli", default=None, help="separately fit frozen Pauli (IonPauli); default: the CT model's")
    ap.add_argument("--ct-sets", required=True)
    ap.add_argument("--walk-eda", default=None)
    ap.add_argument("--force-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-file", type=int, default=None, help="evenly strided subset of each force file")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    P = IonPotential.from_runs(a.ct_run, a.disp, a.pauli)
    ck_args = torch.load(Path(a.ct_run) / "model.pt", weights_only=False)["args"]
    eda = eda_frame(P, a, ck_args)
    eda.to_csv(out / "total_eda.csv", index=False)
    print("EDA frames, MAE kJ/mol (all / whole clusters n>3):")
    for name, _ in TERMS:
        line = []
        for s in ("train", "val", "test"):
            d = eda[(eda.split == s) & np.isfinite(eda[f"{name}_ref"]) & ((eda.ct_mask) if name in ("ct", "int") else True)]
            c = d[d.full & (d.n > 3)]
            line.append(f"{s} {np.abs(d[name] - d[f'{name}_ref']).mean():.2f} / {np.abs(c[name] - c[f'{name}_ref']).mean():.2f}")
        print(f"  {name:6s} " + " | ".join(line))
    frc, atoms = force_frame(P, a.force_dir, a.max_per_file)
    frc.to_csv(out / "total_force.csv", index=False)
    atoms.to_csv(out / "force_atoms.csv", index=False)
    print("force set (never trained):")
    for n, g in frc.groupby("n"):
        print(f"  n={n:2d} frames {len(g):3d}  E_int MAE {np.abs(g.int - g.int_ref).mean():7.2f} kJ/mol "
              f"({np.abs(g.int - g.int_ref).mean() / n:.2f}/ion, bias {(g.int - g.int_ref).mean():+.2f})  "
              f"F MAE {g.f_mae.mean():.2f} rms {g.f_rms.mean():.2f} kJ/mol/A (ref rms {g.f_ref_rms.mean():.1f})")
    plot(eda, frc, atoms, out / "ionct_total.png")


if __name__ == "__main__":
    main()
