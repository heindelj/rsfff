"""Regenerate notebooks/eda_rsff_plotting.ipynb.

Emits plain notebook JSON rather than using nbformat, which is not in the environment.
The notebook is only a presentation layer: the analysis lives in
``scripts/dispersion_analysis.py`` and ``scripts/dispersion_plots.py``, and
``scripts/run_dispersion_analysis.py`` runs the same code headlessly to regenerate every
figure in ``notebooks/figures/``.
"""

import json
from pathlib import Path

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    CELLS.append({
        "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    })


md("""
# Range-separated dispersion vs. Q-Chem ALMO-EDA

The model is `rsfff.ff.dispersion.TTDispersion`: a Tang–Toennies damped $C_6$ backbone whose
per-atom coefficients come from environment-aware SOAP descriptors, plus a short-range
**pairwise** correction head, range-separated by a learnable Fermi switch.

$$E_\\mathrm{disp} = \\sum_{i<j}^{\\text{inter}} \\Big[\\, S_\\mathrm{Fermi}(r_{ij})\\, T_\\mathrm{cut}(r_{ij})\\,
\\big(-f_6(b_{ij} r_{ij})\\, C_6^{ij} / r_{ij}^6\\big) \\;+\\; W(r_{ij})\\, \\Delta E_{ij} \\,\\Big]$$

Trained on all 9597 EDA-labeled water clusters (w2–w5) against the `eda_disp` component.

**Caveat worth keeping in view:** `eda_disp` under ωB97X-V is the VV10 nonlocal-correlation
contribution to the *frozen* interaction, and `eda_mod_pauli` is *defined* as
`cls_pauli − disp`. The two are entangled by the decomposition, so the fitted $C_6$ is not a
transferable Casimir–Polder coefficient.
""")

code("""
%load_ext autoreload
%autoreload 2

import os, sys
from pathlib import Path

# Resolve the repo root and work from there: every path below is repo-root-relative.
# Both steps are absolute and idempotent -- a relative sys.path entry like "../scripts"
# would be resolved at *import* time, i.e. after the chdir, and would miss.
ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
os.chdir(ROOT)
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch

import dispersion_plots as dp
from dispersion_analysis import (
    CLUSTERS, load_model, load_clusters, predict, atom_parameters,
    many_body_table, strongest_frames,
)

dp.use_style()

model, cfg, state = load_model(
    "checkpoints/eda_water_disp/best.pt", "configs/eda_water_disp_full.yaml"
)
datasets = load_clusters()

print(f"working from {ROOT}")
print(f"checkpoint from epoch {state['epoch']}, val loss {state['val_loss']:.4f}")
print(f"learned range separation r0 = {float(model.r0.detach()):.3f} A "
      f"(started at {cfg.dispersion.r0_init})")
print({tag: len(ds) for tag, ds in datasets.items()})
""")

md("""
## 1. Accuracy

`corr_fraction` is $|E_\\mathrm{corr}| / (|E_\\mathrm{FF}| + |E_\\mathrm{corr}|)$ — the share of the
pair energy the neural head supplied. It starts at exactly 0 (the head is zero-initialized,
so training begins as the pure force field) and the $r_0$ penalty pushes against it growing.
""")

code("""
preds = {tag: predict(model, datasets[tag], tag) for tag in CLUSTERS}

print(f"{'cluster':<9}{'n':<7}{'MAE':<9}{'RMSE':<9}{'R^2':<10}{'<corr frac>':<12}")
for tag, p in preds.items():
    print(f"{tag:<9}{len(p.ref):<7}{p.mae:<9.4f}{p.rmse:<9.4f}{p.r2:<10.5f}"
          f"{p.corr_fraction.mean():<12.4f}")

err = np.concatenate([p.error for p in preds.values()])
print(f"\\noverall MAE {np.abs(err).mean():.4f} kJ/mol over {len(err)} frames")
""")

code("""
fig = dp.correlation_panels(preds, path="notebooks/figures/disp_correlation.png")
""")

md("""
## 2. The learned effective dispersion coefficients

This is the payoff of letting the descriptors see the environment: $C_6$ is no longer a
per-element constant but an **effective** coefficient that responds to what surrounds the atom.
The dashed line is the per-species value the environment MLP modulates around.
""")

code("""
records = {k: np.concatenate([
    atom_parameters(model, datasets[t], n_frames=150)[0][k] for t in CLUSTERS
]) for k in ("Z", "c6", "b", "n_fragments", "contact")}
_, priors = atom_parameters(model, datasets["w2"], n_frames=1)

for z, sym in ((1, "H"), (8, "O")):
    m = records["Z"] == z
    c6, b = records["c6"][m], records["b"][m]
    print(f"{sym}:  C6 = {c6.mean():7.3f} +/- {c6.std():.3f}  "
          f"[{c6.min():.3f}, {c6.max():.3f}]   prior {priors[z][0]:.3f}")
    print(f"    b  = {b.mean():7.3f} +/- {b.std():.3f}  "
          f"[{b.min():.3f}, {b.max():.3f}]   prior {priors[z][1]:.3f}")
""")

code("""
fig = dp.parameter_panels(records, priors, path="notebooks/figures/disp_parameters.png")
""")

md("""
Hydrogen's distribution comes out **bimodal**. Cluster size turns out not to explain it — the
medians barely move from dimer to pentamer. What does explain it is *local* engagement: the
distance from each atom to the nearest atom in another molecule, which for a hydrogen separates
H-bond donors (~1.8 Å to the acceptor oxygen) from free OH hydrogens.
""")

code("""
fig = dp.parameter_vs_environment(
    records, priors, path="notebooks/figures/disp_parameters_env.png"
)
""")

md("""
## 3. Many-body expansion

For a cluster of $N$ molecules the interaction decomposes exactly as

$$E(1\\ldots N) = \\sum_k E^{(k)}, \\qquad
E^{(k)} = \\sum_{|S| = k} \\sum_{T \\subseteq S} (-1)^{|S| - |T|} E(T)$$

Each subset is evaluated **with only that subset present**, so the descriptors see exactly the
sub-cluster. A pair sum with per-element coefficients is strictly additive and every
$E^{(k \\geq 3)}$ vanishes identically — `tests/test_many_body.py` asserts precisely that, which
is what makes a nonzero value here meaningful rather than a bug.

We take the 167 most strongly interacting frames of each of w3/w4/w5 (~500 clusters),
ranked by total EDA interaction energy.
""")

code("""
mbe = many_body_table(model, datasets, per_cluster=167)

print(f"{'cluster':<9}{'n':<6}{'total':<10}{'2-body':<10}{'3-body':<10}"
      f"{'4-body':<10}{'5-body':<10}{'|MB|/|tot|':<10}")
for tag in ("w3", "w4", "w5"):
    m = mbe["tag"] == tag
    print(f"{tag:<9}{m.sum():<6}{mbe['total'][m].mean():<10.3f}"
          f"{mbe['two_body'][m].mean():<10.3f}{mbe['e3'][m].mean():<10.3f}"
          f"{mbe['e4'][m].mean():<10.3f}{mbe['e5'][m].mean():<10.3f}"
          f"{np.abs(mbe['many_body'][m] / mbe['total'][m]).mean() * 100:<10.2f}")

print(f"\\nnon-additivity from effective C6:      {mbe['mb_ff'].mean():+.4f} kJ/mol")
print(f"non-additivity from correction head:   {mbe['mb_corr'].mean():+.4f} kJ/mol")
""")

md("""
### Is the many-body content *right*?

There is no reference many-body dispersion in this dataset — that would need EDA calculations
on every sub-cluster. But there is one indirect handle worth taking seriously.

The model's 2-body sum is built purely from **isolated dimer** evaluations, and dimers are
exactly where we do have direct reference data: 2401 w2 frames, fit to 0.045 kJ/mol. So
`reference − model 2-body` is a reasonable proxy for the *true* many-body dispersion of each
cluster. Comparing it against the model's own many-body terms is the closest thing to a
validation available.

Read the result with the caveat that it is **not fully independent**: the model was trained on
the w3/w4/w5 totals, so it is partly guaranteed to fit them. What is *not* guaranteed is that
the required correction has the right size and grows the right way with cluster size — and it
does, from ~0.13 kJ/mol at trimers to ~0.79 at pentamers.
""")

code("""
print(f"{'cluster':<9}{'MAE full':<11}{'MAE 2-body':<13}{'model MB':<12}{'reference asks':<15}")
for tag in ("w3", "w4", "w5"):
    m = mbe["tag"] == tag
    full = np.abs(mbe["total"][m] - mbe["ref"][m]).mean()
    two = np.abs(mbe["two_body"][m] - mbe["ref"][m]).mean()
    print(f"{tag:<9}{full:<11.4f}{two:<13.4f}"
          f"{(mbe['total'][m] - mbe['two_body'][m]).mean():<+12.4f}"
          f"{(mbe['ref'][m] - mbe['two_body'][m]).mean():<+15.4f}")

model_mb = mbe["total"] - mbe["two_body"]
needed_mb = mbe["ref"] - mbe["two_body"]
print(f"\\nmodel many-body vs (reference - model 2-body): "
      f"r = {np.corrcoef(model_mb, needed_mb)[0, 1]:.4f}, "
      f"slope = {np.polyfit(model_mb, needed_mb, 1)[0]:.3f}")
""")

code("""
fig = dp.many_body_panels(mbe, path="notebooks/figures/disp_many_body.png")
""")

md("""
### Does the non-additivity actually buy anything?

The many-body content above is exact *for the model*, but the dataset has **no reference
many-body dispersion** — that would need EDA calculations on every sub-cluster, which we don't
have. So we test it indirectly, with two controls:

| variant | flexibility | many-body |
|---|---|---|
| **environment-aware** (full) | environment $C_6$ + correction head | yes |
| **intra-fragment** | *identical capacity*, descriptors grouped by fragment | **zero, by construction** |
| **per-species only** | rigid $C_6$/$b$, no correction head | zero |

The **intra-fragment** row is the control that matters. It keeps every parameter the full model
has and only changes what each atom is allowed to *see*, so its pair sum is strictly additive
(asserted in `tests/test_many_body.py::test_intra_fragment_features_are_additive`) while its
capacity is unchanged. Any gap between it and the full model is many-body content and nothing
else.

The per-species row is a weaker control: it removes flexibility *and* non-additivity at once,
so on dimers — where many-body is zero by definition — its gap measures flexibility alone.
Reading that gap as "many-body" would be a mistake.
""")

code("""
variants = {"environment-aware (full)": preds}
for label, run in (
    ("intra-fragment (additive, same capacity)", "intra"),
    ("per-species only (additive, rigid)", "additive"),
):
    m, _, s = load_model(
        f"checkpoints/eda_water_disp_{run}/best.pt",
        f"configs/eda_water_disp_{run}.yaml",
    )
    variants[label] = {tag: predict(m, datasets[tag], tag) for tag in CLUSTERS}

print(f"{'variant':<44}" + "".join(f"{t:<10}" for t in CLUSTERS))
for label, p in variants.items():
    print(f"{label:<44}" + "".join(f"{p[t].mae:<10.4f}" for t in CLUSTERS))

intra = variants["intra-fragment (additive, same capacity)"]
print("\\nmany-body gain (full vs additive-of-equal-capacity):")
for tag in CLUSTERS:
    print(f"  {tag}: {intra[tag].mae:.4f} -> {preds[tag].mae:.4f} kJ/mol "
          f"({intra[tag].mae / preds[tag].mae:.2f}x)")
""")

code("""
fig = dp.additive_comparison(
    variants, path="notebooks/figures/disp_additive_control.png"
)
""")

md("""
## 4. Scratch space

`mbe` is a dict of flat numpy arrays (`tag`, `n_fragments`, `total`, `two_body`, `many_body`,
`e3`/`e4`/`e5`, `mb_ff`, `mb_corr`, `ref`), and `preds[tag]` carries `pred`/`ref`/`ff`/`corr`
per frame — all in kJ/mol. `rsfff.ff.many_body.mbe_decompose` runs a single cluster if you
want to look at one in detail.
""")

code("""
# one pentamer, in full
from rsfff.ff.many_body import mbe_decompose
from rsfff.ff.units import KJMOL_PER_HARTREE as K

idx = strongest_frames(datasets["w5"], 1)[0]
frame = datasets["w5"].flat_batch([idx])
res = mbe_decompose(model, frame.positions, frame.atomic_numbers, frame.fragment_idx)

print(f"reference eda_disp {float(frame.eda['disp']) * K:8.3f} kJ/mol")
print(f"model total        {float(res.total) * K:8.3f} kJ/mol")
for k, v in res.by_order.items():
    print(f"  {k}-body          {float(v) * K:8.3f}")
""")

nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "rsfff", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.15"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
path = Path("notebooks/eda_rsff_plotting.ipynb")
path.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {path} with {len(CELLS)} cells")
