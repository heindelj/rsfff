"""Light refit of the ions' permanent densities against ALMO's frozen electrostatics (ion ct ff plan).

    python scripts/ion_elec_refit.py CT_SETS [--walk-eda DIR] [--params IN] [--out OUT] [--lam 1.0]

The permanent densities of ion_params.py come from each free ion's ESP alone; in the clusters the shaped-density
electrostatics then overbinds by ~6 kJ/mol per ion against ALMO cls_elec (penetration at contact: Na-Cl -3.4 kJ/mol
at 2.8 A, Cl-Cl -4.7). ALMO's frozen electrostatics is exactly pairwise additive over free-ion densities, so the fix
belongs in those densities. Here the shell exponents (Na: b; Cl: b1, b2) and the Cl shell split (n1, n2 = 12 - n1,
so the charge and nuclear Z stay fixed) are refined against cls_elec of every frame of the CT training data
(per-ion squared error, kJ/mol), with a penalty lam * sum (ln p / p0)^2 that keeps them near the ESP fit. The free
ion's charge, and so its far field, is unchanged; the change of its potential outside 1.5 vdW-ish radii is reported.

Writes OUT (a copy of IN with the refit shells and an 'elec_refit' record). b_ind, the response shapes, the frontier
shapes and alpha/C are left as fitted; the 'channels' section must be regenerated (ion_ct_pairs.py) since the
frontier potentials depend on the permanent densities.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import warnings
from pathlib import Path

import numpy as np
import torch

from rsfff.ff.ionct import IonTable
from rsfff.ff.ionct import kernels as K
from rsfff.ff.multipole import slater_one_center_damp
from rsfff.train.train_ionct import KJ, _geom_key, load_frames

warnings.filterwarnings("ignore")
ZMAP = {"Na": 11, "Cl": 17}


def elec_labels(ct_sets, walk_eda):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ionct_parity import elec_labels as f
    return f(ct_sets, walk_eda)


def flat_pairs(frames, tab):
    """All intra-frame pairs of all frames, concatenated: positions (P,2,3) as one tensor with frame index."""
    pos, sp, fi, n = [], [], [], []
    off = 0
    I, J, F = [], [], []
    for k, (sym, X, *_rest) in enumerate(frames):
        m = len(sym)
        pos.append(torch.tensor(X, dtype=torch.float64))
        sp.append(tab.species_index(torch.tensor([ZMAP[a] for a in sym])))
        i, j = torch.triu_indices(m, m, 1)
        I.append(i + off)
        J.append(j + off)
        F.append(torch.full((i.shape[0],), k))
        n.append(m)
        off += m
    return torch.cat(pos), torch.cat(sp), torch.stack([torch.cat(I), torch.cat(J)]), torch.cat(F), torch.tensor(n)


class Shells(torch.nn.Module):
    """ln b for Na (1 shell), Cl (2 shells), and the Cl split as n1 = 12 sigmoid(t)."""

    def __init__(self, tab: IonTable):
        super().__init__()
        self.tab0 = tab
        self.na, self.cl = tab.symbols.index("Na"), tab.symbols.index("Cl")
        self.p0 = {"b_Na": tab.shell_b[self.na, 0], "b1_Cl": tab.shell_b[self.cl, 0], "b2_Cl": tab.shell_b[self.cl, 1],
                   "n1_Cl": tab.shell_n[self.cl, 0]}
        self.ln = torch.nn.ParameterDict({k: torch.nn.Parameter(torch.zeros((), dtype=torch.float64)) for k in self.p0})

    def values(self):
        return {k: self.p0[k] * self.ln[k].exp() for k in self.p0}

    def table(self):
        v = self.values()
        nt = self.tab0.Z[self.cl] - self.tab0.charge[self.cl]                 # 12 electrons in the Cl shells
        b = self.tab0.shell_b.clone()
        n = self.tab0.shell_n.clone()
        b = b.index_put((torch.tensor([self.na, self.cl, self.cl]), torch.tensor([0, 0, 1])),
                        torch.stack([v["b_Na"], v["b1_Cl"], v["b2_Cl"]]))
        n = n.index_put((torch.tensor([self.cl, self.cl]), torch.tensor([0, 1])), torch.stack([v["n1_Cl"], nt - v["n1_Cl"]]))
        return dataclasses.replace(self.tab0, shell_b=b, shell_n=n)

    def penalty(self):
        return sum(p ** 2 for p in self.ln.values())


def ion_potential(tab, s, r_bohr):
    """Potential of a free ion (species index s) at distance r (bohr)."""
    phi = tab.Z[s] / r_bohr
    for t in range(tab.shell_n.shape[1]):
        phi = phi - tab.shell_n[s, t] * (1 - slater_one_center_damp(tab.shell_b[s, t] * r_bohr, 0)[0]) / r_bohr
    return phi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ct_sets")
    ap.add_argument("--walk-eda", default="/mnt/user-data/uploads/research--rsfff/train/nacl/data/eda")
    ap.add_argument("--params", default=str(Path(__file__).resolve().parents[1] / "params" / "ions_wb97mv_tzvpd.json"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--iters", type=int, default=200)
    a = ap.parse_args()
    tab = IonTable.load(a.params)
    frames = load_frames(a.ct_sets, a.walk_eda)
    el = elec_labels(a.ct_sets, a.walk_eda)
    ref = torch.tensor([el[_geom_key(sym, X)] for sym, X, *_ in frames], dtype=torch.float64)
    pos, sp, pairs, fidx, n = flat_pairs(frames, tab)
    nf = len(frames)

    def energies(t):
        e = K.permanent_pair_energy(pos, sp, pairs, t) * KJ
        return torch.zeros(nf, dtype=torch.float64).index_add(0, fidx, e)

    sh = Shells(tab)
    with torch.no_grad():
        e0 = energies(tab)
    err0 = ((e0 - ref) / n)
    opt = torch.optim.LBFGS(sh.parameters(), lr=0.5, max_iter=a.iters, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-10, tolerance_change=1e-12)

    def closure():
        opt.zero_grad()
        err = (energies(sh.table()) - ref) / n
        loss = (err ** 2).mean() + a.lam * sh.penalty()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        t1 = sh.table()
        e1 = energies(t1)
        err1 = (e1 - ref) / n
        r = torch.linspace(2.0, 10.0, 81, dtype=torch.float64)
        dphi = {s: float(((ion_potential(t1, k, r) - ion_potential(tab, k, r)).abs() / ion_potential(tab, k, r).abs()).max())
                for k, s in enumerate(tab.symbols)}
    big = (n > 3)
    rep = {"lam": a.lam, "params_old": {k: float(v) for k, v in sh.p0.items()},
           "params_new": {k: float(v) for k, v in sh.values().items()},
           "per_ion_rms_old": float(err0.pow(2).mean().sqrt()), "per_ion_rms_new": float(err1.pow(2).mean().sqrt()),
           "per_ion_mean_old": float(err0.mean()), "per_ion_mean_new": float(err1.mean()),
           "cluster_per_ion_mean_old": float(err0[big].mean()), "cluster_per_ion_mean_new": float(err1[big].mean()),
           "max_rel_dphi_free_ion_r2-10bohr": dphi}
    print(json.dumps(rep, indent=1))
    if a.out:
        P = json.load(open(a.params))
        out = copy.deepcopy(P)
        v = sh.values()
        out["ions"]["Na"]["shells"][0]["b"] = float(v["b_Na"])
        out["ions"]["Cl"]["shells"][0]["b"] = float(v["b1_Cl"])
        out["ions"]["Cl"]["shells"][1]["b"] = float(v["b2_Cl"])
        out["ions"]["Cl"]["shells"][0]["n"] = float(v["n1_Cl"])
        out["ions"]["Cl"]["shells"][1]["n"] = float(t1.shell_n[sh.cl, 1])
        out["elec_refit"] = rep
        out.pop("channels", None)
        json.dump(out, open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
