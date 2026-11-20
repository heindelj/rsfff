#!/usr/bin/env python
"""Frontier orbitals of the free ions and their radial shapes: what the CT coupling of a pair should decay like.

    python scripts/cmm_frontier_shapes.py --out params/frontier_wb97mv_tzvpd.json [--params TABLE.json]
                                          [--ions Li Na K Cs F Cl Br I] [--xc wb97m-v] [--basis def2-tzvpd]

Per ion, from one SCF: the acceptor orbital of a cation (its lowest s virtual, the ns orbital of the atom) and the
donor orbital of an anion (its np HOMO, sigma towards the partner), with

    eps, zeta_inf = sqrt(2 |eps|)     the orbital energy and the asymptotic decay of the amplitude
    zeta_loc(r)   = -d ln|phi| / dr   the local decay at a few radii (how far the orbital is from one exponential)

and fits of the radial amplitude on the valence lobe (outside the last radial node, to where it has fallen to 2e-4 of
its maximum; inside the node the shape is zero: the core oscillations hardly overlap a neighbour),

    phi(r) = sum_t e^{-zeta_t r} sum_k c_tk r^k   for r > r_node      (times cos(theta) for the p orbital)

with two fitted exponents (at least a factor 1.25 apart), "k01": k = 0, 1 each (4 coefficients), "k012": k = 0, 1, 2
(6). Least squares of the relative error (floor 2 % of the lobe's maximum), linear in c, the exponents scanned. One
exponential does not do: the cations' amplitude rises to a maximum at 2 - 4 bohr and reaches zeta_inf only beyond
8 - 10 bohr, the halides' has a compact lobe (zeta about 1) and a weak diffuse tail.

Per cation-anion pair: the two-centre overlap S(R) of the two orbitals (s with p sigma), exact (the ions' own basis
sets) and from each pair of fitted shapes, 2 - 8 A. The model's coupling is W^2 = qD qA x kernel(R): the kernel
should have the shape of S(R)^2. Errors are given relative to the largest S of the pair (S passes through zero at
short range for the large ions) and, beyond the maximum of S, relative to S itself. With ``--params``: also S(R)
from the species table's frontier density shapes, amplitude = sqrt(density). Atomic units; R in Angstrom.

``--table OUT.json`` writes instead the overlap table of the model's ``ct_kernel = "frontier"``
(rsfff.ff.cmm.ct_overlap): for every ordered pair D>A of the ions the overlap S(R) of D's donor orbital (its
highest occupied shell; p sigma towards the partner, 1s for Li+) with A's acceptor orbital (its lowest s virtual), on
a uniform grid, straight from the orbitals (no fitted shapes). An anion's acceptor is unbound (eps > 0): a
discretized continuum function of the basis set, the same one the labels of that basis have.

    python scripts/cmm_frontier_shapes.py --table params/ct_overlap_wb97mv_tzvpd.json
"""
import argparse
import itertools
import json
import math

import numpy as np
from pyscf import dft, gto

CHARGE = {"Li": 1, "Na": 1, "K": 1, "Rb": 1, "Cs": 1, "F": -1, "Cl": -1, "Br": -1, "I": -1}
ECP = {"Rb", "Cs", "I"}
BOHR = 0.529177210903
R_PAIR = np.arange(2.0, 8.01, 0.5)                              # A
R_GRID = np.exp(np.linspace(math.log(0.02), math.log(40.0), 900))
FLOOR = 0.02                                                    # relative-error floor of the fit (of the lobe's max)
TAIL = 2e-4                                                     # the fit ends where |phi| < TAIL x the lobe's maximum
FITS = {"k01": dict(powers=(0, 1)), "k012": dict(powers=(0, 1, 2))}
SCAN = np.arange(0.25, 5.0001, 0.025)                           # exponents tried
RATIO = 1.25                                                    # smallest ratio of the two exponents


def _mol(atoms, basis, charge):
    ecp = {s: basis for s in {a.split()[0] for a in atoms.split(";")} if s in ECP}
    return gto.M(atom=atoms, basis=basis, charge=charge, ecp=ecp or None, verbose=0)


def frontier(sym, xc, basis):
    """{mol, coef (nao,), l, eps}: the cation's lowest s virtual / the anion's p_z HOMO."""
    mol = _mol(f"{sym} 0 0 0", basis, CHARGE[sym])
    mf = dft.RKS(mol)
    mf.xc = xc
    if xc.lower().endswith("-v"):
        mf.nlc = "vv10"
    mf.grids.level = 4
    mf.conv_tol = 1e-10
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"{sym}: SCF not converged")
    C, e, S = mf.mo_coeff, mf.mo_energy, mol.intor("int1e_ovlp")
    lab = mol.ao_labels(fmt=False)                              # (atom, symbol, nl, m)
    nocc = mol.nelectron // 2

    def weight(c, sel):                                          # Mulliken weight of the selected AOs in orbital c
        return float(sum(c[i] * (S @ c)[i] for i in sel))
    if CHARGE[sym] > 0:
        s_ao = [i for i, x in enumerate(lab) if x[2][-1] == "s"]
        k = next(k for k in range(nocc, C.shape[1]) if weight(C[:, k], s_ao) > 0.9)
        return dict(mol=mol, coef=C[:, k], l=0, eps=float(e[k]), role="acceptor", index=k - nocc)
    pz = [i for i, x in enumerate(lab) if x[2][-1] == "p" and x[3] == "z"]
    P = C[:, nocc - 3:nocc]                                       # the degenerate p shell: rotate p_z out of it
    M = P[pz].T @ (S @ P)[pz]
    c = P @ np.linalg.eigh((M + M.T) / 2)[1][:, -1]
    if weight(c, pz) < 0.99:
        raise RuntimeError(f"{sym}: the three highest occupied orbitals are not a p shell")
    return dict(mol=mol, coef=c, l=1, eps=float(e[nocc - 1]), role="donor", index=0)


def orbitals(sym, xc, basis):
    """{"donor": {mol, coef, l, eps}, "acceptor": {...}}: the highest occupied shell (its p_z member, or the s
    orbital) and the lowest s virtual of the ion."""
    mol = _mol(f"{sym} 0 0 0", basis, CHARGE[sym])
    mf = dft.RKS(mol)
    mf.xc = xc
    if xc.lower().endswith("-v"):
        mf.nlc = "vv10"
    mf.grids.level = 4
    mf.conv_tol = 1e-10
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"{sym}: SCF not converged")
    C, e, S = mf.mo_coeff, mf.mo_energy, mol.intor("int1e_ovlp")
    lab = mol.ao_labels(fmt=False)
    nocc = mol.nelectron // 2
    sel = {t: [i for i, x in enumerate(lab) if x[2][-1] == t] for t in "sp"}
    pz = [i for i in sel["p"] if lab[i][3] == "z"]

    def weight(c, idx):
        return float(sum(c[i] * (S @ c)[i] for i in idx))
    k = next(k for k in range(nocc, C.shape[1]) if weight(C[:, k], sel["s"]) > 0.9)
    acc = dict(mol=mol, coef=C[:, k], l=0, eps=float(e[k]))
    if weight(C[:, nocc - 1], sel["s"]) > 0.9:
        don = dict(mol=mol, coef=C[:, nocc - 1], l=0, eps=float(e[nocc - 1]))
    else:
        P = C[:, nocc - 3:nocc]
        M = P[pz].T @ (S @ P)[pz]
        c = P @ np.linalg.eigh((M + M.T) / 2)[1][:, -1]
        if weight(c, pz) < 0.99:
            raise RuntimeError(f"{sym}: the three highest occupied orbitals are not a p shell")
        don = dict(mol=mol, coef=c, l=1, eps=float(e[nocc - 1]))
    return dict(donor=don, acceptor=acc)


def overlap_table(orbs, basis, r0=0.5, h=0.02, r1=10.0):
    """{"D>A": [S(R)]} on the grid r0, r0 + h, ... (Angstrom) for every ordered pair: the donor orbital of D at the
    origin (p_z towards the partner) with the acceptor orbital of A at (0, 0, R); the sign such that S > 0 where |S|
    is largest."""
    R = r0 + h * np.arange(int(round((r1 - r0) / h)) + 1)
    out = {}
    for a, oa in orbs.items():
        for b, ob in orbs.items():
            mb = ob["acceptor"]["mol"].copy()
            S = np.empty(len(R))
            for n, r in enumerate(R):
                mb.set_geom_(f"{b} 0 0 {r:.10f}", unit="Angstrom")
                S[n] = oa["donor"]["coef"] @ gto.intor_cross("int1e_ovlp", oa["donor"]["mol"], mb) @ ob["acceptor"]["coef"]
            out[f"{a}>{b}"] = (S * np.sign(S[np.argmax(np.abs(S))])).tolist()
    return R, out


def write_table(a):
    orbs = {s: orbitals(s, a.xc, a.basis) for s in a.ions}
    for s, o in orbs.items():
        print(f"{s:2s} donor l = {o['donor']['l']} eps {o['donor']['eps']:+.4f}   acceptor eps {o['acceptor']['eps']:+.4f}"
              f"{'' if o['acceptor']['eps'] < 0 else '  (unbound)'}", flush=True)
    R, S = overlap_table(orbs, a.basis)
    json.dump(dict(level=f"{a.xc}/{a.basis}", symbols=list(a.ions), r0=float(R[0]), h=float(R[1] - R[0]), n=len(R),
                   donor={s: dict(l=o["donor"]["l"], eps=o["donor"]["eps"]) for s, o in orbs.items()},
                   acceptor={s: dict(l=0, eps=o["acceptor"]["eps"], bound=o["acceptor"]["eps"] < 0)
                             for s, o in orbs.items()}, S=S), open(a.table, "w"))
    print("wrote", a.table)


def amplitude(fr, r):
    """The radial amplitude on the +z axis (phi = R(r) for s, R(r) cos(theta) for p_z), positive in the tail."""
    y = dft.numint.eval_ao(fr["mol"], np.outer(r, [0.0, 0.0, 1.0])) @ fr["coef"]
    return y * np.sign(y[np.argmin(np.abs(r - 6.0))])


def lobe(r, y):
    """(i0, i1): the valence lobe, from the last radial node to where the amplitude has fallen to TAIL of its
    maximum there (beyond, the Gaussian basis decays too fast)."""
    flips = [i for i in np.nonzero(np.sign(y[1:]) != np.sign(y[:-1]))[0]          # not the noise of the far tail
             if r[i] < 12.0 and np.abs(y[i + 1:]).max() > 1e-4]
    i0 = int(flips[-1]) + 1 if flips else 0
    peak = i0 + int(np.argmax(np.abs(y[i0:])))
    below = np.nonzero(np.abs(y[peak:]) < TAIL * abs(y[peak]))[0]
    return i0, (peak + int(below[0]) if len(below) else len(r) - 1)


def evaluate(shape, r):
    v = sum(c * r ** k * np.exp(-z * r) for z, row in zip(shape["zeta"], shape["c"])
            for k, c in zip(shape["powers"], row))
    return np.where(r >= shape.get("cut", 0.0), v, 0.0)


def fit(r, y, powers):
    """The best shape of the family on the valence lobe: {zeta, powers, c, cut, error, range, tail_ok}."""
    i0, i1 = lobe(r, y)
    rr, yy = r[i0:i1 + 1], y[i0:i1 + 1]
    w = np.sqrt(np.gradient(rr)) / (np.abs(yy) + FLOOR * np.abs(yy).max())
    best = None
    for zs in itertools.combinations(SCAN, 2):
        if zs[1] < RATIO * zs[0]:
            continue
        A = np.stack([rr ** k * np.exp(-z * rr) for z in zs for k in powers], 1)
        c, *_ = np.linalg.lstsq(A * w[:, None], yy * w, rcond=None)
        res = float(np.linalg.norm((A @ c - yy) * w))
        if best is None or res < best[0]:
            best = (res, zs, c)
    res, zs, c = best
    shape = dict(zeta=[float(z) for z in zs], powers=list(powers), c=c.reshape(len(zs), len(powers)).tolist(),
                 cut=float(rr[0]) if i0 else 0.0)
    f = evaluate(shape, rr)
    far = evaluate(shape, np.linspace(rr[-1], 30.0, 200))
    shape.update(error=float(np.sqrt(np.mean(((f - yy) / (np.abs(yy) + FLOOR * np.abs(yy).max())) ** 2))),
                 range=[float(rr[0]), float(rr[-1])],
                 tail_ok=bool((far > 0).all() and (np.diff(far) < 0).all()))       # positive and decaying beyond
    return shape


def fukui_amplitude(k, b):
    """sqrt of the normalized density r^k e^{-b r}: the amplitude a density shape of the table stands for."""
    norm = b ** (k + 3) / (4 * math.pi * math.factorial(k + 2))
    return dict(zeta=[b / 2], powers=[k / 2], c=[[math.sqrt(norm * 4 * math.pi)]], density=True)


def pair_overlaps(fc, fa, shapes_c, shapes_a, basis):
    """{name: [S(R)]}: exact, and from each pair of shapes (cation s, anion p sigma), on the dimer's grid."""
    out = {"exact": []}
    out.update({n: [] for n in shapes_c})
    sc, sa = fc["mol"].atom_symbol(0), fa["mol"].atom_symbol(0)
    n1 = fc["mol"].nao
    for R in R_PAIR:
        mol = _mol(f"{sc} 0 0 0; {sa} 0 0 {R}", basis, 0)
        out["exact"].append(abs(float(fc["coef"] @ mol.intor("int1e_ovlp")[:n1, n1:] @ fa["coef"])))
        g = dft.gen_grid.Grids(mol)
        g.level = 5
        g.build()
        d = g.coords - np.array([0.0, 0.0, R / BOHR])
        ra, rb = np.linalg.norm(g.coords, axis=1), np.linalg.norm(d, axis=1)
        cos = -d[:, 2] / np.maximum(rb, 1e-12)
        for n in shapes_c:
            # an s orbital is R / sqrt(4 pi) only for the density shapes (the fits are the amplitude itself)
            ya = evaluate(shapes_c[n], np.maximum(ra, 1e-12))
            yb = evaluate(shapes_a[n], np.maximum(rb, 1e-12))
            if shapes_c[n].get("density"):
                ya, yb = ya / math.sqrt(4 * math.pi), yb * math.sqrt(3 / (4 * math.pi))
            out[n].append(abs(float(g.weights @ (ya * yb * cos))))
    return out


def local_decay(R, y):
    return (-np.diff(np.log(np.asarray(y))) / (np.diff(R) / BOHR)).tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="the shapes and their fits")
    ap.add_argument("--table", default=None, help="write the model's overlap table (every ordered pair) instead")
    ap.add_argument("--params", default=None, help="species table: compare with its frontier (Fukui) shapes")
    ap.add_argument("--ions", nargs="*", default=["Li", "Na", "K", "Cs", "F", "Cl", "Br", "I"])
    ap.add_argument("--xc", default="wb97m-v")
    ap.add_argument("--basis", default="def2-tzvpd")
    a = ap.parse_args()
    if a.table:
        return write_table(a)
    if not a.out:
        ap.error("--out or --table")
    tab = json.load(open(a.params))["ions"] if a.params else {}
    res = {"level": f"{a.xc}/{a.basis}", "R_pair_ang": R_PAIR.tolist(), "ions": {}, "pairs": {}}
    fr, shapes = {}, {}
    for sym in a.ions:
        f = fr[sym] = frontier(sym, a.xc, a.basis)
        y = amplitude(f, R_GRID)
        zinf = math.sqrt(2 * abs(f["eps"]))
        i0, i1 = lobe(R_GRID, y)
        radii = [3.0, 4.0, 6.0, 8.0, 10.0]
        zl = [float(-(np.log(abs(np.interp(q + 0.05, R_GRID, y))) - np.log(abs(np.interp(q - 0.05, R_GRID, y)))) / 0.1)
              for q in radii]
        shapes[sym] = {n: fit(R_GRID, y, **kw) for n, kw in FITS.items()}
        ion = dict(role=f["role"], l=f["l"], eps=f["eps"], zeta_inf=zinf, virtual_index=f["index"],
                   last_node=float(R_GRID[i0]), peak=float(R_GRID[i0 + int(np.argmax(np.abs(y[i0:i1 + 1])))]),
                   zeta_local={f"{q:g}": v for q, v in zip(radii, zl)}, fits=shapes[sym])
        if sym in tab and "frontier" in tab[sym]:
            t = tab[sym]["frontier"][f["role"]]
            ion["table"] = dict(k=t["k"], b=t["b"], source=t.get("source", "outer shell"))
            shapes[sym]["table"] = fukui_amplitude(t["k"], t["b"])
        res["ions"][sym] = ion
        print(f"{sym:2s} {f['role']:8s} eps {f['eps']:+.4f}  zeta_inf {zinf:.3f}  node {ion['last_node']:.2f}  peak "
              f"{ion['peak']:.2f}  local decay at 3/4/6/8/10 bohr " + " ".join(f"{v:.2f}" for v in zl) + "  | fit error "
              + ", ".join(f"{n} {100 * s['error']:.1f} %{'' if s['tail_ok'] else ' (tail!)'}"
                          for n, s in shapes[sym].items() if "error" in s), flush=True)
    for c in [s for s in a.ions if CHARGE[s] > 0]:
        for an in [s for s in a.ions if CHARGE[s] < 0]:
            names = [n for n in shapes[c] if n in shapes[an]]
            S = pair_overlaps(fr[c], fr[an], {n: shapes[c][n] for n in names}, {n: shapes[an][n] for n in names}, a.basis)
            ex = np.array(S["exact"])
            out = R_PAIR >= R_PAIR[int(np.argmax(ex))]             # from the maximum of S outwards
            p = dict(S=S, R_max=float(R_PAIR[int(np.argmax(ex))]),
                     local_decay={n: local_decay(R_PAIR[out], np.array(v)[out]) for n, v in S.items()},
                     error_of_max={n: float(np.max(np.abs(np.array(S[n]) - ex)) / ex.max()) for n in names},
                     error_outer={n: float(np.max(np.abs(np.array(S[n]) / ex - 1)[out])) for n in names})
            res["pairs"][f"{c}-{an}"] = p
            print(f"{c}-{an}: S largest at {p['R_max']:.1f} A; error over 2-8 A / S_max, and beyond the maximum / S: "
                  + ", ".join(f"{n} {100 * p['error_of_max'][n]:.1f} %, {100 * p['error_outer'][n]:.0f} %"
                              for n in names), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
