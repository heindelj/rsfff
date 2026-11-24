#!/usr/bin/env python
"""The tabulated CT coupling between the free ions' frontier orbitals (``CmmConfig.ct_kernel = "table"``).

    python scripts/cmm_ct_coupling.py --out params/ct_coupling_wb97mv_tzvpd.json
           [--ions Li Na K Cs F Cl Br I] [--donors F Cl Br I] [--xc wb97m-v] [--basis def2-tzvpd]
           [--r0 1.0 --h 0.2 --n 41] [--breath 2 --params IONS.json [--base TABLE.json]] [--pairs "Cl>Na" ...] [--merge PART.json ...]

Per ordered pair D>A and distance R: the two free ions are placed R apart, their determinants are antisymmetrized
(the frozen density, D = 2 C (C^T S C)^-1 C^T over all occupied orbitals of both ions) and the Fock matrix F of that
density is built. With

    i   the donor orbital of D: the sigma member of its highest occupied shell (p sigma towards the partner for a
        halide; 1s for Li+), projected out of the acceptor ion's occupied space, normalized;
    a   the acceptor orbital of A: its lowest s virtual, projected out of the whole occupied space, normalized;

the table holds W = <i | F | a> (Hartree). It is the one-electron coupling of the transfer D -> A with the Pauli
projection and the partner's full potential in it and nothing else: no third ion, no environment factor, no fitted
amplitude. A closed-shell donor pair couples with sqrt(2) W; the model's two-state channel uses W_model^2 = 2 W^2.
An anion's acceptor is unbound: the basis set's discretized continuum function (as in the overlap table).

Also stored per pair, for analysis: ``S`` the overlap of the bare orbitals, ``a_norm`` the squared norm of the
projected acceptor, ``gap`` the orbital gap F_aa - F_ii of the projected orbitals, ``J`` = (ii|aa) and ``K`` = (ia|ia)
(the singlet single-excitation gap is gap - J + 2 K), and per ion the frontier orbital energies.

``--breath K --params IONS.json`` adds ``dW_donor`` and ``dW_acceptor``: the derivative of W with respect to the
donor's or the acceptor's breathing amplitude z_b (electrons; the model's l = 0 response, whose density is
z_b (g_b - n / N) with g_b the species table's breathing shape), on every K-th grid point between ``--breath-range``
(:func:`pair_breath`). An ion without a breathing response has a zero column. ``--base TABLE.json`` takes W and the
rest from an existing table and computes only these.

The VV10 term is left out of the Fock matrix (it is a small, smooth part of the potential). One SCF per ion and one
Fock build per pair and distance: about a second each for the light pairs. ``--pairs`` / ``--merge`` split the work.
Atomic units; R in Angstrom.
"""
import argparse
from pathlib import Path
import math
import json

import numpy as np
from pyscf import dft, gto, scf
from scipy.linalg import eigh  # noqa: F401  (kept for users extending the script)

CHARGE = {"Li": 1, "Na": 1, "K": 1, "Rb": 1, "Cs": 1, "F": -1, "Cl": -1, "Br": -1, "I": -1}
ECP = {"Rb", "Cs", "I"}


def _mol(atoms, basis, charge):
    ecp = {s: basis for s in {a.split()[0] for a in atoms.split(";")} if s in ECP}
    return gto.M(atom=atoms, basis=basis, charge=charge, ecp=ecp or None, verbose=0, unit="Angstrom")


def _rks(mol, xc, level):
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.nlc = False
    mf.grids.level = level
    mf.conv_tol = 1e-10
    return mf


def ion(sym, xc, basis):
    """Free ion: {mol, C, e, nocc, ks (index of the lowest s virtual)}."""
    mol = _mol(f"{sym} 0 0 0", basis, CHARGE[sym])
    mf = _rks(mol, xc, 4)
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"{sym}: SCF not converged")
    C, S = mf.mo_coeff, mol.intor("int1e_ovlp")
    nocc = mol.nelectron // 2
    lab = mol.ao_labels(fmt=False)
    s_ao = [k for k, l in enumerate(lab) if l[2][-1] == "s"]
    sw = lambda c: float(sum(c[k] * (S @ c)[k] for k in s_ao))                      # noqa: E731
    ks = next(k for k in range(nocc, C.shape[1]) if sw(C[:, k]) > 0.8)
    # the highest occupied shell (every occupied orbital within 1e-4 Ha of the HOMO) and its sigma member along z:
    # the combination with the largest <z^2>. Signs fixed on the free ion (largest AO coefficient positive), so that
    # W(R) is continuous where the overlap changes sign.
    top = [k for k in range(nocc) if mf.mo_energy[nocc - 1] - mf.mo_energy[k] < 1e-4]
    T = C[:, top]
    zz = mol.intor("int1e_rr").reshape(3, 3, mol.nao, mol.nao)[2, 2]
    sig = T @ np.linalg.eigh(T.T @ zz @ T)[1][:, -1]
    fix = lambda c: c * np.sign(c[np.argmax(np.abs(c))])                            # noqa: E731
    acc = fix(C[:, ks])
    if len(top) > 1:        # p sigma: positive lobe towards +z, where the partner sits
        lab_p = [k for k, l in enumerate(lab) if l[2][-1] == "p" and l[3] == "z"]
        sig = sig * np.sign(sig[lab_p][np.argmax(np.abs(sig[lab_p]))])
    else:
        sig = fix(sig)
    return dict(sym=sym, mol=mol, C=C, e=mf.mo_energy, nocc=nocc, ks=ks, sig=sig, acc=acc)


def pair_point(d, a, R, xc, basis):
    """Donor ion ``d`` at the origin, acceptor ion ``a`` at R on z: {W, S, a_norm, gap, J, K}."""
    mol = _mol(f"{d['sym']} 0 0 0; {a['sym']} 0 0 {R}", basis, CHARGE[d["sym"]] + CHARGE[a["sym"]])
    nd, na = d["mol"].nao, a["mol"].nao
    if mol.nao != nd + na:
        raise RuntimeError("basis mismatch")
    S = mol.intor("int1e_ovlp")
    pd = lambda c: np.vstack([c, np.zeros((na,) + c.shape[1:])])                    # noqa: E731
    pa = lambda c: np.vstack([np.zeros((nd,) + c.shape[1:]), c])                    # noqa: E731
    Od, Oa = pd(d["C"][:, :d["nocc"]]), pa(a["C"][:, :a["nocc"]])
    acc = pa(a["acc"][:, None])[:, 0]
    # the acceptor sits at +z of the donor (its sigma lobe points there); seen from the acceptor the donor is at -z,
    # which does not matter for an s acceptor
    don = pd(d["sig"][:, None])[:, 0]
    Occ = np.hstack([Od, Oa])
    D = 2 * Occ @ np.linalg.inv(Occ.T @ S @ Occ) @ Occ.T
    mf = _rks(mol, xc, 3)
    F = mf.get_hcore() + mf.get_veff(mol, D)
    proj = lambda C_, v: v - C_ @ np.linalg.solve(C_.T @ S @ C_, C_.T @ S @ v)      # noqa: E731
    ap = proj(Occ, acc)
    a_norm = float(ap @ S @ ap)
    ap = ap / np.sqrt(a_norm)
    ip = proj(Oa, don)
    ip = ip / np.sqrt(ip @ S @ ip)
    vj, _ = scf.hf.get_jk(mol, [np.outer(ip, ip), np.outer(ip, ap)], hermi=0)
    return dict(W=float(ip @ F @ ap), S=float(don @ S @ acc), a_norm=a_norm,
                gap=float(ap @ F @ ap - ip @ F @ ip), J=float(ap @ vj[0] @ ap), K=float(ip @ vj[1] @ ap))


def _scaled_basis(sym, basis, lam):
    return [[sh[0]] + [[row[0] * lam * lam] + list(row[1:]) for row in sh[1:]] for sh in gto.basis.load(basis, sym)]


GHOSTS = (1.6, 0.6)                  # exponent factors of the two ghost copies that carry the deformed orbitals


def _breath_shape(parents, r):
    """g_b(r) of the species table's breathing parent: sum_t sum_n w_tn N_n r^n e^(-b_t r), each term normalized."""
    out = np.zeros_like(r)
    for t in parents["breath"]:
        b = float(t["b"])
        for k, w in enumerate(t["w"]):
            if w:
                out += w * b ** (k + 3) / (4 * math.pi * math.factorial(k + 2)) * r ** k * np.exp(-b * r)
    return out


def breath_orbitals(x, basis, parents):
    """The ion's breathing response in its orbitals: chi_i = g_b / (2 n) phi_i for every occupied phi_i, n the ion's
    density, so that phi_i + z chi_i changes the density by z g_b to first order (z (g_b - n / N) once normalized:
    the model's breathing response). Least-squares coefficients (nao (1 + len(GHOSTS)), nocc) over the ion's basis and
    its ghost copies, and the relative L2 error of the density change they give. None for an ion that does not
    breathe."""
    if not float((parents or {}).get("a0") or 0.0) or not parents.get("breath"):
        return None
    sym = x["sym"]
    atoms = f"{sym}@1 0 0 0; " + "; ".join(f"ghost-{sym}{k} 0 0 0" for k in range(len(GHOSTS)))
    bas = {sym + "@1": basis}
    bas.update({f"ghost-{sym}{k}": _scaled_basis(sym, basis, math.sqrt(q)) for k, q in enumerate(GHOSTS)})
    mol = gto.M(atom=atoms, basis=bas, ecp=({sym + "@1": basis} if sym in ECP else None), charge=CHARGE[sym], verbose=0)
    g = dft.gen_grid.Grids(x["mol"])
    g.level = 6
    g.build()
    ao, w, r = dft.numint.eval_ao(mol, g.coords), g.weights, np.linalg.norm(g.coords, axis=1)
    phi = ao[:, :x["mol"].nao] @ x["C"][:, :x["nocc"]]
    n = 2 * (phi ** 2).sum(1)
    gb = _breath_shape(parents, r)
    tgt = 0.5 * np.where(n > 1e-9, gb / np.maximum(n, 1e-9), 0.0)[:, None] * phi
    ev, U = np.linalg.eigh(mol.intor("int1e_ovlp"))
    keep = ev > 1e-7
    c = U[:, keep] @ ((U[:, keep].T @ (ao.T @ (w[:, None] * tgt))) / ev[keep][:, None])
    dn = 4 * (phi * (ao @ c)).sum(1)
    return c, float(np.sqrt((w * (dn - gb) ** 2).sum() / (w * gb ** 2).sum()))


def pair_breath(d, a, R, xc, basis, dz=-0.05):
    """(dW / dz_D, dW / dz_A, W) at R: the change of W = <i|F|a> with the donor's or the acceptor's breathing
    amplitude, the ion's occupied orbitals deformed to phi + z chi (``d["chi"]``, :func:`breath_orbitals`; None: 0).
    Everything follows the deformed ion: the frozen density and its Fock matrix, the donor orbital (the same
    combination of the ion's orbitals), its projection out of the acceptor's occupied space and the acceptor
    orbital's out of the whole occupied space, so W(z) = <i|F - E_i|a>. The acceptor's virtual orbital itself is not
    deformed. W is linear in an ion's shape: the derivative is the coupling of its breathing function to the other
    ion, and it belongs with the variational amplitude. One-sided difference (``dz`` < 0: a contraction)."""
    sd, sa = d["sym"], a["sym"]
    ng = len(GHOSTS)
    gD, gA = d.get("chi") is not None, a.get("chi") is not None
    if not (gD or gA):
        return 0.0, 0.0, None
    atoms = f"{sd}@1 0 0 0; {sa}@2 0 0 {R}"
    bas = {sd + "@1": basis, sa + "@2": basis}
    for s, on, z, k0 in ((sd, gD, 0.0, 0), (sa, gA, R, ng)):
        if on:
            atoms += "".join(f"; ghost-{s}{k0 + k} 0 0 {z}" for k in range(ng))
            bas.update({f"ghost-{s}{k0 + k}": _scaled_basis(s, basis, math.sqrt(q)) for k, q in enumerate(GHOSTS)})
    ecp = {s + t: basis for s, t in ((sd, "@1"), (sa, "@2")) if s in ECP}
    mol = gto.M(atom=atoms, basis=bas, ecp=ecp or None, charge=CHARGE[sd] + CHARGE[sa], verbose=0, unit="Angstrom")
    mol0 = _mol(f"{sd} 0 0 0; {sa} 0 0 {R}", basis, CHARGE[sd] + CHARGE[sa])
    nd, na = d["mol"].nao, a["mol"].nao
    n = mol.nao
    if n != (1 + ng * gD) * nd + (1 + ng * gA) * na:
        raise RuntimeError("basis mismatch")
    S = mol.intor("int1e_ovlp")
    off = {"d": (0, nd, nd + na), "a": (nd, na, nd + na + ng * nd * gD)}

    def put(c, ion):                    # coefficients over the ion's basis (+ its ghosts) in the dimer's
        o, m, g0 = off[ion]
        v = np.zeros((n,) + c.shape[1:])
        v[o:o + m] = c[:m]
        if c.shape[0] > m:
            v[g0:g0 + ng * m] = c[m:]
        return v
    g = dft.gen_grid.Grids(mol0)        # no grid on the ghosts (they sit on the nuclei)
    g.level = 3
    g.build()
    mf = _rks(mol, xc, 3).density_fit()     # the fitting error cancels in the difference
    mf.grids = g
    h1 = mf.get_hcore()
    proj = lambda C_, v: v - C_ @ (np.linalg.pinv(C_.T @ S @ C_, rcond=1e-10) @ (C_.T @ S @ v))     # noqa: E731
    nrm = lambda v: v / np.sqrt(v @ S @ v)                                                         # noqa: E731
    a0 = put(a["acc"][:, None], "a")[:, 0]
    comb = d["C"][:, :d["nocc"]].T @ d["mol"].intor("int1e_ovlp") @ d["sig"]     # the donor orbital in the ion's

    def W(zD, zA):
        Od = put(d["C"][:, :d["nocc"]], "d") + (zD * put(d["chi"][0], "d") if zD else 0.0)
        Oa = put(a["C"][:, :a["nocc"]], "a") + (zA * put(a["chi"][0], "a") if zA else 0.0)
        Occ = np.hstack([Od, Oa])
        D = 2 * Occ @ np.linalg.pinv(Occ.T @ S @ Occ, rcond=1e-12) @ Occ.T
        F = h1 + mf.get_veff(mol, D)
        return float(nrm(proj(Oa, Od @ comb)) @ F @ nrm(proj(Occ, a0)))
    w0 = W(0.0, 0.0)
    return ((W(dz, 0.0) - w0) / dz if gD else 0.0), ((W(0.0, dz) - w0) / dz if gA else 0.0), w0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ions", nargs="+", default=["Li", "Na", "K", "Cs", "F", "Cl", "Br", "I"])
    ap.add_argument("--donors", nargs="+", default=None, help="donor species (default: the anions among --ions)")
    ap.add_argument("--pairs", nargs="+", default=None, help='only these ordered pairs, e.g. "Cl>Na"')
    ap.add_argument("--merge", nargs="+", default=None, help="merge part files into --out instead of computing")
    ap.add_argument("--xc", default="wb97m-v")
    ap.add_argument("--basis", default="def2-tzvpd")
    ap.add_argument("--r0", type=float, default=1.0)
    ap.add_argument("--h", type=float, default=0.2)
    ap.add_argument("--n", type=int, default=41)
    ap.add_argument("--breath", type=int, default=0, metavar="K",
                    help="also the derivatives dW/dz_b with the donor's and the acceptor's breathing amplitude, on "
                         "every K-th grid point (0: none); needs --params")
    ap.add_argument("--params", default=None, help="species table with the breathing parents (ions_*.json)")
    ap.add_argument("--breath-range", type=float, nargs=2, default=(1.4, 7.4), metavar=("R0", "R1"),
                    help="compute the derivatives between these distances; inside R0 the value at R0, beyond R1 the "
                         "logarithmic derivative at R1 times W")
    ap.add_argument("--base", default=None, help="an existing table: take W and the rest from it (with --breath)")
    a = ap.parse_args()
    if a.merge:
        parts = [json.load(open(p)) for p in a.merge]
        out = parts[0]
        for p in parts[1:]:
            if any(p[k] != out[k] for k in ("level", "r0", "h", "n")):
                raise SystemExit("part files differ in level or grid")
            for k in ("W", "S", "a_norm", "gap", "J", "K", "ions", "dW_donor", "dW_acceptor"):
                if k in p:
                    out.setdefault(k, {}).update(p[k])
            for k in ("breath_every", "breath_note", "breath_params"):
                if k in p:
                    out[k] = p[k]
        out["symbols"] = sorted(out["ions"], key=lambda s: a.ions.index(s) if s in a.ions else 99)
        json.dump(out, open(a.out, "w"))
        print(f"{a.out}: {len(out['W'])} pairs")
        return
    donors = a.donors or [s for s in a.ions if CHARGE[s] < 0]
    pairs = a.pairs or [f"{d}>{b}" for d in donors for b in a.ions]
    need = sorted({s for p in pairs for s in p.split(">")}, key=a.ions.index)
    ions = {s: ion(s, a.xc, a.basis) for s in need}
    r = a.r0 + a.h * np.arange(a.n)
    out = dict(level=f"{a.xc}/{a.basis} (no VV10 in the Fock matrix)", symbols=need, r0=a.r0, h=a.h, n=a.n,
               note="W = <i|F|a>, Hartree: donor's highest occupied sigma (projected out of the acceptor ion's occupied "
                    "space) with the acceptor's lowest s virtual (projected out of the occupied space), F the Fock "
                    "matrix of the frozen dimer; W_model^2 = 2 W^2 (scripts/cmm_ct_coupling.py)",
               ions={s: dict(eps_donor=float(v["e"][v["nocc"] - 1]), eps_acceptor=float(v["e"][v["ks"]])) for s, v in ions.items()},
               W={}, S={}, a_norm={}, gap={}, J={}, K={})
    base = json.load(open(a.base)) if a.base else None
    if base is not None and any(base[k] != out[k] for k in ("level", "r0", "h", "n")):
        raise SystemExit("--base: another level or grid")
    if a.breath:
        if not a.params:
            raise SystemExit("--breath needs --params")
        par = json.load(open(a.params))["ions"]
        for s, v in ions.items():
            v["chi"] = breath_orbitals(v, a.basis, par[s].get("parents"))
            if v["chi"] is not None:
                print(f"{s}: breathing density represented to {v['chi'][1]:.3f} (relative L2)", flush=True)
        out["breath_every"] = a.breath
        out["breath_params"] = Path(a.params).name
        out["breath_note"] = ("dW_donor, dW_acceptor: dW / dz_b (Hartree per electron) with the donor's / the acceptor's "
                              "breathing amplitude, the model's l = 0 response with the shapes of breath_params "
                              "(z_b > 0 an expansion); on the grid points 0, K, 2K, ...; zero for an ion that does "
                              "not breathe")
    for p in pairs:
        d, b = p.split(">")
        if base is not None:
            for k in ("W", "S", "a_norm", "gap", "J", "K"):
                out[k][p] = base[k][p]
        else:
            rows = [pair_point(ions[d], ions[b], float(x), a.xc, a.basis) for x in r]
            for k in ("W", "S", "a_norm", "gap", "J", "K"):
                out[k][p] = [row[k] for row in rows]
        w = np.array(out["W"][p])
        if a.breath:
            rc = r[::a.breath]
            wc = w[::a.breath]
            on = [k for k, x in enumerate(rc) if a.breath_range[0] - 1e-9 <= x <= a.breath_range[1] + 1e-9]
            rb = {k: pair_breath(ions[d], ions[b], float(rc[k]), a.xc, a.basis) for k in on}
            if any(v[2] is not None and abs(v[2] - wc[k]) > 2e-4 for k, v in rb.items()):
                raise RuntimeError(f"{p}: the enlarged-basis W differs from the table's")
            for q, name in enumerate(("dW_donor", "dW_acceptor")):
                col = np.zeros(len(rc))
                col[on] = [rb[k][q] for k in on]
                col[:on[0]] = col[on[0]]
                col[on[-1] + 1:] = col[on[-1]] / wc[on[-1]] * wc[on[-1] + 1:]
                out.setdefault(name, {})[p] = [float(v) for v in col]
        print(f"{p}: max |W| {np.abs(w).max():.4f} Ha at {r[np.argmax(np.abs(w))]:.2f} A", flush=True)
        json.dump(out, open(a.out, "w"))


if __name__ == "__main__":
    main()
