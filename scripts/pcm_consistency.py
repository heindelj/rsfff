"""QM / force-field consistency of the continuum (the cpcm plan's S5 + S6 on the complete model's frozen densities).

Runs the pyscf ALMO-EDA in C-PCM (``rsfff.qcgen.almo``) and the force-field continuum (``rsfff.ff.complete.solvent``)
on the same ions with the same cavity, and compares what does not depend on any trained parameter:

1. the cavity (points, exponents, switching);
2. the frozen surface potential: the ion table's nuclei + Slater shells (smeared) against the superposed solvated
   fragment densities (the QM "sum" state) and the antisymmetrized frozen density;
3. the continuum energies of the frozen decomposition: G_cplx[V_A], G_own[V_A], e_cav, e_scr (cls_elec's screened
   cross term) -- the FF's e_cav / e_scr are what the model's cls_elec and frozen terms are trained around;
4. the round trip: the FF's equilibrated frozen-state charges as pyscf Gaussian MM charges (radii = 1/zeta) around
   the QM frozen density, checking q . V_QM against pyscf's own embedding energy (the export convention).

    python scripts/pcm_consistency.py [--R 2.6] [--eps 78.3553] [--basis def2-tzvpd] [--ng 302] [--params ...]
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from rsfff.ff import cpcm
from rsfff.ff.complete.solvent import permanent_surface_potential
from rsfff.ff.ionct.params import IonTable
from rsfff.qcgen.almo import Fragment, Continuum, almo_eda, KJ, surface_potential, pcm_energy

ORDER = {110: 17, 194: 23, 302: 29, 590: 41}


def main(a):
    tab = IonTable.load(a.params)
    sym = ["Na", "Cl"]
    pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, a.R]])
    pcm = Continuum(eps=a.eps, lebedev_order=ORDER[a.ng])
    spec = cpcm.CPCMSpec(eps=a.eps, ng=a.ng)
    frags = [Fragment([s], pos[k:k + 1], charge=+1 if s == "Na" else -1) for k, s in enumerate(sym)]
    print(f"QM: {a.basis} wB97M-V, eps {a.eps}, {a.ng} points/sphere ...", flush=True)
    out = almo_eda(frags, basis=a.basis, pcm=pcm, verbose=0)
    sv, st = out["solvent"], out["states"]
    t = out["terms"]
    print({k: round(v, 3) for k, v in t.items()})

    # --- FF side
    X = torch.tensor(pos / cpcm.BOHR_ANG)
    sp = torch.tensor([tab.symbols.index(s) for s in sym])
    cav = cpcm.build_cavity(X, cpcm.sphere_radii(sym, spec), spec)
    print("\n1. cavity: points", cav.npoints, "vs", len(sv["grid_coords"]),
          "max |dr|", np.abs(cav.coords.numpy() - sv["grid_coords"]).max(),
          "max |dzeta|", np.abs(cav.zeta.numpy() - sv["charge_exp"]).max(),
          "max |dF|", np.abs(cav.F.numpy() - sv["switch_fun"]).max())
    S = cpcm.s_matrix(cav)
    Va = permanent_surface_potential(cav, X, sp, tab)              # (P, 2)
    V0 = Va.sum(1)
    f = spec.f
    Vq_sup, Vq_frz = sv["V_sup"], sv["V_frz"]
    d = V0.numpy() - Vq_sup
    print("2. frozen surface potential (FF shells vs QM superposed fragments): rms %.2e max %.2e (V rms %.3f);"
          % (np.sqrt((d ** 2).mean()), np.abs(d).max(), np.sqrt((Vq_sup ** 2).mean())),
          "vs antisymmetrized: rms %.2e" % np.sqrt(((V0.numpy() - Vq_frz) ** 2).mean()))
    for k, s in enumerate(sym):
        dk = Va[:, k].numpy() - sv["V_frag_cplx"][k]
        print(f"   {s}: per-ion rms {np.sqrt((dk ** 2).mean()):.2e} max {np.abs(dk).max():.2e}")
    # --- energies
    q_ff, G_ff = cpcm.solve(S, V0, f)
    G_frag_ff = [float(cpcm.energy(S, Va[:, k], f)) for k in range(2)]
    G_own_ff = []
    for k in range(2):
        c1 = cpcm.build_cavity(X[k:k + 1], cpcm.sphere_radii([sym[k]], spec), spec)
        G_own_ff.append(float(cpcm.energy(cpcm.s_matrix(c1), permanent_surface_potential(c1, X[k:k + 1], sp[k:k + 1], tab)[:, 0], f)))
    e_cav_ff = sum(G_frag_ff) - sum(G_own_ff)
    e_scr_ff = float(G_ff) - sum(G_frag_ff)
    print("3. continuum energies (kJ/mol)            FF        QM")
    for k, s in enumerate(sym):
        print(f"   G_cplx[{s}]        {G_frag_ff[k] * KJ:10.3f} {sv['G_frag_cplx'][k] * KJ:10.3f}")
        print(f"   G_own[{s}]         {G_own_ff[k] * KJ:10.3f} {sv['G_frag_own'][k] * KJ:10.3f}")
    print(f"   G_cplx[frozen]     {float(G_ff) * KJ:10.3f} {sv['G_sup'] * KJ:10.3f}  (QM superposed; antisymmetrized {sv['G_frz'] * KJ:.3f})")
    print(f"   e_cav              {e_cav_ff * KJ:10.3f} {t['cav']:10.3f}")
    print(f"   e_scr (cls_elec)   {e_scr_ff * KJ:10.3f} {t['cls_elec_scr']:10.3f}")
    print(f"   sum q (frozen)     {float(q_ff.sum()):10.4f} {sv['q_frz'].sum():10.4f}")
    # --- 4. round trip: FF charges around the QM frozen density
    from pyscf import qmmm
    mol = st["mol"]
    sys_ = st["system"]
    coords, q, radii = cpcm.to_pyscf_mm(cav, q_ff)
    mf = sys_.mf.__class__(mol, xc=sys_.mf.xc)
    mf = qmmm.mm_charge(mf, coords, q, radii, unit="Bohr")
    D = st["D_frz"]
    h_mm = mf.get_hcore() - sys_.mf.get_hcore()                    # the charges' operator on the electrons
    e_elec = float(np.einsum("ij,ji->", h_mm, D))
    e_nuc = float(mf.energy_nuc() - mol.energy_nuc())
    e_mm = e_elec + e_nuc
    p = sys_.pcm
    V_qm = surface_potential(p, D)
    print(f"4. round trip: q_FF . V_QM(frozen) = {float(q_ff.numpy() @ V_qm) * KJ:.6f} kJ/mol, pyscf MM embedding energy = "
          f"{e_mm * KJ:.6f} kJ/mol (diff {(float(q_ff.numpy() @ V_qm) - e_mm) * KJ:.2e})")
    print(f"   QM G with FF charges held fixed, 1/2 bookkeeping: {(e_mm - 0.5 * float(q_ff.numpy() @ V_qm)) * KJ:.3f} vs "
          f"QM equilibrated G[frozen] {sv['G_frz'] * KJ:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=float, default=2.6)
    ap.add_argument("--eps", type=float, default=78.3553)
    ap.add_argument("--basis", default="def2-tzvpd")
    ap.add_argument("--ng", type=int, default=302)
    ap.add_argument("--params", default=str(Path(__file__).resolve().parents[1] / "params" / "ions_wb97mv_tzvpd_pf2.json"))
    main(ap.parse_args())
