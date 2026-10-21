"""One command per ion: the complete model's parameter-table entry from monomer references only.

    python scripts/ion_table.py --db DB --ions Li Na K F Cl Br --out params/ions_<level>.json [--work WORK]

Per ion, in order (each step cached under WORK so the command is resumable; nothing from an EDA is used):

  1. monomer references: the elst_db monomer (SCF + fitted density, built by data/elst_db/build.py), its inner
     ESP grid and its uniform-field / field-gradient CPKS responses (``rsfff.qcgen.pol_ref.field_response``);
  2. joint fit (``ion_params.py``'s): Slater 1s shells on a pseudo-nucleus Z_eff, free-ion alpha (exact) and C,
     the single-exponent response parents (kept as the T = 1 fallback);
  3. frontier densities: the Fukui fits of DB/shapes/ion_shells.json (one r^3 e^{-br} term) and the CT energies
     I_donor / A_acceptor (anions: EA from the Fukui Delta-SCF; cations: IP of the atom from the same, IP2 from NIST);
  4. shell refinement on frozen pair scans: the exact-ee pair energies of every pair among the requested ions
     (``elst_ref.pair_energy``) refit the shells' (n, b) with a log penalty toward the ESP fit (``ion_elec_pairfit``'s
     idea, species-generic), R >= --r-min;
  5. response parents on the shell exponents and the l = 0 breathing shape (``ion_parents.py``: the CPKS response
     to a point charge by l).

The validation numbers of every step go into ions[s]["fit"]; ``--report`` prints them for an existing table.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from rsfff.ff.ionct import kernels as K
from rsfff.ff.ionct.params import IonTable
from rsfff.qcgen import elst_db as Db
from rsfff.qcgen import elst_fit as EF
from rsfff.qcgen import elst_ref as E
from rsfff.qcgen import joint_fit as J
from rsfff.qcgen import pol_fit as PF
from rsfff.qcgen import pol_ref as R

EV = 27.211386
KJ = 2625.4996
BOHR = K.BOHR_ANG
HERE = Path(__file__).resolve().parent


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: per ion: database name, atomic number, pseudo-nucleus Z_eff (the electrons it absorbs are not in the shells),
#: number of permanent shells, free induced/permanent exponent ratio, Fukui role, IP2 of the atom (eV, NIST; cations)
#: Z_eff: Na+ 9 / Cl- 11 are the validated choices (Stage A parity); the others follow the same pattern (cation: the
#: outer s-p octet on a pseudo-nucleus; anion: the octet plus the next shell) and are checked by the ESP-fit metrics.
REGISTRY = {
    "Li": dict(db="Lip", Z=3, zv=3.0, charge=1, n_shells=1, free=False, role="acceptor", ip2=75.6400),
    "Na": dict(db="Nap", Z=11, zv=9.0, charge=1, n_shells=1, free=False, role="acceptor", ip2=47.2864),
    "K": dict(db="Kp", Z=19, zv=9.0, charge=1, n_shells=1, free=False, role="acceptor", ip2=31.6250),
    "F": dict(db="Fm", Z=9, zv=7.0, charge=-1, n_shells=2, free=True, role="donor", ip2=None),
    "Cl": dict(db="Clm", Z=17, zv=11.0, charge=-1, n_shells=2, free=True, role="donor", ip2=None),
    "Br": dict(db="Brm", Z=35, zv=11.0, charge=-1, n_shells=2, free=True, role="donor", ip2=None),
}
FUKUI_K = 3


# ---- 1 + 2: monomer references and the joint fit -------------------------------------------------------------
def joint_fit(db, sym, reg, log):
    name = reg["db"]
    t0 = time.time()
    rec = R.field_response(db, name, field_gradients=True)
    m = db.get(name)
    grid = Db.monomer_grid(db, name, scales=EF.INNER_SCALES, cache="grids_inner")
    z = reg["Z"]
    perm = EF.FitConfig(max_rank=0, n_shells=reg["n_shells"], z={z: reg["zv"]}, b_min=1.2, taper=(5.0, 6.0))
    pol = PF.PolConfig(quadrupole=True, n_shape=3, alpha="exact", c_fixed={z: PF.reference_c(rec)})
    cfg = J.JointConfig(perm=perm, pol=pol, perm_min_scale=0.5, free_ratio=reg["free"])
    jf, met = J.fit_joint(grid, rec, [z], db.index[name]["charge"], m.coords, cfg)
    sp = jf.sync()
    b, sw = (x.detach().numpy()[0] for x in sp.shells())
    q, Z = float(sp.q.detach()[0]), float(sp.Z.detach()[0])
    pf = jf.pol

    def weights(w):
        w = np.where(w < 1e-6, 0.0, w)
        return (w / w.sum()).tolist()

    Pi = pf.blocks().detach().numpy()[0]
    out = {"symbol": sym, "db_name": name, "charge": q, "Z": Z,
           "shells": [{"n": float((Z - q) * w), "b": float(bb)} for bb, w in sorted(zip(b, sw), key=lambda t: -t[0])],
           "alpha": float(np.trace(Pi[:3, :3]) / 3), "alpha_ref": float(np.trace(rec["alpha"]) / 3),
           "C": float(np.trace(Pi[3:, 3:]) / 5) if Pi.shape[0] > 3 else None,
           "b_ind": float(pf.b().detach()[0]), "w_dip": weights(pf.w().detach().numpy()[0]),
           "b_quad": float(pf.b_quad().detach()[0]), "w_quad": weights(pf.w_quad().detach().numpy()[0]),
           "ratio": float(met["joint"]["ratio"][jf.elements.index(z)]),
           "fit": {"perm_phi_rel": {s: v["phi_rms"] / v["phi_ref_rms"] for s, v in met["joint"]["perm"].items()},
                   "resp_F": met["joint"]["resp"]["scales"], "resp_G": met["joint"]["resp"].get("scales_G")}}
    log(f"  {sym}: joint fit {time.time() - t0:.0f}s: Z_eff {Z:.0f}, shells "
        f"{[(round(s['n'], 3), round(s['b'], 3)) for s in out['shells']]}, alpha {out['alpha']:.3f} "
        f"(ref {out['alpha_ref']:.3f}), C {out['C']:.2f}, ESP rel err "
        f"{ {k: round(v, 3) for k, v in list(out['fit']['perm_phi_rel'].items())[:4]} }")
    return out


# ---- 3: frontier ---------------------------------------------------------------------------------------------
def frontier(p, reg, shapes):
    fk = shapes[reg["db"]]
    outer = min(s["b"] for s in p["shells"])
    fr = {"donor": {"k": 0, "b": outer}, "acceptor": {"k": 0, "b": outer}}
    fr[reg["role"]] = {"k": FUKUI_K, "b": float(fk["fits"][str(FUKUI_K)][0]), "fit_err": float(fk["fits"][str(FUKUI_K)][1]),
                       "source": "Fukui"}
    p["frontier"] = fr
    if reg["role"] == "donor":
        p["I_donor"], p["A_acceptor"] = float(fk["I_eV"]) / EV, None
    else:
        p["I_donor"], p["A_acceptor"] = reg["ip2"] / EV, float(fk["I_eV"]) / EV


# ---- 4: shells on frozen pair scans ----------------------------------------------------------------------------
def pair_scans(db, syms, path, log):
    """{'A-B': {'R': [A], 'E': [kJ/mol]}} frozen pair energies (exact ee) for every pair among syms (resumable)."""
    scans = json.load(open(path)) if path.exists() else {}
    Rg = np.concatenate([np.arange(1.4, 4.0, 0.05), np.arange(4.0, 8.01, 0.25)])
    from itertools import combinations_with_replacement
    for a, b in combinations_with_replacement(syms, 2):
            key = f"{a}-{b}"
            if key in scans or f"{b}-{a}" in scans:
                continue
            ma, mb = db.get(REGISTRY[a]["db"]), db.get(REGISTRY[b]["db"])
            A = ma.place(shift=-ma.coords[0])
            t0 = time.time()
            e = []
            for r in Rg:
                B = mb.place(shift=np.array([0.0, 0.0, r / BOHR]) - mb.coords[0])
                e.append(E.pair_energy(A, B, exact_ee=True)["total"] * KJ)
            scans[key] = {"R": Rg.tolist(), "E": e}
            json.dump(scans, open(path, "w"))
            log(f"  pair scan {key}: {len(Rg)} points, {time.time() - t0:.0f}s")
    return scans


def _load_table(doc, syms):
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(doc, fh)
    return IonTable.load(fh.name, syms)


def refine_shells(ions, syms, scans, r_min, lam, iters, log):
    """Refit every requested ion's shell (n_t, b_t) to the frozen pair scans with a log penalty toward the ESP fit;
    the last shell's electron count closes the charge. Writes the shells back into ions[s]."""
    from rsfff.ff.complete.elst import permanent_electrostatics
    tab0 = _load_table({"ions": ions}, syms)
    frames, ref = [], []
    for key, d in scans.items():
        a, b = key.split("-")
        if a not in syms or b not in syms:
            continue
        for r, e in zip(d["R"], d["E"]):
            if r >= r_min:
                frames.append((a, b, r))
                ref.append(e)
    ref = torch.tensor(ref, dtype=torch.float64)
    S, T = tab0.shell_b.shape
    pos = torch.tensor([[[0, 0, 0], [0, 0, r]] for _, _, r in frames], dtype=torch.float64).reshape(-1, 3)
    sp = torch.tensor([[syms.index(a), syms.index(b)] for a, b, _ in frames]).reshape(-1)
    fr = torch.arange(len(frames)).repeat_interleave(2)
    live = tab0.shell_n > 0
    lb0 = torch.log(tab0.shell_b)
    ln0 = torch.log(tab0.shell_n.clamp(min=1e-3))
    lb = torch.nn.Parameter(lb0.clone())
    ln = torch.nn.Parameter(ln0.clone())
    ne = tab0.shell_n.sum(1)

    def table():
        b = torch.where(live, lb.exp(), tab0.shell_b)
        n = torch.where(live, ln.exp(), tab0.shell_n)
        # close the charge on each ion's last live shell
        last = (live.cumsum(1) == live.sum(1, keepdim=True)) & live
        n = torch.where(last, (ne - (n * (live & ~last)).sum(1))[:, None].expand_as(n), n)
        return dataclasses.replace(tab0, shell_b=b, shell_n=n)

    opt = torch.optim.LBFGS([lb, ln], max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        tab = table()
        e = permanent_electrostatics(pos, sp, fr, tab) * KJ
        pen = lam * (((lb - lb0) ** 2 + (ln - ln0) ** 2) * live).sum()
        loss = ((e - ref) ** 2).mean() + pen
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        tab = table()
        e = permanent_electrostatics(pos, sp, fr, tab) * KJ
        e0 = permanent_electrostatics(pos, sp, fr, tab0) * KJ
    rep = {}
    for key in sorted({f"{a}-{b}" for a, b, _ in frames}):
        m = torch.tensor([f"{a}-{b}" == key for a, b, _ in frames])
        rep[key] = {"rms": float((e[m] - ref[m]).pow(2).mean().sqrt()), "rms_esp": float((e0[m] - ref[m]).pow(2).mean().sqrt()),
                    "max": float((e[m] - ref[m]).abs().max())}
    for k, s in enumerate(syms):
        ions[s]["shells"] = [{"n": float(tab.shell_n[k, t]), "b": float(tab.shell_b[k, t])} for t in range(T) if live[k, t]]
        ions[s].setdefault("fit", {})["pair_scans"] = {key: v for key, v in rep.items() if s in key.split("-")}
        ions[s]["shells_note"] = f"refined on frozen pair scans R >= {r_min} A, lam {lam} (ion_table.py)"
    log("  pair-scan refit (rms kJ/mol, ESP -> refined): " + ", ".join(f"{k} {v['rms_esp']:.2f} -> {v['rms']:.2f}" for k, v in rep.items()))


# ---- main --------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--ions", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default=None, help="cache dir (default: <out stem>_work next to --out)")
    ap.add_argument("--shapes", default=None, help="ion_shells.json (default DB/shapes/ion_shells.json)")
    ap.add_argument("--r-min", type=float, default=2.3)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--skip", nargs="*", default=[], choices=["joint", "pairs", "parents"], help="steps to skip")
    ap.add_argument("--report", action="store_true", help="print the fit metrics of --out and exit")
    a = ap.parse_args()
    out = Path(a.out)
    log = lambda s: print(s, flush=True)
    if a.report:
        doc = json.load(open(out))
        for s, p in doc["ions"].items():
            log(f"{s}: " + json.dumps(p.get("fit", {}), indent=None)[:800])
        return
    work = Path(a.work or out.with_name(out.stem + "_work"))
    work.mkdir(parents=True, exist_ok=True)
    db = Db.MonomerDB(a.db)
    shapes = json.load(open(a.shapes or Path(a.db) / "shapes" / "ion_shells.json"))["fukui"]
    doc = json.load(open(out)) if out.exists() else {"level": "wB97M-V/def2-TZVPD",
                                                     "units": "atomic (bohr, e, Hartree); alpha bohr^3; C: Theta = C g"}
    ions = doc.setdefault("ions", {})
    for s in a.ions:
        if s not in REGISTRY:
            raise SystemExit(f"{s}: not in the registry ({list(REGISTRY)})")
        cache = work / f"{s}_joint.json"
        if "joint" in a.skip and s in ions:
            continue
        if cache.exists():
            p = json.load(open(cache))
        else:
            p = joint_fit(db, s, REGISTRY[s], log)
            json.dump(p, open(cache, "w"), indent=1)
        frontier(p, REGISTRY[s], shapes)
        ions[s] = {**ions.get(s, {}), **p}
        json.dump(doc, open(out, "w"), indent=1)
    if "pairs" not in a.skip:
        scans = pair_scans(db, list(a.ions), work / "pair_scans.json", log)
        refine_shells(ions, list(a.ions), scans, a.r_min, a.lam, a.iters, log)
        json.dump(doc, open(out, "w"), indent=1)
    if "parents" not in a.skip:
        IP = _load_script("ion_parents")
        IP.DB = str(Path(a.db).resolve())
        tab = _load_table(doc, list(a.ions))
        for s in a.ions:
            IP.DBNAME[s] = REGISTRY[s]["db"]
            scan = IP.response_scan(s, work / f"{s}_resp_scan.json")
            parents, report = IP.fit_ion(s, tab, scan)
            ions[s]["parents"] = parents
            ions[s].setdefault("fit", {})["parents"] = report
            log(f"  {s}: parents {json.dumps(report)}")
            json.dump(doc, open(out, "w"), indent=1)
    doc["built_by"] = "scripts/ion_table.py"
    json.dump(doc, open(out, "w"), indent=1)
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
