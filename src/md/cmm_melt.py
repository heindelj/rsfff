"""Molten salts on a cmm model: density and partial RDFs at given temperatures (NPT, P = 0), and two-phase
(solid | liquid) coexistence runs that bracket the melting temperature.

    python -m rsfff.md.cmm_melt MODEL --salt NaCl --mode liquid --temps 1100 1200 --rep 4 --ps 20 --equil-ps 10
    python -m rsfff.md.cmm_melt MODEL --salt NaCl --mode coexist --temps 1000 1050 1100 1150 --rep 4 --ps 50

liquid: the crystal (rep^3 conventional cells), its volume set to ~the liquid's, melted at ``--t-melt`` for
``--premelt-ps`` (NVT), then one replica per temperature (one model call per step), NPT (stochastic cell
rescaling, isotropic) for ``--equil-ps`` and ``--ps`` of production: the density, the potential energy and the
partial RDFs averaged every ``--every`` steps.

coexist: a rep x rep x 2 rep crystal; its upper half melted at ``--t-melt`` with the lower half held fixed (NVT,
the box stretched along z by the melt's expansion), then one replica per temperature with the barostat on z only
(x, y keep the crystal's lattice at that temperature, from a short NPT run of the bulk crystal first) for ``--ps``.
Above the melting point the crystal melts (the potential energy rises), below it the liquid freezes (it falls):
the slope of E_pot(t) per temperature brackets T_m. A per-layer order parameter (the fraction of ions whose 6
nearest neighbours sit near the crystal's octahedral / cubic directions) is recorded along z.

Output ``--json``: per temperature the time series (t, E_pot, T, V, P, cell) every ``--every`` steps and the
averages; liquid mode adds the RDFs; ``--xyz`` writes the final configurations. Units: ps, K, g/cm^3, eV/atom.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from .cmm_bench import A_EXPT, B2, crystal
from .cmm_dyn import BatchMD, partial_rdf

AMU_A3_TO_G_CM3 = 1.66053906660


def density(atoms):
    return float(atoms.get_masses().sum() / atoms.get_volume() * AMU_A3_TO_G_CM3)


def premelt(pot, at, T, ps, dt_fs, seed, fixed=None, gamma_ps=10.0, log=print):
    """NVT at T for ps (positions of ``fixed`` atoms held: their velocities and forces zeroed)."""
    sim = BatchMD(pot, [at], dt_fs=dt_fs, T=T, gamma_ps=gamma_ps, P_GPa=None, seed=seed)
    n = int(round(ps * 1000 / dt_fs))
    if fixed is not None:
        x0 = sim.X[0][fixed].copy()

        def hold(s):
            s.X[0][fixed] = x0
            s.V[0][fixed] = 0.0
        hold(sim)
        sim.run(n, callback=hold, every=1)
    else:
        sim.run(n)
    log(f"premelt {ps} ps at {T} K: {sim.thermo()[0]}  ({sim.wall / max(n, 1):.3f} s/step)")
    return sim.atoms(0)


def order_profile(atoms, nbins=8, salt=None):
    """Per z-slab: the mean over ions of the bond-orientational order q4 of their 6 nearest unlike neighbours
    (rock salt ~0.76 at 0 K; the CsCl structure's 8 neighbours give its own value; a liquid ~0.1-0.2)."""
    from ase.neighborlist import neighbor_list
    from scipy.special import sph_harm_y
    i, j, D = neighbor_list("ijD", atoms, 5.0 if salt not in B2 else 5.0)
    d = np.linalg.norm(D, axis=1)
    sym = atoms.numbers
    q4 = np.zeros(len(atoms))
    k = 8 if salt in B2 else 6
    for a in range(len(atoms)):
        m = (i == a) & (sym[j] != sym[a])
        if m.sum() < k:
            continue
        sel = np.argsort(d[m])[:k]
        v = D[m][sel]
        r = np.linalg.norm(v, axis=1)
        th, ph = np.arccos(np.clip(v[:, 2] / r, -1, 1)), np.arctan2(v[:, 1], v[:, 0])
        qlm = np.array([sph_harm_y(4, mm, th, ph).mean() for mm in range(-4, 5)])
        q4[a] = np.sqrt(4 * np.pi / 9 * (np.abs(qlm) ** 2).sum())
    z = atoms.positions[:, 2] % atoms.cell[2, 2]
    edges = np.linspace(0, atoms.cell[2, 2], nbins + 1)
    return [float(q4[(z >= edges[b]) & (z < edges[b + 1])].mean()) for b in range(nbins)]


def finish(out, sim, a, t0, partial=False):
    out["wall_s"] = round(time.time() - t0, 1)
    out["s_per_step"] = round(sim.wall / max(sim.steps, 1), 4)
    out["partial"] = partial
    keys = ("rho_mean", "rho_std", "epot_mean", "rdf_first_peak", "prod_ps") if a.mode == "liquid" else \
        ("lattice", "epot_slope_eV_atom_ps", "verdict")
    out["summary"] = {k: {kk: v.get(kk) for kk in keys} for k, v in out["temps"].items()}
    if a.json:
        tmp = a.json + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f)
        os.replace(tmp, a.json)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--params", default=None)
    ap.add_argument("--salt", required=True)
    ap.add_argument("--mode", choices=["liquid", "coexist"], default="liquid")
    ap.add_argument("--temps", nargs="+", type=float, required=True)
    ap.add_argument("--rep", type=int, default=4, help="conventional cells per side (rock salt 8 ions, CsCl 2)")
    ap.add_argument("--dt", type=float, default=2.0)
    ap.add_argument("--ps", type=float, default=20.0, help="production (liquid) or coexistence run length")
    ap.add_argument("--equil-ps", type=float, default=10.0)
    ap.add_argument("--premelt-ps", type=float, default=10.0)
    ap.add_argument("--t-melt", type=float, default=2500.0, help="temperature of the pre-melt")
    ap.add_argument("--expand", type=float, default=1.20, help="volume factor of the pre-melt cell over the crystal")
    ap.add_argument("--crystal-ps", type=float, default=5.0, help="coexist: NPT of the bulk crystal per temperature")
    ap.add_argument("--every", type=int, default=25)
    ap.add_argument("--gamma", type=float, default=5.0, help="Langevin friction 1/ps")
    ap.add_argument("--tau-p", type=float, default=2.0)
    ap.add_argument("--beta", type=float, default=0.1, help="compressibility used by the barostat, 1/GPa")
    ap.add_argument("--ewald-tol", type=float, default=1e-6)
    ap.add_argument("--cg-rtol", type=float, default=1e-7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--table-cache", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--xyz", default=None)
    ap.add_argument("--premelt-dt", type=float, default=None, help="time step of the pre-melt (default --dt)")
    ap.add_argument("--block-ps", type=float, default=2.5, help="liquid: production block; --json rewritten after each")
    ap.add_argument("--wall-h", type=float, default=None,
                    help="stop production at the first block boundary past this many hours (results kept)")
    a = ap.parse_args(argv)
    from .ions import CmmPotential
    pot = CmmPotential.from_checkpoint(a.model, a.params, a.device, a.table_cache, ewald_tol=a.ewald_tol,
                                       cg_rtol=a.cg_rtol)
    if a.json:
        BatchMD.dump_path = os.path.join(os.path.dirname(os.path.abspath(a.json)), "failed_frames.xyz")
    t0 = time.time()
    rows, out = [], {"salt": a.salt, "mode": a.mode, "args": vars(a), "temps": {}}
    log = lambda *x: print(*x, flush=True)   # noqa: E731
    T = np.asarray(a.temps, float)
    nstep = lambda ps: int(round(ps * 1000 / a.dt))   # noqa: E731

    if a.mode == "liquid":
        base = crystal(a.salt, a.rep)
        base.set_cell(base.cell * a.expand ** (1 / 3), scale_atoms=True)
        liq = premelt(pot, base, a.t_melt, a.premelt_ps, a.premelt_dt or a.dt, a.seed, log=log)
        sim = BatchMD(pot, [liq.copy() for _ in T], dt_fs=a.dt, T=T, gamma_ps=a.gamma, P_GPa=0.0, baro="iso",
                      tau_p_ps=a.tau_p, compressibility_GPa=a.beta, seed=a.seed + 1)
        sim.run(nstep(a.equil_ps))
        log(f"equilibrated {a.equil_ps} ps: {[round(t['V'], 1) for t in sim.thermo()]} A^3  ({sim.wall / sim.steps:.3f} s/step)")
        series = [[] for _ in T]
        rdf_sum = [None for _ in T]
        nrdf = 0

        def sample(s):
            nonlocal nrdf
            th = s.thermo()
            for r, t in enumerate(th):
                at = s.atoms(r)
                series[r].append({"t_ps": s.steps * a.dt / 1000, **t, "rho": density(at)})
                rr, g = partial_rdf(at, nbins=150)
                if rdf_sum[r] is None:
                    rdf_sum[r] = {"r": rr.tolist(), **{k: np.zeros_like(v) for k, v in g.items()}}
                for k, v in g.items():
                    rdf_sum[r][k] += v
            nrdf += 1
        def summarise():
            for r, t in enumerate(T):
                s = series[r]
                if not s:
                    continue
                rho = np.array([x["rho"] for x in s])
                ep = np.array([x["epot_eV_atom"] for x in s])
                rdf = {k: (v / nrdf).tolist() if k != "r" else v for k, v in rdf_sum[r].items()}
                peaks = {k: rdf["r"][int(np.argmax(v))] for k, v in rdf.items() if k != "r"}
                out["temps"][str(t)] = {"rho_mean": float(rho.mean()), "rho_std": float(rho.std()),
                                        "epot_mean": float(ep.mean()), "T_mean": float(np.mean([x["T"] for x in s])),
                                        "P_mean": float(np.mean([x["P_GPa"] for x in s])), "rdf": rdf,
                                        "rdf_first_peak": peaks, "prod_ps": s[-1]["t_ps"] - s[0]["t_ps"], "series": s}
        nblk = max(1, int(round(a.ps / a.block_ps)))
        for b in range(nblk):
            sim.run(nstep(a.ps / nblk), callback=sample, every=a.every)
            summarise()
            finish(out, sim, a, t0, partial=b + 1 < nblk)
            log(f"prod {(b + 1) * a.ps / nblk:.1f} ps: rho {[round(v['rho_mean'], 4) for v in out['temps'].values()]}  "
                f"({sim.wall / sim.steps:.3f} s/step)")
            if a.wall_h and time.time() - t0 > a.wall_h * 3600 and b + 1 < nblk:
                log(f"wall limit {a.wall_h} h reached after {(b + 1) * a.ps / nblk:.1f} ps of production")
                out["stopped_early"] = True
                break
        for t, v in out["temps"].items():
            log(f"T {float(t):.0f}: rho {v['rho_mean']:.4f} +- {v['rho_std']:.4f} g/cm3, E_pot {v['epot_mean']:.4f} "
                f"eV/atom, peaks {v['rdf_first_peak']}")
    else:
        # the crystal's lattice at each temperature (bulk NPT), then the two-phase cell on that lattice
        base = crystal(a.salt, a.rep)
        csim = BatchMD(pot, [base.copy() for _ in T], dt_fs=a.dt, T=T, gamma_ps=a.gamma, P_GPa=0.0, baro="iso",
                       tau_p_ps=a.tau_p, compressibility_GPa=a.beta, seed=a.seed)
        lat = [[] for _ in T]

        def lsample(s):
            for r, t in enumerate(s.thermo()):
                lat[r].append(t["cell"][0])
        csim.run(nstep(a.crystal_ps), callback=lsample, every=a.every)
        half = [float(np.mean(v[len(v) // 2:])) for v in lat]
        log(f"crystal lattice (rep {a.rep}) per T: {dict(zip(T.tolist(), [round(x / a.rep, 4) for x in half]))}")
        reps = []
        for r, t in enumerate(T):
            ac = half[r] / a.rep
            slab = crystal(a.salt, a.rep, a=A_EXPT[a.salt]).repeat((1, 1, 2))
            slab.set_cell(np.diag([ac * a.rep, ac * a.rep, ac * 2 * a.rep]), scale_atoms=True)
            z = slab.positions[:, 2]
            top = z >= slab.cell[2, 2] / 2
            # melt the top half with the bottom held; the box stretched along z for the liquid's lower density
            Lz = slab.cell[2, 2]
            stretch = 1 + 0.5 * (a.expand - 1)
            slab.set_cell(np.diag([slab.cell[0, 0], slab.cell[1, 1], Lz * stretch]), scale_atoms=False)
            slab.positions[top, 2] = Lz / 2 + (slab.positions[top, 2] - Lz / 2) * (1 + (a.expand - 1))
            liq = premelt(pot, slab, a.t_melt, a.premelt_ps, a.premelt_dt or a.dt, a.seed + 10 + r, fixed=np.where(~top)[0], log=log)
            reps.append(liq)
        sim = BatchMD(pot, reps, dt_fs=a.dt, T=T, gamma_ps=a.gamma, P_GPa=0.0, baro="z", tau_p_ps=a.tau_p,
                      compressibility_GPa=a.beta, seed=a.seed + 2)
        series = [[] for _ in T]

        def csample(s):
            for r, t in enumerate(s.thermo()):
                series[r].append({"t_ps": s.steps * a.dt / 1000, **t})
        prof = [[] for _ in T]
        nblk = max(1, int(round(a.ps / 5.0)))
        for b in range(nblk):
            sim.run(nstep(a.ps / nblk), callback=csample, every=a.every)
            for r in range(len(T)):
                prof[r].append(order_profile(sim.atoms(r), salt=a.salt))
            log(f"{(b + 1) * a.ps / nblk:.1f} ps: E_pot {[round(s[-1]['epot_eV_atom'], 4) for s in series]}  "
                f"({sim.wall / sim.steps:.3f} s/step)")
        for r, t in enumerate(T):
            s = series[r]
            tt = np.array([x["t_ps"] for x in s])
            ep = np.array([x["epot_eV_atom"] for x in s])
            k = len(tt) // 5                                       # skip the first fifth
            slope = float(np.polyfit(tt[k:], ep[k:], 1)[0]) if len(tt) > k + 2 else float("nan")
            out["temps"][str(t)] = {"lattice": half[r] / a.rep, "epot_slope_eV_atom_ps": slope,
                                    "verdict": "melts" if slope > 0 else "freezes", "order_profile": prof[r],
                                    "series": s}
            log(f"T {t:.0f}: dE_pot/dt {slope * 1e3:+.4f} meV/atom/ps -> {out['temps'][str(t)]['verdict']}; "
                f"q4 per slab {[round(x, 2) for x in prof[r][-1]]}")
    finish(out, sim, a, t0)
    if a.xyz:
        from ase.io import write
        write(a.xyz, [sim.atoms(r) for r in range(sim.R)])
    return out


if __name__ == "__main__":
    main()
