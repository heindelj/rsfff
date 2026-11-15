"""MD throughput of a cmm model on periodic alkali-halide cells (crystal and a disordered melt-like cell).

    python -m rsfff.md.cmm_bench MODEL --salts NaCl CsI --reps 2 3 4 --steps 20 --device cuda --json bench.json

For each salt and supercell: the ideal crystal at the experimental room-temperature lattice constant, and a
"melt" cell (the same atoms, the volume +20 %, every ion displaced by ~0.6 A and relaxed for a few steps at high
temperature) are run for ``--steps`` velocity-Verlet steps (dt 2 fs, Langevin at ``--temperature``) after
``--warmup`` steps; reported: ms per step (one energy + forces + virial evaluation), atom-steps per second, ns/day
at dt, the joint-solve CG iterations per step, peak GPU memory. ``--batch B`` also times B independent replicas
evaluated in one model call (the throughput of a melting-point or replica sweep).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

A_EXPT = {"LiF": 4.027, "LiCl": 5.130, "LiBr": 5.501, "LiI": 6.000, "NaF": 4.634, "NaCl": 5.640, "NaBr": 5.977,
          "NaI": 6.473, "KF": 5.347, "KCl": 6.293, "KBr": 6.600, "KI": 7.066, "CsF": 6.008, "CsCl": 4.123,
          "CsBr": 4.296, "CsI": 4.567}
B2 = {"CsCl", "CsBr", "CsI"}


def crystal(salt, rep, a=None):
    from ase.build import bulk
    import re
    m, x = re.findall(r"[A-Z][a-z]?", salt)
    a = a or A_EXPT[salt]
    at = bulk(salt, "cesiumchloride" if salt in B2 else "rocksalt", a=a, cubic=True)
    return at.repeat((rep, rep, rep))


def melt_like(at, seed=0, expand=1.20, sigma=0.6):
    at = at.copy()
    at.set_cell(at.cell * expand ** (1 / 3), scale_atoms=True)
    rng = np.random.default_rng(seed)
    at.positions += rng.normal(scale=sigma / np.sqrt(3), size=at.positions.shape)
    return at


def _sync(dev):
    import torch
    if str(dev).startswith("cuda"):
        torch.cuda.synchronize()


def run_md(pot, atoms_list, steps, warmup, dt_fs, T, seed=0, gamma_ps=10.0):
    """Velocity Verlet with a Langevin thermostat (BAOAB-free, plain), all replicas in one evaluate() per step.
    Returns per-step wall times (s) of the measured steps and the CG iterations."""
    from ase import units
    n = [len(a) for a in atoms_list]
    Z = [a.numbers for a in atoms_list]
    X = [a.positions.copy() for a in atoms_list]
    C = [a.cell.array.copy() for a in atoms_list]
    M = [a.get_masses()[:, None] for a in atoms_list]
    rng = np.random.default_rng(seed)
    kT = units.kB * T
    V = [rng.normal(size=x.shape) * np.sqrt(kT / m) for x, m in zip(X, M)]
    dt = dt_fs * units.fs
    gamma = gamma_ps * 1e-3 / units.fs
    c1 = np.exp(-gamma * dt)
    HEV = 27.211386245988

    def forces(X):
        E, F, vir = pot.evaluate(Z, X, C, stress=True)
        return E, [f * HEV for f in F]

    E, F = forces(X)
    times, iters, Es = [], [], []
    for step in range(warmup + steps):
        t0 = time.perf_counter()
        V = [v + 0.5 * dt * f / m for v, f, m in zip(V, F, M)]
        X = [x + dt * v for x, v in zip(X, V)]
        E, F = forces(X)
        V = [v + 0.5 * dt * f / m for v, f, m in zip(V, F, M)]
        V = [c1 * v + np.sqrt((1 - c1 ** 2) * kT / m) * rng.normal(size=v.shape) for v, m in zip(V, M)]
        _sync(pot.device)
        if step >= warmup:
            times.append(time.perf_counter() - t0)
            iters.append(int(pot.last.get("cg_iters", 0)))
            ke = 0.5 * float((M[0] * V[0] ** 2).sum())
            Es.append(float(E[0]) * HEV + ke)
    return np.array(times), np.array(iters), np.array(Es), X


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--params", default=None)
    ap.add_argument("--salts", nargs="+", default=["NaCl", "CsI"])
    ap.add_argument("--reps", nargs="+", type=int, default=[2, 3, 4])
    ap.add_argument("--phases", nargs="+", default=["crystal", "melt"])
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--dt", type=float, default=2.0)
    ap.add_argument("--temperature", type=float, default=1200.0)
    ap.add_argument("--gamma", type=float, default=10.0, help="Langevin friction (1/ps); 0: NVE (reports the drift)")
    ap.add_argument("--batch", nargs="+", type=int, default=[1])
    ap.add_argument("--cutoff", type=float, default=None)
    ap.add_argument("--ewald-tol", type=float, default=1e-10)
    ap.add_argument("--cg-rtol", type=float, default=None, help="joint-solve CG tolerance (default: the model's 1e-10)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--table-cache", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    import torch
    from .ions import CmmPotential
    t0 = time.perf_counter()
    pot = CmmPotential.from_checkpoint(a.model, a.params, a.device, a.table_cache, cutoff=a.cutoff, cg_rtol=a.cg_rtol,
                                       ewald_tol=a.ewald_tol)
    info = {"device": a.device, "load_s": round(time.perf_counter() - t0, 2), "torch": torch.__version__}
    if a.device.startswith("cuda"):
        info["gpu"] = torch.cuda.get_device_name(0)
    print(json.dumps(info), flush=True)
    rows = []
    for salt in a.salts:
        for rep in a.reps:
            for phase in a.phases:
                for B in a.batch:
                    base = crystal(salt, rep)
                    reps = [base if phase == "crystal" else melt_like(base, seed=b) for b in range(B)]
                    pot._x = None
                    if a.device.startswith("cuda"):
                        torch.cuda.reset_peak_memory_stats()
                    try:
                        tt, it, Es, _ = run_md(pot, reps, a.steps, a.warmup, a.dt, a.temperature, gamma_ps=a.gamma)
                    except Exception as exc:  # noqa: BLE001 -- OOM etc.: record and go on
                        rows.append({"salt": salt, "rep": rep, "phase": phase, "batch": B, "n": len(base),
                                     "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
                        print(json.dumps(rows[-1]), flush=True)
                        if a.device.startswith("cuda"):
                            torch.cuda.empty_cache()
                        continue
                    ms = 1e3 * float(np.median(tt))
                    r = {"salt": salt, "rep": rep, "phase": phase, "batch": B, "n": len(base),
                         "ms_per_step": round(ms, 2), "ms_min": round(1e3 * tt.min(), 2),
                         "atom_steps_per_s": round(B * len(base) / (ms / 1e3), 1),
                         "ns_per_day": round(86400 / (ms / 1e3) * a.dt * 1e-6, 3),
                         "cg_iters": float(np.mean(it)),
                         "E_drift_meV_atom": (round(1e3 * (Es[-1] - Es[0]) / len(base), 4) if a.gamma == 0 else None),
                         "E_rms_meV_atom": (round(1e3 * float(np.std(Es)) / len(base), 4) if a.gamma == 0 else None),
                         "cutoff": pot.cutoff}
                    if a.device.startswith("cuda"):
                        r["peak_mem_GB"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
                    rows.append(r)
                    print(json.dumps(r), flush=True)
    if a.json:
        summary = {f"{r['salt']}/{r['n']}/{r['phase']}/B{r['batch']}": r.get("ms_per_step", r.get("error"))
                   for r in rows}
        json.dump({"info": info, "rows": rows, "summary": summary, "args": vars(a)}, open(a.json, "w"), indent=1)
    return rows


if __name__ == "__main__":
    main()
