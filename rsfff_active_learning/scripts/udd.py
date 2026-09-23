#!/usr/bin/env python3
"""Uncertainty-driven dynamics outside the loop: pack, run, report. For trying settings on an
interactive GPU node before the loop spends its budget on them.

    python scripts/udd.py --waters 32 --replicas 8 --time-ps 10 --out runs/udd/w32
    python scripts/udd.py --waters 8 --replicas 4 --time-ps 1 --bias-mode matched --out runs/udd/w8m
    python scripts/udd.py --waters 16 --bias-weight 0 --out runs/udd/w16_unbiased     # reference

Writes to ``--out``:

    frames.npz        every recorded frame (positions, committee energies, sigmas, bias ratio, ...)
    summary.json      settings, per-replica failures/restarts, timing, sigma statistics
    traj_r00.xyz      replica 0 as a plain xyz trajectory (for ChemLab / VMD / ase gui)
    top.extxyz        the --top most uncertain pooled frames (one per 100 fs window)
    state.pt          checkpoint while running; rerunning the same command resumes it
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from rsfff_al.build import pack_waters  # noqa: E402
from rsfff_al.dynamics import DynamicsConfig  # noqa: E402

DEFAULT_COMMITTEE = HERE.parent / "committees" / "film_committee_100k"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--committee", default=str(DEFAULT_COMMITTEE),
                    help="committee dir / committee.json / .pt (default committees/film_committee_100k)")
    ap.add_argument("--waters", type=int, required=True)
    ap.add_argument("--replicas", type=int, default=4)
    ap.add_argument("--start", default=None, help="xyz/extxyz start structure(s) instead of packmol")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--window-fs", type=float, default=100.0)
    ap.add_argument("--deadline-min", type=float, default=None,
                    help="checkpoint and exit after this many minutes (rerun to continue)")
    for f in dataclasses.fields(DynamicsConfig):
        if f.name in ("seed", "min_contact"):
            continue
        flag = "--" + f.name.replace("_", "-").replace("-K", "-k")
        kind = {"bool": lambda s: s.lower() in ("1", "true", "yes")}.get(
            str(f.type), float if "float" in str(f.type) else int if "int" in str(f.type) else str)
        ap.add_argument(flag, dest=f.name, type=kind, default=None)
    a = ap.parse_args(argv)
    cfg_kw = {f.name: getattr(a, f.name) for f in dataclasses.fields(DynamicsConfig)
              if f.name not in ("seed", "min_contact") and getattr(a, f.name, None) is not None}
    cfg = DynamicsConfig(**cfg_kw, seed=a.seed)

    from rsfff_al.committee import Committee, Topology
    from rsfff_al.dynamics import run_replicas
    from rsfff_al.select import pool_indices, scores

    a.out.mkdir(parents=True, exist_ok=True)
    committee = Committee.load(a.committee, device=a.device)
    n = a.waters
    if a.start:
        from easyal import read_extxyz

        frames = read_extxyz(a.start)
        starts = [np.asarray(f["arrays"]["pos"], float) for f in frames][:a.replicas]
        starts = [starts[i % len(starts)] for i in range(a.replicas)]
    else:
        starts = [pack_waters(n, seed=a.seed * 1000 + k, workdir=a.out / "packmol")["positions"]
                  for k in range(a.replicas)]
    print(f"{committee.n_members} members on {committee.device}; {a.replicas} x (H2O){n}; "
          f"{json.dumps(cfg.to_dict())}", flush=True)
    t0 = time.time()
    out = run_replicas(committee, Topology.water(n, device=committee.device), np.stack(starts),
                       cfg, checkpoint=a.out / "state.pt",
                       log=lambda s: print(s, flush=True),
                       deadline=t0 + 60 * a.deadline_min if a.deadline_min else None)
    if out is None:
        print(f"deadline reached; rerun the same command to continue from {a.out / 'state.pt'}")
        return 3

    np.savez_compressed(a.out / "frames.npz",
                        **{k: v for k, v in out.items() if isinstance(v, np.ndarray)})
    se, sf = out["sigma_energy"], out["sigma_forces"]
    prod = out["phase"] == 1
    summary = {
        "waters": n, "replicas": a.replicas, "committee": committee.describe(),
        "config": out["config"], "n_steps": out["n_steps"], "wall_seconds": out["wall_seconds"],
        "s_per_step": round(out["wall_seconds"] / max(out["n_steps"], 1), 4),
        "replica_status": out["replicas"],
        "n_frames": int(len(se)),
        "sigma_energy_per_water_mHa": _stats(se[prod] / n * 1e3),
        "sigma_forces_mHa_per_A": _stats(sf[prod] * 1e3),
        "bias_ratio": _stats(out["bias_ratio"][prod]),
        "bias_on_fraction": float(np.mean(out["bias_on"][prod])) if prod.any() else None,
        "temperature_K": _stats(out["temperature"][prod]),
        "max_force_Ha_per_A": _stats(out["max_force"][prod]),
    }
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))

    species = ["O", "H", "H"] * n
    with open(a.out / "traj_r00.xyz", "w") as fh:
        for j in np.flatnonzero(out["replica"] == 0):
            fh.write(f"{3 * n}\nt={out['time_fs'][j]:.1f} fs sigma_E={se[j] * 1e3:.3f} mHa "
                     f"sigma_F={sf[j] * 1e3:.2f} mHa/A\n")
            fh.writelines(f"{s} {x:.6f} {y:.6f} {z:.6f}\n"
                          for s, (x, y, z) in zip(species, out["positions"][j]))
    sc = scores(se, sf, "both")
    rows = pool_indices(out["replica"], out["time_fs"], sc, window_fs=a.window_fs)
    best = rows[np.argsort(-sc[rows])][:a.top]
    from easyal import new_frame, write_extxyz

    write_extxyz(a.out / "top.extxyz", [new_frame(species, out["positions"][j], {
        "replica": int(out["replica"][j]), "time_fs": float(out["time_fs"][j]),
        "sigma_energy": float(se[j]), "sigma_forces": float(sf[j]), "score": float(sc[j]),
        "committee_energies": [float(e) for e in out["energies"][j]]},
        fragment_idx=[m for m in range(n) for _ in range(3)]) for j in best])
    print(json.dumps({k: summary[k] for k in ("s_per_step", "n_frames",
                                              "sigma_energy_per_water_mHa", "bias_ratio",
                                              "temperature_K")}, indent=1))
    for r in out["replicas"]:
        print(f"replica {r['replica']}: {r['status']}, {r['restarts']} restarts"
              + "".join(f"\n    {x['time_fs']:.0f} fs: {x['reason']}" for x in r["failures"]))
    (a.out / "state.pt").unlink(missing_ok=True)
    return 0


def _stats(x):
    x = np.asarray(x, float)
    if x.size == 0:
        return None
    return {"mean": float(x.mean()), "median": float(np.median(x)),
            "p95": float(np.percentile(x, 95)), "max": float(x.max())}


if __name__ == "__main__":
    sys.exit(main())
