"""M6 of the torchff port: is the kernel backend the same force field as the torch backend?

Two checks on a trained FilmModel checkpoint (``checkpoints/water_film_full`` by default):

1. **Parity** on the water clusters in ``benchmarks/structures`` (w4-w23): energy, forces,
   the four interaction components (elst / pauli / disp / induction) and -- on a subset --
   the force-loss gradient into every network parameter (the double backward that training
   depends on), evaluated on the torch backend and on the torchff backend, with the largest
   absolute and relative differences per quantity.
2. **NVE** on a larger cluster (216 waters cut from ``external/torchff-lib/examples/water_216.pdb``
   by default) with velocity Verlet on the torchff backend: total-energy drift and RMS
   fluctuation relative to the kinetic energy, plus the same run on the torch backend for a
   short stretch so the two trajectories can be compared step by step.

Results go to ``benchmarks/validate/results/<tag>_<stamp>.md`` and ``.json``.

    python benchmarks/validate/validate_backends.py --device cuda
    python benchmarks/validate/validate_backends.py --device cpu --skip-nve   # Mac smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "profile"))

from profile_film import _batch_to, read_water  # noqa: E402
from rsfff.ff import backend as ff_backend  # noqa: E402
from rsfff.md.film_driver import load_film_model, make_batch  # noqa: E402

HARTREE_EV = 27.211386245988
BOHR_ANG = 0.529177210903
KB_HARTREE = 3.166811563e-6           # Hartree / K
AMU_ME = 1822.888486209               # electron masses per amu
FS_AU = 41.341373335                  # atomic time units per fs
MASSES = {1: 1.00782503, 8: 15.99491462}


# --------------------------------------------------------------------------------------
# parity
# --------------------------------------------------------------------------------------

def evaluate(model, pos, z, frag, device, *, param_grad: bool):
    b = _batch_to(make_batch(pos, z, frag, n_frames=1), device)
    b.positions.requires_grad_(True)
    out = model(b, with_induction=True)
    (g,) = torch.autograd.grad(out.energy.sum(), b.positions, create_graph=param_grad)
    res = dict(
        energy=out.energy.detach(),
        forces=(-g).detach(),
        **{f"e_{k}": v.detach() for k, v in out.interaction.items()},
    )
    if out.solver:
        res["cg_iters"] = int(out.solver["ind"][0])
    if param_grad:
        loss = out.energy.pow(2).mean() + g.pow(2).sum(-1).mean()
        grads = torch.autograd.grad(loss, [p for p in model.parameters() if p.requires_grad], allow_unused=True)
        res["param_grad"] = torch.cat([x.reshape(-1) for x in grads if x is not None]).detach()
    return res


def parity(model, structures, device, param_grad_every):
    rows = []
    for k, path in enumerate(structures):
        pos, z, frag = read_water(path)
        name = Path(path).stem
        pg = param_grad_every > 0 and k % param_grad_every == 0
        ff_backend.set_backend("torch")
        a = evaluate(model, pos, z, frag, device, param_grad=pg)
        ff_backend.set_backend("torchff")
        c = evaluate(model, pos, z, frag, device, param_grad=pg)
        ff_backend.set_backend("auto")
        row = dict(structure=name, n_atoms=int(z.size), cg_iters=(a.get("cg_iters"), c.get("cg_iters")))
        for key in a:
            if key == "cg_iters":
                continue
            x, y = a[key], c[key]
            diff = (x - y).abs()
            scale = x.abs().max().clamp(min=1e-30)
            row[key] = dict(
                torch=float(x.sum()) if key == "energy" or key.startswith("e_") else None,
                max_abs=float(diff.max()), max_rel=float(diff.max() / scale),
            )
        rows.append(row)
        print(f"  {name:14s} " + " ".join(
            f"{key}={row[key]['max_abs']:.1e}" for key in ("energy", "forces", "e_elst", "e_pauli", "e_disp", "e_induction", "param_grad")
            if key in row), flush=True)
    return rows


# --------------------------------------------------------------------------------------
# NVE
# --------------------------------------------------------------------------------------

def nve(model, pos_ang, z, frag, device, *, steps, dt_fs, temperature, seed, log_every=10):
    """Velocity Verlet in atomic units on the current backend. Returns per-step energies."""
    n = int(z.size)
    mass = torch.tensor([MASSES[int(a)] * AMU_ME for a in z], dtype=torch.float64, device=device)[:, None]
    x = torch.as_tensor(np.asarray(pos_ang) / BOHR_ANG, dtype=torch.float64, device=device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    v = torch.randn(n, 3, generator=gen, dtype=torch.float64).to(device) * torch.sqrt(KB_HARTREE * temperature / mass)
    v = v - (mass * v).sum(0) / mass.sum()                       # no centre-of-mass drift
    dt = dt_fs * FS_AU

    def force(x_bohr):
        b = _batch_to(make_batch((x_bohr * BOHR_ANG).detach().cpu().numpy(), z, frag, n_frames=1), device)
        b.positions.requires_grad_(True)
        out = model(b, with_induction=True)
        (g,) = torch.autograd.grad(out.energy.sum(), b.positions)
        return float(out.energy.sum()), -g.detach() * BOHR_ANG      # Hartree/bohr

    e_pot, f = force(x)
    log = []
    t0 = time.perf_counter()
    for step in range(steps + 1):
        e_kin = float(0.5 * (mass * v * v).sum())
        log.append(dict(step=step, e_pot=e_pot, e_kin=e_kin, e_tot=e_pot + e_kin,
                        temperature=2 * e_kin / (3 * n * KB_HARTREE)))
        if step % log_every == 0:
            print(f"    step {step:5d}  E_tot {e_pot + e_kin:+.8f} Ha  T {log[-1]['temperature']:6.1f} K  "
                  f"({(time.perf_counter() - t0) / max(step, 1) * 1e3:.0f} ms/step)", flush=True)
        if step == steps:
            break
        v = v + 0.5 * dt * f / mass
        x = x + dt * v
        e_pot, f = force(x)
        v = v + 0.5 * dt * f / mass
    return log


def nve_summary(log, dt_fs):
    e = np.array([r["e_tot"] for r in log])
    ek = np.array([r["e_kin"] for r in log])
    t = np.arange(len(e)) * dt_fs
    slope = np.polyfit(t, e, 1)[0] if len(e) > 2 else 0.0
    return dict(
        steps=len(e) - 1, dt_fs=dt_fs, t_ps=t[-1] / 1000.0,
        e_tot_drift_per_ps=float(slope * 1000.0),
        e_tot_rms=float(e.std()), e_kin_mean=float(ek.mean()),
        e_tot_rms_over_e_kin=float(e.std() / max(ek.mean(), 1e-30)),
        temperature_mean=float(np.mean([r["temperature"] for r in log])),
    )


# --------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------

def format_parity(rows):
    keys = ["energy", "forces", "e_elst", "e_pauli", "e_disp", "e_induction", "param_grad"]
    lines = ["| structure | atoms | CG iters (torch/torchff) | " + " | ".join(f"{k} max abs (rel)" for k in keys) + " |",
             "|---|---|---|" + "---|" * len(keys)]
    for r in rows:
        cells = []
        for k in keys:
            if k in r:
                cells.append(f"{r[k]['max_abs']:.1e} ({r[k]['max_rel']:.1e})")
            else:
                cells.append("-")
        lines.append(f"| {r['structure']} | {r['n_atoms']} | {r['cg_iters'][0]}/{r['cg_iters'][1]} | " + " | ".join(cells) + " |")
    worst = {k: max((r[k]["max_rel"] for r in rows if k in r), default=float("nan")) for k in keys}
    lines.append("\nWorst relative difference per quantity: " + ", ".join(f"{k} {v:.1e}" for k, v in worst.items()))
    lines.append("Energies in Hartree, forces in Hartree/Angstrom; rel = max abs / max |torch value|. "
                 "param_grad = gradient of the E+F loss into every network parameter (double backward), on a subset.")
    return "\n".join(lines)


def format_nve(summaries, compare):
    lines = ["| backend | steps | dt (fs) | t (ps) | T mean (K) | E_tot drift (Ha/ps) | E_tot RMS (Ha) | E_tot RMS / <E_kin> |",
             "|---|---|---|---|---|---|---|---|"]
    for name, s in summaries.items():
        lines.append(f"| {name} | {s['steps']} | {s['dt_fs']} | {s['t_ps']:.3f} | {s['temperature_mean']:.1f} | "
                     f"{s['e_tot_drift_per_ps']:+.2e} | {s['e_tot_rms']:.2e} | {s['e_tot_rms_over_e_kin']:.2e} |")
    if compare:
        lines.append(f"\nSame initial conditions on both backends for {compare['steps']} steps: max |E_tot(torch) - E_tot(torchff)| "
                     f"= {compare['max_abs_e_tot']:.2e} Ha, max |E_pot diff| = {compare['max_abs_e_pot']:.2e} Ha.")
    lines.append("E_tot RMS / <E_kin> well below 1e-2 and no systematic drift is what a conserving integrator + smooth "
                 "force field gives at this time step; a drift on torchff that torch does not show means a kernel "
                 "derivative is wrong somewhere the unit tests did not reach.")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=str(REPO_ROOT / "checkpoints/water_film_full/best.pt"))
    ap.add_argument("--structures", nargs="+", default=sorted(
        str(p) for p in (REPO_ROOT / "benchmarks/structures").glob("w*_mp2_avtz.xyz")))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--param-grad-every", type=int, default=3,
                    help="double-backward parity on every k-th structure (0 = never)")
    ap.add_argument("--skip-parity", action="store_true")
    ap.add_argument("--skip-nve", action="store_true")
    ap.add_argument("--nve-structure", default=str(REPO_ROOT / "external/torchff-lib/examples/water_216.pdb"))
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--dt", type=float, default=0.5, help="fs")
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--compare-steps", type=int, default=50,
                    help="rerun this many NVE steps on the torch backend from the same start for a step-by-step comparison")
    ap.add_argument("--max-neighbors", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO_ROOT / "benchmarks/validate/results"))
    ap.add_argument("--tag", default=None)
    args = ap.parse_args(argv)

    if not ff_backend.HAVE_TORCHFF:
        raise SystemExit("torchff is not importable; nothing to validate against")
    device = torch.device(args.device)
    torch.set_default_dtype(torch.float64)
    model, config = load_film_model(args.checkpoint, device=str(device))
    model.max_num_neighbors = int(args.max_neighbors)
    tag = args.tag or (torch.cuda.get_device_name(device).replace(" ", "_") if device.type == "cuda"
                       else platform.processor() or "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"checkpoint": args.checkpoint, "torch": torch.__version__, "host": platform.node(), "device": str(device)}
    md = [f"# Backend validation: {tag}, {time.strftime('%Y-%m-%d %H:%M')}\n",
          f"checkpoint `{args.checkpoint}`, torch {torch.__version__}, device {device}\n"]

    if not args.skip_parity:
        print("== parity, torch vs torchff", flush=True)
        rows = parity(model, args.structures, device, args.param_grad_every)
        report["parity"] = rows
        md += ["## Parity: torch backend vs torchff backend\n", format_parity(rows), ""]

    if not args.skip_nve:
        pos, z, frag = read_water(args.nve_structure)
        name = Path(args.nve_structure).stem
        print(f"== NVE on {name} ({z.size} atoms), {args.steps} steps of {args.dt} fs at {args.temperature} K", flush=True)
        ff_backend.set_backend("torchff")
        log_f = nve(model, pos, z, frag, device, steps=args.steps, dt_fs=args.dt,
                    temperature=args.temperature, seed=args.seed)
        summaries = {"torchff": nve_summary(log_f, args.dt)}
        compare = None
        if args.compare_steps > 0:
            print(f"== NVE on torch backend, {args.compare_steps} steps from the same start", flush=True)
            ff_backend.set_backend("torch")
            log_t = nve(model, pos, z, frag, device, steps=args.compare_steps, dt_fs=args.dt,
                        temperature=args.temperature, seed=args.seed)
            summaries["torch"] = nve_summary(log_t, args.dt)
            k = min(len(log_t), len(log_f))
            compare = dict(
                steps=k - 1,
                max_abs_e_tot=float(max(abs(a["e_tot"] - b["e_tot"]) for a, b in zip(log_t[:k], log_f[:k]))),
                max_abs_e_pot=float(max(abs(a["e_pot"] - b["e_pot"]) for a, b in zip(log_t[:k], log_f[:k]))),
            )
        ff_backend.set_backend("auto")
        report["nve"] = dict(structure=name, n_atoms=int(z.size), summaries=summaries, compare=compare, log_torchff=log_f)
        md += [f"## NVE: {name}, {z.size} atoms\n", format_nve(summaries, compare), ""]

    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = out_dir / f"{tag}_{stamp}"
    with open(base.with_suffix(".json"), "w") as fh:
        json.dump(report, fh, indent=1)
    with open(base.with_suffix(".md"), "w") as fh:
        fh.write("\n".join(md))
    print(f"\nwrote {base.with_suffix('.md')} and .json")
    print("\n".join(md))


if __name__ == "__main__":
    main()
