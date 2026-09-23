"""Uncertainty-driven Langevin dynamics: many replicas of one cluster size, in lockstep.

The force the integrator follows
--------------------------------
::

    F = (1 - w) Fbar  +  w * kappa * grad sigma_E  +  F_wall

``Fbar`` is the committee-mean force and ``sigma_E`` the committee's standard deviation of the
total energy (:mod:`rsfff_al.committee`). ``w`` is the bias weight (``0.2``: "0.8 along the
true gradient, 0.2 along the uncertainty gradient"). The sign pushes *up* the uncertainty.

``bias_mode`` sets ``kappa``:

``"matched"`` (default)  ``kappa = |Fbar| / |grad sigma_E|`` per replica, per step (norms over
    all atoms): the two directions are mixed 80/20 *by magnitude*, which is the "step 0.8
    along the true gradient, 0.2 along the uncertainty gradient" rule taken literally. Not
    conservative -- the thermostat (and the temperature guard) is what keeps it bounded.
``"raw"``  ``kappa`` is a constant (``bias_kappa``, default 1). The force is then ``-grad U``
    with ``U = (1 - w) Ebar - w kappa sigma_E``: conservative, so the thermostat samples a
    well-defined biased ensemble ``exp(-U / kT)``. For film_committee_100k on small clusters
    ``|grad sigma| / |Fbar|`` is ~0.05-0.1, so with ``kappa = 1`` the bias is ~1% of the force
    and the run is effectively unbiased; ``kappa`` of order 10-30 is where it starts to act.
    Note the ``(1 - w)`` on the mean surface is itself a heating: at 300 K the model surface
    is sampled as if at ``300 / 0.8 = 375 K`` (``scale_mean_force=False`` keeps the full
    ``Fbar`` and only adds the bias).

Every record carries ``bias_ratio`` = |bias term| / |mean-force term|, the number to look at
when choosing between them.

The bias is switched off for a replica whose ``sigma_E`` per atom exceeds ``bias_sigma_max``
(Hartree/atom): past that point the committee is already lost and pushing further only buys
geometries nobody should run DFT on.

Integrator and guards
---------------------
BAOAB Langevin (Leimkuhler-Matthews) in Angstrom / fs / amu, every replica its own time step
(warmup replicas use ``warmup_dt_fs`` and ``warmup_friction``: packmol packings carry strain,
and the start is where a trajectory is most likely to blow up). The trajectories start from the
packings as they are -- no minimization -- so the relaxation out of a bad packing is sampled
too.

Per replica, a step is a **failure** when: an energy or force is not finite or a member raised;
``|Fbar|`` on some atom exceeds ``max_force``; the thermostat-smoothed temperature exceeds
``max_temperature``; an O-H bond exceeds ``max_oh``; or two atoms of different waters come
closer than ``min_contact`` (per element pair). ``sigma_abort`` (Hartree/atom) optionally
fails on runaway uncertainty.

A failure **rewinds** the replica to its snapshot at least ``rewind_fs`` earlier, redraws the
velocities and carries on (the frames it recorded on the way stay, tagged with
``fs_to_failure``, since what a model does just before it breaks is exactly what it has not
been taught). After ``max_restarts`` rewinds the replica is retired and keeps what it has.

Resuming
--------
``run_replicas(..., checkpoint=path)`` writes the whole state (positions, velocities, clocks,
snapshots, RNG, records) every ``checkpoint_every_s`` seconds and on exit, and picks it up
again when called with the same path, so a trajectory batch longer than one allocation just
continues in the next.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from .committee import Committee, Evaluation, Topology

__all__ = ["DynamicsConfig", "run_replicas", "HARTREE_EV", "flat_bottom_wall"]

HARTREE_EV = 27.211386245988
ACC = 9.648533212e-3          # (eV/Angstrom)/amu -> Angstrom/fs^2
KB_EV = 8.617333262e-5        # eV/K


@dataclass
class DynamicsConfig:
    # thermostat and length
    temperature_K: float = 300.0
    timestep_fs: float = 0.5
    friction_per_fs: float = 0.01
    time_ps: float = 10.0                  # production time per replica (after warmup)
    warmup_fs: float = 250.0
    warmup_dt_fs: float = 0.25
    warmup_friction: float = 0.1
    # bias
    bias_weight: float = 0.2
    bias_mode: str = "matched"             # "matched" | "raw" | "none"
    bias_kappa: float = 1.0
    scale_mean_force: bool = True
    bias_sigma_max: float | None = 1e-3    # Hartree/atom; bias off above
    # confinement (flat-bottom sphere around the running COM; oxygens at R, hydrogens R+h_slack)
    wall_k: float = 0.5                    # Hartree/Angstrom^2
    wall_margin: float = 2.0               # Angstrom added to the starting radius
    wall_h_slack: float = 1.2
    # recording
    stride: int = 20                       # steps between recorded frames
    # guards
    max_force: float = 0.5                 # Hartree/Angstrom, per atom |Fbar|
    max_temperature: float | None = None   # K, default 3 x temperature_K
    temperature_tau_fs: float = 100.0
    max_oh: float = 1.3                    # Angstrom (train/scripts/check_data.py OH_MAX)
    min_contact: dict = field(default_factory=lambda: {"OO": 2.0, "OH": 1.15, "HH": 1.0})
    sigma_abort: float | None = None       # Hartree/atom
    check_every: int = 5
    rewind_fs: float = 200.0
    max_restarts: int = 5
    seed: int = 0
    checkpoint_every_s: float = 600.0

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def t_max(self) -> float:
        return self.max_temperature or 3.0 * self.temperature_K


def flat_bottom_wall(pos: torch.Tensor, top: Topology, radius: torch.Tensor, k: float,
                     h_slack: float):
    """Batched ``rsfff.md.confine.flat_bottom_sphere``: energy (R,) Hartree and force
    (R, N, 3) Hartree/Angstrom, the sphere centered on each replica's running COM (so the net
    force is exactly zero)."""
    if k == 0.0:
        return pos.new_zeros(pos.shape[0]), torch.zeros_like(pos)
    x = pos.detach().requires_grad_(True)
    m = top.masses
    com = (m[None, :, None] * x).sum(1, keepdim=True) / m.sum()
    shell = radius[:, None] + torch.where(top.atomic_numbers == 1, h_slack, 0.0)[None, :]
    excess = ((x - com).norm(dim=-1) - shell).clamp(min=0.0)
    e = 0.5 * k * (excess ** 2).sum(-1)
    if not e.requires_grad or float(e.detach().abs().max()) == 0.0:
        return e.detach(), torch.zeros_like(pos)
    (g,) = torch.autograd.grad(e.sum(), x)
    return e.detach(), -g


def _pair_masks(top: Topology):
    z = top.atomic_numbers
    inter = top.fragment_idx[:, None] != top.fragment_idx[None, :]
    iu = torch.triu(torch.ones_like(inter), diagonal=1)
    o, h = z == 8, z == 1
    return {
        "OO": inter & iu & o[:, None] & o[None, :],
        "OH": inter & ((o[:, None] & h[None, :]) | (h[:, None] & o[None, :])) & iu,
        "HH": inter & iu & h[:, None] & h[None, :],
    }


def _oh_pairs(top: Topology):
    z = top.atomic_numbers.cpu().numpy()
    frag = top.fragment_idx.cpu().numpy()
    pairs = []
    for f in range(int(frag.max()) + 1):
        atoms = np.flatnonzero(frag == f)
        oxy = atoms[z[atoms] == 8]
        if len(oxy) != 1:
            continue
        pairs += [(int(oxy[0]), int(a)) for a in atoms if z[a] == 1]
    return torch.tensor(pairs, dtype=torch.long, device=top.atomic_numbers.device)


class _Guards:
    def __init__(self, top: Topology, cfg: DynamicsConfig):
        self.cfg = cfg
        self.masks = {k: v for k, v in _pair_masks(top).items() if k in cfg.min_contact}
        self.oh = _oh_pairs(top)
        self.n_atoms = top.n_atoms

    def geometry(self, pos: torch.Tensor) -> list[str]:
        """'' for a sane replica, else why not. pos (R, N, 3)."""
        reasons = [""] * pos.shape[0]
        if len(self.oh):
            r_oh = (pos[:, self.oh[:, 0]] - pos[:, self.oh[:, 1]]).norm(dim=-1)   # (R, B)
            worst = r_oh.max(-1).values
            for i in torch.nonzero(worst > self.cfg.max_oh).flatten().tolist():
                reasons[i] = f"O-H bond {float(worst[i]):.2f} A > {self.cfg.max_oh}"
        d = torch.cdist(pos, pos)
        for name, mask in self.masks.items():
            if not bool(mask.any()):
                continue
            dmin = d[:, mask].min(-1).values
            limit = float(self.cfg.min_contact[name])
            for i in torch.nonzero(dmin < limit).flatten().tolist():
                if not reasons[i]:
                    reasons[i] = f"intermolecular {name} contact {float(dmin[i]):.2f} A < {limit}"
        return reasons




_REC_KEYS = ("positions", "replica", "time_fs", "phase", "energies", "sigma_energy",
             "sigma_forces", "max_force", "temperature", "bias_ratio", "bias_on", "restart",
             "wall_energy", "fs_to_failure")


def run_replicas(committee: Committee, top: Topology, start_positions, cfg: DynamicsConfig,
                 *, checkpoint=None, log=print, tags=None, deadline=None) -> dict | None:
    """Run ``len(start_positions)`` replicas of one topology for ``cfg.warmup_fs + 1000 *
    cfg.time_ps`` fs each; return the recorded frames and a per-replica summary.

    ``start_positions`` (R, N, 3) Angstrom. Returns a dict of numpy arrays, one row per
    recorded frame -- ``positions`` (M, N, 3), ``replica``, ``time_fs``, ``phase`` (0 warmup,
    1 production), ``energies`` (M, K), ``sigma_energy`` (Hartree, total), ``sigma_forces``
    (Hartree/Angstrom), ``max_force`` (largest per-atom |Fbar|), ``temperature``
    (instantaneous), ``bias_ratio``, ``bias_on``, ``restart``, ``wall_energy``,
    ``fs_to_failure`` (-1 unless the replica failed later in the same restart) -- plus
    ``replicas`` (per-replica status and failures), ``n_steps``, ``wall_seconds``, ``config``.

    ``deadline`` (``time.time()`` value): past it, save the checkpoint and return ``None``;
    calling again with the same ``checkpoint`` continues.
    """
    dev = committee.device
    dtype = torch.get_default_dtype()
    start = torch.as_tensor(np.asarray(start_positions), dtype=dtype, device=dev)
    n_rep, n_atoms = start.shape[0], start.shape[1]
    guards = _Guards(top, cfg)
    mass = top.masses.to(dev, dtype)[None, :, None]                         # (1, N, 1)
    t_total = cfg.warmup_fs + 1000.0 * cfg.time_ps
    kT = KB_EV * cfg.temperature_K
    dof = 3 * n_atoms
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(cfg.seed))

    def maxwell(n):
        sd = torch.sqrt(kT * ACC / mass)                                    # Angstrom/fs
        v = torch.randn((n, n_atoms, 3), generator=gen, device=dev, dtype=dtype) * sd
        p = (mass * v).sum(1, keepdim=True)
        return v - p / mass.sum()                                           # no COM drift

    # --- state ---------------------------------------------------------------------------
    state_file = Path(checkpoint) if checkpoint else None
    if state_file is not None and state_file.exists():
        s = torch.load(state_file, map_location=dev, weights_only=False)
        pos, vel, clock, t_ema = s["pos"], s["vel"], s["clock"], s["t_ema"]
        alive, restarts, radius = s["alive"], s["restarts"], s["radius"]
        snaps, rec, reps, since = s["snaps"], s["rec"], s["reps"], s["since"]
        gen.set_state(s["gen"])
        n_steps, wall_before = s["n_steps"], s["wall_seconds"]
        log(f"resumed {state_file.name}: {int(alive.sum())}/{n_rep} replicas running")
    else:
        pos = start.clone()
        vel = maxwell(n_rep)
        clock = torch.zeros(n_rep, dtype=dtype, device=dev)
        t_ema = torch.full((n_rep,), cfg.temperature_K, dtype=dtype, device=dev)
        alive = torch.ones(n_rep, dtype=torch.bool, device=dev)
        restarts = torch.zeros(n_rep, dtype=torch.long, device=dev)
        com = (mass * pos).sum(1, keepdim=True) / mass.sum()
        radius = (pos - com).norm(dim=-1).max(-1).values + cfg.wall_margin
        snaps = [[(0.0, pos[i].detach().cpu().clone())] for i in range(n_rep)]  # (t, positions)
        rec = {k: [] for k in _REC_KEYS}
        reps = [{"replica": i, "failures": [], "status": "running"} for i in range(n_rep)]
        since = [[] for _ in range(n_rep)]         # record rows since the last (re)start
        n_steps, wall_before = 0, 0.0
    t_start = time.perf_counter()

    def save():
        if state_file is None:
            return
        tmp = state_file.with_name(f".{state_file.name}.{os.getpid()}.tmp")
        torch.save({"pos": pos, "vel": vel, "clock": clock, "t_ema": t_ema, "alive": alive,
                    "restarts": restarts, "radius": radius, "snaps": snaps, "rec": rec,
                    "reps": reps, "since": since, "gen": gen.get_state(), "n_steps": n_steps,
                    "wall_seconds": wall_before + time.perf_counter() - t_start}, tmp)
        os.replace(tmp, state_file)

    def evaluate(idx, x):
        """Driving force (Hartree/Angstrom) on replicas ``idx`` at ``x``, and diagnostics."""
        ev: Evaluation = committee.evaluate(x, top)
        fbar, sig = ev.mean_forces, ev.sigma_energy
        drive = fbar
        bias_on = torch.zeros(len(idx), dtype=torch.bool, device=dev)
        ratio = torch.zeros(len(idx), dtype=dtype, device=dev)
        if cfg.bias_mode != "none" and cfg.bias_weight > 0 and ev.n_members > 1:
            w = cfg.bias_weight
            gsig = ev.grad_sigma()
            bias_on = torch.isfinite(sig)
            if cfg.bias_sigma_max is not None:
                bias_on &= sig / n_atoms <= cfg.bias_sigma_max
            fnorm = fbar.flatten(1).norm(dim=-1)
            gnorm = gsig.flatten(1).norm(dim=-1)
            if cfg.bias_mode == "matched":
                kappa = fnorm / gnorm.clamp(min=1e-12)
            elif cfg.bias_mode == "raw":
                kappa = torch.full_like(sig, float(cfg.bias_kappa))
            else:
                raise ValueError(f"bias_mode {cfg.bias_mode!r}: raw | matched | none")
            a = (1.0 - w) if cfg.scale_mean_force else 1.0
            biased = a * fbar + (w * kappa)[:, None, None] * gsig
            drive = torch.where(bias_on[:, None, None], biased, fbar)
            ratio = (w * kappa * gnorm) / (a * fnorm).clamp(min=1e-12)
        e_wall, f_wall = flat_bottom_wall(x, top, radius[idx], cfg.wall_k, cfg.wall_h_slack)
        return {"idx": idx, "drive": drive + f_wall, "ev": ev, "bias_on": bias_on,
                "ratio": ratio, "e_wall": e_wall}

    def fail(i: int, reason: str):
        """Rewind replica ``i`` to a snapshot ``rewind_fs`` back, or retire it."""
        t_fail = float(clock[i])
        reps[i]["failures"].append({"time_fs": round(t_fail, 2), "reason": reason,
                                    "restart": int(restarts[i])})
        for j in since[i]:
            rec["fs_to_failure"][j] = t_fail - rec["time_fs"][j]
        since[i] = []
        if int(restarts[i]) >= cfg.max_restarts:
            alive[i] = False
            reps[i]["status"] = "retired"
            log(f"  replica {i}: {reason} at {t_fail:.0f} fs; retired "
                f"after {int(restarts[i])} restarts")
            return
        restarts[i] += 1
        back = [s for s in snaps[i] if s[0] <= t_fail - cfg.rewind_fs] or snaps[i][:1]
        t_back, x_back = back[-1]
        snaps[i] = [s for s in snaps[i] if s[0] <= t_back]
        pos[i] = x_back.to(dev)
        vel[i] = maxwell(1)[0]
        clock[i] = t_back
        t_ema[i] = cfg.temperature_K
        log(f"  replica {i}: {reason} at {t_fail:.0f} fs; rewound to {t_back:.0f} fs "
            f"(restart {int(restarts[i])})")

    cache = None
    last_save = time.perf_counter()
    while True:
        finished = alive & (clock >= t_total - 1e-9)
        for i in torch.nonzero(finished).flatten().tolist():
            alive[i] = False
            reps[i]["status"] = "complete"
        idx = torch.nonzero(alive).flatten()
        if len(idx) == 0:
            break
        if cache is None or not torch.equal(cache["idx"], idx):
            cache = evaluate(idx, pos[idx])

        warm = clock[idx] < cfg.warmup_fs
        dt = torch.where(warm, cfg.warmup_dt_fs, cfg.timestep_fs).to(dtype)[:, None, None]
        gamma = torch.where(warm, cfg.warmup_friction, cfg.friction_per_fs).to(dtype)[:, None, None]
        x, v = pos[idx], vel[idx]
        v = v + 0.5 * dt * cache["drive"] * (HARTREE_EV * ACC) / mass              # B
        x = x + 0.5 * dt * v                                                        # A
        c1 = torch.exp(-gamma * dt)                                                 # O
        v = c1 * v + torch.sqrt((1 - c1 ** 2) * kT * ACC / mass) * torch.randn(
            v.shape, generator=gen, device=dev, dtype=dtype)
        x = x + 0.5 * dt * v                                                        # A
        cache = evaluate(idx, x.detach())
        v = v + 0.5 * dt * cache["drive"] * (HARTREE_EV * ACC) / mass              # B
        pos[idx], vel[idx] = x.detach(), v.detach()
        clock[idx] += dt.flatten()
        n_steps += 1

        t_inst = (mass * v ** 2).sum((1, 2)) / ACC / (dof * KB_EV)                 # 2 KE / (dof kB)
        a_ema = (dt.flatten() / cfg.temperature_tau_fs).clamp(max=1.0)
        t_ema[idx] = (1 - a_ema) * t_ema[idx] + a_ema * t_inst

        # --- guards (one host sync per step) ---------------------------------------------
        ev = cache["ev"]
        fmax = ev.mean_forces.norm(dim=-1).max(-1).values
        host = torch.stack([ev.failed.to(dtype), fmax, t_ema[idx],
                            ev.sigma_energy / n_atoms]).cpu().numpy()
        reasons = []
        for bad, fm, te, sa in host.T:
            if bad or not np.isfinite(fm):
                reasons.append("committee evaluation failed or not finite")
            elif fm > cfg.max_force:
                reasons.append(f"|F| {fm:.3f} Ha/A > {cfg.max_force}")
            elif te > cfg.t_max:
                reasons.append(f"T {te:.0f} K > {cfg.t_max:.0f}")
            elif cfg.sigma_abort is not None and sa > cfg.sigma_abort:
                reasons.append(f"sigma_E/atom {sa:.2e} > {cfg.sigma_abort}")
            else:
                reasons.append("")
        record_now = n_steps % cfg.stride == 0
        if record_now or n_steps % cfg.check_every == 0:
            geo = guards.geometry(x.detach())
            reasons = [r or g for r, g in zip(reasons, geo)]

        if record_now:
            sig_f = ev.sigma_forces.cpu().numpy()
            sig_e = ev.sigma_energy.cpu().numpy()
            energies = ev.energy.T.cpu().numpy()
            ratio = cache["ratio"].cpu().numpy()
            bias_on = cache["bias_on"].cpu().numpy()
            e_wall = cache["e_wall"].cpu().numpy()
            t_host = t_inst.cpu().numpy()
            clock_host = clock[idx].cpu().numpy()
            x_host = x.detach().cpu()
            for li, i in enumerate(idx.tolist()):
                if reasons[li]:
                    continue
                t_i = float(clock_host[li])
                since[i].append(len(rec["replica"]))
                for key, val in (("positions", x_host[li].numpy()), ("replica", i),
                                 ("time_fs", t_i), ("phase", int(t_i > cfg.warmup_fs)),
                                 ("energies", energies[li]), ("sigma_energy", float(sig_e[li])),
                                 ("sigma_forces", float(sig_f[li])),
                                 ("max_force", float(host[1, li])),
                                 ("temperature", float(t_host[li])),
                                 ("bias_ratio", float(ratio[li])),
                                 ("bias_on", bool(bias_on[li])),
                                 ("restart", int(restarts[i])),
                                 ("wall_energy", float(e_wall[li])), ("fs_to_failure", -1.0)):
                    rec[key].append(val)
                snaps[i].append((t_i, x_host[li].clone()))
                if len(snaps[i]) > 64:            # keep the start and what a rewind can reach
                    keep_from = t_i - 4 * cfg.rewind_fs
                    snaps[i] = snaps[i][:1] + [s for s in snaps[i][1:] if s[0] >= keep_from]

        if any(reasons):
            for li, reason in enumerate(reasons):
                if reason:
                    fail(int(idx[li]), reason)
            cache = None                          # re-evaluate the rewound set

        now = time.perf_counter()
        if record_now and n_steps % (cfg.stride * 100) == 0:
            a = torch.nonzero(alive).flatten()
            m = max(len(a), 1)
            log(f"  step {n_steps}: {len(a)}/{n_rep} running, t {float(clock[a].min()) if len(a) else 0:.0f}-"
                f"{float(clock[a].max()) if len(a) else 0:.0f} fs, <T> "
                f"{float(t_ema[a].mean()) if len(a) else 0:.0f} K, sigma_E/atom "
                f"{np.mean(rec['sigma_energy'][-m:]) / n_atoms * 1e3:.3f} mHa, bias ratio "
                f"{np.mean(rec['bias_ratio'][-m:]):.3f}, "
                f"{(now - t_start) / max(n_steps, 1):.3f} s/step")
        if state_file is not None and now - last_save > cfg.checkpoint_every_s:
            save()
            last_save = now
        if deadline is not None and time.time() > deadline:
            save()
            log(f"  deadline reached after {n_steps} steps; state saved to {state_file}")
            return None

    for i, r in enumerate(reps):
        r["restarts"] = int(restarts[i])
        r["final_time_fs"] = round(float(clock[i]), 2)
    save()
    out = {k: np.asarray(v) for k, v in rec.items()}
    out.update(replicas=reps, n_steps=n_steps, config=cfg.to_dict(), tags=dict(tags or {}),
               wall_seconds=round(wall_before + time.perf_counter() - t_start, 2),
               n_atoms=n_atoms, n_members=committee.n_members)
    return out
