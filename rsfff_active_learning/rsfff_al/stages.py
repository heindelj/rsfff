"""easyAL stages for the uncertainty-driven walk: build -> dynamics -> select (-> label -> train
-> assess, stages 2 and 3 of the plan).

    build     PackmolBuild           (H2O)n packings for this iteration's sizes, not minimized
    dynamics  UncertaintyDynamics    committee-biased Langevin, replicas batched per size
    select    SelectUncertain        k most uncertain per size, EDA subset stamped on the frames

``ctx.model`` is a committee: a directory with ``committee.json`` (the train stage's output,
or ``film_committee_100k`` as the initial model), and every stage that needs it loads it with
:meth:`rsfff_al.committee.Committee.load`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from easyal import Build, Contract, Pending, Sample, new_frame

from . import schedule as sched
from .build import pack_waters

__all__ = ["PackmolBuild", "UncertaintyDynamics", "SelectUncertain", "STRUCTURE", "SAMPLED",
           "SELECTED"]

CARRIED = ("charge", "multiplicity", "n_waters", "n_fragments", "fragment_charges",
           "fragment_multiplicities", "traj_id")
STRUCTURE = Contract(info=CARRIED, arrays=["fragment_idx"])
SAMPLED = Contract(info=(*CARRIED, "time_fs", "sigma_energy", "sigma_forces",
                         "committee_energies"), arrays=["fragment_idx"])
SELECTED = Contract(info=(*SAMPLED.info, "labels"), arrays=["fragment_idx"])


def _plan(ctx):
    p = ctx.params
    return sched.plan_iteration(
        ctx.iteration, schedule=p.get("schedule"), eda_policy=p.get("eda_policy"),
        pool_multiplier=p.get("pool_multiplier", 4.0),
        candidates_per_trajectory=p.get("candidates_per_trajectory", 100),
        min_trajectories=p.get("min_trajectories", 2))


class PackmolBuild(Build):
    """One packing per trajectory the schedule asks for (see :mod:`rsfff_al.schedule`).

    ``schedule``, ``eda_policy``, ``pool_multiplier``, ``candidates_per_trajectory``,
    ``min_trajectories``   the budget (pass the same values to ``SelectUncertain``)
    ``tolerance`` (2.0 A), ``density`` (1.0), ``padding`` (1.5 A)   packmol
    ``seed``   base seed; iteration i, size n, trajectory k packs with a seed of its own
    """

    produces = STRUCTURE

    def run(self, ctx):
        p = ctx.params
        plan = _plan(ctx)
        seed0 = int(p.get("seed", 20260922))
        frames, failed = [], []
        for row in plan:
            n = row["n_waters"]
            for k in range(row["trajectories"]):
                seed = seed0 + 1_000_000 * ctx.iteration + 1000 * n + k
                try:
                    packed = pack_waters(n, seed=seed, workdir=ctx.scratch,
                                         tolerance=p.get("tolerance", 2.0),
                                         density=p.get("density", 1.0),
                                         padding=p.get("padding", 1.5))
                except RuntimeError as exc:
                    failed.append(str(exc))
                    continue
                info = dict(packed["info"], traj_id=f"i{ctx.iteration:02d}_w{n:03d}_t{k:03d}")
                frames.append(new_frame(packed["species"], packed["positions"], info,
                                        fragment_idx=[m for m in range(n) for _ in range(3)]))
        if failed:
            ctx.note(f"packmol failed {len(failed)} times: {failed[:3]}")
        ctx.log(n_structures=len(frames), n_failed=len(failed), plan=plan)
        return frames


class UncertaintyDynamics(Sample):
    """Committee-biased Langevin from every packing; writes the candidate pool.

    Parameters are :class:`rsfff_al.dynamics.DynamicsConfig` fields (``time_ps``,
    ``bias_weight``, ``bias_mode``, ``temperature_K``, ...) plus

    ``device``          ``auto`` | ``cuda`` | ``cpu``
    ``max_replicas``    replicas per batch (all of one size), default 16
    ``window_fs``       one candidate per this much trajectory (default 100)
    ``include_warmup``  let warmup frames into the pool (default True: the relaxation out of
                        a raw packing is data too)
    ``score``           ``both`` | ``energy`` | ``forces``, for choosing each window's frame
    ``time_budget_s``   stop after this long and report Pending (the batch checkpoints, so the
                        next run continues it); default: no budget

    Each batch leaves ``scratch/w{n}_b{j}.npz`` (every recorded frame, with the committee
    energies) and ``.json`` (per-replica failures and restarts). A batch with its ``.npz`` is
    never rerun.
    """

    name = "dynamics"
    requires = STRUCTURE
    produces = SAMPLED

    def run(self, ctx):
        import torch

        from .committee import Committee, Topology
        from .dynamics import DynamicsConfig, run_replicas

        p = dict(ctx.params)
        deadline = time.time() + float(p["time_budget_s"]) if p.get("time_budget_s") else None
        fields = set(DynamicsConfig.__dataclass_fields__)
        cfg_kw = {k: v for k, v in p.items() if k in fields}
        committee = Committee.load(ctx.model, device=p.get("device", "auto"))
        ctx.log(committee=committee.describe())

        frames = ctx.read()
        by_size: dict[int, list[dict]] = {}
        for f in frames:
            by_size.setdefault(int(f["info"]["n_waters"]), []).append(f)
        max_rep = int(p.get("max_replicas", 16))
        batches = [(n, j, group[s:s + max_rep])
                   for n, group in sorted(by_size.items())
                   for j, s in enumerate(range(0, len(group), max_rep))]

        n_done = 0
        for n, j, group in batches:
            stem = ctx.scratch / f"w{n:03d}_b{j:02d}"
            if Path(f"{stem}.npz").exists():
                n_done += 1
                continue
            if deadline is not None and time.time() > deadline:
                raise Pending(f"time budget spent: {n_done}/{len(batches)} batches done")
            cfg = DynamicsConfig(**{**cfg_kw, "seed": int(p.get("seed", 0)) + 7919 * n + j
                                    + 104729 * ctx.iteration})
            top = Topology.water(n, device=committee.device)
            start = np.stack([np.asarray(f["arrays"]["pos"], float) for f in group])
            ctx.note(f"{n} waters, batch {j}: {len(group)} replicas, "
                     f"{cfg.warmup_fs:.0f} fs warmup + {cfg.time_ps} ps")
            out = run_replicas(committee, top, start, cfg, checkpoint=f"{stem}.state.pt",
                               log=lambda s: print(s, flush=True),
                               deadline=deadline)
            if out is None:
                raise Pending(f"time budget spent inside {stem.name}; it resumes from "
                              f"its checkpoint")
            traj_ids = [f["info"]["traj_id"] for f in group]
            np.savez_compressed(
                f"{stem}.tmp.npz", **{k: v for k, v in out.items()
                                      if isinstance(v, np.ndarray)},
                traj_id=np.array(traj_ids))
            Path(f"{stem}.tmp.npz").replace(f"{stem}.npz")
            Path(f"{stem}.json").write_text(json.dumps(
                {k: out[k] for k in ("replicas", "n_steps", "wall_seconds", "config",
                                     "n_atoms", "n_members")}
                | {"traj_id": traj_ids}, indent=1))
            Path(f"{stem}.state.pt").unlink(missing_ok=True)
            n_done += 1

        return self._pool(ctx, by_size, batches)

    def _pool(self, ctx, by_size, batches):
        from .select import pool_indices, scores

        p = ctx.params
        out, per_size = [], {}
        for n, j, group in batches:
            stem = ctx.scratch / f"w{n:03d}_b{j:02d}"
            d = np.load(f"{stem}.npz")
            meta = json.loads(Path(f"{stem}.json").read_text())
            if len(d["replica"]) == 0:
                continue
            sc = scores(d["sigma_energy"], d["sigma_forces"], p.get("score", "both"))
            rows = pool_indices(d["replica"], d["time_fs"], sc,
                                window_fs=float(p.get("window_fs", 100.0)), phase=d["phase"],
                                include_warmup=bool(p.get("include_warmup", True)))
            src = group[0]["info"]
            fails = sum(len(r["failures"]) for r in meta["replicas"])
            retired = sum(r["status"] == "retired" for r in meta["replicas"])
            stat = per_size.setdefault(n, {"replicas": 0, "failures": 0, "retired": 0,
                                           "frames": 0, "candidates": 0, "seconds": 0.0,
                                           "steps": 0})
            stat["replicas"] += len(group)
            stat["failures"] += fails
            stat["retired"] += int(retired)
            stat["frames"] += int(len(d["replica"]))
            stat["candidates"] += int(len(rows))
            stat["seconds"] += float(meta["wall_seconds"])
            stat["steps"] += int(meta["n_steps"])
            stat.setdefault("sigma_energy_per_water_mHa", []).append(
                float(np.median(d["sigma_energy"])) / n * 1e3)
            stat.setdefault("bias_ratio", []).append(float(np.median(d["bias_ratio"])))
            for r in rows:
                rep = int(d["replica"][r])
                info = {k: src[k] for k in CARRIED if k != "traj_id"}
                info.update(
                    traj_id=str(d["traj_id"][rep]),
                    source=f"{group[rep]['info'].get('source', '')}|udd{d['time_fs'][r]:.0f}",
                    time_fs=round(float(d["time_fs"][r]), 2),
                    phase="warmup" if int(d["phase"][r]) == 0 else "production",
                    restart=int(d["restart"][r]),
                    fs_to_failure=round(float(d["fs_to_failure"][r]), 1),
                    sigma_energy=float(d["sigma_energy"][r]),
                    sigma_forces=float(d["sigma_forces"][r]),
                    committee_energies=[float(e) for e in d["energies"][r]],
                    committee_mean_energy=float(np.mean(d["energies"][r])),
                    max_force=float(d["max_force"][r]),
                    temperature=round(float(d["temperature"][r]), 1),
                    bias_ratio=float(d["bias_ratio"][r]),
                )
                out.append(new_frame(["O", "H", "H"] * n, d["positions"][r], info,
                                     fragment_idx=[m for m in range(n) for _ in range(3)]))
        for stat in per_size.values():
            for key in ("sigma_energy_per_water_mHa", "bias_ratio"):
                stat[key] = round(float(np.median(stat[key])), 6)
            stat["s_per_step"] = round(stat["seconds"] / max(stat["steps"], 1), 4)
        ctx.log(per_size=per_size, n_candidates=len(out))
        if not out:
            raise RuntimeError("no candidate frames survived the dynamics")
        return out


class SelectUncertain(Sample):
    """The ``labels`` most uncertain candidates per size; ``n_eda`` of them get an EDA.

    Budget parameters as in :class:`PackmolBuild` (``schedule``, ``eda_policy``); ``score``
    (``both``); ``max_per_traj`` (default ``ceil(2 k / trajectories)``);
    ``exclude_failure_within_fs`` drops candidates that close before a blow-up (default 0:
    keep them -- where a model breaks is where it most needs data).

    Each selected frame gets ``labels = "force,eda"`` or ``"force"``, which is all the label
    stage reads to decide which Q-Chem jobs to submit.
    """

    name = "select"
    requires = SAMPLED
    produces = SELECTED

    def run(self, ctx):
        from .select import choose, scores

        p = ctx.params
        plan = {r["n_waters"]: r for r in _plan(ctx)}
        frames = ctx.read()
        by_size: dict[int, list[int]] = {}
        for i, f in enumerate(frames):
            by_size.setdefault(int(f["info"]["n_waters"]), []).append(i)
        out, stats = [], {}
        for n, rows in sorted(by_size.items()):
            want = plan.get(n)
            if want is None:
                continue
            info = [frames[i]["info"] for i in rows]
            sc = scores([x["sigma_energy"] for x in info], [x["sigma_forces"] for x in info],
                        p.get("score", "both"))
            window = float(p.get("exclude_failure_within_fs", 0.0))
            exclude = [0 <= float(x.get("fs_to_failure", -1)) < window for x in info]
            chosen, eda = choose(sc, [x["traj_id"] for x in info], want["labels"],
                                 n_eda=want["eda"], max_per_traj=p.get("max_per_traj"),
                                 exclude=exclude)
            eda_set = set(eda.tolist())
            for rank, c in enumerate(chosen):
                f = frames[rows[c]]
                new_info = dict(f["info"], labels="force,eda" if c in eda_set else "force",
                                select_rank=rank, select_score=float(sc[c]),
                                selected_from=len(rows))
                out.append({"info": new_info, "arrays": dict(f["arrays"])})
            stats[n] = {"pool": len(rows), "chosen": int(len(chosen)), "eda": int(len(eda)),
                        "short": max(0, want["labels"] - len(chosen)),
                        "median_score_chosen": float(np.median(sc[chosen])) if len(chosen) else None}
        short = {n: s["short"] for n, s in stats.items() if s["short"]}
        if short:
            ctx.note(f"fewer candidates than labels for sizes {short}: raise "
                     f"pool_multiplier or time_ps")
        ctx.log(per_size=stats, n_selected=len(out),
                n_eda=sum(s["eda"] for s in stats.values()))
        if not out:
            raise RuntimeError("nothing selected")
        return out
