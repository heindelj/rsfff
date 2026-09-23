"""How good is a committee on labeled frames, and the assess stage built on it.

:func:`score_frames` evaluates a committee on labeled frames (the rsfff training schema:
forces in Hartree/bohr, energy in Hartree) and returns, per frame, the committee-mean errors
and the committee's own spread, in kJ/mol units a chemist reads:

    e_err     |Ebar - E| per water, kJ/mol
    f_mae     mean |Fbar - F| over components, kJ/mol/A
    sigma_e   std of the total energy over members, per water, kJ/mol
    sigma_f   DeePMD max-atom force deviation, kJ/mol/A

:class:`CommitteeAssess` runs it on this iteration's labels with the new committee and the one
the iteration started from, per size. Those frames were chosen because the old committee
disagreed on them and are now (90% of them) in training, so what the numbers answer is
"did the new labels get learned" (``f_mae_new`` vs ``f_mae_old``) and "did the disagreement
the sampler chased go away" (``sigma_drop``). A held-out test that is never trained on is
the next thing to add (AL4 on the board).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from easyal import Assess

__all__ = ["score_frames", "summarize", "CommitteeAssess"]

KJMOL = 2625.4996394798254
BOHR_PER_ANGSTROM = 1.0 / 0.529177210903


def score_frames(committee, frames, *, batch: int = 16) -> list[dict]:
    import torch

    from .committee import Topology

    by_topology = defaultdict(list)
    for i, f in enumerate(frames):
        key = (tuple(f["arrays"]["species"]), tuple(int(v) for v in f["arrays"]["fragment_idx"]))
        by_topology[key].append(i)
    rows: list[dict | None] = [None] * len(frames)
    z_of = {"O": 8, "H": 1}
    for (species, frag), idx in by_topology.items():
        top = Topology.from_arrays([z_of[s] for s in species], frag, device=committee.device)
        n_water = max(frag) + 1
        for s in range(0, len(idx), batch):
            chunk = idx[s:s + batch]
            pos = torch.tensor(np.stack([np.asarray(frames[i]["arrays"]["pos"], float)
                                         for i in chunk]))
            ev = committee.evaluate(pos, top)
            e = ev.energy.cpu().numpy()                         # (K, B)
            fm = ev.mean_forces.cpu().numpy()                   # (B, N, 3) Ha/A
            se = ev.sigma_energy.cpu().numpy()
            sf = ev.sigma_forces.cpu().numpy()
            for b, i in enumerate(chunk):
                info, arr = frames[i]["info"], frames[i]["arrays"]
                row = {"n_waters": n_water, "sigma_e": se[b] / n_water * KJMOL,
                       "sigma_f": sf[b] * KJMOL, "failed": bool(ev.failed[b])}
                if "energy" in info:
                    row["e_err"] = abs(e[:, b].mean() - float(info["energy"])) / n_water * KJMOL
                if "forces" in arr:
                    ref = np.asarray(arr["forces"], float) * BOHR_PER_ANGSTROM
                    row["f_mae"] = float(np.abs(fm[b] - ref).mean()) * KJMOL
                rows[i] = row
    return rows


def summarize(rows, key_fn=lambda r: r["n_waters"]) -> dict:
    groups = defaultdict(list)
    for r in rows:
        groups[key_fn(r)].append(r)
    out = {}
    for k, rs in sorted(groups.items()):
        out[k] = {"n": len(rs)}
        for m in ("e_err", "f_mae", "sigma_e", "sigma_f"):
            vals = [r[m] for r in rs if m in r and np.isfinite(r[m])]
            if vals:
                out[k][m] = round(float(np.mean(vals)), 4)
    return out


class CommitteeAssess(Assess):
    """New committee vs the iteration's starting committee on this iteration's labels.

    ``device`` (``auto``); ``n_iterations`` -- the loop is converged when this many
    iterations are complete (default: the length of the size schedule).
    """

    def run(self, ctx):
        from easyal import read_extxyz

        from .committee import Committee

        p = ctx.params
        frames = read_extxyz(ctx.output_of("label"))
        new = Committee.load(ctx.input, device=p.get("device", "auto"))
        rows_new = score_frames(new, frames)
        old = Committee.load(ctx.model, device=p.get("device", "auto"))
        rows_old = score_frames(old, frames)
        s_new, s_old = summarize(rows_new), summarize(rows_old)
        per_size = {}
        for n in s_new:
            a, b = s_new[n], s_old.get(n, {})
            per_size[str(n)] = {"n": a["n"], **{f"{k}_new": v for k, v in a.items() if k != "n"},
                                **{f"{k}_old": v for k, v in b.items() if k != "n"}}
        mean = lambda rows, k: float(np.mean([r[k] for r in rows if k in r]))  # noqa: E731
        metrics = {
            "n_frames": len(frames),
            "f_mae_new": mean(rows_new, "f_mae"), "f_mae_old": mean(rows_old, "f_mae"),
            "e_err_new": mean(rows_new, "e_err"), "e_err_old": mean(rows_old, "e_err"),
            "sigma_f_new": mean(rows_new, "sigma_f"), "sigma_f_old": mean(rows_old, "sigma_f"),
            "per_size": per_size,
        }
        metrics["sigma_drop"] = metrics["sigma_f_new"] / max(metrics["sigma_f_old"], 1e-12)
        metrics["iteration"] = ctx.iteration
        return metrics

    def converged(self, history):
        from . import schedule as sched

        n = int(self.params.get("n_iterations") or len(self.params.get("schedule")
                                                           or sched.DEFAULT_SCHEDULE))
        return len(history) >= n
