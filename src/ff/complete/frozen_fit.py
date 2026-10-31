"""The frozen fit: the monatomic ions' permanent shells and the CP Pauli kernel's charges and exponent scales, fitted
together to EDA pair scans (cls_elec, mod_pauli). The first step of the complete model, before any environment
dependence is trained; the training then pulls w and s toward these values (``CompleteModel.pauli_prior``) with the
shells fixed (ct_model's frozen-fit workflow; prototype and form comparison 2026-10-26).

Per pair, the same expressions the model evaluates:

    electrostatics  :func:`rsfff.ff.complete.elst.permanent_electrostatics` on the shells (Z_eff, n_t, b_t), fitted on
                    its penetration part E - q_i q_j / r
    Pauli           sum_tu w_t w_u f(b_t s_i, b_u s_j; r) / r   (``pauli_form = "shells"``: f the Slater overlap
                    complement of unequal exponents), one exponent scale s per species, s = exp(ls) unbounded

``shared`` species keep s = 1 (their Pauli exponents are the electrostatic ones): Li+, whose core is so compact that
the kernel's decay is its partner's and the Pauli data cannot fix its own exponent (s ran to ~10^4 with no change in
the loss).

Loss: weighted mean squares, residual / (|E| / e0 + 1) (relative above e0, absolute below), of the electrostatics +
mu x the Pauli, + lam x sum (log b - log b0)^2 + (log n - log n0)^2 toward the prior shells (ions[s]['shells_esp'],
the monomer ESP fit, else ions[s]['shells']). The prior is what keeps the anions' diffuse outer shells: the
electrostatics barely see them, and the Pauli term alone drains them (I- outer shell 1.1 -> 0.27 e at lam = 0.1).
Points with R >= window / 0.9 x the contact distance (:func:`pair_window`: window (rho_i + rho_j) for an unlike
pair, Shannon radii).

Steps: (1) the shells on the electrostatics alone; (2) w and s on those shells from several s starts (the Pauli loss
has local minima in s); (3) everything together.

    python -m rsfff.ff.complete.frozen_fit --params BASE.json --labels LABELS.jsonl --out OUT.json [--mu 1 --lam 1]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import torch

from ..ionct import kernels as K
from ..ionct.params import IonTable
from ..multipole import slater_two_center_damp_unequal
from .elst import permanent_electrostatics

KJ = 2625.4996
#: Shannon ionic radii (CN 6, Angstrom): the window's scale
RADIUS = {"Li": 0.76, "Na": 1.02, "K": 1.38, "Rb": 1.52, "Cs": 1.67, "F": 1.33, "Cl": 1.81, "Br": 1.96, "I": 2.20}

__all__ = ["PairPoints", "pair_points", "read_labels", "fit", "pauli_energy", "contact", "pair_window", "RADIUS"]


@dataclasses.dataclass
class PairPoints:
    """Two-ion frames: species indices a, b (P,), separation r (P,) Angstrom, the EDA's frozen electrostatics and
    Pauli (P,) kJ/mol (NaN where a label lacks the term), and a key "A-B" per point."""
    a: torch.Tensor
    b: torch.Tensor
    r: torch.Tensor
    elec: torch.Tensor
    pauli: torch.Tensor
    key: list

    def __len__(self):
        return len(self.r)

    def subset(self, m):
        m = torch.as_tensor(m)
        idx = m.nonzero().flatten().tolist() if m.dtype == torch.bool else list(m)
        return PairPoints(self.a[m], self.b[m], self.r[m], self.elec[m], self.pauli[m], [self.key[i] for i in idx])


def pair_points(rows, symbols, *, window: float = 0.75) -> PairPoints:
    """``rows``: dicts with ``symbols`` (two), ``r`` (Angstrom) and the EDA terms ``cls_elec`` and ``mod_pauli``
    (kJ/mol). Keeps the pairs among ``symbols`` with r >= :func:`pair_window`, one point per (pair, r) (the first),
    the pair ordered as in ``symbols``."""
    idx = {s: k for k, s in enumerate(symbols)}
    a, b, r, e, p, key, seen = [], [], [], [], [], [], set()
    for row in rows:
        s1, s2 = row["symbols"]
        if s1 not in idx or s2 not in idx:
            continue
        if idx[s1] > idx[s2]:
            s1, s2 = s2, s1
        rr = float(row["r"])
        k = (s1, s2, round(rr, 4))
        if k in seen or rr < pair_window(s1, s2, window) or row.get("cls_elec") is None:
            continue
        seen.add(k)
        a.append(idx[s1]); b.append(idx[s2]); r.append(rr); key.append(f"{s1}-{s2}")
        e.append(float(row["cls_elec"]))
        p.append(float(row["mod_pauli"]) if row.get("mod_pauli") is not None else float("nan"))
    t = lambda x: torch.tensor(x, dtype=torch.float64)
    return PairPoints(torch.tensor(a, dtype=torch.long), torch.tensor(b, dtype=torch.long), t(r), t(e), t(p), key)


def read_labels(path) -> list[dict]:
    """Rows for :func:`pair_points` from a ChemLab ``{ labelled = "stage" }`` dataset (JSON lines: ``xyz`` the job's
    geometry, the report's fields at the top level or under ``report.data``): every two-atom frame with an EDA."""
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        rep = d.get("report", {}).get("data", d) if isinstance(d.get("report"), dict) else d
        eda = rep.get("es.eda")
        if not eda:
            continue
        sym, pos = _xyz(d.get("xyz") or "")
        if len(sym) != 2:
            continue
        info = _comment(d.get("xyz") or "")
        if info.get("field") not in (None, "", "0") or info.get("pcm"):
            continue
        rows.append({"symbols": sym, "r": float(np.linalg.norm(np.subtract(pos[1], pos[0]))),
                     "cls_elec": eda.get("cls_elec", eda.get("elec")), "mod_pauli": eda.get("mod_pauli", eda.get("pauli")),
                     "disp": eda.get("disp"), "job": d.get("job")})
    return rows


def _xyz(text):
    lines = text.splitlines()
    if len(lines) < 2:
        return [], []
    n = int(lines[0].split()[0])
    sym, pos = [], []
    for ln in lines[2:2 + n]:
        f = ln.split()
        sym.append(f[0]); pos.append([float(x) for x in f[1:4]])
    return sym, pos


def _comment(text):
    lines = text.splitlines()
    out = {}
    for tok in (lines[1].split() if len(lines) > 1 else []):
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v.strip('"')
    return out


def pauli_energy(w, bp, a, b, r_ang):
    """sum_tu w[a, t] w[b, u] f(bp[a, t], bp[b, u]; r) / r (Hartree), the model's shells-form monopole Pauli."""
    rb = r_ang / K.BOHR_ANG
    e = rb.new_zeros(len(rb))
    for t in range(w.shape[1]):
        for u in range(w.shape[1]):
            f = slater_two_center_damp_unequal(bp[a, t], bp[b, u], rb, 0)[0]
            e = e + w[a, t] * w[b, u] * f / rb
    return e


def _table(doc, symbols, key="shells"):
    ions = {s: dict(doc["ions"][s]) for s in symbols}
    for s in symbols:
        ions[s]["shells"] = ions[s].get(key) or ions[s]["shells"]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump({"ions": ions}, fh)
    return IonTable.load(fh.name, symbols)


def _lbfgs(params, loss, iters):
    opt = torch.optim.LBFGS(params, max_iter=iters, line_search_fn="strong_wolfe", tolerance_grad=1e-10, tolerance_change=1e-14)

    def closure():
        opt.zero_grad()
        L = loss()
        L.backward()
        return L
    for _ in range(3):
        opt.step(closure)


def fit(doc: dict, symbols, pts: PairPoints, *, mu: float = 1.0, lam: float = 1.0, shared=("Li",), e0: float = 10.0,
        s_starts=(0.5, 0.75, 1.0, 1.35, 1.8), iters: int = 500, log=print) -> tuple[dict, dict]:
    """The frozen fit of ``symbols`` (every one of them in ``doc['ions']``) to ``pts``. Returns the new document (a
    copy: shells, the prior kept as ``shells_esp``, ``pauli = {w, s}`` and ``fit['frozen']`` per ion) and the metrics."""
    symbols = list(symbols)
    tab0 = _table(doc, symbols, "shells_esp")          # the prior's shells
    S, T = tab0.shell_n.shape
    live = tab0.shell_n > 0
    last = (live.cumsum(1) == live.sum(1, keepdim=True)) & live
    ne = tab0.shell_n.sum(1)
    lb0, ln0 = torch.log(tab0.shell_b), torch.log(tab0.shell_n.clamp(min=1e-3))
    lb, ln = torch.nn.Parameter(lb0.clone()), torch.nn.Parameter(ln0.clone())
    lw = torch.nn.Parameter(torch.where(live, ln0 - 1.0, torch.full_like(lb0, -30.0)))
    ls = torch.nn.Parameter(torch.zeros(S, dtype=torch.float64))
    free_s = torch.tensor([s not in shared for s in symbols], dtype=torch.float64)

    a, b, r = pts.a, pts.b, pts.r
    pos = torch.stack([torch.zeros_like(r), torch.zeros_like(r), torch.zeros_like(r), torch.zeros_like(r),
                       torch.zeros_like(r), r], 1).reshape(-1, 3)
    sp = torch.stack([a, b], 1).reshape(-1)
    fr = torch.arange(len(r)).repeat_interleave(2)
    coul = tab0.charge[a] * tab0.charge[b] / (r / K.BOHR_ANG) * KJ
    pen_ref = pts.elec - coul
    has_p = torch.isfinite(pts.pauli)
    p_ref = torch.nan_to_num(pts.pauli)

    def shells():
        bb = torch.where(live, lb.exp(), tab0.shell_b)
        n = torch.where(live, ln.exp(), tab0.shell_n)
        n = torch.where(last, (ne - (n * (live & ~last)).sum(1))[:, None].expand_as(n), n)
        return n, bb

    def predict():
        n, bb = shells()
        tab = dataclasses.replace(tab0, shell_n=n, shell_b=bb)
        pe = permanent_electrostatics(pos, sp, fr, tab) * KJ - coul
        w = torch.where(live, lw.exp(), torch.zeros_like(lw))
        scale = (ls * free_s).exp()
        pp = pauli_energy(w, bb * scale[:, None], a, b, r) * KJ
        return pe, pp

    def losses():
        pe, pp = predict()
        le = (((pe - pen_ref) / (pen_ref.abs() / e0 + 1.0)) ** 2).mean()
        res_p = torch.where(has_p, (pp - p_ref) / (p_ref.abs() / e0 + 1.0), torch.zeros_like(pp))
        lp = (res_p ** 2).sum() / has_p.sum().clamp(min=1)
        prior = (((lb - lb0) ** 2 + (ln - ln0) ** 2) * live).sum()
        return le, lp, prior

    # (1) the shells on the electrostatics
    _lbfgs([lb, ln], lambda: (lambda L: L[0] + lam * L[2])(losses()), iters)
    with torch.no_grad():
        log(f"  shells on electrostatics: weighted rms {math.sqrt(float(losses()[0])):.3f}")
    # (2) w and s on those shells, from several s starts (the lowest loss kept)
    best = None
    for s0 in s_starts:
        with torch.no_grad():
            ls.copy_(torch.full_like(ls, math.log(s0)))
            lw.copy_(torch.where(live, ln.detach() - 1.0, torch.full_like(lw, -30.0)))
            _, pp = predict()
            ok = has_p & (p_ref > 1) & (pp > 0)
            if ok.any():   # a common offset first: E scales as exp(2 d)
                lw += 0.5 * torch.median(torch.log(p_ref[ok] / pp[ok]))
        _lbfgs([lw], lambda: losses()[1], iters)
        _lbfgs([lw, ls], lambda: losses()[1], iters)
        with torch.no_grad():
            L = float(losses()[1])
        log(f"  pauli from s = {s0}: weighted rms {math.sqrt(L):.4f}")
        if best is None or L < best[0]:
            best = (L, lw.detach().clone(), ls.detach().clone())
    with torch.no_grad():
        lw.copy_(best[1]); ls.copy_(best[2])
    pauli_only = math.sqrt(best[0])
    # (3) together
    _lbfgs([lb, ln, lw, ls], lambda: (lambda L: L[0] + mu * L[1] + lam * L[2])(losses()), iters)

    with torch.no_grad():
        le, lp, pr = (float(x) for x in losses())
        pe, pp = predict()
        n, bb = shells()
        w = torch.where(live, lw.exp(), torch.zeros_like(lw))
        scale = (ls * free_s).exp()
    metrics = {"weighted_rms": {"elec": math.sqrt(le), "pauli": math.sqrt(lp), "pauli_on_elec_shells": pauli_only},
               "prior": pr, "mu": mu, "lam": lam, "e0": e0, "shared": list(shared), "points": len(pts),
               "pairs": _pair_metrics(pts, pe, pen_ref, pp, p_ref, has_p)}
    metrics["median_contact"] = {t: _median([v[f"{t}_rms_contact"] for v in metrics["pairs"].values()]) for t in ("elec", "pauli")}
    log(f"  joint (mu {mu}, lam {lam}): weighted rms elec {math.sqrt(le):.3f}, pauli {math.sqrt(lp):.3f} "
        f"(on the elec shells {pauli_only:.3f}); median contact rms elec {metrics['median_contact']['elec']:.2f}, "
        f"pauli {metrics['median_contact']['pauli']:.2f} kJ/mol")
    out = json.loads(json.dumps(doc))
    for k, s in enumerate(symbols):
        ion = out["ions"][s]
        ion.setdefault("shells_esp", ion["shells"] if "shells_esp" not in doc["ions"][s] else doc["ions"][s]["shells_esp"])
        ion["shells"] = [{"n": float(n[k, t]), "b": float(bb[k, t])} for t in range(T) if live[k, t]]
        ion["pauli"] = {"w": [float(w[k, t]) for t in range(T) if live[k, t]], "s": float(scale[k]),
                        "form": "shells", "shared": s in shared}
        ion["shells_note"] = (f"frozen fit (rsfff.ff.complete.frozen_fit): shells + CP Pauli on EDA pair scans, "
                              f"mu {mu}, lam {lam}, R >= pair_window")
        ion.setdefault("fit", {})["frozen"] = {key: v for key, v in metrics["pairs"].items() if s in key.split("-")}
    return out, metrics


def contact(a: str, b: str) -> float:
    """Where a pair sits in a crystal or cluster (Angstrom): unlike 0.9 (rho_a + rho_b); like 0.9 sqrt2 (rho + the
    smallest counter-ion), the closest a like pair gets."""
    cat = {"Li", "Na", "K", "Rb", "Cs"}
    if (a in cat) != (b in cat):
        return 0.9 * (RADIUS[a] + RADIUS[b])
    partner = min(RADIUS[x] for x in (set(RADIUS) - cat if a in cat else cat))
    return 0.9 * math.sqrt(2) * (0.5 * (RADIUS[a] + RADIUS[b]) + partner)


def pair_window(a: str, b: str, window: float = 0.75) -> float:
    """The shortest distance read (Angstrom): window / 0.9 x the contact -- window (rho_a + rho_b) for an unlike pair,
    the same fraction of its own contact for a like pair (whose compressed geometries no condensed phase reaches)."""
    return window / 0.9 * contact(a, b)


def _pair_metrics(pts, pe, pen_ref, pp, p_ref, has_p):
    out = {}
    for key in sorted(set(pts.key)):
        m = torch.tensor([k == key for k in pts.key])
        a, b = key.split("-")
        mc = m & (pts.r >= contact(a, b)) & (pts.r <= contact(a, b) + 1.5)
        rms = lambda x, mm: float(x[mm].pow(2).mean().sqrt()) if mm.any() else float("nan")
        out[key] = {"n": int(m.sum()), "elec_rms": rms(pe - pen_ref, m), "elec_rms_contact": rms(pe - pen_ref, mc),
                    "pauli_rms": rms(pp - p_ref, m & has_p), "pauli_rms_contact": rms(pp - p_ref, mc & has_p)}
    return out


def _median(vals):
    v = [x for x in vals if math.isfinite(x)]
    return float(np.median(v)) if v else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--params", required=True, help="the base ion table (shells_esp = the prior, else shells)")
    ap.add_argument("--labels", required=True, nargs="+", help="JSON lines of labelled pair scans (ChemLab labelled)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ions", nargs="*", default=None, help="default: every ion with pair points")
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--shared", nargs="*", default=["Li"])
    ap.add_argument("--window", type=float, default=0.75)
    ap.add_argument("--metrics", default=None, help="write the metrics here (JSON)")
    a = ap.parse_args(argv)
    doc = json.load(open(a.params))
    rows = [r for p in a.labels for r in read_labels(p)]
    have = sorted({s for r in rows for s in r["symbols"]} & set(doc["ions"]), key=list(RADIUS).index)
    symbols = a.ions or have
    pts = pair_points(rows, symbols, window=a.window)
    print(f"{len(pts)} points over {len(set(pts.key))} pairs of {', '.join(symbols)}", flush=True)
    out, metrics = fit(doc, symbols, pts, mu=a.mu, lam=a.lam, shared=tuple(a.shared))
    json.dump(out, open(a.out, "w"), indent=1)
    if a.metrics:
        json.dump(metrics, open(a.metrics, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
