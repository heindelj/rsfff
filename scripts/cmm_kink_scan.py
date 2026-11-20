#!/usr/bin/env python
"""Find kinks of a cmm model along smooth paths: jumps of the first and second derivative.

    python scripts/cmm_kink_scan.py MODEL.pt [--params TABLE.json] [--step 0.002] [--order 1 2] [--set KEY=VALUE ...]

Paths: the eight cation-anion / anion-anion dimers of LiF, NaCl, KBr, CsI, the four like-cation dimers, and a
uniformly scaled rock-salt cube of each salt. Along each, on a grid of ``--step`` Angstrom: the raw readouts u, the CT
log-factors ln q / q0, the encoder's inputs and every energy term (kJ/mol). A jump of the k-th derivative is an isolated
spike of the (k + 1)-th difference, reported with its position, its size (per A^k) and the smooth background there.
Order 1 is a kink (a jump in the forces if it is in an energy); order 2 a jump in the curvature.

``--set`` overrides config fields of the loaded model before the scan (same weights, another functional form), e.g.
``--set env_switch=smooth ct_q_headroom=0.2`` to see what a kink is made of. Sources known on 2026-11-18
(claude/cmm_smoothness.md): the clip of the one-sided CT charges (order 1, ``ct_q_headroom``), the cubic step of the
overlap windows (order 2, ``env_switch``); the quintic channel taper at cutoff - taper is C^2 (order 3).
"""
import argparse
import ast

import numpy as np
import torch

from rsfff.ff.cmm import load_cmm
from rsfff.ff.cmm.encoder import READOUTS
from rsfff.train import cmm_data as D

KJ = 2625.4996394799
SALTS = {"LiF": ("Li", "F", 2.01), "NaCl": ("Na", "Cl", 2.82), "KBr": ("K", "Br", 3.30), "CsI": ("Cs", "I", 3.95)}


def paths(step, symbols):
    """{name: (x, symbols, [positions (n, 3)])}"""
    P = {}
    cube = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], float)
    for salt, (c, a, d0) in SALTS.items():
        if c not in symbols or a not in symbols:
            continue
        for p, q, lo in ((c, a, 0.62 * d0), (a, a, 0.95 * d0), (c, c, 0.8 * d0)):
            x = np.arange(lo, 8.0, step)
            P[f"{p}-{q}"] = (x, [p, q], [np.array([[0, 0, 0], [0, 0, r]], float) for r in x])
        s = [c if (i + j + k) % 2 == 0 else a for i, j, k in cube.astype(int)]
        x = np.arange(0.78 * d0, 1.5 * d0, step)
        P[f"{salt} cube"] = (x, s, [cube * r for r in x])
    return P


def evaluate(m, sym, X, chunk=256):
    """{quantity: (n, columns)} along the path; per-atom quantities for the first two atoms."""
    n, out = len(sym), {}
    Z = [D.ATOMIC_NUMBER[a] for a in sym]
    for k in range(0, len(X), chunk):
        Xs = X[k:k + chunk]
        bt = m.precompute(torch.tensor(np.concatenate(Xs)), torch.tensor(Z * len(Xs)),
                          torch.arange(len(Xs)).repeat_interleave(n))
        o = m(bt)
        sp = bt["species"]
        r = {f"u {q}": m._u[q].reshape(len(Xs), n)[:, :2] for q in READOUTS}
        lq = (torch.log(m.ct_charges(sp)) - m.ct_ln_q0[sp]).reshape(len(Xs), n, 2)[:, :2]
        r["ln qD/qD0"], r["ln qA/qA0"] = lq[..., 0], lq[..., 1]
        Xc, G = m._inputs(bt)
        r["input chan"] = Xc.reshape(len(Xs), n, -1)[:, :2].reshape(len(Xs), -1)
        if G is not None:
            r["input gated"] = G.reshape(len(Xs), n, -1)[:, :2].reshape(len(Xs), -1)
        for q in ("elst", "pauli", "disp", "pol", "ct", "int"):
            r[f"E {q}"] = o[f"e_{q}"][:, None] * KJ
        for q, v in r.items():
            out.setdefault(q, []).append(v.numpy())
    return {q: np.concatenate(v) for q, v in out.items()}


def jumps(x, y, order, ratio=15.0, w=12, skip=3):
    """[(position, jump of the order-th derivative, background)]: isolated spikes of the (order + 1)-th difference."""
    h = x[1] - x[0]
    d = np.diff(y, n=order + 1) / h ** order
    a = np.abs(d)
    floor = 1e-9 * np.abs(y).max() / h ** order
    out = []
    for i in range(w, len(d) - w):
        ref = np.median(np.concatenate([a[i - w:i - skip], a[i + skip + 1:i + w + 1]])) + floor
        if a[i] > ratio * ref and a[i] >= a[i - skip:i + skip + 1].max():
            out.append((x[i + (order + 1) // 2], d[i - 1:i + 2].sum(), ref))     # a jump between two grid points
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--params", default=None, help="the species table (default: the checkpoint's)")
    ap.add_argument("--step", type=float, default=0.002)
    ap.add_argument("--order", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides for the scan")
    ap.add_argument("--min", type=float, default=1e-6,
                    help="smallest quantity (its largest value on the path) reported")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    m = load_cmm(a.model, params=a.params).eval()
    m.table()
    for kv in a.set:
        k, v = kv.split("=", 1)
        try:
            v = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            pass
        setattr(m.config, k, tuple(v) if isinstance(v, list) else v)
    m.config.__post_init__()
    total = {o: 0 for o in a.order}
    for name, (x, sym, X) in paths(a.step, set(m.tab.symbols)).items():
        r = evaluate(m, sym, X)
        for order in a.order:
            for q, y in r.items():
                for c in range(y.shape[1]):
                    if np.abs(y[:, c]).max() < a.min:
                        continue
                    for pos, j, ref in jumps(x, y[:, c], order):
                        total[order] += 1
                        print(f"{name:10s} order {order}  {q:12s}[{c:2d}]  x = {pos:6.3f} A  jump {j:+.3e}  "
                              f"(background {ref:.1e})")
    print("jumps found:", ", ".join(f"order {o}: {n}" for o, n in total.items()))


if __name__ == "__main__":
    main()
