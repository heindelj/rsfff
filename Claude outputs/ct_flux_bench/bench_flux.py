"""Where does the flux-CT complete model spend its time?  NaCl rock-salt clusters of growing size.

Timers wrap the model's stages (monkeypatched) and the torch.linalg.solve calls inside them.  Modes:
  fwd    no_grad forward (inference energy)
  train  forward + backward of the energy wrt parameters (energy-only training step)
  force  forward + forces by autograd wrt positions (MD / force evaluation)
"""
import argparse, json, time, math
from collections import defaultdict
from pathlib import Path

import torch
from rsfff.ff.complete import CompleteModel, DensityEnvironment, RadialTable, batched_precompute
from rsfff.ff.complete import model as CM
from rsfff.ff.ionct import ct as CT
from rsfff.ff.ionct.params import IonTable

torch.set_default_dtype(torch.float64)
PARAMS = Path.home() / "rsfff/params/ions_wb97mv_tzvpd_pf2p.json"
T = defaultdict(float)
CNT = defaultdict(int)
STACK = []


def timed(name, fn):
    def w(*a, **k):
        STACK.append(name)
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            T[name] += time.perf_counter() - t0
            CNT[name] += 1
            STACK.pop()
    return w


# the dense solves, attributed to the stage that calls them
_solve = torch.linalg.solve
def solve(A, B, *a, **k):
    t0 = time.perf_counter()
    out = _solve(A, B, *a, **k)
    T[f"  linalg.solve[{STACK[-1] if STACK else '?'}] {tuple(A.shape)}x{B.shape[-1]}"] += time.perf_counter() - t0
    return out
torch.linalg.solve = solve

for meth in ("frozen_terms", "xpol_solve", "flux_operator", "flux_solve", "direct_ct_env", "confinement",
             "alpha_inverse", "xpol_pairs"):
    setattr(CM.CompleteModel, meth, timed(meth, getattr(CM.CompleteModel, meth)))
CM.flux_pair_terms = timed("  flux_pair_terms", CM.flux_pair_terms)


def cluster(n_side, a=2.82, jitter=0.08, seed=0):
    g = torch.tensor([(i, j, k) for i in range(n_side[0]) for j in range(n_side[1]) for k in range(n_side[2])],
                     dtype=torch.float64)
    Z = torch.where(g.sum(1) % 2 == 0, 11, 17)
    X = g * a + jitter * torch.randn(g.shape[0], 3, generator=torch.Generator().manual_seed(seed))
    return X - X.mean(0), Z


SIZES = {2: (2, 1, 1), 8: (2, 2, 2), 18: (3, 3, 2), 27: (3, 3, 3), 36: (4, 3, 3), 48: (4, 4, 3), 64: (4, 4, 4),
         100: (5, 5, 4), 125: (5, 5, 5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="2,8,18,27,36,48,64")
    ap.add_argument("--modes", default="fwd,train,force")
    ap.add_argument("--ct", default="flux")
    ap.add_argument("--cutoff", type=float, default=None, help="override the CT channel cutoff (A)")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", default="bench.jsonl")
    ap.add_argument("--atomic", action="store_true", help="atomic-space flux solve (flux_atomic.py)")
    a = ap.parse_args()
    if a.atomic:
        import sys; sys.path.insert(0, str(Path(__file__).parent))
        import flux_atomic as FA
        CM.CompleteModel.flux_operator = timed("flux_operator", FA.flux_operator_atomic)
        CM.CompleteModel.flux_solve = timed("flux_solve", FA.flux_solve_atomic)
    tab, chan = IonTable.load(PARAMS), CT.ChannelTable.load(PARAMS)
    if a.cutoff is not None:
        chan.cutoff = a.cutoff
    table = RadialTable(tab, h=0.05)
    feat = DensityEnvironment(tab)
    torch.manual_seed(3)
    m = CompleteModel(tab, chan, feat.n_feat, feature_names=feat.feature_names, pseudo=False, frozen=True,
                      pauli_form="shells", confine=("Cl",), env_cap=3.0, xpol=True, xpol_perp=True,
                      xpol_kappa_env="scalefree", ct_model=a.ct)
    # warm the lazily built tables
    X, Z = cluster(SIZES[2])
    with torch.no_grad():
        m(batched_precompute(X, Z, torch.zeros(2, dtype=torch.long), tab, table, chan, inv_form=m.inv_form))
    f = open(a.out, "a")
    for n in map(int, a.sizes.split(",")):
        X0, Z = cluster(SIZES[n])
        fr = torch.zeros(n, dtype=torch.long)
        for mode in a.modes.split(","):
            best = None
            for rep in range(a.reps):
                T.clear(); CNT.clear()
                m.zero_grad(set_to_none=True)
                t0 = time.perf_counter()
                if mode == "fwd":
                    with torch.no_grad():
                        tp = time.perf_counter()
                        bt = batched_precompute(X0, Z, fr, tab, table, chan, inv_form=m.inv_form)
                        T["precompute"] += time.perf_counter() - tp
                        tf = time.perf_counter()
                        o = m(bt)
                        T["forward(total)"] += time.perf_counter() - tf
                elif mode == "train":
                    with torch.no_grad():
                        tp = time.perf_counter()
                        bt = batched_precompute(X0, Z, fr, tab, table, chan, inv_form=m.inv_form)
                        T["precompute"] += time.perf_counter() - tp
                    tf = time.perf_counter()
                    o = m(bt)
                    T["forward(total)"] += time.perf_counter() - tf
                    tb = time.perf_counter()
                    (o["e_ct"].sum() + o["e_pol"].sum() + o["e_int"].sum()).backward()
                    T["backward"] += time.perf_counter() - tb
                else:
                    X = X0.clone().requires_grad_(True)
                    tp = time.perf_counter()
                    bt = batched_precompute(X, Z, fr, tab, table, chan, inv_form=m.inv_form)
                    T["precompute"] += time.perf_counter() - tp
                    tf = time.perf_counter()
                    o = m(bt)
                    T["forward(total)"] += time.perf_counter() - tf
                    tb = time.perf_counter()
                    F, = torch.autograd.grad(o["e_int"].sum(), X)
                    T["backward"] += time.perf_counter() - tb
                T["TOTAL"] = time.perf_counter() - t0
                if best is None or T["TOTAL"] < best["TOTAL"]:
                    best = dict(T)
            nC = int(bt["ch_index"].shape[1])
            rec = {"ct": a.ct + ("-atomic" if a.atomic else ""), "cutoff": chan.cutoff, "n": n, "mode": mode, "nC": nC, "nmax": int(bt["nmax"]),
                   "e_ct": float(o["e_ct"].sum()), "e_pol": float(o["e_pol"].sum()), "e_int": float(o["e_int"].sum()),
                   "t": best}
            f.write(json.dumps(rec) + "\n"); f.flush()
            print(f"\n== {rec['ct']} cut={chan.cutoff} n={n} {mode}: channels {nC}  E_ct {rec['e_ct']*2625.5:.3f} kJ/mol  "
                  f"TOTAL {best['TOTAL']:.3f}s")
            for k, v in sorted(best.items(), key=lambda kv: -kv[1]):
                if v > 0.002:
                    print(f"   {k:60s} {v:8.3f}")


if __name__ == "__main__":
    main()
