"""Two-state CT of a trained complete model (``--ct-model twostate``): the NaCl dimer along R and held-out clusters,
linear response (one outer step) against the converged two-state minimum.

    python scripts/twostate_scan.py CACHE.pt RUN/model.pt [--frames 40] [--outer 30]

Prints, per dimer distance, the Cl- -> Na+ gap (raw and floored), the direct energy e_c, the populations q (linear
response / converged) and E_CT; then, over cluster frames, E_CT in both modes and the largest population. Run from
research/rsfff (the runs' relative paths)."""
from __future__ import annotations

import argparse

import torch

from rsfff.ff.complete import batched_precompute
from rsfff.ff.ionct import IonTable
from rsfff.ff.ionct import ct as CT
from rsfff.ff.ionct.params import ATOMIC_NUMBER as Zmap
from rsfff.train.train_ionct import build_model, radial_table

HK = 2625.4996


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("model")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--min-ions", type=int, default=4)
    ap.add_argument("--outer", type=int, default=30)
    a = ap.parse_args()
    ck = torch.load(a.model, weights_only=False)
    args = ck["args"]
    tab, chan = IonTable.load(args["params"]), CT.ChannelTable.load(args["params"])
    model = build_model(args, tab, chan, ck["n_feat"], None, ck.get("feature_names"))
    model.load_state_dict(ck["state"])
    model.eval()
    if model.ct_model != "twostate":
        raise SystemExit(f"{a.model}: ct_model {model.ct_model!r}, not twostate")
    table = radial_table(tab, args["params"], args.get("cache_dir"))
    print("I, A (table + shift):", {x: [round(float(model.ts_I0[k] + model.ts_dI[k]), 4),
                                        round(float(model.ts_A0[k] + model.ts_dA[k]), 4)]
                                    for k, x in enumerate(tab.symbols)}, f"gap floor {model.ct_gap_min}")

    def run(bt, outer):
        model.ct_outer, model.ct_outer_tol = outer, 1e-12
        with torch.no_grad():
            o = model(bt)
        return o, dict(model._ts_info)

    n1 = model.ct_outer
    print(f"\nNaCl dimer, Cl- -> Na+   (E in kJ/mol)\n{'R/A':>5} {'Delta':>8} {'Delta_f':>8} {'e_c':>9} {'q_LR':>9} {'q_conv':>9} "
          f"{'E_LR':>8} {'E_conv':>8}")
    for R in (2.2, 2.4, 2.6, 2.8, 3.0, 3.4, 4.0, 5.0, 6.0, 7.0):
        X = torch.tensor([[0.0, 0, 0], [R, 0, 0]], dtype=torch.float64)
        bt = batched_precompute(X, torch.tensor([11, 17]), torch.zeros(2, dtype=torch.long), tab, table, chan,
                                inv_form=model.inv_form)
        sp = bt["species"]
        with torch.no_grad():
            model.env_prepare(bt, sp) if getattr(model, "env", None) is not None else None
            I = model.scaled(bt["feats"])
            *_, e_c, Dl, Draw = model.twostate_operator(bt, sp, model.ct_charges_from(I, sp), 9 if model.breath else 8)
        o1, _ = run(bt, 1)
        on, _ = run(bt, a.outer)
        c = int((sp[bt["ch_index"][0]] == tab.symbols.index("Cl")).nonzero()[0])
        print(f"{R:5.2f} {float(Draw[c]):8.4f} {float(Dl[c]):8.4f} {float(e_c[c]) * HK:9.3f} {float(o1['p'][c]):9.2e} "
              f"{float(on['p'][c]):9.2e} {float(o1['e_ct'][0]) * HK:8.3f} {float(on['e_ct'][0]) * HK:8.3f}")

    caches = [c for c in torch.load(a.cache, weights_only=False) if c.species.shape[0] >= a.min_ions][:a.frames]
    d, rel, qmax, it = [], [], [], []
    for c in caches:
        Z = torch.tensor([Zmap[tab.symbols[int(s)]] for s in c.species])
        bt = batched_precompute(c.pos.double(), Z, torch.zeros(len(Z), dtype=torch.long), tab, table, chan,
                                inv_form=model.inv_form)
        o1, _ = run(bt, 1)
        on, info = run(bt, a.outer)
        e1, en = float(o1["e_ct"][0]), float(on["e_ct"][0])
        n = c.species.shape[0]
        d.append(abs(e1 - en) * HK / n)
        rel.append(abs(e1 - en) / max(abs(en), 1e-12))
        qmax.append(float(on["p"].max()))
        it.append(info["outer"])
    t = torch.tensor
    print(f"\n{len(caches)} clusters (>= {a.min_ions} ions): |E_LR - E_conv| per ion mean {t(d).mean():.4f} max {t(d).max():.4f} "
          f"kJ/mol, relative mean {t(rel).mean():.3f} max {t(rel).max():.3f}; max q {max(qmax):.3e}; outer steps "
          f"{min(it)}-{max(it)}")
    model.ct_outer = n1


if __name__ == "__main__":
    main()
