"""Which flux channels matter?  Per channel: type, distance, kernel energy e_c, its share of E_CT and |q_c|, for the
untrained model on a 5x5x5 rock-salt cluster; then E_CT and timing vs the channel cutoff."""
import time, sys
from pathlib import Path
import torch
from rsfff.ff.complete import CompleteModel, DensityEnvironment, RadialTable, batched_precompute
from rsfff.ff.ionct import ct as CT
from rsfff.ff.ionct.params import IonTable
sys.path.insert(0, str(Path(__file__).parent))
torch.set_default_dtype(torch.float64)
PARAMS = Path.home() / "rsfff/params/ions_wb97mv_tzvpd_pf2p.json"
KJ = 2625.4996


def cluster(ns, a=2.82, jitter=0.08, seed=0):
    g = torch.tensor([(i, j, k) for i in range(ns[0]) for j in range(ns[1]) for k in range(ns[2])], dtype=torch.float64)
    Z = torch.where(g.sum(1) % 2 == 0, 11, 17)
    X = g * a + jitter * torch.randn(g.shape[0], 3, generator=torch.Generator().manual_seed(seed))
    return X - X.mean(0), Z


tab, chan = IonTable.load(PARAMS), CT.ChannelTable.load(PARAMS)
table = RadialTable(tab, h=0.05)
feat = DensityEnvironment(tab)
torch.manual_seed(3)
m = CompleteModel(tab, chan, feat.n_feat, feature_names=feat.feature_names, pseudo=False, frozen=True,
                  pauli_form="shells", confine=("Cl",), env_cap=3.0, xpol=True, xpol_perp=True,
                  xpol_kappa_env="scalefree", ct_model="flux")
ns = tuple(int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "5,5,5").split(","))
X, Z = cluster(ns)
n = X.shape[0]
fr = torch.zeros(n, dtype=torch.long)
with torch.no_grad():
    bt = batched_precompute(X, Z, fr, tab, table, chan, inv_form=m.inv_form)
    o = m(bt)
i, j = bt["ch_index"]
r = bt["ch_r"]
ty = bt["ch_type"]
names = list(chan.types)
e_c, e_ch, q = o["W"], o["e_ch"], o["p"]
E = float(o["e_ct"].sum())
print(f"n={n} channels={i.shape[0]}  E_CT={E*KJ:.3f} kJ/mol  sum e_c={float(e_c.sum())*KJ:.3f}")
bins = [0, 3.0, 3.5, 4.2, 5.0, 6.0, 7.0, 8.01]
print(f"{'r bin (A)':>12} {'type':>6} {'#ch':>6} {'max|e_c| kJ':>12} {'sum e_c kJ':>11} {'sum share kJ':>12} {'max|q|':>9}")
for lo, hi in zip(bins[:-1], bins[1:]):
    for t, nm in enumerate(names):
        k = (r >= lo) & (r < hi) & (ty == t)
        if k.any():
            print(f"{lo:5.1f}-{hi:4.1f} {nm:>8} {int(k.sum()):6d} {float(e_c[k].abs().max())*KJ:12.2e} "
                  f"{float(e_c[k].sum())*KJ:11.3e} {float(e_ch[k].sum())*KJ:12.3e} {float(q[k].abs().max()):9.2e}")

print("\ncutoff sweep (same model; the channel taper moves with the cutoff)")
for cut in (8.0, 6.0, 5.0, 4.5, 4.2, 3.6):
    chan.cutoff = cut
    with torch.no_grad():
        bt = batched_precompute(X, Z, fr, tab, table, chan, inv_form=m.inv_form)
        t0 = time.perf_counter()
        o = m(bt)
        dt = time.perf_counter() - t0
    print(f"cut {cut:4.1f}: channels {bt['ch_index'].shape[1]:6d}  E_CT {float(o['e_ct'].sum())*KJ:11.3f}  "
          f"E_pol {float(o['e_pol'].sum())*KJ:11.3f}  dE_CT/ion vs 8 A {(float(o['e_ct'].sum())-E)*KJ/n:+.4f}  "
          f"forward {dt:.3f}s")
