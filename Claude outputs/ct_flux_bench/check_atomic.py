"""Atomic-space flux solve == channel-space flux solve: energies, channel charges, moments, parameter gradients and
forces, on a padded multi-frame batch (2, 8, 27 ions, an applied field), then timing vs size."""
import time, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parent))
from rsfff.ff.complete import CompleteModel, DensityEnvironment, RadialTable, batched_precompute
from rsfff.ff.complete import model as CM
from rsfff.ff.ionct import ct as CT
from rsfff.ff.ionct.params import IonTable
import flux_atomic as FA

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
with torch.no_grad():                                   # asymmetric directions and non-trivial heads, as in the tests
    m.ct_lnq0[0, 0] -= 2.0
    for p in m.head_ctq.parameters():
        p.add_(0.2 * torch.randn_like(p))

parts = [cluster((2, 1, 1), a=2.9), cluster((2, 2, 2), seed=1), cluster((3, 3, 3), seed=2)]
X0 = torch.cat([p[0] + 30.0 * k for k, p in enumerate(parts)])
Z = torch.cat([p[1] for p in parts])
fr = torch.cat([torch.full((p[0].shape[0],), k, dtype=torch.long) for k, p in enumerate(parts)])
field = torch.tensor([[0.0, 0, 0], [0.002, -0.001, 0.0005], [0.0, 0.001, 0.0]])


def run():
    m.zero_grad(set_to_none=True)
    X = X0.clone().requires_grad_(True)
    bt = batched_precompute(X, Z, fr, tab, table, chan, inv_form=m.inv_form)
    o = m(bt, field=field)
    E = (o["e_int"] * torch.tensor([1.0, 0.7, 1.3])).sum() + 0.5 * o["e_ct"].sum()
    gX, = torch.autograd.grad(E, X, retain_graph=True)
    E.backward()
    gp = {k: p.grad.clone() for k, p in m.named_parameters() if p.grad is not None}
    return o, gX, gp


o1, g1, p1 = run()
orig = FA.patch(CM.CompleteModel)
o2, g2, p2 = run()
FA.unpatch(CM.CompleteModel, orig)


def rel(a, b):
    return float((a - b).abs().max() / (b.abs().max() + 1e-300))


print("E_CT per frame (kJ/mol) channel:", (o1["e_ct"] * KJ).tolist())
print("E_CT per frame (kJ/mol) atomic :", (o2["e_ct"] * KJ).tolist())
for k in ("e_ct", "e_int", "e_pol", "p", "e_ch", "dq", "x"):
    print(f"  {k:6s} max rel diff {rel(o2[k].detach(), o1[k].detach()):.2e}")
print(f"  forces max rel diff {rel(g2, g1):.2e}")
worst = max(((k, rel(p2[k], p1[k])) for k in p1 if float(p1[k].abs().max()) > 0), key=lambda kv: kv[1])
print(f"  parameter grads: {len(p1)} tensors, worst {worst[0]} {worst[1]:.2e}")
print("  flux_lnv grad", p1["flux_lnv"].tolist(), p2["flux_lnv"].tolist())
