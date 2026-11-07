"""A training-like batch: 32 frames of mixed size (dimers .. 25 ions, padded to the largest), channel-space vs
atomic-space flux solve. Steps: energy-only (loss on E, backward) and force (forces with create_graph, loss on E and F,
backward = double backward)."""
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
gen = torch.Generator().manual_seed(0)


def frame(n):
    """n ions cut from a jittered rock-salt lattice (the n sites closest to a random point), charge-neutral if n even."""
    g = torch.tensor([(i, j, k) for i in range(-3, 4) for j in range(-3, 4) for k in range(-3, 4)], dtype=torch.float64)
    c = torch.rand(3, generator=gen) - 0.5
    d = (g - c).norm(dim=1)
    idx = torch.argsort(d)[:n]
    g = g[idx]
    Z = torch.where(g.sum(1) % 2 == 0, 11, 17)
    X = g * 2.82 + 0.12 * torch.randn(n, 3, generator=gen)
    return X, Z


sizes = [2] * 8 + [3] * 6 + [4, 5, 6, 6, 7, 8, 8, 9, 9, 9] + [int(x) for x in sys.argv[1].split(",")]
frames = [frame(n) for n in sizes]
X0 = torch.cat([f[0] + 40.0 * k for k, f in enumerate(frames)])
Z = torch.cat([f[1] for f in frames])
fr = torch.cat([torch.full((f[0].shape[0],), k, dtype=torch.long) for k, f in enumerate(frames)])

tab, chan = IonTable.load(PARAMS), CT.ChannelTable.load(PARAMS)
table = RadialTable(tab, h=0.05)
feat = DensityEnvironment(tab)
torch.manual_seed(3)
m = CompleteModel(tab, chan, feat.n_feat, feature_names=feat.feature_names, pseudo=False, frozen=True,
                  pauli_form="shells", confine=("Cl",), env_cap=3.0, xpol=True, xpol_perp=True,
                  xpol_kappa_env="scalefree", ct_model="flux")


def step(kind):
    m.zero_grad(set_to_none=True)
    X = X0.clone().requires_grad_(kind == "force")
    t0 = time.perf_counter()
    bt = batched_precompute(X, Z, fr, tab, table, chan, inv_form=m.inv_form)
    t1 = time.perf_counter()
    o = m(bt)
    t2 = time.perf_counter()
    loss = (o["e_int"] ** 2).sum() + (o["e_ct"] ** 2).sum()
    if kind == "force":
        F, = torch.autograd.grad(o["e_int"].sum(), X, create_graph=True)
        loss = loss + (F ** 2).sum()
    t3 = time.perf_counter()
    loss.backward()
    t4 = time.perf_counter()
    return dict(precompute=t1 - t0, forward=t2 - t1, forces=t3 - t2, backward=t4 - t3, total=t4 - t0), bt


with torch.no_grad():
    m(batched_precompute(X0[:2], Z[:2], fr[:2] * 0, tab, table, chan, inv_form=m.inv_form))      # build tables
for label, atomic in (("channel", False), ("atomic", True)):
    orig = FA.patch(CM.CompleteModel) if atomic else None
    for kind in ("energy", "force"):
        best = None
        for _ in range(2):
            t, bt = step(kind)
            best = t if best is None or t["total"] < best["total"] else best
        cnt = torch.bincount(bt["ch_frame"], minlength=bt["B"])
        print(f"{label:8s} {kind:6s} B={bt['B']} nmax={bt['nmax']} channels={int(cnt.sum())} Cmax={int(cnt.max())}  "
              + "  ".join(f"{k} {v:.3f}" for k, v in best.items()), flush=True)
    if atomic:
        FA.unpatch(CM.CompleteModel, orig)
