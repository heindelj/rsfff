import numpy as np, pandas as pd, hullkit as H, json, time
Q = np.load('pool_Q_k6.npy'); q = pd.read_pickle('pool_ions.pkl'); V0 = np.load('hull_vertices_k6.npy')
import pickle
pool = pickle.load(open('pool_feats.pkl', 'rb'))
def sane(p):   # guard: no like-ion collapse, no unlike contact < 1.8 A, no ion > 6 A from every other ion
    R = np.linalg.norm(p['pos'][:, None] - p['pos'][None], axis=-1); np.fill_diagonal(R, np.inf)
    u = p['Z'][:, None] != p['Z'][None]
    return R[~u].min() > 2.4 and R[u].min() > 1.8 and R.min(1).max() < 6.0
ok = np.array([sane(p) for p in pool]); print('sane frames', ok.sum(), 'of', len(ok))
q = q[ok[q.frame.values]]
trajs = q.traj.unique(); rng = np.random.default_rng(0); rng.shuffle(trajs)
half = set(map(tuple, trajs[: len(trajs) // 2]))
isA = q.traj.map(lambda t: tuple(t) in half).values
fA = np.unique(q.frame[isA]); fB = np.unique(q.frame[~isA])
fA = rng.choice(fA, min(500, len(fA)), replace=False); fB = rng.choice(fB, min(500, len(fB)), replace=False)
ionsA = {f: q.index.values[q.frame.values == f] for f in fA}
evalB = np.concatenate([q.index.values[q.frame.values == f] for f in fB])
sig = q.groupby('frame').sigE.first()
CHK = [0, 5, 10, 20, 40, 80]

def coverage(V, idx):
    d = H.hull_distance(V, Q[idx])
    return float((d > 1e-5).mean()), float(np.percentile(d, 90)), float(np.percentile(d, 99))

def run(order_fn, name):
    V = V0.copy(); res = []; chosen = []
    for m in range(max(CHK) + 1):
        if m in CHK:
            res.append((m,) + coverage(V, evalB))
        if m == max(CHK): break
        f = order_fn(V, chosen)
        chosen.append(f); V = np.vstack([V, Q[ionsA[f]]])
    print(name, [(r[0], round(r[1], 3), round(r[2], 3)) for r in res], flush=True)
    return res, chosen

state = {}
def greedy(V, chosen):
    if 'd' not in state:
        state['d'] = {f: H.hull_distance(V, Q[ionsA[f]]) for f in fA}
    else:          # distances only shrink as the hull grows: recompute the still-positive ones
        newV = V
        for f in fA:
            if f in chosen: continue
            dd = state['d'][f]; pos = dd > 1e-5
            if pos.any(): dd[pos] = H.hull_distance(newV, Q[ionsA[f]][pos])
    cand = [f for f in fA if f not in chosen]
    return max(cand, key=lambda f: state['d'][f].max())

order_rand = list(rng.permutation(fA))
order_sig = list(sig.loc[fA].sort_values(ascending=False).index)
out = {}
t = time.time()
for sd in range(3):
    o_r = list(np.random.default_rng(10 + sd).permutation(fA))
    out[f'random{sd}'] = run(lambda V, c, o_r=o_r: o_r[len(c)], f'random{sd}')
out['sigma_E'] = run(lambda V, c: order_sig[len(c)], 'sigma_E')
out['greedy_hull'] = run(greedy, 'greedy_hull')
print(f'[{time.time()-t:.0f}s]')
sel = out['greedy_hull'][1]
print('greedy picks: n ions', [int(q.n[q.frame == f].iloc[0]) for f in sel[:20]])
json.dump({k: [list(r) for r in v[0]] for k, v in out.items()}, open('greedy_demo.json', 'w'), indent=1)
pd.to_pickle(out, 'greedy_demo.pkl')
