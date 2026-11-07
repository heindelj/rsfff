import numpy as np, pandas as pd, hullkit as H, json, time, sys
from scipy.stats import spearmanr
k = int(sys.argv[1]) if len(sys.argv) > 1 else 10
I = np.load('I.npy'); df = pd.read_pickle('ions.pkl'); P = pd.read_pickle('frames.pkl')
tr = df.split.values == 'train'
W = H.Whitener(I[tr], k=k)
Xtr = np.unique(W(I[tr]).round(10), axis=0)
t = time.time()
vtx = H.extreme_points(Xtr, tol=1e-7)
print(f'k={k}: {len(Xtr)} unique train ions, {len(vtx)} hull vertices [{time.time()-t:.0f}s]', flush=True)
V = Xtr[vtx]
np.save(f'hull_vertices_k{k}.npy', V)
Q = W(I[~tr])
d = H.hull_distance(V, Q)
dk = H.knn_dist(Xtr, Q, k=5)
q = df[~tr].copy(); q['d'] = d; q['dknn'] = dk
fr = q.groupby('frame').agg(d=('d', 'max'), dknn=('dknn', 'max'), n=('n', 'first'), split=('split', 'first'),
                             workflow=('workflow', 'first'), full=('full', 'first'))
for t_ in ('pol', 'ct', 'pauli', 'disp', 'int'):
    fr[f'err_{t_}'] = P.loc[fr.index, f'err_{t_}'].values
fr.to_pickle(f'hull_err_k{k}.pkl')
out = dict(k=k, n_vertices=int(len(vtx)), n_train_unique=int(len(Xtr)),
           frac_ions_outside=float((d > 1e-5).mean()), frac_frames_outside=float((fr.d > 1e-5).mean()))
rows = []
for name, g in [('all', fr), ('n=2', fr[fr.n == 2]), ('n=3', fr[fr.n == 3]), ('n>=4', fr[fr.n >= 4])]:
    r = dict(subset=name, n=len(g), frac_out=round(float((g.d > 1e-5).mean()), 3))
    for t_ in ('pol', 'pauli', 'ct', 'int'):
        e = g[f'err_{t_}']; ok = e.notna()
        r[f'rho_hull_{t_}'] = round(spearmanr(g.d[ok], e[ok]).correlation, 3)
        r[f'rho_knn_{t_}'] = round(spearmanr(g.dknn[ok], e[ok]).correlation, 3)
        inn, outn = e[ok & (g.d <= 1e-5)], e[ok & (g.d > 1e-5)]
        r[f'med_in_{t_}'] = round(float(inn.median()), 3) if len(inn) else None
        r[f'med_out_{t_}'] = round(float(outn.median()), 3) if len(outn) else None
    rows.append(r)
out['by_subset'] = rows
print(pd.DataFrame(rows).T.to_string())
json.dump(out, open(f'hull_err_k{k}.json', 'w'), indent=1)
