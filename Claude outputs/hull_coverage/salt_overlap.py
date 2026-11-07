import numpy as np, pandas as pd, hullkit as H, json
I = np.load('I.npy'); df = pd.read_pickle('ions.pkl')
df['salt'] = df.workflow.str.extract(r'seed-(\w+)-')[0].fillna('nacl')
tr = (df.split.values == 'train')
W = H.Whitener(I[tr], k=6)
X = W(I)
salts = ['nacl', 'lif', 'lii', 'kbr', 'csi']
hulls = {}
for s in salts:
    m = tr & (df.salt.values == s)
    P = np.unique(X[m].round(10), axis=0)
    hulls[s] = P[H.extreme_points(P, tol=1e-7)]
    print(s, m.sum(), 'ions', len(hulls[s]), 'vertices', flush=True)
rng = np.random.default_rng(0)
M = pd.DataFrame(index=salts, columns=salts, dtype=float)
for a in salts:          # hull of a
    for b in salts:      # ions of b
        m = np.where(tr & (df.salt.values == b))[0]
        m = rng.choice(m, min(1500, len(m)), replace=False)
        M.loc[a, b] = (H.hull_distance(hulls[a], X[m]) <= 1e-5).mean()
print('fraction of column-salt ions inside row-salt training hull (k=6):'); print(M.round(3))
# and: each salt's ions inside the hull of the other four salts together
loo = {}
for b in salts:
    P = np.vstack([hulls[a] for a in salts if a != b])
    m = np.where(tr & (df.salt.values == b))[0]; m = rng.choice(m, min(1500, len(m)), replace=False)
    loo[b] = float((H.hull_distance(P, X[m]) <= 1e-5).mean())
print('inside hull of the other four salts:', {k: round(v, 3) for k, v in loo.items()})
M.to_csv('salt_overlap_k6.csv'); json.dump(loo, open('salt_loo_k6.json', 'w'))
