import numpy as np, pandas as pd, hullkit as H, json
I = np.load('I.npy'); df = pd.read_pickle('ions.pkl')
tr = df.split.values == 'train'
X = I[tr]; S = X / X.std(0)
S = np.unique(S.round(8), axis=0)
print('unique train ions', len(S))
rng = np.random.default_rng(1)
Z = S[rng.choice(len(S), 2500, replace=False)]
print('TwoNN (dedup train)', H.twonn_id(S[rng.choice(len(S), 6000, replace=False)]))
sel, curve = H.greedy_ii_selection(Z, n_max=16)
out = [(H.NAMES[s], round(c, 4)) for s, c in zip(sel, curve)]
for o in out: print(o)
groups = {p: [H.NAMES.index(f"{p}:{q}") for q in H.INV] for p in H.PROBES}
gsel, gcurve = H.greedy_ii_selection(Z, n_max=6, groups=groups)
print('probe groups:', list(zip(gsel, np.round(gcurve, 4))))
json.dump(dict(single=out, groups=list(zip(gsel, [float(c) for c in gcurve]))), open('select_ii.json', 'w'), indent=1)
