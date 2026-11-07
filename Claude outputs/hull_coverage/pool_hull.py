import numpy as np, pandas as pd, pickle, torch, hullkit as H, json, time
from scipy.stats import spearmanr
k = 6
I = np.load('I.npy'); df = pd.read_pickle('ions.pkl')
scale = torch.load('/mnt/user-data/uploads/research--rsfff/train/halides/m26/five_salts_polarfilt/model.pt',
                   weights_only=False)['state']['feat_scale'].numpy()
tr = df.split.values == 'train'
W = H.Whitener(I[tr], k=k)
V = np.load(f'hull_vertices_k{k}.npy')
pool = pickle.load(open('pool_feats.pkl', 'rb'))
rows, Q = [], []
for fi, p in enumerate(pool):
    R = np.linalg.norm(p['pos'][:, None] - p['pos'][None], axis=-1)
    unlike = p['Z'][:, None] != p['Z'][None]
    cn = ((R < 3.3) & unlike).sum(1)
    for a in range(p['n']):
        rows.append(dict(frame=fi, n=p['n'], sym='Na' if p['Z'][a] == 11 else 'Cl', cn=int(cn[a]), sigE=p['sigE'] / p['n'],
                         traj=(p['round'], p['traj'])))
    Q.append(W(p['feats'] / scale))
q = pd.DataFrame(rows); Q = np.concatenate(Q)
np.save('pool_Q_k6.npy', Q); q.to_pickle('pool_ions.pkl')
# training full-cluster ions' coordination for comparison
t = time.time()
rng = np.random.default_rng(0)
sub = rng.choice(len(Q), 8000, replace=False)
d = H.hull_distance(V, Q[sub])
qs = q.iloc[sub].copy(); qs['d'] = d
print(f'{len(pool)} pool frames, {len(Q)} ions; 8000 sampled [{time.time()-t:.0f}s]')
print('frac pool ions outside training hull (k=6):', (d > 1e-5).mean())
print(qs.groupby('cn').d.agg(['count', lambda x: (x > 1e-5).mean(), 'median']).rename(columns={'<lambda_0>': 'frac_out'}))
print(qs.groupby('sym').d.agg(['count', lambda x: (x > 1e-5).mean(), 'median']))
qs.to_pickle('pool_sub_d_k6.pkl')
# per-frame: max distance vs committee sigma
pf = qs.groupby('frame').agg(d=('d', 'max'), sigE=('sigE', 'first'), n=('n', 'first'))
print('frame-level spearman(max d, sigma_E/ion):', spearmanr(pf.d, pf.sigE).correlation, 'frames', len(pf))
