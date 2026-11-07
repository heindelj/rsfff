import numpy as np, pandas as pd, hullkit as H, json
from scipy.stats import spearmanr
I = np.load('I.npy'); df = pd.read_pickle('ions.pkl')
tr = df.split.values == 'train'
X = I[tr]; S = X / X.std(0)
rng = np.random.default_rng(0)
res = {}
# intrinsic dimension
sub = rng.choice(len(S), 6000, replace=False)
res['twonn_all_train'] = H.twonn_id(S[sub])
full = tr & df.full.values
res['twonn_full_cluster_ions'] = H.twonn_id(S[(df.full.values[tr])])
res['n_full_ions'] = int(full.sum())
# per-species ID
res['twonn_by_species'] = {s: round(H.twonn_id(S[df.sym.values[tr] == s]), 2) for s in H.SYMS if (df.sym.values[tr] == s).sum() > 300}
# Spearman redundancy
rho = spearmanr(S[sub]).correlation
np.fill_diagonal(rho, 0)
pairs = [(H.NAMES[i], H.NAMES[j], round(rho[i, j], 4)) for i in range(42) for j in range(i + 1, 42) if abs(rho[i, j]) > 0.98]
res['n_pairs_rho_gt_0.98'] = len(pairs)
res['pairs_rho_gt_0.98'] = pairs
# greedy: how many columns explain the rest linearly (CUR-like pivoted QR on standardized cols)
from scipy.linalg import qr
_, R, piv = qr(S[sub], pivoting=True, mode='economic')
d = np.abs(np.diag(R)); res['pivoted_qr_order'] = [H.NAMES[p] for p in piv[:20]]
res['pivoted_qr_rdiag_rel'] = list(np.round(d / d[0], 4)[:42])
# information imbalance: groups
groups = {p: [H.NAMES.index(f"{p}:{q}") for q in H.INV] for p in H.PROBES}
groups['frontier'] = groups['donor'] + groups['acceptor']
groups['coulomb'] = [i for i, n in enumerate(H.NAMES) if ':V' in n]
groups['overlap'] = [i for i, n in enumerate(H.NAMES) if ':S' in n]
groups['cross'] = [i for i, n in enumerate(H.NAMES) if n.endswith('V1.S1')]
allc = list(range(42))
ii = {}
Wsub = rng.choice(len(S), 3000, replace=False)
Z = S[Wsub]
for g, cols in groups.items():
    rest = [c for c in allc if c not in cols]
    ii[g] = dict(rest_to_full=round(H.information_imbalance(Z[:, rest], Z, n=3000), 4),
                 full_to_rest=round(H.information_imbalance(Z, Z[:, rest], n=3000), 4),
                 group_to_rest=round(H.information_imbalance(Z[:, cols], Z[:, rest], n=3000), 4),
                 rest_to_group=round(H.information_imbalance(Z[:, rest], Z[:, cols], n=3000), 4))
res['information_imbalance'] = ii
json.dump(res, open('redundancy.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != 'pairs_rho_gt_0.98'}, indent=1))
for p in pairs: print(p)
