import torch, numpy as np, pandas as pd, hullkit as H
fr = torch.load('frames_m26.pt', weights_only=False)
pr = pd.read_csv('/mnt/user-data/uploads/research--rsfff/train/halides/m26/five_salts_polarfilt/predictions.csv')
ck = torch.load('/mnt/user-data/uploads/research--rsfff/train/halides/m26/five_salts_polarfilt/model.pt', weights_only=False)
scale = ck['state']['feat_scale'].numpy()
idx = H.match_predictions(fr, pr)
df, F = H.ion_table(fr)
df['split'] = pr.split.values[idx[df.frame.values]]
I = F / scale
# per-frame errors (kJ/mol, per ion)
P = pr.iloc[idx].reset_index(drop=True)
for t in ('pol', 'ct', 'pauli', 'disp', 'int'):
    P[f'err_{t}'] = (P[t] - P[f'{t}_ref']).abs() / P['n']
np.save('I.npy', I); df.to_pickle('ions.pkl'); P.to_pickle('frames.pkl')
