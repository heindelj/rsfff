import torch, numpy as np, pickle
from ase.io import read
from rsfff.ff.ionct.params import IonTable
from rsfff.ff.complete.environment import DensityEnvironment
torch.set_num_threads(1)
tab = IonTable.load('/mnt/user-data/uploads/research--rsfff/train/halides/params_frozen_halides_45pairs.json')
env = DensityEnvironment(tab, inv_form='sq')
out = []
for r in (1, 2, 3):
    A = read(f'/mnt/user-data/uploads/research--rsfff/sample/nacl/round{r}/pool.xyz', index='::5')
    for k, a in enumerate(A):
        f = env(a.positions, a.numbers, np.zeros(len(a), dtype=int)).inv_feats.numpy()
        out.append(dict(round=r, i=k * 5, n=len(a), Z=a.numbers.copy(), pos=a.positions.copy(), feats=f,
                        traj=a.info.get('traj_id'), t=a.info.get('time_fs'), sigE=a.info.get('sigma_energy'),
                        sigF=a.info.get('sigma_forces'), T=a.info.get('temperature'), contact=a.info.get('contact')))
        if len(out) % 200 == 0:
            print(len(out), flush=True); pickle.dump(out, open('pool_feats.pkl', 'wb'))
pickle.dump(out, open('pool_feats.pkl', 'wb')); print('done', len(out))
