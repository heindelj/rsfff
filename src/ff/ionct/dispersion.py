"""Tang-Toennies damped dispersion between ions, with environment-dependent dispersion coefficients.

    E_disp = - sum_{i<j} s_i s_j [ f_6(b_ij r) C6_ij / r^6 + f_8(b_ij r) C8_ij / r^8 ]
    C6_ij = sqrt(C6_i C6_j),  C8_ij = sqrt(C8_i C8_j),  b_ij = sqrt(b_i b_j)
    s_i   = exp(1/2 [u(h_i) - u(h_i^0)])      (network on the ion's environment invariants; s = 1 when isolated)

so C6_ij -> s_i^2 s_j^2 ... in effect each ion's C6 is scaled by s_i^2. ALMO-EDA dispersion of NaCl clusters is
7-15 % below the sum of its pair values (density-dependent nonlocal correlation: the ions' response is confined
in a cluster); a pairwise form cannot carry that, an environment scale of C6 does. Without a network (n_feat=None)
this is the plain pairwise TT model with per-species constants.

Fit by ``scripts/ion_frozen_fit.py`` against ALMO ``disp``. Atomic units inside; positions in Angstrom.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..damping import tang_toennies
from ..units import BOHR_ANG
from .kernels import all_pairs

#: starting values (C6 Ha bohr^6, b 1/bohr, C8 Ha bohr^8): C6, b from the CMM ion model
#: (rsfff.ff.dispersion.DEFAULT_C6_PRIOR); C8 rough
DISP_INIT = {"Na": (5.5, 1.98, 30.0), "Cl": (662.0, 1.07, 12000.0)}


class IonDispersion(nn.Module):
    def __init__(self, symbols, c6, b, c8=None, *, use_c8=True, n_feat=None, hidden=32, emb=8, feat_scale=None):
        super().__init__()
        self.symbols = tuple(symbols)
        f = lambda v: nn.Parameter(torch.log(torch.as_tensor(v, dtype=torch.float64)))
        self.ln_c6, self.ln_b = f(c6), f(b)
        self.use_c8 = use_c8
        self.ln_c8 = f(c8 if c8 is not None else [DISP_INIT[x][2] for x in symbols])
        self.n_feat, self.hidden, self.emb_dim = n_feat, hidden, emb
        if n_feat:
            self.register_buffer("feat_scale", torch.ones(n_feat, dtype=torch.float64) if feat_scale is None
                                 else feat_scale.clone().double())
            self.emb = nn.Embedding(len(symbols), emb).double()
            self.net = nn.Sequential(nn.Linear(n_feat + emb, hidden), nn.SiLU(), nn.Linear(hidden, 1)).double()
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    @classmethod
    def initial(cls, symbols, use_c8=True, **kw):
        return cls(symbols, [DISP_INIT[x][0] for x in symbols], [DISP_INIT[x][1] for x in symbols],
                   [DISP_INIT[x][2] for x in symbols], use_c8=use_c8, **kw)

    def config(self):
        return {"symbols": list(self.symbols), "use_c8": self.use_c8, "n_feat": self.n_feat, "hidden": self.hidden,
                "emb": self.emb_dim}

    @classmethod
    def load(cls, path):
        ck = torch.load(path, weights_only=False)
        c = ck["config"]
        m = cls(c["symbols"], [1.0] * len(c["symbols"]), [1.0] * len(c["symbols"]), [1.0] * len(c["symbols"]),
                use_c8=c["use_c8"], n_feat=c["n_feat"], hidden=c["hidden"], emb=c["emb"])
        m.load_state_dict(ck["state"])
        return m

    def save(self, path, **extra):
        torch.save({"config": self.config(), "state": self.state_dict(), **extra}, path)

    def constants(self):
        return {"c6": self.ln_c6.exp().tolist(), "b": self.ln_b.exp().tolist(), "c8": self.ln_c8.exp().tolist()}

    def env_scale(self, species, feats):
        """(N,) log s_i: zero for an isolated ion (zero features) or without a network."""
        if not self.n_feat or feats is None:
            return torch.zeros(species.shape[0], dtype=torch.float64, device=species.device)
        e = self.emb(species)
        return 0.5 * (self.net(torch.cat([feats / self.feat_scale, e], -1))
                      - self.net(torch.cat([torch.zeros_like(feats), e], -1)))[:, 0]

    def pair_energy(self, pos_ang, species, pairs, feats=None):
        i, j = pairs
        r = (pos_ang[j] - pos_ang[i]).norm(dim=-1) / BOHR_ANG
        b = torch.exp(0.5 * (self.ln_b[species[i]] + self.ln_b[species[j]]))
        c6 = torch.exp(0.5 * (self.ln_c6[species[i]] + self.ln_c6[species[j]]))
        e = -tang_toennies(b * r, 6) * c6 / r ** 6
        if self.use_c8:
            c8 = torch.exp(0.5 * (self.ln_c8[species[i]] + self.ln_c8[species[j]]))
            e = e - tang_toennies(b * r, 8) * c8 / r ** 8
        ls = self.env_scale(species, feats)
        return e * torch.exp(ls[i] + ls[j])

    def forward(self, pos_ang, species, batch_idx, n_frames=None, pairs=None, feats=None):
        """(B,) Hartree per frame. ``pairs`` (2, P) within frames, if precomputed (all_pairs is O(N^2) in the
        batch); ``feats`` (N, F) the environment invariants (needed when the model has a network)."""
        B = int(batch_idx.max()) + 1 if n_frames is None else n_frames
        pairs = all_pairs(batch_idx) if pairs is None else pairs
        return pos_ang.new_zeros(B).index_add(0, batch_idx[pairs[0]], self.pair_energy(pos_ang, species, pairs, feats))
