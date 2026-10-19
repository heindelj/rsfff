"""Frozen Pauli repulsion between ions on its own (CMM monopole Slater kernel, environment-scaled), fit to ALMO
``mod_pauli`` independently of the response model.

    E_P = sum_{i<j} qP_i qP_j f_1(b_ij r) / r,   b_ij = sqrt(b_i b_j),   qP_i = qP_s exp(u(h_i) - u(h_i^0))

f_1 is the overlap part of the two-centre Slater Coulomb (CMM's short-range kernel); u is a network on the ion's
environment invariants (qP = qP_s for an isolated ion). The same per-species form as the frozen part of
``kernels.pauli_kernel``; here it is not shared with the polarization.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..multipole import slater_two_center_damp
from ..units import BOHR_ANG
from .kernels import all_pairs

#: (qP e, bP 1/bohr) from the pair fit to mod_pauli (model.PAULI_INIT)
PAULI_INIT = {"Na": (0.978, 1.360), "Cl": (6.575, 1.577)}


class IonPauli(nn.Module):
    def __init__(self, symbols, q, b, *, n_feat=None, hidden=32, emb=8, feat_scale=None):
        super().__init__()
        self.symbols = tuple(symbols)
        f = lambda v: nn.Parameter(torch.log(torch.as_tensor(v, dtype=torch.float64)))
        self.ln_q, self.ln_b = f(q), f(b)
        self.n_feat, self.hidden, self.emb_dim = n_feat, hidden, emb
        if n_feat:
            self.register_buffer("feat_scale", torch.ones(n_feat, dtype=torch.float64) if feat_scale is None
                                 else feat_scale.clone().double())
            self.emb = nn.Embedding(len(symbols), emb).double()
            self.net = nn.Sequential(nn.Linear(n_feat + emb, hidden), nn.SiLU(), nn.Linear(hidden, 1)).double()
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    @classmethod
    def initial(cls, symbols, **kw):
        return cls(symbols, [PAULI_INIT[x][0] for x in symbols], [PAULI_INIT[x][1] for x in symbols], **kw)

    def config(self):
        return {"kind": "pauli", "symbols": list(self.symbols), "n_feat": self.n_feat, "hidden": self.hidden,
                "emb": self.emb_dim}

    @classmethod
    def load(cls, path):
        ck = torch.load(path, weights_only=False)
        c = ck["config"]
        m = cls(c["symbols"], [1.0] * len(c["symbols"]), [1.0] * len(c["symbols"]), n_feat=c["n_feat"],
                hidden=c["hidden"], emb=c["emb"])
        m.load_state_dict(ck["state"])
        return m

    def save(self, path, **extra):
        torch.save({"config": self.config(), "state": self.state_dict(), **extra}, path)

    def constants(self):
        return {"q": self.ln_q.exp().tolist(), "b": self.ln_b.exp().tolist()}

    def env_scale(self, species, feats):
        """(N,) log of the environment scale of qP: zero for an isolated ion or without a network."""
        if not self.n_feat or feats is None:
            return torch.zeros(species.shape[0], dtype=torch.float64, device=species.device)
        e = self.emb(species)
        return (self.net(torch.cat([feats / self.feat_scale, e], -1))
                - self.net(torch.cat([torch.zeros_like(feats), e], -1)))[:, 0]

    def pair_energy(self, pos_ang, species, pairs, feats=None):
        i, j = pairs
        r = (pos_ang[j] - pos_ang[i]).norm(dim=-1) / BOHR_ANG
        lq = self.ln_q[species] + self.env_scale(species, feats)
        b = torch.exp(0.5 * (self.ln_b[species[i]] + self.ln_b[species[j]]))
        return torch.exp(lq[i] + lq[j]) * slater_two_center_damp(b * r, 0)[0] / r

    def forward(self, pos_ang, species, batch_idx, n_frames=None, pairs=None, feats=None):
        B = int(batch_idx.max()) + 1 if n_frames is None else n_frames
        pairs = all_pairs(batch_idx) if pairs is None else pairs
        return pos_ang.new_zeros(B).index_add(0, batch_idx[pairs[0]], self.pair_energy(pos_ang, species, pairs, feats))
