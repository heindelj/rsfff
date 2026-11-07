"""Environment heads (complete model plan, C2): every emitted parameter is the isolated-atom value times a factor
that is exactly 1 when the environment vanishes,

    theta_i = theta0[species_i] * exp( a . I_i + NN(I_i) - NN(0) )

with I_i the scaled environment invariants of atom i. The linear term and the network are shared by all species
(the species enters only through theta0, the centre atom's own parameters); an optional species embedding can be
concatenated to the network input, in which case NN(0) is evaluated with the same embedding so the isolated limit
still holds. Both last layers start at zero, so training starts from the isolated-atom force field.

With ``log_k > 0`` the head reads slog(I) = sign(I) log(1 + |I| / k) / log(1 + 1 / k) in place of I (0 at 0, linear
below k, logarithmic above): the overlap and penetration invariants decay as e^(-bR), so in the log they are close to
linear in R, and the linear term a . slog(I) makes theta a power of the overlap instead of exp(a e^(-bR))."""
from __future__ import annotations

import torch
from torch import nn

__all__ = ["EnvironmentHead"]


def _mlp(i, h, o, depth):
    layers, d = [], i
    for _ in range(depth):
        layers += [nn.Linear(d, h), nn.SiLU()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class EnvironmentHead(nn.Module):
    """``forward(I, species=None) -> (N, n_out)``: the log-factor ``a . I + NN(I) - NN(0)``, exactly zero at I = 0."""

    def __init__(self, n_feat: int, n_out: int, hidden: int = 64, depth: int = 2, n_species: int = 0, emb: int = 8):
        super().__init__()
        self.lin = nn.Linear(n_feat, n_out, bias=False)
        nn.init.zeros_(self.lin.weight)
        self.emb = nn.Embedding(n_species, emb) if n_species else None
        self.net = _mlp(n_feat + (emb if n_species else 0), hidden, n_out, depth)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.log_k = 0.0

    def forward(self, I: torch.Tensor, species: torch.Tensor | None = None) -> torch.Tensor:
        if self.log_k > 0:
            k = self.log_k
            I = torch.sign(I) * torch.log1p(I.abs() / k) / torch.log1p(torch.tensor(1.0 / k, dtype=I.dtype))
        z0 = torch.zeros_like(I)
        if self.emb is not None:
            e = self.emb(species)
            I, z0 = torch.cat([I, e], -1), torch.cat([z0, e], -1)
        return self.lin(I[:, : self.lin.in_features]) + self.net(I) - self.net(z0)


class SharedEnvironment(nn.Module):
    """m33 environment network: one (or one per term) small trunk on the three-body channels, a narrow latent z, and a
    linear readout per emitted parameter,

        z_i   = phi(T_i) - phi(0)                       (phi: MLP to ``latent`` numbers; ``mode="per_term"``: one phi per term)
        u_p,i = W_p z_i + a_p . T_i [+ b_p . B_i + g_p(B_i) - g_p(0)]       (B = the two-body channels, only for
                                                                             readouts with ``two_body``)

    Every readout's last layer starts at zero and nothing has a bias path at zero input, so u_p = 0 exactly for an
    isolated atom (and, for readouts without two-body input, for every dimer). ``readouts``: (name, term, two_body)
    or (name, term, two_body, extra); readouts with ``extra`` also get a linear + small-MLP branch on ``n_extra``
    additional per-atom inputs (``forward(X, E)``), exactly zero where those inputs all vanish."""

    def __init__(self, n_in, three_cols, two_cols, readouts, latent=4, hidden=64, depth=2, mode="shared", tb_hidden=16,
                 n_extra=0, extra_capacity="full"):
        super().__init__()
        if mode not in ("shared", "per_term"):
            raise ValueError(f"mode {mode!r}: shared or per_term")
        self.mode, self.latent = mode, int(latent)
        self.register_buffer("three_cols", torch.tensor(list(three_cols), dtype=torch.long), persistent=False)
        self.register_buffer("two_cols", torch.tensor(list(two_cols), dtype=torch.long), persistent=False)
        self.names = [r[0] for r in readouts]
        self.terms = {r[0]: r[1] for r in readouts}
        self.two_body = {r[0]: bool(r[2]) for r in readouts}
        self.extra = {r[0]: bool(r[3]) if len(r) > 3 and n_extra else False for r in readouts}
        self.n_extra = int(n_extra)
        n3, n2 = len(three_cols), len(two_cols)
        keys = ["shared"] if mode == "shared" else sorted(set(self.terms.values()))
        self.trunks = nn.ModuleDict({k: _mlp(n3, hidden, self.latent, depth) for k in keys})
        self.W = nn.ModuleDict({p: nn.Linear(self.latent, 1, bias=False) for p in self.names})
        self.lin3 = nn.ModuleDict({p: nn.Linear(n3, 1, bias=False) for p in self.names})
        tb = [p for p in self.names if self.two_body[p]]
        self.lin2 = nn.ModuleDict({p: nn.Linear(n2, 1, bias=False) for p in tb})
        self.g2 = nn.ModuleDict({p: _mlp(n2, tb_hidden, 1, 1) for p in tb})
        ex = [p for p in self.names if self.extra[p]]
        self.line = nn.ModuleDict({p: nn.Linear(self.n_extra, 1, bias=False) for p in ex})
        # the extra branch gets a full head's capacity (hidden x depth): it carries a whole parameter (e.g. the CT
        # strengths) when its inputs are the ones that parameter needs
        eh, ed = (hidden, depth) if extra_capacity == "full" else (tb_hidden, 1)     # "small": the first m33 runs
        self.ge = nn.ModuleDict({p: _mlp(self.n_extra, eh, 1, ed) for p in ex})
        self.zero_()

    @torch.no_grad()
    def zero_(self):
        """Every readout to zero (the trunks keep their weights): all u_p = 0."""
        for p in self.names:
            nn.init.zeros_(self.W[p].weight)
            nn.init.zeros_(self.lin3[p].weight)
        for p in self.lin2:
            nn.init.zeros_(self.lin2[p].weight)
            nn.init.zeros_(self.g2[p][-1].weight)
            nn.init.zeros_(self.g2[p][-1].bias)
        for p in self.line:
            nn.init.zeros_(self.line[p].weight)
            nn.init.zeros_(self.ge[p][-1].weight)
            nn.init.zeros_(self.ge[p][-1].bias)

    def latents(self, X):
        """{trunk key: (N, latent)} z = phi(T) - phi(0)."""
        T = X[:, self.three_cols]
        T0 = torch.zeros_like(T[:1])
        live = (T != 0).any(1, keepdim=True)          # exactly 0 where every three-body channel is (dimers, isolated)
        return {k: torch.where(live, m(T) - m(T0), torch.zeros_like(T[:, :1]).expand(-1, self.latent))
                for k, m in self.trunks.items()}

    def forward(self, X, E=None):
        """X (N, n_in) scaled channels (E (N, n_extra): the extra inputs) -> {name: (N,)} log-factors."""
        T = X[:, self.three_cols]
        B = X[:, self.two_cols]
        Z = self.latents(X)
        B0 = torch.zeros_like(B[:1])
        out = {}
        for p in self.names:
            z = Z["shared"] if self.mode == "shared" else Z[self.terms[p]]
            u = self.W[p](z)[:, 0] + self.lin3[p](T)[:, 0]
            if self.two_body[p]:
                live2 = (B != 0).any(1)
                g = (self.g2[p](B) - self.g2[p](B0))[:, 0]
                u = u + self.lin2[p](B)[:, 0] + torch.where(live2, g, torch.zeros_like(g))
            if self.extra[p]:
                livee = (E != 0).any(1)
                ge = (self.ge[p](E) - self.ge[p](torch.zeros_like(E[:1])))[:, 0]
                u = u + self.line[p](E)[:, 0] + torch.where(livee, ge, torch.zeros_like(ge))
            out[p] = u
        return out
