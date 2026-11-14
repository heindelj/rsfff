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
        u_p,i = W_p z_i [+ a_p . T_i] [+ b_p . B_i + g_p(B_i) - g_p(0)]       (B = the two-body channels, only for
                                                                             readouts with ``two_body``)

    Every readout's last layer starts at zero and nothing has a bias path at zero input, so u_p = 0 exactly for an
    isolated atom (and, for readouts without two-body input, for every dimer). ``readouts``: (name, term, two_body)
    or (name, term, two_body, extra); readouts with ``extra`` also get a linear + small-MLP branch on ``n_extra``
    additional per-atom inputs (``forward(X, E)``), exactly zero where those inputs all vanish.

    ``cond_dim > 0`` (m35): the encoders -- the trunk(s) and the two-body branches g_p -- also read a per-atom
    conditioning vector c (the centre atom's species embedding, :mod:`.species`), by FiLM (``cond_mode="film"``,
    bb.nn.ConditionedTrunk: per-layer scale and shift generated from c, zero at initialisation) or by concatenation
    (``"concatenate"``); ``forward(X, E, c)``. The readouts (W_p, a_p, b_p) stay shared by every species. The isolated
    and dimer limits hold per atom: z = phi(T, c) - phi(0, c), g = g(B, c) - g(0, c).

    ``skip`` (False by default since 2026-11-11): the per-readout linear skip a_p . T. Without it every readout's
    three-body dependence goes through the shared (conditioned) latent alone, u_p = W_p z [+ the two-body branch];
    widen ``latent`` if it needs more. Checkpoints from before carry it (train_ionct.build_model restores it)."""

    def __init__(self, n_in, three_cols, two_cols, readouts, latent=4, hidden=64, depth=2, mode="shared", tb_hidden=16,
                 n_extra=0, extra_capacity="full", cond_dim=0, cond_mode="film", extra_latent=False, skip=False,
                 two_latent=False):
        super().__init__()
        # two_latent (2026-11-12): the two-body channels join the trunk's input and no readout has a two-body branch --
        # every readout is W_p z of one latent that carries the dimer-level and the many-body dependence together
        self.two_latent = bool(two_latent)
        # extra_latent (m35): the extra inputs join the three-body channels as the trunk's input -- one latent from
        # one feature set for every readout -- instead of a per-readout side branch. The latent is then live wherever
        # either is non-zero (on dimers too, through the extra inputs)
        self.extra_latent = bool(extra_latent) and n_extra > 0
        self.cond_dim, self.cond_mode = int(cond_dim), cond_mode
        if self.cond_dim and cond_mode not in ("film", "concatenate"):
            raise ValueError(f"cond_mode {cond_mode!r}: film or concatenate")
        if mode not in ("shared", "per_term"):
            raise ValueError(f"mode {mode!r}: shared or per_term")
        self.mode, self.latent = mode, int(latent)
        self.register_buffer("three_cols", torch.tensor(list(three_cols), dtype=torch.long), persistent=False)
        self.register_buffer("two_cols", torch.tensor(list(two_cols), dtype=torch.long), persistent=False)
        self.names = [r[0] for r in readouts]
        self.terms = {r[0]: r[1] for r in readouts}
        self.two_body = {r[0]: bool(r[2]) and not self.two_latent for r in readouts}
        self.extra = {r[0]: bool(r[3]) if len(r) > 3 and n_extra and not self.extra_latent else False
                      for r in readouts}
        self.n_extra = int(n_extra)
        n3, n2 = len(three_cols), len(two_cols)
        keys = ["shared"] if mode == "shared" else sorted(set(self.terms.values()))
        n_trunk = n3 + (n2 if self.two_latent else 0) + (self.n_extra if self.extra_latent else 0)
        self.trunks = nn.ModuleDict({k: self._encoder(n_trunk, hidden, self.latent, depth) for k in keys})
        self.W = nn.ModuleDict({p: nn.Linear(self.latent, 1, bias=False) for p in self.names})
        self.skip = bool(skip)
        self.lin3 = nn.ModuleDict({p: nn.Linear(n3, 1, bias=False) for p in self.names} if self.skip else {})
        tb = [p for p in self.names if self.two_body[p]]
        self.lin2 = nn.ModuleDict({p: nn.Linear(n2, 1, bias=False) for p in tb})
        self.g2 = nn.ModuleDict({p: self._encoder(n2, tb_hidden, 1, 1) for p in tb})
        ex = [p for p in self.names if self.extra[p]]
        self.line = nn.ModuleDict({p: nn.Linear(self.n_extra, 1, bias=False) for p in ex})
        # the extra branch gets a full head's capacity (hidden x depth): it carries a whole parameter (e.g. the CT
        # strengths) when its inputs are the ones that parameter needs
        eh, ed = (hidden, depth) if extra_capacity == "full" else (tb_hidden, 1)     # "small": the first m33 runs
        self.ge = nn.ModuleDict({p: _mlp(self.n_extra, eh, 1, ed) for p in ex})
        self.zero_()

    def _encoder(self, i, h, o, depth):
        return _mlp(i, h, o, depth) if not self.cond_dim else _CondMLP(i, h, o, depth, self.cond_dim, self.cond_mode)

    def _enc(self, net, x, c, x0):
        """net(x, c) - net(x0, c): x0 the zero input (one row, broadcast). One pass of the network: unconditioned, the
        zero row rides along as an extra row; conditioned, :meth:`_CondMLP.diff` (the FiLM generators run once)."""
        if not self.cond_dim:
            y = net(torch.cat([x, x0], 0))
            return y[:-1] - y[-1:]
        return net.diff(x, x0, c)

    def _enc_ref(self, net, x, c, x0):
        """Two-pass form of :meth:`_enc` (kept for tests)."""
        if not self.cond_dim:
            return net(x) - net(x0)
        return net(x, c) - net(x0.expand_as(x), c)

    @torch.no_grad()
    def zero_(self):
        """Every readout to zero (the trunks keep their weights): all u_p = 0."""
        for p in self.names:
            nn.init.zeros_(self.W[p].weight)
            if p in self.lin3:
                nn.init.zeros_(self.lin3[p].weight)
        for p in self.lin2:
            nn.init.zeros_(self.lin2[p].weight)
            last = self.g2[p][-1] if not self.cond_dim else self.g2[p].out
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        for p in self.line:
            nn.init.zeros_(self.line[p].weight)
            nn.init.zeros_(self.ge[p][-1].weight)
            nn.init.zeros_(self.ge[p][-1].bias)

    def latents(self, X, c=None, E=None, ref=False):
        """{trunk key: (N, latent)} z = phi(T, c) - phi(0, c) (T: the three-body channels, with the extra inputs
        appended when ``extra_latent``)."""
        if self.cond_dim and c is None:
            raise ValueError("SharedEnvironment with cond_dim needs the conditioning vectors c")
        T = X[:, self.three_cols]
        if getattr(self, "two_latent", False):
            T = torch.cat([T, X[:, self.two_cols]], 1)
        if self.extra_latent:
            if E is None:
                raise ValueError("SharedEnvironment with extra_latent needs the extra inputs E")
            T = torch.cat([T, E], 1)
        T0 = torch.zeros_like(T[:1])
        live = (T != 0).any(1, keepdim=True)          # exactly 0 where every three-body channel is (dimers, isolated)
        enc = self._enc_ref if ref else self._enc
        return {k: torch.where(live, enc(m, T, c, T0), torch.zeros_like(T[:, :1]).expand(-1, self.latent))
                for k, m in self.trunks.items()}

    def forward(self, X, E=None, c=None):
        """X (N, n_in) scaled channels (E (N, n_extra): the extra inputs; c (N, cond_dim): the conditioning) ->
        {name: (N,)} log-factors. The readouts W_p (and the skips a_p) of every parameter on the same latent are one
        matmul; the two-body and extra branches, when present, are added per parameter."""
        T = X[:, self.three_cols]
        B = X[:, self.two_cols]
        Z = self.latents(X, c, E)
        groups = {}
        for p in self.names:
            groups.setdefault("shared" if self.mode == "shared" else self.terms[p], []).append(p)
        out = {}
        for k, ps in groups.items():
            U = Z[k] @ torch.cat([self.W[p].weight for p in ps], 0).T                    # (N, len(ps))
            if any(p in self.lin3 for p in ps):         # the skips a_p . T (zero rows for readouts without one)
                U = U + T @ torch.cat([self.lin3[p].weight if p in self.lin3 else
                                       T.new_zeros(1, T.shape[1]) for p in ps], 0).T
            for m, p in enumerate(ps):
                out[p] = U[:, m]
        B0 = torch.zeros_like(B[:1])
        for p in self.names:
            u = out[p]
            if self.two_body[p]:
                live2 = (B != 0).any(1)
                g = self._enc(self.g2[p], B, c, B0)[:, 0]
                u = u + self.lin2[p](B)[:, 0] + torch.where(live2, g, torch.zeros_like(g))
            if self.extra[p]:
                livee = (E != 0).any(1)
                ge = (self.ge[p](E) - self.ge[p](torch.zeros_like(E[:1])))[:, 0]
                u = u + self.line[p](E)[:, 0] + torch.where(livee, ge, torch.zeros_like(ge))
            out[p] = u
        return {p: out[p] for p in self.names}

    def _forward_ref(self, X, E=None, c=None):
        """Per-readout, two-pass reference of :meth:`forward` (kept for tests).

        X (N, n_in) scaled channels (E (N, n_extra): the extra inputs; c (N, cond_dim): the conditioning) ->
        {name: (N,)} log-factors."""
        T = X[:, self.three_cols]
        B = X[:, self.two_cols]
        Z = self.latents(X, c, E, ref=True)
        B0 = torch.zeros_like(B[:1])
        out = {}
        for p in self.names:
            z = Z["shared"] if self.mode == "shared" else Z[self.terms[p]]
            u = self.W[p](z)[:, 0]
            if p in self.lin3:
                u = u + self.lin3[p](T)[:, 0]
            if self.two_body[p]:
                live2 = (B != 0).any(1)
                g = self._enc_ref(self.g2[p], B, c, B0)[:, 0]
                u = u + self.lin2[p](B)[:, 0] + torch.where(live2, g, torch.zeros_like(g))
            if self.extra[p]:
                livee = (E != 0).any(1)
                ge = (self.ge[p](E) - self.ge[p](torch.zeros_like(E[:1])))[:, 0]
                u = u + self.line[p](E)[:, 0] + torch.where(livee, ge, torch.zeros_like(ge))
            out[p] = u
        return out


class _CondMLP(nn.Module):
    """``depth`` conditioned hidden layers (bb.nn.ConditionedTrunk, ending on an activation) and a linear output:
    the conditioned counterpart of ``_mlp(i, h, o, depth)``; ``forward(x, c)``."""

    def __init__(self, i, h, o, depth, cond_dim, cond_mode="film"):
        super().__init__()
        from bb.nn.conditioning import ConditionedTrunk
        self.trunk = ConditionedTrunk(i, h, depth, d_c=cond_dim, mode=cond_mode)
        self.out = nn.Linear(h, o)

    def forward(self, x, c):
        return self.out(self.trunk(x, c))

    def diff(self, x, x0, c):
        """net(x, c) - net(x0, c) in one pass: the FiLM generators run once on c and both inputs go through the
        layers stacked (2, N, .) (x0: zero row(s), broadcast to x)."""
        tr = self.trunk
        h = torch.stack([x, x0.expand_as(x)], 0)
        if tr.mode == "concatenate":
            h = torch.cat([h, c.expand(2, -1, -1)], -1)
        for k, layer in enumerate(tr.layers):
            h = layer(h, tr.generators[k](c) if tr.generators is not None else None)
        y = self.out(h)
        return y[0] - y[1]
