"""Vectorized complete-model blocks against the loop forms they replace (kept as ``*_ref``): values and gradients."""
import pytest
import torch

from rsfff.ff.complete.model import CompleteModel

f64 = torch.float64


def _pauli_inputs(seed, N=10, T=3, P=40):
    g = torch.Generator().manual_seed(seed)
    qPs = torch.rand(N, T, generator=g, dtype=f64)
    qPs[:, -1] *= (torch.rand(N, generator=g) < 0.5).to(f64)                 # some shells switched off
    bPs = 1.0 + 2.0 * torch.rand(N, T, generator=g, dtype=f64)
    bPs[0, -1] = bPs[1, 0]                                                    # an equal-exponent shell pair
    i = torch.randint(0, N, (P,), generator=g)
    j = (i + 1 + torch.randint(0, N - 1, (P,), generator=g)) % N
    dr = torch.randn(P, 3, generator=g, dtype=f64) * 2.5
    i[0], j[0] = 0, 1
    rb = dr.norm(dim=-1)
    polym = torch.randn(N, 9, generator=g, dtype=f64) * 0.1
    outer = torch.randint(0, T, (N,), generator=g)
    return qPs, bPs, i, j, rb, dr, (polym, outer)


@pytest.mark.parametrize("moments", [False, True])
@pytest.mark.parametrize("T", [1, 2, 3])
def test_pauli_pair_energy_matches_loop(moments, T):
    qPs, bPs, i, j, rb, dr, mom = _pauli_inputs(T, T=T)
    mom = mom if moments else None
    new = CompleteModel.pauli_pair_energy(qPs, bPs, i, j, rb, dr, mom)
    ref = CompleteModel._pauli_pair_energy_ref(qPs, bPs, i, j, rb, dr, mom)
    assert new.shape == ref.shape == rb.shape
    torch.testing.assert_close(new, ref, rtol=1e-11, atol=1e-11 * float(ref.abs().max()))


@pytest.mark.parametrize("moments", [False, True])
def test_pauli_pair_energy_gradients_match_loop(moments):
    qPs, bPs, i, j, rb, dr, mom = _pauli_inputs(7)
    grads = []
    for f in (CompleteModel.pauli_pair_energy, CompleteModel._pauli_pair_energy_ref):
        q, b, d = (t.clone().requires_grad_(True) for t in (qPs, bPs, dr))
        m = (mom[0].clone().requires_grad_(True), mom[1]) if moments else None
        e = f(q, b, i, j, d.norm(dim=-1), d, m)
        ins = (q, b, d) + ((m[0],) if moments else ())
        grads.append(torch.autograd.grad((e * torch.linspace(0.5, 1.5, e.numel(), dtype=f64)).sum(), ins))
    for a, b in zip(*grads):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-10 * float(b.abs().max()))


# --- fused environment heads (one trunk pass, one readout matmul) against the two-pass, per-readout form ---------

def _randomize(m, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(0.3 * torch.randn(p.shape, generator=g, dtype=p.dtype))


READOUTS = [("a", "pauli", True, True), ("b", "pauli", False, False), ("c", "disp", True, False),
            ("d", "ct", False, True)]


@pytest.mark.parametrize("cfg", [
    dict(mode="shared"), dict(mode="per_term"), dict(mode="shared", skip=True),
    dict(mode="shared", cond_dim=3, cond_mode="film"), dict(mode="per_term", cond_dim=3, cond_mode="concatenate"),
    dict(mode="shared", cond_dim=3, cond_mode="film", two_latent=True, n_extra=2, extra_latent=True),
    dict(mode="shared", cond_dim=3, cond_mode="film", n_extra=2, skip=True),
])
def test_shared_environment_fused_matches_reference(cfg):
    from rsfff.ff.complete.heads import SharedEnvironment
    n_extra = cfg.pop("n_extra", 0)
    env = SharedEnvironment(9, (2, 3, 4, 5, 6, 7, 8), (0, 1), READOUTS, latent=3, hidden=8, depth=2, tb_hidden=4,
                            n_extra=n_extra, **cfg).double()
    _randomize(env, 1)
    g = torch.Generator().manual_seed(2)
    N = 11
    X = torch.randn(N, 9, generator=g, dtype=f64)
    X[0, 2:] = 0.0                                          # a dimer row (no three-body channels)
    X[1] = 0.0                                              # an isolated atom
    E = torch.randn(N, n_extra, generator=g, dtype=f64) if n_extra else None
    if E is not None:
        E[1] = 0.0
    c = torch.randn(N, cfg.get("cond_dim", 0), generator=g, dtype=f64) if cfg.get("cond_dim") else None
    w = {p: torch.randn(N, generator=g, dtype=f64) for p, *_ in READOUTS}

    def run(fn):
        env.zero_grad()
        out = fn(X, E, c)
        sum(w[p] * out[p] for p in out).sum().backward()
        return out, {n: p.grad.clone() for n, p in env.named_parameters() if p.grad is not None}
    new, gn = run(env.forward)
    ref, gr = run(env._forward_ref)
    assert list(new) == list(ref) == [r[0] for r in READOUTS]
    for p in ref:
        torch.testing.assert_close(new[p], ref[p], rtol=1e-12, atol=1e-14)
        assert float(new[p][1].abs()) == 0.0                # exactly zero for the isolated atom
    assert set(gn) == set(gr)
    for n in gr:
        torch.testing.assert_close(gn[n], gr[n], rtol=1e-11, atol=1e-13)
