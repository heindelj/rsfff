import sys
from pathlib import Path
p = Path(sys.argv[1]) / "tests/test_complete_flux.py"
s = p.read_text()
def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (old, s.count(old))
    s = s.replace(old, new)
rep('''"""Charge-flux CT (``CompleteModel(ct_model="flux")``): the pair's zero-field energy is the direct-CT kernel's,
the compliance vanishes with it, the parallel CT polarizability of a pair, the joint minimum (stationarity in z and q),
the dimer's zero perpendicular CT response, and forces as derivatives of the energy."""''',
'''"""Charge-flux CT (``CompleteModel(ct_model="flux")``): the pair's zero-field energy is the direct-CT kernel's,
the compliance vanishes with it, the parallel CT polarizability of a pair, the joint minimum (stationarity in z and q),
the dimer's zero perpendicular CT response, and forces as derivatives of the energy. The channel-level checks use the
channel-space reference solve (``flux_operator_channel`` / ``flux_solve_channel``); the default atomic-space solve is
checked against it (energies, charges, moments, forces, parameter gradients) and through ``m(bt)``."""''')
rep('    flux = m.flux_operator(bt, sp, m.ct_charges_from(I, sp), 9 if m.breath else 8)',
    '    flux = m.flux_operator_channel(bt, sp, m.ct_charges_from(I, sp), 9 if m.breath else 8)')
rep('    """The flux block and the xpol solution of a batch, as the forward builds them."""',
    '    """The channel-space flux block and the xpol solution of a batch, as the forward builds them."""')
rep('m.flux_solve(bt, ', 'm.flux_solve_channel(bt, ', count=3)
rep('''    assert float(o["e_ct"]) == pytest.approx(float(e), rel=1e-10)
    assert torch.isfinite(o["e_int"]).all()
''', '''    assert float(o["e_ct"]) == pytest.approx(float(e), rel=1e-10)      # the forward: the atomic solve
    assert torch.isfinite(o["e_int"]).all()


def test_atomic_solve_equals_channel_solve(setup):
    """A padded three-frame batch (2, 8, 27 ions) in different applied fields: the atomic-space solve (channels
    eliminated through K = G diag(s^2) G^T) reproduces the channel-space Schur complement in E_CT, the channel charges
    and energies, the atomic charges, the joint moments, the forces and every parameter gradient."""
    tab, chan, table, feat, make, keep = setup
    m = make()
    parts = [(torch.tensor([[0.0, 0, 0], [2.9, 0, 0]], dtype=torch.float64), torch.tensor([11, 17])), _cluster(),
             _cluster27()]
    X0 = torch.cat([x + 30.0 * k for k, (x, _) in enumerate(parts)])
    Z = torch.cat([z for _, z in parts])
    fr = torch.cat([torch.full((x.shape[0],), k, dtype=torch.long) for k, (x, _) in enumerate(parts)])
    field = torch.tensor([[0.0, 0, 0], [0.002, -0.001, 0.0005], [0.0, 0.001, 0.0]], dtype=torch.float64)
    w = torch.tensor([1.0, 0.7, 1.3], dtype=torch.float64)

    def run(form):
        m.flux_solver = form
        m.zero_grad(set_to_none=True)
        X = X0.clone().requires_grad_(True)
        bt = batched_precompute(X, Z, fr, tab, table, chan, inv_form=m.inv_form)
        o = m(bt, field=field)
        E = (o["e_int"] * w).sum() + 0.5 * o["e_ct"].sum()
        gX, = torch.autograd.grad(E, X, retain_graph=True)
        E.backward()
        return o, gX, {k: p.grad.clone() for k, p in m.named_parameters() if p.grad is not None}

    o1, g1, p1 = run("channel")
    keep(m)
    o2, g2, p2 = run("atomic")
    assert m.flux_solver == "atomic"

    def close(a, b, tol=1e-11):
        return float((a - b).abs().max()) <= tol * float(b.abs().max()) + 1e-300
    for k in ("e_ct", "e_int", "e_pol", "p", "e_ch", "dq", "x"):
        assert close(o2[k].detach(), o1[k].detach()), k
    assert close(g2, g1)
    assert set(p1) == set(p2)
    for k in p1:
        assert close(p2[k], p1[k], 1e-9), k
    assert float(o2["e_ct"].detach().min()) < 0


def _cluster27(a=2.82):
    g = torch.tensor([(i, j, k) for i in range(3) for j in range(3) for k in range(3)], dtype=torch.float64)
    Z = torch.where(g.sum(1) % 2 == 0, 11, 17)
    X = g * a + 0.08 * torch.randn(g.shape[0], 3, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    return X, Z
''')
p.write_text(s)
print("patched", p)
