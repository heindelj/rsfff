"""Site polarizability / induced-density fits (rsfff.qcgen.pol_fit)."""

import numpy as np
import pytest
import torch
from scipy.integrate import quad

from rsfff.qcgen import pol_fit as PF

D = PF.DTYPE
H = PF.HYDROGEN_EXACT


def _exact_h_R(r):
    """Hydrogen atom, exact first-order response to F_z = 1: potential = R(r) cos(theta) (radial quadrature
    of the Dalgarno-Lewis density (2/pi)(r + r^2/2) e^{-2r} cos(theta))."""
    rho1 = lambda s: (2 / np.pi) * (s + s * s / 2) * np.exp(-2 * s)
    return 4 * np.pi / 3 * (quad(lambda s: s ** 3 * rho1(s), 0, r)[0] / r ** 2 + r * quad(rho1, r, np.inf)[0])


def _h_model(n_shape=3):
    w = torch.zeros(1, n_shape + 1, dtype=D)
    w[0, :3] = torch.tensor(H["w"], dtype=D)
    return torch.tensor([[[H["alpha"], 0, 0], [0, H["alpha"], 0], [0, 0, H["alpha"]]]], dtype=D), torch.tensor([H["b"]], dtype=D), w


def _h_record(scales=(0.2, 0.3, 0.5, 0.8, 1.0, 1.4, 2.0, 3.0), n=60, rng=0):
    rng = np.random.default_rng(rng)
    pts, sc = [], []
    for s in scales:
        u = rng.normal(size=(n, 3))
        pts.append(u / np.linalg.norm(u, axis=1)[:, None] * s * 1.2 / 0.52917721092)
        sc += [s] * n
    P = torch.tensor(np.concatenate(pts), dtype=D)
    A, b, w = _h_model()
    phi, E = PF.response_potential_field(torch.zeros(1, 3, dtype=D), A, b, w, P, create_graph=False)
    return {"coords": np.zeros((1, 3)), "points": P.numpy(), "scale": np.array(sc), "alpha": H["alpha"] * np.eye(3),
            "dphi_dF": phi.detach().numpy(), "dE_dF": E.detach().numpy()}


def test_exact_hydrogen_response_is_a_shaped_dipole_density():
    A, b, w = _h_model()
    r = np.array([0.2, 0.7, 1.5, 3.0, 6.0, 10.0])
    P = torch.tensor(np.stack([np.zeros_like(r), np.zeros_like(r), r], 1), dtype=D)
    phi, _ = PF.response_potential_field(torch.zeros(1, 3, dtype=D), A, b, w, P, create_graph=False)
    ref = np.array([_exact_h_R(x) for x in r])
    assert np.allclose(phi[:, 2].detach().numpy(), ref, atol=1e-11)
    assert np.allclose(phi[:, :2].detach().numpy(), 0.0, atol=1e-14)


def test_fit_recovers_hydrogen_alpha_exponent_and_shape():
    rec = _h_record()
    mod, met = PF.fit_field_response(rec, [1], PF.PolConfig(n_shape=2))
    assert abs(met["alpha_iso_model"] - 4.5) < 1e-7
    assert abs(met["b"][0] - 2.0) < 1e-7
    assert np.allclose(met["w"][0], [1 / 6, 1 / 2, 1 / 3], atol=1e-7)
    assert max(v["phi_rel"] for v in met["scales"].values()) < 1e-7


def test_alpha_constraint_modes_on_a_1s_density():
    """With a Slater 1s dipole density the grid alone does not return alpha = 4.5; 'exact' enforces it,
    'penalty' moves toward it."""
    rec = _h_record()
    free = PF.fit_field_response(rec, [1], PF.PolConfig(), b_starts=(1.0,))[1]
    exact = PF.fit_field_response(rec, [1], PF.PolConfig(alpha="exact"), b_starts=(1.0,))[1]
    pen = PF.fit_field_response(rec, [1], PF.PolConfig(alpha="penalty", alpha_weight=10.0), b_starts=(1.0,))[1]
    assert abs(free["alpha_iso_model"] - 4.5) > 1e-3
    assert abs(exact["alpha_iso_model"] - 4.5) < 1e-12
    assert abs(pen["alpha_iso_model"] - 4.5) < abs(free["alpha_iso_model"] - 4.5)
    tgt = PF.fit_field_response(rec, [1], PF.PolConfig(alpha="exact", alpha_target=4.0), b_starts=(1.0,))[1]
    assert abs(tgt["alpha_iso_model"] - 4.0) < 1e-12


def test_exact_parameterizations_hold_for_any_parameters():
    rng = np.random.default_rng(1)
    M = rng.normal(size=(3, 3))
    T = M @ M.T + 5 * np.eye(3)
    X = np.array([[0.0, 0.0, 0.2214], [0.0, 1.4309, -0.8865], [0.0, -1.4309, -0.8865]])
    for aniso, quad, cf in ((False, False, False), (True, False, False), (True, True, False), (False, True, True), (True, True, True)):
        mod = PF.PolFit([8, 1, 1], PF.PolConfig(aniso=aniso, quadrupole=quad, charge_flow=cf, alpha="exact"),
                        alpha_ref=T, coords=X)
        with torch.no_grad():
            for p in mod.parameters():
                p.add_(torch.randn_like(p))
        Pi = mod.blocks()
        assert (torch.linalg.eigvalsh(Pi) > -1e-10).all()                       # every site response PSD
        if mod.full_tensor_exact():
            assert torch.allclose(mod.molecular_alpha(), torch.tensor(T, dtype=D), atol=1e-10)
        else:
            assert abs(float(torch.trace(mod.molecular_alpha())) - np.trace(T)) < 1e-10
    mod = PF.PolFit([8, 1, 1], PF.PolConfig(alpha="exact", alpha_fixed={1: 1.2}), alpha_ref=T)
    a = mod.site_alpha()[:, 0, 0]
    assert torch.allclose(a[1:], torch.tensor([1.2, 1.2], dtype=D)) and abs(float(a.sum()) - np.trace(T) / 3) < 1e-12
    mod = PF.PolFit([8, 1], PF.PolConfig(b_fixed={1: 2.0}, b_min=1.5, tie_elements=True), b_init=0.1)
    assert float(mod.b()[1]) == 2.0 and float(mod.b()[0]) >= 1.5
    mod = PF.PolFit([8, 1], PF.PolConfig(b_sites=[2.3, 1.9]))
    assert torch.allclose(mod.b(), torch.tensor([2.3, 1.9], dtype=D)) and not mod.rb.requires_grad


def test_split_charge_response_is_the_sqe_minimum():
    X = np.array([[0.0, 0.0, 0.2214], [0.0, 1.4309, -0.8865], [0.0, -1.4309, -0.8865]])
    mod = PF.PolFit([8, 1, 1], PF.PolConfig(charge_flow=True), coords=X)
    with torch.no_grad():
        mod.ls.copy_(torch.tensor([-0.3, 0.4], dtype=D))
        mod.leta.copy_(torch.tensor([0.2, -0.5, 1.0], dtype=D))
    phi = torch.tensor([0.03, -0.02, 0.05], dtype=D)
    # direct minimization of E(p) = sum 1/2 eta q^2 + q phi + 1/2 sum p^2 / s over the bond transfers
    B, H, S = mod.Bm, torch.diag(mod.hardness()), torch.diag(mod.compliance())
    p = -torch.linalg.solve(B.T @ H @ B + torch.linalg.inv(S), B.T @ phi)
    dq, _, _ = mod.induced(torch.zeros(1, 3, 3, dtype=D), None, phi[None])
    assert torch.allclose(dq[0], B @ p, atol=1e-14) and abs(float(dq.sum())) < 1e-14
    acf = mod.charge_flow_alpha()
    assert (torch.linalg.eigvalsh(acf) > -1e-12).all() and abs(float(acf[0, 0])) < 1e-12      # in-plane, PSD
    # uniform field: dipole of dq = alpha_cf F
    Fv = torch.tensor([0.0, 0.01, -0.02], dtype=D)
    dq, _, _ = mod.induced(torch.zeros(1, 3, 3, dtype=D), None, -(torch.tensor(X) @ Fv)[None])
    assert torch.allclose(torch.tensor(X).T @ dq[0], acf @ Fv, atol=1e-14)
    # eta = 0: independent bonds, dq = -s (phi_i - phi_j)
    mod0 = PF.PolFit([8, 1, 1], PF.PolConfig(charge_flow=True, hardness="none"), coords=X)
    s = mod0.compliance()
    dq, _, _ = mod0.induced(torch.zeros(1, 3, 3, dtype=D), None, phi[None])
    ref = torch.stack([-s[0] * (phi[0] - phi[1]) - s[1] * (phi[0] - phi[2]), s[0] * (phi[0] - phi[1]), s[1] * (phi[0] - phi[2])])
    assert torch.allclose(dq[0], ref, atol=1e-14)


def test_quadrupole_conventions():
    rng = np.random.default_rng(5)
    th = torch.tensor(rng.normal(size=5), dtype=D)
    G = PF.QUAD_BASIS.T.reshape(3, 3, 5) @ torch.tensor(rng.normal(size=5), dtype=D)    # some traceless G
    G = 0.5 * (G + G.T)
    Theta = PF.spherical_to_cartesian_quadrupole(th)
    assert abs(float(-(Theta * G).sum() / 3 + th @ PF.grad_drive(G))) < 1e-14          # U = -Theta:G/3 = -theta.g
    # induced quadrupole potential = the permanent model's damped quadrupole (elst_fit)
    X = torch.tensor([[0.2, -0.1, 0.3]], dtype=D)
    P = torch.tensor(rng.normal(size=(15, 3)) * 3, dtype=D)
    b = torch.tensor([1.7], dtype=D)
    w = torch.tensor([[0.2, 0.3, 0.5]], dtype=D)
    T = PF.kernels(X, b, w, P, 2)
    phi, E = PF.moments_potential_field(T, torch.zeros(1, 1, dtype=D), torch.zeros(1, 1, 3, dtype=D), Theta[None, None])
    sp = PF.F.SiteParams(torch.zeros(1, dtype=D), torch.zeros(1, 3, dtype=D), Theta[None], torch.zeros(1, dtype=D), b, w)
    ref_phi, ref_E = PF.F.potential_field(X, sp, P, max_rank=2, create_graph=False)
    assert torch.allclose(phi[:, 0], ref_phi, atol=1e-12) and torch.allclose(E[..., 0], ref_E, atol=1e-12)
    # gradient columns: E = -grad phi, the drive is that of B_k
    Xs = torch.tensor(rng.normal(size=(3, 3)), dtype=D)
    Ek, gk, phik = PF.gradient_columns(Xs, [0.1, 0.0, -0.2])
    h = 1e-6
    for a in range(3):
        e = torch.zeros(3, dtype=D); e[a] = h
        dphi = (PF.gradient_columns(Xs + e, [0.1, 0.0, -0.2])[2] - PF.gradient_columns(Xs - e, [0.1, 0.0, -0.2])[2]) / (2 * h)
        assert torch.allclose(Ek[..., a], -dphi, atol=1e-8)
    assert torch.allclose(gk[:, 0], PF.grad_drive(PF.QUAD_BASIS))


def test_gradient_response_cpks_matches_finite_and_quadrupoles_help(tmp_path):
    from rsfff.qcgen import elst_db as Db, pol_ref as R
    db = Db.MonomerDB(tmp_path)
    db.add("w", ["O", "H", "H"], np.array([[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]]),
           xc="b3lyp", basis="def2-svpd")
    rec = R.field_response(db, "w", field_gradients=True, grid_kw=dict(density=1.0))
    fin = R.field_response(db, "w", field_gradients=True, grid_kw=dict(density=1.0), cache="pol_fd", method="finite",
                           step=5e-4)
    for k in ("alpha", "dmu_dG", "dQ_dG"):
        assert np.allclose(rec[k], fin[k], atol=2e-3 * np.abs(rec[k]).max()), k
    X = db.get("w").coords
    base = PF.fit_response(rec, [8, 1, 1], PF.PolConfig(aniso=True, alpha="exact"), coords=X, b_starts=(1.0,))[1]
    quad = PF.fit_response(rec, [8, 1, 1], PF.PolConfig(aniso=True, alpha="exact", quadrupole=True), coords=X,
                           b_starts=(1.0,))[1]
    err = lambda m, key: float(np.mean([v["phi_rel"] for v in m[key].values()]))
    assert err(quad, "scales_G") < err(base, "scales_G")


def test_reference_c_of_the_hydrogen_convention():
    """A record built from the exact H quadrupole response (C = 15) returns C = 15."""
    g0 = PF.grad_drive(PF.QUAD_BASIS)                        # drives of the five unit gradients
    Theta = PF.spherical_to_cartesian_quadrupole(15.0 * g0)  # (5, 3, 3) Buckingham
    Q = (2.0 / 3.0) * Theta.numpy()                          # traceless second moments
    assert abs(PF.reference_c({"dQ_dG": np.moveaxis(Q, 0, -1)}) - 15.0) < 1e-10


def test_qm_hydrogen_atom_matches_exact(tmp_path):
    """UHF (exact for one electron) in aug-cc-pVQZ, finite field -> alpha, and the fit returns the exact shape."""
    from rsfff.qcgen import elst_db as Db, elst_fit as EF, pol_ref as R
    db = Db.MonomerDB(tmp_path)
    db.add("H", ["H"], np.zeros((1, 3)), charge=0, spin=1, xc="hf", basis="aug-cc-pvqz")
    rec = R.field_response(db, "H", grid_kw=dict(scales=EF.INNER_SCALES, cache="grids_inner", density=1.0))
    assert rec["method"] == "finite"
    assert abs(np.trace(rec["alpha"]) / 3 - 4.5) < 0.03
    mod, met = PF.fit_field_response(rec, [1], PF.PolConfig(n_shape=3), b_starts=(1.0,))
    assert abs(met["b"][0] - 2.0) < 0.1


def test_fast_kernels_match_autograd_potential():
    rng = np.random.default_rng(4)
    X = torch.tensor(rng.normal(size=(3, 3)), dtype=D)
    P = torch.tensor(rng.normal(size=(20, 3)) * 4, dtype=D)
    M = torch.tensor(rng.normal(size=(3, 3, 3)), dtype=D)
    A = M @ M.transpose(1, 2)
    dq = torch.tensor(rng.normal(size=(3, 3)), dtype=D)
    b = torch.tensor([1.3, 2.0, 2.6], dtype=D)
    w = torch.softmax(torch.tensor(rng.normal(size=(3, 3)), dtype=D), 1)
    for ww in (None, w):
        fast = PF.response_potential_field(X, A, b, ww, P, dq=dq)
        slow = []
        for x in range(3):
            sp = PF.F.SiteParams(dq[:, x], A[:, :, x], torch.zeros(3, 3, 3, dtype=D), torch.zeros(3, dtype=D), b, ww)
            slow.append(PF.F.potential_field(X, sp, P, max_rank=1, create_graph=False))
        assert torch.allclose(fast[0], torch.stack([s_[0] for s_ in slow], 1), atol=1e-12)
        assert torch.allclose(fast[1], torch.stack([s_[1] for s_ in slow], 2), atol=1e-12)


def test_exact_hydrogen_quadrupole_response_is_a_shaped_quadrupole_density():
    """G = B_0 (phi_ext = -1/2 r^2 P2): exact induced charge density (r^2/2 + r^3/3) e^{-2r} P2 / pi (Dalgarno-Lewis)
    = second derivative of the r^0..r^3 parent with weights (7, 21, 32, 20)/80 at b = 2, and C = 15."""
    rho2 = lambda s: (s * s / 2 + s ** 3 / 3) * np.exp(-2 * s) / np.pi
    R2 = lambda r: 4 * np.pi / 5 * (quad(lambda s: s ** 4 * rho2(s), 0, r)[0] / r ** 3 + r ** 2 * quad(lambda s: rho2(s) / s, r, np.inf)[0])
    w = torch.tensor([PF.HYDROGEN_EXACT["w_quad"]], dtype=D)
    r = np.array([0.3, 1.0, 2.0, 4.0, 8.0])
    P = torch.tensor(np.stack([np.zeros_like(r), np.zeros_like(r), r], 1), dtype=D)
    T = PF.kernels(torch.zeros(1, 3, dtype=D), torch.tensor([2.0], dtype=D), None, P, 2, w_quad=w)
    g0 = PF.grad_drive(PF.QUAD_BASIS[0])
    Theta = PF.spherical_to_cartesian_quadrupole(PF.HYDROGEN_EXACT["c"] * g0)
    phi, _ = PF.moments_potential_field(T, torch.zeros(1, 1, dtype=D), torch.zeros(1, 1, 3, dtype=D), Theta[None, None])
    assert np.allclose(phi[:, 0].numpy(), [R2(x) for x in r], atol=1e-11)


def test_second_moments_match_the_induced_density():
    """Model molecular second moment = second moment of the site charge/dipole/quadrupole densities."""
    X = np.array([[0.0, 0.0, 0.2214], [0.0, 1.4309, -0.8865], [0.0, -1.4309, -0.8865]])
    mod = PF.PolFit([8, 1, 1], PF.PolConfig(aniso=True, quadrupole=True, charge_flow=True), coords=X)
    with torch.no_grad():
        for p in mod.parameters():
            p.add_(0.3 * torch.randn_like(p))
    E, g, ph = PF.gradient_columns(mod.X, [0.1, 0.2, -0.1])
    dq, mu, Th = mod.induced(E, g, ph)
    M = mod.second_moments(dq, mu, Th, [0.1, 0.2, -0.1])
    # far-field potential of the moments vs the quadrupole term of the model's own potential
    P = torch.tensor([[0.0, 0.0, 4000.0], [3000.0, 0.0, 0.0], [0.0, 2500.0, 2500.0]], dtype=D)
    T = mod.kernels(P)
    phi, _ = PF.moments_potential_field(T, dq, mu, Th)
    o = torch.tensor([0.1, 0.2, -0.1], dtype=D)
    d = P - o
    r = d.norm(dim=1)
    dip = torch.einsum("cna->ca", mu) + torch.einsum("cn,na->ca", dq, mod.X - o)
    pred = (d @ dip.T) / r[:, None] ** 3 + 1.5 * torch.einsum("pa,cab,pb->pc", d, M, d) / r[:, None] ** 5
    assert torch.allclose(phi, pred, rtol=1e-4, atol=1e-16)
