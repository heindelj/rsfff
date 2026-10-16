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
    for aniso in (False, True):
        mod = PF.PolFit([8, 1, 1], PF.PolConfig(aniso=aniso, alpha="exact"), alpha_ref=T)
        with torch.no_grad():
            for p in mod.parameters():
                p.add_(torch.randn_like(p))
        A = mod.site_alpha()
        if aniso:      # full tensor exact, each site positive definite
            assert torch.allclose(A.sum(0), torch.tensor(T, dtype=D), atol=1e-10)
            assert (torch.linalg.eigvalsh(A) > 0).all()
        else:          # isotropic mean exact
            assert abs(float(torch.trace(A.sum(0))) - np.trace(T)) < 1e-10
            assert (A[:, 0, 0] > 0).all()
    mod = PF.PolFit([8, 1, 1], PF.PolConfig(alpha="exact", alpha_fixed={1: 1.2}), alpha_ref=T)
    a = mod.site_alpha()[:, 0, 0]
    assert torch.allclose(a[1:], torch.tensor([1.2, 1.2], dtype=D)) and abs(float(a.sum()) - np.trace(T) / 3) < 1e-12
    mod = PF.PolFit([8, 1], PF.PolConfig(b_fixed={1: 2.0}, b_min=1.5, tie_elements=True), b_init=0.1)
    assert float(mod.b()[1]) == 2.0 and float(mod.b()[0]) >= 1.5


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


def test_charge_flow_alpha_and_constraints():
    X = np.array([[0.0, 0.0, 0.2214], [0.0, 1.4309, -0.8865], [0.0, -1.4309, -0.8865]])     # water, bohr
    T = np.diag([9.6, 10.1, 9.8])
    mod = PF.PolFit([8, 1, 1], PF.PolConfig(charge_flow=True, alpha="exact"), alpha_ref=T, coords=X)
    assert mod.bonds == [(0, 1), (0, 2)]
    with torch.no_grad():
        for p in mod.parameters():
            p.add_(torch.randn_like(p))
    A, k = mod.pieces()
    acf = mod.charge_flow_alpha()
    assert (k > 0).all() and (torch.linalg.eigvalsh(acf) > -1e-12).all() and abs(float(acf[0, 0])) < 1e-12
    assert abs(float(torch.trace(mod.molecular_alpha())) - np.trace(T)) < 1e-10        # isotropic mean exact
    dq = mod.dq_dF(k)
    assert torch.allclose(dq.sum(0), torch.zeros(3, dtype=D), atol=1e-14)
    # charge-flow dipole per unit field = alpha_cf
    assert torch.allclose(torch.as_tensor(X).T @ dq, acf, atol=1e-12)
    # environment: dq from potential differences, uniform field reproduces dq_dF
    F_ = np.array([0.0, 0.01, 0.0])
    assert torch.allclose(mod.dq_env(-X @ F_), dq @ torch.tensor(F_), atol=1e-14)
