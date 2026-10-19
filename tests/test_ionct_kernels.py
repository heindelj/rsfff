"""rsfff.ff.ionct.kernels: shaped-density electrostatics, induction and frontier potentials of monatomic ions."""
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from rsfff.ff.ionct import IonTable
from rsfff.ff.ionct import kernels as K
from rsfff.ff.multipole import spherical_to_cartesian_quadrupole
from rsfff.ff.units import BOHR_ANG

PARAMS = Path(__file__).resolve().parents[1] / "params" / "ions_wb97mv_tzvpd.json"
pytestmark = pytest.mark.skipif(not PARAMS.exists(), reason="params/ions_wb97mv_tzvpd.json not built")
DT = torch.float64


@pytest.fixture(scope="module")
def tab():
    return IonTable.load(PARAMS)


def _slater_pot(r, b):
    """Potential of a normalized 1s Slater density."""
    return (1 - (1 + b * r / 2) * np.exp(-b * r)) / r


def _shape_pot(r, k, b):
    x = b * r
    qin = 1 - np.exp(-x) * sum(x ** m / math.factorial(m) for m in range(k + 3))
    out = b / (k + 2) * np.exp(-x) * sum(x ** m / math.factorial(m) for m in range(k + 2))
    return qin / r + out


def _ion_pot(tab, s, r):
    v = float(tab.Z[s]) / r
    for n, b in zip(tab.shell_n[s].tolist(), tab.shell_b[s].tolist()):
        v = v - n * _slater_pot(r, b)
    return v


def _U(tab, s, x):
    """U(x) = int_0^x V(t) t dt for the ion's spherical potential (closed form)."""
    out = float(tab.Z[s]) * x
    for n, b in zip(tab.shell_n[s].tolist(), tab.shell_b[s].tolist()):
        I = (1 - np.exp(-b * x)) / b + (1 - np.exp(-b * x) * (1 + b * x)) / (2 * b)
        out = out - n * (x - I)
    return out


def _avg_numeric(tab, s_src, k, b, R):
    """Potential of ion s_src averaged over a normalized r^k e^{-br} shape centred at distance R: the sphere
    average of a spherical potential about a displaced centre, (U(R + r) - U(|R - r|)) / (2 r R), integrated over
    the shape's radial density by quadrature."""
    rr = np.linspace(1e-6, 60.0 / b, 200001)
    w = 4 * np.pi * rr ** 2 * b ** (k + 3) * rr ** k * np.exp(-b * rr) / (4 * np.pi * math.factorial(k + 2))
    A = (_U(tab, s_src, R + rr) - _U(tab, s_src, np.abs(R - rr))) / (2 * rr * R)
    return np.trapezoid(w * A, rr)


def _pair(r_ang, s=(0, 1)):
    X = torch.tensor([[0.0, 0.0, 0.0], [0.1, -0.2, r_ang]], dtype=DT)
    X[1] *= r_ang / X[1].norm()
    return X, torch.tensor(s), torch.zeros(2, dtype=torch.long)


@pytest.mark.parametrize("r", [1.8, 2.4, 3.5, 6.0])
def test_permanent_pair_energy_is_the_coulomb_integral_of_the_two_densities(tab, r):
    X, s, b = _pair(r)
    e = K.permanent_pair_energy(X, s, K.all_pairs(b), tab).item()
    R = r / BOHR_ANG
    # ion 0 (nucleus + shells) in the potential of ion 1: nucleus point + each shell averaged (1s, k = 0)
    ref = float(tab.Z[0]) * _ion_pot(tab, 1, R)
    for n, bb in zip(tab.shell_n[0].tolist(), tab.shell_b[0].tolist()):
        if n > 0:
            ref -= n * _avg_numeric(tab, 1, 0, bb, R)
    assert e == pytest.approx(ref, rel=2e-5, abs=1e-7)
    assert e == pytest.approx(-1 / R, rel=0.3)         # penetration: ~20 % at 1.8 A


def test_isolated_ion_in_a_uniform_field(tab):
    F = torch.tensor([[0.003, -0.002, 0.01]], dtype=DT)
    for s in range(len(tab.symbols)):
        r = K.induction(torch.zeros(1, 3, dtype=DT), torch.tensor([s]), torch.zeros(1, dtype=torch.long), tab, field=F)
        assert torch.allclose(r.mu, tab.alpha[s] * F, rtol=1e-12)
        assert r.energy.item() == pytest.approx(-0.5 * float(tab.alpha[s]) * float((F ** 2).sum()), rel=1e-12)
        assert r.theta.abs().max() < 1e-14


def test_far_point_charge_induces_theta_equal_C_g(tab):
    """Cl- far from a Na+: theta = C g with g_k = (1/3) B_k : G (pol_fit), up to the field's own dipole response."""
    R_ang = 30.0
    X = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, R_ang]], dtype=DT)
    s = torch.tensor([1, 0])                                     # Cl at the origin, Na at +z
    r = K.induction(X, s, torch.zeros(2, dtype=torch.long), tab)
    R = R_ang / BOHR_ANG
    # field gradient at the origin of a +1 charge at R z: E = -(1/R^2) z (pointing away from Na), G = dE_a/dx_b
    G = torch.diag(torch.tensor([1.0, 1.0, -2.0], dtype=DT)) / R ** 3
    B = spherical_to_cartesian_quadrupole(torch.eye(5, dtype=DT))
    g = torch.einsum("kab,ab->k", B, G) / 3
    assert torch.allclose(r.theta[0], tab.C[1] * g, rtol=2e-3, atol=1e-12)
    assert r.mu[0, 2].item() == pytest.approx(-float(tab.alpha[1]) / R ** 2, rel=2e-3)


@pytest.mark.parametrize("role", ["donor", "acceptor"])
def test_frontier_potential_of_a_permanent_ion(tab, role):
    for r in (2.4, 3.2, 5.0):
        X, s, b = _pair(r)
        Phi = K.frontier_potential_matrix(X, s, b, tab, role)
        R = r / BOHR_ANG
        for i, j in ((0, 1), (1, 0)):
            k = int(tab.front_w[role][s[i]].argmax())
            ref = _avg_numeric(tab, int(s[j]), k, float(tab.front_b[role][s[i]]), R)
            assert Phi[i, j].item() == pytest.approx(ref, rel=2e-4, abs=1e-7)
        assert Phi[0, 0] == 0 and Phi[1, 1] == 0


def test_frontier_potential_of_induced_dipoles_tends_to_the_point_dipole(tab):
    X = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 12.0]], dtype=DT)
    s, b = torch.tensor([1, 0]), torch.zeros(2, dtype=torch.long)
    x = torch.zeros(2, 8, dtype=DT)
    x[1, :3] = torch.tensor([0.3, -0.2, 0.5])
    P0 = K.frontier_potential_matrix(X, s, b, tab, "donor")
    P1 = K.frontier_potential_matrix(X, s, b, tab, "donor", x=x)
    d = (X[0] - X[1]) / BOHR_ANG
    point = float(x[1, :3] @ d) / float(d.norm()) ** 3
    assert (P1[0, 1] - P0[0, 1]).item() == pytest.approx(point, rel=1e-4)


def test_forces_by_autograd_match_finite_differences(tab):
    torch.manual_seed(0)
    X0 = torch.tensor([[0.0, 0.0, 0.0], [2.5, 0.3, 0.1], [0.2, 2.6, -0.3], [2.7, 2.4, 0.4]], dtype=DT)
    s = torch.tensor([0, 1, 1, 0])
    b = torch.zeros(4, dtype=torch.long)
    F = torch.tensor([[0.0, 0.004, -0.002]], dtype=DT)

    def energy(X):
        e = K.permanent_pair_energy(X, s, K.all_pairs(b), tab).sum()
        ind = K.induction(X, s, b, tab, field=F)
        phi = K.frontier_potential_matrix(X, s, b, tab, "donor", x=ind.x).sum()
        return e + ind.energy.sum() + 0.1 * phi

    X = X0.clone().requires_grad_(True)
    g, = torch.autograd.grad(energy(X), X, create_graph=True)
    h = 1e-5
    for a in range(4):
        for c in range(3):
            Xp, Xm = X0.clone(), X0.clone()
            Xp[a, c] += h
            Xm[a, c] -= h
            fd = (energy(Xp) - energy(Xm)).item() / (2 * h)
            assert g[a, c].item() == pytest.approx(fd, rel=1e-6, abs=1e-9)
    # double backward exists (force training)
    (g ** 2).sum().backward()
    assert X.grad is not None and torch.isfinite(X.grad).all()


def test_frames_are_independent(tab):
    X, s, _ = _pair(2.4)
    one = K.induction(X, s, torch.zeros(2, dtype=torch.long), tab)
    XX = torch.cat([X, X + 50.0])
    two = K.induction(XX, torch.cat([s, s]), torch.tensor([0, 0, 1, 1]), tab)
    assert torch.allclose(two.energy, one.energy.repeat(2), rtol=1e-12)


@pytest.mark.parametrize("role", ["donor", "acceptor"])
def test_frontier_field_is_minus_the_gradient_of_the_frontier_potential(tab, role):
    X = torch.tensor([[0, 0, 0], [2.7, 0.3, 0.1], [0.2, 2.6, -0.3], [2.9, 2.8, 0.4]], dtype=torch.float64)
    s = torch.tensor([1, 0, 0, 1])
    b = torch.zeros(4, dtype=torch.long)
    x = torch.randn(4, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(0)) * 0.1
    EV, EKk = K.frontier_field_operator(X, s, b, tab, role)
    E = EV + (EKk * x[None, :, None, :]).sum(-1)
    h = 1e-5
    for i0 in range(4):
        num = torch.zeros(4, 3, dtype=torch.float64)
        for c in range(3):
            Xp, Xm = X.clone(), X.clone()
            Xp[i0, c] += h * K.BOHR_ANG
            Xm[i0, c] -= h * K.BOHR_ANG
            Vp, Kp = K.frontier_operator(Xp, s, b, tab, role)
            Vm, Km = K.frontier_operator(Xm, s, b, tab, role)
            num[:, c] = -((Vp[i0] + (Kp[i0] * x).sum(-1)) - (Vm[i0] + (Km[i0] * x).sum(-1))) / (2 * h)
        assert torch.allclose(num, E[i0], atol=1e-8)


def test_shaped_overlap_matches_quadrature():
    import math
    import numpy as np

    def rho(r, b, n):
        return b ** (n + 3) / (4 * math.pi * math.factorial(n + 2)) * r ** n * np.exp(-b * r)

    def num(b1, n1, b2, n2, R):
        z = np.linspace(-25, 25 + R, 1201)
        s = np.linspace(0, 25, 801)
        Z, S = np.meshgrid(z, s, indexing="ij")
        f = rho(np.sqrt(Z ** 2 + S ** 2), b1, n1) * rho(np.sqrt((Z - R) ** 2 + S ** 2), b2, n2) * 2 * math.pi * S
        return np.trapezoid(np.trapezoid(f, s, axis=1), z)

    for b1, n1, b2, n2 in ((1.873, 3, 1.437, 3), (1.873, 3, 1.2577, 0), (3.546, 0, 1.2577, 0)):
        for R in (0.0, 3.0, 7.0):
            w1 = torch.zeros(1, 4, dtype=torch.float64)
            w1[0, n1] = 1
            w2 = torch.zeros(1, 4, dtype=torch.float64)
            w2[0, n2] = 1
            o = K.shaped_overlap(torch.tensor([b1], dtype=torch.float64), w1, torch.tensor([b2], dtype=torch.float64),
                                 w2, torch.tensor([R], dtype=torch.float64)).item()
            assert o == pytest.approx(num(b1, n1, b2, n2, R), rel=2e-3)


def test_point_potential_far_field_and_exclusion(tab):
    X = torch.tensor([[0, 0, 0], [0, 0, 2.8], [2.5, 1.0, 0.3]], dtype=torch.float64)
    s = torch.tensor([0, 1, 1])
    q = tab.charge[s]
    pts = torch.tensor([[9.0, 8.0, 7.0], [0.0, 0.0, 1.4]], dtype=torch.float64)
    V, Kp = K.point_potential_operator(X, s, tab, pts, torch.tensor([-1, 1]))
    r = (pts[0] - X).norm(dim=-1) / K.BOHR_ANG
    assert V[0].sum().item() == pytest.approx((q / r).sum().item(), rel=1e-6)
    assert V[1, 1].item() == 0.0 and Kp[1, 1].abs().max().item() == 0.0
