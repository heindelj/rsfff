"""pyscf is the oracle for the cavity, S and the solve; closed forms for the smeared potentials."""
import math
import numpy as np
import pytest
import torch

from rsfff.ff import cpcm

pyscf = pytest.importorskip("pyscf")
from pyscf import gto
from pyscf.solvent import pcm as ppcm

torch.set_default_dtype(torch.float64)
BOHR = cpcm.BOHR_ANG


def _mol(atoms):
    return gto.M(atom=atoms, unit="Angstrom", basis="sto-3g", charge=0, verbose=0)


@pytest.mark.parametrize("switching", ["SWIG", "ISWIG"])
@pytest.mark.parametrize("ng", [110, 302])
def test_surface_matches_pyscf(switching, ng):
    mol = _mol("Na 0 0 0; Cl 0 0 2.4; O 1.5 1.2 0.3; H 2.2 1.1 0.9; H 1.0 2.0 0.1")
    spec = cpcm.CPCMSpec(eps=78.3553, ng=ng, switching=switching)
    p = spec.to_pyscf().build(mol)
    pos = torch.tensor(mol.atom_coords())
    radii = cpcm.sphere_radii([mol.atom_symbol(i) for i in range(mol.natm)], spec)
    cav = cpcm.build_cavity(pos, radii, spec)
    surf = p.surface
    assert cav.npoints == len(surf["grid_coords"])
    assert np.abs(cav.coords.numpy() - surf["grid_coords"]).max() < 1e-12
    assert np.abs(cav.zeta.numpy() - surf["charge_exp"]).max() < 1e-12
    assert np.abs(cav.F.numpy() - surf["switch_fun"]).max() < 1e-12
    assert np.abs(cav.w.numpy() - surf["weights"]).max() < 1e-12
    assert np.abs(cav.area.numpy() - surf["area"]).max() < 1e-12
    S = cpcm.s_matrix(cav)
    assert (np.abs(S.numpy() - p._intermediates["S"]) / (1 + np.abs(p._intermediates["S"]))).max() < 1e-12
    # the solve on pyscf's own surface potential
    V = torch.tensor(p.v_grids_n)
    q, G = cpcm.solve(S, V, spec.f)
    K, R = p._intermediates["K"], p._intermediates["R"]
    q_ref = np.linalg.solve(K, R @ p.v_grids_n)
    assert np.abs(q.numpy() - q_ref).max() < 1e-10
    assert abs(float(G) - 0.5 * q_ref @ p.v_grids_n) < 1e-10


def test_nuclear_potential_is_the_smeared_point_charge_potential():
    mol = _mol("Na 0 0 0; Cl 0 0 2.4")
    spec = cpcm.CPCMSpec()
    p = spec.to_pyscf().build(mol)
    pos = torch.tensor(mol.atom_coords())
    cav = cpcm.build_cavity(pos, cpcm.sphere_radii(["Na", "Cl"], spec), spec)
    V = cpcm.surface_potential_charges(cav, pos, torch.tensor(mol.atom_charges(), dtype=torch.float64))
    assert np.abs(V.numpy() - p.v_grids_n).max() < 1e-10


def test_cosmo_and_gas_phase_f():
    assert cpcm.f_epsilon(78.3553) == pytest.approx(77.3553 / 78.3553)
    assert cpcm.f_epsilon(78.3553, "COSMO") == pytest.approx(77.3553 / 78.8553)
    assert cpcm.f_epsilon(1.0) == 0.0 and cpcm.f_epsilon(float("inf")) == 1.0


def test_slater_smeared_potential_against_quadrature():
    from scipy.integrate import quad
    spec = cpcm.CPCMSpec(ng=110)
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
    cav = cpcm.build_cavity(pos, torch.tensor([3.0, 2.5]), spec)
    n, b = 2.0, 1.3
    V = cpcm.slater_shell_surface_potential(cav, pos[:1], torch.tensor([n]), torch.tensor([b]))
    Rs = (cav.coords - pos[0]).norm(dim=-1).numpy()
    for i in range(0, cav.npoints, 23):
        R, z = Rs[i], float(cav.zeta[i])
        phi = lambda r: (1 - (1 + b * r / 2) * math.exp(-b * r)) / r
        f = lambda r: phi(r) * r * (math.exp(-z * z * (r - R) ** 2) - math.exp(-z * z * (r + R) ** 2))
        ref, _ = quad(f, 0, R + 12 / z, limit=200)
        ref *= z / (math.sqrt(math.pi) * R)
        assert float(V[i]) == pytest.approx(-n * ref, abs=1e-11)


def test_slater_smeared_potential_limits():
    spec = cpcm.CPCMSpec(ng=110)
    pos = torch.tensor([[0.0, 0.0, 0.0]])
    cav = cpcm.build_cavity(pos, torch.tensor([6.0]), spec)
    n = torch.tensor([3.0])
    # a very tight shell is a point charge
    V = cpcm.slater_shell_surface_potential(cav, pos, n, torch.tensor([400.0]))
    Vp = cpcm.surface_potential_charges(cav, pos, -n)
    assert torch.allclose(V, Vp, atol=1e-10)
    # a very narrow Gaussian sees the unsmeared Slater potential
    cav2 = cpcm.Cavity(cav.coords, cav.zeta * 1e3, cav.F, cav.w, cav.R, cav.atom, cav.nvec, cav.radii, spec)
    b = torch.tensor([0.9])
    V2 = cpcm.slater_shell_surface_potential(cav2, pos, n, b)
    R = cav.coords.norm(dim=-1)
    ref = -n * (1 - (1 + b * R / 2) * torch.exp(-b * R)) / R
    assert torch.allclose(V2, ref, atol=1e-10)


def test_born_and_gauss():
    spec = cpcm.CPCMSpec(eps=78.3553, ng=590)
    pos = torch.tensor([[0.0, 0.0, 0.0]])
    R0 = 3.0
    cav = cpcm.build_cavity(pos, torch.tensor([R0]), spec)
    S = cpcm.s_matrix(cav)
    V = cpcm.surface_potential_charges(cav, pos, torch.tensor([1.0]))
    q, G = cpcm.solve(S, V, spec.f)
    assert float(q.sum()) == pytest.approx(-spec.f, rel=1e-3)             # Gauss
    assert float(G) == pytest.approx(-0.5 * spec.f / R0, rel=2e-3)          # Born


def test_forces_by_autograd_match_finite_differences():
    spec = cpcm.CPCMSpec(eps=4.0, ng=110)
    sym = ["Na", "Cl", "Cl"]
    pos0 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 4.4], [3.9, 1.0, 1.0]])
    qa = torch.tensor([1.0, -1.0, -1.0])

    def G_of(pos):
        cav = cpcm.build_cavity(pos, cpcm.sphere_radii(sym, spec), spec)
        V = cpcm.surface_potential_charges(cav, pos, qa)
        return cpcm.solve(cpcm.s_matrix(cav), V, spec.f)[1]

    p = pos0.clone().requires_grad_(True)
    g, = torch.autograd.grad(G_of(p), p)
    h = 1e-4
    for a, x in ((1, 2), (2, 0), (0, 1)):
        dp = torch.zeros_like(pos0); dp[a, x] = h
        fd = (float(G_of(pos0 + dp)) - float(G_of(pos0 - dp))) / (2 * h)
        assert float(g[a, x]) == pytest.approx(fd, abs=1e-7)


def test_joint_blocks_recover_the_sequential_solution_when_decoupled():
    spec = cpcm.CPCMSpec(eps=10.0, ng=50)
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
    cav = cpcm.build_cavity(pos, torch.tensor([3.0, 3.0]), spec)
    S = cpcm.s_matrix(cav)
    V0 = cpcm.surface_potential_charges(cav, pos, torch.tensor([1.0, -1.0]))
    A = torch.diag(torch.tensor([2.0, 3.0]))
    L = torch.tensor([0.1, -0.2])
    J = torch.zeros(cav.npoints, 2)
    H, g = cpcm.joint_blocks(A, L, J, V0, S, spec.f)
    y = -torch.linalg.solve(H, g)
    e = 0.5 * g @ y
    e_ref = -0.5 * L @ torch.linalg.solve(A, L) + cpcm.energy(S, V0, spec.f)
    assert float(e) == pytest.approx(float(e_ref), abs=1e-12)
