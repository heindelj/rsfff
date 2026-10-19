"""Reference polarization (rsfff.qcgen.pol_ref): environments, CPKS response, induced observables.

Small basis and a functional without VV10 so CPKS and finite-field SCF agree to numerical precision.
"""

import numpy as np
import pytest
from dataclasses import replace

from rsfff.qcgen import elst_ref as E
from rsfff.qcgen import pol_ref as P

XC = "b3lyp"
WATER = (["O", "H", "H"], np.array([[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]]))


@pytest.fixture(scope="module")
def water():
    return E.Monomer.from_scf(*WATER, xc=XC, basis="def2-svpd")


@pytest.fixture(scope="module")
def water0(water):
    p = water.place()
    return p, P.scf(p, xc=XC, conv_tol=1e-12)


def test_uniform_field_sign_and_cpks_alpha(water0):
    p, mf0 = water0
    with p.mol.with_common_orig((0, 0, 0)):
        V = p.mol.intor("int1e_r", comp=3)
    d1 = P.linear_response(mf0, V)
    alpha = np.stack([P.induced_moments(p.mol, d1[x])[0] for x in range(3)], 1)
    h = 1e-3
    fd = np.zeros((3, 3))
    for y in range(3):
        F = np.zeros(3); F[y] = h
        dp = P._total_dm(P.scf(p, P.Environment(field=F), XC, dm0=P._total_dm(mf0), conv_tol=1e-12))
        dm = P._total_dm(P.scf(p, P.Environment(field=-F), XC, dm0=P._total_dm(mf0), conv_tol=1e-12))
        fd[:, y] = P.induced_moments(p.mol, (dp - dm) / 2)[0] / h
    assert np.allclose(alpha, fd, atol=2e-3 * np.abs(fd).max())
    assert np.allclose(alpha, alpha.T, atol=1e-4) and np.all(np.linalg.eigvalsh(alpha) > 0)   # induced dipole along F


def test_partner_first_order_energy_is_frozen_electrostatics(water):
    pa = water.place()
    pb = water.place(E.random_rotation(1), np.array([0.3, 5.2, 1.0]))
    env = P.Environment(partners=[pb])
    e1 = float(np.einsum("ij,ji->", env.h1(pa.mol), pa.dm)) + env.nuclear(pa.mol)
    assert abs(e1 - E.pair_energy(pa, pb, exact_ee=True)["total"]) < 1e-9


def test_point_charges_match_partner_of_bare_charges(water):
    """h1 of point charges = -sum q/|r - r_k| (via a 'partner' made of nuclei only, i.e. a ghost density)."""
    pa = water.place()
    xyz = np.array([[0.0, 4.0, 3.0], [-3.5, 0.0, -2.0]])
    q = np.array([0.7, -0.4])
    env = P.Environment(charge_coords=xyz, charges=q)
    j3 = E.df.incore.aux_e2(pa.mol, E.gto.fakemol_for_charges(xyz), intor="int3c2e", aosym="s1")
    assert np.allclose(env.h1(pa.mol), -np.einsum("ijk,k->ij", j3, q))
    # potential/field/gradient of the environment are consistent (E = -grad phi, G = dE/dr)
    x = np.array([[0.4, -0.3, 0.8]])
    h = 1e-4
    g = np.array([(env.potential(x + h * e) - env.potential(x - h * e))[0] / (2 * h) for e in np.eye(3)])
    assert np.allclose(env.field_at(x)[0], -g, atol=1e-7)
    G = np.stack([(env.field_at(x + h * e) - env.field_at(x - h * e))[0] / (2 * h) for e in np.eye(3)], 1)
    assert np.allclose(env.field_gradient_at(x)[0], G, atol=1e-6)


def test_polarize_charges_linear_limit_and_observables(water0):
    p, mf0 = water0
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(40, 3)) * 4.0
    pts = pts[np.linalg.norm(pts, axis=1) > 3.0]
    xyz = np.array([[0.0, 0.0, 7.0], [5.0, 3.0, -4.0]])
    dev = {}
    for s in (1.0, 0.1):
        env = P.Environment(charge_coords=xyz, charges=s * np.array([0.02, -0.01]))
        r = P.polarize(p, env, pts, XC, mf0=mf0, conv_tol=1e-12)
        dev[s] = np.abs(r["ddm"] - r["ddm_lin"]).max() / np.abs(r["ddm_lin"]).max()
        assert r["e_pol"] < 0
    # full SCF -> linear response as the environment weakens; the remainder is second order (hyperpolarization)
    assert dev[0.1] < 1e-3 and 0.05 < dev[0.1] / dev[1.0] < 0.15
    assert np.linalg.norm(r["dmu"] - r["dmu_lin"]) < 2e-3 * np.linalg.norm(r["dmu_lin"]) and abs(r["e_pol"] - r["e_pol_lin"]) < 1e-4 * abs(r["e_pol_lin"])
    # the induced density is neutral; far away its potential is its dipole's
    assert abs(np.trace(r["ddm"] @ p.mol.intor("int1e_ovlp"))) < 1e-7
    far = np.array([[0.0, 30.0, 0.0], [0.0, 0.0, -40.0]])
    o = r["origin"]
    d = far - o
    rr = np.linalg.norm(d, axis=1)
    Q = r["dQ"]       # second moment of the (neutral) induced density about the origin
    mult = (d @ r["dmu"]) / rr ** 3 + (3 * np.einsum("pa,ab,pb->p", d, Q, d) - rr ** 2 * np.trace(Q)) / (2 * rr ** 5)
    assert np.allclose(P.induced_potential(p, r["ddm"], far), mult, rtol=2e-3)
    assert np.allclose(r["dE"], -np.stack([(P.induced_potential(p, r["ddm"], pts + 1e-4 * e)
                                             - P.induced_potential(p, r["ddm"], pts - 1e-4 * e)) / 2e-4
                                            for e in np.eye(3)], 1), atol=1e-8)


def test_strong_field_is_nonlinear_and_variational(water0):
    p, mf0 = water0
    env = P.Environment(charge_coords=np.array([[0.0, 0.0, 5.0]]), charges=np.array([1.0]))
    r = P.polarize(p, env, None, XC, mf0=mf0)
    assert r["e_pol"] < 0
    assert abs(r["e_pol"] - r["e_pol_lin"]) > 1e-3 * abs(r["e_pol"])      # a unit charge at 5 bohr is not linear


def test_mutual_pair_converges(water):
    pa = water.place()
    pb = water.place(E.random_rotation(3), np.array([0.0, 0.0, 5.6]))
    da1, db1, _ = P.polarize_pair(pa, pb, mutual=False, xc=XC)
    da, db, info = P.polarize_pair(pa, pb, mutual=True, xc=XC)
    Sa, Sb = pa.mol.intor("int1e_ovlp"), pb.mol.intor("int1e_ovlp")
    assert info["iterations"] > 1
    assert abs(np.trace((da - pa.dm) @ Sa)) < 1e-6 and abs(np.trace((db - pb.dm) @ Sb)) < 1e-6
    # mutual induction enhances the response of a water dimer (same sign, larger norm)
    n1, n2 = np.linalg.norm(da1 - pa.dm), np.linalg.norm(da - pa.dm)
    assert n2 > n1 * 0.99
