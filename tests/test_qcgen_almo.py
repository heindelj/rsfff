"""ALMO-EDA in pyscf: closure, invariances, the continuum bookkeeping. Small basis / GGA to keep it fast."""
import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
from rsfff.qcgen.almo import Fragment, Continuum, almo_eda, KJ, UniformField

XC, BASIS = "pbe", "def2-svp"


def nacl(R=2.6, shift=(0.0, 0.0, 0.0)):
    s = np.asarray(shift)
    return [Fragment(["Na"], np.array([[0.0, 0.0, 0.0]]) + s, charge=1),
            Fragment(["Cl"], np.array([[0.0, 0.0, R]]) + s, charge=-1)]


@pytest.fixture(scope="module")
def vac():
    return almo_eda(nacl(), xc=XC, basis=BASIS)


@pytest.fixture(scope="module")
def solv():
    return almo_eda(nacl(), xc=XC, basis=BASIS, pcm=Continuum(eps=20.0, lebedev_order=17))


def test_closure_and_signs(vac):
    t, e = vac["terms"], vac["energies"]
    assert (e["full"] - sum(e["fragments"])) * KJ == pytest.approx(t["int"], abs=1e-9)
    assert t["frz"] == pytest.approx(t["cav"] + t["cls_elec"] + t["disp"] + t["mod_pauli"], abs=1e-9)
    assert t["cav"] == 0.0                              # vacuum
    assert t["pol"] < 0 and t["ct"] < 0 and t["mod_pauli"] > 0 and t["cls_elec"] < 0 and t["disp"] < 0
    assert vac["info"]["scfmi_gmax"] < 2e-5
    assert vac["info"]["ortho"]["gmax"] < 1e-5 and vac["info"]["ortho"]["omega"] > 0


def test_frozen_decompositions_agree_on_the_frozen_total():
    c = almo_eda(nacl(), xc=XC, basis=BASIS, with_pol=False, frozen_decomp="classical")
    o = almo_eda(nacl(), xc=XC, basis=BASIS, with_pol=False, frozen_decomp="orthogonal")
    assert c["terms"]["frz"] == pytest.approx(o["terms"]["frz"], abs=1e-9)
    assert c["terms"]["cls_elec"] == pytest.approx(o["terms"]["cls_elec"], abs=1e-9)
    assert c["terms"]["disp"] != pytest.approx(o["terms"]["disp"], abs=1e-3)
    assert "pol" not in c["terms"]


def test_scfmi_is_a_minimum_between_frozen_and_full(vac):
    e = vac["energies"]
    assert e["frz"] > e["scfmi"] > e["full"]


def test_terms_are_translation_invariant_in_a_field():
    F = [0.0, 0.002, 0.004]
    a = almo_eda(nacl(), xc=XC, basis=BASIS, field=F, with_ct=False)
    b = almo_eda(nacl(shift=(1.0, -2.0, 0.5)), xc=XC, basis=BASIS, field=F, with_ct=False)
    for k in ("cls_elec", "mod_pauli", "frz", "pol"):
        assert a["terms"][k] == pytest.approx(b["terms"][k], abs=1e-5)
    # the absolute energies shift by -Q_tot F.d per fragment; the dimer is neutral so the sum is invariant
    assert sum(a["energies"]["fragments"]) == pytest.approx(sum(b["energies"]["fragments"]), abs=1e-8)


def test_field_changes_polarization_but_env_and_field_agree():
    F = np.array([0.0, 0.0, 0.01])
    a = almo_eda(nacl(), xc=XC, basis=BASIS, field=F, with_ct=False)
    b = almo_eda(nacl(), xc=XC, basis=BASIS, env=UniformField(F), with_ct=False)
    assert a["terms"]["pol"] == pytest.approx(b["terms"]["pol"], abs=1e-9)
    c = almo_eda(nacl(), xc=XC, basis=BASIS, with_ct=False)
    assert abs(a["terms"]["pol"] - c["terms"]["pol"]) > 0.01


def test_continuum_bookkeeping(solv):
    t, e, s = solv["terms"], solv["energies"], solv["solvent"]
    assert (e["full"] - sum(e["fragments"])) * KJ == pytest.approx(t["int"], abs=1e-9)
    assert t["frz"] == pytest.approx(t["cav"] + t["cls_elec"] + t["disp"] + t["mod_pauli"], abs=1e-9)
    assert t["cav"] > 0                                   # moving a solvated ion into the bigger cavity costs
    assert t["cls_elec_scr"] > 0 and abs(t["cls_elec"]) < abs(t["cls_elec_vac"])   # screening
    # Gauss: the apparent charge of a neutral solute is ~0, of each fragment -f * q
    assert abs(s["q_frz"].sum()) < 0.05
    f = s["f_epsilon"]
    from rsfff.qcgen.almo import pcm_charges
    p = solv["states"]["system"].pcm
    assert pcm_charges(p, s["V_frag_cplx"][0]).sum() == pytest.approx(-f, abs=0.05)
    # the screened cross term is the quadratic cross term of G
    assert t["cls_elec_scr"] / KJ == pytest.approx(s["G_sup"] - sum(s["G_frag_cplx"]), abs=1e-12)
    # every state's G is 1/2 q.V of the returned charges
    assert s["G_frz"] == pytest.approx(0.5 * s["q_frz"] @ s["V_frz"], abs=1e-10)
    assert s["G_full"] == pytest.approx(0.5 * s["q_full"] @ s["V_full"], abs=1e-8)


def test_continuum_screens_every_interaction(vac, solv):
    assert abs(solv["terms"]["frz"]) < abs(vac["terms"]["frz"])
    assert abs(solv["terms"]["ct"]) < abs(vac["terms"]["ct"])


def test_ferf_space_and_polarization():
    from rsfff.qcgen.almo import ferf_virtuals, multipole_operators
    from rsfff.qcgen.backend import make_mol, make_mf
    mol = make_mol(["O", "H", "H"], [[0, 0, 0.12], [0, 0.76, -0.47], [0, -0.76, -0.47]], basis=BASIS)
    mf = make_mf(mol, XC)
    mf.kernel()
    assert multipole_operators(mol, "MDQ").shape == (9, mol.nao, mol.nao)
    V = ferf_virtuals(mf, "DQ")
    nocc = int((mf.mo_occ > 0).sum())
    assert V.shape[1] <= 8 * nocc
    S = mol.intor("int1e_ovlp")
    assert abs(V.T @ S @ V - np.eye(V.shape[1])).max() < 1e-10            # orthonormal
    assert abs(V.T @ S @ mf.mo_coeff[:, mf.mo_occ > 0]).max() < 1e-10    # virtual
    Vc = ferf_virtuals(mf, "D", coupled=True)
    assert Vc.shape[1] <= 3 * nocc
    # the FERF polarization is a constrained version of the ALMO one: between it and zero
    f = almo_eda(nacl(), xc=XC, basis=BASIS, with_ct=False, pol_model="ferf", frozen_decomp="classical")
    a = almo_eda(nacl(), xc=XC, basis=BASIS, with_ct=False, pol_model="almo", frozen_decomp="classical")
    assert a["terms"]["pol"] <= f["terms"]["pol"] + 1e-5 < 0
    assert f["info"]["ferf"]["orders"] == "MDQ"


@pytest.fixture(scope="module")
def hartree():
    return almo_eda(nacl(), xc=XC, basis=BASIS, with_ct=False, frozen_decomp="classical", hartree=True)


def test_hartree_split_closes_and_has_the_right_signs(hartree):
    t = hartree["terms"]
    assert t["pol"] == pytest.approx(t["pol_elec"] + t["pol_exch"], abs=1e-9)
    assert t["pol_exch"] == pytest.approx(t["dP_onH"] + t["R_AS"], abs=1e-9)
    assert t["pol"] == pytest.approx(t["pol_elec_act"] + t["pol_exch_act"], abs=1e-9)
    assert t["pol_elec"] < 0 and t["R_AS"] <= 1e-9 and t["H_cost"] >= -1e-9
    assert t["pol_elec_act"] >= t["pol_elec"] - 1e-9          # x_H is the minimum of H
    assert hartree["info"]["hartree"]["gmax"] < 1e-6


def test_hartree_state_is_a_minimum_of_H_in_the_domain(hartree):
    import scipy.linalg as sla
    from rsfff.qcgen.almo import HartreeProduct
    st = hartree["states"]
    hp = HartreeProduct(st["system"], st["atom_slices"])
    C_H, bases = st["C_H"], st["bases"]
    S = st["system"].S
    H_H = hp.energy(C_H)
    rng = np.random.default_rng(0)
    for _ in range(3):
        Cs = []
        for c, b in zip(C_H, bases):
            u = b.T @ S @ c                                   # own-space coordinates (m, no)
            m, no = u.shape
            K = np.zeros((m, m))
            K[:no, no:] = 0.02 * rng.standard_normal((no, m - no))
            K = K - K.T
            # rotate in the basis in which u's columns come first
            Q, _ = np.linalg.qr(np.hstack([u, rng.standard_normal((m, m - no))]))
            Cs.append(b @ (Q @ sla.expm(K) @ Q.T @ u))
        assert hp.energy(Cs) > H_H


def test_hartree_long_range_limit_is_classical_induction():
    far = almo_eda(nacl(R=8.0), xc=XC, basis=BASIS, with_ct=False, frozen_decomp="classical", hartree=True)
    t = far["terms"]
    assert abs(t["pol_exch"]) < 0.02                           # no overlap: antisymmetry is irrelevant
    # -1/2 alpha E^2 of each ion in the other's field (E = 1/R^2, au), alpha of the isolated ions by finite field
    from rsfff.qcgen.backend import make_mol, make_mf
    R = 8.0 / 0.52917721092
    e_cls, F = 0.0, 1e-3
    for sym, q in (("Na", 1), ("Cl", -1)):
        mol = make_mol([sym], [[0, 0, 0]], charge=q, basis=BASIS)
        e = []
        for sgn in (1, -1):
            mf = make_mf(mol, XC)
            h = mf.get_hcore() + sgn * F * mol.intor("int1e_r", comp=3)[2]
            mf.get_hcore = lambda *a, h=h, **k: h
            mf.conv_tol = 1e-11
            e.append(mf.kernel())
        mf0 = make_mf(mol, XC)
        e0 = mf0.kernel()
        alpha = -(e[0] + e[1] - 2 * e0) / F ** 2
        e_cls -= 0.5 * alpha / R ** 4
    assert t["pol_elec"] == pytest.approx(e_cls * KJ, rel=0.15)
