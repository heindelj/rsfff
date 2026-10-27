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
