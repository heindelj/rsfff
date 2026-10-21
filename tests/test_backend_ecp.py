"""def2 basis sets imply the def2-ECP for Rb and heavier atoms (what Q-Chem does with ECP = def2-ecp)."""
import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
from rsfff.qcgen.backend import default_ecp, make_mol  # noqa: E402


@pytest.mark.parametrize("sym, charge, core, zeff", [("Rb", 1, 28, 9), ("Cs", 1, 46, 9), ("I", -1, 28, 25), ("Cl", -1, 0, 17)])
def test_def2_ecp_core_counts(sym, charge, core, zeff):
    m = make_mol([sym], np.zeros((1, 3)), charge=charge, basis="def2-tzvpd")
    assert m.atom_nelec_core(0) == core
    assert m.atom_charge(0) == zeff
    assert m.nelectron == zeff - charge


def test_default_ecp_rules():
    assert default_ecp(["Rb"], "def2-tzvpd") == "def2-tzvpd"
    assert default_ecp(["Na", "Cl"], "def2-tzvpd") is None
    assert default_ecp(["I"], "cc-pvdz") is None
