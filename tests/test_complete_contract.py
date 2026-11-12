"""Breathing as shell contraction (rsfff.ff.complete.contract; theory/parameter_free_xpol.tex, step 1)."""
import math

import numpy as np
import pytest
import torch

from rsfff.ff.complete.contract import ContractTable, _antiderivative, _rho_s, contraction_modes
from rsfff.ff.ionct.params import IonTable

torch.set_default_dtype(torch.float64)
B = 0.529177210903


@pytest.fixture(scope="module")
def tab():
    t = IonTable.load(symbols=("Na", "Cl"))
    t.a0 = torch.tensor([0.0, 2.79])          # Cl breathes, Na does not
    return t


@pytest.fixture(scope="module")
def ct(tab):
    return ContractTable(tab, r_max=6.0, h=0.02)


def test_modes_carry_no_charge(tab):
    r = np.linspace(0, 60, 600001)[1:]
    for m in contraction_modes(tab, 1, r):
        assert abs(np.sum(4 * np.pi * r * r * m) * (r[1] - r[0])) < 1e-6
    assert all(not np.any(m) for m in contraction_modes(tab, 0, r))


def test_overlap_rows_are_scaling_derivatives(ct, tab):
    """ct:ovl:t = dO_t/d ln b_t and ct:ovl2:t = 1/2 d^2 O_t / d (ln b_t)^2 of shell t's overlap with the partner."""
    r = np.arange(0.005, 30, 0.01)
    G = _antiderivative(_rho_s(tab, 0))
    Ra = 2.8
    R = Ra / B
    y = ct(torch.tensor([1]), torch.tensor([0]), torch.tensor([Ra]))[0]
    nsh = [t for t in range(tab.shell_n.shape[1]) if float(tab.shell_n[1, t]) > 0]
    for t in nsh[:2]:
        n, b = float(tab.shell_n[1, t]), float(tab.shell_b[1, t])

        def O(e):
            bb = b * math.exp(e)
            m = n * bb ** 3 / (8 * np.pi) * np.exp(-bb * r)
            return (2 * np.pi / R) * np.sum(r * m * (G(r + R) - G(np.abs(r - R)))) * 0.01
        h = 1e-3
        d1, d2 = (O(h) - O(-h)) / (2 * h), (O(h) - 2 * O(0) + O(-h)) / h ** 2
        assert math.isclose(float(y[ct.index[f"ct:ovl:{t}"]]), d1, rel_tol=1e-3, abs_tol=1e-9)
        assert math.isclose(float(y[ct.index[f"ct:ovl2:{t}"]]), 0.5 * d2, rel_tol=1e-3, abs_tol=1e-9)


def test_drive_and_coupling_are_short_ranged(ct):
    y = ct(torch.tensor([1, 1]), torch.tensor([0, 1]), torch.tensor([5.9, 5.9]))
    near = ct(torch.tensor([1]), torch.tensor([1]), torch.tensor([3.2]))[0]
    assert y.abs().max() < 0.05 * near.abs().max()                # no 1/R tail: the modes carry no charge
    assert abs(float(near[ct.index["ct:J:00"]])) > 1e-6          # Cl-Cl modes overlap at contact
    assert math.isclose(float(near[ct.index["ct:J:01"]]), float(near[ct.index["ct:J:10"]]), rel_tol=1e-6)


def test_translation_rows_are_derivatives(tab):
    """tr:S1 / tr:S2 are the first and second R-derivatives (bohr) of the shell overlap tr:S."""
    from rsfff.ff.complete.contract import TranslateTable
    tt = TranslateTable(tab, r_max=6.0, h=0.02)
    R = torch.tensor([2.6, 2.6 + 1e-3, 2.6 - 1e-3])
    y = tt(torch.tensor([1, 1, 1]), torch.tensor([0, 0, 0]), R)
    hb = 1e-3 / B
    for t in range(2):
        S, S1, S2 = (y[:, tt.index[f"tr:{k}:{t}"]] for k in ("S", "S1", "S2"))
        if float(S[0]) == 0.0:
            continue
        assert math.isclose(float(S1[0]), float(S[1] - S[2]) / (2 * hb), rel_tol=1e-3)
        assert math.isclose(float(S2[0]), float(S1[1] - S1[2]) / (2 * hb), rel_tol=1e-3)


def test_strain_rows_match_brute_force(tab):
    """st:d / st:q1-3 against the overlap of the strained shell det(1 + eS) s_t((1 + eS) r) with the partner's density by
    3D quadrature about the (compact) partner, finite differences in e."""
    from rsfff.ff.complete.gram import StrainTable, _shell_density
    st = StrainTable(tab, r_max=6.0, h=0.02)
    Ra = 2.6
    rng = np.random.default_rng(3)
    M = rng.normal(size=(3, 3)); S = M + M.T; S -= np.trace(S) / 3 * np.eye(3); S /= np.sqrt(np.trace(S @ S))
    Rh = rng.normal(size=3); Rh /= np.linalg.norm(Rh); R = Ra / B * Rh
    dr_ = 0.01
    r = np.arange(700) * dr_ + dr_ / 2
    xg, wg = np.polynomial.legendre.leggauss(40)
    ph = np.arange(40) * 2 * np.pi / 40
    stt = np.sqrt(1 - xg ** 2)
    rh = np.stack([(stt[:, None] * np.cos(ph)).ravel(), (stt[:, None] * np.sin(ph)).ravel(), np.repeat(xg, 40)], 1)
    wa = np.repeat(wg, 40) * 2 * np.pi / 40
    rho = _shell_density(tab, 0, torch.tensor(r)).numpy() * r * r * dr_
    v = r[:, None, None] * rh[None] + R
    y = st(torch.tensor([1]), torch.tensor([0]), torch.tensor([Ra]))[0]
    I = ((Rh @ S @ Rh) ** 2, Rh @ S @ S @ Rh, np.trace(S @ S))
    for t in range(2):
        b = float(tab.shell_b[1, t])

        def O(e):
            u = np.linalg.norm(v @ (np.eye(3) + e * S).T, axis=-1)
            return np.linalg.det(np.eye(3) + e * S) * np.sum(b ** 3 / (8 * np.pi) * np.exp(-b * u) * wa * rho[:, None])
        h = 2e-3
        d1, d2 = (O(h) - O(-h)) / (2 * h), (O(h) - 2 * O(0) + O(-h)) / (2 * h * h)
        assert math.isclose(float(y[st.index[f"st:d:{t}"]]) * (Rh @ S @ Rh), d1, rel_tol=2e-3)
        q = sum(float(y[st.index[f"st:q{k}:{t}"]]) * x for k, x in zip((1, 2, 3), I))
        assert math.isclose(q, d2, rel_tol=2e-3)
