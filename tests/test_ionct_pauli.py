"""The Pauli kernel on permanent + induced moments (pol_form="pauli"): frozen energy, exclusion sign, assembly into the
padded induction Hessian, positive stiffening and the reduced response."""
import json
import warnings
from pathlib import Path

import pytest
import torch

from rsfff.ff.ionct import IonTable
from rsfff.ff.ionct import ct as CT
from rsfff.ff.ionct import kernels as K
from rsfff.ff.ionct.model import IonCTModel, collate, precompute
from rsfff.ff.multipole import slater_two_center_damp

PARAMS = Path(__file__).resolve().parents[1] / "params" / "ions_wb97mv_tzvpd.json"
HAS = PARAMS.exists() and "channels" in json.load(open(PARAMS))
pytestmark = pytest.mark.skipif(not HAS, reason="params with a 'channels' section not built")
DT = torch.float64
BOHR = K.BOHR_ANG


@pytest.fixture(scope="module")
def setup():
    warnings.filterwarnings("ignore")
    from bb.features import SOAPFeaturizer
    tab, chan = IonTable.load(PARAMS), CT.ChannelTable.load(PARAMS)
    feat = SOAPFeaturizer(cutoff=6.0, n_max=6, l_max=3, neighbor_types=(11, 17), selected_lambdas=(0,)).double()
    return tab, chan, feat


FRAMES = [
    (["Na", "Cl"], [[0, 0, 0], [0, 0, 2.5]]),
    (["Cl", "Na", "Cl"], [[0, 0, 0], [2.7, 0.2, 0], [4.9, 1.8, 0.3]]),
    (["Na", "Cl", "Na", "Cl"], [[0, 0, 0], [2.6, 0, 0], [2.7, 2.6, 0.1], [0.1, 2.7, -0.2]]),
    (["Cl"], [[0, 0, 0]]),
]


def _model(setup, caches, **kw):
    tab, chan, _ = setup
    torch.manual_seed(0)
    m = IonCTModel(tab, chan, caches[0].feats.shape[1], pol_form="pauli", ct_form="direct", **kw)
    with torch.no_grad():               # nontrivial anisotropic stiffening and kappa
        m.pauli_s2_raw.add_(0.4)
        m.pauli_kappa_raw.add_(torch.tensor([[0.2, -0.3], [0.5, 0.1]], dtype=DT))
    return m


def test_frozen_pauli_is_the_cmm_monopole_kernel():
    pos = torch.tensor([[0, 0, 0], [0.3, -0.4, 2.7]], dtype=DT)
    qP, bP = torch.tensor([1.1, 6.0], dtype=DT), torch.tensor([1.4, 1.6], dtype=DT)
    pairs, e0, *_ = K.pauli_kernel(pos, torch.zeros(2, dtype=torch.long), qP, bP, torch.ones(2, 8, dtype=DT),
                                   torch.ones(2, 3, dtype=DT))
    r = pos.diff(dim=0).norm() / BOHR
    f = slater_two_center_damp((1.4 * 1.6) ** 0.5 * r, 0)
    f = f[0] if f.dim() else f
    assert e0.item() == pytest.approx((1.1 * 6.0 * f / r).item(), rel=1e-12)


def test_exclusion_pushes_electrons_away_from_the_neighbour():
    # Cl- at the origin, its neighbour on +z. The electrostatic tensor convention first: a charge-dipole pair with mu
    # pointing at a positive charge is repulsive.
    X = torch.tensor([[0, 0, 0], [0, 0, 5.0]], dtype=DT)
    P = K._poly_map(DT, "cpu")
    T1 = K._tensor(X[1:] - X[:1], torch.ones(5, 1, dtype=DT))
    assert (T1[:, 0, :] @ P)[0, 2] > 0
    # Pauli: mu along +z (electron density shifted to -z, away from the neighbour) lowers E_P -> g_z < 0 on the Cl
    _, _, g, *_ = K.pauli_kernel(X * BOHR, torch.zeros(2, dtype=torch.long), torch.tensor([1.0, 6.0], dtype=DT),
                                 torch.tensor([1.4, 1.6], dtype=DT), torch.ones(2, 8, dtype=DT),
                                 torch.zeros(2, 3, dtype=DT))
    assert g[0, 2] < 0 and g[1, 2] > 0
    assert g[:, :2].abs().max() < 1e-14


def test_assembled_quadratic_matches_the_pairwise_energy(setup):
    tab, chan, feat = setup
    caches = [precompute(s, X, tab, chan, feat) for s, X in FRAMES]
    m = _model(setup, caches, pauli_nn=True)
    with torch.no_grad():
        for p in m.pauli_q.parameters():
            p.normal_(0, 0.05)
    bt = collate(caches)
    h, h0 = m.latents(bt["feats"], bt["species"])
    e, g, A = m.pauli(h, h0, bt)
    qP, bP, kappa, stiff = m.pauli_params(h, h0, bt["species"])
    torch.manual_seed(1)
    x = 0.3 * torch.randn(bt["species"].shape[0], 8, dtype=DT)
    xp = torch.zeros(bt["B"], bt["nmax"], 8, dtype=DT).masked_scatter(bt["mask"][..., None].expand(-1, -1, 8), x)
    quad = e + (xp * torch.zeros_like(xp).masked_scatter(bt["mask"][..., None].expand(-1, -1, 8), g)).sum((1, 2)) \
        + 0.5 * torch.einsum("bi,bij,bj->b", xp.reshape(bt["B"], -1), A, xp.reshape(bt["B"], -1))
    # direct: sum over pairs of p_j^T T p_i with p = [qP, -kappa x], plus 1/2 x H x on site
    X = bt["pos"] / BOHR
    P = K._poly_map(DT, "cpu")
    fr = bt["frame"]
    ref = torch.zeros(bt["B"], dtype=DT)
    for a in range(len(fr)):
        for b in range(a + 1, len(fr)):
            if fr[a] != fr[b]:
                continue
            dr = (X[b] - X[a])[None]
            r = dr.norm()
            T = K._tensor(dr, slater_two_center_damp(torch.sqrt(bP[a] * bP[b]) * r, 2)[:, None])[0]
            pa = torch.zeros(10, dtype=DT)
            pa[0] = qP[a]
            pa = pa + P @ (-kappa[a] * x[a])
            pb = torch.zeros(10, dtype=DT)
            pb[0] = qP[b]
            pb = pb + P @ (-kappa[b] * x[b])
            F = T[0, 0]
            e_ = (dr / r)[0]
            for u, v in ((a, b), (b, a)):
                Hd = qP[v] * F * (stiff[u, 0] * torch.eye(3, dtype=DT) + stiff[u, 1] * (torch.outer(e_, e_) - torch.eye(3, dtype=DT) / 3))
                ref[fr[a]] += 0.5 * x[u, :3] @ Hd @ x[u, :3] + 0.5 * qP[v] * F * stiff[u, 2] * (x[u, 3:] ** 2).sum()
            ref[fr[a]] += pb @ T @ pa
    assert torch.allclose(quad, ref, rtol=1e-10, atol=1e-14)


def test_stiffening_is_positive_and_reduces_the_response(setup):
    tab, chan, feat = setup
    caches = [precompute(s, X, tab, chan, feat) for s, X in FRAMES[:1]]
    m = _model(setup, caches)
    bt = collate(caches)
    h, h0 = m.latents(bt["feats"], bt["species"])
    qP, bP, kappa, stiff = m.pauli_params(h, h0, bt["species"])
    *_, Hon = K.pauli_kernel(bt["pos"], bt["frame"], qP, bP, kappa, stiff)
    assert torch.linalg.eigvalsh(Hon).min() > 0
    # induced dipole of the Cl- in a field along the bond: smaller with the stiffening than without
    F = torch.tensor([[0, 0, 1e-3]], dtype=DT)
    with torch.no_grad():
        dmu = (m(bt, field=F)["x"] - m(bt, field=-F)["x"])[0, 1, 2]
        m.pauli_s0_raw.fill_(-40.0)
        dmu0 = (m(bt, field=F)["x"] - m(bt, field=-F)["x"])[0, 1, 2]
    assert 0 < dmu < dmu0


def test_pauli_model_isolated_ion_and_gradients(setup):
    tab, chan, feat = setup
    caches = [precompute(s, X, tab, chan, feat) for s, X in FRAMES]
    m = _model(setup, caches, pauli_nn=True)
    out = m(collate(caches))
    assert out["e_pol"][3].abs().item() < 1e-14 and out["e_pauli"][3].item() == 0.0
    assert (out["e_pauli"][:3] > 0).all()
    (out["e_pol"].sum() + out["e_pauli"].sum()).backward()
    for name in ("pauli_lnq", "pauli_lnb", "pauli_kappa_raw", "pauli_s0_raw", "pauli_s2_raw"):
        assert getattr(m, name).grad.abs().sum() > 0, name


def test_breathing_kernel_matches_finite_differences():
    tab = IonTable.load(PARAMS)
    pos = torch.tensor([[0, 0, 0], [2.7, 0.3, 0.1], [-0.4, 2.8, 0.2]], dtype=DT)
    sp = torch.tensor([1, 0, 0])
    b = torch.zeros(3, dtype=torch.long)
    qP, bP = torch.tensor([6.0, 1.0, 1.0], dtype=DT), torch.tensor([1.6, 1.4, 1.4], dtype=DT)
    kap = torch.tensor([1.0, 0.7, 0.7], dtype=DT)
    pairs, D, Hd, Ho = K.breathing_kernel(pos, b, sp, tab, qP, bP, kap)

    def energy(a):
        i, j = pairs
        r = (pos[j] - pos[i]).norm(dim=-1) / BOHR
        u = torch.sqrt(bP[i] * bP[j]) * r * torch.exp(0.5 * (kap[i] * a[i] + kap[j] * a[j]))
        return (K._scaled_permanent_pair(r, sp[i], sp[j], tab, a[i], a[j])
                + qP[i] * qP[j] * slater_two_center_damp(u, 0)[0] / r).sum()

    h = 1e-4
    for k in range(3):
        e = torch.zeros(3, dtype=DT)
        e[k] = h
        assert D[k].item() == pytest.approx(((energy(e) - energy(-e)) / (2 * h)).item(), rel=1e-6)
        assert Hd[k].item() == pytest.approx(((energy(e) - 2 * energy(0 * e) + energy(-e)) / h ** 2).item(), rel=1e-4)
    e = torch.zeros(3, dtype=DT)
    e[0], e[1] = h, h
    f = torch.zeros(3, dtype=DT)
    f[0], f[1] = h, -h
    mixed = (energy(e) - energy(f) - energy(-f) + energy(-e)) / (4 * h * h)
    assert Ho[0].item() == pytest.approx(mixed.item(), rel=1e-4)
