"""The reference-electrostatics parameter network (rsfff.ff.ref_elec) and its training physics."""

import numpy as np
import torch

from rsfff.ff import ref_elec as RE
from rsfff.qcgen import elst_fit as EF
from rsfff.qcgen import pol_fit as PF
from rsfff.train import train_ref_elec as T

D = torch.float64
WATER = np.array([[0.0, 0.0, 0.2214], [0.0, 1.4309, -0.8865], [0.0, -1.4309, -0.8865]])
CL = np.zeros((1, 3))


def _model():
    torch.manual_seed(0)
    return RE.RefElecModel(RE.RefElecConfig()).to(D)


def test_forward_shapes_charges_and_ties():
    model = _model()
    mons = [{"numbers": [8, 1, 1], "coords": WATER, "charge": 0}, {"numbers": [17], "coords": CL, "charge": -1}]
    batch = RE.make_batch(mons, dtype=D)
    p = model(batch)
    parts = RE.split_params(p, batch)
    assert abs(float(parts[0]["q"].sum())) < 1e-10 and abs(float(parts[1]["q"].sum()) + 1) < 1e-10
    assert parts[0]["b"].shape == (3, 2) and torch.allclose(parts[0]["sw"][:, 1], torch.zeros(3, dtype=D))  # O, H: 1 shell
    assert torch.allclose(parts[1]["sw"].sum(1), torch.ones(1, dtype=D)) and float(parts[1]["Z"][0]) == 11.0
    assert torch.allclose(parts[0]["b_ind"], parts[0]["b"][:, 0])                 # induced = outermost permanent shell
    assert (parts[0]["b"] >= model.cfg.b_min).all() and (parts[0]["b"][:, 1] > parts[0]["b"][:, 0]).all()
    assert parts[0]["bond_index"].shape[1] == 3 and parts[1]["bond_index"].shape[1] == 0
    assert (torch.linalg.eigvalsh(parts[0]["alpha"]) > 0).all()


def test_flow_kernel_matches_pol_fit_sqe():
    torch.manual_seed(1)
    eta = torch.rand(3, dtype=D) + 0.1
    s = torch.rand(3, dtype=D) + 0.1
    bi = torch.tensor([[0, 0, 1], [1, 2, 2]])
    K = RE.flow_kernel(eta, s, bi, 3)
    # direct minimization of the SQE functional over the transfers
    B = torch.zeros(3, 3, dtype=D)
    B[bi[0], torch.arange(3)] = 1.0
    B[bi[1], torch.arange(3)] = -1.0
    phi = torch.tensor([0.02, -0.03, 0.01], dtype=D)
    p = -torch.linalg.solve(B.T @ torch.diag(eta) @ B + torch.diag(1 / s), B.T @ phi)
    assert torch.allclose(-K @ phi, B @ p, atol=1e-14)


def test_fast_permanent_field_matches_elst_fit():
    model = _model()
    batch = RE.make_batch([{"numbers": [8, 1, 1], "coords": WATER, "charge": 0}], dtype=D)
    p = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in RE.split_params(model(batch), batch)[0].items()}
    sp = EF.SiteParams(p["q"], p["mu"], p["theta"], p["Z"], p["b"], None, p["sw"])
    P = torch.tensor(np.random.default_rng(0).normal(size=(30, 3)) * 4, dtype=D)
    a = T.permanent_potential_field(torch.as_tensor(WATER), sp, P)
    b = EF.potential_field(torch.as_tensor(WATER), sp, P, max_rank=2, create_graph=False)
    assert torch.allclose(a[0], b[0], atol=1e-12) and torch.allclose(a[1], b[1], atol=1e-12)
