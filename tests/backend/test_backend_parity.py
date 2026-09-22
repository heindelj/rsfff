"""``rsfff.ff.backend``: the torch path and the torchff path are the same function.

On CPU (here) the torchff path runs ``torchff.ffterms``' pure-torch references, so this pins
that those references *are* rsfff's formulas; on a CUDA machine with the compiled extension
the same tests exercise the kernels, including double backward. Both are run through the
whole film model, not just the leaves, so the wiring in ``FilmModel.forward`` is covered.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

from rsfff.ff import backend
from rsfff.ff.film.bonded import BondedTopology
from rsfff.train.data import Batch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "film"))
from film_helpers import water_cluster_batch  # noqa: E402
from test_model import small_model  # noqa: E402

needs_torchff = pytest.mark.skipif(not backend.HAVE_TORCHFF, reason="torchff not installed")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def both_backends():
    yield
    backend.set_backend("auto")


def _to(batch: Batch, device):
    from dataclasses import fields, replace
    kw = {}
    for f in fields(batch):
        v = getattr(batch, f.name)
        if torch.is_tensor(v):
            kw[f.name] = v.to(device)
    return replace(batch, **kw)


def _run(model, batch, name: str, *, second_order: bool):
    backend.set_backend(name)
    batch = _to(batch, DEVICE)
    batch.positions = batch.positions.detach().clone().requires_grad_(True)
    out = model(batch)
    (forces,) = torch.autograd.grad(out.energy.sum(), batch.positions, create_graph=second_order)
    res = dict(
        energy=out.energy.detach(),
        forces=forces.detach(),
        bonded=out.energy_bonded.detach(),
        **{f"e_{k}": v.detach() for k, v in out.interaction.items()},
    )
    if second_order:
        # the force-loss gradient into every parameter: this is the double backward that a
        # first-order-only kernel would silently get wrong
        loss = forces.pow(2).sum() + out.energy.pow(2).sum()
        grads = torch.autograd.grad(loss, [p for p in model.parameters() if p.requires_grad], allow_unused=True)
        res["param_grad"] = torch.cat([g.reshape(-1) for g in grads if g is not None]).detach()
    return res


@needs_torchff
@pytest.mark.parametrize("n_waters", [1, 3])
def test_film_model_agrees_across_backends(both_backends, n_waters):
    model = small_model(seed=3, randomize=True).to(DEVICE)
    batch = water_cluster_batch(n_waters, jitter=0.08, seed=11)
    a = _run(model, batch, "torch", second_order=True)
    b = _run(model, batch, "torchff", second_order=True)
    for key in a:
        assert torch.allclose(a[key], b[key], rtol=1e-9, atol=1e-11), key


@needs_torchff
def test_leaf_dispersion_matches(both_backends):
    torch.manual_seed(0)
    pos = (torch.rand(9, 3, dtype=torch.float64, device=DEVICE) * 4).requires_grad_(True)
    ii, jj = torch.triu_indices(9, 9, 1)
    pair_index = torch.stack([ii, jj]).to(DEVICE)
    c6 = (torch.rand(pair_index.shape[1], dtype=torch.float64, device=DEVICE) * 30 + 1).requires_grad_(True)
    b = (torch.rand(pair_index.shape[1], dtype=torch.float64, device=DEVICE) + 1.0).requires_grad_(True)
    backend.set_backend("torch")
    e_t = backend.tt_dispersion(pos, pair_index, c6, b)
    backend.set_backend("torchff")
    e_f = backend.tt_dispersion(pos, pair_index, c6, b)
    assert torch.allclose(e_t, e_f, rtol=1e-12, atol=1e-15)
    g_t = torch.autograd.grad(e_t.sum(), (pos, c6, b))
    g_f = torch.autograd.grad(e_f.sum(), (pos, c6, b))
    for x, y in zip(g_t, g_f):
        assert torch.allclose(x, y, rtol=1e-11, atol=1e-14)


@needs_torchff
@pytest.mark.parametrize("K", [1, 4, 10])
def test_leaf_slater_elec_matches_to_second_order(both_backends, K):
    """The elst leaf (M4): energies, first derivatives and the force-loss double backward
    agree between the torch formulas and torchff (references on CPU, kernels on CUDA)."""
    torch.manual_seed(0)
    n = 9
    pos = (torch.rand(n, 3, dtype=torch.float64, device=DEVICE) * 4).requires_grad_(True)
    ii, jj = torch.triu_indices(n, n, 1)
    pair_index = torch.stack([ii, jj]).to(DEVICE)
    b = (torch.rand(n, dtype=torch.float64, device=DEVICE) + 1.5).requires_grad_(True)
    gate = torch.rand(pair_index.shape[1], dtype=torch.float64, device=DEVICE).requires_grad_(True)
    m = (torch.randn(n, K, dtype=torch.float64, device=DEVICE) * 0.3).requires_grad_(True)
    m_nuc = torch.zeros(n, K, dtype=torch.float64, device=DEVICE)
    m_nuc[:, 0] = torch.randint(1, 8, (n,)).to(m_nuc)
    m_nuc.requires_grad_(True)
    leaves = (pos, b, gate, m, m_nuc)

    def run(name):
        backend.set_backend(name)
        e = backend.slater_elec_pair_energy(pos, pair_index, b, gate, m, m_nuc)
        w = torch.linspace(0.5, 1.5, e.numel(), dtype=e.dtype, device=e.device)
        g = torch.autograd.grad((e * w).sum(), leaves, create_graph=True)
        loss = sum((gk * gk).sum() for gk in g)
        h = torch.autograd.grad(loss, leaves)
        return e.detach(), [x.detach() for x in g], [x.detach() for x in h]

    e_t, g_t, h_t = run("torch")
    e_f, g_f, h_f = run("torchff")
    assert torch.allclose(e_t, e_f, rtol=1e-12, atol=1e-15)
    for x, y in zip(g_t, g_f):
        assert torch.allclose(x, y, rtol=1e-10, atol=1e-13)
    for x, y in zip(h_t, h_f):
        assert torch.allclose(x, y, rtol=1e-9, atol=1e-12)


@needs_torchff
@pytest.mark.parametrize("K", [1, 4, 10])
def test_leaf_slater_pauli_matches_to_second_order(both_backends, K):
    """The Pauli leaf (M3): energies, first derivatives and the force-loss double backward."""
    torch.manual_seed(0)
    n = 9
    pos = (torch.rand(n, 3, dtype=torch.float64, device=DEVICE) * 4).requires_grad_(True)
    ii, jj = torch.triu_indices(n, n, 1)
    pair_index = torch.stack([ii, jj]).to(DEVICE)
    P = pair_index.shape[1]
    b = (torch.rand(P, dtype=torch.float64, device=DEVICE) + 1.5).requires_grad_(True)
    a_i = (torch.randn(P, K, dtype=torch.float64, device=DEVICE) * 0.3).requires_grad_(True)
    a_j = (torch.randn(P, K, dtype=torch.float64, device=DEVICE) * 0.3).requires_grad_(True)
    leaves = (pos, b, a_i, a_j)

    def run(name):
        backend.set_backend(name)
        e = backend.slater_pauli_pair_energy(pos, pair_index, a_i, a_j, b)
        w = torch.linspace(0.5, 1.5, e.numel(), dtype=e.dtype, device=e.device)
        g = torch.autograd.grad((e * w).sum(), leaves, create_graph=True)
        loss = sum((gk * gk).sum() for gk in g)
        h = torch.autograd.grad(loss, leaves)
        return e.detach(), [x.detach() for x in g], [x.detach() for x in h]

    e_t, g_t, h_t = run("torch")
    e_f, g_f, h_f = run("torchff")
    assert torch.allclose(e_t, e_f, rtol=1e-12, atol=1e-15)
    for x, y in zip(g_t, g_f):
        assert torch.allclose(x, y, rtol=1e-10, atol=1e-13)
    for x, y in zip(h_t, h_f):
        assert torch.allclose(x, y, rtol=1e-9, atol=1e-12)


@needs_torchff
def test_leaf_bonded_matches(both_backends):
    from rsfff.ff.film.state import StateDescriptor
    from film_helpers import make_projector
    batch = _to(water_cluster_batch(2, jitter=0.1, seed=5), DEVICE)
    projector = make_projector().to(DEVICE)
    species = projector.species_index(batch.atomic_numbers)
    state = StateDescriptor.from_batch(batch, species, projector.featurizer.n_species)
    topo = BondedTopology.from_state(state, batch.atomic_numbers)
    nb, na = topo.bond_index.shape[1], topo.angle_index.shape[1]
    from rsfff.ff.film.bonded import BondedParameters
    params = BondedParameters(
        r_eq=torch.full((nb,), 1.81, dtype=torch.float64, device=DEVICE).requires_grad_(True),
        d=torch.full((nb,), 0.2, dtype=torch.float64, device=DEVICE).requires_grad_(True),
        k=torch.full((nb,), 0.54, dtype=torch.float64, device=DEVICE).requires_grad_(True),
        cos_theta_eq=torch.full((na,), -0.25, dtype=torch.float64, device=DEVICE).requires_grad_(True),
        k_theta=torch.full((na,), 0.17, dtype=torch.float64, device=DEVICE).requires_grad_(True),
    )
    pos = batch.positions.detach().clone().requires_grad_(True)
    backend.set_backend("torch")
    eb_t, ea_t = backend.bonded_energy(pos, topo, params)
    backend.set_backend("torchff")
    eb_f, ea_f = backend.bonded_energy(pos, topo, params)
    assert torch.allclose(eb_t, eb_f, rtol=1e-12, atol=1e-15)
    assert torch.allclose(ea_t, ea_f, rtol=1e-12, atol=1e-15)
    leaves = (pos, params.r_eq, params.d, params.k, params.cos_theta_eq, params.k_theta)
    g_t = torch.autograd.grad((eb_t.sum() + ea_t.sum()), leaves)
    g_f = torch.autograd.grad((eb_f.sum() + ea_f.sum()), leaves)
    for x, y in zip(g_t, g_f):
        assert torch.allclose(x, y, rtol=1e-11, atol=1e-14)


def test_backend_selection():
    backend.set_backend("torch")
    assert backend.active_backend(torch.zeros(1)) == "torch"
    with pytest.raises(ValueError):
        backend.set_backend("nope")
    backend.set_backend("auto")
    assert backend.active_backend(torch.zeros(1)) == "torch"  # CPU tensor -> torch path


@needs_torchff
def test_coupled_solve_on_the_fly_operator_matches_precomputed(both_backends):
    """The torchff path hands the solve no (P, K, K) tensors; the kernel (or its reference)
    rebuilds the pair operator per matvec. Same solution, same energy, same adjoint."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from rsfff.ff.coupled_solve import coupled_solve, coupled_energy, multipoles_from_state
    from rsfff.ff.polarization import build_coupled_system
    from rsfff.ff.response import ResponseParameters
    torch.manual_seed(4)
    n, dt = 9, torch.float64
    pos = (torch.rand(n, 3, dtype=dt) * 4).to(DEVICE)
    batch_idx = torch.zeros(n, dtype=torch.long, device=DEVICE)
    bond_index = torch.tensor([[0, 0, 3, 3, 6, 6], [1, 2, 4, 5, 7, 8]], device=DEVICE)
    bond_batch = torch.zeros(6, dtype=torch.long, device=DEVICE)
    ii, jj = torch.triu_indices(n, n, 1)
    pair_index = torch.stack([ii, jj]).to(DEVICE)
    gate = torch.rand(pair_index.shape[1], dtype=dt, device=DEVICE)

    def leaves():
        return dict(
            chi=torch.randn(n, dtype=dt, device=DEVICE) * 0.1,
            eta=torch.rand(n, dtype=dt, device=DEVICE) + 0.5,
            q0=torch.zeros(n, dtype=dt, device=DEVICE),
            compliance=torch.rand(6, dtype=dt, device=DEVICE) + 0.1,
            alpha=torch.eye(3, dtype=dt, device=DEVICE).expand(n, 3, 3) * 2.0,
            cquad=torch.full((n,), 0.5, dtype=dt, device=DEVICE),
            z=torch.tensor([6.0, 1, 1] * 3, dtype=dt, device=DEVICE),
            b=torch.rand(n, dtype=dt, device=DEVICE) + 1.5,
            mu0=torch.randn(n, 3, dtype=dt, device=DEVICE) * 0.1,
            quad0=torch.randn(n, 5, dtype=dt, device=DEVICE) * 0.05,
        )

    def run(name):
        backend.set_backend(name)
        torch.manual_seed(11)
        p = {k: v.clone().requires_grad_(True) for k, v in leaves().items()}
        pos_ = pos.clone().requires_grad_(True)
        gate_ = gate.clone().requires_grad_(True)
        rp = ResponseParameters(chi=p["chi"], eta=p["eta"], q0=p["q0"], compliance=p["compliance"],
                                chivec=None, alpha=p["alpha"], chiquad=None, cquad=p["cquad"],
                                z=p["z"], b=p["b"], mu0=p["mu0"], quad0=p["quad0"])
        sys_, _ = build_coupled_system(rp, positions=pos_, batch_idx=batch_idx, n_systems=1,
                                       bond_index=bond_index, bond_batch=bond_batch,
                                       pair_index=pair_index, gate=gate_, max_rank=2)
        assert sys_.on_the_fly == (name == "torchff")
        x, n_iter = coupled_solve(sys_, rtol=1e-12, atol=1e-14)
        q, mu, th = multipoles_from_state(sys_, x)
        e = coupled_energy(sys_, x)
        # a non-variational consumer of the solution, so the adjoint path is exercised
        loss = e.sum() + (q ** 3).sum() + (mu ** 2).sum() * 0.3 + th.abs().sum() * 0.1
        grads = torch.autograd.grad(loss, [pos_, gate_] + list(p.values()))
        return q.detach(), mu.detach(), th.detach(), e.detach(), [g.detach() for g in grads]

    a = run("torch")
    b_ = run("torchff")
    for x, y in zip(a[:4], b_[:4]):
        assert torch.allclose(x, y, rtol=1e-8, atol=1e-10)
    for x, y in zip(a[4], b_[4]):
        assert torch.allclose(x, y, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("name", ["torch", pytest.param("torchff", marks=needs_torchff)])
def test_force_loss_gradient_matches_central_differences(both_backends, name):
    """M5b, end to end: the force-loss gradient into a network parameter is the derivative of
    the force loss -- including the coupled solve's adjoint term, whose ``theta``-dependence
    the second backward used to drop (see ``_CoupledSolve``). Central differences on one
    parameter element, both backends.
    """
    backend.set_backend(name)
    torch.manual_seed(0)
    model = small_model(seed=3, randomize=True).to(DEVICE).double()
    batch = _to(water_cluster_batch(3, jitter=0.08, seed=11), DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]

    def loss_value(create_graph):
        b = batch
        b.positions = b.positions.detach().clone().requires_grad_(True)
        out = model(b)
        (forces,) = torch.autograd.grad(out.energy.sum(), b.positions, create_graph=create_graph)
        return forces.pow(2).sum() + out.energy.pow(2).sum()

    loss = loss_value(True)
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    # the largest-gradient elements of a few tensors, so the check is not on something tiny
    checked = 0
    h = 1e-4
    for p, g in zip(params, grads):
        if g is None or g.numel() == 0 or float(g.abs().max()) < 1e-6:
            continue
        idx = int(g.abs().reshape(-1).argmax())
        flat = p.data.reshape(-1)
        old = float(flat[idx])
        flat[idx] = old + h
        lp = float(loss_value(False))
        flat[idx] = old - h
        lm = float(loss_value(False))
        flat[idx] = old
        fd = (lp - lm) / (2 * h)
        an = float(g.reshape(-1)[idx])
        assert abs(fd - an) < 2e-5 * max(1.0, abs(fd)), (p.shape, idx, fd, an)
        checked += 1
        if checked == 4:
            break
    assert checked >= 2
