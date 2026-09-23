"""The pairing model: accounting, vertex behaviour, forces, the film ablation."""

import pytest
import torch

from pairing_helpers import make_model, water_cluster_batch

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture(autouse=True)
def _f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    yield
    torch.set_default_dtype(old)


def test_water_is_water():
    model = make_model()
    out = model(water_cluster_batch(2))
    p = out.bond_order
    r = out.r[out.sub_index]
    bonded = r < 1.2
    assert torch.allclose(p[bonded], torch.ones_like(p[bonded]), atol=1e-8)
    assert float(p[~bonded].max()) < 1e-8
    assert float(out.unpaired.max()) < 1e-8
    # the co-membership is one on every assigned intra pair (bonds and the 1-3 pair)
    assert torch.allclose(out.p_intra[out.is_intra], torch.ones(int(out.is_intra.sum())), atol=1e-8)
    assert float(out.p_intra[~out.is_intra].max()) < 1e-8
    assert float(out.interaction["cross"].abs()) < 1e-8
    # each O-H bond is worth about the pyCMM well depth at initialization
    e_int = (out.fragment_energy - out.energy_ref)
    assert bool(((e_int < -0.3) & (e_int > -0.5)).all())


def test_isolated_fragment_vertex():
    model = make_model()
    out = model(water_cluster_batch(1, jitter=0.0))
    assert float(out.interaction["induction"].abs()) == 0.0
    assert float(out.interaction["cross"].abs()) == 0.0
    for name in ("elst", "pauli", "disp"):
        assert float(out.interaction[name].abs()) == 0.0


def test_total_is_exact_sum_of_buckets():
    model = make_model()
    out = model(water_cluster_batch(3))
    total = out.fragment_energy.sum() + sum(v.sum() for v in out.interaction.values())
    assert torch.allclose(total, out.energy.sum())


def test_forces_match_finite_differences():
    model = make_model()
    batch = water_cluster_batch(2)
    pos = batch.positions.clone().requires_grad_(True)
    batch.positions = pos
    energy = model(batch).energy.sum()
    force = -torch.autograd.grad(energy, pos)[0]
    base = pos.detach().clone()
    h = 1e-4
    for atom, axis in ((0, 1), (1, 0), (4, 2)):
        plus = base.clone(); plus[atom, axis] += h
        minus = base.clone(); minus[atom, axis] -= h
        batch.positions = plus; e_plus = model(batch).energy.sum()
        batch.positions = minus; e_minus = model(batch).energy.sum()
        fd = -(e_plus - e_minus) / (2 * h)
        assert abs(float(fd) - float(force[atom, axis])) < 1e-6 * max(1.0, abs(float(fd)))


def test_force_loss_gradient_matches_finite_differences():
    """d/dtheta of the force loss runs through the bond order's second derivatives."""
    model = make_model()
    batch = water_cluster_batch(2)
    param = model.network.pairing_heads.d_log_kappa

    def loss():
        pos = batch.positions.detach().clone().requires_grad_(True)
        batch.positions = pos
        e = model(batch).energy.sum()
        f = torch.autograd.grad(e, pos, create_graph=True)[0]
        return f.pow(2).sum()

    g = torch.autograd.grad(loss(), param)[0]
    h = 1e-4
    with torch.no_grad():
        param[0] += h
    lp = float(loss())
    with torch.no_grad():
        param[0] -= 2 * h
    lm = float(loss())
    with torch.no_grad():
        param[0] += h
    fd = (lp - lm) / (2 * h)
    assert abs(fd - float(g[0])) < 1e-5 * max(1.0, abs(fd))


def test_fermi_ablation_runs():
    model = make_model(range_gate="fermi")
    out = model(water_cluster_batch(2))
    assert torch.isfinite(out.energy).all()


def test_stretched_bond_switches_the_classical_terms_on():
    model = make_model()
    batch = water_cluster_batch(1, jitter=0.0)
    out0 = model(batch)
    stretched = batch.positions.clone()
    stretched[1] = stretched[0] + 3.0 * (stretched[1] - stretched[0]) / (stretched[1] - stretched[0]).norm()
    batch.positions = stretched
    out1 = model(batch)
    i_bond = (out1.pair_index[0] == 0) & (out1.pair_index[1] == 1)
    assert float(out0.p_intra[(out0.pair_index[0] == 0) & (out0.pair_index[1] == 1)]) > 1 - 1e-8
    assert float(out1.p_intra[i_bond]) < 0.5
    # a stretched O-H is higher in energy than the equilibrium one
    assert float(out1.fragment_energy - out0.fragment_energy) > 0.05


def test_film_fit_runs_with_forces_on_a_pairing_output():
    """The shared cluster loss (EDA channels + forces, create_graph) accepts the pairing model."""
    from rsfff.train.config import Config
    from rsfff.train.train_film import film_fit

    model = make_model()
    batch = water_cluster_batch(2)
    pos = batch.positions.detach().clone().requires_grad_(True)
    batch.positions = pos
    batch.forces = torch.zeros_like(pos)
    batch.fragment_energy = torch.full((2,), -76.4)
    batch.eda = {k: torch.zeros(1) for k in ("cls_elec", "mod_pauli", "disp", "pol", "ct")}
    cfg = Config()
    cfg.film.model = "pairing"
    out = model(batch)
    loss, metrics, _ = film_fit(out, batch, cfg, training=True, with_forces=True)
    assert torch.isfinite(loss)
    assert metrics["bo_fail"] == 0.0
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert model.network.pairing_heads.d_log_kappa.grad is not None
