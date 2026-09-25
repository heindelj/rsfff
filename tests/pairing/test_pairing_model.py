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
    assert torch.allclose(p[bonded], torch.ones_like(p[bonded]), atol=1e-6)
    # a hydrogen-bonded pair keeps a trace of bond order at the priors (the pairing exponent
    # is soft and the formal charges can move by ~1e-3 to buy it); training sharpens both
    assert float(p[~bonded].max()) < 5e-3
    assert float(out.unpaired.max()) < 1e-6
    assert float(out.formal_charge.abs().max()) < 5e-3
    assert torch.allclose(out.p_intra[out.is_intra], torch.ones(int(out.is_intra.sum())), atol=1e-6)
    assert float(out.p_intra[~out.is_intra].max()) < 5e-3
    assert float(out.interaction["cross"].abs().max()) < 1e-3
    # each O-H bond is worth about the pyCMM well depth at initialization
    e_int = (out.fragment_energy - out.energy_ref)
    assert bool(((e_int < -0.3) & (e_int > -0.5)).all())


def test_isolated_fragment_vertex():
    model = make_model()
    out = model(water_cluster_batch(1, jitter=0.0))
    # the env-dressed solve warm-starts from the isolated one, so the two pairing energies
    # agree to the solver's tolerance rather than bitwise
    assert float(out.interaction["induction"].abs()) < 1e-9
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


def _ion_batch():
    """H3O+ and OH-, one frame each (in one frame the total charge would be zero, and the
    solve cannot separate an ion pair without the environment's electrostatics)."""
    from rsfff.train.data import Batch

    h3o = torch.tensor([
        [0.0, 0.0, 0.0754], [0.9408, 0.0, -0.2010],
        [-0.4704, -0.8147, -0.2010], [-0.4704, 0.8147, -0.2010],
    ])
    oh = torch.tensor([[0.0, 0.0, -0.1072], [0.0, 0.0, 0.8577]])
    positions = torch.cat((h3o, oh))
    return Batch(
        positions=positions.to(torch.get_default_dtype()),
        atomic_numbers=torch.tensor([8, 1, 1, 1, 8, 1]),
        batch_idx=torch.tensor([0, 0, 0, 0, 1, 1]),
        n_systems=2,
        energy=torch.zeros(2),
        fragment_idx=torch.tensor([0, 0, 0, 0, 1, 1]),
        fragment_charge=torch.tensor([1.0, -1.0]),
        fragment_two_s=torch.zeros(2),
        fragment_to_batch=torch.tensor([0, 1]),
        n_fragments=2,
    )


def test_formal_charges_come_out_of_the_solve():
    """H3O+ puts its charge on the oxygen (capacity 3, three bonds); OH- likewise (-1, one)."""
    model = make_model()
    out = model(_ion_batch())
    q = out.formal_charge
    assert abs(float(q[0]) - 1.0) < 0.02 and float(q[1:4].abs().max()) < 0.02
    assert abs(float(q[4]) + 1.0) < 0.05 and abs(float(q[5])) < 0.05
    v = out.electronic_state.valence
    assert abs(float(v[0]) - 3.0) < 0.02 and abs(float(v[4]) - 1.0) < 0.05
    p = out.bond_order
    r = out.r[out.sub_index]
    bonded = r < 1.2
    assert int(bonded.sum()) == 4
    assert torch.allclose(p[bonded], torch.ones_like(p[bonded]), atol=1e-3)
    assert float(out.unpaired.max()) < 0.05  # q_O of OH- is -0.987 at priors, so v0 = 1.013
    assert float(out.interaction["cross"].abs().max()) < 1e-3


def test_multiplicity_constraint_unpairs_two_electrons():
    model = make_model()
    batch = water_cluster_batch(1, jitter=0.0)
    batch.fragment_two_s = torch.tensor([2.0])
    out = model(batch)
    assert abs(float(out.unpaired.sum()) - 2.0) < 1e-6
    singlet = model(water_cluster_batch(1, jitter=0.0))
    assert float(out.energy - singlet.energy) > 0.1     # a broken bond's worth


def test_charged_reference_removes_the_ionization_offset():
    """The atomic E0(q) inside the state energy makes an O-H worth about the same in H2O,
    H3O+ and OH-: against neutral atoms the ions differ by the ionization / affinity."""
    from rsfff.ff.pairing.electronic_state import e0_terms

    chi, eta = torch.tensor(0.2808177263), torch.tensor(0.4523349594)
    e, _, _ = e0_terms(torch.tensor([1.0, -1.0, 0.0]), chi, eta)
    ip, ea = chi + 0.5 * eta, chi - 0.5 * eta
    assert abs(float(e[0]) - float(ip)) < 0.01 and abs(float(e[1]) + float(ea)) < 0.01 and abs(float(e[2])) < 1e-12

    model = make_model()
    out = model(_ion_batch())
    e_int = out.fragment_energy - out.energy_ref              # against neutral atoms
    per_bond = torch.stack((
        (e_int[0] - out.electronic_state.q.new_tensor(float(ip))) / 3.0,    # minus IP(O)
        (e_int[1] + out.electronic_state.q.new_tensor(float(ea))) / 1.0,    # plus EA(O)
    ))
    water = model(water_cluster_batch(1, jitter=0.0))
    per_bond_water = float((water.fragment_energy - water.energy_ref)[0]) / 2.0
    assert bool(((per_bond - per_bond_water).abs() < 0.08).all())
