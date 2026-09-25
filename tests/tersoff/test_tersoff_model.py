"""The Tersoff model: the pairing model's invariants on the explicit state, plus what only
makes sense here (the dimer discriminator, the closed-form formal charges, the rules)."""

import pytest
import torch

from tersoff_helpers import (
    ion_batch,
    make_model,
    make_pairing_model,
    water_cluster_batch,
    water_dimer_batch,
    zundel_batch,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture(autouse=True)
def _f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    yield
    torch.set_default_dtype(old)


def _bonded(out, r_max=1.2):
    r = out.r[out.sub_index]
    return r < r_max


def test_water_is_water():
    """Intact water reproduces the film accounting: c one on bonds and 1-3 pairs, zero
    elsewhere, no cross energy, each O-H worth the pyCMM well at the priors."""
    model = make_model()
    out = model(water_cluster_batch(2))
    p = out.bond_order
    bonded = _bonded(out)
    assert torch.allclose(p[bonded], torch.ones_like(p[bonded]), atol=1e-6)
    assert float(p[~bonded].max()) < 5e-3
    assert float(out.unpaired.max()) < 1e-6
    assert float(out.formal_charge.abs().max()) < 1e-9          # neutral, readout at zero
    assert torch.allclose(out.p_intra[out.is_intra], torch.ones(int(out.is_intra.sum())), atol=1e-6)
    assert float(out.p_intra[~out.is_intra].max()) < 5e-3
    assert float(out.interaction["cross"].abs().max()) < 1e-3
    e_int = out.fragment_energy - out.energy_ref
    assert bool(((e_int < -0.3) & (e_int > -0.5)).all())
    assert out.bo_solver is not None and all(bool(v[1].all()) for v in out.bo_solver.values())


def test_dimer_discriminator():
    """A hydrogen-bonded dimer at the priors: the covalent O-H keeps p = 1 and the hydrogen
    bond -- whose *raw* order is ~0.35 -- is squeezed out under waterfill, and leaks under
    rebo (the pairing solve's valence competition, in closed form or not)."""
    model = make_model()
    out = model(water_dimer_batch())
    bonded = _bonded(out)
    assert float(out.bond_order[bonded].min()) > 1 - 1e-6
    assert float(out.bond_order[~bonded].max()) < 1e-2
    assert float(out.interaction["cross"].abs().max()) < 1e-4
    from rsfff.ff.tersoff.bond_order import raw_bond_order

    b = raw_bond_order(out.coupling, out.kappa_pair, model.temperature)
    assert float(b[~bonded].max()) > 0.3
    rebo = make_model(tersoff_saturation="rebo")
    out_r = rebo(water_dimer_batch())
    assert float(out_r.bond_order[_bonded(out_r)].min()) < 0.9      # recorded, not a target


def test_capacity_bound_everywhere():
    model = make_model()
    for batch in (water_cluster_batch(3), water_dimer_batch(), ion_batch(), zundel_batch(shift=0.1)):
        out = model(batch)
        i, j = out.pair_index[:, out.sub_index]
        n = int(batch.positions.shape[0])
        coord = torch.zeros(n).index_add_(0, i, out.bond_order).index_add_(0, j, out.bond_order)
        deg = torch.bincount(torch.cat((i, j)), minlength=n)
        assert bool((coord <= out.electronic_state.valence + 1e-3 + 2e-3 * deg).all())


def test_isolated_fragment_vertex():
    model = make_model()
    out = model(water_cluster_batch(1, jitter=0.0))
    assert float(out.interaction["induction"].abs()) < 1e-9
    assert float(out.interaction["cross"].abs()) == 0.0
    for name in ("elst", "pauli", "disp"):
        assert float(out.interaction[name].abs()) == 0.0


def test_total_is_exact_sum_of_buckets():
    model = make_model()
    out = model(water_cluster_batch(3))
    total = out.fragment_energy.sum() + sum(v.sum() for v in out.interaction.values())
    assert torch.allclose(total, out.energy.sum())


@pytest.mark.parametrize("sat", ["waterfill", "rebo"])
def test_forces_match_finite_differences(sat):
    model = make_model(tersoff_saturation=sat)
    batch = water_dimer_batch()
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
    """d/dtheta of the force loss runs through the explicit state's second derivatives."""
    model = make_model()
    batch = water_dimer_batch()
    param = model.network.pairing_heads.d_log_kappa

    def force_loss():
        pos = batch.positions.detach().clone().requires_grad_(True)
        batch.positions = pos
        energy = model(batch).energy.sum()
        force = -torch.autograd.grad(energy, pos, create_graph=True)[0]
        return (force * force).sum()

    loss = force_loss()
    g = torch.autograd.grad(loss, param)[0]
    h = 1e-4
    with torch.no_grad():
        param[0] += h
    lp = float(force_loss())
    with torch.no_grad():
        param[0] -= 2 * h
    lm = float(force_loss())
    with torch.no_grad():
        param[0] += h
    fd = (lp - lm) / (2 * h)
    assert abs(fd - float(g[0])) < 1e-5 * max(1.0, abs(fd))


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
    assert float(out1.fragment_energy - out0.fragment_energy) > 0.05
    # homolytic: the unpaired count grows toward two (the raw order is J / kappa, and the
    # soft prior exponent leaves ~0.17 of it at 3 A), no formal charge moves
    assert float(out1.unpaired.sum()) > 1.5 and float(out0.unpaired.sum()) < 1e-6
    assert float(out1.formal_charge.abs().max()) < 1e-9


def test_formal_charges_from_the_projection():
    """H3O+ puts its charge on the oxygen (capacity 3, three bonds), OH- likewise (-1, one):
    the heavy-atom prior of the first pass and the overload weights of the second agree."""
    for rule in ("overload", "heavy_atoms"):
        model = make_model(tersoff_formal_charge=rule)
        out = model(ion_batch())
        q = out.formal_charge
        assert abs(float(q[0]) - 1.0) < 0.01 and float(q[1:4].abs().max()) < 0.01, rule
        assert abs(float(q[4]) + 1.0) < 0.01 and abs(float(q[5])) < 0.01, rule
        v = out.electronic_state.valence
        assert abs(float(v[0]) - 3.0) < 0.01 and abs(float(v[4]) - 1.0) < 0.01, rule
        bonded = _bonded(out)
        assert int(bonded.sum()) == 4
        assert float(out.bond_order[bonded].min()) > 0.99, rule
        assert float(out.unpaired.max()) < 0.01
        assert float(out.interaction["cross"].abs().max()) < 1e-3


def test_zundel_hands_the_charge_over():
    """The shared proton: an even split at the midpoint; off it, the overload rule moves the
    charge (and the third capacity) to the oxygen the proton sits on, which the heavy-atom
    prior cannot."""
    model = make_model()
    mid = model(zundel_batch())
    i, j = mid.pair_index[:, mid.sub_index]
    p1 = float(mid.bond_order[(i == 0) & (j == 3)])
    p2 = float(mid.bond_order[(i == 3) & (j == 4)])
    assert abs(p1 - p2) < 1e-6 and 0.45 < p1 < 0.5
    assert abs(float(mid.formal_charge[0]) - float(mid.formal_charge[4])) < 1e-6
    off = model(zundel_batch(shift=0.3))
    i, j = off.pair_index[:, off.sub_index]
    assert float(off.bond_order[(i == 3) & (j == 4)]) > 0.95
    assert float(off.bond_order[(i == 0) & (j == 3)]) < 0.05
    assert float(off.formal_charge[4]) > 0.9 and float(off.formal_charge[0]) < 0.1
    prior = make_model(tersoff_formal_charge="heavy_atoms")(zundel_batch(shift=0.3))
    assert abs(float(prior.formal_charge[4]) - 0.5) < 1e-6


def test_charged_reference_removes_the_ionization_offset():
    from rsfff.ff.pairing.electronic_state import e0_terms

    chi, eta = torch.tensor(0.2808177263), torch.tensor(0.4523349594)
    ip = chi + 0.5 * eta
    ea = chi - 0.5 * eta
    model = make_model()
    out = model(ion_batch())
    e_int = out.fragment_energy - out.energy_ref
    per_bond = torch.stack((
        (e_int[0] - float(ip)) / 3.0,
        e_int[1] + float(ea),
    ))
    water = make_model()(water_cluster_batch(1, jitter=0.0))
    ref = float((water.fragment_energy - water.energy_ref) / 2.0)
    assert bool(((per_bond - ref).abs() < 0.05).all())
    assert abs(float(e0_terms(torch.tensor([1.0]), chi, eta)[0]) - float(ip)) < 0.01


def test_matches_pairing_on_intact_water():
    """The ablation: on intact water the two models agree to the pairing barriers' size."""
    torch.manual_seed(0)
    tersoff = make_model()
    torch.manual_seed(0)
    pairing = make_pairing_model()
    batch = water_cluster_batch(2)
    a, b = tersoff(batch), pairing(batch)
    assert torch.allclose(a.bond_order, b.bond_order, atol=5e-3)
    assert abs(float(a.energy - b.energy)) < 0.05


def test_film_fit_runs_with_forces_on_a_tersoff_output():
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
    cfg.film.model = "tersoff"
    out = model(batch)
    loss, metrics, _ = film_fit(out, batch, cfg, training=True, with_forces=True)
    assert torch.isfinite(loss)
    assert metrics["bo_fail"] == 0.0
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert model.network.pairing_heads.d_log_kappa.grad is not None
    assert model.network.topology_heads.q_mlp[-1].weight.grad is not None


def test_config_roundtrip_builds_the_model():
    from rsfff.train.config import load_config
    from rsfff.train.build_pairing import build_model

    cfg = load_config("configs/water_tersoff.yaml")
    assert cfg.film.model == "tersoff"
    model = build_model(cfg.features, cfg.film, [1, 8], torch.tensor([-0.5, -75.0]))
    assert type(model).__name__ == "TersoffModel"
    assert model.saturation == "waterfill" and model.formal_charge == "overload"
