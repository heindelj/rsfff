"""Diabatic mixture: the dissociation-limit properties that must hold by construction.

These mirror the free-atom limit test of the monomer stack (test_monomer_stack.py) one level up:
the whole point of the mixture machinery is that pulling a bond apart collapses the model
*exactly* onto the sum of its fragment reference predictions, with integer charges and a
continuous force. None of that is trained — it follows from the dissociated diabat partitioning
into isolated fragments, the renormalized envelope gate, and the C² compliance switch — so the
tests run on a small randomly-initialized model, not the checkpoint.
"""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch

from rsfff.mlip import (
    AtomicStateReference,
    DiabaticStateLibrary,
    EnvelopeConfig,
    MixtureModel,
    assign_from_headers,
    build_monomer_model,
    enumerate_diabats,
    mixture_channel_graph,
)
from rsfff.train.data import Batch

LIBRARY = "data/diabatic_states.yaml"
STATES = "data/atomic_reference_states.json"
ATOMIC_NUMBER = {"H": 1, "O": 8}

# Geometries (O is atom 0, the stretched O-H is atom 1).
GEOMETRY = {
    "oh-":  (["O", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.97]]),
    "h2o":  (["O", "H", "H"], [[0, 0, 0.12], [0, 0.76, -0.47], [0, -0.76, -0.47]]),
    "h3o+": (["O", "H", "H", "H"],
             [[0, 0, 0.07], [0, 0.94, -0.20], [0.82, -0.47, -0.20], [-0.82, -0.47, -0.20]]),
}


@pytest.fixture(scope="module")
def library():
    return DiabaticStateLibrary.from_yaml(LIBRARY)


@pytest.fixture(scope="module")
def env():
    return EnvelopeConfig(
        bound_envelope=dict(hi1=1.6, hi0=3.0),
        channel_envelope=dict(lo0=1.3, lo1=2.6),
        switch_r_on=1.6, switch_r_off=4.5, beta=0.0, tau=1.0,
    )


@pytest.fixture(scope="module")
def models():
    """A small randomized monomer stack and its mixture wrapper (weight-agnostic tests)."""
    # Module-scoped: built before the function-scoped float64 autouse fixture runs, so set the
    # dtype here too or the params come out float32 while the test inputs are float64.
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    features = types.SimpleNamespace(
        cutoff=5.0, n_max=3, l_max=2, selected_lambdas=(0, 1, 2), backend="e3nn"
    )
    monomer_cfg = types.SimpleNamespace(
        emb_dim=12, weight_channels=4, hidden=32, depth=1, equiv_channels=8
    )
    sqe = types.SimpleNamespace(
        s_init=0.5, s_floor=0.0, n_radial=6, eta_init=0.5, eta_floor=0.05, psd_floor=1e-4
    )
    states = AtomicStateReference.from_json(STATES, [1, 8])
    monomer = build_monomer_model(
        features, monomer_cfg, sqe, [1, 8],
        torch.tensor([-0.5013141188705883, -75.00924018194253]), states,
    )
    g = torch.Generator().manual_seed(2)
    with torch.no_grad():
        for p in monomer.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    monomer.eval()
    mix = MixtureModel(monomer)
    mix.eval()
    return monomer, mix


def one_system(symbols, positions):
    n = len(symbols)
    return Batch(
        positions=torch.tensor(np.asarray(positions, dtype=float)),
        atomic_numbers=torch.tensor([ATOMIC_NUMBER[s] for s in symbols], dtype=torch.long),
        batch_idx=torch.zeros(n, dtype=torch.long), n_systems=1,
        energy=torch.zeros(1), forces=torch.zeros(n, 3),
    )


def mono_call(monomer, library, symbols, positions, config_type):
    """Standalone MonomerModel prediction for one (sub-)fragment, for exact-limit comparison."""
    b = one_system(symbols, positions)
    a = assign_from_headers(library, symbols, config_type=config_type)
    b.fragment_idx = torch.tensor(a.fragment_idx)
    b.fragment_charge = torch.tensor(a.fragment_charge, dtype=torch.float64)
    b.fragment_two_s = torch.tensor(a.fragment_two_s, dtype=torch.float64)
    b.n_fragments = 1
    b.bond_index = torch.tensor(a.bond_index)
    b.bond_batch = torch.zeros(a.bond_index.shape[1], dtype=torch.long)
    return monomer(b)


def stretched(config_type, r):
    symbols, pos0 = GEOMETRY[config_type]
    pos0 = np.array(pos0, dtype=float)
    unit = (pos0[1] - pos0[0]) / np.linalg.norm(pos0[1] - pos0[0])
    pos = pos0.copy()
    pos[1] = pos0[0] + r * unit
    return symbols, pos


# ---------------------------------------------------------------------------
# Channel-graph classification
# ---------------------------------------------------------------------------

def test_finest_refinement_marks_the_broken_bond_inter(library):
    """The stretched O-H is inter-fragment on the finest refinement; the spectators are intra."""
    das = enumerate_diabats(library, ["O", "H", "H", "H"], config_type="h3o+")
    bond_index, is_inter = mixture_channel_graph(das)
    inter = {frozenset((int(bond_index[0, e]), int(bond_index[1, e])))
             for e in range(bond_index.shape[1]) if is_inter[e]}
    assert inter == {frozenset((0, 1))}                       # only O0-H1 is broken
    assert is_inter.sum() == 1 and (~is_inter).sum() == 2     # two spectator O-H stay intra


# ---------------------------------------------------------------------------
# Equilibrium recovery
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("config_type", ["oh-", "h2o", "h3o+"])
def test_equilibrium_recovers_the_monomer_stack(models, library, env, config_type):
    """At a sampled geometry (Ω_dissoc=0, switch=1) the mixture *is* the monomer stack."""
    monomer, mix = models
    symbols, pos = stretched(config_type, 0.98)
    das = enumerate_diabats(library, symbols, config_type=config_type)
    out = mix(one_system(symbols, pos), das, env)
    ref = mono_call(monomer, library, symbols, pos, config_type)
    assert float(out.weights[0]) == pytest.approx(1.0, abs=1e-13)     # bound diabat only
    assert torch.allclose(out.energy, ref.energy[0], atol=1e-11)
    assert torch.allclose(out.charges, ref.charges, atol=1e-11)


# ---------------------------------------------------------------------------
# Dissociation limits (exact = sum of standalone fragment predictions)
# ---------------------------------------------------------------------------

def test_oh_minus_collapses_to_atomic_references(models, library, env):
    """OH- -> O-(atom) + H(atom): integer charges and E = E_mono(O-) + E_mono(H) exactly."""
    monomer, mix = models
    symbols, pos = stretched("oh-", 12.0)
    das = enumerate_diabats(library, symbols, config_type="oh-")
    out = mix(one_system(symbols, pos), das, env)
    assert float(out.charges[0]) == pytest.approx(-1.0, abs=1e-10)
    assert float(out.charges[1]) == pytest.approx(0.0, abs=1e-10)
    e_o = mono_call(monomer, library, ["O"], [[0, 0, 0]], "o-").energy[0]
    e_h = mono_call(monomer, library, ["H"], [[0, 0, 0]], "h").energy[0]
    assert torch.allclose(out.energy, e_o + e_h, atol=1e-10)


@pytest.mark.parametrize(
    "config_type, remainder, remainder_key, remainder_charge",
    [("h3o+", [0, 2, 3], "h2o", 0.0), ("h2o", [0, 2], "oh-", -1.0)],
)
def test_proton_loss_collapses_to_monomer_plus_proton(
    models, library, env, config_type, remainder, remainder_key, remainder_charge
):
    """H3O+/H2O -> (trained monomer) + H+: leaving proton is exactly +1, energy is exact."""
    monomer, mix = models
    symbols, pos = stretched(config_type, 12.0)
    das = enumerate_diabats(library, symbols, config_type=config_type)
    out = mix(one_system(symbols, pos), das, env)
    assert float(out.charges[1]) == pytest.approx(1.0, abs=1e-9)                     # leaving proton
    assert float(out.charges[remainder].sum()) == pytest.approx(remainder_charge, abs=1e-9)
    rem_syms = [symbols[i] for i in remainder]
    e_rem = mono_call(monomer, library, rem_syms, pos[remainder], remainder_key).energy[0]
    e_hp = mono_call(monomer, library, ["H"], [[0, 0, 0]], "h+").energy[0]
    assert torch.allclose(out.energy, e_rem + e_hp, atol=1e-9)


# ---------------------------------------------------------------------------
# Normalization, conservation, continuity, symmetry (across the scan)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("config_type", ["oh-", "h2o", "h3o+"])
def test_weights_normalized_and_charge_conserved(models, library, env, config_type):
    """Σc_K = 1 (so total charge stays exactly Q) at every separation."""
    _, mix = models
    symbols = GEOMETRY[config_type][0]
    q_total = float(library[config_type].charge)
    for r in np.linspace(0.9, 6.0, 12):
        symbols, pos = stretched(config_type, r)
        das = enumerate_diabats(library, symbols, config_type=config_type)
        out = mix(one_system(symbols, pos), das, env)
        assert float(out.weights.sum()) == pytest.approx(1.0, abs=1e-12)
        assert float(out.charges.sum()) == pytest.approx(q_total, abs=1e-10)


def test_force_is_continuous_across_switch_and_envelope(models, library, env):
    """Analytic -dE/dx matches central finite difference across the crossover (§4.4.4).

    Sampled at radii straddling the envelope onset, the crossover, the envelope close, and the
    switch-off radius -- the boundaries where a naive gate/switch would spike the force.
    """
    _, mix = models
    symbols, _ = GEOMETRY["oh-"]
    das = enumerate_diabats(library, symbols, config_type="oh-")
    unit = np.array([0.0, 0.0, 1.0])
    step = 1e-6
    for r in (1.35, 2.2, 3.0, 4.5):
        pos = np.array([[0.0, 0, 0], r * unit])
        b = one_system(symbols, pos)
        b.positions.requires_grad_(True)
        out = mix(b, das, env)
        (grad,) = torch.autograd.grad(out.energy, b.positions)
        f_h_analytic = -float(grad[1, 2])

        e_plus = float(mix(one_system(symbols, pos + np.array([[0, 0, 0], step * unit])), das, env).energy)
        e_minus = float(mix(one_system(symbols, pos - np.array([[0, 0, 0], step * unit])), das, env).energy)
        f_h_numeric = -(e_plus - e_minus) / (2 * step)
        assert abs(f_h_analytic - f_h_numeric) < 1e-6, f"force discontinuity near r={r}"


def test_rotation_equivariance(models, library, env):
    """Energy/charges invariant; dipole/α equivariant under a global rotation."""
    from e3nn import o3

    _, mix = models
    symbols, pos = stretched("h2o", 2.0)                      # in the crossover, both diabats live
    das = enumerate_diabats(library, symbols, config_type="h2o")
    out = mix(one_system(symbols, pos), das, env)

    rot = o3.rand_matrix().to(torch.float64)
    out_rot = mix(one_system(symbols, pos @ rot.T.numpy()), das, env)
    assert torch.allclose(out_rot.energy, out.energy, atol=1e-10)
    assert torch.allclose(out_rot.charges, out.charges, atol=1e-10)
    assert torch.allclose(out_rot.dipole, out.dipole @ rot.T, atol=1e-10)
    assert torch.allclose(out_rot.alpha, rot @ out.alpha @ rot.T, atol=1e-9)
