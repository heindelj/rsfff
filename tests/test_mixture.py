"""Latent-space diabatic adiabaticization: the architectural properties that hold by construction.

The model blends the active diabats' *feature bundles* into one adiabatic feature, adds a
nonlinear correction that is exactly zero at every pure-state vertex, and decodes once through the
frozen monomer heads (docs/mixture_of_diabatic_embeddings.md, Revision 3). None of that is
trained -- it follows from the ``4c_Kc_J`` prefactor, the geometric overlap shutdown, the
envelope-gated proxy coefficients, and the C² compliance switch -- so these tests run on a small
randomly-initialized model, not the checkpoint.

The nine systems the design targets appear here: the bare atoms H, H⁺, O, O⁻, O⁺ (single-diabat
vertices), the radicals/ions OH, OH⁻ (one dissociation channel each), H3O⁺ (one channel), and H2O
(the flagship two-channel homolytic-vs-heterolytic active space).
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
    "oh_q0_m2":   (["O", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.97]]),
    "oh_q-1_m1":  (["O", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.97]]),
    "h2o_q0_m1":  (["O", "H", "H"], [[0, 0, 0.12], [0, 0.76, -0.47], [0, -0.76, -0.47]]),
    "h3o_q+1_m1": (["O", "H", "H", "H"],
             [[0, 0, 0.07], [0, 0.94, -0.20], [0.82, -0.47, -0.20], [-0.82, -0.47, -0.20]]),
    "o2_q-1_m2":  (["O", "O"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.35]]),
}


@pytest.fixture(scope="module")
def library():
    return DiabaticStateLibrary.from_yaml(LIBRARY)


@pytest.fixture(scope="module")
def env():
    return EnvelopeConfig(
        bound_envelope=dict(hi1=1.6, hi0=3.0),
        channel_envelope=dict(lo0=1.3, lo1=2.6),
        switch_r_on=1.6, switch_r_off=4.5,
        overlap_r_on=1.6, overlap_r_off=4.5, tau=0.05,
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
    # Default small-init correction: nonzero through the crossover (‖Δz‖ ~ 1e-2) yet gentle
    # enough that the untrained decoder stays smooth (a large random Ψ can push the PSD-clamped
    # α head into a kink). The vertex identity is enforced by 4c_Kc_J regardless of the scale.
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
    das = enumerate_diabats(library, ["O", "H", "H", "H"], config_type="h3o_q+1_m1")
    bond_index, is_inter = mixture_channel_graph(das)
    inter = {frozenset((int(bond_index[0, e]), int(bond_index[1, e])))
             for e in range(bond_index.shape[1]) if is_inter[e]}
    assert inter == {frozenset((0, 1))}                       # only O0-H1 is broken
    assert is_inter.sum() == 1 and (~is_inter).sum() == 2     # two spectator O-H stay intra


def test_water_is_a_three_diabat_active_space(library):
    """H2O enumerates bound + homolytic (OH+H) + heterolytic (OH-+H+); all break the same bond."""
    das = enumerate_diabats(library, ["O", "H", "H"], config_type="h2o_q0_m1")
    assert len(das) == 3
    keys = [tuple(s.key for s, _ in a.fragments) for a in das]
    assert keys == [("h2o_q0_m1",), ("oh_q0_m2", "h_q0_m2"), ("oh_q-1_m1", "h_q+1_m1")]
    # both dissociated diabats refine to the same {O,H2}|{H1} partition
    _, is_inter = mixture_channel_graph(das)
    assert is_inter.sum() == 1


# ---------------------------------------------------------------------------
# Pure-state vertex identity (§7.4, §13.2): the correction vanishes and the shared decoder
# reproduces the monomer stack whenever a single diabat is active.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "config_type, symbols", [("h_q0_m2", ["H"]), ("h_q+1_m1", ["H"]), ("o_q0_m3", ["O"]), ("o_q-1_m2", ["O"]),
                             ("o_q+1_m4", ["O"])]
)
def test_bare_atom_is_a_pure_state_vertex(models, library, env, config_type, symbols):
    """A single-diabat system: c=1, no correction, and the mixture *is* the monomer stack."""
    monomer, mix = models
    out = mix(one_system(symbols, [[0.0, 0.0, 0.0]]),
              enumerate_diabats(library, symbols, config_type=config_type), env)
    ref = mono_call(monomer, library, symbols, [[0.0, 0.0, 0.0]], config_type)
    assert float(out.weights[0]) == pytest.approx(1.0, abs=1e-13)
    assert float(out.correction_norm) == 0.0                         # exactly zero, not ~0
    assert torch.allclose(out.energy, ref.energy[0], atol=1e-12)
    assert torch.allclose(out.charges, ref.charges, atol=1e-12)


@pytest.mark.parametrize("config_type", ["oh_q0_m2", "oh_q-1_m1", "h2o_q0_m1", "h3o_q+1_m1"])
def test_equilibrium_recovers_the_monomer_stack(models, library, env, config_type):
    """At a sampled geometry (Ω_dissoc=0, switch=1) the mixture *is* the monomer stack.

    Only the bound diabat is valid, so c_bound=1 is a vertex: μ is its feature, Δz is exactly 0,
    and the single shared decode reproduces MonomerModel to machine precision.
    """
    monomer, mix = models
    symbols, pos = stretched(config_type, 0.98)
    das = enumerate_diabats(library, symbols, config_type=config_type)
    out = mix(one_system(symbols, pos), das, env)
    ref = mono_call(monomer, library, symbols, pos, config_type)
    assert float(out.weights[0]) == pytest.approx(1.0, abs=1e-13)     # bound diabat only
    assert float(out.correction_norm) == 0.0
    assert torch.allclose(out.energy, ref.energy[0], atol=1e-11)
    assert torch.allclose(out.charges, ref.charges, atol=1e-11)


def test_correction_is_active_in_the_crossover_and_zero_at_both_ends(models, library, env):
    """‖Δz‖ is exactly 0 at every vertex and strictly positive where two diabats genuinely mix."""
    _, mix = models
    das = enumerate_diabats(library, ["O", "H"], config_type="oh_q-1_m1")
    norms = {}
    for r in (0.98, 2.0, 12.0):
        symbols, pos = stretched("oh_q-1_m1", r)
        norms[r] = float(mix(one_system(symbols, pos), das, env).correction_norm)
    assert norms[0.98] == 0.0                                        # bound vertex (Ω_dissoc=0)
    assert norms[12.0] == 0.0                                        # dissociated vertex + 𝒪=0
    assert norms[2.0] > 1e-4                                         # genuine two-state mixture


# ---------------------------------------------------------------------------
# Dissociation limits (exact for the single-channel systems)
# ---------------------------------------------------------------------------

def test_oh_minus_collapses_to_the_covalent_reference(models, library, env):
    """OH- pulls apart to O- + H (the covalent cover), not O + H- (ionic).

    OH- now carries competing covalent/ionic covers (like water), so the asymptote is a
    *near*-vertex: the ionic cover keeps a small proxy weight, and the covalent O- charge is near
    -1, not an exact integer. The covalent cover wins because O- binds its electron here while H-
    does not (IP/EA-seeded self-energies).
    """
    _, mix = models
    symbols, pos = stretched("oh_q-1_m1", 12.0)
    das = enumerate_diabats(library, symbols, config_type="oh_q-1_m1")
    out = mix(one_system(symbols, pos), das, env)
    assert float(out.weights[0]) == pytest.approx(0.0, abs=1e-6)      # bound closed by the envelope
    assert float(out.weights[1]) > 0.9                               # covalent (O- + H) dominates
    assert float(out.weights[1]) > float(out.weights[2])             # ... over ionic (O + H-)
    assert float(out.charges[0]) == pytest.approx(-1.0, abs=0.1)      # ~ -1 on oxygen
    assert float(out.charges[1]) == pytest.approx(0.0, abs=0.1)       # ~ 0 on the leaving H
    assert float(out.charges.sum()) == pytest.approx(-1.0, abs=1e-10)
    assert float(out.correction_norm) < 1e-3                          # overlap 𝒪 has shut the mix off


def test_oh_radical_collapses_to_neutral_atoms(models, library, env):
    """OH -> O + H (covalent): both fragments near-neutral, a near-vertex like water's homolysis."""
    _, mix = models
    symbols, pos = stretched("oh_q0_m2", 12.0)
    das = enumerate_diabats(library, symbols, config_type="oh_q0_m2")
    out = mix(one_system(symbols, pos), das, env)
    assert float(out.weights[1]) > 0.9                               # covalent (O + H) dominates
    assert float(out.charges[0]) == pytest.approx(0.0, abs=0.05)
    assert float(out.charges[1]) == pytest.approx(0.0, abs=0.05)
    assert float(out.charges.sum()) == pytest.approx(0.0, abs=1e-10)
    assert float(out.correction_norm) < 1e-3


def test_superoxide_collapses_to_atomic_references(models, library, env):
    """O2- -> O-(atom) + O(atom): a single-channel diatomic, so an *exact* vertex at large r."""
    monomer, mix = models
    symbols, pos = stretched("o2_q-1_m2", 12.0)
    das = enumerate_diabats(library, symbols, config_type="o2_q-1_m2")
    assert len(das) == 2                                             # bound + one dissociation cover
    out = mix(one_system(symbols, pos), das, env)
    assert float(out.weights[1]) == pytest.approx(1.0, abs=1e-9)     # single channel -> exact vertex
    assert float(out.charges[0]) == pytest.approx(-1.0, abs=1e-9)
    assert float(out.charges[1]) == pytest.approx(0.0, abs=1e-9)
    e_om = mono_call(monomer, library, ["O"], [[0, 0, 0]], "o_q-1_m2").energy[0]
    e_o = mono_call(monomer, library, ["O"], [[0, 0, 0]], "o_q0_m3").energy[0]
    assert torch.allclose(out.energy, e_om + e_o, atol=1e-9)


def test_hydronium_proton_loss_collapses_to_water_plus_proton(models, library, env):
    """H3O+ -> H2O + H+: the leaving proton is exactly +1, energy is exact (single channel)."""
    monomer, mix = models
    symbols, pos = stretched("h3o_q+1_m1", 12.0)
    das = enumerate_diabats(library, symbols, config_type="h3o_q+1_m1")
    out = mix(one_system(symbols, pos), das, env)
    assert float(out.charges[1]) == pytest.approx(1.0, abs=1e-9)
    assert float(out.charges[[0, 2, 3]].sum()) == pytest.approx(0.0, abs=1e-9)
    e_rem = mono_call(monomer, library, ["O", "H", "H"], pos[[0, 2, 3]], "h2o_q0_m1").energy[0]
    e_hp = mono_call(monomer, library, ["H"], [[0, 0, 0]], "h_q+1_m1").energy[0]
    assert torch.allclose(out.energy, e_rem + e_hp, atol=1e-9)


def test_water_dissociates_homolytically(models, library, env):
    """H2O pulls apart to OH + H, not OH- + H+ (gas-phase): the proxy picks the neutral channel.

    With two dissociation channels sharing an envelope the asymptote is a *near*-vertex -- the
    ionic cover keeps an exponentially small proxy weight -- so this checks the neutral channel
    dominates and the leaving H is near-neutral, not that the charge is an exact integer.
    """
    _, mix = models
    symbols, pos = stretched("h2o_q0_m1", 12.0)
    das = enumerate_diabats(library, symbols, config_type="h2o_q0_m1")
    out = mix(one_system(symbols, pos), das, env)
    c = out.weights
    assert float(c[0]) == pytest.approx(0.0, abs=1e-6)               # bound closed by the envelope
    assert float(c[1]) > 0.98                                        # homolytic (OH+H) dominates
    assert float(c[1]) > float(c[2])                                 # ... over heterolytic (OH-+H+)
    assert float(out.charges[1]) < 0.05                              # leaving H nearly neutral
    assert float(out.correction_norm) < 1e-3                         # overlap 𝒪 has shut the mix off


# ---------------------------------------------------------------------------
# Normalization, conservation, continuity, symmetry (across the scan)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("config_type", ["oh_q0_m2", "oh_q-1_m1", "h2o_q0_m1", "h3o_q+1_m1"])
def test_weights_normalized_and_charge_conserved(models, library, env, config_type):
    """Σc_K = 1 (so total charge stays exactly Q) at every separation."""
    _, mix = models
    q_total = float(library[config_type].charge)
    for r in np.linspace(0.9, 6.0, 12):
        symbols, pos = stretched(config_type, r)
        das = enumerate_diabats(library, symbols, config_type=config_type)
        out = mix(one_system(symbols, pos), das, env)
        assert float(out.weights.sum()) == pytest.approx(1.0, abs=1e-12)
        assert float(out.charges.sum()) == pytest.approx(q_total, abs=1e-10)


@pytest.mark.parametrize("config_type", ["oh_q-1_m1", "h2o_q0_m1"])
def test_force_is_continuous_across_switch_envelope_and_overlap(models, library, env, config_type):
    """Analytic -dE/dx matches central finite difference across the crossover (§4.4.4, §13.3).

    Sampled at radii straddling the envelope onset, the crossover (where Δz peaks), the envelope
    close, and the switch-off radius -- every boundary where a naive gate/switch/overlap would
    spike the force. H2O adds the three-diabat homolytic/heterolytic crossover.
    """
    _, mix = models
    symbols, pos0 = GEOMETRY[config_type]
    das = enumerate_diabats(library, symbols, config_type=config_type)
    pos0 = np.array(pos0, dtype=float)
    unit = (pos0[1] - pos0[0]) / np.linalg.norm(pos0[1] - pos0[0])
    unit_t = torch.tensor(unit)
    step = 1e-6
    for r in (1.35, 1.7, 2.2, 3.0, 4.5):
        pos = pos0.copy()
        pos[1] = pos0[0] + r * unit
        b = one_system(symbols, pos)
        b.positions.requires_grad_(True)
        out = mix(b, das, env)
        (grad,) = torch.autograd.grad(out.energy, b.positions)
        f_h_analytic = -float(grad[1] @ unit_t)          # force on the leaving H along the bond

        dp = np.zeros_like(pos)
        dp[1] = step * unit
        e_plus = float(mix(one_system(symbols, pos + dp), das, env).energy)
        e_minus = float(mix(one_system(symbols, pos - dp), das, env).energy)
        f_h_numeric = -(e_plus - e_minus) / (2 * step)
        assert abs(f_h_analytic - f_h_numeric) < 1e-6, f"force discontinuity near r={r}"


def test_rotation_equivariance(models, library, env):
    """Energy/charges invariant; dipole/α equivariant under a global rotation (in the crossover)."""
    from e3nn import o3

    _, mix = models
    symbols, pos = stretched("h2o_q0_m1", 2.0)                      # in the crossover, all diabats live
    das = enumerate_diabats(library, symbols, config_type="h2o_q0_m1")
    out = mix(one_system(symbols, pos), das, env)

    rot = o3.rand_matrix().to(torch.float64)
    out_rot = mix(one_system(symbols, pos @ rot.T.numpy()), das, env)
    assert torch.allclose(out_rot.energy, out.energy, atol=1e-10)
    assert torch.allclose(out_rot.charges, out.charges, atol=1e-10)
    assert torch.allclose(out_rot.dipole, out.dipole @ rot.T, atol=1e-10)
    assert torch.allclose(out_rot.alpha, rot @ out.alpha @ rot.T, atol=1e-9)
