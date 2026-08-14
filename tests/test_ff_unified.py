"""The unified pair model: one list, learned range separation, routing that is not physics.

The load-bearing test here is :func:`test_init_matches_existing_terms`. It pins the unified
model against the per-term modules it replaces by giving both the *same* parameter heads and
forcing the range separation into the per-term modules' single-global-``r0`` form. That is a
wiring test, not an accuracy test -- it says the union pair list, the per-channel tapers, the
extracted kernels and the routing add up to exactly the energies the validated modules
produce, so any later difference is attributable to the physics that changed and not to a
transcription error.

The rest of the file guards the properties that must survive the change (exact one-body,
exactly-two-body electrostatics, charge conservation, forces) and pins the two behaviours that
are genuinely new: a same-fragment pair at range now gets real electrostatics, and relabelling
a dimer as one fragment moves energy between the decomposition's buckets without changing the
total.
"""

import numpy as np
import pytest
import torch

from rsfff.features.features import DensityExpansion, FlatLambdaSOAPFeaturizer
from rsfff.ff.dispersion import (
    DispersionParameterHeads,
    TTDispersion,
    build_log_priors,
)
from rsfff.ff.electrostatics import SlaterElectrostatics
from rsfff.ff.many_body import mbe_decompose
from rsfff.ff.multipole import irrep2_to_spherical
from rsfff.ff.pairs import union_pairs
from rsfff.ff.pauli import PauliMultipoleHeads, SlaterPauli, build_pauli_priors
from rsfff.ff.range_priors import RANGE_CHANNELS, build_range_priors
from rsfff.ff.response import (
    ElectrostaticParameterHeads,
    FragmentResponse,
    build_elec_priors,
)
from rsfff.ff.unified import (
    ClassicalSpec,
    EnvironmentResidual,
    FragmentStateEmbedding,
    RangeSeparationHeads,
    UnifiedPairModel,
)
from rsfff.ff.units import KJMOL_PER_HARTREE
from rsfff.mlip.sqe import PairComplianceHead
from rsfff.mlip.unified_head import ChannelSpec, UnifiedPairHead
from rsfff.train.data import Batch

torch.set_default_dtype(torch.float64)

NEIGHBOR_TYPES = [1, 8]
E0 = torch.tensor([-0.4941110651, -75.0780656005])   # H, O at wB97M-V/def2-TZVPD
MAX_RANK = 2


def call_pair_head(model, feats, pair_index, r, which=0):
    """``UnifiedPairHead`` takes features already gathered onto pairs; do that here.

    ``which`` selects the energies (0) or the per-pair log-r0 deviations (1).
    """
    i, j = pair_index[0], pair_index[1]
    return model.pair_head(
        feats.inv_feats[i], feats.inv_feats[j],
        feats.species_idx[i], feats.species_idx[j], r,
    )[which]

_WATER = np.array([[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469]])


def water_cluster(n, spacing=3.0, jitter=0.05, seed=0):
    rng = np.random.default_rng(seed)
    blocks = [
        _WATER + np.array([k * spacing, 0.0, 0.0]) + rng.normal(scale=jitter, size=(3, 3))
        for k in range(n)
    ]
    positions = torch.tensor(np.concatenate(blocks))
    return positions, torch.tensor([8, 1, 1] * n), torch.arange(n).repeat_interleave(3)


def make_batch(positions, numbers, fragment_idx, batch_idx=None):
    n_frag = int(fragment_idx.max()) + 1
    if batch_idx is None:
        batch_idx = torch.zeros(positions.shape[0], dtype=torch.long)
    n_sys = int(batch_idx.max()) + 1
    f2b = batch_idx.new_zeros(n_frag).scatter_(0, fragment_idx, batch_idx)
    return Batch(
        positions=positions.clone(),
        atomic_numbers=numbers,
        batch_idx=batch_idx,
        n_systems=n_sys,
        energy=torch.zeros(n_sys),
        fragment_idx=fragment_idx,
        fragment_charge=torch.zeros(n_frag),
        fragment_two_s=torch.zeros(n_frag),
        fragment_to_batch=f2b,
        n_fragments=n_frag,
    )


def build_parts(*, fragment_state_dim=0, alpha_init=40.0, seed=0):
    """Every submodule, so a test can hand the same instances to both model shapes."""
    torch.manual_seed(seed)
    featurizer = FlatLambdaSOAPFeaturizer(
        cutoff=5.0, n_max=4, l_max=3, neighbor_types=NEIGHBOR_TYPES,
        selected_lambdas=(0, 1, 2), backend="e3nn", density_channels=8,
    )
    n_species = len(NEIGHBOR_TYPES)
    p1, p2 = featurizer.feature_dims.get(1), featurizer.feature_dims.get(2)
    fragment_state = FragmentStateEmbedding(fragment_state_dim)
    p0 = featurizer.feature_dims[0] + fragment_state.dim
    voigt = featurizer.backend.irrep6_to_voigt()

    log_z, log_b, q0 = build_elec_priors(NEIGHBOR_TYPES)
    response = FragmentResponse(
        ElectrostaticParameterHeads(
            p0, p1, p2, n_species,
            log_z_prior=log_z, log_b_prior=log_b, q0_prior=q0,
            irrep6_to_voigt=voigt, irrep2_to_spherical_map=irrep2_to_spherical(voigt),
            emb_dim=8, hidden=24, depth=1, equiv_channels=6, max_rank=MAX_RANK,
        ),
        PairComplianceHead(p0, hidden=24, depth=1, cutoff=5.0, s_init=0.5),
    )
    log_c6, log_bd = build_log_priors(NEIGHBOR_TYPES, b_prior="per_element")
    disp_params = DispersionParameterHeads(
        p0, n_species, log_c6_prior=log_c6, log_b_prior=log_bd,
        emb_dim=8, hidden=24, depth=1,
    )
    log_q, log_bp, mu_scale, quad_scale = build_pauli_priors(NEIGHBOR_TYPES)
    pauli_params = PauliMultipoleHeads(
        p0, p1, n_species,
        log_q_prior=log_q, log_b_prior=log_bp, dipole_scale=mu_scale,
        p2=p2, quad_scale=quad_scale, irrep2_to_spherical=irrep2_to_spherical(voigt),
        emb_dim=8, hidden=24, depth=1, equiv_channels=6, max_rank=MAX_RANK,
    )
    range_heads = RangeSeparationHeads(
        p0, n_species, log_r0_prior=build_range_priors(NEIGHBOR_TYPES),
        alpha_init=alpha_init, emb_dim=8, hidden=24, depth=1,
    )
    corr = dict(r_on=4.0, r_off=5.0)
    pair_head = UnifiedPairHead(
        p0, n_species,
        {
            "elst": ChannelSpec(**corr, energy_scale=3e-3),
            "pauli": ChannelSpec(**corr, energy_scale=3e-3),
            "disp": ChannelSpec(**corr, energy_scale=1e-3),
            "bond": ChannelSpec(r_on=2.5, r_off=4.0, energy_scale=0.2),
        },
        range_channels=RANGE_CHANNELS,
        emb_dim=8, hidden=24, depth=1,
    )
    return dict(
        featurizer=featurizer, response=response, disp_params=disp_params,
        pauli_params=pauli_params, range_heads=range_heads, pair_head=pair_head,
        fragment_state=fragment_state,
    )


def make_model(parts=None, *, environment=False, **kw):
    p = parts or build_parts(**kw)
    del kw
    env = None
    if environment:
        f = p["featurizer"]
        env = EnvironmentResidual(
            f.feature_dims[0] + p["fragment_state"].dim,
            f.feature_dims.get(1), f.feature_dims.get(2), len(NEIGHBOR_TYPES),
            emb_dim=8, hidden=24, depth=1,
        )
    return UnifiedPairModel(
        p["featurizer"], p["response"], p["disp_params"], p["pauli_params"],
        p["range_heads"], p["pair_head"], p["fragment_state"], E0,
        environment=env, max_rank=MAX_RANK,
    )


def wake_environment(model, scale=0.3, seed=101):
    """Move ``g`` off its zero initialization so the environment path is actually live."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for m in (model.environment.inv_mlp, model.environment.vec_gate,
                  model.environment.equiv_gate):
            if m is not None:
                m[-1].weight.add_(scale * torch.randn(m[-1].weight.shape, generator=g))
                m[-1].bias.add_(scale * torch.randn(m[-1].bias.shape, generator=g))
    return model


def randomize(model, scale=0.05, seed=1):
    """Move every zero-initialized readout off zero so couplings are actually exercised."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(scale * torch.randn(p.shape, generator=g))
    return model


def force_constant_r0(model, value, channels=RANGE_CHANNELS):
    """Collapse the per-atom, per-channel ``r0`` to one global scalar, as the per-term
    modules have. Only then are the two gatings comparable."""
    with torch.no_grad():
        model.range_heads.log_r0_prior.fill_(float(np.log(value)))
        for name in channels:
            model.range_heads.d_log_r0[name].zero_()


# ---------------------------------------------------------------------------
# The reproduction assertion
# ---------------------------------------------------------------------------

def test_init_matches_existing_terms():
    """Same parameters, same gating -> bit-for-bit the per-term modules' backbone energies.

    ``r0`` is collapsed to a global scalar per channel and ``alpha`` set to each per-term
    module's own value, which is the only configuration in which the two range separations
    describe the same function. Pauli has no Fermi switch at all in ``SlaterPauli``, so its
    unified counterpart is driven to a gate of 1 by putting ``r0`` far below every distance
    in the system.
    """
    parts = build_parts()
    model = randomize(make_model(parts))
    positions, numbers, frag = water_cluster(4, seed=3)
    batch = make_batch(positions, numbers, frag)

    elst_r0, elst_alpha = 1.5, 8.0
    disp_r0, disp_alpha = 2.0, 8.0
    with torch.no_grad():
        # The per-term modules have one global scalar r0, so the per-pair correction has to
        # be off for the two gatings to describe the same function at all.
        for lin in model.pair_head.range_readout.values():
            lin.weight.zero_()
            lin.bias.zero_()
        model.range_heads.d_log_r0["elst"].zero_()
        model.range_heads.d_log_r0["pauli"].zero_()
        model.range_heads.d_log_r0["disp"].zero_()
        # One prior table is shared, so set each channel's offset relative to it.
        model.range_heads.log_r0_prior.zero_()          # r0_i = exp(d_log_r0)
        model.range_heads.d_log_r0["elst"].fill_(float(np.log(elst_r0)))
        model.range_heads.d_log_r0["disp"].fill_(float(np.log(disp_r0)))
        model.range_heads.d_log_r0["pauli"].fill_(float(np.log(1e-4)))   # gate == 1
        for name, a in (("elst", elst_alpha), ("pauli", 40.0), ("disp", disp_alpha)):
            model.range_heads.alpha_raw[name].fill_(
                float(torch.log(torch.expm1(torch.tensor(a))))
            )

    out = model(batch)

    # The per-term modules, sharing the very same parameter head instances.
    elec = SlaterElectrostatics(
        parts["response"], None, cutoff=12.0, taper_width=1.0,
        r0_init=elst_r0, alpha=elst_alpha, max_rank=MAX_RANK,
    )
    pauli = SlaterPauli(parts["pauli_params"], None, cutoff=7.0, taper_width=1.0,
                        max_rank=MAX_RANK, inter_only=True)
    disp = TTDispersion(parts["disp_params"], None, cutoff=10.0, taper_width=1.0,
                        r0_init=disp_r0, alpha=disp_alpha, inter_only=True)

    feats = parts["featurizer"](batch, batch.fragment_idx)
    ref = {
        "elst": elec(batch, feats).energy_ff,
        "pauli": pauli(batch, feats).energy_ff,
        "disp": disp(batch, feats).energy_ff,
    }
    for name, want in ref.items():
        got = out.interaction_ff[name]
        assert torch.allclose(got, want, atol=1e-12, rtol=0), (
            f"{name}: unified {got.tolist()} vs per-term {want.tolist()} "
            f"(diff {(got - want).abs().max():.3e} Ha)"
        )


def test_intra_classical_leak_is_small_at_init():
    """The r0 priors keep the covalent region out of the classical forms.

    This is what stops training diverging on step one: at 0.96 Angstrom the Slater Pauli form
    between O and H is ~1000 kJ/mol, and the union pair list no longer masks it away.
    """
    model = make_model()
    positions, numbers, frag = water_cluster(3, seed=1)
    out = model(make_batch(positions, numbers, frag))

    i, j = out.pair_index
    z = numbers
    oh = ((z[i] == 8) & (z[j] == 1)) | ((z[i] == 1) & (z[j] == 8))
    intra_oh = out.is_intra & oh
    assert bool(intra_oh.any())
    # Covalent O-H must be switched off hard, and the residual energy kept under a kJ/mol.
    for name in ("pauli", "elst"):
        assert float(out.gate[name][intra_oh].detach().max()) < 1e-2, name
        leak = (out.e_pair_ff[name][intra_oh].detach().abs().max()) * KJMOL_PER_HARTREE
        assert float(leak) < 1.0, f"{name} leaks {float(leak):.3f} kJ/mol into a covalent O-H"

    # Intramolecular H-H is switched off too. An earlier prior left it fully on, on the
    # grounds that intra/inter H-H "overlap" -- true of the min/max over 225k pairs, false of
    # the distributions (0.01% contamination). See rsfff.ff.range_priors.
    #
    # Asserted in aggregate rather than per pair: the intra H-H distribution has a tail that
    # reaches into the switch (its 99.9th percentile is 1.679 against r0 1.75), so a bound on
    # the worst single gate is a bound on the tail, not on the physics. What matters is the
    # total classical energy this routes into `fragment_energy`, which was ~165 kJ/mol per
    # trimer under the old prior.
    hh = (z[i] == 1) & (z[j] == 1)
    assert bool((out.is_intra & hh).any())
    total = sum(
        float(out.e_pair_ff[c][out.is_intra].detach().sum()) for c in out.e_pair_ff
    ) * KJMOL_PER_HARTREE
    assert abs(total) < 1.0, f"{total:+.2f} kJ/mol of classical energy leaking into a fragment"

    # ... while inter-fragment pairs stay fully on, which is what makes the H-H threshold a
    # real trade rather than a free one.
    inter = ~out.is_intra
    assert float(out.gate["elst"][inter & (out.r < 6.0)].detach().min()) > 0.98


def test_accounting_identity_no_double_count_no_gap():
    """Every pair appears exactly once, in exactly one bucket.

    This is the invariant that has to survive when ``is_intra`` softens into a mixture
    weight, so it is worth pinning at the pair level rather than just checking the
    per-fragment sum.
    """
    model = randomize(make_model())
    positions, numbers, frag = water_cluster(3, seed=5)
    out = model(make_batch(positions, numbers, frag))

    assert torch.allclose(
        out.fragment_energy,
        out.energy_ref + out.energy_internal + out.energy_bond,
        atol=0, rtol=0,
    )
    # Every channel is structurally zero wherever it does not belong, so the identity is
    # simply "classical everywhere + every correction that exists".
    total = (
        out.energy_ref.sum()
        + out.energy_internal.sum()
        + sum(v.sum() for v in out.e_pair_ff.values())
        + sum(v.sum() for v in out.e_pair_corr.values())
    )
    assert torch.allclose(out.energy.sum(), total, atol=1e-12), (
        f"accounting identity off by {float(out.energy.sum() - total):.3e} Ha"
    )
    # The partition: bond only on intra pairs, interaction corrections only on inter pairs.
    assert float(out.e_pair_corr["bond"][~out.is_intra].detach().abs().max()) == 0.0
    for name in ("elst", "pauli", "disp"):
        assert float(out.e_pair_corr[name][out.is_intra].detach().abs().max()) == 0.0, name
    assert float(out.e_pair_corr["bond"][out.is_intra].detach().abs().max()) > 0.0


# ---------------------------------------------------------------------------
# The two genuinely new behaviours
# ---------------------------------------------------------------------------

def test_same_fragment_pair_at_range_gets_electrostatics():
    """The capability the per-term stack does not have at all.

    ``OneBodyEnergy`` is ``E0 + E_internal + bond_head`` with the bond envelope dead at
    4 Angstrom, so two atoms of one fragment 6 Angstrom apart interact not at all. Here they
    are fully switched on and their energy is routed to ``fragment_energy``.
    """
    model = make_model()
    # One fragment: a water plus a far-away H, well past the bond head's 4 A envelope.
    positions = torch.tensor(_WATER.tolist() + [[6.0, 0.0, 0.0]])
    numbers = torch.tensor([8, 1, 1, 1])
    frag = torch.zeros(4, dtype=torch.long)
    batch = make_batch(positions, numbers, frag)
    batch.fragment_charge = torch.tensor([1.0])   # H3O+ composition, keep the solve sane

    out = model(batch)
    i, j = out.pair_index
    far = (out.r > 5.0) & out.is_intra
    assert bool(far.any()), "the far intra pair must be in the list"
    assert float(out.gate["elst"][far].min()) > 0.99, "and fully switched on"

    # The bond channel really is silent there, so the energy can only be classical.
    feats = model.featurizer(batch, frag)
    dE = call_pair_head(model, feats, out.pair_index, out.r)
    assert float(dE["bond"][far].abs().max()) == 0.0


def test_relabelling_moves_every_pair_into_the_fragment_bucket():
    """Relabelling a dimer as one fragment empties the interaction channels.

    And it genuinely changes the total, which is correct rather than a leak. Two reasons,
    both asserted here because it is easy to assume otherwise:

    * the bond channel has no inter-fragment counterpart, so the nine pairs that switch
      buckets *gain* an energy channel -- this is where a charge-transfer energy would
      surface when a pair transitions from inter to intra;
    * the descriptors are fragment-confined, so they are partition-dependent themselves.

    What must hold in both labellings is the accounting identity, checked above.
    """
    model = randomize(make_model())
    positions, numbers, frag = water_cluster(2, spacing=2.9, seed=7)
    two = model(make_batch(positions, numbers, frag))
    one = model(make_batch(positions, numbers, torch.zeros_like(frag)))

    assert float(sum(v.sum() for v in two.interaction.values()).abs()) > 1e-4
    assert float(sum(v.sum() for v in one.interaction.values()).abs()) == 0.0
    assert int(two.is_intra.sum()) == 6 and int(one.is_intra.sum()) == 15

    # The descriptor itself moves, so nothing downstream of it can be partition-invariant.
    h_two = model.featurizer(make_batch(positions, numbers, frag), frag).inv_feats
    zeros = torch.zeros_like(frag)
    h_one = model.featurizer(make_batch(positions, numbers, zeros), zeros).inv_feats
    assert float((h_two - h_one).abs().max()) > 1e-3

    # ... and the accounting identity survives in the relabelled system.
    every_pair = sum(
        (one.e_pair_ff[c] + one.e_pair_corr[c]).sum() for c in one.e_pair_ff
    )
    total = (
        one.energy_ref.sum() + one.energy_internal.sum()
        + every_pair + one.e_pair_corr["bond"][one.is_intra].sum()
    )
    assert torch.allclose(one.energy.sum(), total, atol=1e-12)


# ---------------------------------------------------------------------------
# Properties that must survive
# ---------------------------------------------------------------------------

def test_fragment_energy_is_exactly_one_body():
    """A fragment's energy inside a cluster equals that fragment alone, to round-off."""
    model = randomize(make_model())
    positions, numbers, frag = water_cluster(4, seed=11)
    cluster = model(make_batch(positions, numbers, frag))

    alone = model(make_batch(positions[:3], numbers[:3], frag[:3]))
    assert torch.allclose(
        cluster.fragment_energy[0], alone.fragment_energy[0], atol=1e-12, rtol=0
    )


def test_interaction_channels_are_exactly_two_body():
    """No many-body content in any interaction channel, with fragment-confined features."""
    model = randomize(make_model())
    positions, numbers, frag = water_cluster(3, seed=13)

    class ChannelOnly(torch.nn.Module):
        """`mbe_decompose` needs energy_ff/energy_corr; expose one channel at a time."""

        def __init__(self, inner, name):
            super().__init__()
            self.inner = inner
            self.name = name

        def forward(self, batch):
            out = self.inner(batch)
            return type("O", (), {
                "energy": out.interaction[self.name],
                "energy_ff": out.interaction_ff[self.name],
                "energy_corr": out.interaction_corr[self.name],
            })()

    for name in ("elst", "pauli", "disp"):
        mbe = mbe_decompose(
            ChannelOnly(model, name), positions, numbers, frag, split_components=False
        )
        assert abs(mbe.by_order.get(3, 0.0)) < 1e-12, f"{name} has 3-body content"


def test_fragment_charge_is_conserved():
    model = randomize(make_model())
    positions, numbers, frag = water_cluster(3, seed=17)
    batch = make_batch(positions, numbers, frag)
    out = model(batch)
    q = out.response.charges
    per_frag = q.new_zeros(int(batch.n_fragments)).index_add_(0, frag, q)
    assert float(per_frag.abs().max()) < 1e-12


def test_rotation_and_translation_invariance():
    model = randomize(make_model())
    positions, numbers, frag = water_cluster(3, seed=19)
    base = model(make_batch(positions, numbers, frag))

    theta = torch.tensor(0.7)
    c, s = torch.cos(theta), torch.sin(theta)
    rot = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    moved = positions @ rot.T + torch.tensor([1.3, -2.1, 0.7])
    other = model(make_batch(moved, numbers, frag))

    assert torch.allclose(base.energy, other.energy, atol=1e-10)
    for name in base.interaction:
        assert torch.allclose(
            base.interaction[name], other.interaction[name], atol=1e-10
        ), name


def test_forces_match_central_differences():
    model = randomize(make_model())
    positions, numbers, frag = water_cluster(2, seed=23)
    batch = make_batch(positions, numbers, frag)
    batch.positions.requires_grad_(True)
    energy = model(batch).energy.sum()
    (grad,) = torch.autograd.grad(energy, batch.positions)

    h = 1e-5
    for atom, comp in ((0, 0), (1, 2), (4, 1)):
        shifted = []
        for sign in (+1, -1):
            p = positions.clone()
            p[atom, comp] += sign * h
            shifted.append(float(model(make_batch(p, numbers, frag)).energy.sum()))
        fd = (shifted[0] - shifted[1]) / (2 * h)
        assert float(grad[atom, comp]) == pytest.approx(fd, abs=1e-6, rel=1e-6)


# ---------------------------------------------------------------------------
# Range separation and the correction trunk
# ---------------------------------------------------------------------------

def test_r0_and_alpha_stay_positive_and_are_connected_to_the_graph():
    """Positive by construction, and reachable by autograd.

    Deliberately *not* asserting the gradient is large. At ``alpha = 40`` the Fermi switch is
    exactly saturated on ~97% of pairs, so ``r0`` receives gradients of order 1e-14: the
    classes are cleanly separated and there is nothing in the crossover to learn from. That is
    an absence of signal in this data, not a broken connection, and
    :func:`test_range_separation_is_saturated_on_water` measures it directly rather than
    letting it hide behind a threshold here.
    """
    model = randomize(make_model(), scale=0.3, seed=29)
    positions, numbers, frag = water_cluster(2, seed=31)
    out = model(make_batch(positions, numbers, frag))
    for name in RANGE_CHANNELS:
        assert float(out.r0[name].min()) > 0.0
        assert float(out.r0_pair[name].min()) > 0.0
        assert float(out.alpha[name]) > 0.0
    out.energy.sum().backward()
    for name in RANGE_CHANNELS:
        for p in (model.range_heads.d_log_r0[name],
                  model.range_heads.alpha_raw[name],
                  model.pair_head.range_readout[name].weight):
            assert p.grad is not None, name
            assert torch.isfinite(p.grad).all(), name


def test_range_separation_has_a_populated_crossover_and_so_can_learn():
    """A switch only learns from pairs sitting on its shoulder, and the priors put some there.

    This is a real constraint on where a prior may be placed, not a formality. A decisive
    switch is flat on both sides, so ``r0`` receives gradient *only* from pairs inside the
    crossover. The superseded ``r0(H,H) = 1.30`` sat below the entire intramolecular H-H
    range (1.355-1.729), leaving every H-H pair saturated at gate 1 and every O-H saturated
    at 0 -- which is exactly why a 60-epoch training run reported ``gate_intra`` pinned at
    0.3333 to four decimal places and never moved. That constancy was an absence of gradient,
    not agreement.

    At ``r0(H,H) = 1.75`` the intramolecular H-H tail straddles the crossover, so the
    parameter is reachable. Asserted here so that a future prior change cannot silently
    re-freeze it.
    """
    model = make_model()
    positions, numbers, frag = water_cluster(4, seed=151)
    out = model(make_batch(positions, numbers, frag))

    transition = (out.gate["elst"] > 1e-6) & (out.gate["elst"] < 1 - 1e-6)
    assert int(transition.sum()) > 0, (
        "no pair lies in the crossover, so r0 has no gradient anywhere and the 'learned' "
        "range separation is a fixed hyperparameter"
    )
    # Those pairs are the intramolecular ones, which is the population the switch is deciding
    # about; the intermolecular pairs should be firmly switched on.
    assert bool(out.is_intra[transition].all())

    out.energy.sum().backward()
    for param, label in (
        (model.range_heads.d_log_r0["elst"], "per-element r0"),
        (model.pair_head.range_readout["elst"].weight, "per-pair r0"),
    ):
        grad = float(param.grad.abs().sum())
        assert grad > 1e-8, f"{label} gradient is {grad:.3e}; the parameter cannot move"


def test_pairwise_r0_combination_is_symmetric_and_reduces_to_r0_i():
    """The geometric mean is order-independent, and equal endpoints give back the value."""
    model = make_model()
    force_constant_r0(model, 1.37)
    positions, numbers, frag = water_cluster(2, seed=37)
    out = model(make_batch(positions, numbers, frag))
    for name in RANGE_CHANNELS:
        assert torch.allclose(
            out.r0[name], torch.full_like(out.r0[name], 1.37), atol=1e-12
        ), name


def test_pair_r0_starts_at_the_per_element_value_and_can_leave_it():
    """Zero-init, so the pair correction begins at exactly the per-element combination."""
    model = make_model()
    positions, numbers, frag = water_cluster(3, seed=131)
    out = model(make_batch(positions, numbers, frag))
    i, j = out.pair_index
    for name in RANGE_CHANNELS:
        base = (0.5 * (out.r0[name][i].log() + out.r0[name][j].log())).exp()
        assert torch.allclose(out.r0_pair[name], base, atol=1e-12, rtol=0), name

    # ... and once the trunk is live it is genuinely per pair, not a per-element constant.
    # The perturbation is gentle on purpose: `max_log_dev * tanh(.)` saturates for large
    # readout outputs, and a saturated tanh is as constant -- and as gradient-free -- as no
    # correction at all. A scale of 0.4 here was enough to pin every pair at the bound.
    live = randomize(make_model(), scale=0.05, seed=137)
    out = live(make_batch(positions, numbers, frag))
    i, j = out.pair_index
    hh = (numbers[i] == 1) & (numbers[j] == 1)
    spread = out.r0_pair["elst"][hh].detach()
    assert float(spread.max() - spread.min()) > 1e-6, (
        "r0 is constant across all H-H pairs; the pair correction is not reaching it"
    )
    # and it stays within the bound the prior anchors it to
    base = live.range_heads.log_r0_prior.exp().max()
    assert float(spread.max()) < float(base) * 2.1


def test_pair_r0_separates_same_element_pairs_that_per_atom_r0_cannot():
    """The reason this exists: topology, not distance.

    Two H-H pairs of the *same* element pair at the *same* separation but in different
    bonding environments must be able to receive different range separations. A per-atom
    ``r0`` gives one threshold per element pair and cannot; a pair-level one can, because the
    trunk sees both atoms' descriptors. (Water's own geminal H-H sits at 1.51 A where
    distance alone decides, which is why this data cannot exhibit the problem -- so it is
    constructed here.)
    """
    model = randomize(make_model(), scale=0.05, seed=139)
    positions, numbers, frag = water_cluster(2, seed=149)
    out = model(make_batch(positions, numbers, frag))
    i, j = out.pair_index
    hh = ((numbers[i] == 1) & (numbers[j] == 1)).nonzero().squeeze(-1)
    assert hh.numel() >= 2

    # Same element pair, and we compare at a *fixed* distance so only the environment can
    # be responsible for any difference.
    feats = model.featurizer(make_batch(positions, numbers, frag), frag)
    fixed_r = torch.full((hh.numel(),), 1.9, dtype=positions.dtype)
    dev = model.pair_head(
        feats.inv_feats[i[hh]], feats.inv_feats[j[hh]],
        feats.species_idx[i[hh]], feats.species_idx[j[hh]], fixed_r,
    )[1]["elst"]
    assert float(dev.max() - dev.min()) > 1e-6, (
        "at fixed r and fixed elements the range separation is constant; it can only be a "
        "function of distance and species, which is what a per-atom r0 already was"
    )


def test_corrections_are_exactly_zero_beyond_r_off():
    """Compact support: past ``r_off`` only the classical form exists.

    This is the property that stops a correction channel absorbing the long-range tail the
    classical form is there to provide.
    """
    model = randomize(make_model(), scale=0.5, seed=41)
    positions, numbers, frag = water_cluster(3, spacing=4.0, seed=43)
    batch = make_batch(positions, numbers, frag)
    feats = model.featurizer(batch, frag)
    out = model(batch)
    dE = call_pair_head(model, feats, out.pair_index, out.r)
    for name, spec in model.pair_head.channels.items():
        beyond = out.r >= spec.r_off
        if bool(beyond.any()):
            assert float(dE[name][beyond].abs().max()) == 0.0, name


def test_pair_head_is_symmetric_under_swap():
    model = randomize(make_model(), seed=47)
    positions, numbers, frag = water_cluster(2, seed=53)
    batch = make_batch(positions, numbers, frag)
    feats = model.featurizer(batch, frag)
    out = model(batch)
    flipped = out.pair_index.flip(0)
    for which in (0, 1):        # energies, then the per-pair r0 deviations
        a = call_pair_head(model, feats, out.pair_index, out.r, which)
        b = call_pair_head(model, feats, flipped, out.r, which)
        for name in a:
            assert torch.allclose(a[name], b[name], atol=0, rtol=0), (which, name)


# ---------------------------------------------------------------------------
# The pair list and the forward-compatibility slots
# ---------------------------------------------------------------------------

def test_union_pairs_keeps_intra_pairs_beyond_any_cutoff():
    positions, numbers, frag = water_cluster(2, seed=59)
    positions[2] = torch.tensor([0.0, 0.0, 40.0])       # still fragment 0
    batch = make_batch(positions, numbers, frag)
    pair_index, r, is_intra, pair_frag = union_pairs(
        batch.positions, batch.batch_idx, frag, 12.0, max_num_neighbors=4
    )
    sel = (pair_index[0] == 0) & (pair_index[1] == 2)
    assert bool(sel.any()), "a 40 A intra pair must survive both the cutoff and the truncation"
    want = float((positions[0] - positions[2]).norm())
    assert float(r[sel]) == pytest.approx(want, abs=1e-9)
    assert bool(is_intra[sel])
    assert int(pair_frag[~is_intra].max()) == -1


def test_fragment_state_block_is_inert_for_neutral_singlets():
    """Zero for (Q, 2S) = (0, 0) whatever the weights are, and non-zero once charged."""
    block = FragmentStateEmbedding(4)
    with torch.no_grad():           # move it well off its initialization
        for p in block.parameters():
            p.add_(0.7 * torch.randn(p.shape))
    positions, numbers, frag = water_cluster(2, seed=61)
    batch = make_batch(positions, numbers, frag)

    neutral = block(batch, frag, positions.dtype, positions.device)
    assert float(neutral.abs().max()) == 0.0

    batch.fragment_charge = torch.tensor([1.0, 0.0])
    charged = block(batch, frag, positions.dtype, positions.device)
    assert float(charged[frag == 0].abs().max()) > 0.0
    assert float(charged[frag == 1].abs().max()) == 0.0


def test_fragment_state_dim_zero_changes_nothing():
    positions, numbers, frag = water_cluster(2, seed=67)
    batch = make_batch(positions, numbers, frag)
    a = make_model(build_parts(fragment_state_dim=0, seed=5))(batch).energy
    b = make_model(build_parts(fragment_state_dim=4, seed=5))(batch).energy
    # Same seed, and the block contributes an exactly-zero column, so the two agree.
    assert torch.allclose(a, b, atol=1e-12)


# ---------------------------------------------------------------------------
# Environment-aware descriptors
# ---------------------------------------------------------------------------

def test_environment_residual_is_inert_at_initialization():
    """``g`` is zero-init, so turning environment awareness on starts from exactly the
    fragment-confined model rather than from a different one."""
    positions, numbers, frag = water_cluster(3, seed=97)
    batch = make_batch(positions, numbers, frag)
    off = make_model(build_parts(seed=9))(batch)
    on = make_model(build_parts(seed=9), environment=True)(batch)
    assert torch.allclose(off.energy, on.energy, atol=1e-12, rtol=0)
    assert float(on.environment_norm.abs().max()) == 0.0
    assert off.environment_norm is None


def test_environment_awareness_reaches_interactions_and_not_the_fragment_energy():
    """The whole point of the split.

    A live ``g`` must move the interaction channels -- that is the effective many-body
    physics -- while leaving ``fragment_energy`` bitwise untouched, because that label is an
    isolated-fragment energy with no environment dependence to fit.

    Getting this to be *exactly* zero rather than merely small is why the Pauli and dispersion
    parameter heads are evaluated on both feature streams. Left on ``h_env`` alone the
    violation was ~1.1 kJ/mol at the environment strength this model trains to -- about half
    the target ``ob_mae``, which is a systematic error comfortably large enough to be
    mistaken for fit error.
    """
    parts = build_parts(seed=11)
    base = randomize(make_model(parts, environment=True), seed=13)
    positions, numbers, frag = water_cluster(3, seed=101)
    batch = make_batch(positions, numbers, frag)
    before = base(batch)

    after = wake_environment(base)(batch)
    assert float(after.environment_norm.mean()) > 0.0
    moved_interaction = float(
        (after.interaction["disp"] - before.interaction["disp"]).abs().max()
    ) * KJMOL_PER_HARTREE
    moved_fragment = float(
        (after.fragment_energy - before.fragment_energy).abs().max()
    ) * KJMOL_PER_HARTREE
    assert moved_interaction > 0.1, (
        f"g barely reached the interactions ({moved_interaction:.4f} kJ/mol); the test is "
        f"not exercising the environment path"
    )
    assert moved_fragment == 0.0, (
        f"{moved_fragment:.6f} kJ/mol of environment reached an isolated-fragment label"
    )


def test_fragment_energy_is_exactly_one_body_with_environment_on():
    """Environment awareness costs nothing in the 1-body term's exactness.

    Every route by which the surroundings could reach an isolated-fragment label is closed:
    the response solve and the bond channel read the fragment-confined stream, and the Pauli
    and dispersion parameters are evaluated on it too for intra-fragment pairs. Asserted at
    both environment settings, so a regression in either path is caught.
    """
    positions, numbers, frag = water_cluster(4, seed=103)
    cluster_b = make_batch(positions, numbers, frag)
    alone_b = make_batch(positions[:3], numbers[:3], frag[:3])

    off = randomize(make_model(environment=False), seed=17)
    assert torch.allclose(
        off(cluster_b).fragment_energy[0], off(alone_b).fragment_energy[0],
        atol=1e-12, rtol=0,
    ), "the fragment-confined path must stay exactly one-body"

    on = wake_environment(randomize(make_model(environment=True), seed=17))
    violation = float(
        (on(cluster_b).fragment_energy[0] - on(alone_b).fragment_energy[0]).abs()
    ) * KJMOL_PER_HARTREE
    assert violation == 0.0, f"one-body violation of {violation:.6f} kJ/mol with g live"


def test_environment_gives_the_dispersion_channel_many_body_content():
    """Fragment-confined dispersion is rigorously two-body; environment-aware is not.

    That is the trade being made, so assert both halves of it rather than only the gain.
    """
    positions, numbers, frag = water_cluster(3, seed=107)

    class ChannelOnly(torch.nn.Module):
        def __init__(self, inner, name):
            super().__init__()
            self.inner, self.name = inner, name

        def forward(self, batch):
            out = self.inner(batch)
            return type("O", (), {
                "energy": out.interaction[self.name],
                "energy_ff": out.interaction_ff[self.name],
                "energy_corr": out.interaction_corr[self.name],
            })()

    confined = randomize(make_model(build_parts(seed=19)), seed=23)
    aware = wake_environment(randomize(make_model(build_parts(seed=19), environment=True), seed=23))
    three = [
        abs(mbe_decompose(ChannelOnly(m, "disp"), positions, numbers, frag,
                          split_components=False).by_order.get(3, 0.0))
        for m in (confined, aware)
    ]
    assert three[0] < 1e-12, "fragment-confined dispersion must have no 3-body content"
    assert three[1] > 1e-9, "environment-aware dispersion should have some"


def test_environment_path_keeps_rotation_invariance():
    """``g`` gates equivariant channels with invariant scalars; a scalar MLP on components
    would break this silently."""
    model = wake_environment(randomize(make_model(environment=True), seed=29))
    positions, numbers, frag = water_cluster(3, seed=109)
    base = model(make_batch(positions, numbers, frag))
    theta = torch.tensor(0.9)
    c, s = torch.cos(theta), torch.sin(theta)
    rot = torch.tensor([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    other = model(make_batch(positions @ rot.T + torch.tensor([0.4, 1.1, -0.6]), numbers, frag))
    assert torch.allclose(base.energy, other.energy, atol=1e-10)
    for name in base.interaction:
        assert torch.allclose(base.interaction[name], other.interaction[name], atol=1e-10), name


def test_environment_forces_match_central_differences():
    model = wake_environment(randomize(make_model(environment=True), seed=31), scale=0.15)
    positions, numbers, frag = water_cluster(2, seed=113)
    batch = make_batch(positions, numbers, frag)
    batch.positions.requires_grad_(True)
    (grad,) = torch.autograd.grad(model(batch).energy.sum(), batch.positions)
    h = 1e-5
    for atom, comp in ((0, 1), (3, 0)):
        shifted = []
        for sign in (+1, -1):
            p = positions.clone()
            p[atom, comp] += sign * h
            shifted.append(float(model(make_batch(p, numbers, frag)).energy.sum()))
        fd = (shifted[0] - shifted[1]) / (2 * h)
        assert float(grad[atom, comp]) == pytest.approx(fd, abs=1e-6, rel=1e-6)


def test_featurizer_pair_path_matches_independent_builds():
    """``also_ungrouped=True`` returns the same two descriptors as two separate calls."""
    positions, numbers, frag = water_cluster(3, seed=127)
    batch = make_batch(positions, numbers, frag)
    f = FlatLambdaSOAPFeaturizer(
        cutoff=5.0, n_max=4, l_max=3, neighbor_types=NEIGHBOR_TYPES,
        selected_lambdas=(0, 1, 2), backend="e3nn", density_channels=8,
    )
    grouped, full = f(batch, frag, also_ungrouped=True)
    assert torch.allclose(grouped.inv_feats, f(batch, frag).inv_feats, atol=1e-12)
    assert torch.allclose(full.inv_feats, f(batch).inv_feats, atol=1e-12)
    # and they are genuinely different descriptors
    assert float((grouped.inv_feats - full.inv_feats).abs().max()) > 1e-3
    # group_idx=None makes the pair degenerate rather than raising
    a, b = f(batch, None, also_ungrouped=True)
    assert a is b


def test_masked_scatter_shares_one_geometric_basis():
    """``h_frag`` and ``h_full`` differ only by an edge mask over one ``edge_expansion``.

    The plumbing the environment-aware descriptor will use. The fragment-grouped neighbor
    list is a strict subset of the frame-grouped one, so masking gives bit-identical results
    to searching again -- which is what makes the two descriptors' difference attributable to
    the environment alone.
    """
    from torch_cluster import radius_graph

    positions, numbers, frag = water_cluster(3, seed=71)
    batch = make_batch(positions, numbers, frag)
    density = DensityExpansion(n_max=4, l_max=3, cutoff=5.0, n_species=2)
    species = (numbers == 8).long()
    n = positions.shape[0]

    e_full = radius_graph(positions, r=5.0, batch=batch.batch_idx, loop=False)
    e_frag = radius_graph(positions, r=5.0, batch=frag, loop=False)
    mask = frag[e_full[0]] == frag[e_full[1]]
    assert set(zip(e_frag[0].tolist(), e_frag[1].tolist())) == set(
        zip(e_full[0, mask].tolist(), e_full[1, mask].tolist())
    ), "fragment edges must be a subset of frame edges for the mask trick to be valid"

    RY = density.edge_expansion(positions, e_full)
    masked = density.scatter_species(RY, e_full, species, n, edge_mask=mask)
    assert torch.equal(masked, density(positions, species, e_frag, n))
    assert torch.equal(
        density.scatter_species(RY, e_full, species, n),
        density(positions, species, e_full, n),
    )
    # and the two really are different descriptors, not the same thing twice
    assert float((masked - density(positions, species, e_full, n)).abs().max()) > 1e-3


def test_batching_matches_separate_frames():
    model = randomize(make_model(), seed=73)
    pa, na, fa = water_cluster(2, seed=79)
    pb, nb, fb = water_cluster(3, seed=83)
    single = [
        model(make_batch(pa, na, fa)).energy,
        model(make_batch(pb, nb, fb)).energy,
    ]
    batched = model(make_batch(
        torch.cat((pa, pb)), torch.cat((na, nb)),
        torch.cat((fa, fb + int(fa.max()) + 1)),
        batch_idx=torch.cat((
            torch.zeros(pa.shape[0], dtype=torch.long),
            torch.ones(pb.shape[0], dtype=torch.long),
        )),
    )).energy
    assert torch.allclose(batched, torch.cat(single), atol=1e-10)


def test_missing_partition_raises():
    model = make_model()
    positions, numbers, frag = water_cluster(2, seed=89)
    batch = make_batch(positions, numbers, frag)
    batch.fragment_idx = None
    with pytest.raises(ValueError, match="fragment_idx"):
        model(batch)
