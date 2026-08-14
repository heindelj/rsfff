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
        emb_dim=8, hidden=24, depth=1,
    )
    return dict(
        featurizer=featurizer, response=response, disp_params=disp_params,
        pauli_params=pauli_params, range_heads=range_heads, pair_head=pair_head,
        fragment_state=fragment_state,
    )


def make_model(parts=None, **kw):
    p = parts or build_parts(**kw)
    return UnifiedPairModel(
        p["featurizer"], p["response"], p["disp_params"], p["pauli_params"],
        p["range_heads"], p["pair_head"], p["fragment_state"], E0, max_rank=MAX_RANK,
    )


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

    # The H-H case does NOT separate (see rsfff.ff.range_priors) and is deliberately left on.
    hh = (z[i] == 1) & (z[j] == 1)
    assert float(out.gate["elst"][out.is_intra & hh].detach().min()) > 0.5


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
    every_pair = sum(
        (out.e_pair_ff[c] + out.e_pair_corr[c]).sum() for c in out.e_pair_ff
    )
    bond = out.e_pair_corr["bond"][out.is_intra].sum()
    total = (
        out.energy_ref.sum() + out.energy_internal.sum() + every_pair + bond
    )
    assert torch.allclose(out.energy.sum(), total, atol=1e-12), (
        f"accounting identity off by {float(out.energy.sum() - total):.3e} Ha"
    )
    # The bond channel is intra-only, which is exactly why relabelling is not a no-op.
    assert float(out.e_pair_corr["bond"][~out.is_intra].abs().sum()) > 0.0


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
    dE = model.pair_head(feats.inv_feats, feats.species_idx, out.pair_index, out.r)
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

def test_r0_and_alpha_stay_positive_and_carry_gradient():
    model = randomize(make_model(), scale=0.3, seed=29)
    positions, numbers, frag = water_cluster(2, seed=31)
    out = model(make_batch(positions, numbers, frag))
    for name in RANGE_CHANNELS:
        assert float(out.r0[name].min()) > 0.0
        assert float(out.alpha[name]) > 0.0
    out.energy.sum().backward()
    for name in RANGE_CHANNELS:
        assert model.range_heads.d_log_r0[name].grad is not None
        assert float(model.range_heads.d_log_r0[name].grad.abs().sum()) > 0.0
        assert float(model.range_heads.alpha_raw[name].grad.abs()) > 0.0


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
    dE = model.pair_head(feats.inv_feats, feats.species_idx, out.pair_index, out.r)
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
    a = model.pair_head(feats.inv_feats, feats.species_idx, out.pair_index, out.r)
    b = model.pair_head(feats.inv_feats, feats.species_idx, flipped, out.r)
    for name in a:
        assert torch.allclose(a[name], b[name], atol=0, rtol=0), name


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
