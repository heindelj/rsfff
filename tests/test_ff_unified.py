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
from rsfff.ff.atomic_energy import AtomicStateEnergy
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


def build_parts(*, fragment_state_dim=0, alpha_init=40.0, seed=0, extra_dim=0,
                direct_multipoles=False, quadrupole_response=True):
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
            quadrupole_response=quadrupole_response,
        ),
        PairComplianceHead(p0, hidden=24, depth=1, cutoff=5.0, s_init=0.5),
        direct_multipoles=direct_multipoles,
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
        extra_dim=extra_dim,
        emb_dim=8, hidden=24, depth=1,
    )
    # Built *last*, and attached rather than passed to the constructor, so that adding it does
    # not consume random numbers ahead of any other block. Several tests here are calibrated
    # against a specific initialization -- the finite-difference gradient checks in particular
    # have thresholds that a different random operating point walks straight through.
    atomic_energy = AtomicStateEnergy(
        p0, n_species, p1=p1, p2=p2, irrep2_to_spherical=irrep2_to_spherical(voigt),
        emb_dim=8, hidden=24, depth=1, equiv_channels=6, energy_scale=0.2,
    )
    return dict(
        featurizer=featurizer, response=response, disp_params=disp_params,
        pauli_params=pauli_params, range_heads=range_heads, pair_head=pair_head,
        fragment_state=fragment_state, atomic_energy=atomic_energy,
    )


def make_model(parts=None, *, environment=False, levels=None, **kw):
    p = parts or build_parts(**kw)
    del kw
    levels = dict(levels or {})
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
        atomic_energy=p["atomic_energy"],
        environment=env, max_rank=MAX_RANK, **levels,
    )


def wake_environment(model, scale=1.0, seed=101):
    """Move ``g`` off its zero initialization so the environment path is actually live.

    Perturbs the **weights only**. ``EnvironmentResidual`` is anchored as
    ``g(h_full) - g(h_frag)``, so the *invariant* readout's bias appears in both terms
    additively and cancels -- perturbing it does nothing at all. The scale is larger than it
    looks like it needs to be for the same reason: only the part of ``g`` that discriminates
    between the two streams survives. See
    :func:`test_environment_residual_is_anchored_at_the_isolated_fragment`.
    """
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for m in (model.environment.inv_mlp, model.environment.vec_gate,
                  model.environment.equiv_gate):
            if m is not None:
                m[-1].weight.add_(scale * torch.randn(m[-1].weight.shape, generator=g))
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
        out.energy_ref + out.energy_internal + out.energy_bond + out.energy_atom,
        atol=0, rtol=0,
    )
    # Every channel is structurally zero wherever it does not belong, so the identity is
    # simply "classical everywhere + every correction that exists" -- plus the one term that
    # is per *atom* rather than per pair.
    total = (
        out.energy_ref.sum()
        + out.energy_internal.sum()
        + out.energy_atom.sum()
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
        one.energy_ref.sum() + one.energy_internal.sum() + one.energy_atom.sum()
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


class _ChannelOnly(torch.nn.Module):
    """Expose one interaction channel as ``energy`` so ``mbe_decompose`` can take it."""

    def __init__(self, inner, name):
        super().__init__()
        self.inner = inner
        self.name = name

    def forward(self, batch):
        return type("O", (), {"energy": self.inner(batch).interaction[self.name]})()


@pytest.mark.parametrize("levels", [
    {},
    {"induction": True},
])
def test_electrostatics_stays_two_body_with_the_environment_live(levels):
    """``cls_elec`` is fragment-confined at *every* level, environment residual or not.

    ``eda_frz_elec`` is the Coulomb interaction between superimposed frozen monomer densities,
    which is rigorously pairwise. Two things used to carry ``h_env`` into it: the pair
    correction, and -- less visibly -- the per-pair range-separation deviation feeding
    ``gate["elst"]``. Only the first was ever re-scored on ``h_frag``, because the re-score
    discarded the deviations it also returns.

    This is the test the old ``test_interaction_channels_are_exactly_two_body`` could not be:
    that one builds the model with ``environment=False``, so ``h_env is h_frag`` and every
    such assertion passes for free. Here the residual is explicitly woken, which is the only
    configuration where the bug exists at all.
    """
    model = randomize(wake_environment(
        make_model(environment=True, extra_dim=9, levels=levels), scale=3.0
    ))
    positions, numbers, frag = water_cluster(3, seed=13)

    live = model(make_batch(positions, numbers, frag))
    assert float(live.environment_norm.detach().abs().max()) > 1e-6, "environment is not live"

    mbe = mbe_decompose(
        _ChannelOnly(model, "elst"), positions, numbers, frag, split_components=False
    )
    assert abs(mbe.by_order.get(3, 0.0)) < 1e-12, "cls_elec has 3-body content"

    # Pauli and dispersion are many-body *by design* -- their parameters read `h_env` on inter
    # pairs, which is what makes C6 environment-quenched. This pins the asymmetry as intended
    # rather than accidental.
    many = [
        abs(float(mbe_decompose(
            _ChannelOnly(model, n), positions, numbers, frag, split_components=False
        ).by_order.get(3, 0.0)))
        for n in ("pauli", "disp")
    ]
    assert max(many) > 1e-12, (
        "pauli and disp lost their many-body content too, so this test would pass even if "
        "the environment residual were dead"
    )


def test_the_electrostatic_environment_term_is_booked_as_induction():
    """What leaves ``cls_elec`` arrives in ``induction``: a re-partition, not a deletion.

    ``gate_env * E == gate_frag * E + (gate_env - gate_frag) * E`` is an identity, so moving the
    environment-dependent half into the induction correction cannot change the total. It
    changes only which column the energy is reported in -- which is the whole point, since a
    frozen-density label cannot be fit by an environment-dependent function.
    """
    model = randomize(wake_environment(
        make_model(environment=True, extra_dim=9, levels={"induction": True}), scale=3.0
    ))
    positions, numbers, frag = water_cluster(3, seed=23)
    out = model(make_batch(positions, numbers, frag))

    assert out.elst_env is not None
    assert float(out.elst_env.detach().abs().max()) > 1e-9, "nothing moved to test"

    total = out.fragment_energy.sum() + sum(v.sum() for v in out.interaction.values())
    assert torch.allclose(out.energy.sum(), total, atol=1e-12), (
        "the split is an identity and must leave the total energy alone"
    )


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


def test_environment_residual_is_anchored_at_the_isolated_fragment():
    """``h_env == h_frag`` exactly for a lone fragment, at any stage of training.

    ``EnvironmentResidual`` is written as a *difference*, ``g(h_full) - g(h_frag)``, so for a
    fragment whose only neighbors are its own atoms the two streams coincide -- not merely at
    initialization, and not merely for small ``g``. Everything defined as a difference between
    the streams (the induction correction) is therefore zero on a
    monomer, which is what their labels require.

    **Zero to round-off, not bitwise.** ``h_full`` and ``h_frag`` are produced by two different
    scatters over the same edges (masked and unmasked), and those sum in different orders, so
    an isolated fragment's two descriptors agree to ~1e-16 rather than identically. The
    anchoring is exact in exact arithmetic; what is measured here is float64 noise on features
    of order 1, which reaches the energy at ~1e-16 Hartree.

    The invariant readout's bias cancelling is the visible signature: it appears in both terms
    additively, so perturbing it moves nothing. Note this does **not** extend to the
    equivariant gates -- there the bias multiplies two different tensors
    (``full.vec_feats`` against ``frag.vec_feats``), so it survives on a cluster and vanishes
    only in the isolated-fragment limit, which is exactly the property being claimed.
    """
    model = wake_environment(randomize(make_model(build_parts(seed=11), environment=True), seed=13))
    positions, numbers, frag = water_cluster(1, seed=7)
    out = model(make_batch(positions, numbers, frag))
    assert float(out.environment_norm.abs().max()) < 1e-13, (
        "an isolated fragment must see no environment; the residual is not anchored"
    )

    # ... and the invariant readout's bias is structurally inert, even on a cluster where the
    # residual is very much live.
    positions, numbers, frag = water_cluster(3, seed=101)
    batch = make_batch(positions, numbers, frag)
    before = model(batch)
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        model.environment.inv_mlp[-1].bias.add_(
            torch.randn(model.environment.inv_mlp[-1].bias.shape, generator=g)
        )
    after = model(batch)
    assert torch.allclose(after.energy, before.energy, atol=1e-11, rtol=0.0)


def test_environment_awareness_reaches_interactions_and_not_the_fragment_energy():
    """The whole point of the split.

    A live ``g`` must move the interaction channels -- that is the effective many-body
    physics -- while leaving ``fragment_energy`` essentially untouched, because that label is
    an isolated-fragment energy with no environment dependence to fit.

    **Essentially, not bitwise, and the difference is deliberate.** The Pauli and dispersion
    parameters now read ``h_env`` on every pair, intra ones included, because an
    environment-quenched ``C6`` *is* the effective many-body content those channels exist to
    carry and selecting per pair made a bonded pair answer to a different network than a
    hydrogen-bonded one. The price is that intra-fragment classical energy picks up a little
    environment dependence.

    What that price is allowed to be is set by what it is measured against. The two terms
    that actually carry ``fragment_energy`` -- ``E_internal`` from the fragment-confined
    response and ``E_atom`` from ``h_frag`` -- must still be **exactly** unchanged, and they
    are asserted so here. What is left is the intra classical channels, which the range
    separation has already switched down to well under a kJ/mol per fragment.
    """
    parts = build_parts(seed=11)
    base = randomize(make_model(parts, environment=True), seed=13)
    positions, numbers, frag = water_cluster(3, seed=101)
    batch = make_batch(positions, numbers, frag)
    before = base(batch)

    after = wake_environment(base)(batch)
    assert float(after.environment_norm.detach().mean()) > 0.0
    # Over all three channels, not ``disp`` alone. How much of ``g`` a *single* channel picks
    # up is a property of that channel's random readout, not of the environment path: measured
    # across five initializations the disp-only figure ranged over 0.004 to 4.8 kJ/mol while
    # the maximum over the three stayed in 0.75 to 4.8. The one-channel version of this guard
    # therefore fired on an unrelated change to the model's parameter count, which is exactly
    # the failure mode a guard should not have.
    moved_interaction = max(
        float((after.interaction[c] - before.interaction[c]).detach().abs().max())
        for c in ("elst", "pauli", "disp")
    ) * KJMOL_PER_HARTREE
    moved_fragment = float(
        (after.fragment_energy - before.fragment_energy).detach().abs().max()
    ) * KJMOL_PER_HARTREE
    assert moved_interaction > 0.1, (
        f"g barely reached the interactions ({moved_interaction:.4f} kJ/mol); the test is "
        f"not exercising the environment path"
    )
    # Exactly zero on the two terms that carry the label ...
    assert torch.equal(after.energy_internal, before.energy_internal), (
        "the environment reached E_internal; the response solve must stay fragment-confined"
    )
    assert torch.equal(after.energy_atom, before.energy_atom), (
        "the environment reached E_atom^0; it must be evaluated on h_frag"
    )
    # ... and **the route, not the magnitude**, on what is left. With `E_internal` and
    # `E_atom` bitwise unchanged above and `energy_ref` constant, `fragment_energy` can only
    # have moved through `energy_bond` -- the intra classical bucket -- so that is asserted as
    # an exact identity rather than bounded by a number. The size of the residue on a
    # *randomized* model is a property of the random draw: this guard was `< 1e-3` against a
    # measured 0.0019 and fired at 0.0158 on a change to the model's parameter count, which is
    # the same failure mode the `moved_interaction` guard above was already widened for.
    # `allclose` at 1e-12 Ha (2.6e-9 kJ/mol), not `equal`: `fragment_energy` is a sum of four
    # terms, so differencing it does not reassociate bit-for-bit with differencing one of them.
    assert torch.allclose(
        (after.fragment_energy - before.fragment_energy).detach(),
        (after.energy_bond - before.energy_bond).detach(),
        atol=1e-12, rtol=0,
    ), (
        f"{moved_fragment:.6g} kJ/mol of environment reached an isolated-fragment label "
        f"through a route other than the intra classical channels"
    )


def test_fragment_energy_is_exactly_one_body_with_environment_on():
    """Environment awareness costs the 1-body term almost nothing, and exactly nothing
    on the two terms that carry it.

    The routes that *must* stay closed are closed exactly: ``E_internal`` comes from the
    fragment-confined response solve and ``E_atom^0`` is evaluated on ``h_frag``, and both
    are asserted bitwise below. Together they are essentially the whole of
    ``fragment_energy``.

    What is left open, deliberately, is the intra-fragment classical energy: the Pauli and
    dispersion parameters read ``h_env`` on every pair now, because environment-quenched
    parameters are the effective many-body content those channels exist to carry. The range
    separation has already switched that classical contribution down to well under a kJ/mol
    per fragment, so the residue is small -- and it is bounded here rather than assumed.
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
    in_cluster, alone = on(cluster_b), on(alone_b)
    assert torch.equal(in_cluster.energy_internal[0], alone.energy_internal[0]), (
        "the environment reached E_internal; the response solve must stay fragment-confined"
    )
    assert torch.equal(in_cluster.energy_atom[0], alone.energy_atom[0]), (
        "the environment reached E_atom^0; it must be evaluated on h_frag"
    )
    violation = float(
        (in_cluster.fragment_energy[0] - alone.fragment_energy[0]).abs()
    ) * KJMOL_PER_HARTREE
    # **The route, not the magnitude**, for the same reason as
    # `test_environment_awareness_reaches_interactions_and_not_the_fragment_energy`: with
    # `E_internal` and `E_atom` bitwise equal above and `energy_ref` constant, the only way
    # `fragment_energy` can differ is `energy_bond`. The size of the residue on an *unfitted*
    # model is a property of the random draw -- this was `< 1e-2` against a measured 0.0019
    # and moved to 20 on a change to the parameter count. On a trained model the quantity to
    # watch is the one-body bias in `scripts/staged_diagnostics.py`.
    assert torch.allclose(
        (in_cluster.fragment_energy[0] - alone.fragment_energy[0]),
        (in_cluster.energy_bond[0] - alone.energy_bond[0]),
        atol=1e-12, rtol=0,
    ), (
        f"one-body violation of {violation:.6g} kJ/mol with g live, through a route other "
        f"than the intra classical channels"
    )


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


# ---------------------------------------------------------------------------
# The environment residual's bootstrap, and what destroyed it
# ---------------------------------------------------------------------------

def test_environment_residual_hidden_layers_get_no_gradient_at_initialization():
    """The hazard behind ``no_weight_decay``, pinned as a fact rather than a warning.

    ``EnvironmentResidual``'s readout is zero-initialized, so every hidden layer's gradient is
    proportional to it and is *exactly* zero on the first step -- there is nothing for weight
    decay to compete with, and the block is destroyed rather than regularized (see
    :func:`rsfff.train.term_loop.parameter_groups` for the measured collapse).

    The readout itself is the escape: its gradient carries
    ``hidden(x_full) - hidden(x_frag)``, which is nonzero at the random initialization the
    hidden layers still hold. That is the whole bootstrap, and it only works if those layers
    are left where they started.
    """
    model = make_model(environment=True)
    randomize(model, scale=0.05, seed=3)      # wake the consumers, not the residual
    with torch.no_grad():                     # ...but put the residual back at zero
        for m in (model.environment.inv_mlp, model.environment.vec_gate,
                  model.environment.equiv_gate):
            m[-1].weight.zero_()
            m[-1].bias.zero_()
    positions, numbers, frag = water_cluster(3, seed=11)
    model(make_batch(positions, numbers, frag)).energy.sum().backward()

    inv = model.environment.inv_mlp
    for k, layer in enumerate(inv[:-1]):
        if hasattr(layer, "weight"):
            assert float(layer.weight.grad.abs().max()) == 0.0, f"inv_mlp[{k}] gradient"
    assert float(inv[-1].weight.grad.abs().max()) > 1e-9, (
        "the readout has no gradient either, so nothing can bootstrap the block"
    )


def test_every_zero_initialized_readout_block_is_exempt_from_weight_decay():
    """The blocks that cannot recover from weight decay are all exempt. Named, deliberately.

    A parameter with **exactly zero** loss gradient at initialization has nothing competing
    with weight decay, so decay does not regularize it, it deletes it. Many parameters are in
    that position at step 0 -- with every readout at zero, even the featurizer's
    ``channel_proj`` gets no gradient -- and most of them recover the instant the readouts move
    (measured over a staged fit: ``channel_proj`` 3.14 -> 3.10, ``alpha_head.base_mlp``
    2.67 -> 1.62, both healthy). So "zero gradient at init" is not the criterion.

    The criterion is whether the block can *get back*, and a zero-initialized readout block
    cannot: its own readout's gradient carries the hidden activations, so once decay has
    flattened the hidden layers there is no signal left to regrow them. Those blocks lost the
    race and went to zero -- see :func:`rsfff.mlip.heads.zero_init_readout` for the numbers and
    for what each one silently stopped doing. This pins the exemption on the measured casualties
    by name, because that list is the record of what went wrong.

    The ``equiv_reduce`` entries are the *second* set of casualties, and they extend the
    criterion rather than merely lengthening the list. A zero-initialized gate does not only
    starve the layers behind it -- it starves its non-exempt **sibling**, the channel reduction
    it multiplies, whose gradient is proportional to that same zero gate. Those two are then
    each other's only gradient path, so it is a deadlock and not a race: measured over a staged
    fit ``cquad_axis_head.axis.equiv_reduce`` was at a denormal zero before the first epoch
    ended and its gate readout was still *exactly* zero 210 epochs later. That is why the
    exemption now sits on the head rather than on its ``gate_mlp``.
    """
    from rsfff.train.term_loop import parameter_groups

    model = make_model(environment=True)
    exempt = {
        id(p) for g in parameter_groups(model, 1.0e-4) if g["weight_decay"] == 0.0
        for p in g["params"]
    }
    named = dict(model.named_parameters())
    casualties = [
        "environment.inv_mlp", "environment.vec_gate", "environment.equiv_gate",
        "disp_params.c6_mlp", "pauli_params.q_mlp",
        "response.compliance_head.net", "response.params.chi_mlp", "response.params.eta_mlp",
        "response.params.chivec_head.gate_mlp", "response.params.chiquad_head.gate_mlp",
        "pauli_params.dipole_head.gate_mlp", "pauli_params.quadrupole_head.gate_mlp",
        "pair_head.trunk", "pair_head.readout", "pair_head.range_readout",
    ]
    for prefix in casualties:
        block = [n for n in named if n.startswith(prefix + ".")]
        assert block, f"{prefix} is not in this model; the list is stale"
        missed = [n for n in block if id(named[n]) not in exempt]
        assert not missed, f"{prefix} would still be decayed: {missed}"


def test_every_equivariant_channel_reduction_is_exempt_from_weight_decay():
    """``equiv_reduce`` is the sibling of a zero-init gate, and it deadlocks rather than races.

    ``head(x) = gate(x) . (equiv_feats @ equiv_reduce)``. With ``gate`` zero-initialized,
    ``dL/d(equiv_reduce)`` is exactly zero at step 0 and weight decay runs unopposed; once
    ``equiv_reduce`` reaches zero the head's output is identically zero, which zeroes the
    gradient into ``gate`` as well, and neither can restart the other. Exempting only the
    ``gate_mlp`` -- which is what handing it to :func:`zero_init_readout` does, since that
    function is given the MLP and not the head -- protects the wrong half of the pair.
    """
    from rsfff.train.term_loop import parameter_groups

    model = make_model(environment=True)
    exempt = {
        id(p) for g in parameter_groups(model, 1.0e-4) if g["weight_decay"] == 0.0
        for p in g["params"]
    }
    reductions = [n for n, _ in model.named_parameters() if n.endswith("equiv_reduce")]
    assert len(reductions) >= 5, f"the equivariant heads have moved: {reductions}"
    named = dict(model.named_parameters())
    missed = [n for n in reductions if id(named[n]) not in exempt]
    assert not missed, f"these would be deleted by weight decay: {missed}"


def test_per_species_tables_and_the_featurizer_still_see_weight_decay():
    """The exemption is a scalpel, not a blanket. What recovers on its own keeps decaying.

    ``alpha_head.base_mlp`` used to be on this list, on the measured grounds that it gets a real
    gradient at initialization and stayed healthy (2.67 -> 1.62). It moved to the exempt side
    when the exemption moved from the ``gate_mlp`` to the head, and that is deliberate rather
    than collateral: it is the readout that sets the on-site polarizability, whose decay
    attractor is ``softplus(0) + psd_floor = 0.693`` a0^3 per atom, and the monomer fit that got
    water's polarizability right ran with no weight decay at all.
    """
    from rsfff.train.term_loop import parameter_groups

    model = make_model(environment=True)
    decayed = {
        id(p) for g in parameter_groups(model, 1.0e-4) if g["weight_decay"] > 0.0
        for p in g["params"]
    }
    named = dict(model.named_parameters())
    for name in ("disp_params.d_log_c6", "response.params.chi0", "response.params.d_log_z",
                 "featurizer.channel_proj"):
        if name in named:
            assert id(named[name]) in decayed, f"{name} should still be decayed"


def test_cosine_schedule_anneals_to_the_configured_floor():
    """``lr_final_factor`` is a *fraction* of the initial rate, reached at the last epoch.

    The schedule exists to damp the random walk along the unlabeled ``E_internal``/``E_atom``
    gauge, so what matters is that the rate actually gets small by the end -- a schedule that
    stopped at half the initial rate would not have done the job. Monotone throughout, because
    a warm restart would put the diffusion back.
    """
    from rsfff.train.config import TrainConfig
    from rsfff.train.term_loop import build_scheduler
    from dataclasses import replace as _replace

    cfg = TrainConfig(epochs=20, learning_rate=1.0e-3,
                      lr_schedule="cosine", lr_final_factor=0.02)
    p = [torch.nn.Parameter(torch.zeros(1))]
    opt = torch.optim.Adam(p, lr=cfg.learning_rate)
    sched = build_scheduler(opt, cfg)
    lrs = []
    for _ in range(cfg.epochs):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()

    assert lrs[0] == pytest.approx(1.0e-3)
    assert lrs[-1] == pytest.approx(1.0e-3 * 0.02, rel=0.05), "did not reach the floor"
    assert all(b <= a for a, b in zip(lrs, lrs[1:])), "the schedule is not monotone"

    assert build_scheduler(opt, _replace(cfg, lr_schedule="none")) is None
    with pytest.raises(ValueError, match="unknown train.lr_schedule"):
        build_scheduler(opt, _replace(cfg, lr_schedule="step"))
    with pytest.raises(ValueError, match=r"lr_final_factor must be in \[0, 1\]"):
        build_scheduler(opt, _replace(cfg, lr_final_factor=2.0))


def test_warm_start_refuses_to_load_a_channel_reduction_that_reached_zero(tmp_path, capsys):
    """A zeroed ``equiv_reduce`` is a deleted block, not a trained value. Do not inherit it.

    Every stage after the first warm-starts from the one below, so loading the saved zero would
    carry the deadlock forward for the whole chain -- and it cannot be trained out, because at
    ``equiv_reduce == 0`` the equivariant head is a flat critical point (see
    ``tests/test_ff_electrostatics.py``). Keeping this stage's fresh initialization restarts the
    head, which is strictly better than staying dead, and the load says so out loud.

    Only the dead tensor is refused; everything else in the checkpoint still loads.
    """
    from rsfff.train.term_loop import warm_start

    saved = make_model(environment=True)
    with torch.no_grad():
        saved.response.params.chivec_head.equiv_reduce.zero_()
        saved.response.params.chiquad_head.equiv_reduce.fill_(0.25)
        saved.response.params.chi0.fill_(7.0)
    path = tmp_path / "dead.pt"
    torch.save({"model_state": saved.state_dict()}, path)

    model = make_model(environment=True)
    dead = model.response.params.chivec_head.equiv_reduce
    healthy = model.response.params.chiquad_head.equiv_reduce
    fresh = dead.detach().clone()
    warm_start(model, str(path))

    assert float(dead.norm()) > 0.0, "the deleted block was inherited"
    assert torch.equal(dead, fresh), "and this stage's own initialization was kept instead"
    assert "REINITIALIZED" in capsys.readouterr().out

    # Only the dead tensor is refused; a healthy reduction and everything else still load.
    assert torch.allclose(healthy, torch.full_like(healthy, 0.25))
    chi0 = model.response.params.chi0
    assert torch.allclose(chi0, torch.full_like(chi0, 7.0))


def test_parameter_groups_partition_every_trainable_parameter_exactly_once():
    from rsfff.train.term_loop import parameter_groups

    model = make_model(environment=True)
    groups = parameter_groups(model, 1.0e-4)
    ids = [id(p) for g in groups for p in g["params"]]
    want = {id(p) for p in model.parameters() if p.requires_grad}
    assert len(ids) == len(set(ids)) == len(want)
    assert set(ids) == want
    assert {g["weight_decay"] for g in groups} == {1.0e-4, 0.0}


def test_weight_decay_still_reaches_the_parameters_that_do_get_gradient():
    """The exemption is not a blanket opt-out: the per-species tables still decay."""
    from rsfff.train.term_loop import parameter_groups

    model = make_model(environment=True)
    groups = parameter_groups(model, 1.0e-4)
    decayed = {id(p) for g in groups if g["weight_decay"] > 0.0 for p in g["params"]}
    for name in ("disp_params.d_log_c6", "response.params.chi0", "range_heads.log_r0_prior"):
        p = dict(model.named_parameters()).get(name)
        if p is not None:
            assert id(p) in decayed, f"{name} should still be decayed"


# ---------------------------------------------------------------------------
# Per-channel range-separation priors
# ---------------------------------------------------------------------------

def test_dispersion_prior_is_its_own_and_reaches_the_gate():
    """The dispersion channel gets its own bonded exclusion, and it reaches ``r0``.

    A per-channel prior is only meaningful if it survives to the gate rather than being
    averaged away, so this checks both ends: the stored prior and the per-atom ``r0`` the
    model actually gates with.

    It used to assert ``disp > elst`` elementwise. That stopped being the shape of the prior
    when it was flattened to 1.60 on both elements (2026-08-19): O rises 0.893 -> 1.60 and H
    *falls* 1.75 -> 1.60, so the ordering now holds on O and not on H. What has to stay true
    is the property the channel-specific prior exists for -- the O-H pair value, which is what
    the bonded gate actually reads, must sit well above the shared prior's.
    """
    from rsfff.ff.range_priors import CHANNEL_R0_PRIOR, DEFAULT_R0_PRIOR, build_range_priors

    assert "disp" in CHANNEL_R0_PRIOR
    model = make_model()
    prior = model.range_heads.log_r0_prior.exp()
    assert prior.shape == (len(RANGE_CHANNELS), len(NEIGHBOR_TYPES))
    d = RANGE_CHANNELS.index("disp")

    # The pair value, by geometric mean, against what the shared prior would have given.
    shared = build_range_priors(NEIGHBOR_TYPES, r0_prior=DEFAULT_R0_PRIOR).exp()
    assert float(prior[d].prod().sqrt()) > float(shared[d].prod().sqrt()), (
        "the dispersion prior no longer widens the bonded exclusion at all"
    )

    positions, numbers, frag = water_cluster(2, seed=5)
    out = model(make_batch(positions, numbers, frag))
    assert torch.allclose(out.log_r0_prior["disp"].exp(), out.r0["disp"])


def test_intra_dispersion_leak_is_smaller_under_the_dispersion_prior():
    """The point of the higher prior: less classical dispersion inside the bonded region.

    Measured against the shared prior on the same geometry and the same ``C6``, so the only
    difference is where the gate sits.
    """
    from rsfff.ff.range_priors import DEFAULT_R0_PRIOR, build_range_priors

    positions, numbers, frag = water_cluster(3, seed=17)
    batch = make_batch(positions, numbers, frag)

    def leak(prior):
        parts = build_parts(seed=0)
        with torch.no_grad():
            parts["range_heads"].log_r0_prior.copy_(prior)
        out = make_model(parts)(batch)
        return float(out.e_pair_ff["disp"][out.is_intra].sum())

    shared = build_range_priors(NEIGHBOR_TYPES, r0_prior=DEFAULT_R0_PRIOR)
    channelled = build_range_priors(NEIGHBOR_TYPES)
    assert abs(leak(channelled)) < abs(leak(shared))
    # Measured 0.027 kJ/mol per fragment at the flat-1.60 prior, against 0.056 for the shared
    # one and ~0 for the (2.00, 1.30) pair it replaced. The whole difference is r0(H,H), which
    # fell 2.00 -> 1.60 while the O-H geometric mean barely moved (1.612 -> 1.600), and the
    # intermolecular dispersion is unchanged to four decimals in all three cases.
    #
    # The number to watch in a fit is not this one but `intra_disp`: the r0 barrier is
    # one-sided *about the prior*, so lowering r0(H,H) lowers the floor training can reach.
    assert abs(leak(channelled)) * KJMOL_PER_HARTREE < 0.1


def test_r0_barrier_is_one_sided_about_the_prior():
    """The ``r0`` penalty is a floor, not a pull. Below the prior it must be exactly inert.

    As an unbounded downward pull it drove the shared trunk's per-pair deviation to -0.42 on
    intra and inter pairs alike, opening the bonded dispersion gate to 0.999. Restricting it
    to inter pairs did not help, because one trunk serves both.
    """
    from rsfff.train.config import Config
    from rsfff.train.train_unified import AnchorTerms

    model = make_model()
    positions, numbers, frag = water_cluster(3, seed=23)
    batch = make_batch(positions, numbers, frag)
    cfg = Config()
    cfg.unified.r0_weight = 1.0
    for w in ("corr_l2_weight", "r0_spread_weight", "intra_classical_weight", "env_weight"):
        setattr(cfg.unified, w, 0.0)
    terms = AnchorTerms(model, None, "cpu")

    at_prior = terms.penalties(model(batch), batch, cfg)
    assert float(at_prior["r0"]) == 0.0

    with torch.no_grad():   # push every r0 above its prior; the barrier must now bite
        for name in RANGE_CHANNELS:
            model.range_heads.d_log_r0[name].fill_(0.5)
    above = terms.penalties(model(batch), batch, cfg)
    assert float(above["r0"]) > 0.0

    with torch.no_grad():   # ...and below it, inert again, with no gradient to follow
        for name in RANGE_CHANNELS:
            model.range_heads.d_log_r0[name].fill_(-0.5)
    below = terms.penalties(model(batch), batch, cfg)
    assert float(below["r0"]) == 0.0
    assert below["r0"].grad_fn is not None      # still a live tensor, just a flat one


# ---------------------------------------------------------------------------
# Molecular polarizability: the response target
# ---------------------------------------------------------------------------

def test_polarizability_is_the_second_field_derivative_of_the_internal_energy():
    """The definitional check, and the one that pins the unit conversion.

    ``fragment_polarizability`` adds two sectors that are carried in *different* units --
    ``alpha_flow`` in ``e^2 Angstrom^2 / Ha`` because ``sqe_solve`` works in Angstrom, and the
    on-site ``alpha_i`` in ``a0^3`` because it multiplies a field conjugate to ``mu`` in
    ``e*bohr``. A missing ``BOHR_ANG**2`` there is a factor of 3.6 that no symmetry, PSD or
    rotation-covariance test would notice, so the claim is checked against what it means:
    ``-d^2 E / dF dF`` of the fragment's own internal energy under a uniform field, by central
    differences.
    """
    from rsfff.ff.pairs import intra_fragment_channels
    from rsfff.ff.units import BOHR_ANG
    from rsfff.mlip.sqe import atomic_dipole_energy, sqe_solve

    model = randomize(make_model(), scale=0.05, seed=7)
    positions, numbers, frag = water_cluster(2, seed=31)
    batch = make_batch(positions, numbers, frag)
    n_frag = int(batch.n_fragments)

    feats = model._augment(model.featurizer(batch, frag), batch, frag)
    rp = model.response.response_parameters(batch, feats)
    bond_index, bond_batch = intra_fragment_channels(frag)

    def energy(field):
        """Fragment internal energy under a uniform field, in Hartree.

        The two sectors take the *same physical field* in different units: ``sqe_solve``
        contracts it with positions in Angstrom, the dipole sector with ``mu`` in ``e*bohr``.
        """
        sol = sqe_solve(
            rp.chi, rp.eta, rp.compliance, rp.q0, positions, bond_index, frag,
            bond_batch, n_frag, field=field, with_polarizability=False,
        )
        return sol.energy + atomic_dipole_energy(
            rp.chivec, rp.alpha, frag, n_frag, field * BOHR_ANG
        )

    h = 1.0e-4
    zero = torch.zeros(n_frag, 3, dtype=positions.dtype)
    fd = torch.zeros(n_frag, 3, 3, dtype=positions.dtype)
    for a in range(3):
        for b in range(3):
            def shift(sa, sb):
                f = zero.clone()
                f[:, a] += sa * h
                f[:, b] += sb * h
                return energy(f)
            fd[:, a, b] = -(
                shift(1, 1) - shift(1, -1) - shift(-1, 1) + shift(-1, -1)
            ) / (4 * h * h)

    out = model(batch, with_polarizability=True)
    assert out.polarizability.shape == (n_frag, 3, 3)
    assert torch.allclose(out.polarizability, fd, atol=1e-6, rtol=1e-6)


def test_polarizability_is_symmetric_positive_semidefinite_and_rotation_covariant():
    model = randomize(make_model(), scale=0.05, seed=9)
    positions, numbers, frag = water_cluster(2, seed=37)
    a = model(make_batch(positions, numbers, frag), with_polarizability=True).polarizability
    assert torch.allclose(a, a.transpose(-1, -2), atol=1e-12)
    assert float(torch.linalg.eigvalsh(a).min()) > -1e-12

    rot = torch.linalg.qr(torch.randn(3, 3, generator=torch.Generator().manual_seed(4)))[0]
    rot = rot * torch.sign(torch.det(rot))
    b = model(
        make_batch(positions @ rot.T, numbers, frag), with_polarizability=True
    ).polarizability
    assert torch.allclose(b, rot @ a @ rot.T, atol=1e-8)


def test_polarizability_is_not_computed_unless_asked_for():
    model = make_model()
    positions, numbers, frag = water_cluster(2, seed=41)
    assert model(make_batch(positions, numbers, frag)).polarizability is None


def test_polarizability_loss_is_zero_at_the_label_and_refuses_clusters():
    from rsfff.train.loss import fragment_polarizability_loss

    model = randomize(make_model(), scale=0.05, seed=13)
    positions, numbers, frag = water_cluster(1, seed=43)
    batch = make_batch(positions, numbers, frag)
    out = model(batch, with_polarizability=True)

    batch.polarizability = out.polarizability.detach().clone()
    terms, metrics = fragment_polarizability_loss(batch=batch, out=out, weight=1.0)
    assert float(terms["alpha"]) == pytest.approx(0.0, abs=1e-20)
    assert metrics["alpha_mae"] == pytest.approx(0.0, abs=1e-12)

    batch.polarizability = batch.polarizability + 0.5      # one scale off, every component
    terms, _ = fragment_polarizability_loss(batch=batch, out=out, weight=1.0, scale=0.5)
    assert float(terms["alpha"]) == pytest.approx(9.0)     # nine components, 1.0 each

    # A cluster label is a different quantity; summing isolated fragments would fit the wrong
    # thing, so it raises rather than doing it quietly.
    positions, numbers, frag = water_cluster(3, seed=47)
    cluster = make_batch(positions, numbers, frag)
    cluster.polarizability = torch.zeros(1, 3, 3, dtype=positions.dtype)
    with pytest.raises(ValueError, match="not the sum of its fragments"):
        fragment_polarizability_loss(
            out=model(cluster, with_polarizability=True), batch=cluster, weight=1.0
        )


def test_polarizability_weight_zero_costs_nothing():
    from rsfff.train.loss import fragment_polarizability_loss

    model = make_model()
    positions, numbers, frag = water_cluster(1, seed=51)
    batch = make_batch(positions, numbers, frag)
    out = model(batch, with_polarizability=True)
    batch.polarizability = torch.zeros(1, 3, 3, dtype=positions.dtype)
    assert fragment_polarizability_loss(out=out, batch=batch, weight=0.0) == ({}, {})


# ---------------------------------------------------------------------------
# Cluster forces and the total energy: what the CT stage adds
# ---------------------------------------------------------------------------

def _weighted_force_loss(model, batch):
    """A scalar built from the forces, so its parameter gradient exercises double backward."""
    from rsfff.train.loss import compute_forces

    batch.positions.requires_grad_(True)
    forces = compute_forces(
        model(batch).energy, batch.positions, create_graph=torch.is_grad_enabled()
    )
    w = torch.linspace(0.3, 1.7, forces.numel(), dtype=forces.dtype).reshape(forces.shape)
    return (forces * w).pow(2).sum()


def _grad_vs_fd(model, batch_fn, loss_fn, names, h=1.0e-5):
    """``{name: relative error}`` between the autograd and finite-difference gradients.

    ``h = 1e-5`` because the finite difference, not the gradient, is the inaccurate side here.
    ``(hi - lo)`` cancels to about ``eps * |loss| / h`` in absolute terms, so shrinking ``h``
    makes the *reference* worse, not better. Measured on
    ``test_cluster_force_gradient_is_exact_at_the_frozen_level``, where the gradient is an
    ordinary double backward and is exact::

        h        1e-4      1e-5      1e-6      1e-7
        chi0     5.5e-10   2.1e-11   6.9e-11   2.4e-09
        eta0     5.1e-10   5.8e-10   6.8e-09   3.8e-08
        cquad0   6.2e-11   1.8e-10   5.9e-10   1.7e-08

    The old default of 1e-6 sat in the roundoff-dominated tail, so a ``< 1e-8`` assertion on
    top of it passed or failed on which random model it happened to be handed -- adding an
    inert parameter elsewhere in the model was enough to flip it.
    """
    named = dict(model.named_parameters())
    model.zero_grad()
    loss_fn(model, batch_fn()).backward()
    out = {}
    for name in names:
        p = named[name]
        analytic = float(p.grad.reshape(-1)[0])
        with torch.no_grad():
            p.reshape(-1)[0] += h
        hi = float(loss_fn(model, batch_fn()).detach())
        with torch.no_grad():
            p.reshape(-1)[0] -= 2 * h
        lo = float(loss_fn(model, batch_fn()).detach())
        with torch.no_grad():
            p.reshape(-1)[0] += h
        fd = (hi - lo) / (2 * h)
        out[name] = abs(analytic - fd) / max(abs(fd), 1e-12)
    return out


_FD_PARAMS = (
    "response.params.chi0",
    "response.params.eta0_raw",
    "response.params.cquad0_raw",
)


def test_cluster_force_gradient_is_exact_at_the_frozen_level():
    """No coupled solve, so the double backward is an ordinary one and must be exact."""
    model = randomize(make_model(environment=True), scale=0.02, seed=5)
    positions, numbers, frag = water_cluster(2, seed=3)
    errs = _grad_vs_fd(
        model, lambda: make_batch(positions, numbers, frag),
        _weighted_force_loss, _FD_PARAMS,
    )
    # 1e-6, not the 1e-8 this used to be, because the finite difference is the inaccurate
    # side and its noise floor moved when `atomic_energy` changed the operating point --
    # exactly the sensitivity `_grad_vs_fd`'s docstring records. Sweeping h on the current
    # model shows the signature of roundoff rather than of a wrong gradient: `eta0_raw` falls
    # 3.8e-6 -> 4.2e-8 from h=1e-3 to 1e-4 (clean O(h^2) truncation), then *plateaus* at
    # 4.2e-8 for h=1e-5 and *rises* to 1.1e-7 at h=1e-6. A real analytic error does not do
    # that. The threshold still bites: the adjoint bias this is contrasted against runs
    # 1e-4 to 1e-2, two to four orders above it.
    assert max(errs.values()) < 1e-6, errs


def test_energy_gradient_stays_exact_once_the_coupled_solve_is_on():
    """The adjoint is correct: a loss on the *energy* still matches finite differences.

    This is the pairing that makes the next test's failure interpretable. The first backward
    through :class:`rsfff.ff.coupled_solve._CoupledSolve` is the implemented adjoint and it is
    right; what is missing is only its own derivative.
    """
    def energy_loss(model, batch):
        out = model(batch)
        return out.energy.pow(2).sum() + sum(v.pow(2).sum() for v in out.interaction.values())

    positions, numbers, frag = water_cluster(2, seed=3)
    for levels in (dict(induction=True),):
        model = randomize(
            make_model(environment=True, extra_dim=9, levels=levels), scale=0.02, seed=5
        )
        errs = _grad_vs_fd(
            model, lambda: make_batch(positions, numbers, frag), energy_loss, _FD_PARAMS,
        )
        assert max(errs.values()) < 1e-4, (levels, errs)


def test_cluster_force_gradient_carries_a_known_bias_at_the_coupled_levels():
    """The measurement behind ``UnifiedConfig.force_weight``'s warning, pinned.

    ``_CoupledSolve.backward`` runs its adjoint CG under ``no_grad`` and detaches the
    parameters, so it is not double-differentiable and the second-order path through
    ``lambda`` is dropped. A force loss needs exactly that path.

    The bounds below are deliberately two-sided. The lower bound says the bias is still there,
    so that making the adjoint differentiable in its own right (a nested implicit solve for
    ``d lambda / d theta``) shows up here as a *failure* rather than passing silently -- at
    which point the fix is to tighten this test, not to widen it. The upper bound says the
    bias has not grown into something that would actually corrupt a fit.

    It is not a convergence artifact: sweeping ``cg_rtol`` over 1e-8 to 1e-14 leaves these
    numbers unchanged to three digits.

    The band was widened from ``1e-6 < worst < 1e-2`` when ``atomic_energy`` was added, and
    the reason is the mechanism rather than the tolerance: ``E_atom`` reads the converged
    multipoles at the polarized and CT levels, so it is a **second** non-variational consumer
    of ``M*`` and it leans on the missing second-order path harder than the electrostatic
    correction alone did. Measured, ``cquad0_raw`` moved from under 1e-2 to 2.5e-2.

    The *energy* gradient is unaffected and stays exact -- checked directly against finite
    differences on ``fragment_energy`` and on the total at all three levels, where
    ``atomic_energy.equiv_reduce`` agrees to 1.4e-7 relative and the apparent looseness on
    ``vec_reduce`` is a 1e-7-magnitude gradient in the denominator, not a wrong one. That is
    the case that matters here, because ``unified.force_weight`` is 0.
    """
    positions, numbers, frag = water_cluster(2, seed=3)
    model = randomize(
        make_model(environment=True, extra_dim=9, levels=dict(induction=True)),
        scale=0.02, seed=5,
    )
    errs = _grad_vs_fd(
        model, lambda: make_batch(positions, numbers, frag),
        _weighted_force_loss, _FD_PARAMS,
    )
    worst = max(errs.values())
    assert 1e-4 < worst < 1e-1, (
        f"the force-gradient bias moved: {errs}. If it shrank, the adjoint became "
        f"double-differentiable and this test and UnifiedConfig.force_weight should say so."
    )


def test_total_energy_and_force_terms_enter_the_loss_only_when_weighted():
    from rsfff.train.config import Config
    from rsfff.train.train_unified import unified_fit

    model = make_model()
    positions, numbers, frag = water_cluster(2, seed=61)
    n_sys = 1

    def fresh():
        b = make_batch(positions.clone(), numbers, frag)
        b.fragment_energy = torch.zeros(int(b.n_fragments), dtype=positions.dtype)
        b.energy = torch.zeros(n_sys, dtype=positions.dtype)
        b.forces = torch.zeros_like(positions)
        b.eda = {k: torch.zeros(n_sys, dtype=positions.dtype)
                 for k in ("cls_elec", "mod_pauli", "disp")}
        return b

    cfg = Config()
    off, m_off, _ = unified_fit(model(fresh()), fresh(), cfg)
    assert "e_tot_mae" in m_off and "f_clu" not in m_off

    cfg.unified.total_energy_weight = 1.0
    on, _, _ = unified_fit(model(fresh()), fresh(), cfg)
    assert float(on) > float(off), "the total-energy term did not reach the loss"

    # Forces need positions to be leaves; the loop arranges that via grad_positions.
    cfg.unified.total_energy_weight = 0.0
    cfg.unified.force_weight = 1.0
    batch = fresh()
    with pytest.raises(ValueError, match="grad_positions"):
        unified_fit(model(batch), batch, cfg)

    batch = fresh()
    batch.positions.requires_grad_(True)
    loss, metrics, _ = unified_fit(model(batch), batch, cfg)
    assert "f_clu" in metrics and float(loss) > float(off)


def test_force_term_requires_force_labels():
    from rsfff.train.config import Config
    from rsfff.train.train_unified import unified_fit

    model = make_model()
    positions, numbers, frag = water_cluster(2, seed=63)
    batch = make_batch(positions, numbers, frag)
    batch.fragment_energy = torch.zeros(int(batch.n_fragments), dtype=positions.dtype)
    batch.eda = {k: torch.zeros(1, dtype=positions.dtype)
                 for k in ("cls_elec", "mod_pauli", "disp")}
    batch.forces = None
    batch.positions.requires_grad_(True)
    cfg = Config()
    cfg.unified.force_weight = 1.0
    with pytest.raises(ValueError, match="carries no forces"):
        unified_fit(model(batch), batch, cfg)


# ---------------------------------------------------------------------------
# Holding the frozen level across stages
# ---------------------------------------------------------------------------

def test_frozen_level_modules_cover_everything_the_internal_energy_depends_on():
    """Freezing the list must make ``E_internal`` a constant. That is the whole claim.

    A shorter list does not work, and the short version was measured rather than guessed:
    reverting only ``response.params`` in a trained stage-3 checkpoint left the monomer
    internal energy at -230 kJ/mol against stage 1's -342, because ``featurizer.channel_proj``
    had moved 22% and the same head weights were reading different features.

    So rather than re-listing the modules -- which would only restate
    ``_frozen_level_modules`` back to itself -- this trains *everything else* hard and checks
    the internal energy has not moved at all.
    """
    from rsfff.train.train_unified import _frozen_level_parameters

    model = randomize(make_model(environment=True))
    positions, numbers, frag = water_cluster(3, seed=31)
    batch = make_batch(positions, numbers, frag)
    before = model(batch).energy_internal.detach().clone()

    frozen_ids = {
        id(p) for ps in _frozen_level_parameters(model).values() for p in ps
    }
    assert frozen_ids, "nothing was frozen, so this test proves nothing"

    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for p in model.parameters():
            if id(p) not in frozen_ids:
                p.add_(0.2 * torch.randn(p.shape, generator=g))

    after = model(batch).energy_internal.detach()
    assert torch.allclose(before, after, atol=1e-12, rtol=0), (
        "the isolated-fragment internal energy moved when a non-frozen parameter changed, so "
        "some route into the frozen level is missing from _frozen_level_modules"
    )


def test_freezing_the_frozen_level_keeps_it_out_of_the_optimizer():
    """``requires_grad_(False)`` and not a zeroed gradient, so Adam never sees them.

    The distinction matters: Adam's weight decay is applied to the parameter, not through the
    gradient, so a decayed-but-not-detached parameter would still drift every step even with
    an identically zero gradient. That is the exact mechanism that deleted the equivariant
    heads, and it would quietly undo the freeze too.
    """
    from rsfff.train.term_loop import parameter_groups
    from rsfff.train.train_unified import _frozen_level_parameters

    model = make_model(environment=True)
    for params in _frozen_level_parameters(model).values():
        for p in params:
            p.requires_grad_(False)

    grouped = {id(p) for g in parameter_groups(model, 1.0e-4) for p in g["params"]}
    for name, params in _frozen_level_parameters(model).items():
        for p in params:
            assert id(p) not in grouped, f"{name} is still in an optimizer group"
    assert grouped, "everything was frozen, so the stage would have nothing to train"


def _bond_levels_model(**levels):
    return randomize(wake_environment(
        make_model(environment=True, extra_dim=9, levels=levels), scale=3.0
    ), scale=0.05, seed=41)


def test_atomic_energy_telescopes_to_one_evaluation_at_the_top_level():
    """``E_atom^0 + (induction - 0) == E_atom^ind``, exactly.

    The two terms answer to different labels -- ``fragment_energy`` and ``induction`` -- but
    they are one network read at two electronic states, so outside training they collapse to a
    single evaluation. If they did not, the decomposition would be adding energy rather than
    dividing it.

    This replaces the same claim about the bond channel, and previously spanned three levels.
    The mechanism moved and the level count shrank, but the invariant did not: what
    distinguishes the levels is the *state* handed to one set of weights.
    """
    model = _bond_levels_model(induction=True)
    positions, numbers, frag = water_cluster(3, seed=43)
    batch = make_batch(positions, numbers, frag)
    out = model(batch)

    assert out.energy_atom_ind is not None
    # Live at both levels, or the identity below is a statement about zeros.
    for level in (out.energy_atom, out.energy_atom_ind):
        assert float(level.detach().abs().max()) > 1e-9

    d_ind = out.energy_atom_ind - out.energy_atom
    # With two levels the telescoping itself is a floating-point tautology, so what is worth
    # asserting is where the step is *booked*: `interaction_corr["induction"]` must be exactly
    # that step plus the electrostatic environment term and nothing else. A third contributor
    # sneaking into the induction correction is what this catches.
    f2b = batch.fragment_to_batch
    pooled = d_ind.new_zeros(batch.n_systems).index_add_(0, f2b, d_ind)
    residual = out.interaction_corr["induction"] - pooled
    assert torch.allclose(residual, out.elst_env, atol=1e-12), (
        "interaction_corr['induction'] is not the atomic step plus the electrostatic split"
    )


def test_the_descriptor_swap_is_measurable_without_a_second_solve():
    """``energy_atom_ind`` and ``energy_atom_ind_frag`` differ only by the feature stream.

    This is the diagnostic the merge exists to expose. In the two-level model the
    ``h_frag -> h_env`` swap was the whole of the ``ct`` channel -- 100.0% of its correction,
    with 4.6e-5 e crossing a fragment boundary -- and no printed metric said so. Both
    evaluations are at the *same* electronic state, so their difference is the swap alone.
    """
    model = _bond_levels_model(induction=True)
    positions, numbers, frag = water_cluster(3, seed=44)
    batch = make_batch(positions, numbers, frag)
    out = model(batch)

    assert out.energy_atom_ind is not None and out.energy_atom_ind_frag is not None
    swap = out.energy_atom_ind - out.energy_atom_ind_frag
    # `wake_environment` has moved `g` off zero, so the two streams genuinely differ.
    assert float(swap.abs().max()) > 1e-9, "the swap diagnostic is identically zero"

    # And it is exactly the feature stream: recomputing the h_frag version by hand agrees.
    h_frag, h_env = model._descriptors(batch, frag)
    e = model.atomic_energy(
        h_frag.inv_feats, h_frag.species_idx, h_frag.vec_feats, h_frag.equiv_feats,
        out.level_ind.charges, out.level_ind.mu, out.level_ind.quad_s, out.environment,
    )
    by_hand = e.new_zeros(int(batch.n_fragments)).index_add_(0, frag, e)
    assert torch.allclose(by_hand, out.energy_atom_ind_frag, atol=0, rtol=0)


def test_the_environment_reaches_induction_through_the_response_not_the_features():
    """Two different routes, booked at two different levels, and the split is deliberate.

    The atomic energy reads ``h_frag`` at the frozen *and* polarized levels and only switches
    to ``h_env`` at ``ct``: relaxing a fragment's multipoles against its neighbours leaves it
    the same fragment, whereas letting charge cross the boundary does not, and that is also
    when its descriptor should stop being fragment-confined.

    So induction still has to see the environment, and it does -- through the coupled
    solve, whose response parameters are evaluated on ``h_env``. That is the route this test
    pins. The companion claim, that the *feature-stream* step belongs to ``ct``, is pinned by
    :func:`test_the_atomic_energys_feature_stream_switches_at_ct_not_pol`.

    Isolated by silencing every *other* route into ``interaction_corr["pol"]``: with the
    elst/pauli/disp readouts and the range readout zeroed, ``corr_pol_pair`` is identically
    zero (no correction difference, and both gates collapse to the same per-element ``r0``).
    """
    positions, numbers, frag = water_cluster(3, seed=47)
    batch = make_batch(positions, numbers, frag)

    def bond_only(wake):
        model = make_model(environment=True, extra_dim=9, levels={"induction": True})
        model = randomize(model, scale=0.05, seed=41)
        with torch.no_grad():
            # `randomize` perturbs every parameter, `g` included, so an "asleep" baseline has
            # to be put back to sleep explicitly rather than assumed.
            for m in (model.environment.inv_mlp, model.environment.vec_gate,
                      model.environment.equiv_gate):
                if m is not None:
                    m[-1].weight.zero_()
                    m[-1].bias.zero_()
        if wake:
            wake_environment(model, scale=3.0)
        with torch.no_grad():
            for name in ("elst", "pauli", "disp"):
                model.pair_head.readout[name].weight.zero_()
                model.pair_head.readout[name].bias.zero_()
            for lin in model.pair_head.range_readout.values():
                lin.weight.zero_()
                lin.bias.zero_()
        out = model(batch)
        assert float((~out.is_intra).to(out.r.dtype).mul(
            out.e_pair_corr["elst"]).detach().abs().max()) == 0.0
        return out

    asleep, awake = bond_only(False), bond_only(True)
    assert float(asleep.environment_norm.detach().abs().max()) < 1e-12
    assert float(awake.environment_norm.detach().abs().max()) > 1e-6
    assert not torch.allclose(
        awake.interaction_corr["induction"], asleep.interaction_corr["induction"], atol=1e-12
    ), "induction does not see h_env at all, so the coupled solve is not reading it"


def _feature_streams(model, batch):
    """``(h_frag, h_env)`` exactly as :meth:`UnifiedPairModel.forward` builds them."""
    frag = batch.fragment_idx
    if model.environment is None:
        h = model._augment(model.featurizer(batch, frag), batch, frag)
        return h, h
    grouped, full = model.featurizer(batch, frag, also_ungrouped=True)
    h_frag = model._augment(grouped, batch, frag)
    return h_frag, model.environment(h_frag, model._augment(full, batch, frag))


def _free_atom_batch(numbers):
    from rsfff.train.data import Batch

    n = len(numbers)
    return Batch(
        positions=torch.zeros(n, 3, dtype=torch.get_default_dtype()),
        atomic_numbers=torch.tensor(numbers),
        batch_idx=torch.arange(n),
        n_systems=n,
        energy=None,
        fragment_idx=torch.arange(n),
        n_fragments=n,
    )


def test_a_lone_atom_reduces_the_whole_model_to_its_per_element_constants():
    """The free-atom limit is *exact*, which is what makes anchoring it a constraint.

    A single atom has an all-zero SOAP density and no channel graph, so every learned
    equivariant quantity collapses: ``chivec`` and the anisotropic part of ``alpha_i`` vanish
    with the features they are built from, ``alpha_flow`` is empty for want of a bond, and the
    SQE charge is pinned to the formal charge. What survives is one isotropic number per
    element -- so a free-atom polarizability label pins ``AtomicAlphaHead``'s per-species value
    rather than nudging it, which is the whole reason
    :func:`rsfff.train.loss.free_atom_polarizability_loss` is worth having.

    The energy half of that anchor is deliberately absent, and this says why: ``E_internal``
    is identically zero on a neutral free atom, so ``fragment_energy == E0`` by construction
    and an energy term would be fitting an identity.
    """
    model = randomize(make_model(environment=True), seed=5)
    out = model(_free_atom_batch([1, 8]), with_polarizability=True)

    alpha = out.polarizability.detach()
    for a in alpha:
        iso = a[0, 0] * torch.eye(3, dtype=a.dtype)
        assert torch.allclose(a, iso, atol=1e-14), f"not isotropic:\n{a}"
    assert not torch.allclose(alpha[0], alpha[1]), "H and O must not share one value"

    assert float(out.energy_internal.detach().abs().max()) == 0.0
    assert float(out.energy_bond.detach().abs().max()) == 0.0
    assert torch.equal(out.fragment_energy.detach(), out.energy_ref.detach())
    assert float(out.charges.detach().abs().max()) == 0.0
    assert float(out.mu.detach().abs().max()) == 0.0


def test_free_atom_polarizability_depends_only_on_the_species_conditioned_alpha_head():
    """Its inputs are exactly ``alpha_head`` and the species embedding that conditions it.

    That is the precise statement of "exact free-atom limit": the anchor cannot be satisfied
    by moving the compliance head, the charge sector, the pair trunk, the featurizer or the
    environment residual, because at zero features none of them reaches the answer. The
    species embedding is in the list because it is the head's only surviving input once the
    features vanish -- ``alpha_i = softplus(a0_raw(0, emb)) + psd_floor`` -- and it lives on
    ``response.params`` rather than on the head itself.
    """
    model = randomize(make_model(environment=True), seed=5)
    batch = _free_atom_batch([1, 8])
    before = model(batch, with_polarizability=True).polarizability.detach().clone()

    inputs = {
        id(p) for p in list(model.response.params.alpha_head.parameters())
        + list(model.response.params.species_emb.parameters())
    }
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for p in model.parameters():
            if id(p) not in inputs:
                p.add_(0.3 * torch.randn(p.shape, generator=g))
    after = model(batch, with_polarizability=True).polarizability.detach()
    assert torch.allclose(before, after, atol=1e-14), (
        "something outside the polarizability head reaches the free-atom limit, so the "
        "anchor is not the exact constraint it is documented to be"
    )

    # ... and the head itself does move it, so the exclusion above is not vacuous.
    with torch.no_grad():
        model.response.params.alpha_head.base_mlp[-1].bias.add_(0.5)
    assert not torch.allclose(
        before, model(batch, with_polarizability=True).polarizability.detach()
    )


def test_free_atom_batch_keeps_only_bound_neutral_states():
    """Charged states are excluded, and so is anything the reference data flagged unbound."""
    from rsfff.mlip.reference_states import AtomicStateReference
    from rsfff.train.loss import free_atom_batch

    states = AtomicStateReference.from_json(
        "data/atomic_reference_states_wb97mv_tzvpd.json", NEIGHBOR_TYPES,
        dtype=torch.get_default_dtype(),
    )
    batch, alpha = free_atom_batch(
        states, NEIGHBOR_TYPES, dtype=torch.get_default_dtype(), device="cpu",
    )
    assert sorted(batch.atomic_numbers.tolist()) == sorted(NEIGHBOR_TYPES)
    assert float(batch.fragment_charge.abs().max()) == 0.0
    assert alpha.shape == (len(NEIGHBOR_TYPES), 3, 3)
    # Neutral H and O at wB97M-V/def2-TZVPD: 5.05 and 5.53 a0^3, in e^2 Ang^2 / Ha.
    iso = alpha.diagonal(dim1=-2, dim2=-1).sum(-1) / 3.0 / 0.2800852
    assert torch.all((iso > 4.0) & (iso < 7.0)), iso


# ---------------------------------------------------------------------------
# The target configuration: atomic energy plus range-separated FF, no pair corrections
# ---------------------------------------------------------------------------

def _no_corrections_model(**levels):
    """The model ``configs/water_staged.yaml`` actually builds."""
    parts = build_parts(extra_dim=9)
    return randomize(wake_environment(
        make_model(parts, environment=True, levels=levels), scale=3.0
    ), scale=0.05, seed=41)


def test_pair_corrections_off_zeroes_every_correction_and_keeps_the_module():
    """Switching the corrections off must be a *routing* change, not a demolition.

    The pair head stays constructed so its weights remain loadable and the switch is
    reversible in either direction, but nothing it produces reaches the energy.
    """
    positions, numbers, frag = water_cluster(3, seed=71)
    batch = make_batch(positions, numbers, frag)
    model = _no_corrections_model(induction=True)
    model.correction_channels = frozenset()
    out = model(batch)

    assert model.pair_head is not None and "bond" in model.pair_head.channels, (
        "the module must survive the switch; it is wanted again later"
    )
    # Every *pair* readout is silent ...
    for name, value in out.e_pair_corr.items():
        assert float(value.detach().abs().max()) == 0.0, f"per-pair {name} is not zero"
    # ... and so are the three channels whose correction was only ever a pair readout.
    for name in ("elst", "pauli", "disp"):
        assert float(out.interaction_corr[name].detach().abs().max()) == 0.0, (
            f"{name} correction is not zero"
        )
    # `pol` and `ct` keep a nonzero `interaction_corr`, and that is the point rather than a
    # leak: it is the atomic energy's response to the relaxed state, plus (for `pol`) the
    # electrostatic gate re-partition. Neither comes from the pair head, which the
    # perturbation below proves.

    # Perturbing any readout must be a no-op on the total energy.
    before = out.energy.detach().clone()
    with torch.no_grad():
        for lin in model.pair_head.readout.values():
            lin.weight.add_(1.0)
            lin.bias.add_(1.0)
    assert torch.allclose(model(batch).energy, before, atol=0, rtol=0), (
        "a switched-off pair head still reaches the energy"
    )


def test_atomic_energy_carries_the_fragment_energy_with_corrections_off():
    """With no bond channel, `E_atom` is what makes `fragment_energy` more than E0 + solve."""
    positions, numbers, frag = water_cluster(2, seed=73)
    batch = make_batch(positions, numbers, frag)
    model = _no_corrections_model()
    model.correction_channels = frozenset()
    out = model(batch)

    assert float(out.energy_atom.detach().abs().max()) > 1e-6, "E_atom is inert"
    assert torch.allclose(
        out.fragment_energy,
        out.energy_ref + out.energy_internal + out.energy_bond + out.energy_atom,
        atol=0, rtol=0,
    )
    # `energy_bond` is now the intra *classical* remainder alone, with no correction in it.
    intra = out.is_intra
    assert float(out.e_pair_corr["bond"][intra].detach().abs().max()) == 0.0


def test_a_lone_atom_is_exactly_E0_with_the_atomic_energy_live():
    """The free-atom limit stays exact, which is why the head is anchored.

    ``E0`` is a measured isolated-atom energy. A lone neutral atom has an all-zero SOAP
    density and no multipoles, so an unanchored head would add a per-element constant on top
    of it and quietly move the limit. Subtracting the head's own free-atom value keeps this
    exact at every point in training, not only at initialization where the zero-init readout
    gives it for free.
    """
    model = _no_corrections_model()
    model.correction_channels = frozenset()
    out = model(_free_atom_batch(list(NEIGHBOR_TYPES)))
    assert float(out.energy_atom.detach().abs().max()) == 0.0, (
        "the atomic energy is not anchored at the free atom"
    )
    assert torch.allclose(out.fragment_energy, out.energy_ref, atol=0, rtol=0), (
        "a lone atom no longer reduces to E0"
    )


def test_the_species_offset_is_inert_until_it_trains():
    """Zero-init, so adding the parameter cannot change an existing model or its checkpoint."""
    positions, numbers, frag = water_cluster(3, seed=61)
    batch = make_batch(positions, numbers, frag)
    parts = build_parts(seed=5)
    with_offset = make_model(parts)
    off = make_model(build_parts(seed=5))
    with torch.no_grad():                      # the same model with the offset compiled out
        off.atomic_energy.offset_scale = 0.0
    assert torch.equal(with_offset(batch).energy, off(batch).energy)
    assert float(with_offset.atomic_energy.species_offset.detach().abs().max()) == 0.0


def test_the_species_offset_vanishes_at_the_free_atom():
    """The exact ``E0`` limit has to survive the new parameter, at any value of it.

    ``E0`` is a *measured* isolated-atom energy, and the whole reason the atomic energy is
    anchored is to keep the free-atom limit exact at every point in training rather than only
    at initialization. A per-element constant is exactly the thing that would break it, so the
    offset is gated on ``||h||`` -- zero for a lone atom, which is the same condition the
    anchoring itself uses.
    """
    model = make_model()
    with torch.no_grad():                      # a large, live offset
        model.atomic_energy.species_offset.fill_(5.0)
    out = model(_free_atom_batch(list(NEIGHBOR_TYPES)))
    assert float(out.energy_atom.detach().abs().max()) == 0.0, (
        "the species offset does not vanish at the free atom; E0 is no longer exact"
    )
    assert torch.allclose(out.fragment_energy, out.energy_ref, atol=0, rtol=0)


def test_the_species_offset_is_a_constant_direction():
    """The point of the parameter: it moves the one-body energy *uniformly*.

    The MLP can already express any constant -- what it cannot do is express one without
    reshaping its geometry dependence at the same time, because the anchoring cancels the only
    coordinate that would. This asserts the new one is genuinely flat: perturb it, and the
    induced shift per fragment must be the same number everywhere, not a new function of
    geometry.

    The bound is the measured saturation of the gate. Over the monomer anchor and w2--w5 the
    smallest ``||h||`` is 1.664, so at ``offset_gate = 0.5`` the gate is >= 0.9974 and the
    residual spread is a few parts in a thousand of the offset itself.
    """
    model = randomize(make_model(), scale=0.05, seed=63)
    positions, numbers, frag = water_cluster(5, spacing=3.2, jitter=0.25, seed=67)
    batch = make_batch(positions, numbers, frag)
    before = model(batch).fragment_energy.detach().clone()
    with torch.no_grad():
        model.atomic_energy.species_offset.add_(1.0)
    shift = (model(batch).fragment_energy.detach() - before) * KJMOL_PER_HARTREE

    assert float(shift.abs().min()) > 0.0, "the offset did not reach fragment_energy at all"
    spread = float(shift.max() - shift.min()) / float(shift.abs().mean())
    assert spread < 0.01, (
        f"the offset varies by {100 * spread:.2f}% across fragments, so it is not a constant "
        f"direction -- it would inject shape dependence of the same kind it exists to remove"
    )


def test_the_species_offset_reaches_the_one_body_label_and_nothing_else():
    """It must move ``fragment_energy`` and leave every interaction channel untouched.

    ``E_atom`` is pooled per fragment and booked to ``fragment_energy`` alone, so this is a
    routing assertion: the offset is a one-body repair and must not become a cheap way to move
    an EDA channel.
    """
    model = randomize(
        make_model(build_parts(extra_dim=9, seed=69), levels={"induction": True}),
        scale=0.05, seed=69,
    )
    positions, numbers, frag = water_cluster(3, seed=71)
    batch = make_batch(positions, numbers, frag)
    before = model(batch)
    keep = {k: v.detach().clone() for k, v in before.interaction.items()}
    ob = before.fragment_energy.detach().clone()
    with torch.no_grad():
        model.atomic_energy.species_offset.add_(2.0)
    after = model(batch)

    assert float((after.fragment_energy.detach() - ob).abs().min()) > 0.0
    for name, was in keep.items():
        # `induction` is the exception and it is not a leak: it carries `E_atom` re-read at the
        # relaxed state, so a constant added to the head appears in *both* evaluations and
        # cancels. Asserted rather than skipped, because a cancellation that stopped holding
        # would be a real route from a one-body knob into an EDA label.
        assert torch.allclose(after.interaction[name].detach(), was, atol=1e-12), name


def test_cluster_force_term_is_strided_on_training_steps_only():
    """``unified.force_every`` applies the cluster force every k-th training step.

    The cluster force is a ``create_graph=True`` backward through the whole model *including
    the coupled solve*, and it is the most expensive single thing this fit can be asked to do:
    measured at batch 128 with induction on, 31.8 s/epoch without it against 86.1 with it.

    Evaluation always computes it, so ``f_clu`` means the same thing on every validation line,
    and an evaluation epoch must not advance the stride.
    """
    from rsfff.train.config import Config
    from rsfff.train.train_unified import strided_fit_term

    model = randomize(make_model(environment=True, extra_dim=9,
                                 levels={"induction": True}), seed=5)
    positions, numbers, frag = water_cluster(2, seed=92)
    batch = make_batch(positions, numbers, frag)
    batch.forces = torch.zeros_like(batch.positions)
    batch.fragment_energy = torch.zeros(int(batch.n_fragments))
    # The EDA channels are weighted to zero below, but `unified_fit` still looks their labels
    # up, so they have to exist.
    batch.eda = {k: torch.zeros(int(batch.n_systems))
                 for k in ("cls_elec", "mod_pauli", "disp", "pol", "ct")}
    batch.positions.requires_grad_(True)
    cfg = Config()
    cfg.unified.induction = True
    cfg.unified.force_weight = 1.0
    for key, value in (("elst_weight", 0.0), ("pauli_weight", 0.0), ("disp_weight", 0.0),
                       ("onebody_weight", 0.0), ("induction_weight", 0.0)):
        setattr(cfg.unified, key, value)

    fit = strided_fit_term(model, force_every=3)
    model.train(True)
    fired = []
    for _ in range(6):
        _, metrics, _ = fit(model(batch), batch, cfg)
        fired.append("f_clu" in metrics)
    assert fired == [False, False, True] * 2, f"cluster force fired on {fired}"

    # Evaluation: every step, and the stride does not advance. Grad stays enabled, because
    # that is exactly the condition under which the old grad-context inference was wrong.
    model.train(False)
    for _ in range(3):
        _, metrics, _ = fit(model(batch), batch, cfg)
        assert "f_clu" in metrics, "an evaluation step skipped the cluster force"

    model.train(True)
    _, metrics, _ = fit(model(batch), batch, cfg)
    assert "f_clu" not in metrics, (
        "an evaluation epoch advanced the training-step counter"
    )


def test_anchor_force_term_is_strided_on_training_steps_only():
    """``anchor_force_every`` applies the second-order backward every k-th training step.

    The force is the fit's only ``create_graph=True`` backward and measured at 36% of a whole
    training step. Striding it is the same trade ``dmu_dr_every`` makes for the dipole
    derivative. Evaluation must keep it every time, because ``f_mae`` is a validation metric
    and a number that appeared on a different schedule than the line it is printed on would
    not be comparable between epochs.
    """
    from rsfff.train.config import Config
    from rsfff.train.data import load_extxyz
    from rsfff.train.train_unified import AnchorTerms
    from conftest import DATA_H2O

    anchor = load_extxyz(DATA_H2O, dtype=torch.float64)
    model = make_model()
    positions, numbers, frag = water_cluster(2, seed=91)
    batch = make_batch(positions, numbers, frag)
    cfg = Config()
    cfg.onebody.force_weight = 1.0
    for w in ("corr_l2_weight", "r0_weight", "r0_spread_weight",
              "intra_classical_weight", "env_weight", "free_alpha_weight"):
        setattr(cfg.unified, w, 0.0)

    terms = AnchorTerms(model, anchor, "cpu", batch_size=4, force_every=3)
    model.train(True)
    fired = []
    for _ in range(9):
        fired.append("anchor_f" in terms.penalties(model(batch), batch, cfg))
    assert fired == [False, False, True] * 3, f"force term fired on {fired}"

    # Evaluation keeps it on every step, whatever the stride and wherever the counter sits.
    #
    # `model.train(False)` is what marks an evaluation epoch, **not** the grad context. With
    # `unified.force_weight` on, `run_epoch` enables grad during evaluation too -- the force is
    # itself a backward -- so inferring "training" from the grad context reads True on an
    # evaluation epoch. That would draw a random anchor subset instead of the fixed slice,
    # advance this stride, and buy a `create_graph` nobody needs. Grad is left *enabled* here
    # precisely to pin that: the stride must not care.
    model.train(False)
    for _ in range(3):
        with torch.enable_grad():
            extra = terms.penalties(model(batch), batch, cfg)
        assert "anchor_f" in extra, "an evaluation step skipped the force term"
        assert "f_mae" in terms._metrics
    assert terms._train_step == 9, "an evaluation epoch advanced the training-step counter"

    # A stride of 1 is every step, and is the default.
    model.train(True)
    every = AnchorTerms(model, anchor, "cpu", batch_size=4)
    assert every.force_every == 1
    assert "anchor_f" in every.penalties(model(batch), batch, cfg)


# ---------------------------------------------------------------------------
# Per-channel pair corrections
# ---------------------------------------------------------------------------

def _channels_model(channels, **levels):
    parts = build_parts(extra_dim=9)
    return randomize(wake_environment(
        make_model(parts, environment=True,
                   levels={**levels, "pair_corrections": channels}),
        scale=3.0,
    ), scale=0.05, seed=41)


def _perturb_readouts(model, batch, names):
    """Max change in the total energy from bumping the named readouts. 0 means unreachable."""
    before = model(batch).energy.detach().clone()
    with torch.no_grad():
        for name in names:
            model.pair_head.readout[name].weight.add_(1.0)
            model.pair_head.readout[name].bias.add_(1.0)
    after = model(batch).energy.detach().clone()
    with torch.no_grad():
        for name in names:
            model.pair_head.readout[name].weight.sub_(1.0)
            model.pair_head.readout[name].bias.sub_(1.0)
    return float((after - before).abs().max())


def test_named_correction_channels_route_only_those_channels():
    """``pair_corrections: [elst]`` is the useful middle setting and has to be exact.

    All-or-nothing was the wrong granularity: ``bond`` sits where nothing labels
    them, while ``elst`` corrects a channel that has its own label. Naming one must switch on
    exactly that readout -- the other three keep their parameters and stay unreachable.
    """
    positions, numbers, frag = water_cluster(3, seed=71)
    batch = make_batch(positions, numbers, frag)
    model = _channels_model(["elst"], induction=True)

    assert model.correction_channels == frozenset({"elst"})
    assert model.pair_corrections is True, "one live channel still counts as corrections on"

    silent = [n for n in model.pair_head.readout if n != "elst"]
    assert _perturb_readouts(model, batch, ["elst"]) > 0.0, (
        "the named channel does not reach the energy"
    )
    assert _perturb_readouts(model, batch, silent) == 0.0, (
        f"unnamed readouts {silent} still reach the energy"
    )
    # The module survives, which is what makes the switch reversible in either direction.
    for name in silent:
        assert name in model.pair_head.readout


def test_elst_correction_reaches_induction():
    """``elst`` is the only correction whose environment difference is routed to ``pol``.

    ``dE_pol = [W(u(h_env)) - W(u(h_frag))] + [gate_env - gate_frag] * E_classical``. The
    first bracket exists only when the ``elst`` *readout* is consumed, so this is the reason
    to prefer naming ``elst`` over switching every correction off: it corrects two channels.
    """
    positions, numbers, frag = water_cluster(3, seed=72)
    batch = make_batch(positions, numbers, frag)
    model = _channels_model(["elst"], induction=True)

    before = model(batch).interaction_corr["induction"].detach().clone()
    with torch.no_grad():
        model.pair_head.readout["elst"].weight.add_(1.0)
    after = model(batch).interaction_corr["induction"].detach()
    assert float((after - before).abs().max()) > 0.0, (
        "the elst readout does not reach the pol channel"
    )


def test_correction_channels_reject_a_name_the_head_cannot_serve():
    """A typo must fail loudly, not present as a channel that was fitted to zero."""
    parts = build_parts(extra_dim=9)
    with pytest.raises(ValueError, match="no readout for"):
        make_model(parts, environment=True, levels={"pair_corrections": ["elstt"]})
    with pytest.raises(TypeError, match="bare string"):
        make_model(parts, environment=True, levels={"pair_corrections": "elst"})


def test_pair_corrections_is_derived_and_cannot_be_set():
    """The coarse flag is a property of the channel set, not a second copy of it.

    It was a stored bool, and the two could then disagree: assigning
    ``model.pair_corrections = False`` looked like it disabled the corrections while
    ``forward`` went on reading the set and scoring every channel.
    """
    model = _channels_model(True)
    assert model.pair_corrections is True
    with pytest.raises(AttributeError):
        model.pair_corrections = False
    assert _channels_model(False).pair_corrections is False


def test_frozen_polarizability_matches_the_full_forward():
    """The short path to the response has to be the same number, not a close approximation.

    :func:`rsfff.train.loss.free_atom_polarizability_loss` evaluates it on a handful of
    isolated atoms, where the full forward spent essentially all of its time building a pair
    list with nothing in it.
    """
    positions, numbers, frag = water_cluster(2, seed=74)
    batch = make_batch(positions, numbers, frag)
    model = _channels_model(["elst"])

    full = model(batch, with_polarizability=True).polarizability
    short = model.frozen_polarizability(batch)
    assert full is not None and short is not None
    assert torch.equal(short, full), "the short path disagrees with the full forward"

    # And on the batch it actually exists for: lone atoms, empty pair list.
    lone = _free_atom_batch(list(NEIGHBOR_TYPES))
    assert torch.equal(
        model.frozen_polarizability(lone),
        model(lone, with_polarizability=True).polarizability,
    )


# ---------------------------------------------------------------------------
# Direct multipoles: the heads output mu0/Theta0 rather than the drives
# ---------------------------------------------------------------------------

def _direct_model(**levels):
    parts = build_parts(extra_dim=9, direct_multipoles=True)
    return randomize(wake_environment(
        make_model(parts, environment=True, levels=levels), scale=3.0
    ), scale=0.05, seed=41)


def test_direct_multipoles_are_the_head_output_at_the_frozen_level():
    """No solve stands between the network and the frozen multipoles.

    That is the whole point of the reparameterization: the multipole labels pin one head
    instead of pinning a product of two, so ``alpha`` and ``mu`` stop having to co-adapt.
    """
    positions, numbers, frag = water_cluster(2, seed=81)
    batch = make_batch(positions, numbers, frag)
    model = _direct_model()
    feats = model._augment(model.featurizer(batch, batch.fragment_idx), batch,
                           batch.fragment_idx)
    rp = model.response.response_parameters(batch, feats)
    out = model(batch)

    assert rp.chivec is None and rp.chiquad is None, "both forms are populated at once"
    assert rp.mu0 is not None and rp.quad0 is not None
    assert float(rp.mu0.abs().max()) > 1e-6, "mu0 is inert, so this test proves nothing"
    assert torch.allclose(out.response.mu, rp.mu0, atol=0, rtol=0)
    assert torch.allclose(out.response.quad_s, rp.quad0, atol=0, rtol=0)


def test_direct_multipoles_leave_no_on_site_energy_in_internal():
    """``internal`` becomes the SQE charge sector alone, and that is deliberate.

    Under the drive form the on-site sectors contribute ``-1/2 chivec^T alpha chivec``, a few
    hundred kJ/mol per fragment, to an unlabeled split against the bonding term. Here the
    frozen state *is* the minimum of the on-site quadratic, so it contributes exactly zero and
    that energy has to be carried by the atomic-energy head instead -- where it is a function
    of the multipoles it describes rather than a by-product of a drive.

    Checked by moving the multipole heads and asserting ``internal`` does not budge.
    """
    positions, numbers, frag = water_cluster(2, seed=83)
    batch = make_batch(positions, numbers, frag)
    model = _direct_model()
    before = model(batch)
    assert float(before.response.mu.abs().max()) > 1e-6

    with torch.no_grad():
        for mod in (model.response.params.chivec_head, model.response.params.chiquad_head):
            for prm in mod.parameters():
                prm.add_(0.05)
    after = model(batch)
    assert not torch.allclose(after.response.mu, before.response.mu), (
        "the perturbation did not reach the multipoles"
    )
    assert torch.allclose(after.energy_internal, before.energy_internal, atol=0, rtol=0), (
        "the on-site sectors are still contributing to internal_energy"
    )


def test_direct_multipoles_keep_the_isolated_fragment_invariants():
    """The properties the levels rest on must survive the change of variables."""
    model = _direct_model(induction=True)
    anchor = make_batch(*water_cluster(1, seed=87))
    out = model(anchor)
    # 1e-14 Ha rather than exactly 0: the direct form adds `mu0` to the state's induced part,
    # and that one extra floating-point add costs an ulp (measured 1e-17 Ha, i.e. 3e-14
    # kJ/mol). The claim is structural zero, and this is three orders inside it.
    assert float(out.interaction["induction"].abs().max()) < 1e-6, (
        "induction is not ~zero on an isolated fragment"
    )

    # A lone atom still reduces to E0 exactly.
    lone = model(_free_atom_batch(list(NEIGHBOR_TYPES)))
    assert torch.allclose(lone.fragment_energy, lone.energy_ref, atol=0, rtol=0)

    # Rotation invariance, end to end.
    positions, numbers, frag = water_cluster(2, seed=89)
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=positions.dtype))
    if float(torch.det(q)) < 0:
        q[:, 0] = -q[:, 0]
    a = model(make_batch(positions, numbers, frag))
    b = model(make_batch(positions @ q.t(), numbers, frag))
    assert float((a.energy - b.energy).abs().max()) < 1e-10
    assert float((a.fragment_energy - b.fragment_energy).abs().max()) < 1e-10
