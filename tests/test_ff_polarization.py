"""The induction level: is its label really a relaxation of the frozen level?

Polarization and charge transfer are **one term** here. They were two, separated only by their
labels and by the atomic energy's feature stream, and that swap turned out to be a free ~20
kJ/mol knob only ``ct`` could reach -- measured at 99.99% of that channel with 4.6e-5 e
crossing a fragment boundary. Explicit inter-fragment charge flow is gone until reactivity
needs it; the fragment now supplies the channel graph and the environment supplies the
parameters on it.

The two levels share one energy functional and differ only in what is allowed to move, so
almost everything worth testing here is an *identity between levels* rather than a number:

* switch off what can move, and ``E_ind`` must be exactly zero;
* on an isolated fragment it must vanish, because there is no environment to respond to;
* and ``fragment_energy`` must not move at all when induction is switched on, because it is
  Q-Chem's *isolated*-fragment energy and nothing at a higher level may touch it.

The last one is the sharpest. It is the property the whole arrangement exists to preserve,
and it is the one an ordinary polarizable force field gives up when it switches on
intramolecular induced electrostatics.

Solver-level correctness (the operator, the adjoint, batching, the degenerate limits) lives in
``tests/test_ff_coupled_solve.py``; this file assumes it.
"""

import pytest
import torch

from rsfff.ff.environment import (
    N_PAIR_INVARIANTS,
    electrostatic_environment,
    environment_pair_invariants,
)
from rsfff.ff.multipole import build_polytensor, damped_interaction_tensor
from rsfff.ff.pairs import intra_fragment_channels, union_channels
from rsfff.ff.units import BOHR_ANG, KJMOL_PER_HARTREE
from rsfff.mlip.sqe import PairComplianceHead
from rsfff.mlip.switch import pairwise_switch

from test_ff_unified import (  # noqa: E402  -- shared fixtures for the unified model
    build_parts,
    make_batch,
    make_model,
    randomize,
    wake_environment,
    water_cluster,
)

LEVELS = dict(induction=True)


def build(seed=11, *, live_env=True, environment=True, **levels):
    """A model with the levels on and every zero-initialized readout moved off zero.

    ``environment=False`` removes the residual entirely, so ``h_env`` *is* ``h_frag`` and the
    two levels genuinely share response parameters. Note that ``live_env=False`` alone does
    **not** achieve that: :func:`randomize` perturbs every parameter including the environment
    MLP, so ``g`` is off zero either way.
    """
    parts = build_parts(seed=seed, extra_dim=N_PAIR_INVARIANTS)
    model = randomize(
        make_model(parts, environment=environment, levels={**LEVELS, **levels}), seed=13
    )
    return wake_environment(model) if (environment and live_env) else model


@pytest.fixture
def cluster():
    positions, numbers, frag = water_cluster(3, seed=101)
    return make_batch(positions, numbers, frag)


# ---------------------------------------------------------------------------
# each label is a relaxation of the level below it
# ---------------------------------------------------------------------------

def test_permanent_quadrupoles_survive_into_the_coupled_level_without_a_response(cluster):
    """**The regression test for the trap this change was built around.**

    With ``quadrupole_response`` off, each atom keeps a permanent quadrupole and nothing moves
    it. ``multipoles_from_state`` used to size the quadrupole sector from ``has_quad``, i.e.
    from whether the *state* carried a variable for it -- so dropping ``cquad`` made ``theta``
    empty, the ``theta.numel()`` guard skipped ``quad0``, and ``_to_polytensor`` zero-filled
    the block. The frozen level would have carried permanent quadrupoles and the coupled level
    silently would not: ``cls_elec`` and ``induction`` describing different molecules.

    Sizing the sector from the permanent multipole instead makes "no polarizability" mean a
    rigid moment, which is what it should mean. Asserted bitwise, because it is an identity:
    with nothing to move the quadrupole, the induced level's quadrupoles *are* the frozen
    level's.
    """
    # `environment=False` so `h_env is h_frag` and both levels read the *same* quadrupole
    # head. With the environment live the permanent moment is legitimately different between
    # the levels -- the response parameters are environment-aware, which is what induction is
    # -- and that would mask the thing being tested here.
    parts = build_parts(seed=11, extra_dim=N_PAIR_INVARIANTS,
                        direct_multipoles=True, quadrupole_response=False)
    model = randomize(make_model(parts, environment=False, levels=LEVELS), seed=13)
    out = model(cluster)

    assert out.response.quad_s is not None, "the permanent quadrupole head is not built"
    assert float(out.response.quad_s.detach().abs().max()) > 1e-9, (
        "the permanent quadrupoles are identically zero, so this asserts nothing"
    )
    assert out.level_ind is not None and out.level_ind.quad_s is not None, (
        "the coupled level lost its quadrupoles entirely"
    )
    assert torch.equal(out.level_ind.quad_s, out.response.quad_s), (
        "the coupled level's quadrupoles differ from the frozen level's with the quadrupole "
        "response switched off and the streams shared -- they are supposed to be rigid"
    )
    # The dipole sector, by contrast, still moves: this is not a model with nothing to relax.
    assert not torch.allclose(out.level_ind.mu, out.response.mu, atol=1e-12), (
        "nothing relaxed at all, so the quadrupole check above is vacuous"
    )


def test_quadrupole_response_needs_the_direct_parameterization():
    """``Theta = -C chiquad`` cannot express a rigid quadrupole: refuse rather than zero it."""
    import pytest as _pytest

    with _pytest.raises(ValueError, match="needs direct_multipoles"):
        build_parts(seed=3, direct_multipoles=False, quadrupole_response=False)


def test_induction_vanishes_when_nothing_can_relax(cluster):
    """Freeze every response degree of freedom and ``E_pol`` must be exactly zero.

    With no polarizability and no charge flow the coupled minimum *is* the frozen point, so
    the difference between the levels is not merely small -- it is the same number twice. This
    is what catches a sign error in the coupling operator, because a sign error would leave the
    frozen and polarized electrostatics disagreeing even at identical multipoles.

    ``environment=False`` is a precondition, not a convenience. ``E_pol`` is a pure relaxation
    only while the two levels share response parameters; with any environment residual at all
    they do not, and the difference legitimately picks up that parameter change too. Measured
    with a live ``g`` and every degree of freedom frozen, ``E_pol`` is -55 kJ/mol at
    *identical* multipoles -- entirely the ``chi``/``eta`` difference between the two streams.
    That is the caveat behind logging ``pol_ff`` during training rather than asserting its sign.
    """
    from dataclasses import replace as dc_replace

    model = build(environment=False)
    # Freeze the response at the interface rather than by reaching into head internals: every
    # polarizability and every channel compliance to exactly zero, so nothing can move.
    inner = model.response.response_parameters

    def frozen_parameters(*args, **kwargs):
        rp = inner(*args, **kwargs)
        return dc_replace(
            rp,
            alpha=torch.zeros_like(rp.alpha),
            cquad=None if rp.cquad is None else torch.zeros_like(rp.cquad),
            compliance=torch.zeros_like(rp.compliance),
        )

    model.response.response_parameters = frozen_parameters
    out = model(cluster)
    assert abs(float(out.interaction_ff["induction"].detach())) * KJMOL_PER_HARTREE < 1e-6


def test_induction_is_non_positive_at_initialization():
    """``E_pol <= 0`` by the variational principle, wherever the levels share parameters.

    The frozen energy is the *same functional* evaluated at the frozen multipoles, and the
    frozen multipoles minimize only its internal part -- so letting the coupling in can only
    lower it. That holds exactly while ``h_env == h_frag``, which is where a fresh model starts.

    It stops being guaranteed once ``g`` trains the two levels' response parameters apart,
    which is why ``pol_ff`` is a logged training metric rather than an assertion. Both regimes
    are exercised here: the guarantee is *asserted* where it holds and only *reported* where it
    does not, so this test cannot quietly start certifying something that is not true.
    """
    shared = build(environment=False)      # h_env is h_frag, so the levels share parameters
    live = build()                         # a trained-apart g, where the guarantee lapses
    worst_shared = worst_live = -float("inf")
    for n_frag in (2, 3, 4):
        positions, numbers, frag = water_cluster(n_frag, seed=17 + n_frag)
        batch = make_batch(positions, numbers, frag)
        worst_shared = max(
            worst_shared,
            float(shared(batch).interaction_ff["induction"].detach()) * KJMOL_PER_HARTREE,
        )
        worst_live = max(
            worst_live,
            float(live(batch).interaction_ff["induction"].detach()) * KJMOL_PER_HARTREE,
        )
    assert worst_shared <= 1e-9, (
        f"E_pol must be a relaxation when the levels share parameters; most positive value "
        f"was {worst_shared:.3e} kJ/mol"
    )
    print(f"\n  E_pol with a trained-apart g: most positive {worst_live:+.3f} kJ/mol")


def test_an_isolated_fragment_has_no_induction():
    """The sharpest single check on the anchoring.

    A lone water has nothing to polarize against and nowhere to transfer charge to, so both
    labels must vanish. Three separate constructions have to hold at once for this to pass:
    the anchored environment residual (``h_env == h_frag``), the anchored bond corrections
    (which are differences between stream/field settings), and the range separation switching
    the intramolecular coupling off.

    The residual is not identically zero -- the range separation leaves a little intramolecular
    electrostatics behind, and that remainder is what appears here. It is measured rather than
    assumed, because on a fragment larger than a monomer it would be genuinely nonzero.
    """
    positions, numbers, frag = water_cluster(1, seed=7)
    out = build()(make_batch(positions, numbers, frag))
    ind = abs(float(out.interaction["induction"].detach())) * KJMOL_PER_HARTREE
    # 1e-4 rather than 0: `M^ind != M^frozen` even alone, because the coupled level minimizes
    # with the intramolecular electrostatics *inside* the functional while the frozen level
    # adds them afterwards. That relaxation is ~2.6e-5 e. The old `ct` bound was exactly zero
    # because `ct` differenced two coupled solves; induction differences against the frozen
    # one, so it inherits the looser -- and correct -- bound that `pol` always had.
    assert ind < 1e-4, f"isolated monomer induction {ind:.3e} kJ/mol"


def test_fragment_energy_is_untouched_by_the_higher_levels(cluster):
    """The label-integrity check, and the reason the bond field term is anchored at zero field.

    ``fragment_energy`` is Q-Chem's isolated-fragment energy. Switching on polarization and
    charge transfer must not move it by *anything*, or the 1-body term is fitting a function
    that cannot match its target. Pauli and dispersion must not move either -- they read the
    same stream at both settings.

    The electrostatic channel is the deliberate exception: its correction moves from the
    environment-aware stream to the fragment-confined one, and the difference becomes the
    polarization correction. That makes ``cls_elec`` rigorously two-body, which is what the
    interaction between frozen monomer densities is.
    """
    parts = build_parts(seed=11, extra_dim=N_PAIR_INVARIANTS)
    frozen = wake_environment(
        randomize(make_model(parts, environment=True), seed=13)
    )(cluster)
    parts = build_parts(seed=11, extra_dim=N_PAIR_INVARIANTS)
    lifted = wake_environment(
        randomize(make_model(parts, environment=True, levels=LEVELS), seed=13)
    )(cluster)

    assert torch.allclose(
        frozen.fragment_energy, lifted.fragment_energy, atol=0.0, rtol=0.0
    ), "a higher level reached an isolated-fragment label"
    for name in ("pauli", "disp"):
        assert torch.allclose(
            frozen.interaction[name], lifted.interaction[name], atol=0.0, rtol=0.0
        ), name


def test_accounting_identity_covers_every_channel(cluster):
    """Every pair appears once, in one bucket: no double counting and no gap."""
    out = build()(cluster)
    total = out.fragment_energy.sum() + sum(v.sum() for v in out.interaction.values())
    assert torch.allclose(out.energy.sum(), total, atol=1e-11)
    assert torch.allclose(
        out.interaction["induction"],
        out.interaction_ff["induction"] + out.interaction_corr["induction"],
        atol=1e-13,
    )


def test_induction_conserves_charge_per_fragment(cluster):
    """No charge crosses a fragment boundary, which is what dropping explicit CT means.

    This is the inverse of what this file used to assert. The CT level conserved charge only
    per *frame* and let fragment charges move -- that was the whole content of its label. With
    the inter-fragment channels gone the induction level is back to per-*fragment*
    conservation, structurally: `union_channels` at cutoff 0 returns the intra-fragment graph,
    and the incidence matrix of a graph confined to a fragment cannot move charge out of it.
    """
    out = build()(cluster)
    frag = cluster.fragment_idx
    n_frag = int(cluster.n_fragments)
    q = out.level_ind.charges.detach()
    per_frag = q.new_zeros(n_frag).index_add_(0, frag, q)
    assert torch.allclose(per_frag, torch.zeros_like(per_frag), atol=1e-12), (
        "induction moved charge across a fragment boundary; the channel graph is not "
        "intra-fragment"
    )

def test_levels_are_rotation_and_translation_invariant(cluster):
    """Where a mis-symmetrized field invariant would surface.

    The electrostatic environment is a vector and a rank-2 tensor per atom, so it can only
    reach a pair head through rotation invariants. If ``F_i . rhat`` were fed raw instead of
    the swap-even combinations, this would still be rotation-invariant but the *pair ordering*
    would matter; the undirected pair list makes that ordering arbitrary, so it would show up
    as noise rather than a clean failure. Both are checked -- ordering below.
    """
    model = build()
    out = model(cluster)
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=torch.Generator().manual_seed(3)))
    q = q * torch.sign(torch.det(q))
    moved = make_batch(
        cluster.positions @ q.t() + torch.tensor([1.3, -0.7, 2.0]),
        cluster.atomic_numbers,
        cluster.fragment_idx,
    )
    turned = model(moved)
    assert torch.allclose(
        out.interaction["induction"], turned.interaction["induction"], atol=1e-10
    )
    assert torch.allclose(out.energy, turned.energy, atol=1e-10)


def test_environment_pair_invariants_survive_swapping_the_pair(cluster):
    """The constraint ``OneBodyEnvironment`` documented, checked directly.

    An undirected pair list stores each pair once in an arbitrary order, so any quantity fed
    to the pair head must be even under the swap. The field contractions are the trap:
    ``rhat`` flips sign, so ``(F_i - F_j).rhat`` survives (both factors flip) while
    ``(F_i + F_j).rhat`` does not and has to be taken in magnitude.
    """
    model = build()
    out = model(cluster)
    pair_index, r = out.pair_index, out.r
    i, j = pair_index[0], pair_index[1]
    positions = cluster.positions
    dr_au = (positions[j] - positions[i]) / BOHR_ANG
    r_au = r / BOHR_ANG
    t_point = damped_interaction_tensor(dr_au, None, 1.0 / r_au, max_rank=2)
    m = build_polytensor(
        out.level_ind.charges, out.level_ind.mu,
        None if out.level_ind.quad_s is None else
        __import__("rsfff.ff.multipole", fromlist=["x"]).spherical_to_cartesian_quadrupole(
            out.level_ind.quad_s
        ),
        max_rank=2,
    )
    env = electrostatic_environment(
        positions, pair_index, t_point, out.gate["elst"], m, max_rank=2
    )
    r_hat = (positions[j] - positions[i]) / r.unsqueeze(-1)
    forward = environment_pair_invariants(env, pair_index, r_hat)
    flipped = environment_pair_invariants(
        env, torch.stack((j, i)), -r_hat
    )
    assert forward.shape[-1] == N_PAIR_INVARIANTS
    assert torch.allclose(forward, flipped, atol=1e-12), (
        "an environment invariant is odd under the pair swap, so the correction would depend "
        "on the arbitrary storage order of an undirected pair"
    )


# ---------------------------------------------------------------------------
# the channel graph
# ---------------------------------------------------------------------------

def test_forces_match_central_differences_through_the_coupled_solve():
    """End-to-end forces with both levels live -- the adjoint's real test.

    ``tests/test_ff_coupled_solve.py`` checks the adjoint against the dense solve in isolation;
    this checks it where it actually has to work, with the environment features feeding the
    bond channel through the converged multipoles. The CG tolerance is tightened well below the
    finite-difference step, because the solve is only differentiated to the accuracy it is
    converged to.
    """
    model = build(seed=23, cg_rtol=1e-13, cg_atol=1e-15)
    positions, numbers, frag = water_cluster(2, seed=113)
    batch = make_batch(positions, numbers, frag)
    batch.positions.requires_grad_(True)
    (grad,) = torch.autograd.grad(model(batch).energy.sum(), batch.positions)

    h = 1e-5
    for atom, comp in ((0, 1), (3, 0), (4, 2)):
        shifted = []
        for sign in (+1, -1):
            p = positions.clone()
            p[atom, comp] += sign * h
            shifted.append(float(model(make_batch(p, numbers, frag)).energy.sum().detach()))
        fd = (shifted[0] - shifted[1]) / (2 * h)
        assert float(grad[atom, comp]) == pytest.approx(fd, abs=1e-6, rel=1e-6), (
            f"atom {atom} component {comp}"
        )


def test_union_channels_keeps_every_bond_at_any_distance():
    """A stretched bond's channel must never be dropped by a radius or an envelope."""
    positions, numbers, frag = water_cluster(3, seed=5)
    batch = make_batch(positions, numbers, frag)
    intra, _ = intra_fragment_channels(frag)
    want = {tuple(p) for p in intra.t().tolist()}
    for cutoff in (0.0, 2.0, 5.0, 20.0):
        bond_index, bond_batch, from_radius = union_channels(
            batch.positions, batch.batch_idx, frag, cutoff
        )
        got = {tuple(p) for p in bond_index.t().tolist()}
        assert want <= got, f"a bonded channel was dropped at cutoff {cutoff}"
        keep = torch.tensor([tuple(p) in want for p in bond_index.t().tolist()])
        assert not bool(from_radius[keep].any()), (
            "a bonded channel was marked for enveloping; a stretched bond would then lose its "
            "channel at the cutoff"
        )
        assert torch.equal(bond_batch, batch.batch_idx[bond_index[0]])


def test_charge_transfer_channels_close_smoothly_at_the_cutoff():
    """Without an envelope the forces are discontinuous where a channel appears.

    The compliance must reach exactly zero at ``r_off`` and be smooth there, which is the
    bounded ``s -> 0`` limit the whole SQE parameterization is built around.
    """
    head = PairComplianceHead(4, hidden=8, depth=1, cutoff=5.0, s_init=0.5)
    h = torch.randn(2, 4, generator=torch.Generator().manual_seed(0))
    bond_index = torch.tensor([[0], [1]])
    r_off, r_on = 5.0, 4.0
    previous = None
    for d in (3.0, 4.5, 4.9, 4.999, 5.0, 5.5):
        positions = torch.tensor([[0.0, 0.0, 0.0], [d, 0.0, 0.0]])
        envelope = pairwise_switch(torch.tensor([d]), r_on, r_off)
        s = float(head(h, positions, bond_index, envelope=envelope).detach())
        if previous is not None:
            assert s <= previous + 1e-12, "the envelope must be monotone over the taper"
        previous = s
    assert s == 0.0, "compliance must be exactly zero past the cutoff"
