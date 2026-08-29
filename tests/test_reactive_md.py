"""Confined, routing-weight-biased MD on the mediated model (:mod:`rsfff.md`).

The test that matters most is the first. **The enumerator must reproduce the corpus.**
Everything else here checks that a force is the gradient of the energy it claims to be, which
is table stakes; but if :func:`rsfff.md.enumerate_group` produced a different family of
decompositions than the one stored in ``data/wb97mv_tzvpd``, the dynamics would sample a
partition the mediator was never fitted on and every frame it generated would be off-manifold
in a way no force check could see.

The second-most important is :func:`test_energy_is_smooth_when_a_candidate_closes`. Dropping a
closed candidate shrinks the contested set, which changes ``Omega`` for the candidates that
survive -- continuous only because the factor removed is the bump at a *covalent* O-H
distance, where it is flat at exactly 1. That is an argument about the rank-0 assignment, and
arguments are what regression tests are for. It is checked as a convergence rate, following
``tests/test_mediator.py``: a magnitude threshold passes even when the accounting has a step
in it, because proton transfer is a stiff coordinate and the second difference is dominated by
the bond rather than by the bookkeeping.
"""

from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.io import read

from rsfff.ff.mediator import MediatorHead
from rsfff.ff.mixture_model import intra_pairs_unsorted, mixture_forward
from rsfff.md import (
    HarmonicBias,
    MediatedCalculator,
    ambiguity,
    enumerate_group,
    flat_bottom_sphere,
    load_mediated_model,
    logit,
    transfer_delta,
)
from rsfff.md.assign import DEFAULT_BUMP, AssignmentError, rank_oh_fragment_assignments
from rsfff.mlip.heads import env_parameters
from rsfff.mlip.reference_states import AtomicStateReference
from rsfff.qcgen.multifrag import read_multifrag_extxyz
from rsfff.train.build_expert import build_expert_model
from rsfff.train.config import Config
from rsfff.train.data import load_reference_energies

NEIGHBOR_TYPES = (1, 8)
CHECKPOINT = Path("checkpoints/ion_mediator_v4_full/best.pt")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

#: The four multi-fragmentation ion files, with the total charge each was harvested at.
ION_CORPUS = [
    ("data/wb97mv_tzvpd/w1_h3o+_wb97mv_tzvpd.xyz", +1),
    ("data/wb97mv_tzvpd/w2_h3o+_wb97mv_tzvpd.xyz", +1),
    ("data/wb97mv_tzvpd/w1_oh-_wb97mv_tzvpd.xyz", -1),
    ("data/wb97mv_tzvpd/w2_oh-_wb97mv_tzvpd.xyz", -1),
]

_Z = {"O": 8, "H": 1}


# ---------------------------------------------------------------------------------------
# A small untrained model, so the force checks do not need the checkpoint
# ---------------------------------------------------------------------------------------

def _config() -> Config:
    """The cheap configuration ``tests/test_mediator.py`` uses, for the same reason."""
    cfg = Config()
    cfg.dtype = "float64"
    cfg.features.cutoff, cfg.features.n_max, cfg.features.l_max = 5.0, 3, 2
    cfg.features.selected_lambdas = [0, 1, 2]
    cfg.features.density_channels = 4
    cfg.elec.max_rank = 2
    cfg.elec.hidden, cfg.elec.depth, cfg.elec.equiv_channels = 16, 2, 4
    cfg.elec.direct_multipoles = True
    cfg.elec.quadrupole_response = False
    cfg.dispersion.hidden, cfg.pauli.hidden = 8, 8
    cfg.pauli.equiv_channels = 4
    cfg.expert.environment_features = True
    cfg.expert.induction = False
    cfg.expert.bond_hidden, cfg.expert.bond_equiv_channels = 16, 4
    cfg.expert.r0_hidden = 16
    cfg.expert.fragment_state_dim = 4
    return cfg


@pytest.fixture(scope="module")
def model():
    torch.set_default_dtype(torch.float64)
    e0 = load_reference_energies(
        "data/atomic_references_wb97mv_tzvpd.json", NEIGHBOR_TYPES
    ).double()
    states = AtomicStateReference.from_json(
        "data/atomic_reference_states_wb97mv_tzvpd.json", NEIGHBOR_TYPES, dtype=torch.float64
    )
    torch.manual_seed(0)
    m = build_expert_model(_config(), NEIGHBOR_TYPES, e0, states).double()
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, torch.nn.Linear):
                mod.weight.normal_(0.0, 0.05)
        for _n, p in env_parameters(m):
            p.normal_(0.0, 0.05)

    probe = enumerate_group(_zundel(2.45, 1.05)[0], _zundel(2.45, 1.05)[1], 1).batch(0)
    em = m.emit(probe, probe.fragment_idx, bond_index=intra_pairs_unsorted(probe.fragment_idx))
    p_frag = int(em.iso.inv_feats.shape[-1])
    p_env = int(em.joined.inv_feats.shape[-1]) - p_frag
    # Randomize the mediator readout. It is zero-initialized by design -- an untrained
    # mediator must be the envelope and nothing else -- but that makes the score path
    # identically flat, so a force check on it would pass with the features detached.
    m.mediator = MediatorHead(p_frag, p_env, hidden=8, depth=2, bump=dict(DEFAULT_BUMP)).double()
    with torch.no_grad():
        for mod in m.mediator.modules():
            if isinstance(mod, torch.nn.Linear):
                mod.weight.normal_(0.0, 0.05)
                mod.bias.normal_(0.0, 0.05)
    return m


def _unit(theta_deg, phi_deg, sign=1.0):
    t, p = math.radians(theta_deg), math.radians(phi_deg)
    return np.array([sign * math.cos(t), math.sin(t) * math.cos(p), math.sin(t) * math.sin(p)])


def _zundel(r_oo: float, d: float, r_oh: float = 0.98, theta: float = 112.0):
    """H5O2+ with the bridging proton at ``d`` along the O-O axis. Charge +1."""
    o1, o2 = np.zeros(3), np.array([r_oo, 0.0, 0.0])
    pos = [o1]
    pos += [o1 + r_oh * _unit(theta, phi, -1.0) for phi in (60.0, 300.0)]
    pos += [o2]
    pos += [o2 + r_oh * _unit(theta, phi, +1.0) for phi in (120.0, 240.0)]
    pos += [np.array([d, 0.0, 0.0])]
    return (torch.tensor(np.array(pos), dtype=torch.float64),
            torch.tensor([8, 1, 1, 8, 1, 1, 1]))


def _canonical(fragment_idx) -> frozenset:
    """A decomposition as a set of atom-groups, so arbitrary fragment ids do not matter."""
    f = np.asarray(fragment_idx)
    return frozenset(frozenset(np.nonzero(f == k)[0].tolist()) for k in np.unique(f))


# ---------------------------------------------------------------------------------------
# 1. The enumerator against the corpus it has to reproduce
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("path,charge", ION_CORPUS)
def test_enumeration_reproduces_the_stored_decompositions(path, charge):
    """**The test this module exists for.**

    ``rsfff.md.assign`` is a port of the enumeration in
    ``qchem_roundtrip/scripts/qchem_roundtrip.py``, which cannot be imported (it runs on a
    cluster with no ``rsfff`` install) and so cannot be shared. A port that drifted would be
    invisible: the dynamics would run, produce plausible geometries, and quietly sample a
    different family of decompositions than the mediator was fitted on. This pins it frame by
    frame against the files the mediator was actually trained from.
    """
    if not Path(path).exists():
        pytest.skip(f"{path} not present")
    frames = read_multifrag_extxyz(path)
    assert frames, f"{path} parsed to no frames"

    for i, frame in enumerate(frames):
        pos = torch.as_tensor(np.asarray(frame["positions"]), dtype=torch.float64)
        z = torch.as_tensor([_Z[s] for s in frame["symbols"]])
        group = enumerate_group(pos, z, charge)
        mine = {_canonical(f) for f in group.fragments.numpy()}
        stored = {_canonical(f) for f in frame["fragment_idx"]}
        assert mine == stored, (
            f"{path} frame {i}: enumerate_group produced {len(mine)} decompositions, the file "
            f"stores {len(stored)}, and they differ. The MD would sample a partition family "
            f"the mediator was never trained on."
        )


def test_enumeration_charges_track_the_hosting_fragment():
    """The excess charge sits on the fragment holding the extra (or missing) proton."""
    pos, z = _zundel(2.45, 1.05)
    group = enumerate_group(pos, z, 1)
    # Every candidate puts the +1 on whichever fragment ends up holding three hydrogens.
    for m in range(group.fragments.shape[0]):
        frag = group.fragments[m]
        charged = frag[group.atom_charge[m] > 0.5].unique()
        assert charged.numel() == 1, (
            f"decomposition {m} charges {charged.numel()} fragments; exactly one must be the "
            f"ion, or the frame carries a composition the model has never seen"
        )
        host = int(charged[0])
        n_h = int(((frag == host) & (group.atomic_numbers == 1)).sum())
        assert n_h == 3, f"the cation of decomposition {m} has {n_h} hydrogens, expected 3"
        assert float(group.atom_charge[m].sum()) == pytest.approx(float((frag == host).sum()))

    # The bridging proton must be one of the atoms actually put in play.
    moved = {int((group.fragments[m] != group.fragments[0]).nonzero()[0])
             for m in range(1, group.fragments.shape[0])}
    assert 6 in moved, f"the bridge (atom 6) generated no candidate; moved atoms were {moved}"


def test_neutral_cluster_enumerates_exactly_one_decomposition():
    """No competition means no mediator: a neutral cluster runs the single-fragmentation model.

    ``M = 1`` is what makes the mediator free on unreactive frames, and ``contested`` empty is
    what ``mixture_forward`` needs to reduce to ``FragmentExpertModel.forward``.
    """
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0],
         [2.8, 0.0, 0.0], [3.2, 0.9, 0.0], [3.2, -0.9, 0.0]], dtype=torch.float64
    )
    group = enumerate_group(pos, torch.tensor([8, 1, 1, 8, 1, 1]), 0)
    assert group.fragments.shape[0] == 1
    assert group.contested.numel() == 0


def test_distant_competitor_is_dropped_by_the_envelope():
    """Pull the oxygens apart and the alternative host closes: M falls back to 1."""
    close = enumerate_group(*_zundel(2.45, 1.05), 1)
    far = enumerate_group(*_zundel(6.0, 0.98), 1)
    # Compact geometries keep more than the bridge alive: `contact_distance` reads the nearest
    # atom of the host, so a terminal hydrogen sitting 2 Angstrom from the other water's
    # hydrogens is inside the envelope too. The trained mediator gives those ~1e-5, and
    # carrying them is what keeps the accounting continuous -- screening them out on the O-H
    # distance is exactly the step `one_hop_candidates` documents.
    assert close.fragments.shape[0] > 1
    assert far.fragments.shape[0] == 1, (
        "a proton 5 Angstrom from the second oxygen is not a candidate for it; leaving it "
        "enumerated costs a full featurization for exactly zero weight"
    )


def test_assignment_is_globally_optimal_not_greedy():
    """Branch and bound, not nearest-oxygen.

    Greedy fails exactly where this matters: take the shared proton for the nearer oxygen
    first and a later hydrogen can be left with no capacity, forcing a globally worse
    assignment. Here O0 is over-subscribed if each hydrogen simply grabs its nearest oxygen.
    """
    symbols = ["O", "O", "H", "H", "H", "H"]
    coords = np.array([[0.0, 0.0, 0.0], [2.6, 0.0, 0.0],
                       [-0.96, 0.0, 0.0], [0.5, 0.9, 0.0],
                       [1.0, 0.0, 0.0], [3.3, 0.7, 0.0]])
    idx, cost = rank_oh_fragment_assignments(symbols, coords, 0)[0].fragment_idx, None
    counts = np.bincount(np.asarray(idx)[2:], minlength=2)
    assert list(counts) == [2, 2], f"expected two hydrogens per oxygen, got {counts}"


def test_non_oh_system_is_refused():
    """Better a clear error than a silent, wrong fragmentation of something else."""
    with pytest.raises(AssignmentError, match="O/H clusters only"):
        rank_oh_fragment_assignments(["O", "H", "C"], np.zeros((3, 3)), 0)


def test_candidate_set_is_stable_under_small_displacements():
    """The set must not depend on an argmin, because an argmin jumps.

    This is the defect the base-relative generator exists to remove. Re-ranking the
    assignments every step made ``M`` fall from 3 to 2 partway across a transfer in a 4-water
    cluster -- not because a candidate had lost its weight, but because the *reference* the
    hop count was measured from had flipped. Here the base is held, so walking the proton all
    the way across changes the set only at the ends, where the weight is already zero.
    """
    pos, z = _zundel(2.45, 0.98)
    base = enumerate_group(pos, z, 1).fragments[0].numpy()
    counts = []
    for d in np.linspace(0.98, 1.47, 25):
        g = enumerate_group(_zundel(2.45, float(d))[0], z, 1, base=base)
        counts.append(g.fragments.shape[0])
    assert len(set(counts)) == 1, (
        f"the candidate count moved through {sorted(set(counts))} while the proton crossed a "
        f"single hydrogen bond; the enumeration is not stable and the forces will step"
    )


def test_committing_the_base_follows_the_chemistry(trained):
    """After a transfer the held base must move, or the enumeration goes stale.

    The base names which fragment is the ion. Once the proton has genuinely moved, a base that
    still says otherwise puts the candidate that *is* the chemistry at the edge of its
    envelope instead of the centre of a fresh one. The commit is hysteretic, so it happens
    once and decisively rather than every time the proton rattles.
    """
    from ase import Atoms

    pos, z = _zundel(2.45, 0.98)
    atoms = Atoms(numbers=z.numpy(), positions=pos.numpy())
    calc = MediatedCalculator(trained, 1, with_induction=False, commit_threshold=0.9)
    atoms.calc = calc
    atoms.get_potential_energy()
    assert calc.n_commits == 0, "a settled geometry must not commit"
    before = calc._base.copy()

    # Walk the proton all the way onto the other oxygen.
    for d in np.linspace(1.0, 1.6, 13):
        atoms.set_positions(_zundel(2.45, float(d))[0].numpy())
        atoms.get_potential_energy()
    assert calc.n_commits >= 1, (
        "the proton transferred and the base never followed; every later step then enumerates "
        "around an ion that is no longer there"
    )
    assert not np.array_equal(calc._base, before)

    calc.reset_base()
    assert calc._base is None and calc.n_commits == 0


def test_transfer_delta_is_the_O_H_asymmetry_not_the_contact_distance():
    """``delta`` must read O-H distances, or the restraint moves the wrong atoms.

    Built on the mediator's ``rho`` -- the nearest atom of the host -- the coordinate is
    frequently a H...H distance on a compact cluster, and restraining it to zero is satisfied
    by rotating a neighbouring water while the proton stays put: measured, ``cv`` reached
    +0.001 while the real O-H asymmetry sat at -0.713 Angstrom.
    """
    pos, z = _zundel(2.45, 1.10)
    group = enumerate_group(pos, z, 1)
    cv = float(transfer_delta(group.positions, group.atomic_numbers,
                              group.fragments, group.contested))
    expected = float((pos[6] - pos[0]).norm() - (pos[6] - pos[3]).norm())
    assert cv == pytest.approx(expected, abs=1e-3), (
        f"delta reported {cv:+.4f} where the O-H asymmetry is {expected:+.4f}"
    )
    # Zero exactly at the midpoint, which is what makes `target=0` mean "the transfer point".
    mid = _zundel(2.45, 2.45 / 2)[0]
    g2 = enumerate_group(mid, z, 1)
    assert float(transfer_delta(g2.positions, g2.atomic_numbers,
                                g2.fragments, g2.contested)) == pytest.approx(0.0, abs=1e-3)


@pytest.mark.parametrize("n_waters", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("ion", ["h3o+", "oh-"])
def test_seeded_clusters_have_something_to_mediate(n_waters, ion):
    """Every cluster the driver can build must enumerate a competing decomposition.

    Both halves of this were wrong at first and neither is visible from the formula. Trimming
    a benchmark cluster by *centrality* picks the two oxygens nearest the centre of mass, and
    in a cyclic tetramer those are a diagonal pair 3.9 Angstrom apart -- not hydrogen bonded,
    so no proton placement between them is ever in range. And placing the extra proton along
    the lone-pair bisector of the most solvated oxygen points it at the vacuum about half the
    time. Either way the run starts at ``M = 1``, the bias has nothing to act on, and the
    failure surfaces 8000 steps later as a guard-rail abort rather than as a bad geometry.
    """
    from run_reactive_md import build_cluster

    atoms, charge = build_cluster(n_waters, ion)
    assert charge == (1 if ion == "h3o+" else -1)

    n_o = sum(1 for z in atoms.get_atomic_numbers() if z == 8)
    n_h = sum(1 for z in atoms.get_atomic_numbers() if z == 1)
    assert n_o == n_waters + 1, (
        f"--n-waters counts waters *besides* the ion, so {n_waters} must give {n_waters + 1} "
        f"oxygens; got {n_o}"
    )
    assert n_h == 2 * n_o + charge

    group = enumerate_group(
        torch.as_tensor(atoms.get_positions(), dtype=torch.float64),
        torch.as_tensor(atoms.get_atomic_numbers()),
        charge,
    )
    assert group.fragments.shape[0] > 1, (
        f"{ion} + {n_waters} waters ({atoms.get_chemical_formula()}) enumerates one "
        f"decomposition; the ion has no acceptor in range and the run cannot sample anything"
    )
    assert group.contested.numel() > 0


# ---------------------------------------------------------------------------------------
# 2-3. The added energy terms are the gradients they claim to be
# ---------------------------------------------------------------------------------------

def _numerical_gradient(fn, pos, h=1e-6):
    grad = torch.zeros_like(pos)
    flat = pos.reshape(-1)
    for k in range(flat.numel()):
        up, dn = pos.clone().reshape(-1), pos.clone().reshape(-1)
        up[k] += h
        dn[k] -= h
        grad.reshape(-1)[k] = (fn(up.reshape(pos.shape)) - fn(dn.reshape(pos.shape))) / (2 * h)
    return grad


def test_confinement_force_matches_finite_differences():
    torch.manual_seed(0)
    z = torch.tensor([8, 1, 1, 8, 1, 1])
    pos = torch.randn(6, 3, dtype=torch.float64) * 3.0
    kw = dict(radius=1.5, k=0.4, h_slack=1.2)

    p = pos.clone().requires_grad_(True)
    (analytic,) = torch.autograd.grad(flat_bottom_sphere(p, z, **kw), p)
    numeric = _numerical_gradient(lambda x: float(flat_bottom_sphere(x, z, **kw)), pos)
    assert torch.allclose(analytic, numeric, atol=1e-7), (
        f"max deviation {float((analytic - numeric).abs().max()):.3e}"
    )


def test_confinement_is_inert_inside_and_exerts_no_net_force_outside():
    """Two claims: flat-bottomed, and translation-invariant.

    The second is the reason the sphere is centered on the running center of mass rather than
    on the origin. A fixed center pushes the whole cluster whenever it is not concentric, and
    that drift then has to be removed from every trajectory.
    """
    z = torch.tensor([8, 1, 1, 8, 1, 1])
    tight = torch.tensor(
        [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0],
         [2.0, 0.0, 0.0], [2.4, 0.9, 0.0], [2.4, -0.9, 0.0]], dtype=torch.float64
    )
    assert float(flat_bottom_sphere(tight, z, radius=6.0, k=0.4)) == 0.0

    spread = tight * 4.0
    p = spread.clone().requires_grad_(True)
    e = flat_bottom_sphere(p, z, radius=1.0, k=0.4)
    assert float(e.detach()) > 0.0, (
        "the wall must actually engage for the next assertion to mean anything"
    )
    (grad,) = torch.autograd.grad(e, p)
    assert torch.allclose(grad.sum(0), torch.zeros(3, dtype=torch.float64), atol=1e-12), (
        f"net force {grad.sum(0).tolist()} is not zero; the confinement is not "
        f"translation-invariant and the trajectory will drift"
    )


def test_bias_force_matches_finite_differences(model):
    """The whole point: ``dE/dR`` through the routing weight, checked numerically.

    ``w`` depends on positions twice over -- through the contact distance into the C² validity
    bump, and through the lambda-SOAP features into the score net. A finite-difference check on
    the *biased* total is the only thing that confirms both paths are live; an analytic-only
    test would pass with either one silently detached.
    """
    bias = HarmonicBias(cv="ambiguity", k=2.0, target=0.4)
    # Two things about this geometry are deliberate. The second oxygen sits on the *falling
    # edge* of the envelope, so both routes into `w` are live -- the envelope through the
    # contact distance and the score net through the features -- where at a symmetric Zundel
    # both envelopes are flat at 1 and the first path contributes exactly zero. And the
    # membership is genuinely lopsided (w ~ 0.71/0.29), because `A` is *stationary* at equal
    # weights: dA/dw = -2w is uniform there and the weights sum to a constant, so a gradient
    # check run at a 50/50 membership measures nothing at all.
    pos, z = _zundel(3.00, 0.98)

    def biased(x):
        group = enumerate_group(x, z, 1)
        out = mixture_forward(model, group, model.mediator)
        return bias(out, group)[0]

    p = pos.clone().requires_grad_(True)
    (analytic,) = torch.autograd.grad(biased(p), p)
    numeric = _numerical_gradient(lambda x: float(biased(x).detach()), pos, h=1e-6)
    assert torch.allclose(analytic, numeric, atol=1e-6, rtol=1e-4), (
        f"max deviation {float((analytic - numeric).abs().max()):.3e}; the bias gradient does "
        f"not match the bias energy it is supposed to differentiate"
    )
    assert float(analytic.abs().max()) > 1e-6, "the bias exerted no force at all"


def test_logit_gradient_is_conditioned_where_ambiguity_is_not(trained):
    """**Why the default coordinate is the logit and not the raw ambiguity.**

    A softmax weight saturates -- ``dw/dR`` goes as ``w(1 - w)`` -- so a coordinate built
    straight from ``w`` has a gradient that collapses in the decided region and peaks at the
    crossover. A single force constant then cannot serve both: sized to move the decided
    region it is catastrophic at the crossover, and measured, a harmonic on ``ambiguity`` at
    the ``k = 10`` needed to move a relaxation reached 51.6 eV/Angstrom and aborted a
    trajectory at step 40. The logit cancels that ``w(1 - w)`` exactly.

    Pinned as the *ratio* of the largest to the smallest gradient across a transfer scan,
    which is the quantity that decides whether one ``k`` works everywhere.
    """
    base = enumerate_group(*_zundel(2.45, 0.98), 1).fragments[0].numpy()
    g_logit, g_amb = [], []
    for d in np.linspace(0.95, 1.50, 12):
        pos, z = _zundel(2.45, float(d))
        p = pos.clone().requires_grad_(True)
        group = enumerate_group(p, z, 1, base=base)
        out = mixture_forward(trained, group, trained.mediator, with_induction=False)
        for fn, store in ((logit, g_logit), (ambiguity, g_amb)):
            (grad,) = torch.autograd.grad(fn(out.mediator.weights), p, retain_graph=True)
            store.append(float(grad.abs().max()))

    spread = lambda v: max(v) / max(min(v), 1e-30)          # noqa: E731
    assert spread(g_logit) < 10.0, (
        f"the logit gradient spans {spread(g_logit):.0f}x across the scan; it is supposed to "
        f"be the well-conditioned coordinate"
    )
    assert spread(g_amb) > 50.0 * spread(g_logit), (
        f"ambiguity spans only {spread(g_amb):.0f}x against the logit's "
        f"{spread(g_logit):.0f}x -- if they are comparable this test no longer justifies the "
        f"default and the choice should be revisited"
    )


def test_linear_tail_caps_the_bias_force():
    """Past ``max_deviation`` the restraint is linear, so the force stops growing.

    Switching a pure harmonic on at ``logit = 6.35`` with ``k = 0.03`` deposits about 16 eV
    into a 13-atom cluster in a single step, which measured out at 7700 K within twenty steps.
    The tail bounds the force; the ramp in ``scripts/run_reactive_md.py`` handles the rest.
    """
    class _Out:
        def __init__(self, w):
            self.mediator = types.SimpleNamespace(weights=w)

    bias = HarmonicBias(cv="ambiguity", k=2.0, target=0.0, max_deviation=0.1)
    group = types.SimpleNamespace(positions=torch.zeros(1, 3, dtype=torch.float64))

    def energy_at(a: float) -> torch.Tensor:
        # Two weights giving ambiguity exactly `a`: 1 - (w^2 + (1-w)^2) = a.
        w0 = 0.5 * (1.0 + math.sqrt(max(0.0, 1.0 - 2.0 * a)))
        w = torch.tensor([w0, 1.0 - w0], dtype=torch.float64, requires_grad=True)
        e, cv = bias(_Out(w), group)
        return e, cv, w

    # C^1 at the join. The two one-sided *slopes* must agree, which is the claim; the values
    # differ by slope * 2h and comparing them to 1e-8 would only be testing h.
    h = 1e-6
    at = lambda a: float(energy_at(a)[0])                            # noqa: E731
    assert at(0.1) == pytest.approx(0.5 * 2.0 * 0.1**2, rel=1e-4)
    left = (at(0.1) - at(0.1 - h)) / h
    right = (at(0.1 + h) - at(0.1)) / h
    assert left == pytest.approx(right, rel=1e-4), (
        f"one-sided slopes {left:.6f} and {right:.6f} disagree at the join; the tail is not "
        f"C1 and the bias force steps there"
    )
    assert left == pytest.approx(2.0 * 0.1, rel=1e-3)

    # Beyond the join the energy is linear, so dE/dcv is constant at k * max_deviation.
    for a in (0.2, 0.3, 0.4):
        e0, _, _ = energy_at(a)
        e1, _, _ = energy_at(a + 1e-5)
        slope = float((e1 - e0) / 1e-5)
        assert slope == pytest.approx(2.0 * 0.1, rel=1e-3), (
            f"slope {slope:.5f} at deviation {a}; the linear tail is not capping the force"
        )


def test_ambiguity_is_smooth_where_occupancy_would_kink():
    """``1 - sum w^2`` rather than ``1 - max w``, and this is the difference.

    At a 50/50 membership ``max`` has a corner, so ``occupancy`` has a discontinuous
    derivative exactly at the geometry a bias would drive the system to sit at. The polynomial
    form does not.
    """
    w = torch.tensor([0.5, 0.5], dtype=torch.float64, requires_grad=True)
    for eps in (1e-4, -1e-4):
        d = torch.tensor([eps, -eps], dtype=torch.float64)
        a = ambiguity((w + d).detach().requires_grad_(True))
        assert float(a) == pytest.approx(0.5 - 2 * eps**2, abs=1e-12)
    (g,) = torch.autograd.grad(ambiguity(w), w)
    assert torch.allclose(g, torch.tensor([-1.0, -1.0], dtype=torch.float64))


# ---------------------------------------------------------------------------------------
# 3b. The geometry guard, and starting from a file
# ---------------------------------------------------------------------------------------

def test_guard_keeps_a_shared_proton_and_rejects_a_stranded_one():
    """**The guard the first harvest needed and did not have.**

    40% of that harvest had a proton stranded on a long hydrogen bond -- neither of the two
    protonation states nor a transition between them. The pair below is the whole distinction:
    same proton, same fraction of the way across, different O-O.
    """
    from run_reactive_md import geometry_defect

    kw = dict(min_distance=0.6, max_oh=1.45, max_oo=2.75)

    pos, z = _zundel(2.42, 1.21)                      # midway on a compressed bond
    shared = Atoms(numbers=z.numpy(), positions=pos.numpy())
    assert geometry_defect(shared, **kw) is None, (
        "a proton shared across a 2.42 Angstrom O-O is exactly what the run looks for"
    )

    pos, z = _zundel(3.10, 1.55)                      # same proton, but the bond is long
    stretched = Atoms(numbers=z.numpy(), positions=pos.numpy())
    defect = geometry_defect(stretched, **kw)
    assert defect is not None and defect.negative_space, (
        "a stretched bond is rejected as a transition structure but kept as negative space"
    )


def test_the_guard_reads_geometry_only():
    """It must not depend on the enumeration, and this is why.

    An earlier version asked the enumeration which host a contested proton would move to. That
    answer depends on the *held base*, so the same geometry was judged one way mid-trajectory
    and another way when re-read from the file: re-classifying a harvest found 3 rejects out of
    500 where the guard that produced it had rejected 35%. A curated corpus has to agree with
    the sampler that produced it, so the rule reads positions and nothing else.
    """
    import inspect

    from run_reactive_md import geometry_defect

    params = list(inspect.signature(geometry_defect).parameters)
    assert params[0] == "atoms", f"geometry_defect takes {params}"
    forbidden = {"results", "group", "model", "calc", "fragments", "contested", "base"}
    assert not forbidden & set(params), (
        f"geometry_defect takes {params}; anything carrying model or enumeration state makes "
        f"the verdict depend on the held base, so a harvest and the guard that produced it "
        f"disagree"
    )


def test_a_wide_cluster_with_a_compressed_reacting_pair_is_kept():
    """Distant oxygens elsewhere in the cluster must not condemn a good transition structure.

    The test measures the two oxygens nearest the most-stretched hydrogen, not the widest pair
    in the cluster. An optimized H3O+(H2O)3 has oxygens more than 4 Angstrom apart across the
    cluster while every hydrogen bond in it is ~2.6.
    """
    from run_reactive_md import geometry_defect

    atoms = read(str(Path("data/hydronium_clusters_ccdb/asp-H2O_3--H3O+.xyz")), index=0)
    d = atoms.get_all_distances()
    oxy = np.flatnonzero(atoms.get_atomic_numbers() == 8)
    assert d[np.ix_(oxy, oxy)].max() > 4.0, "the isomer no longer has distant oxygens"
    assert geometry_defect(atoms, min_distance=0.6, max_oh=1.45, max_oo=2.75) is None


@pytest.mark.parametrize("path,frame,charge", [
    ("data/hydroxide_clusters/jp5b03893_si_002.xyz", 6, -1),
    ("data/hydroxide_clusters/jp5b03893_si_002.xyz", 25, -1),
    ("data/hydronium_clusters_ccdb/asp-H2O_3--H3O+.xyz", 0, +1),
])
def test_the_guard_accepts_optimized_starting_geometries(path, frame, charge):
    """**The test that would have caught the bug that cost 72 runs.**

    An optimized cluster is by definition not a stranded proton, so the guard must accept every
    one of them. It did not: the O-O test was applied to the globally most-stretched hydrogen,
    which on a hydronium is the excess proton on a short bond but on a hydroxide is an ordinary
    water donating a normal 2.8 Angstrom hydrogen bond. Every hydroxide isomer was rejected at
    step 2, before the bias had ramped in, and the whole ion's half of a sweep harvested
    nothing.

    Both ions are parametrized deliberately. The earlier validation measured only hydronium
    output and reported the guard as clean; a guard checked on one ion is not checked.
    """
    from run_reactive_md import geometry_defect

    if not Path(path).exists():
        pytest.skip(f"{path} not present")
    atoms = read(path, index=frame)
    defect = geometry_defect(atoms, min_distance=0.6, max_oh=1.45, max_oo=2.75,
                             transfer_oh=1.25)
    assert defect is None, (
        f"rejected an optimized {'hydroxide' if charge < 0 else 'hydronium'} cluster: "
        f"{defect.reason}"
    )


def test_an_unstretched_hydrogen_on_a_wide_bond_is_not_a_defect():
    """A normal donor sits on a 2.7-2.9 Angstrom bond, and that is not a pathology.

    Measured over the first harvest: of hydrogens at r(O-H) = 0.90-1.05, **46.8% sit between
    oxygens more than 2.75 Angstrom apart** -- that is simply what a water hydrogen bond is.
    Only a hydrogen stretched past ``transfer_oh`` is being transferred, and only for those
    does the O-O distance say anything.
    """
    from run_reactive_md import geometry_defect

    # Two waters 3.2 Angstrom apart, every O-H at a normal 0.98: a long hydrogen bond, and
    # nothing more.
    atoms = Atoms(
        numbers=[8, 1, 1, 8, 1, 1],
        positions=[[0.0, 0.0, 0.0], [0.98, 0.0, 0.0], [-0.24, 0.95, 0.0],
                   [3.2, 0.0, 0.0], [4.18, 0.0, 0.0], [2.96, 0.95, 0.0]],
    )
    nearest = 0.98
    assert geometry_defect(atoms, min_distance=0.6, max_oh=1.45, max_oo=2.75,
                           transfer_oh=1.25) is None

    # Move one hydrogen out to 1.30 on the same wide bond and it becomes the pathology.
    stranded = atoms.copy()
    p = stranded.get_positions()
    p[1] = [1.30, 0.0, 0.0]
    stranded.set_positions(p)
    defect = geometry_defect(stranded, min_distance=0.6, max_oh=1.45, max_oo=2.75,
                             transfer_oh=1.25)
    assert defect is not None and defect.negative_space, (
        f"a proton at 1.30 Angstrom on a 3.2 Angstrom O-O is the stranded case; got {defect}"
    )
    assert nearest < 1.25 < 1.30                      # the threshold sits between the two


def test_disabled_guards_accept_everything():
    """``0`` means off, so an old run can be reproduced exactly."""
    from run_reactive_md import geometry_defect

    pos, z = _zundel(3.10, 1.55)
    atoms = Atoms(numbers=z.numpy(), positions=pos.numpy())
    assert geometry_defect(atoms, min_distance=0.6, max_oh=0.0, max_oo=0.0) is None


def test_only_labelable_rejects_are_marked_as_negative_space():
    """A collapsed frame is a reject, not a training example.

    Negative space is for configurations the reference method can actually be run on. Two
    nuclei 0.3 Angstrom apart will not converge in Q-Chem, and if it did its energy would
    dominate any loss the frame appeared in.
    """
    from run_reactive_md import geometry_defect

    collapsed = Atoms(numbers=[8, 1, 1], positions=[[0, 0, 0], [0.3, 0, 0], [0, 0.96, 0]])
    defect = geometry_defect(collapsed, min_distance=0.6, max_oh=1.45, max_oo=2.75)
    assert defect is not None and not defect.negative_space


@pytest.mark.parametrize("path,frame,expect", [
    ("data/hydronium_clusters_ccdb/asp-H2O_3--H3O+.xyz", 0, +1),
    ("data/hydroxide_clusters/jp5b03893_si_002.xyz", 12, -1),
    ("data/hydroxide_clusters/jp5b03893_si_002.xyz", 71, -1),
])
def test_charge_is_inferred_from_composition(path, frame, expect):
    """No cluster file records its charge, so it has to come from the formula.

    Frame 71 is ``Isomer 7j``, whose atoms are written ``... H, O, H ...`` for one water. It is
    here because a parser that assigned hydrogens by position would read a spurious H3O+/OH-
    pair out of it and still arrive at a total of -1 -- the charge check alone would not catch
    it, which is why the fragment check below exists too.
    """
    from run_reactive_md import load_geometry

    if not Path(path).exists():
        pytest.skip(f"{path} not present")
    atoms, charge = load_geometry(path, frame, None)
    assert charge == expect
    assert len(atoms) > 0


def test_malformed_frame_still_fragments_as_a_hydroxide_cluster():
    """``Isomer 7j`` must be OH- plus seven waters, not a charge-separated pair.

    Its atoms are ordered ``H, O, H`` for one water. Grouping by position gives that water's
    oxygen three hydrogens and the next one only one -- an H3O+ and an OH- invented inside a
    singly-charged anion. Assigning by distance is what avoids it, and this is the frame that
    proves the sweep does not need per-file special casing.
    """
    path = Path("data/hydroxide_clusters/jp5b03893_si_002.xyz")
    if not path.exists():
        pytest.skip("hydroxide set not present")
    atoms = read(str(path), index=71)
    z = np.asarray(atoms.get_atomic_numbers())
    group = enumerate_group(
        torch.as_tensor(atoms.get_positions(), dtype=torch.float64),
        torch.as_tensor(z), -1,
    )
    batch = group.batch(0)
    charges = batch.fragment_charge.numpy().astype(int)
    assert sorted(charges.tolist()) == [-1] + [0] * (len(charges) - 1), (
        f"fragment charges {charges.tolist()} -- a hydroxide cluster must carry exactly one "
        f"anion and no cation"
    )
    frag = group.fragments[0].numpy()
    sizes = sorted(int(((frag == f) & (z == 1)).sum()) for f in set(frag.tolist()))
    assert sizes == [1] + [2] * (len(sizes) - 1), f"hydrogen counts per fragment: {sizes}"


# ---------------------------------------------------------------------------------------
# 4. Continuity where a candidate enters or leaves the enumeration
# ---------------------------------------------------------------------------------------

def _second_difference(model, r_oo: float, d: float, h: float) -> float:
    e = []
    for k in (-1, 0, 1):
        pos, z = _zundel(r_oo + k * h, d)
        group = enumerate_group(pos, z, 1)
        e.append(mixture_forward(model, group, model.mediator).energy)
    return float((e[0] - 2 * e[1] + e[2]).abs().detach())


def test_energy_is_smooth_when_a_candidate_closes(model):
    """No kink where the second host leaves the validity envelope.

    ``hi0 = 2.20 Angstrom`` is where ``Omega`` reaches zero and the candidate is dropped from
    the enumeration entirely. Dropping it also shrinks the contested set, which changes
    ``Omega`` for the survivors -- so this is not only a test of ``validity_bump``'s own C²
    property, it is the test of the argument in ``rsfff.md.assign``'s docstring that the factor
    removed is flat at 1.

    Checked as a convergence rate, following ``tests/test_mediator.py``: for a C² energy the
    second difference falls as ``h^2``, while a step or a kink in the accounting leaves it flat
    or growing. A magnitude threshold would pass either way, because the O-H stretch dominates
    the curvature no matter what the bookkeeping does.
    """
    # Scan r_OO so the bridging proton (fixed at d) drifts out of the far host's envelope.
    # The drop lands between 3.34 and 3.36 -- note it is set by the H...H contact, not the
    # H...O one, because `contact_distance` reads the nearest atom of the host and not its
    # oxygen. Both stencils below straddle it.
    d, centre = 0.98, 3.35
    coarse = _second_difference(model, centre, d, 0.02)
    fine = _second_difference(model, centre, d, 0.01)
    assert coarse > 0.0 and fine > 0.0
    ratio = coarse / fine
    # A looser band than `tests/test_mediator.py` uses, and for a reason: the drop sits *inside*
    # the stencil rather than away from it, so the ratio carries a finite-size term on top of
    # the h^2 scaling. The discriminant is unaffected -- a step in the enumeration pins the
    # ratio near 1, which neither end of this band admits.
    assert 3.0 < ratio < 6.0, (
        f"second difference fell by {ratio:.2f}x when the spacing halved; a C2 energy gives ~4x "
        f"and a step in the enumeration gives ~1x (coarse {coarse:.3e}, fine {fine:.3e}). "
        f"Dropping a closed candidate is not energy-neutral."
    )


def test_candidate_count_actually_changes_across_that_scan():
    """The previous test is vacuous unless the enumeration really does change there."""
    d = 0.98
    inside = enumerate_group(*_zundel(3.30, d), 1).fragments.shape[0]
    outside = enumerate_group(*_zundel(3.40, d), 1).fragments.shape[0]
    assert inside == 2 and outside == 1, (
        f"expected the second candidate to close between r_OO 3.30 and 3.40 (got "
        f"M={inside} inside, "
        f"M={outside} outside); the continuity test is not exercising a drop"
    )


def test_uncontested_geometry_evaluates_without_a_contested_atom(model):
    """``M = 1`` must run, and this used to raise.

    §8 promises that "an atom with one candidate has ``pi = 1`` and costs nothing, so a neutral
    water cluster runs the model of §5-§7 untouched". The corpus is contested by construction,
    so nothing ever exercised it, and ``contact_distance`` reached ``torch.stack([])`` on an
    empty ``D``. Under dynamics the uncontested step is the *majority* of steps -- every
    spectator frame between two transfers -- so this is the path that has to work.
    """
    pos, z = _zundel(3.60, 0.98)
    group = enumerate_group(pos, z, 1)
    assert group.fragments.shape[0] == 1 and group.contested.numel() == 0

    out = mixture_forward(model, group, model.mediator)
    assert torch.isfinite(out.energy)
    assert torch.allclose(out.mediator.weights, torch.ones(1, dtype=torch.float64))
    assert float(ambiguity(out.mediator.weights)) == pytest.approx(0.0, abs=1e-12)

    p = pos.clone().requires_grad_(True)
    e = mixture_forward(model, enumerate_group(p, z, 1), model.mediator).energy
    (grad,) = torch.autograd.grad(e, p)
    assert torch.isfinite(grad).all() and float(grad.abs().max()) > 0.0


# ---------------------------------------------------------------------------------------
# 5. The unbiased limit
# ---------------------------------------------------------------------------------------

def test_zero_bias_is_exactly_the_mediated_energy(model):
    """``k = 0`` must be plain mediated MD **exactly**, not approximately.

    This is the same statement Invariant 1 makes about the mediator itself: the bias is an
    addition to the model, so switching it off has to give the model back bit for bit. A bias
    that returned a small number instead of zero would shift every unbiased reference run.
    """
    pos, z = _zundel(2.45, 1.15)
    group = enumerate_group(pos, z, 1)
    out = mixture_forward(model, group, model.mediator)
    e_bias, _cv = HarmonicBias(k=0.0, target=0.4)(out, group)
    assert float(e_bias) == 0.0
    assert float(flat_bottom_sphere(pos, z, radius=0.0, k=0.0)) == 0.0


# ---------------------------------------------------------------------------------------
# 6-7. The trained model, the calculator, and dynamics
# ---------------------------------------------------------------------------------------

requires_checkpoint = pytest.mark.skipif(
    not CHECKPOINT.exists(), reason=f"{CHECKPOINT} not present"
)


@pytest.fixture(scope="module")
def trained():
    if not CHECKPOINT.exists():
        pytest.skip(f"{CHECKPOINT} not present")
    model, _cfg, _state = load_mediated_model(str(CHECKPOINT))
    return model


@requires_checkpoint
def test_calculator_forces_match_finite_differences(trained):
    """End to end: the number ASE integrates is the gradient of the number ASE reports.

    Model energy, bias and wall in one backward, checked against central differences on the
    total. This is the statement that the whole calculator -- not just each term alone -- is
    conservative, which is what an energy-conserving integrator needs.
    """
    from ase import Atoms

    pos, z = _zundel(2.45, 1.20)
    atoms = Atoms(numbers=z.numpy(), positions=pos.numpy())
    calc = MediatedCalculator(
        trained, 1,
        bias=HarmonicBias(cv="ambiguity", k=0.02, target=0.5),
        radius=3.0, k_confine=0.05, with_induction=False,
    )
    atoms.calc = calc
    forces = atoms.get_forces()

    h = 1e-5
    for atom, axis in ((6, 0), (0, 1), (3, 2)):     # the bridge and both oxygens
        e = []
        for s in (+1, -1):
            shifted = atoms.copy()
            p = shifted.get_positions()
            p[atom, axis] += s * h
            shifted.set_positions(p)
            shifted.calc = MediatedCalculator(
                trained, 1,
                bias=HarmonicBias(cv="ambiguity", k=0.02, target=0.5),
                radius=3.0, k_confine=0.05, with_induction=False,
            )
            e.append(shifted.get_potential_energy())
        numeric = -(e[0] - e[1]) / (2 * h)
        assert numeric == pytest.approx(forces[atom, axis], abs=2e-4), (
            f"atom {atom} axis {axis}: analytic {forces[atom, axis]:.6f} vs finite-difference "
            f"{numeric:.6f} eV/Angstrom"
        )


@requires_checkpoint
def test_calculator_reports_the_unbiased_energy_alongside_the_biased_one(trained):
    """The thermostat sees the biased total; analysis needs the model's own energy.

    Reporting only one of them would mean either a wrong integrator or a second forward pass
    per logged frame, and at ~40 ms a step the second pass is not free.
    """
    from ase import Atoms

    pos, z = _zundel(2.45, 1.20)
    atoms = Atoms(numbers=z.numpy(), positions=pos.numpy())
    bias = HarmonicBias(cv="ambiguity", k=0.5, target=0.5)
    atoms.calc = MediatedCalculator(trained, 1, bias=bias, with_induction=False)
    biased_ev = atoms.get_potential_energy()
    r = atoms.calc.results

    assert r["bias_energy"] > 0.0, "a k=0.5 bias off target must cost something"
    assert r["confine_energy"] == 0.0
    from rsfff.md.calculator import HARTREE_TO_EV
    assert biased_ev == pytest.approx(
        (r["energy_hartree"] + r["bias_energy"]) * HARTREE_TO_EV, rel=1e-12
    )

    atoms.calc = MediatedCalculator(trained, 1, bias=HarmonicBias(k=0.0), with_induction=False)
    atoms.get_potential_energy()
    assert atoms.calc.results["energy_hartree"] == pytest.approx(r["energy_hartree"], rel=1e-12)


@requires_checkpoint
def test_short_nve_trajectory_conserves_energy(trained):
    """Bias and thermostat off: total energy must not drift.

    The end-to-end statement that the reported forces really are ``-dE/dR`` of the reported
    energy. It catches what a single-point finite-difference check cannot -- a term that is
    conservative at one geometry but not along a path, such as an enumeration that flickers.
    """
    from ase import Atoms, units
    from ase.md.verlet import VelocityVerlet
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
    from ase.optimize import FIRE

    pos, z = _zundel(2.45, 1.10)
    atoms = Atoms(numbers=z.numpy(), positions=pos.numpy())
    atoms.calc = MediatedCalculator(
        trained, 1, bias=HarmonicBias(k=0.0), radius=6.0, k_confine=0.02, with_induction=False
    )
    # Relax first. The idealized Zundel above is strained, and released cold it converts that
    # strain into ~4000 K of kinetic energy within twenty steps -- where 0.25 fs is no longer
    # a converged timestep and the 56 meV of *integrator* error swamps anything a
    # discontinuity would contribute. Relaxing makes the test measure what it is named after.
    FIRE(atoms, logfile=None).run(fmax=0.05, steps=200)
    MaxwellBoltzmannDistribution(atoms, temperature_K=50.0, rng=np.random.default_rng(0))
    Stationary(atoms)

    dyn = VelocityVerlet(atoms, timestep=0.25 * units.fs)
    total = []
    for _ in range(40):
        dyn.run(5)
        total.append(atoms.get_total_energy())
    total = np.asarray(total)
    drift = float(abs(total[-1] - total[0]))
    spread = float(total.max() - total.min())
    assert atoms.calc.results["n_commits"] == 0, (
        "the base was rebuilt mid-run; that is a deliberate discrete event and this test "
        "cannot separate it from a defect in the forces"
    )
    assert drift < 5e-3 and spread < 1e-2, (
        f"NVE energy drifted {drift * 1e3:.2f} meV over 200 steps (spread {spread * 1e3:.2f} "
        f"meV) on a 7-atom system; the forces are not the gradient of the reported energy"
    )


@requires_checkpoint
def test_bias_drives_the_membership_to_split(trained):
    """The claim the whole exercise rests on: the bias moves the ambiguity where it wants.

    Relaxing on the biased surface must reach a split membership where the unbiased surface
    reaches a decided one. Without this, the bias is a term that costs time and changes
    nothing.

    ``k = 10`` and not the 0.005-0.02 Ha a naive scaling suggests. The unbiased relaxation
    does not merely sit still -- it runs the O-O distance out to 2.6 Angstrom and localizes
    the proton, and the bias has to outbid that. Measured: ``k = 5`` leaves the ambiguity at
    1e-4, indistinguishable from unbiased; ``k = 10`` reaches 0.4975 at ``r_OO = 2.34`` with
    the proton centred (delta = -0.02). See :class:`rsfff.md.bias.HarmonicBias`.
    """
    from ase import Atoms
    from ase.optimize import FIRE

    pos, z = _zundel(2.45, 0.98)                  # start decided, proton on O1

    def relaxed_ambiguity(k: float) -> float:
        atoms = Atoms(numbers=z.numpy(), positions=pos.numpy())
        atoms.calc = MediatedCalculator(
            trained, 1, bias=HarmonicBias(cv="ambiguity", k=k, target=0.5),
            with_induction=False,
        )
        FIRE(atoms, logfile=None).run(fmax=0.05, steps=60)
        w = np.asarray(atoms.calc.results["weights"])
        return float(1.0 - (w**2).sum())

    plain, biased = relaxed_ambiguity(0.0), relaxed_ambiguity(10.0)
    assert plain < 0.05, (
        f"the unbiased relaxation already reached ambiguity {plain:.3f}; this geometry no "
        f"longer isolates what the bias does"
    )
    assert biased > 0.30, (
        f"the bias moved the ambiguity from {plain:.4f} only to {biased:.4f}; a restraint that "
        f"does not reach the routing weight is not sampling anything new"
    )
