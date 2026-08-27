"""The mediator and the mixture: the five invariants of ``docs/fff_v2.md`` §8.

The one that matters most is the first. **Vertex identity**: with a one-hot membership the
mixture must reproduce the single-fragmentation model *exactly*, not approximately. That is
what makes the mediator an addition to the model of §5-§7 rather than a second model competing
with it, and it is checked here against a genuinely separate implementation --
:func:`rsfff.ff.mixture_model.mixture_forward` shares the parameterization with
``FragmentExpertModel.forward`` but assembles the energy independently, so agreement is a real
statement rather than a tautology about a branch.

The rest: partition of unity, swap symmetry, C² continuity through a crossover, and the fact
that charge genuinely moves between fragments once both compliances are open -- the property
the whole mediator exists to permit.
"""

from __future__ import annotations

import pytest
import torch

from rsfff.ff.mediator import (
    MediatorHead,
    align_fragments,
    contact_distance,
    contested_atoms,
)
from rsfff.ff.mixture_model import (
    MixtureGroup,
    intra_pairs_unsorted,
    mixture_forward,
    routing_weight,
    union_pair_list,
)
from rsfff.mlip.heads import env_parameters
from rsfff.mlip.reference_states import AtomicStateReference
from rsfff.train.build_expert import build_expert_model
from rsfff.train.config import Config
from rsfff.train.data import load_reference_energies

NEIGHBOR_TYPES = (1, 8)


def _config() -> Config:
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
    cfg.expert.compositions = ["H2O", "H3O", "HO"]
    cfg.expert.fragment_state_dim = 4
    return cfg


@pytest.fixture(scope="module")
def model():
    torch.set_default_dtype(torch.float64)
    e0 = load_reference_energies(
        "data/atomic_references_wb97mv_tzvpd.json", NEIGHBOR_TYPES
    ).double()
    states = AtomicStateReference.from_json(
        "data/atomic_reference_states_wb97mv_tzvpd.json", NEIGHBOR_TYPES,
        dtype=torch.float64,
    )
    torch.manual_seed(0)
    m = build_expert_model(_config(), NEIGHBOR_TYPES, e0, states).double()
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, torch.nn.Linear):
                mod.weight.normal_(0.0, 0.05)
        for _n, p in env_parameters(m):
            p.normal_(0.0, 0.05)
    return m


def _h5o2(r_shared: float = 1.05) -> torch.Tensor:
    """H5O2+: two waters bridged by a proton at ``r_shared`` from the first oxygen.

    Atom order is O0 H H | O1 H H | H(bridge) -- the bridge last, so neither decomposition
    finds the atoms already grouped and both exercise the unsorted paths.
    """
    return torch.tensor(
        [
            [0.0, 0.0, 0.0],            # 0 O
            [-0.9, 0.4, 0.3],           # 1 H
            [-0.5, -0.8, -0.5],         # 2 H
            [2.45, 0.0, 0.0],           # 3 O
            [3.0, 0.85, 0.15],          # 4 H
            [3.0, -0.75, -0.45],        # 5 H
            [r_shared, 0.02, 0.03],     # 6 H, the bridge
        ],
        dtype=torch.float64,
    )


def _group(r_shared: float = 1.05) -> MixtureGroup:
    """The two decompositions of H5O2+: the bridge with O0, or the bridge with O1."""
    pos = _h5o2(r_shared)
    z = torch.tensor([8, 1, 1, 8, 1, 1, 1])
    #                       O0 H  H  O1 H  H  Hb
    frag = torch.tensor([[0, 0, 0, 1, 1, 1, 0],      # H3O+ | H2O
                         [0, 0, 0, 1, 1, 1, 1]])     # H2O  | H3O+
    # The host carrying the proton is the cation in each case.
    qa = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                       [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]], dtype=torch.float64)
    sa = torch.zeros_like(qa)
    return MixtureGroup(
        positions=pos, atomic_numbers=z, fragments=frag,
        atom_charge=qa, atom_two_s=sa, contested=torch.tensor([6]),
    )


class _Fixed(torch.nn.Module):
    """A mediator returning a prescribed membership, for the invariants that fix ``pi``."""

    def __init__(self, weights):
        super().__init__()
        self.w = torch.as_tensor(weights, dtype=torch.float64)

    def forward(self, inv, fragments, positions, qa, sa, atoms):
        from rsfff.ff.mediator import MediatorOutput

        return MediatorOutput(
            weights=self.w, omega=torch.ones_like(self.w),
            score=torch.zeros_like(self.w),
            rho=contact_distance(positions, fragments, atoms),
            atoms=torch.as_tensor(atoms).reshape(-1),
        )


# ---------------------------------------------------------------------------------------
# Invariant 1: vertex identity
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("induction", [False, True])
@pytest.mark.parametrize("vertex", [0, 1])
def test_one_hot_mixture_reproduces_the_single_fragmentation_model(model, vertex, induction):
    """**The invariant that makes this an addition rather than a second model.**

    A one-hot membership must give back ``FragmentExpertModel.forward`` exactly. The two
    assemble the energy through separate code, so this is a real cross-check: the mixture
    builds its pair list by enumeration and its solve on a union graph blocked by frame, while
    ``forward`` uses a radius graph and a fragment-blocked solve. They agree because with one
    decomposition the union graph *is* that decomposition's channel graph and no channel
    crosses a fragment, so frame blocking and fragment blocking have the same blocks.
    """
    group = _group()
    one_hot = [0.0, 0.0]
    one_hot[vertex] = 1.0
    mixed = mixture_forward(model, group, _Fixed(one_hot), with_induction=induction)

    batch = group.batch(vertex)
    # `forward` needs atoms grouped by fragment; the mixture deliberately does not.
    order = torch.argsort(group.fragments[vertex], stable=True)
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.shape[0])
    from rsfff.train.data import Batch

    sorted_batch = Batch(
        positions=batch.positions[order],
        atomic_numbers=batch.atomic_numbers[order],
        batch_idx=batch.batch_idx[order],
        n_systems=1,
        energy=batch.energy,
        fragment_idx=group.fragments[vertex][order],
        fragment_charge=batch.fragment_charge,
        fragment_two_s=batch.fragment_two_s,
        fragment_to_batch=batch.fragment_to_batch,
        n_fragments=batch.n_fragments,
    )
    reference = model(sorted_batch, with_induction=induction)

    assert torch.allclose(mixed.energy, reference.energy[0], atol=1e-9, rtol=0.0), (
        f"one-hot mixture {float(mixed.energy.detach()):.12f} != forward "
        f"{float(reference.energy[0].detach()):.12f}; Invariant 1 is what makes the mediator an "
        f"addition to the model rather than a competitor to it"
    )


# ---------------------------------------------------------------------------------------
# Invariant 2: partition of unity, and the accounting that rests on it
# ---------------------------------------------------------------------------------------

def test_routing_weight_counts_every_pair_exactly_once(model):
    group = _group()
    w = torch.tensor([0.6, 0.4], dtype=torch.float64)
    pair_index, _ = union_pair_list(group.positions, group.fragments, 12.0)
    b = routing_weight(group.fragments, w, pair_index)
    assert torch.all(b >= 0.0) and torch.all(b <= 1.0)
    # b + (1 - b) = 1 is trivial; the real claim is that a pair intra in *both*
    # decompositions is fully intra and one intra in neither is fully inter.
    both = (group.fragments[0][pair_index[0]] == group.fragments[0][pair_index[1]]) & (
        group.fragments[1][pair_index[0]] == group.fragments[1][pair_index[1]]
    )
    neither = (group.fragments[0][pair_index[0]] != group.fragments[0][pair_index[1]]) & (
        group.fragments[1][pair_index[0]] != group.fragments[1][pair_index[1]]
    )
    assert torch.allclose(b[both], torch.ones_like(b[both]))
    assert torch.allclose(b[neither], torch.zeros_like(b[neither]))
    # Only pairs involving the contested atom may be fractional.
    fractional = (b > 1e-12) & (b < 1.0 - 1e-12)
    touched = (pair_index[0] == 6) | (pair_index[1] == 6)
    assert torch.all(touched[fractional])


@pytest.mark.parametrize("induction", [False, True])
def test_mixture_energy_is_the_sum_of_its_parts(model, induction):
    group = _group()
    out = mixture_forward(model, group, _Fixed([0.55, 0.45]), with_induction=induction)
    total = (
        out.energy_ref + out.energy_internal + out.energy_intra + out.energy_bond
        + sum(out.energy_inter.values()) + out.energy_induction
    )
    assert torch.allclose(out.energy, total, atol=1e-12, rtol=0.0)
    # Induction must actually do something, or the flag is decorative and the mixture total is
    # short by the whole relaxation.
    assert (abs(float(out.energy_induction.detach())) > 1e-9) == induction


# ---------------------------------------------------------------------------------------
# Invariant 3: swap symmetry
# ---------------------------------------------------------------------------------------

def test_weights_do_not_depend_on_enumeration_order(model):
    """The same physical swap enumerated from either end returns the same weights.

    Not by reading symmetric combinations (§8's suggestion) but because the score is computed
    per decomposition and a softmax over decompositions is permutation-equivariant. That is
    exact, and it generalizes past two candidates, which the pairwise sum/squared-difference
    construction does not.
    """
    group = _group()
    med = MediatorHead(p_frag=8, p_env=8)
    torch.manual_seed(1)
    with torch.no_grad():
        for mod in med.modules():
            if isinstance(mod, torch.nn.Linear):
                mod.weight.normal_(0.0, 0.5)
                mod.bias.normal_(0.0, 0.5)
    med = med.double()

    n_atoms = group.positions.shape[0]
    feats = torch.randn(2, n_atoms, 16, dtype=torch.float64)
    a = med(feats, group.fragments, group.positions,
            group.atom_charge, group.atom_two_s, group.contested)
    flip = torch.tensor([1, 0])
    b = med(feats[flip], group.fragments[flip], group.positions,
            group.atom_charge[flip], group.atom_two_s[flip], group.contested)
    assert torch.allclose(a.weights, b.weights[flip], atol=1e-12, rtol=0.0)


# ---------------------------------------------------------------------------------------
# Invariant 4: C2 continuity through the crossover
# ---------------------------------------------------------------------------------------

def _crossover_second_difference(model, med, centre: float, h: float) -> float:
    """``|E(c-h) - 2E(c) + E(c+h)|`` for the mediated energy at spacing ``h``."""
    e = [
        mixture_forward(model, _group(centre + k * h), _MedAdapter(med)).energy
        for k in (-1, 0, 1)
    ]
    return float((e[0] - 2 * e[1] + e[2]).abs().detach())


def test_energy_is_smooth_through_the_crossover(model):
    """The total must have no kink where the membership swaps.

    Tested as a **convergence rate**, not as a magnitude. Proton transfer is a stiff
    coordinate -- the energy really does curve sharply along a breaking O-H bond -- so "the
    second difference is small" measures the bond, not the accounting. What distinguishes a
    smooth mixture from a switched one is how the second difference behaves as the sampling
    tightens: for a C2 function it falls as ``h^2`` (halving ``h`` quarters it), while a step
    or a kink in the accounting leaves it flat or growing.

    A hard ``is_intra`` under a soft membership -- the failure §8 singles out -- fails this
    even though it would pass a magnitude threshold at coarse spacing.
    """
    med = MediatorHead(p_frag=8, p_env=8).double()   # zero readout: the envelope alone
    centre = 1.2                                     # inside the crossover, both candidates open
    coarse = _crossover_second_difference(model, med, centre, 0.02)
    fine = _crossover_second_difference(model, med, centre, 0.01)
    assert coarse > 0.0 and fine > 0.0
    ratio = coarse / fine
    assert 3.0 < ratio < 5.0, (
        f"second difference fell by {ratio:.2f}x when the spacing halved; a C2 energy gives "
        f"4x and a step in the accounting gives ~1x (coarse {coarse:.3e}, fine {fine:.3e})"
    )


def test_membership_is_genuinely_split_through_the_crossover(model):
    """The mediator must actually mix somewhere, or none of the above is being exercised."""
    med = MediatorHead(p_frag=8, p_env=8).double()
    occupancies = []
    for r in torch.linspace(1.0, 1.5, 11, dtype=torch.float64):
        g = _group(float(r))
        out = mixture_forward(model, g, _MedAdapter(med))
        occupancies.append(float(out.mediator.occupancy.detach()))
    assert max(occupancies) > 0.2, (
        f"the membership never split (max occupancy {max(occupancies):.3f}); the validity "
        f"envelope never opens both candidates and the mediator is decorative"
    )


class _MedAdapter(torch.nn.Module):
    """Feed :class:`MediatorHead` the widths this model's slots actually have."""

    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, inv, fragments, positions, qa, sa, atoms):
        width = inv.shape[-1]
        if self.head.width != width:
            self.head = MediatorHead(
                p_frag=width - width // 2, p_env=width // 2
            ).double()
        return self.head(inv, fragments, positions, qa, sa, atoms)


# ---------------------------------------------------------------------------------------
# The point of the exercise: charge crosses a fragment boundary
# ---------------------------------------------------------------------------------------

def test_charge_flows_between_fragments_when_both_candidates_are_open(model):
    """With both compliances nonzero the solve must let charge cross, and conserve overall.

    Under a single fragmentation ``sqe_solve`` is block diagonal over fragments and each
    fragment's charge is pinned to its formal value. A mixture solves on the union graph, whose
    connected component is the frame, so the two halves are free to exchange -- which is what
    a proton transfer *is*. Total charge is still exact.
    """
    group = _group()
    out = mixture_forward(model, group, _Fixed([0.5, 0.5]))
    total = out.charges.sum()
    assert torch.allclose(total, torch.tensor(1.0, dtype=torch.float64), atol=1e-10)

    # Under decomposition 0 the first fragment is {0,1,2,6} and carries +1 exactly. In the
    # mixture it must not: charge has moved.
    left = out.charges[[0, 1, 2, 6]].sum()
    assert not torch.allclose(left, torch.tensor(1.0, dtype=torch.float64), atol=1e-6), (
        "the mixed solve reproduced the fragment-blocked charge exactly, which means the "
        "union channel graph did not actually open a path between the two hosts"
    )


# ---------------------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------------------

def test_contested_atoms_is_invariant_to_fragment_relabeling():
    # Genuinely the same partition {0,1,2,6} | {3,4,5}, with the two ids exchanged. Nothing
    # moved, so nothing is contested -- a raw id comparison would report all seven atoms.
    frag = torch.tensor([[0, 0, 0, 1, 1, 1, 0],
                         [1, 1, 1, 0, 0, 0, 1]])
    assert contested_atoms(frag).numel() == 0
    # Only the bridge moved. The other six atoms all changed *composition* -- their hosts
    # gained or lost a proton -- but none of them changed address.
    frag2 = torch.tensor([[0, 0, 0, 1, 1, 1, 0],
                          [0, 0, 0, 1, 1, 1, 1]])
    assert contested_atoms(frag2).tolist() == [6]
    # ...and the same swap written with the ids exchanged as well.
    frag3 = torch.tensor([[0, 0, 0, 1, 1, 1, 0],
                          [1, 1, 1, 0, 0, 0, 0]])
    assert contested_atoms(frag3).tolist() == [6]


def test_intra_pairs_unsorted_matches_the_sorted_fast_path():
    from rsfff.ff.pairs import intra_fragment_channels

    frag = torch.tensor([0, 0, 0, 1, 1, 1])
    fast, _ = intra_fragment_channels(frag)
    slow = intra_pairs_unsorted(frag)
    assert torch.equal(fast, slow)


# ---------------------------------------------------------------------------------------
# The key layer (docs/fff_v2.md v3)
# ---------------------------------------------------------------------------------------

def test_keys_are_unit_norm(model):
    """Every key the decoder ever sees lies on the same sphere. Both evaluations."""
    from rsfff.ff.mixture_model import intra_pairs_unsorted

    group = _group()
    em = model.emit(
        group.batch(0), group.fragments[0],
        bond_index=intra_pairs_unsorted(group.fragments[0]),
    )
    for name, key in (("k", em.key), ("k0", em.key0)):
        n = key.norm()
        assert torch.allclose(n, torch.ones_like(n), atol=1e-10), (
            f"{name} is not unit norm (min {float(n.min()):.6f}, max {float(n.max()):.6f})"
        )


def test_mix_keys_is_the_identity_at_a_vertex(model):
    """A one-hot membership returns the vertex key to floating-point exactness.

    **Not bitwise, and the reason is worth knowing.** At a vertex the convex sum *is* the
    vertex key, so renormalizing is mathematically the identity -- but it recomputes a norm
    from already-normalized components, and that sum of squares is only 1.0 to rounding. The
    division therefore moves the last bit or two. The energy-level statement (Invariant 1)
    holds at 1e-9 regardless, which is what actually matters; this pins the key-level one at
    the tightest tolerance it can honestly carry.
    """
    from rsfff.ff.keys import mix_keys
    from rsfff.ff.mixture_model import intra_pairs_unsorted

    group = _group()
    keys = [
        model.emit(
            group.batch(m), group.fragments[m],
            bond_index=intra_pairs_unsorted(group.fragments[m]),
        ).key
        for m in range(2)
    ]
    for vertex in (0, 1):
        w = torch.zeros(2, dtype=torch.float64)
        w[vertex] = 1.0
        mixed = mix_keys(keys, w)
        for block in ("k0", "k1", "k2"):
            got, want = getattr(mixed, block), getattr(keys[vertex], block)
            if want is None:
                assert got is None
                continue
            assert torch.allclose(got, want, atol=1e-14, rtol=0.0), (
                f"{block} moved at a one-hot membership by "
                f"{float((got - want).abs().max()):.3e}"
            )


def test_the_key_is_equivariant(model):
    """Rotate the frame: ``k0`` is invariant, ``k1`` and ``k2`` carry the rotation.

    Checked directly rather than through the energy. A normalization that accidentally broke
    equivariance -- normalizing per ``m`` component, say -- would leave the energy of *this*
    frame untouched and only show up once a rotated one appeared.
    """
    from rsfff.ff.mixture_model import intra_pairs_unsorted

    group = _group()
    theta = torch.tensor(0.7, dtype=torch.float64)
    c, s = theta.cos(), theta.sin()
    R = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)

    def keys_for(positions):
        g = MixtureGroup(
            positions=positions, atomic_numbers=group.atomic_numbers,
            fragments=group.fragments, atom_charge=group.atom_charge,
            atom_two_s=group.atom_two_s, contested=group.contested,
        )
        return model.emit(
            g.batch(0), g.fragments[0],
            bond_index=intra_pairs_unsorted(g.fragments[0]),
        ).key

    plain = keys_for(group.positions)
    turned = keys_for(group.positions @ R.T)

    assert torch.allclose(plain.k0, turned.k0, atol=1e-10), "k0 is not rotation invariant"
    if plain.k1 is not None:
        want = torch.einsum("ab,nbk->nak", R, plain.k1)
        assert torch.allclose(want, turned.k1, atol=1e-10), "k1 does not carry the rotation"


def test_the_mixture_stays_inside_the_vertex_interval(model):
    """**The defect v3 exists to remove**, pinned so it cannot come back.

    v2 mixed parameters, and because the classical forms are nonlinear in them the mediated
    energy left the interval spanned by the two vertices by 162 kJ/mol (H5O2+ total) at the
    crossover. Mixing keys and decoding once should keep the mixture between its endpoints, up
    to the genuine nonlinearity of a single decode.

    The tolerance is generous on purpose: the claim is "no longer a spurious well of order the
    bond energy", not "exactly linear", which key mixing does not promise and should not.
    """
    from rsfff.ff.units import KJMOL_PER_HARTREE

    worst = 0.0
    for r in torch.linspace(1.10, 1.35, 11, dtype=torch.float64):
        g = _group(float(r))
        vertices = []
        for vertex in (0, 1):
            w = [0.0, 0.0]
            w[vertex] = 1.0
            vertices.append(float(mixture_forward(model, g, _Fixed(w)).energy.detach()))
        mixed = float(mixture_forward(model, g, _Fixed([0.5, 0.5])).energy.detach())
        lo, hi = min(vertices), max(vertices)
        worst = max(worst, (lo - mixed), (mixed - hi))
    assert worst * KJMOL_PER_HARTREE < 40.0, (
        f"the mixture leaves the vertex interval by {worst * KJMOL_PER_HARTREE:.1f} kJ/mol; "
        f"v2 parameter mixing gave 162 and that is the regression this guards"
    )
