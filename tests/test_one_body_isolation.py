"""**The headline invariant.** ``fragment_energy`` cannot see the environment. Ever.

``docs/fff_v2.md`` §1 makes one claim the whole architecture is arranged around:

    No ``eta`` enters ``E_f`` anywhere, so ``fragment_energy`` is exactly the isolated-fragment
    quantity its label is -- at every geometry, at any separation from any neighbour, at every
    point in training, with no freeze and nothing to enforce.

If that is true the staged-fit problem this model was built to fix goes away, because "what an
isolated water is" stops being something a cluster fit can drift. If it is *almost* true, the
one-body term is fitting a function its label cannot match and nothing in the loss will say so.
So it is checked here in the two strongest forms available:

**Bitwise**, against the weights: train every environment-slot parameter to arbitrary values and
``fragment_energy`` does not move a single bit. This is exact, because the isolated evaluation
never puts an environment column into the arithmetic at all.

**Numerically**, against the geometry: put a second molecule 20 Angstrom away, then 3 Angstrom
away, and the first fragment's energy is unchanged to ~1e-14 Hartree. Not bitwise, and the
reason is worth knowing: a frame with six atoms scatters a different number of edges than one
with three, and floating-point addition is not associative. That is the same caveat
``FlatLambdaSOAPFeaturizer.forward`` already carries for its paired descriptors -- there is no
*term* here a neighbour contributes to, only a different summation order.

The 3 Angstrom case is the one that matters. At 20 Angstrom there are no cross-fragment edges
inside the feature cutoff, so ``eta`` is zero and the test would pass on a model that had the
old environment residual. At 3 Angstrom ``eta`` is large, the interaction channels move by tens
of kJ/mol, and only the routing keeps ``fragment_energy`` still.
"""

from __future__ import annotations

import pytest
import torch

from rsfff.mlip.heads import env_parameters
from rsfff.mlip.reference_states import AtomicStateReference
from rsfff.train.build_expert import build_expert_model
from rsfff.train.config import Config
from rsfff.train.data import Batch, load_extxyz, load_reference_energies

from conftest import DATA_H2O

HARTREE_TO_KJMOL = 2625.4996394799
NEIGHBOR_TYPES = (1, 8)


def _config(*, environment: bool = True, induction: bool = True) -> Config:
    cfg = Config()
    cfg.dtype = "float64"
    # A small featurizer: this file is about routing, not about accuracy, and the invariant
    # holds at any width.
    cfg.features.cutoff, cfg.features.n_max, cfg.features.l_max = 5.0, 3, 2
    cfg.features.selected_lambdas = [0, 1, 2]
    cfg.features.density_channels = 4
    cfg.elec.max_rank = 2
    cfg.elec.hidden, cfg.elec.depth, cfg.elec.equiv_channels = 16, 2, 4
    cfg.elec.direct_multipoles = True
    cfg.elec.quadrupole_response = False
    cfg.dispersion.hidden, cfg.pauli.hidden = 8, 8
    cfg.pauli.equiv_channels = 4
    cfg.expert.environment_features = environment
    cfg.expert.induction = induction
    cfg.expert.bond_hidden, cfg.expert.bond_equiv_channels = 16, 4
    cfg.expert.r0_hidden = 16
    cfg.expert.applicability = True
    return cfg


@pytest.fixture(scope="module")
def model_factory():
    torch.set_default_dtype(torch.float64)
    e0 = load_reference_energies(
        "data/atomic_references_wb97mv_tzvpd.json", NEIGHBOR_TYPES
    ).double()
    states = AtomicStateReference.from_json(
        "data/atomic_reference_states_wb97mv_tzvpd.json", NEIGHBOR_TYPES,
        dtype=torch.float64,
    )

    def build(**kwargs):
        torch.manual_seed(0)
        return build_expert_model(
            _config(**kwargs), NEIGHBOR_TYPES, e0, states
        ).double()

    return build


@pytest.fixture(scope="module")
def water():
    """One water, as its own frame."""
    return load_extxyz(DATA_H2O, dtype=torch.float64).flat_batch([0])


def _with_second_copy(mono: Batch, offset: float) -> Batch:
    """The same water plus a translated copy, as one frame with two fragments."""
    shifted = mono.positions + torch.tensor([offset, 0.0, 0.0], dtype=mono.positions.dtype)
    n = mono.positions.shape[0]
    return Batch(
        positions=torch.cat((mono.positions, shifted)),
        atomic_numbers=torch.cat((mono.atomic_numbers, mono.atomic_numbers)),
        batch_idx=torch.zeros(2 * n, dtype=torch.long),
        n_systems=1,
        energy=mono.energy[:1],
        fragment_idx=torch.cat((torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long))),
        n_fragments=2,
    )


def _wake(model):
    """Give every zero-initialized readout and every environment weight a real value.

    Without this the whole file passes vacuously: ``w_env`` starts at zero, so *any* wiring
    would leave the isolated evaluation alone.
    """
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.normal_(0.0, 0.05)
                if module.bias is not None:
                    module.bias.normal_(0.0, 0.05)
        for _name, p in env_parameters(model):
            p.normal_(0.0, 0.05)


# ---------------------------------------------------------------------------------------
# bitwise, against the weights
# ---------------------------------------------------------------------------------------

def test_fragment_energy_is_bitwise_blind_to_the_environment_weights(model_factory, water):
    model = model_factory()
    _wake(model)
    batch = _with_second_copy(water, 3.0)

    before = model(batch)
    with torch.no_grad():
        for _name, p in env_parameters(model):
            p.normal_(0.0, 0.5)
    after = model(batch)

    assert torch.equal(before.fragment_energy, after.fragment_energy)
    assert torch.equal(before.energy_internal, after.energy_internal)
    assert torch.equal(before.energy_bond, after.energy_bond)
    assert torch.equal(before.energy_intra, after.energy_intra)
    # ... while the interaction channels that are *supposed* to respond, did.
    moved = [
        name for name, value in after.interaction.items()
        if not torch.equal(value, before.interaction[name])
    ]
    assert set(moved) >= {"pauli", "disp"}, (
        f"only {moved} responded to the environment weights; if the many-body channels are "
        f"inert this test proves nothing"
    )


def test_the_electrostatic_channel_is_fragment_confined(model_factory, water):
    """``eda_cls_elec`` is rigorously pairwise, so it must read ``theta_0`` like the one-body does.

    Its ``theta - theta_0`` difference is polarization and belongs to induction. With
    ``environment_r0`` off there is no route for it at all, which is what this pins.
    """
    model = model_factory()
    _wake(model)
    batch = _with_second_copy(water, 3.0)

    before = model(batch)
    with torch.no_grad():
        for _name, p in env_parameters(model):
            p.normal_(0.0, 0.5)
    after = model(batch)
    assert torch.equal(before.interaction["elst"], after.interaction["elst"])


def test_applicability_reads_the_environment_and_stays_out_of_the_energy(model_factory, water):
    """``v_f`` is the one quantity here that *should* move with the environment.

    It used to be asserted blind to it. That was the old reading of the head -- "is this
    expert right for this fragment" -- and it is not what the score is for: it says whether
    this *decomposition* is the best description of the system, which is a statement about
    the surroundings. See :class:`rsfff.ff.expert.ApplicabilityHead`.

    What must still hold is that it is an output and not an input: no energy in the model
    reads it, so moving the environment weights may move ``applicability`` freely while
    ``fragment_energy`` stays exactly where it was.
    """
    model = model_factory()
    _wake(model)
    batch = _with_second_copy(water, 3.0)

    before = model(batch)
    assert before.applicability is not None and before.applicability.shape == (2,)
    with torch.no_grad():
        for _name, p in env_parameters(model):
            p.normal_(0.0, 0.5)
    after = model(batch)
    assert not torch.equal(before.applicability, after.applicability)
    assert torch.equal(before.fragment_energy, after.fragment_energy)


def test_still_blind_after_optimizer_steps(model_factory, water):
    """The guarantee is architectural, so it survives training rather than only initialization."""
    model = model_factory()
    _wake(model)
    batch = _with_second_copy(water, 3.0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    alone = model(water).fragment_energy[0].item()
    for _ in range(3):
        opt.zero_grad()
        out = model(batch)
        loss = out.energy.pow(2).sum() + sum(v.pow(2).sum() for v in out.interaction.values())
        loss.backward()
        opt.step()

    assert any(p.grad is not None and p.grad.abs().max() > 0 for _n, p in env_parameters(model))
    after = model(batch)
    after_alone = model(water).fragment_energy[0].item()
    assert abs(after_alone - alone) > 0, "training moved nothing; the test is vacuous"
    assert abs(after.fragment_energy[0].item() - after_alone) < 1e-13


# ---------------------------------------------------------------------------------------
# numerically, against the geometry
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("offset", [20.0, 3.0])
def test_fragment_energy_is_unchanged_by_a_neighbour(model_factory, water, offset):
    """Add a second molecule. The first fragment's energy does not move.

    At 3 Angstrom the interaction channels move by tens of kJ/mol and ``eta`` is large, which is
    what makes this a test of the routing rather than of the cutoff.
    """
    model = model_factory()
    _wake(model)

    alone = model(water)
    together = model(_with_second_copy(water, offset))

    delta = abs(
        together.fragment_energy[0].item() - alone.fragment_energy[0].item()
    ) * HARTREE_TO_KJMOL
    assert delta < 1e-10, f"{delta:.3e} kJ/mol of environment leaked into fragment_energy"

    if offset == 3.0:
        interaction = sum(v.item() for v in together.interaction.values()) * HARTREE_TO_KJMOL
        assert abs(interaction) > 1.0, (
            "the two molecules are not interacting, so this says nothing about routing"
        )
        assert float(together.env_norm.detach().max()) > 1e-3, "eta is zero; nothing was tested"


def test_env_norm_is_exactly_zero_for_a_lone_fragment(model_factory, water):
    """Not small -- zero. The sum that builds ``eta`` is empty."""
    model = model_factory()
    _wake(model)
    assert float(model(water).env_norm.detach().abs().max()) == 0.0
    far = model(_with_second_copy(water, 20.0))
    assert float(far.env_norm.detach().abs().max()) == 0.0


# ---------------------------------------------------------------------------------------
# the ablation
# ---------------------------------------------------------------------------------------

def test_environment_features_off_removes_the_slot_entirely(model_factory, water):
    """``environment_features: false`` is a genuinely two-body model, not a small one."""
    model = model_factory(environment=False)
    _wake(model)
    assert not list(env_parameters(model))
    out = model(_with_second_copy(water, 3.0))
    assert out.env_shift == {}
    assert float(out.env_norm.detach().abs().max()) == 0.0


def test_the_fragment_view_stream_has_no_environment_at_all(model_factory):
    """A batch of many exploded fragments still has ``eta = 0`` on every atom.

    This is what makes the fragment-view stream a direct measurement of the isolated sector
    rather than an approximation to one. It is worth checking as a *batch*: frames are packed
    into one flat tensor, so it rests on the neighbor search respecting ``batch_idx``, and a
    single-frame test would not see a leak across frames.
    """
    from rsfff.train.data import fragment_view, load_extxyz

    from conftest import DATA_W3

    clusters = load_extxyz(DATA_W3, dtype=torch.float64)
    views = fragment_view(clusters, [0, 1, 2, 3])
    assert len(views) == 12
    batch = views.flat_batch(range(12))

    model = model_factory()
    _wake(model)
    out = model(batch, with_induction=False)
    assert float(out.env_norm.detach().abs().max()) == 0.0
    # ... and every interaction channel is an empty sum, so the frame total *is* the fragment
    # energy. That is why `fragment_view` sets `energy` to it, and why the isolated streams
    # pass `with_induction=False`: with the coupled solve on, a lone fragment still relaxes
    # against its own field and the total picks up an energy the label does not know about.
    for name, value in out.interaction.items():
        assert float(value.detach().abs().max()) == 0.0, name
    assert torch.allclose(out.energy, out.fragment_energy, atol=1e-14, rtol=0.0)


def test_a_lone_fragment_has_a_nonzero_induction_residue_with_the_solve_on(model_factory, water):
    """The artifact the isolated streams switch off, pinned so it cannot grow unnoticed.

    ALMO-EDA reports ``eda_pol = eda_ct = 0`` for an isolated monomer by definition. The model
    does not, quite: the coupled level minimizes with the intramolecular electrostatics inside
    the functional while the frozen level adds them afterwards, so the fragment relaxes against
    its own field. It is a property of how the levels are defined, not a bug -- but it is real,
    it is why ``with_induction=False`` exists, and a test that asserted it away would be hiding
    the reason.
    """
    model = model_factory()
    _wake(model)
    with_solve = model(water)
    without = model(water, with_induction=False)

    assert "induction" not in without.interaction
    assert without.level_ind is None
    assert float(with_solve.interaction["induction"].detach().abs().max()) > 0.0
    # The one-body sector is identical either way -- the solve does not touch it.
    assert torch.equal(with_solve.fragment_energy, without.fragment_energy)
