"""The fragment-expert model: the accounting identity, and where the many-body content lives.

Two families of check.

**The accounting.** Every pair appears exactly once, in exactly one bucket, and the pieces sum
to the total. This is the invariant that has to survive when ``is_intra`` eventually softens
into a mixture weight, so it is pinned now rather than after.

**The many-body content, per channel.** ``docs/fff_v2.md`` §6 says which slot each channel reads
and gives the reason for each. That is not a preference -- ``eda_cls_elec`` is the Coulomb
interaction between superimposed frozen monomer densities and so is *rigorously pairwise*, while
the modified Pauli term antisymmetrizes the product of all the monomer densities and the
dispersion is a supersystem difference, so both carry genuine many-body content. The
decomposition here is what says the wiring matches that claim: dispersion and Pauli have real
3-body content, electrostatics has none, and switching the environment slot off takes all of it
away.
"""

from __future__ import annotations

import pytest
import torch

from rsfff.ff.many_body import mbe_decompose
from rsfff.mlip.heads import env_parameters
from rsfff.mlip.reference_states import AtomicStateReference
from rsfff.train.build_expert import build_expert_model
from rsfff.train.config import Config
from rsfff.train.data import load_extxyz, load_reference_energies

from conftest import DATA_W3

NEIGHBOR_TYPES = (1, 8)


def _config(*, environment: bool, induction: bool = False) -> Config:
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
    cfg.expert.environment_features = environment
    cfg.expert.induction = induction
    cfg.expert.bond_hidden, cfg.expert.bond_equiv_channels = 16, 4
    cfg.expert.r0_hidden = 16
    return cfg


@pytest.fixture(scope="module")
def build():
    torch.set_default_dtype(torch.float64)
    e0 = load_reference_energies(
        "data/atomic_references_wb97mv_tzvpd.json", NEIGHBOR_TYPES
    ).double()
    states = AtomicStateReference.from_json(
        "data/atomic_reference_states_wb97mv_tzvpd.json", NEIGHBOR_TYPES,
        dtype=torch.float64,
    )

    def make(*, environment=True, induction=False, wake=True):
        torch.manual_seed(0)
        model = build_expert_model(
            _config(environment=environment, induction=induction),
            NEIGHBOR_TYPES, e0, states,
        ).double()
        if wake:
            with torch.no_grad():
                for module in model.modules():
                    if isinstance(module, torch.nn.Linear):
                        module.weight.normal_(0.0, 0.05)
                for _n, p in env_parameters(model):
                    p.normal_(0.0, 0.05)
        return model

    return make


@pytest.fixture(scope="module")
def w3():
    return load_extxyz(DATA_W3, dtype=torch.float64)


# ---------------------------------------------------------------------------------------
# accounting
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("induction", [False, True])
def test_total_is_the_sum_of_its_parts(build, w3, induction):
    model = build(induction=induction)
    batch = w3.flat_batch([0, 1])
    out = model(batch)

    f2b = batch.fragment_to_batch
    if f2b is None:
        f2b = batch.batch_idx.new_zeros(out.fragment_energy.shape[0]).scatter_(
            0, batch.fragment_idx, batch.batch_idx
        )
    total = out.fragment_energy.new_zeros(batch.n_systems).index_add_(
        0, f2b, out.fragment_energy
    )
    for value in out.interaction.values():
        total = total + value
    assert torch.allclose(out.energy, total, atol=1e-12, rtol=0.0)


def test_fragment_energy_is_its_four_terms(build, w3):
    out = build()(w3.flat_batch([0, 1]))
    assert torch.allclose(
        out.fragment_energy,
        out.energy_ref + out.energy_internal + out.energy_intra + out.energy_bond,
        atol=1e-14, rtol=0.0,
    )


def test_every_pair_lands_in_exactly_one_bucket(build, w3):
    """The invariant that must survive ``is_intra`` softening into a mixture weight."""
    batch = w3.flat_batch([0, 1])
    out = build()(batch)
    pair_batch = batch.batch_idx[out.pair_index[0]]
    inter = (~out.is_intra).to(out.r.dtype)
    intra = out.is_intra.to(out.r.dtype)
    assert torch.equal(inter + intra, torch.ones_like(inter))

    for name, value in out.e_pair.items():
        pooled = value.new_zeros(batch.n_systems).index_add_(
            0, pair_batch, inter * value
        )
        assert torch.allclose(out.interaction[name], pooled, atol=1e-14, rtol=0.0), name

    pooled_intra = out.r.new_zeros(out.energy_intra.shape[0]).index_add_(
        0, out.pair_frag[out.is_intra],
        (intra * sum(out.e_pair.values()))[out.is_intra],
    )
    assert torch.allclose(out.energy_intra, pooled_intra, atol=1e-14, rtol=0.0)
    # inter pairs carry no fragment id, so they cannot be routed to a fragment by accident
    assert int(out.pair_frag[~out.is_intra].max()) == -1


def test_induction_telescopes_through_the_bond_energy(build, w3):
    """``energy_bond + (induction contribution) == energy_bond_ind``, by construction."""
    out = build(induction=True)(w3.flat_batch([0, 1]))
    assert out.energy_bond_ind is not None and out.energy_bond_ind_iso is not None
    # The slot swap alone, separable from the state relaxation. It was 100% of the old `ct`.
    swap = out.energy_bond_ind - out.energy_bond_ind_iso
    assert swap.abs().max() > 0.0, "the h -> [h|eta] swap is inert; ct has nowhere to come from"
    assert "e_bond" in out.env_shift


# ---------------------------------------------------------------------------------------
# many-body content, per channel
# ---------------------------------------------------------------------------------------

class _Channel(torch.nn.Module):
    """Adapts one channel of the model's output to what ``mbe_decompose`` expects."""

    def __init__(self, model, name):
        super().__init__()
        self.model, self.name = model, name

    def forward(self, batch):
        out = self.model(batch)
        return type("E", (), {"energy": out.interaction[self.name]})()


def _many_body(model, name, w3):
    """``|total - E^(2)|`` for one channel on a trimer: everything beyond pairwise."""
    batch = w3.flat_batch([0])
    result = mbe_decompose(
        _Channel(model, name), batch.positions, batch.atomic_numbers,
        batch.fragment_idx, split_components=False,
    )
    return float(result.many_body.abs().sum())


@pytest.mark.parametrize("name", ["pauli", "disp"])
def test_the_many_body_channels_have_real_three_body_content(build, w3, name):
    """These labels are not pairwise, so their parameters read the environment slot."""
    assert _many_body(build(environment=True), name, w3) > 1e-8


@pytest.mark.parametrize("name", ["elst", "pauli", "disp"])
def test_no_channel_has_three_body_content_without_the_environment_slot(build, w3, name):
    """The ablation: fragment-confined parameters make every pair sum rigorously additive."""
    assert _many_body(build(environment=False), name, w3) < 1e-12


def test_electrostatics_stays_pairwise_even_with_the_environment_slot_on(build, w3):
    """``eda_cls_elec`` is a function of the two fragments alone. That is what its label is.

    Contrast the two tests above: this is the same model, the same live environment slot, and
    the channel is *still* exactly additive -- because it reads ``theta_0``.
    """
    assert _many_body(build(environment=True), "elst", w3) < 1e-12
