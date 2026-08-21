"""The two-slot contract: ``eta`` is zero when there is nothing around, and ``P(h, 0)`` is exact.

``docs/fff_v2.md`` §3-4 makes two claims that the rest of the model is built on top of, and
this file is where they are checked rather than asserted in prose:

1. **The environment slot is identically zero for an isolated fragment.** Not small, not zero
   by an anchoring subtraction between two evaluations of a network, not only at
   initialization -- zero because the sum that builds it is empty.
2. **The isolated evaluation is exact at every point in training.** ``P(h, 0)`` is what the
   model says about a fragment on its own, and it must equal the narrow call bitwise no matter
   what the environment weights have been trained to.

The second is the one worth being paranoid about, because the failure is silent: an isolated
evaluation that quietly picked up a little environment would leave ``fragment_energy`` fitting
a function its label cannot match, and nothing in the loss would say so. Every check below
that touches it randomizes ``w_env`` first, because at initialization ``w_env`` is zero and
*everything* passes vacuously.
"""

from __future__ import annotations

import pytest
import torch

from rsfff.features.features import FlatLambdaSOAPFeaturizer
from rsfff.ff.multipole import irrep2_to_spherical
from rsfff.ff.slots import SlotFeatures
from rsfff.mlip.heads import (
    TwoSlotLinear,
    core_parameters,
    env_parameters,
    mlp,
    two_slot_mlp,
)
from rsfff.train.data import load_extxyz
from rsfff.train.term_loop import parameter_groups

from conftest import DATA_H2O, DATA_W3

P_FRAG, P_ENV, P1_FRAG, P1_ENV, P2_FRAG, P2_ENV = 6, 4, 5, 3, 7, 2
N_SPECIES, EMB = 2, 3


@pytest.fixture(scope="module")
def featurizer():
    torch.set_default_dtype(torch.float64)
    return FlatLambdaSOAPFeaturizer(
        cutoff=5.0, n_max=3, l_max=2, neighbor_types=(1, 8),
        selected_lambdas=(0, 1, 2), density_channels=4,
    ).double()


@pytest.fixture(scope="module")
def cluster():
    return load_extxyz(DATA_W3, dtype=torch.float64).flat_batch([0])


# ---------------------------------------------------------------------------------------
# 1. the descriptor
# ---------------------------------------------------------------------------------------

def test_cross_descriptor_vanishes_for_an_isolated_fragment(featurizer):
    """Every lambda, exactly zero. This is the property the whole design rests on."""
    mono = load_extxyz(DATA_H2O, dtype=torch.float64).flat_batch([0])
    _frag, cross = featurizer(mono, mono.fragment_idx, also_cross=True)
    for name in ("inv_feats", "vec_feats", "equiv_feats"):
        block = getattr(cross, name)
        assert float(block.detach().abs().max()) == 0.0, name


def test_cross_descriptor_is_nonzero_in_a_cluster(featurizer, cluster):
    _frag, cross = featurizer(cluster, cluster.fragment_idx, also_cross=True)
    assert float(cross.inv_feats.detach().abs().max()) > 1e-3


def test_fragment_slot_is_unchanged_by_neighbours(featurizer, cluster):
    """A water's fragment descriptor is the same whether or not the other two are present.

    Exact in exact arithmetic and ~1e-16 in float64: the two calls scatter a different number
    of edges, so the sums accumulate in a different order. That is the same caveat
    ``FlatLambdaSOAPFeaturizer.forward`` already carries for its paired descriptors, and it is
    a property of floating-point addition rather than of the model -- there is no term here
    that a neighbour could contribute to.

    The statement `tests/test_one_body_isolation.py` makes about energies starts here: if the
    fragment slot genuinely moved when a neighbour arrived, nothing downstream could be
    environment-independent however the heads were wired.
    """
    from rsfff.train.data import Batch

    frag_all, _ = featurizer(cluster, cluster.fragment_idx, also_cross=True)
    first = cluster.fragment_idx == 0
    alone = Batch(
        positions=cluster.positions[first],
        atomic_numbers=cluster.atomic_numbers[first],
        batch_idx=cluster.batch_idx[first],
        n_systems=1,
        energy=cluster.energy[:1],
        fragment_idx=cluster.fragment_idx[first],
        n_fragments=1,
    )
    frag_one, cross_one = featurizer(alone, alone.fragment_idx, also_cross=True)
    for name in ("inv_feats", "vec_feats", "equiv_feats"):
        a = getattr(frag_all, name)[first]
        b = getattr(frag_one, name)
        assert torch.allclose(a, b, atol=1e-14, rtol=0.0), name
    assert float(cross_one.inv_feats.detach().abs().max()) == 0.0


def test_grouped_is_identical_across_the_two_modes(featurizer, cluster):
    """``also_cross`` and ``also_ungrouped`` differ only in the *second* descriptor.

    Both take the grouped one from the same masked scatter of the same shared spherical
    harmonics, so a change in one mode that moved the other would mean the two paths had
    drifted apart -- which is exactly what the shared ``RY`` exists to prevent.
    """
    a, _full = featurizer(cluster, cluster.fragment_idx, also_ungrouped=True)
    b, _cross = featurizer(cluster, cluster.fragment_idx, also_cross=True)
    assert torch.equal(a.inv_feats, b.inv_feats)
    assert torch.equal(a.vec_feats, b.vec_feats)
    assert torch.equal(a.equiv_feats, b.equiv_feats)


def test_modes_are_mutually_exclusive_and_cross_needs_groups(featurizer, cluster):
    with pytest.raises(ValueError, match="alternative second descriptors"):
        featurizer(cluster, cluster.fragment_idx, also_ungrouped=True, also_cross=True)
    with pytest.raises(ValueError, match="needs a group_idx"):
        featurizer(cluster, None, also_cross=True)


# ---------------------------------------------------------------------------------------
# 2. SlotFeatures
# ---------------------------------------------------------------------------------------

def test_slot_features_layout(featurizer, cluster):
    slots = SlotFeatures(*featurizer(cluster, cluster.fragment_idx, also_cross=True))
    d = slots.dims
    assert d.has_env and d.p0 == d.p0_frag + d.p0_env
    assert slots.isolated() is slots.frag
    joined = slots.joined()
    assert joined.inv_feats.shape[-1] == d.p0
    # fragment first, and the two halves are the untouched originals
    assert torch.equal(joined.inv_feats[:, : d.p0_frag], slots.frag.inv_feats)
    assert torch.equal(joined.inv_feats[:, d.p0_frag :], slots.env.inv_feats)
    assert slots.joined() is joined                      # cached


def test_slot_features_without_env_is_the_single_slot_model(featurizer, cluster):
    frag, _ = featurizer(cluster, cluster.fragment_idx, also_cross=True)
    slots = SlotFeatures(frag, None)
    assert not slots.dims.has_env
    assert slots.joined() is frag and slots.isolated() is frag
    assert float(slots.env_norm().detach().abs().max()) == 0.0


# ---------------------------------------------------------------------------------------
# 3. TwoSlotLinear -- the exactness claim
# ---------------------------------------------------------------------------------------

def test_narrow_input_equals_zero_padded_input_after_training():
    """The load-bearing one. Must hold with ``w_env`` at an arbitrary trained value."""
    torch.manual_seed(0)
    net = two_slot_mlp(P_FRAG, P_ENV, 8, 2, 3, p_tail=EMB)
    dict(net.named_parameters())["0.w_env"].data.normal_()
    for module in net:
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_()
    h, eta, emb = torch.randn(9, P_FRAG), torch.randn(9, P_ENV), torch.randn(9, EMB)
    narrow = net(torch.cat((h, emb), dim=-1))
    padded = net(torch.cat((h, torch.zeros_like(eta), emb), dim=-1))
    joined = net(torch.cat((h, eta, emb), dim=-1))
    assert torch.equal(narrow, padded)
    assert not torch.equal(narrow, joined), "a trained w_env must actually do something"


def test_two_slot_mlp_is_the_plain_mlp_when_the_env_slot_is_off():
    """``p_env = 0`` must reproduce the old module tree *and its parameter names*.

    ``rsfff.ff.v1`` imports these heads from the live tree to keep
    ``checkpoints/water_staged/best.pt`` loadable; `tests/test_v1_checkpoint.py` is the
    end-to-end guard and this is the local one that says which property it depends on.
    """
    a = two_slot_mlp(P_FRAG, 0, 8, 2, 3, p_tail=EMB)
    b = mlp(P_FRAG + EMB, 8, 2, 3)
    assert [n for n, _ in a.named_parameters()] == [n for n, _ in b.named_parameters()]
    assert all(
        pa.shape == pb.shape for pa, pb in zip(a.parameters(), b.parameters())
    )
    assert not list(env_parameters(a))


def test_two_slot_linear_rejects_a_width_it_cannot_read():
    layer = TwoSlotLinear(P_FRAG, P_ENV, 4, p_tail=EMB)
    with pytest.raises(ValueError, match="expected"):
        layer(torch.randn(2, P_FRAG + P_ENV + EMB + 1))


def test_env_and_core_partition_the_parameters_and_the_decay_groups():
    net = two_slot_mlp(P_FRAG, P_ENV, 8, 2, 3, p_tail=EMB)
    env = dict(env_parameters(net))
    core = dict(core_parameters(net))
    assert set(env) == {"0.w_env"}
    assert set(env) | set(core) == set(dict(net.named_parameters()))
    assert not set(env) & set(core)
    decayed, exempt = parameter_groups(net, 1.0e-4)
    assert exempt["weight_decay"] == 0.0
    assert any(p is env["0.w_env"] for p in exempt["params"])
    assert not any(p is env["0.w_env"] for p in decayed["params"])


# ---------------------------------------------------------------------------------------
# 4. every real parameter head
# ---------------------------------------------------------------------------------------

def _heads():
    """Each entry: ``(name, build(p_env_widths) -> head, call(head, feats) -> tensors)``."""
    from rsfff.ff.dispersion import DispersionParameterHeads
    from rsfff.ff.pauli import PauliMultipoleHeads
    from rsfff.ff.range_heads import RangeSeparationHeads
    from rsfff.ff.response import ElectrostaticParameterHeads
    from rsfff.features.equivariant_backend import get_backend

    backend = get_backend("e3nn")
    voigt = backend.irrep6_to_voigt().double()
    to_sph = irrep2_to_spherical(voigt)
    prior = torch.zeros(N_SPECIES, dtype=torch.float64)

    def disp(env):
        return DispersionParameterHeads(
            P_FRAG, N_SPECIES, log_c6_prior=prior, log_b_prior=prior,
            p_env=P_ENV if env else 0, emb_dim=EMB, hidden=8, depth=2,
            environment_c6=True, environment_b=True,
        ).double()

    def pauli(env):
        return PauliMultipoleHeads(
            P_FRAG, P1_FRAG, N_SPECIES, log_q_prior=prior, log_b_prior=prior,
            dipole_scale=torch.ones(N_SPECIES, dtype=torch.float64), p2=P2_FRAG,
            quad_scale=torch.ones(N_SPECIES, dtype=torch.float64),
            irrep2_to_spherical=to_sph,
            p_env=P_ENV if env else 0, p1_env=P1_ENV if env else 0,
            p2_env=P2_ENV if env else 0,
            emb_dim=EMB, hidden=8, depth=2, equiv_channels=4, max_rank=2,
            environment_q=True, environment_b=True,
        ).double()

    def elec(env):
        return ElectrostaticParameterHeads(
            P_FRAG, P1_FRAG, P2_FRAG, N_SPECIES,
            log_z_prior=prior, log_b_prior=prior,
            irrep6_to_voigt=voigt,
            irrep2_to_spherical_map=to_sph,
            p_env=P_ENV if env else 0, p1_env=P1_ENV if env else 0,
            p2_env=P2_ENV if env else 0,
            emb_dim=EMB, hidden=8, depth=2, equiv_channels=4, max_rank=2,
            environment_chi=True, environment_eta=True,
            environment_z=True, environment_b=True,
        ).double()

    def rng(env):
        return RangeSeparationHeads(
            P_FRAG, N_SPECIES,
            log_r0_prior=torch.zeros(3, N_SPECIES, dtype=torch.float64),
            alpha_init=40.0, p_env=P_ENV if env else 0,
            emb_dim=EMB, hidden=8, depth=2, environment_r0=True,
        ).double()

    return {"dispersion": disp, "pauli": pauli, "electrostatics": elec, "range": rng}


def _call(head, inv, vec, equiv, species):
    """Call one head, whatever calling convention it happens to have, and flatten the result."""
    import inspect

    from rsfff.features.features import LambdaFeatures

    params = inspect.signature(head.forward).parameters
    if "feats" in params:                       # ElectrostaticParameterHeads
        out = head(
            LambdaFeatures(
                inv_feats=inv, equiv_feats=equiv, species_idx=species,
                batch_idx=torch.zeros_like(species), vec_feats=vec,
            )
        )
    elif "vec_feats" in params:                 # PauliMultipoleHeads
        out = head(inv, species, vec, equiv)
    else:                                       # dispersion, range separation
        out = head(inv, species)
    if hasattr(out, "_fields") or hasattr(out, "__dataclass_fields__"):
        import dataclasses

        out = tuple(
            getattr(out, f)
            for f in (
                out._fields if hasattr(out, "_fields")
                else [f.name for f in dataclasses.fields(out)]
            )
        )
    flat = []
    for item in (out if isinstance(out, tuple) else (out,)):
        if isinstance(item, dict):
            flat += [v for _k, v in sorted(item.items())]
        elif item is not None:
            flat.append(item)
    return flat


@pytest.mark.parametrize("name", sorted(_heads()))
def test_head_isolated_evaluation_ignores_the_environment_weights(name):
    """**The load-bearing test.** Train ``w_env`` to anything; ``theta_0`` does not move, bitwise.

    This is what "the split is analytic, not procedural" (``docs/fff_v2.md`` §4) means
    operationally. The one-body sector reads only the isolated evaluation, so if this holds for
    every head then ``fragment_energy`` is environment-independent at every point in training,
    with no freeze and nothing to enforce.

    It is bitwise because the isolated call never materializes a zeroed environment block: it
    contracts the narrow feature tensor against the fragment half of the weights and the
    environment half is not in the arithmetic at all. Compare
    :func:`test_isolated_matches_a_zeroed_environment_block`, which checks the *mathematical*
    equivalence to padding and can only be approximate.
    """
    torch.manual_seed(3)
    torch.set_default_dtype(torch.float64)
    head = _heads()[name](env=True)
    # Wake the zero-initialized readouts, so the head's output is not a constant that would
    # make any of this pass vacuously.
    for module in head.modules():
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(0.0, 0.3)

    n = 5
    inv, eta = torch.randn(n, P_FRAG), torch.randn(n, P_ENV)
    vec, vec_env = torch.randn(n, 3, P1_FRAG), torch.randn(n, 3, P1_ENV)
    equiv, equiv_env = torch.randn(n, 5, P2_FRAG), torch.randn(n, 5, P2_ENV)
    species = torch.randint(0, N_SPECIES, (n,))

    def isolated():
        return _call(head, inv, vec, equiv, species)

    def joined():
        return _call(
            head,
            torch.cat((inv, eta), -1),
            torch.cat((vec, vec_env), -1),
            torch.cat((equiv, equiv_env), -1),
            species,
        )

    before_iso, before_joined = isolated(), joined()
    env = list(env_parameters(head))
    assert env, f"{name}: no environment-slot parameters at all"
    with torch.no_grad():
        for _n, p in env:
            p.normal_(0.0, 0.5)
    after_iso, after_joined = isolated(), joined()

    for a, b in zip(before_iso, after_iso):
        assert torch.equal(a, b), f"{name}: theta_0 moved when w_env changed"
    assert any(
        not torch.equal(a, b) for a, b in zip(before_joined, after_joined)
    ), f"{name}: w_env changed nothing even in the joined evaluation"
    assert any(
        not torch.equal(a, b) for a, b in zip(after_iso, after_joined)
    ), f"{name}: the joined and isolated evaluations agree, so the slot is inert"


@pytest.mark.parametrize("name", sorted(_heads()))
def test_isolated_matches_a_zeroed_environment_block(name):
    """``P(h, 0)`` equals the narrow call -- to rounding, which is all it can be.

    The two differ only in reduction length: padding adds ``p_env`` terms that are exactly
    ``0 * w``, and a longer sum accumulates differently in the last bits. Mathematically it is
    an identity, which is why the model takes the narrow path and never pays for the padding.
    """
    torch.manual_seed(3)
    torch.set_default_dtype(torch.float64)
    head = _heads()[name](env=True)
    for module in head.modules():
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(0.0, 0.3)
    with torch.no_grad():
        for _n, p in env_parameters(head):
            p.normal_(0.0, 0.5)

    n = 5
    inv, eta = torch.randn(n, P_FRAG), torch.randn(n, P_ENV)
    vec, vec_env = torch.randn(n, 3, P1_FRAG), torch.randn(n, 3, P1_ENV)
    equiv, equiv_env = torch.randn(n, 5, P2_FRAG), torch.randn(n, 5, P2_ENV)
    species = torch.randint(0, N_SPECIES, (n,))

    isolated = _call(head, inv, vec, equiv, species)
    padded = _call(
        head,
        torch.cat((inv, torch.zeros_like(eta)), -1),
        torch.cat((vec, torch.zeros_like(vec_env)), -1),
        torch.cat((equiv, torch.zeros_like(equiv_env)), -1),
        species,
    )
    for a, b in zip(isolated, padded):
        assert torch.allclose(a, b, atol=1e-13, rtol=1e-12), name


@pytest.mark.parametrize("name", sorted(_heads()))
def test_head_without_env_slot_is_single_slot_throughout(name):
    """``p_env = 0`` must leave a head with no two-slot machinery anywhere inside it.

    That is what keeps ``checkpoints/water_staged/best.pt`` loadable: ``rsfff.ff.v1`` imports
    these heads from the live tree, and a :class:`TwoSlotLinear` where an ``nn.Linear`` used to
    be would rename ``weight`` to ``w_frag`` and break the load. ``tests/test_v1_checkpoint.py``
    is the end-to-end guard; this says locally which property that guard depends on.
    """
    torch.set_default_dtype(torch.float64)
    off = _heads()[name](env=False)
    assert not list(env_parameters(off))
    assert not any(isinstance(m, TwoSlotLinear) for m in off.modules())
    assert all(not n.endswith("_reduce_env") for n, _ in off.named_parameters())

    on = _heads()[name](env=True)
    assert list(env_parameters(on)), f"{name}: switching the env slot on added nothing"
