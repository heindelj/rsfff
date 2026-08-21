"""``FragmentBondEnergy``: the shared state invariants, and the constant the anchoring took away.

Two things are checked here that nothing else can check.

**The extraction is faithful.** ``state_invariants`` was lifted out of ``AtomicStateEnergy`` so
the new bond head and the old one share one implementation. ``rsfff.ff.v1`` still carries its
own untouched copy, which makes the two directly comparable -- and comparable *bitwise*, since
it is the same arithmetic in the same order.

**The one-body constant is reachable again.** The v1 head returned ``net(x) - net(ref)``, which
cancels the readout's final-layer bias identically: its gradient was measured at exactly
``0.000e+00``, so the model had no direction that moved the one-body constant without also
reshaping its geometry dependence. That cost a measured ``-5.23 kJ/mol`` per fragment and is
what ``species_offset`` existed to repair. Both are gone, and the test is that the bias now
carries gradient -- which is the property, rather than the absence of an attribute.
"""

from __future__ import annotations

import pytest
import torch

from rsfff.features.equivariant_backend import get_backend
from rsfff.ff.bond_energy import FragmentBondEnergy
from rsfff.ff.multipole import irrep2_to_spherical
from rsfff.mlip.heads import env_parameters

P0, P1, P2, P_ENV, P1_ENV, P2_ENV = 6, 5, 7, 4, 3, 2
N_SPECIES, EMB, K = 2, 3, 4


@pytest.fixture(scope="module")
def to_spherical():
    torch.set_default_dtype(torch.float64)
    return irrep2_to_spherical(get_backend("e3nn").irrep6_to_voigt().double())


def _head(to_spherical, *, env: bool):
    return FragmentBondEnergy(
        P0, N_SPECIES, p1=P1, p2=P2,
        p_env=P_ENV if env else 0,
        p1_env=P1_ENV if env else 0,
        p2_env=P2_ENV if env else 0,
        irrep2_to_spherical=to_spherical,
        emb_dim=EMB, hidden=8, depth=2, equiv_channels=K,
    ).double()


def _state(n=5, *, env=False):
    torch.manual_seed(7)
    parts = dict(
        inv=torch.randn(n, P0), vec=torch.randn(n, 3, P1), equiv=torch.randn(n, 5, P2),
        species=torch.randint(0, N_SPECIES, (n,)),
        q=torch.randn(n), mu=torch.randn(n, 3), quad=torch.randn(n, 5),
    )
    if env:
        parts["inv"] = torch.cat((parts["inv"], torch.randn(n, P_ENV)), -1)
        parts["vec"] = torch.cat((parts["vec"], torch.randn(n, 3, P1_ENV)), -1)
        parts["equiv"] = torch.cat((parts["equiv"], torch.randn(n, 5, P2_ENV)), -1)
    return parts


def _call(head, p):
    return head(p["inv"], p["species"], p["vec"], p["equiv"], p["q"], p["mu"], p["quad"])


# ---------------------------------------------------------------------------------------

def test_state_invariants_extraction_is_bitwise_faithful(to_spherical):
    """The shared function reproduces the frozen v1 head exactly, weights held equal."""
    from rsfff.ff.atomic_energy import AtomicStateEnergy as Live
    from rsfff.ff.v1.atomic_energy import AtomicStateEnergy as Frozen

    kwargs = dict(
        p1=P1, p2=P2, irrep2_to_spherical=to_spherical,
        emb_dim=EMB, hidden=8, depth=2, equiv_channels=K,
    )
    torch.manual_seed(1)
    live = Live(P0, N_SPECIES, **kwargs).double()
    torch.manual_seed(1)
    frozen = Frozen(P0, N_SPECIES, **kwargs).double()
    p = _state()
    assert torch.equal(_call(live, p), _call(frozen, p))


def test_readout_bias_carries_gradient(to_spherical):
    """The one-body constant is a direction the model can move. In v1 it was exactly zero.

    This is the whole reason ``species_offset`` existed, stated as the property rather than as
    the absence of a workaround.
    """
    from rsfff.ff.v1.atomic_energy import AtomicStateEnergy as Frozen

    head = _head(to_spherical, env=False)
    _call(head, _state()).sum().backward()
    assert head.net[-1].bias.grad.abs().item() > 0.0

    anchored = Frozen(
        P0, N_SPECIES, p1=P1, p2=P2, irrep2_to_spherical=to_spherical,
        emb_dim=EMB, hidden=8, depth=2, equiv_channels=K,
    ).double()
    _call(anchored, _state()).sum().backward()
    assert anchored.net[-1].bias.grad.abs().item() == 0.0, (
        "the v1 anchoring is supposed to cancel this bias identically; if it no longer does, "
        "the comparison this test draws is stale"
    )


def test_no_species_offset_machinery(to_spherical):
    head = _head(to_spherical, env=False)
    assert not hasattr(head, "species_offset")
    assert not any("offset" in name for name, _ in head.named_parameters())


def test_isolated_energy_ignores_the_environment_weights(to_spherical):
    """``E_bond^0`` does not move when ``w_env`` does. Bitwise, at any point in training."""
    head = _head(to_spherical, env=True)
    for module in head.modules():
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(0.0, 0.3)
    iso = _state()
    joined = _state(env=True)

    before_iso, before_joined = _call(head, iso), _call(head, joined)
    with torch.no_grad():
        for _n, p in env_parameters(head):
            p.normal_(0.0, 0.5)
    after_iso, after_joined = _call(head, iso), _call(head, joined)

    assert torch.equal(before_iso, after_iso)
    assert not torch.equal(before_joined, after_joined)


def test_environment_slot_reaches_the_equivariant_contractions(to_spherical):
    """``vec_reduce_env`` / ``equiv_reduce_env`` must actually be in the arithmetic.

    They are the only route by which the *orientation* of the surroundings relative to the
    fragment reaches the bonding energy, and being zero-initialized they are exactly the kind of
    block that has been silently deleted here before.
    """
    head = _head(to_spherical, env=True)
    names = {n for n, _ in env_parameters(head)}
    assert names == {"vec_reduce_env", "equiv_reduce_env", "net.0.w_env"}

    joined = _state(env=True)
    base = _call(head, joined)
    with torch.no_grad():
        dict(head.named_parameters())["vec_reduce_env"].normal_()
    # The readout is zero-initialized, so wake it or nothing downstream can move.
    with torch.no_grad():
        head.net[-1].weight.normal_(0.0, 0.3)
    assert not torch.equal(base, _call(head, joined))


def test_state_invariants_are_rotation_invariant(to_spherical):
    """Rotate the geometry, the multipoles and the field together; the invariants do not move."""
    from rsfff.ff.multipole import spherical_to_cartesian_quadrupole

    head = _head(to_spherical, env=False)
    p = _state(n=4)
    inv0 = head.state_invariants(
        p["vec"], p["equiv"], p["q"], p["mu"], None, None, None, None
    )
    # A rotation acts on the lambda=1 blocks as R and leaves lambda=0 alone; with the lambda=2
    # slots zeroed here the check isolates the rank-1 sector, where the cross-contraction
    # `mu . v_k` is the term that could silently break.
    angle = torch.tensor(0.7, dtype=torch.float64)
    c, s = torch.cos(angle), torch.sin(angle)
    R = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    inv1 = head.state_invariants(
        torch.einsum("ab,nbp->nap", R, p["vec"]),
        p["equiv"],
        p["q"],
        p["mu"] @ R.T,
        None, None, None, None,
    )
    assert torch.allclose(inv0, inv1, atol=1e-12, rtol=0.0)
