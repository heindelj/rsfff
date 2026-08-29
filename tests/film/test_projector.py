"""Phase-1 tests: StateDescriptor reductions and the projected feature blocks.

What is pinned here:

* the internal/environment blocks reproduce the v4 two-slot descriptor at a one-hot ``C``
  (they are the same masked scatters of the same shared basis);
* the cross block transforms correctly under rotation (invariant at lambda=0, covariant at
  lambda=2) -- it is a new contraction, not a reuse;
* the vertex guarantee is exact: a lone fragment has ``x_env``, ``x_cross`` and ``a_env``
  equal to zero because the sums that build them are empty;
* fragment relabeling (permuting the columns of ``C``) changes nothing observable;
* features are continuous along a ``C(lambda)`` blend.
"""

from __future__ import annotations

import torch
from e3nn import o3

from rsfff.ff.film import StateDescriptor
from rsfff.train.data import Batch

from film_helpers import make_projector, make_state, water_cluster_batch


def test_one_hot_matches_v4_two_slot():
    """At a one-hot C the in/env blocks equal the featurizer's also_cross descriptor pair."""
    batch = water_cluster_batch(2)
    proj = make_projector()
    state = make_state(batch, proj)
    out = proj(batch, state)

    grouped, cross = proj.featurizer(batch, batch.fragment_idx, also_cross=True)
    assert torch.allclose(out.x_in.inv_feats, grouped.inv_feats, atol=1e-14)
    assert torch.allclose(out.x_in.vec_feats, grouped.vec_feats, atol=1e-14)
    assert torch.allclose(out.x_in.equiv_feats, grouped.equiv_feats, atol=1e-14)
    assert torch.allclose(out.x_env.inv_feats, cross.inv_feats, atol=1e-14)
    assert torch.allclose(out.x_env.equiv_feats, cross.equiv_feats, atol=1e-14)


def test_cross_block_rotation():
    """lambda=0 cross invariant; lambda=2 cross covariant under the Wigner D."""
    batch = water_cluster_batch(2)
    proj = make_projector(cross_lambdas=(0, 2))
    state = make_state(batch, proj)
    out = proj(batch, state)

    R = o3.rand_matrix().to(batch.positions.dtype)
    rotated = Batch(**{**batch.__dict__, "positions": batch.positions @ R.T})
    out_rot = proj(rotated, state)

    assert torch.allclose(out_rot.cross_inv, out.cross_inv, atol=1e-10)
    D = o3.Irrep(2, 1).D_from_matrix(R).to(out.x_cross[2].dtype)
    expected = torch.einsum("ij,njp->nip", D, out.x_cross[2])
    assert torch.allclose(out_rot.x_cross[2], expected, atol=1e-10)


def test_vertex_exact_zero():
    """A lone fragment's environment block, cross block, and activity are exact zeros."""
    batch = water_cluster_batch(1)
    proj = make_projector()
    state = make_state(batch, proj)
    out = proj(batch, state)

    assert torch.count_nonzero(out.x_env.inv_feats) == 0
    assert torch.count_nonzero(out.x_env.equiv_feats) == 0
    assert torch.count_nonzero(out.cross_inv) == 0
    assert torch.count_nonzero(out.a_env) == 0

    # And on a cluster they are not zero -- the guarantee is about isolation, not a dead slot.
    cl = water_cluster_batch(2)
    st = make_state(cl, proj)
    out2 = proj(cl, st)
    assert out2.x_env.inv_feats.abs().max() > 0
    assert out2.cross_inv.abs().max() > 0
    assert out2.a_env.max() > 0


def test_fragment_relabeling_invariance():
    """Permuting the fragment columns of C changes no projected quantity."""
    batch = water_cluster_batch(3)
    proj = make_projector()
    state = make_state(batch, proj)
    perm = torch.tensor([2, 0, 1])
    relabeled = state.permute_fragments(perm)

    out = proj(batch, state)
    out_p = proj(batch, relabeled)
    assert torch.equal(out.P_edge, out_p.P_edge)
    assert torch.allclose(out.x_in.inv_feats, out_p.x_in.inv_feats, atol=1e-14)
    assert torch.allclose(out.cross_inv, out_p.cross_inv, atol=1e-14)
    assert torch.equal(state.mixing_measure(), relabeled.mixing_measure())
    assert torch.allclose(
        state.local_conditioning(), relabeled.local_conditioning(), atol=1e-15
    )


def test_mixing_measure_and_blend_continuity():
    """u_i is zero at a vertex, maximal at an even split; features move continuously in lam."""
    batch = water_cluster_batch(2)
    proj = make_projector()
    a = make_state(batch, proj)
    b = a.permute_fragments(torch.tensor([1, 0]))

    assert torch.count_nonzero(a.mixing_measure()) == 0
    mid = StateDescriptor.blend(a, b, 0.5)
    assert torch.allclose(mid.mixing_measure(), torch.full((6,), 0.5), atol=1e-14)

    # Endpoint recovery and continuity of the features along the sweep.
    out_a = proj(batch, a)
    out_0 = proj(batch, StateDescriptor.blend(a, b, 0.0))
    assert torch.allclose(out_a.x_in.inv_feats, out_0.x_in.inv_feats, atol=1e-14)

    lam = 0.5
    eps = 1e-4
    f0 = proj(batch, StateDescriptor.blend(a, b, lam - eps)).x_in.inv_feats
    f1 = proj(batch, StateDescriptor.blend(a, b, lam + eps)).x_in.inv_feats
    scale = f0.abs().max()
    assert (f1 - f0).abs().max() / scale < 1e-2


def test_row_sums_and_census():
    batch = water_cluster_batch(2)
    proj = make_projector()
    state = make_state(batch, proj)
    assert torch.allclose(state.C.sum(dim=1), torch.ones(6), atol=1e-15)
    # census: each water is 2 H + 1 O in this featurizer's species order (H=0, O=1)
    expected = torch.tensor([[2.0, 1.0], [2.0, 1.0]])
    assert torch.equal(state.fragment_counts, expected)
