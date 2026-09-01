"""The neighbor-list cap: explicit everywhere, and loud when it is reached.

``torch_cluster.radius_graph`` truncates at ``max_num_neighbors`` without saying so, and its
default is 32 -- below what a 5 A cutoff produces in bulk-like water. These tests pin the
two things that make truncation impossible to miss: the cap is never the library default,
and reaching it is reported.
"""

import inspect
import warnings

import pytest
import torch
from torch_cluster import radius_graph

from rsfff.features.features import FlatLambdaSOAPFeaturizer, FlatStateSOAPFeaturizer
from rsfff.ff import pairs as pairs_mod
from rsfff.neighbors import (
    CAP_EVENTS,
    DEFAULT_MAX_NUM_NEIGHBORS,
    NeighborCapExceeded,
    build_radius_graph,
    reset_cap_events,
    set_strict,
)


@pytest.fixture(autouse=True)
def _clean_state():
    reset_cap_events()
    yield
    reset_cap_events()
    set_strict(False)


def _dense_blob(n=200, box=3.0, seed=0):
    """n atoms inside a 3 A box: at a 5 A cutoff every atom sees every other."""
    torch.manual_seed(seed)
    return torch.rand(n, 3) * box, torch.zeros(n, dtype=torch.long)


def test_capped_row_is_the_query_row():
    """`_check_cap` counts edge[1]; this is the assumption it rests on.

    With torch_cluster's default flow the returned edges are ``(neighbor, center)``, so the
    per-atom cap shows up as a *uniform* degree on row 1. If a torch_cluster upgrade flips
    that convention, the detector would silently stop detecting -- hence this test.
    """
    pos, batch = _dense_blob()
    edge = radius_graph(pos, r=5.0, batch=batch, loop=False, max_num_neighbors=16)
    degree_center = torch.bincount(edge[1], minlength=pos.shape[0])
    degree_neighbor = torch.bincount(edge[0], minlength=pos.shape[0])
    assert int(degree_center.min()) == int(degree_center.max()) == 16
    assert int(degree_neighbor.max()) != 16 or int(degree_neighbor.min()) == 16


def test_truncation_warns_and_is_counted():
    pos, batch = _dense_blob()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_radius_graph(pos, 5.0, batch, context="unit", max_num_neighbors=16)
    assert len(caught) == 1
    assert "max_num_neighbors=16" in str(caught[0].message)
    assert CAP_EVENTS["unit"] == 1


def test_untruncated_list_is_silent():
    pos, batch = _dense_blob(n=40)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        edge = build_radius_graph(pos, 5.0, batch, context="unit")
    assert edge.shape[1] == 40 * 39            # complete graph, nothing dropped
    assert not caught
    assert CAP_EVENTS == {}


def test_strict_mode_raises():
    pos, batch = _dense_blob()
    set_strict(True)
    with pytest.raises(NeighborCapExceeded, match="max_num_neighbors=16"):
        build_radius_graph(pos, 5.0, batch, context="unit", max_num_neighbors=16)


def test_warning_fires_once_per_context():
    """A truncating training run must not emit a warning per step."""
    pos, batch = _dense_blob()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(5):
            build_radius_graph(pos, 5.0, batch, context="unit", max_num_neighbors=16)
    assert len(caught) == 1
    assert CAP_EVENTS["unit"] == 5             # the count still sees every hit


def test_no_caller_inherits_the_library_default():
    """Every builder in the repo sets the cap; none is left at torch_cluster's 32."""
    builders = [
        pairs_mod.inter_fragment_pairs,
        pairs_mod.union_pairs,
        pairs_mod.union_channels,
        FlatLambdaSOAPFeaturizer.__init__,
        FlatStateSOAPFeaturizer.__init__,
    ]
    for fn in builders:
        default = inspect.signature(fn).parameters["max_num_neighbors"].default
        assert default >= DEFAULT_MAX_NUM_NEIGHBORS, fn


def test_featurizer_passes_its_cap_through():
    feat = FlatLambdaSOAPFeaturizer(
        cutoff=5.0, n_max=3, l_max=2, neighbor_types=(1, 8), max_num_neighbors=16,
    )
    pos, batch = _dense_blob()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        feat._build_edges(pos, batch)
    assert len(caught) == 1
    assert CAP_EVENTS["FlatLambdaSOAPFeaturizer"] == 1


def test_pair_builder_reports_its_own_context():
    pos, batch = _dense_blob()
    frag = torch.arange(pos.shape[0]) // 2       # 2-atom fragments, already sorted
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pairs_mod.inter_fragment_pairs(
            pos, batch, 5.0, fragment_idx=frag, max_num_neighbors=16,
        )
    assert len(caught) == 1
    assert CAP_EVENTS["inter_fragment_pairs"] == 1
