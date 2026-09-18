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

from rsfff.features.features import FlatLambdaSOAPFeaturizer, FlatStateSOAPFeaturizer
from rsfff.ff import pairs as pairs_mod
from rsfff.neighbors import (
    BACKEND,
    CAP_EVENTS,
    DEFAULT_MAX_NUM_NEIGHBORS,
    NeighborCapExceeded,
    build_radius_graph,
    radius_graph_torch,
    reset_cap_events,
    set_strict,
)

try:  # only the tests that compare the two implementations need the compiled one
    from torch_cluster import radius_graph
except ImportError:  # pragma: no cover
    radius_graph = None

needs_compiled = pytest.mark.skipif(
    radius_graph is None, reason="compiled torch_cluster is not installed"
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


@needs_compiled
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


# --- the pure-torch fallback ------------------------------------------------------------------
#
# torch_cluster ships only an sdist that imports torch in its setup.py, so a machine without a
# compiler (or without --no-build-isolation) cannot install it. `radius_graph_torch` is the same
# graph in plain torch; these pin that "the same" is meant literally.

def _random_system(seed, max_graphs=3, max_size=25):
    torch.manual_seed(seed)
    sizes = [int(torch.randint(1, max_size, (1,))) for _ in range(int(torch.randint(1, max_graphs + 1, (1,))))]
    batch = torch.cat([torch.full((n,), g, dtype=torch.long) for g, n in enumerate(sizes)])
    return torch.randn(sum(sizes), 3, dtype=torch.float64) * 2.5, batch


def _edges(edge):
    return set(map(tuple, edge.t().tolist()))


@needs_compiled
@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("r", [1.0, 2.5, 5.0])
@pytest.mark.parametrize("loop", [False, True])
def test_fallback_matches_the_compiled_graph(seed, r, loop):
    """Identical edges wherever the cap is not reached -- which is every legitimate call."""
    pos, batch = _random_system(seed)
    cap = 256
    mine = radius_graph_torch(pos, r=r, batch=batch, loop=loop, max_num_neighbors=cap)
    theirs = radius_graph(pos, r=r, batch=batch, loop=loop, max_num_neighbors=cap)
    assert int(torch.bincount(mine[1], minlength=pos.shape[0]).max()) < cap, "cap reached"
    assert _edges(mine) == _edges(theirs)


@needs_compiled
def test_fallback_agrees_about_the_cutoff_itself():
    """A pair sitting exactly on the cutoff is out of both graphs.

    The comparison is strict in the compiled kernel. It is a measure-zero case for random
    coordinates and an everyday one for a built lattice, so it is worth pinning rather than
    leaving to whichever comparison each implementation happened to write.
    """
    pos = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [3.0, 0, 0]], dtype=torch.float64)
    assert _edges(radius_graph_torch(pos, r=1.0, max_num_neighbors=32)) == set()
    assert _edges(radius_graph(pos, r=1.0, max_num_neighbors=32)) == set()
    assert _edges(radius_graph_torch(pos, r=1.001, max_num_neighbors=32)) == {(0, 1), (1, 0)}


@needs_compiled
@pytest.mark.parametrize("seed", range(4))
def test_fallback_truncates_to_the_cap_and_no_further(seed):
    """Over the cap the two differ only in *which* neighbors survive.

    ``torch_cluster`` documents its picks as random and asks its kernel for ``cap + 1`` when
    ``loop=False`` so the self-pair can be dropped, which leaks: an atom can come back with
    ``cap + 1`` neighbors. The fallback removes the self-pair first and keeps the nearest
    ``cap``, so it is deterministic and never over. Either way the environment is wrong, which
    is what ``build_radius_graph`` exists to report.
    """
    pos, batch = _random_system(seed, max_size=30)
    cap = 4
    mine = radius_graph_torch(pos, r=100.0, batch=batch, loop=False, max_num_neighbors=cap)
    theirs = radius_graph(pos, r=100.0, batch=batch, loop=False, max_num_neighbors=cap)
    mine_degree = torch.bincount(mine[1], minlength=pos.shape[0])
    their_degree = torch.bincount(theirs[1], minlength=pos.shape[0])
    assert int(mine_degree.max()) <= cap
    assert bool(((their_degree - mine_degree) <= 1).all())


def test_fallback_keeps_the_nearest_when_it_truncates():
    """Spacings chosen so no atom has two equally near neighbors to choose between."""
    pos = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [2.5, 0, 0], [5.0, 0, 0]], dtype=torch.float64)
    edge = radius_graph_torch(pos, r=10.0, loop=False, max_num_neighbors=1)
    assert {int(t): int(s) for s, t in edge.t().tolist()} == {0: 1, 1: 0, 2: 1, 3: 2}


def test_fallback_row_one_is_the_query_row():
    """The same assumption `_check_cap` rests on, for the fallback."""
    pos = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1.0, 0]], dtype=torch.float64)
    edge = radius_graph_torch(pos, r=1.5, loop=False, max_num_neighbors=32)
    assert torch.bincount(edge[1], minlength=3).tolist() == [2, 2, 2]


def test_fallback_never_crosses_a_graph_boundary():
    pos = torch.tensor([[0.0, 0, 0], [0.05, 0, 0]], dtype=torch.float64)
    batch = torch.tensor([0, 1])
    assert _edges(radius_graph_torch(pos, r=5.0, batch=batch, max_num_neighbors=32)) == set()


def test_fallback_chunking_does_not_change_the_graph():
    pos, batch = _random_system(3, max_size=40)
    whole = radius_graph_torch(pos, r=3.0, batch=batch, max_num_neighbors=64, chunk=10 ** 6)
    chunked = radius_graph_torch(pos, r=3.0, batch=batch, max_num_neighbors=64, chunk=3)
    assert _edges(whole) == _edges(chunked)


def test_fallback_edges_and_empties():
    assert radius_graph_torch(torch.zeros(0, 3), r=1.0).shape == (2, 0)
    assert radius_graph_torch(torch.zeros(1, 3), r=1.0, loop=False).shape == (2, 0)
    assert _edges(radius_graph_torch(torch.zeros(1, 3), r=1.0, loop=True)) == {(0, 0)}


def test_fallback_refuses_an_unsorted_batch():
    pos = torch.randn(4, 3, dtype=torch.float64)
    with pytest.raises(ValueError, match="non-decreasing"):
        radius_graph_torch(pos, r=1.0, batch=torch.tensor([1, 0, 1, 0]))


def test_backend_says_which_one_is_in_use():
    assert BACKEND in ("torch_cluster", "torch")
    assert (BACKEND == "torch_cluster") == (radius_graph is not None)
