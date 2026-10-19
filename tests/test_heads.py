"""Ported response heads: Voigt helpers, closed-form eigenvalues, PSD path, inits."""

import torch

from bb.o3 import get_backend
from rsfff.mlip import (
    AtomicAlphaHead,
    AtomicVectorHead,
    min_eig_sym3x3,
    symmetric_matrix_to_voigt_vector,
    voigt_vector_to_symmetric_matrix,
)


def test_voigt_round_trip():
    v = torch.randn(17, 6)
    m = voigt_vector_to_symmetric_matrix(v)
    assert torch.equal(m, m.transpose(-1, -2))
    assert torch.allclose(symmetric_matrix_to_voigt_vector(m), v)


def test_min_eig_matches_eigvalsh():
    a = torch.randn(64, 3, 3)
    sym = 0.5 * (a + a.transpose(-1, -2))
    expected = torch.linalg.eigvalsh(sym)[:, 0]
    got = min_eig_sym3x3(sym)
    assert (got - expected).abs().max() < 1e-8
    # isotropic (triple root) corner case
    iso = 2.5 * torch.eye(3).expand(4, 3, 3)
    assert (min_eig_sym3x3(iso) - 2.5).abs().max() < 1e-6


def _alpha_head(**kw):
    return AtomicAlphaHead(
        5, 7, 4, get_backend("e3nn").irrep6_to_voigt(),
        hidden=16, depth=1, equiv_channels=3, **kw,
    )


def test_alpha_head_psd_floor():
    head = _alpha_head(positive_isotropic=True, psd_floor=1e-3)
    inv = torch.randn(30, 5)
    emb = torch.randn(30, 4)
    equiv = torch.randn(30, 5, 7)
    alpha = voigt_vector_to_symmetric_matrix(head(inv, emb, equiv))
    eigs = torch.linalg.eigvalsh(alpha)
    assert eigs.min() >= 1e-3 - 1e-10


def test_alpha_head_free_atom_is_isotropic():
    """Zero equivariant features (isolated atom) -> purely isotropic alpha."""
    head = _alpha_head()
    alpha = voigt_vector_to_symmetric_matrix(
        head(torch.zeros(3, 5), torch.randn(3, 4), torch.zeros(3, 5, 7))
    )
    off = alpha - torch.diag_embed(alpha.diagonal(dim1=-2, dim2=-1))
    assert off.abs().max() < 1e-12
    d = alpha.diagonal(dim1=-2, dim2=-1)
    assert (d - d[:, :1]).abs().max() < 1e-12


def test_vector_head_zero_init():
    head = AtomicVectorHead(5, 6, 4, hidden=16, depth=1, equiv_channels=3)
    out = head(torch.randn(10, 5), torch.randn(10, 4), torch.randn(10, 3, 6))
    assert out.abs().max() == 0.0


# --------------------------------------------------------------------------- pre-bb checkpoints
def _old_keys(state):
    """The same weights under the pre-bb key names (``gate_mlp`` / ``base_mlp``, ``equiv_reduce``)."""
    out = {}
    for k, v in state.items():
        k = k.replace("alpha.mlp.", "alpha.base_mlp.").replace("vec.mlp.", "vec.gate_mlp.")
        k = k.replace("reduce_env.2e", "equiv_reduce_env").replace("reduce_env.1o", "equiv_reduce_env")
        k = k.replace("reduce.2e", "equiv_reduce").replace("reduce.1o", "equiv_reduce")
        out[k] = v
    return out


def _two_heads(seed):
    torch.manual_seed(seed)
    m = torch.nn.Module()
    m.alpha = _alpha_head(p_env=2, p2_env=3)
    m.vec = AtomicVectorHead(5, 6, 4, p_env=2, p1_env=3, hidden=16, depth=1, equiv_channels=3)
    with torch.no_grad():
        for p in m.parameters():
            p.normal_()
    return m


def test_old_checkpoint_keys_load_strictly():
    src, dst = _two_heads(0), _two_heads(1)
    old = _old_keys(src.state_dict())
    assert "vec.gate_mlp.0.w_env" in old and "alpha.equiv_reduce" in old
    dst.load_state_dict(old, strict=True)
    for (n, a), (_, b) in zip(src.state_dict().items(), dst.state_dict().items()):
        assert torch.equal(a, b), n
    assert dst.vec.gate_mlp is dst.vec.mlp and dst.alpha.equiv_reduce is dst.alpha.reduce["2e"]


def test_warm_start_maps_old_keys_and_still_refuses_a_dead_reduction(tmp_path):
    """``warm_start`` matches checkpoint keys by name, so it must rename pre-bb keys first --
    otherwise every head in an old checkpoint is silently "not in this model" and dropped."""
    from rsfff.train.term_loop import warm_start

    src, dst = _two_heads(0), _two_heads(1)
    fresh_vec_reduce = dst.vec.reduce["1o"].detach().clone()
    old = _old_keys(src.state_dict())
    old["vec.equiv_reduce"] = torch.zeros_like(old["vec.equiv_reduce"])   # a deleted block
    path = tmp_path / "old.pt"
    torch.save({"model_state": old}, path)

    warm_start(dst, str(path))
    assert torch.equal(dst.alpha.mlp[0].w_frag, src.alpha.mlp[0].w_frag)
    assert torch.equal(dst.alpha.reduce["2e"], src.alpha.reduce["2e"])
    assert torch.equal(dst.vec.mlp[-1].weight, src.vec.mlp[-1].weight)
    # the zeroed reduction was refused under its old name: this model's own value is kept
    assert torch.equal(dst.vec.reduce["1o"], fresh_vec_reduce)
