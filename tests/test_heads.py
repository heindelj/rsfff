"""Ported response heads: Voigt helpers, closed-form eigenvalues, PSD path, inits."""

import torch

from rsfff.features import get_backend
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
