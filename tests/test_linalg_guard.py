"""The finiteness guards in front of the batched dense factorizations (rsfff.linalg_guard).

A non-finite matrix in a batched GPU LU can fault the whole process; the guards factorize the
identity in its place and hand NaN back for that batch entry only.
"""

from __future__ import annotations

import torch

from rsfff.linalg_guard import finite_or_identity, poison
from rsfff.mlip.sqe import sqe_solve

from test_sqe import system  # noqa: F401  -- the H2O + H3O+ ragged batch


def test_finite_matrices_pass_through_untouched():
    g = torch.Generator().manual_seed(0)
    a = torch.randn(4, 3, 3, generator=g, dtype=torch.float64)
    out, ok = finite_or_identity(a)
    assert ok.all() and torch.equal(out, a)


def test_non_finite_entries_become_identity_and_are_poisoned_back():
    a = torch.randn(3, 2, 2, dtype=torch.float64) + 3 * torch.eye(2, dtype=torch.float64)
    a[1, 0, 1] = float("nan")
    a[2, 1, 1] = float("inf")
    safe, ok = finite_or_identity(a)
    assert ok.tolist() == [True, False, False]
    assert torch.isfinite(safe).all() and torch.equal(safe[1], torch.eye(2, dtype=torch.float64))
    x = poison(torch.linalg.inv(safe), ok)
    assert torch.allclose(x[0], torch.linalg.inv(a[0]))
    assert torch.isnan(x[1:]).all()


def test_sqe_nan_in_one_system_stays_in_that_system(system):  # noqa: F811
    ref = sqe_solve(**system)
    eta = system["eta"].clone()
    eta[4] = float("nan")                       # an atom of the H3O+ (system 1)
    sol = sqe_solve(**{**system, "eta": eta})
    assert torch.allclose(sol.charges[:3], ref.charges[:3], atol=1e-13)
    assert torch.isnan(sol.charges[3:]).all()
