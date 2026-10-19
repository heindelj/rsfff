"""The unrolled solve (``rsfff.ff.unrolled_solve``) against the converged one.

What has to hold (``docs/fff_nonvariational.md`` §3, §7): with the physical widths and enough
iterations the loop *is* the CG solution; ``K = 0`` is the uncoupled fragment response (the CG
preconditioner); the response matrix is symmetric for every ``K``, any widths and any global
weights; the response is linear in the drive; a far-apart pair does not feel the width swap;
and the whole thing is differentiable twice, including through the widths.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from rsfff.ff.coupled_solve import (
    CoupledSystem,
    _Preconditioner,
    _grad_state,
    _spherical_to_poly_map,
    coupled_energy,
    coupled_solve_dense,
    multipoles_from_state,
    pcg,
    zero_state,
)
from rsfff.ff.electrostatics import slater_elec_tensors
from rsfff.ff.multipole import build_polytensor
from rsfff.ff.units import BOHR_ANG
from rsfff.ff.unrolled_solve import (
    build_mutual_operator,
    fragment_block_solve,
    internal_solve,
    unrolled_solve,
)


@pytest.fixture(autouse=True)
def _float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def make_geometry(n_frag=3, seed=0, spacing=3.0):
    """Three-atom fragments on a line, bonded-length atoms around each centre."""
    g = torch.Generator().manual_seed(seed)
    n_at = 3 * n_frag
    centers = torch.zeros(n_frag, 3)
    centers[:, 0] = spacing * torch.arange(n_frag, dtype=torch.get_default_dtype())
    offs = torch.randn(n_at, 3, generator=g)
    offs = 0.96 * offs / offs.norm(dim=-1, keepdim=True)
    return centers.repeat_interleave(3, dim=0) + offs


def make_system(positions, *, max_rank=2, seed=0, b=None, gate_intra=False):
    """A coupled system built the way the model builds it: ``slater_elec_tensors`` with
    per-atom ``b``, inter-fragment pairs on (intra pairs gated off, as under exclusions),
    complete intra-fragment SQE channels. Returns ``(sys, b, gate)``."""
    g = torch.Generator().manual_seed(seed + 100)
    n_at = positions.shape[0]
    n_frag = n_at // 3
    frag = torch.repeat_interleave(torch.arange(n_frag), 3)

    rows, cols = [], []
    for f in range(n_frag):
        o = 3 * f
        for a in range(3):
            for c in range(a + 1, 3):
                rows.append(o + a)
                cols.append(o + c)
    bond_index = torch.tensor([rows, cols], dtype=torch.long)
    pi = [(a, c) for a in range(n_at) for c in range(a + 1, n_at)]
    pair_index = torch.tensor(pi, dtype=torch.long).t().contiguous()
    i, j = pair_index[0], pair_index[1]
    gate = ((frag[i] != frag[j]) | gate_intra).to(positions.dtype)

    z = torch.rand(n_at, generator=g) + 0.5
    if b is None:
        b = torch.rand(n_at, generator=g) + 1.5
    dr = (positions[j] - positions[i]) / BOHR_ANG
    r = dr.norm(dim=-1)
    tp, tss, t1i, t1j = (
        gate[:, None, None] * t for t in slater_elec_tensors(dr, r, b, pair_index, max_rank=max_rank)
    )
    alpha = None
    if max_rank >= 1:
        a = torch.randn(n_at, 3, 3, generator=g) * 0.3
        alpha = a @ a.transpose(-1, -2) + 0.5 * torch.eye(3)
    sys = CoupledSystem(
        n_systems=1, n_atoms=n_at,
        batch_idx=torch.zeros(n_at, dtype=torch.long),
        bond_index=bond_index,
        bond_batch=torch.zeros(bond_index.shape[1], dtype=torch.long),
        pair_index=pair_index, max_rank=max_rank,
        chi=torch.randn(n_at, generator=g) * 0.1,
        eta=torch.rand(n_at, generator=g) + 0.5,
        q0=torch.randn(n_at, generator=g) * 0.1,
        compliance=torch.rand(bond_index.shape[1], generator=g) * 0.5,
        chivec=torch.randn(n_at, 3, generator=g) * 0.05 if max_rank >= 1 else None,
        alpha=alpha,
        chiquad=torch.randn(n_at, 5, generator=g) * 0.05 if max_rank >= 2 else None,
        cquad=torch.rand(n_at, generator=g) + 0.5 if max_rank >= 2 else None,
        t_point=tp, t_ss=tss, t_1c_i=t1i, t_1c_j=t1j,
        m_nuc=build_polytensor(z, None, None, max_rank=max_rank),
    )
    return sys, b, gate


def _frag(sys):
    return torch.repeat_interleave(torch.arange(sys.n_atoms // 3), 3), sys.n_atoms // 3


def _max_diff(a, b):
    return max(float((x - y).abs().max()) if x.numel() else 0.0 for x, y in zip(a, b))


def _flatten(x):
    return torch.cat([t.reshape(-1) for t in x])


def total_energy(sys, x, d_map):
    """``E(x) - E(0) = b^T x + 1/2 x^T A x`` through the solver's own gradient."""
    zero = zero_state(sys, torch.float64, "cpu")
    g0 = _flatten(_grad_state(sys, zero, d_map))
    gx = _flatten(_grad_state(sys, x, d_map))
    xf = _flatten(x)
    return 0.5 * xf @ (gx - g0) + g0 @ xf


def solve(sys, mutual, k, **kw):
    frag, n = _frag(sys)
    return unrolled_solve(sys, mutual, n_iter=k, group=frag, n_groups=n, **kw)


# --- the fragment-internal solve -------------------------------------------------------------

def test_internal_solve_zeroes_the_uncoupled_gradient():
    sys, _, _ = make_system(make_geometry(3), max_rank=2)
    d_map = _spherical_to_poly_map(torch.float64, "cpu")
    frag, n = _frag(sys)
    block = fragment_block_solve(sys, frag, n)
    na = sys.n_atoms
    x = internal_solve(sys, (torch.zeros(na), torch.zeros(na, 3), torch.zeros(na, 5)), block)
    unc = replace(sys, t_point=0 * sys.t_point, t_ss=0 * sys.t_ss,
                  t_1c_i=0 * sys.t_1c_i, t_1c_j=0 * sys.t_1c_j)
    assert _max_diff(_grad_state(unc, x, d_map), zero_state(unc, torch.float64, "cpu")) < 1e-12
    assert _max_diff(x, coupled_solve_dense(unc)) < 1e-10


def test_block_solve_per_fragment_equals_per_frame():
    sys, _, _ = make_system(make_geometry(3), max_rank=1)
    rhs = torch.randn(sys.bond_index.shape[1])
    frag, n = _frag(sys)
    assert torch.allclose(
        fragment_block_solve(sys, frag, n)(rhs),
        fragment_block_solve(sys, sys.batch_idx, sys.n_systems)(rhs), atol=1e-12,
    )


def test_closed_channels_need_no_floor():
    sys, _, _ = make_system(make_geometry(2), max_rank=0)
    sys = replace(sys, compliance=torch.zeros_like(sys.compliance))
    rhs = torch.randn(sys.bond_index.shape[1])
    frag, n = _frag(sys)
    v = fragment_block_solve(sys, frag, n)(rhs)
    assert torch.allclose(v, rhs)                      # (I + L 0) v = rhs, finite


def test_k_zero_is_the_preconditioner():
    sys, _, _ = make_system(make_geometry(3), max_rank=2)
    d_map = _spherical_to_poly_map(torch.float64, "cpu")
    x0, _ = solve(sys, build_mutual_operator(sys, b_ind=None), 0)
    g0 = _grad_state(sys, zero_state(sys, torch.float64, "cpu"), d_map)
    pre = _Preconditioner(sys, floor=0.0)((-g0[0], -g0[1], -g0[2]))
    assert _max_diff(x0, pre) < 1e-9


# --- the loop against CG ----------------------------------------------------------------------

@pytest.mark.parametrize("max_rank", [0, 1, 2])
def test_converged_loop_with_physical_widths_is_the_cg_solution(max_rank):
    sys, _, _ = make_system(make_geometry(3), max_rank=max_rank)
    x_star, _ = pcg(sys, rtol=1e-13, atol=1e-15)
    x, info = solve(sys, build_mutual_operator(sys, b_ind=None), 40, with_residual=True)
    assert _max_diff(x, x_star) < 1e-11
    assert float(info.residual.max()) < 1e-13


def test_error_decreases_and_energy_stays_above_the_minimum():
    sys, _, _ = make_system(make_geometry(3), max_rank=2)
    d_map = _spherical_to_poly_map(torch.float64, "cpu")
    x_star = coupled_solve_dense(sys)
    e_star = total_energy(sys, x_star, d_map)
    mutual = build_mutual_operator(sys, b_ind=None)
    errs, gaps = [], []
    for k in range(5):
        x, _ = solve(sys, mutual, k)
        errs.append(_max_diff(x, x_star))
        gaps.append(float(total_energy(sys, x, d_map) - e_star))
    assert all(a > b for a, b in zip(errs, errs[1:]))
    # E(x) - E(x*) = 1/2 r^T A^-1 r >= 0 for a PD functional, second order in the error
    assert all(g >= -1e-14 for g in gaps)
    assert gaps[2] < 1e-2 * gaps[0]


def test_iterate_weights_of_one_are_the_plain_iterate_and_others_move_it():
    sys, _, _ = make_system(make_geometry(3), max_rank=1)
    mutual = build_mutual_operator(sys, b_ind=None)
    x_plain, _ = solve(sys, mutual, 3)
    x_ones, _ = solve(sys, mutual, 3, iterate_weights=torch.ones(3))
    x_w, _ = solve(sys, mutual, 3, iterate_weights=torch.tensor([1.0, 0.5, 2.0]))
    assert _max_diff(x_plain, x_ones) < 1e-14
    assert _max_diff(x_plain, x_w) > 1e-7


# --- symmetry, linearity, locality ------------------------------------------------------------

def _response_matrix(sys, mutual, k, weights=None):
    """``d(q, mu)/d(chi, chivec)`` column by column: the drive enters as ``P^T``, so this is
    ``-P R P^T`` and its symmetry is reciprocity of the induced-moment response."""
    d_map = _spherical_to_poly_map(torch.float64, "cpu")
    n = sys.n_atoms

    def moments(chi, chivec):
        s = replace(sys, chi=chi, chivec=chivec)
        x, _ = solve(s, mutual, k, iterate_weights=weights, d_map=d_map)
        q, mu, _ = multipoles_from_state(s, x, d_map)
        return torch.cat([q, mu.reshape(-1)])

    m0 = moments(sys.chi, sys.chivec)
    cols, eps = [], 1e-4
    for a in range(n):
        chi = sys.chi.clone(); chi[a] += eps
        cols.append((moments(chi, sys.chivec) - m0) / eps)
    for a in range(n):
        for c in range(3):
            cv = sys.chivec.clone(); cv[a, c] += eps
            cols.append((moments(sys.chi, cv) - m0) / eps)
    return torch.stack(cols, dim=1)


@pytest.mark.parametrize("k", [0, 1, 3])
def test_response_matrix_is_symmetric_with_learned_widths_and_weights(k):
    pos = make_geometry(3)
    sys, b, gate = make_system(pos, max_rank=1)
    b_ind = b * torch.exp(-torch.rand(sys.n_atoms, generator=torch.Generator().manual_seed(3)))
    mutual = build_mutual_operator(sys, b_ind=b_ind, positions=pos, gate=gate)
    weights = torch.tensor([1.0, 0.7, 1.3])[:k] if k else None
    G = _response_matrix(sys, mutual, k, weights)
    assert torch.allclose(G, G.T, atol=1e-6 * (1 + float(G.abs().max())))


def test_response_is_linear_in_the_drive():
    sys, _, _ = make_system(make_geometry(3), max_rank=2)
    mutual = build_mutual_operator(sys, b_ind=None)
    x1, _ = solve(sys, mutual, 3)
    doubled = replace(sys, chi=2 * sys.chi, chivec=2 * sys.chivec, chiquad=2 * sys.chiquad,
                      q0=2 * sys.q0, m_nuc=2 * sys.m_nuc)
    x2, _ = solve(doubled, mutual, 3)
    assert _max_diff(tuple(2 * t for t in x1), x2) < 1e-10


def test_mutual_operator_with_physical_widths_is_the_physical_shell_operator():
    pos = make_geometry(2)
    sys, b, gate = make_system(pos, max_rank=2)
    dm = torch.randn(sys.n_atoms, 10)
    op_phys = build_mutual_operator(sys, b_ind=None)
    op_re = build_mutual_operator(sys, b_ind=b, positions=pos, gate=gate)
    assert torch.allclose(op_phys(dm), op_re(dm), atol=1e-10)


@pytest.mark.parametrize("spacing, expect_move", [(12.0, False), (2.6, True)])
def test_far_fragments_do_not_feel_the_width_swap(spacing, expect_move):
    pos = make_geometry(2, spacing=spacing)
    sys, b, gate = make_system(pos, max_rank=1)
    x_phys, _ = solve(sys, build_mutual_operator(sys, b_ind=None), 3)
    x_wide, _ = solve(sys, build_mutual_operator(sys, b_ind=0.5 * b, positions=pos, gate=gate), 3)
    moved = _max_diff(x_phys, x_wide)
    assert (moved > 1e-6) == expect_move, (spacing, moved)


# --- derivatives ----------------------------------------------------------------------------

def test_loop_is_twice_differentiable_including_through_the_widths():
    pos = make_geometry(2, seed=4).requires_grad_(True)
    sys0, b0, gate = make_system(pos.detach(), max_rank=1, seed=4)
    b = b0.clone().requires_grad_(True)
    d_map = _spherical_to_poly_map(torch.float64, "cpu")

    def energy(pos, b):
        i, j = sys0.pair_index[0], sys0.pair_index[1]
        dr = (pos[j] - pos[i]) / BOHR_ANG
        r = dr.norm(dim=-1)
        tp, tss, t1i, t1j = (gate[:, None, None] * t for t in
                             slater_elec_tensors(dr, r, b, sys0.pair_index, max_rank=1))
        live = replace(sys0, t_point=tp, t_ss=tss, t_1c_i=t1i, t_1c_j=t1j)
        mutual = build_mutual_operator(live, b_ind=0.8 * b, positions=pos, gate=gate)
        x, _ = solve(live, mutual, 3, d_map=d_map)
        return total_energy(live, x, d_map)

    e = energy(pos, b)
    (f,) = torch.autograd.grad(e, pos, create_graph=True)
    gb, gp = torch.autograd.grad(f.pow(2).sum(), (b, pos))
    assert torch.isfinite(gb).all() and torch.isfinite(gp).all()

    eps = 1e-5
    for a, c in ((0, 0), (4, 2)):
        p = pos.detach().clone(); p[a, c] += eps
        m = pos.detach().clone(); m[a, c] -= eps
        fd = (energy(p, b.detach()) - energy(m, b.detach())) / (2 * eps)
        assert abs(float(f[a, c]) - float(fd)) < 1e-7
    eb = torch.autograd.grad(energy(pos.detach(), b), b)[0]
    bp = b.detach().clone(); bp[2] += eps
    bm = b.detach().clone(); bm[2] -= eps
    fd = (energy(pos.detach(), bp) - energy(pos.detach(), bm)) / (2 * eps)
    assert abs(float(eb[2]) - float(fd)) < 1e-7
