"""Periodic boundary conditions for the complete model: pair list with lattice shifts, Ewald sums in the model's own
polytensor convention, the matrix-free induction solve and the dispersion tail.

Every long-range quantity of the model is the bare Coulomb tensor of point sources (the ions' net charges, the
induced dipoles / quadrupoles) minus a short-range complement from the radial table that vanishes with the overlap.
Under PBC the bare part splits the Ewald way. Each inverse power of the damped tensor is screened by

    g_n(r) = B_n(r) r^(2n+1) / (2n-1)!!,    g_0 = erfc(a r),   g_n = g_(n-1) + (2 a^2 r^2)^n e^(-a^2 r^2) / (a r sqrt(pi) (2n-1)!!)

(Smith's B_n of erfc(a r)/r), so the real-space tensor is ``damped_interaction_tensor(dr, g)`` and the cluster code
is the special case g_n = 1. :func:`rsfff.ff.complete.tabulate.batched_precompute` multiplies the net charge (and
the 1 of the coupling lam = 1 - f) by g_n on a pair list that includes lattice images; the rest is the reciprocal
sum, the self term and the neutralizing background, :class:`EwaldFrame`, applied once to the charges and at every
solver step to the induced moments.

Convention: ``out_i[b] = sum_j sum_a M_j[a] T(X_j - X_i)[a, b]`` over every j and every lattice image except i
itself, M the (N, 10) polytensors of :func:`rsfff.ff.multipole.build_polytensor` and T the tensor of
:func:`rsfff.ff.multipole.damped_interaction_tensor`, T[a, b] = (-1)^|m_b| d^(m_a + m_b) (1/r) with m the monomials
(1, x, y, z, xx, xy, xz, yy, yz, zz). Column 0 of ``out`` is the potential and ``out @ P`` (P = ``_poly_map``) the
drive of the induced moments, as in the cluster code. Tin-foil boundary; a charged cell gets the uniform
neutralizing background (energy -pi Q^2 / (2 V a^2)).

The induction solve under PBC is preconditioned CG on the operator ``A x = D x + sum_pairs B_ij x_j + P^T recip(P x)``
(never assembled), with first-order implicit differentiation: ``x = x* - A^-1 r(x*)`` where ``r = A x* + L`` is zero
to tolerance and the inverse is applied only in the backward pass (one more CG). Forces and stress are exact to the
CG tolerance; second derivatives (force training) are not supported -- train on clusters, run MD here.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace

import torch

from ..ionct import kernels as K

__all__ = ["EwaldFrame", "PolarizationCatastrophe", "dispersion_switch", "dispersion_tail_integral", "ewald_alpha",
           "erfc_screening", "make_ewald_frames", "periodic_pairs", "recip_apply", "solve_induction_periodic",
           "smooth_switch"]

BOHR = K.BOHR_ANG
NORD = 5
# exponents of the 10 polytensor components (1, x, y, z, xx, xy, xz, yy, yz, zz)
_MONO10 = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2))
_ORDER = (0, 1, 1, 1, 2, 2, 2, 2, 2, 2)


class PolarizationCatastrophe(RuntimeError):
    """The induction operator lost positive definiteness (a CG curvature p.Ap <= 0)."""


# ---- pair list --------------------------------------------------------------------------------------------------

def _torchff_cell_list():
    try:
        import torchff_nblist  # noqa: F401  (registers torch.ops.torchff.build_neighbor_list_*)
        return torch.ops.torchff.build_neighbor_list_cell_list
    except Exception:
        return None


def _pairs_one(X, C, rc, use_torchff=True):
    """Pairs of one frame: X (n, 3), cell C (3, 3) rows = lattice vectors, same length unit as rc."""
    n = X.shape[0]
    inv = torch.linalg.inv(C)
    f = X @ inv
    h = 1.0 / inv.norm(dim=0)                                        # spacing of the lattice planes
    nmax = [int(math.floor(rc / float(ha) + 0.5)) for ha in h]
    if use_torchff and X.is_cuda and max(nmax) == 0:
        op = _torchff_cell_list()
        if op is not None:
            pairs, npairs = op(X.contiguous(), C.contiguous(), rc, -1, None, None, False)
            pairs = pairs[: int(npairs)]
            i, j = pairs.min(1).values, pairs.max(1).values
            return torch.stack([i, j]), -torch.round(f[j] - f[i])
    shifts = torch.tensor(list(itertools.product(*[range(-m, m + 1) for m in nmax])), dtype=X.dtype, device=X.device)
    nz = shifts.ne(0)
    first = torch.where(nz.any(1), nz.to(torch.int8).argmax(1), torch.zeros(len(shifts), dtype=torch.long, device=X.device))
    positive = nz.any(1) & (shifts.gather(1, first[:, None]).squeeze(1) > 0)     # lexicographically positive shifts
    ar = torch.arange(n, device=X.device)
    out_i, out_j, out_s = [], [], []
    chunk = max(1, (1 << 22) // max(1, n * len(shifts)))
    for a in range(0, n, chunk):
        ii = ar[a:a + chunk]
        df = f[None, :, :] - f[ii, None, :]                           # (c, n, 3): f_j - f_i
        base = -torch.round(df)
        s = base[:, :, None, :] + shifts[None, None]                  # (c, n, S, 3)
        d = ((df[:, :, None, :] + s) @ C).norm(dim=-1)                # (c, n, S)
        same = ii[:, None, None] == ar[None, :, None]
        keep = (d < rc) & ((ii[:, None, None] < ar[None, :, None]) | (same & positive[None, None]))
        ci, cj, cs = keep.nonzero(as_tuple=True)
        out_i.append(ii[ci])
        out_j.append(cj)
        out_s.append(s[ci, cj, cs])
    return torch.stack([torch.cat(out_i), torch.cat(out_j)]), torch.cat(out_s)


def periodic_pairs(pos_ang, cell_ang, batch_idx, cutoff, use_torchff=True):
    """Undirected pairs within ``cutoff`` (Angstrom) over all lattice images, frame by frame: atom indices (2, P) with
    i <= j and integer shifts (P, 3) such that ``dr = x_j - x_i + shift @ cell``. An atom's pair with its own image is
    listed once (the lexicographically positive shift). torchff's CUDA cell list when the cutoff is under half the
    cell width and the extension is built; an O(n^2) torch sweep over the needed images otherwise. No gradient: the
    caller rebuilds dr from the live positions and cell."""
    with torch.no_grad():
        pos, cell = pos_ang.detach(), cell_ang.detach()
        B = int(batch_idx.max()) + 1 if batch_idx.numel() else 0
        ps, ss = [], []
        for fr in range(B):
            idx = (batch_idx == fr).nonzero().squeeze(1)
            p, s = _pairs_one(pos[idx], cell[fr], float(cutoff), use_torchff)
            ps.append(idx[p])
            ss.append(s)
        if not ps:
            return pos.new_zeros(2, 0, dtype=torch.long), pos.new_zeros(0, 3)
        return torch.cat(ps, 1), torch.cat(ss)


# ---- Ewald -------------------------------------------------------------------------------------------------------

def ewald_alpha(rc_bohr, tol):
    """Splitting parameter (1/bohr) with erfc(a rc) ~ tol."""
    return math.sqrt(-math.log(tol)) / rc_bohr


def erfc_screening(r, alpha):
    """(NORD, P): the screening g_n of the inverse powers 1/r^(2n+1) for the real-space Ewald tensor (r in bohr)."""
    a = alpha * r
    ex = torch.exp(-a * a) / (a * math.sqrt(math.pi))
    g = [torch.erfc(a)]
    dfact = 1.0
    for n in range(1, NORD):
        dfact *= 2 * n - 1
        g.append(g[-1] + (2 * a * a) ** n * ex / dfact)
    return torch.stack(g)


def _self_tensor(alpha, dtype, device):
    """(10, 10): T_self[a, b] = (-1)^|m_b| d^(m_a + m_b) [erf(a r)/r] at r = 0 (what the reciprocal sum adds of each
    site's own field)."""
    c0 = 2 * alpha / math.sqrt(math.pi)
    c2, c4 = -c0 * alpha ** 2 / 3, c0 * alpha ** 4 / 10
    T = torch.zeros(10, 10, dtype=dtype, device=device)
    for a, ma in enumerate(_MONO10):
        for b, mb in enumerate(_MONO10):
            m = sorted((ma[k] + mb[k] for k in range(3)), reverse=True)
            t = sum(m)
            if t == 0:
                v = c0
            elif t == 2:
                v = 2 * c2 if m[0] == 2 else 0.0
            elif t == 4:
                v = 24 * c4 if m[0] == 4 else (8 * c4 if m[:2] == [2, 2] else 0.0)
            else:
                v = 0.0
            T[a, b] = (-1) ** _ORDER[b] * v
    return T


def _signs(dtype, device):
    even = torch.tensor([1., 0, 0, 0, -1, -1, -1, -1, -1, -1], dtype=dtype, device=device)
    odd = torch.tensor([0., 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=dtype, device=device)
    return even, odd


def _kmono(k):
    """(K, 10): k^m for the 10 monomials."""
    e = torch.tensor(_MONO10, device=k.device)
    return k[:, None, 0] ** e[None, :, 0] * k[:, None, 1] ** e[None, :, 1] * k[:, None, 2] ** e[None, :, 2]


@dataclass
class EwaldFrame:
    """The reciprocal-space part of one frame: atoms ``idx``, Green's function ``G`` (K,) over a half space of
    k-vectors with their monomials ``kmono`` (K, 10), the phases (n, K), the self tensor and the background."""
    idx: torch.Tensor
    kmono: torch.Tensor
    G: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    T_self: torch.Tensor
    bg: torch.Tensor

    def detach(self):
        return replace(self, kmono=self.kmono.detach(), G=self.G.detach(), cos=self.cos.detach(),
                       sin=self.sin.detach(), bg=self.bg.detach())

    def apply(self, M):
        """out (n, 10) of the frame's sources M (n, 10): reciprocal sum, minus self, plus background."""
        even, odd = _signs(M.dtype, M.device)
        Cr, Ci = (M * even) @ self.kmono.T, (M * odd) @ self.kmono.T          # (n, K)
        Sr = (Cr * self.cos - Ci * self.sin).sum(0)
        Si = (Cr * self.sin + Ci * self.cos).sum(0)
        ReW = self.cos * Sr + self.sin * Si
        ImW = self.cos * Si - self.sin * Sr
        out = 2 * (((ReW * self.G) @ self.kmono) * even + ((ImW * self.G) @ self.kmono) * odd)
        out = out - M @ self.T_self
        return torch.cat([out[:, :1] + self.bg * M[:, 0].sum(), out[:, 1:]], 1)


def _kvectors(cell_bohr, kcut):
    """Integer k (K, 3) of a half space with |k| <= kcut for the (detached) cell."""
    rec = 2 * math.pi * torch.linalg.inv(cell_bohr).T                 # rows b_j, a_i . b_j = 2 pi delta
    nmax = [int(math.ceil(kcut * float(cell_bohr[a].norm()) / (2 * math.pi))) for a in range(3)]
    n = torch.tensor(list(itertools.product(range(0, nmax[0] + 1), range(-nmax[1], nmax[1] + 1),
                                            range(-nmax[2], nmax[2] + 1))), dtype=cell_bohr.dtype, device=cell_bohr.device)
    half = (n[:, 0] > 0) | ((n[:, 0] == 0) & (n[:, 1] > 0)) | ((n[:, 0] == 0) & (n[:, 1] == 0) & (n[:, 2] > 0))
    n = n[half]
    return n[(n @ rec).norm(dim=1) <= kcut]


def make_ewald_frames(X, cell_bohr, batch_idx, alpha, kcut):
    """One :class:`EwaldFrame` per frame from live positions (N, 3) and cells (B, 3, 3), both bohr."""
    frames = []
    T_self = _self_tensor(alpha, X.dtype, X.device)
    for fr in range(cell_bohr.shape[0]):
        idx = (batch_idx == fr).nonzero().squeeze(1)
        C = cell_bohr[fr]
        n = _kvectors(C.detach(), kcut)
        k = n @ (2 * math.pi * torch.linalg.inv(C).T)
        k2 = (k * k).sum(1)
        V = torch.linalg.det(C).abs()
        G = 4 * math.pi / V * torch.exp(-k2 / (4 * alpha * alpha)) / k2
        th = X[idx] @ k.T
        frames.append(EwaldFrame(idx, _kmono(k), G, torch.cos(th), torch.sin(th), T_self,
                                 -math.pi / (V * alpha * alpha)))
    return frames


def recip_apply(frames, M):
    """out (N, 10) of all frames' sources M (N, 10)."""
    out = M.new_zeros(M.shape)
    for f in frames:
        out = out.index_copy(0, f.idx, f.apply(M[f.idx]))
    return out


# ---- the induction solve ----------------------------------------------------------------------------------------

def _matvec(x, ainv, cinv, pairs, Bij, frames, P, dof):
    """A x for site variables x (N, 8): on-site blocks, real-space pair blocks, reciprocal part."""
    out = torch.cat([torch.einsum("nab,nb->na", ainv, x[:, :3]), cinv[:, None] * x[:, 3:]], 1)
    ii, jj = pairs
    out = out.index_add(0, jj, torch.einsum("pca,pa->pc", Bij, x[ii]))
    out = out.index_add(0, ii, torch.einsum("pca,pc->pa", Bij, x[jj]))
    out = out + recip_apply(frames, x @ P.T) @ P
    return out * dof


def _cg(matvec, b, precond, x0, rtol, maxiter):
    x = x0.clone() if x0 is not None else torch.zeros_like(b)
    r = b - matvec(x) if x0 is not None else b.clone()
    target = rtol * float(b.abs().max())
    if target == 0.0:
        return torch.zeros_like(b), 0
    z = precond(r)
    p = z
    rz = (r * z).sum()
    for it in range(1, maxiter + 1):
        if float(r.abs().max()) <= target:
            return x, it - 1
        ap = matvec(p)
        pap = (p * ap).sum()
        if pap <= 0:
            raise PolarizationCatastrophe(f"induction operator not positive definite (p.Ap = {float(pap):.3e} at CG "
                                          f"iteration {it})")
        step = rz / pap
        x = x + step * p
        r = r - step * ap
        z = precond(r)
        rz_new = (r * z).sum()
        p = z + (rz_new / rz) * p
        rz = rz_new
    raise RuntimeError(f"induction CG did not converge in {maxiter} iterations (residual {float(r.abs().max()):.3e}, "
                       f"target {target:.3e})")


class _ImplicitZero(torch.autograd.Function):
    """Zero forward (the residual r(x*) vanishes to tolerance); backward applies A^-1 (A symmetric): with
    ``x = x* - _ImplicitZero(r)`` the gradient of x is the implicit-function one, -A^-1 dr/dtheta."""

    @staticmethod
    def forward(ctx, r, solve):
        ctx.solve = solve
        return torch.zeros_like(r)

    @staticmethod
    def backward(ctx, g):
        return ctx.solve(g), None


def solve_induction_periodic(L, ainv, C, pairs, Bij, frames, quadrupoles=True, rtol=1e-10, maxiter=500, x0=None,
                             info=None):
    """Induced site variables x (N, 8) minimizing L.x + 1/2 x.A x under PBC; L (N, 8), ainv (N, 3, 3), C (N,), the
    real-space pair blocks ``Bij`` (P, 8, 8) of the undirected ``pairs`` (2, P) (``bt["pbc"]["A_blocks"]``, plus any
    model-side kernel) and the Ewald ``frames``. ``x0``: a starting guess (the previous MD step). Differentiable to
    first order through the implicit function theorem."""
    P = K._poly_map(L.dtype, L.device)
    dof = L.new_ones(8)
    if not quadrupoles:
        dof[3:] = 0
    cinv = 1.0 / C
    with torch.no_grad():
        d_ainv, d_cinv, d_B = ainv.detach(), cinv.detach(), Bij.detach()
        d_frames = [f.detach() for f in frames]
        alpha = torch.linalg.inv(d_ainv)
        cq = C.detach()

        def mv(v):
            return _matvec(v, d_ainv, d_cinv, pairs, d_B, d_frames, P, dof)

        def pre(v):
            return torch.cat([torch.einsum("nab,nb->na", alpha, v[:, :3]), cq[:, None] * v[:, 3:]], 1) * dof

        def solve(g):
            return _cg(mv, g * dof, pre, None, rtol, maxiter)[0]

        x_star, n_iter = _cg(mv, -L.detach() * dof, pre, None if x0 is None else x0.detach() * dof, rtol, maxiter)
    if info is not None:
        info["cg_iters"] = n_iter
    if not torch.is_grad_enabled():
        return x_star
    r = _matvec(x_star, ainv, cinv, pairs, Bij, frames, P, dof) + L * dof
    return x_star - _ImplicitZero.apply(r, solve)


# ---- short-range switches and the dispersion tail ----------------------------------------------------------------

def smooth_switch(r, r_on, r_off):
    """C^2 quintic switch: 1 below r_on, 0 above r_off."""
    if r_off <= r_on:
        return (r < r_off).to(r.dtype)
    x = ((r - r_on) / (r_off - r_on)).clamp(0, 1)
    return 1 - x * x * x * (10 - 15 * x + 6 * x * x)


def dispersion_switch(r_ang, cutoff, width):
    return smooth_switch(r_ang, cutoff - width, cutoff)


def dispersion_tail_integral(cutoff_ang, width_ang, n=4000):
    """I = int_0^inf (1 - S(r)) r^-4 dr (bohr^-3) for the switched dispersion: the tail energy of the missing pairs
    is -2 pi / V (sum_i sqrt(C6_i))^2 I (Tang-Toennies damping taken as 1 out there)."""
    r_on, r_off = (cutoff_ang - width_ang) / BOHR, cutoff_ang / BOHR
    if r_off <= r_on:
        return 1.0 / (3 * r_off ** 3)
    r = torch.linspace(r_on, r_off, n + 1, dtype=torch.float64)
    S = smooth_switch(r, r_on, r_off)
    inner = torch.trapezoid((1 - S) * r ** -4, r)
    return float(inner) + 1.0 / (3 * r_off ** 3)
