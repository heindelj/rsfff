"""Fused Triton kernels for the two scalar inner problems of the electronic-state solve.

Both inner problems -- the per-pair logit equation ``kappa p + T logit(p) = a`` of
:mod:`bond_order` and the per-atom formal-count stationarity ``h(n) = 0`` of
:mod:`electronic_state` -- are embarrassingly parallel with per-element control flow. In
plain torch that control flow is a few hundred tiny elementwise launches (and a host sync)
per dual-state evaluation, which on a GPU costs more than the arithmetic by two orders of
magnitude. Here each element is one Triton program running the same bracketed iteration as
scalar code, so a state evaluation costs two launches for the inner solves.

The kernels produce the converged root only; the two differentiable Newton polishing steps
that give the exact first and second derivatives stay in torch (a handful of launches).
The torch reference loops in :mod:`electronic_state` / :mod:`bond_order` remain the
fallback on CPU tensors, when Triton is missing, or with ``RSFFF_PAIRING_FUSED=0``; the two
paths agree to roundoff (``tests/pairing/test_pairing_fused.py``), and the torch loop is
the specification.

The formal-count kernel mirrors :func:`electronic_state._formal_count` step for step: warm
start, bracket, Newton with the ``q = 0`` step of ``E0'`` inverted in closed form when the
step would cross it or dominates the local slope, the walls at ``q = +-1`` as hard stops,
and the bracket's own step (regula falsi when the end residuals are within a decade,
bisection otherwise) when Newton fails to shrink the residual.
"""

from __future__ import annotations

import os

import torch

try:  # pragma: no cover - import guard
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    HAVE_TRITON = False

__all__ = ["HAVE_TRITON", "fused_enabled", "pair_logit", "formal_count"]


def fused_enabled(t: torch.Tensor) -> bool:
    """Whether the fused kernels serve tensors like ``t``: CUDA (or the Triton interpreter
    for testing on the host), Triton importable, not switched off by the environment."""
    if not HAVE_TRITON or os.environ.get("RSFFF_PAIRING_FUSED", "1") == "0":
        return False
    if t.is_cuda:
        return True
    return os.environ.get("TRITON_INTERPRET", "0") == "1"


if HAVE_TRITON:

    @triton.jit
    def _pair_logit_kernel(a_ptr, kappa_ptr, x_ptr, temperature, n_elem, MAXITER: tl.constexpr):
        pid = tl.program_id(0)
        if pid < n_elem:
            a = tl.load(a_ptr + pid)
            kappa = tl.load(kappa_ptr + pid)
            # the piecewise-linear start of the torch reference
            if a > kappa:
                x = (a - kappa) / temperature
            elif a < 0.0:
                x = a / temperature
            else:
                x = (a - 0.5 * kappa) / (temperature + 0.25 * kappa)
            s = 1.0 / (1.0 + tl.exp(-x))
            phi = kappa * s + temperature * x - a
            tol = 1.0e-13 * (1.0 + tl.abs(a))
            it = 0
            while (tl.abs(phi) > tol) & (it < MAXITER):
                dphi = kappa * s * (1.0 - s) + temperature
                x_new = x - phi / dphi
                s_new = 1.0 / (1.0 + tl.exp(-x_new))
                phi_new = kappa * s_new + temperature * x_new - a
                k = 0
                while (tl.abs(phi_new) > tl.abs(phi)) & (k < 6):
                    x_new = 0.5 * (x + x_new)
                    s_new = 1.0 / (1.0 + tl.exp(-x_new))
                    phi_new = kappa * s_new + temperature * x_new - a
                    k += 1
                x = x_new
                s = s_new
                phi = phi_new
                it += 1
            tl.store(x_ptr + pid, x)

    @triton.jit
    def _e0_prime(q, chi, eta, eps, wall):
        r = tl.sqrt(q * q + eps * eps)
        ds = q / r
        z = (r - eps - 1.0) / eps
        sig = 1.0 / (1.0 + tl.exp(-z))
        # softplus(z) without overflow
        sp = tl.where(z > 30.0, z, tl.log(1.0 + tl.exp(tl.minimum(z, 30.0))))
        w = eps * sp
        dw = sig * ds
        k = wall * eta
        d2s = eps * eps / (r * r * r)
        de = chi + 0.5 * eta * ds + k * w * dw
        d2e = 0.5 * eta * d2s + k * (dw * dw + w * (sig * (1.0 - sig) / eps * ds * ds + sig * d2s))
        return de, d2e

    @triton.jit
    def _count_residual(n, lam, mu, chi, eta, n0, a, b, shell, T, eps, wall, tiny):
        d = n - n0
        de0, d2e0 = _e0_prime(-d, chi, eta, eps, wall)
        n_c = tl.maximum(n, tiny)
        m_c = tl.maximum(shell - n, tiny)
        db = T * tl.log(n_c / m_c)
        d2b = T * (n_c + m_c) / (n_c * m_c)
        h = -de0 + db - lam * (a + 2.0 * b * d) + mu
        dh = d2e0 + d2b - 2.0 * lam * b
        return h, dh, de0

    @triton.jit
    def _bracket_step(lo, hi, h_lo, h_hi):
        """regula falsi while the end residuals are within a decade, bisection otherwise"""
        both = (h_lo > -1.0e29) & (h_hi < 1.0e29)
        balanced = both & (tl.abs(h_hi) < 10.0 * tl.abs(h_lo)) & (tl.abs(h_lo) < 10.0 * tl.abs(h_hi))
        n_rf = (lo * h_hi - hi * h_lo) / tl.where(both, h_hi - h_lo, 1.0)
        n_fb = tl.where(balanced, n_rf, 0.5 * (lo + hi))
        return tl.minimum(tl.maximum(n_fb, lo), hi)

    @triton.jit
    def _formal_count_kernel(
        lam_ptr, mu_ptr, chi_ptr, eta_ptr, n0_ptr, a_ptr, b_ptr, shell_ptr, ninit_ptr, out_ptr,
        T, eps, wall, tol, mach_eps, tiny, n_elem,
        MAXITER: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid < n_elem:
            lam = tl.load(lam_ptr + pid)
            mu = tl.load(mu_ptr + pid)
            chi = tl.load(chi_ptr + pid)
            eta = tl.load(eta_ptr + pid)
            n0 = tl.load(n0_ptr + pid)
            a = tl.load(a_ptr + pid)
            b = tl.load(b_ptr + pid)
            shell = tl.load(shell_ptr + pid)
            n = tl.load(ninit_ptr + pid)
            n = tl.minimum(tl.maximum(n, 1.0e-9), shell - 1.0e-9)

            lo = 0.0 * n
            hi = shell
            h_lo = -1.0e30 + 0.0 * n
            h_hi = 1.0e30 + 0.0 * n
            h, dh, de0 = _count_residual(n, lam, mu, chi, eta, n0, a, b, shell, T, eps, wall, tiny)
            it = 0
            go = tl.abs(h) > tol
            while go & (it < MAXITER):
                # fold the current point into the bracket
                if h < 0.0:
                    lo = n
                    h_lo = h
                else:
                    hi = n
                    h_hi = h
                n_newton = n - h / dh
                # closed-form inversion of the q = 0 step of E0' where it is the stiff part
                # or where Newton would cross it
                y = (h + de0 - chi) / (0.5 * eta)
                y_c = tl.minimum(tl.maximum(y, -0.999), 0.999)
                n_inv = n0 - eps * y_c / tl.sqrt(1.0 - y_c * y_c)
                q_r = tl.sqrt((n0 - n) * (n0 - n) + eps * eps)
                s_step = 0.5 * eta * eps * eps / (q_r * q_r * q_r)
                crosses = ((n < n0) & (n_newton > n0)) | ((n > n0) & (n_newton < n0))
                use_inv = (tl.abs(y) < 0.999) & (crosses | (s_step > 50.0 * tl.abs(dh - s_step)))
                n_newton = tl.where(use_inv, n_inv, n_newton)
                # the walls at q = +-1 are hard stops
                k_lo = n0 - 1.0
                k_hi = n0 + 1.0
                cross_lo = ((n < k_lo) & (n_newton > k_lo)) | ((n > k_lo) & (n_newton < k_lo))
                n_newton = tl.where(cross_lo, k_lo, n_newton)
                cross_hi = ((n < k_hi) & (n_newton > k_hi)) | ((n > k_hi) & (n_newton < k_hi))
                n_newton = tl.where(cross_hi, k_hi, n_newton)
                inside = (n_newton > lo) & (n_newton < hi)
                n_try = tl.where(inside, n_newton, _bracket_step(lo, hi, h_lo, h_hi))
                h_try, dh_try, de0_try = _count_residual(
                    n_try, lam, mu, chi, eta, n0, a, b, shell, T, eps, wall, tiny
                )
                if inside & (tl.abs(h_try) > tl.abs(h)):
                    # the failed Newton point still tightens the bracket
                    if h_try < 0.0:
                        lo = n_try
                        h_lo = h_try
                    else:
                        hi = n_try
                        h_hi = h_try
                    n_try = _bracket_step(lo, hi, h_lo, h_hi)
                    h_try, dh_try, de0_try = _count_residual(
                        n_try, lam, mu, chi, eta, n0, a, b, shell, T, eps, wall, tiny
                    )
                n = n_try
                h = h_try
                dh = dh_try
                de0 = de0_try
                go = (tl.abs(h) > tol) & (hi - lo > 8.0 * mach_eps * shell)
                it += 1
            tl.store(out_ptr + pid, n)


def pair_logit(a: torch.Tensor, kappa: torch.Tensor, temperature: float, *, maxiter: int = 40) -> torch.Tensor:
    """The converged logit per pair (no autograd); see :func:`bond_order._solve_pair_logit`."""
    a = a.detach().contiguous()
    kappa = kappa.detach().contiguous()
    x = torch.empty_like(a)
    n = int(a.numel())
    if n == 0:
        return x
    _pair_logit_kernel[(n,)](a, kappa, x, float(temperature), n, MAXITER=int(maxiter))
    return x


def formal_count(lam, mu_atom, chi, eta, n0, a, b, shell, temperature, n_init, *, eps: float,
                 wall: float, tol: float, maxiter: int = 60) -> torch.Tensor:
    """The converged formal count per atom (no autograd); see
    :func:`electronic_state._formal_count`."""
    args = [t.detach().contiguous() for t in (lam, mu_atom, chi, eta, n0, a, b, shell, n_init)]
    out = torch.empty_like(args[0])
    n = int(out.numel())
    if n == 0:
        return out
    finfo = torch.finfo(out.dtype)
    _formal_count_kernel[(n,)](
        *args, out, float(temperature), float(eps), float(wall), float(tol),
        float(finfo.eps), float(finfo.tiny), n, MAXITER=int(maxiter),
    )
    return out
