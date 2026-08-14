"""Slater-damped multipole interactions: the shared kernel for the short-range terms.

Two Slater 1s densities overlapping produce a damped multipole interaction tensor. Writing
``T_damped = T * (1 - f_n)`` defines the factors this module returns:

    f_n(u) = p_n(u) * exp(-u),      u = b_ij * r      (dimensionless)

so ``f_n`` is the **overlap complement** ``-(T_damped - T_undamped)/T_undamped`` -- the part
that decays exponentially, with the undamped ``1/r^n`` tail already subtracted off. Every
short-range term is then one sign away from the next:

======================  =========================================================
term                    tensor
======================  =========================================================
Pauli repulsion         ``+f_n * T``  with *Pauli* multipoles
charge penetration      ``-f_n * T``  with the real electrostatic multipoles
Slater-damped Coulomb   ``(1 - f_n) * T``
polarization            ``(1 - f_n) * T`` on the field / field-gradient blocks
======================  =========================================================

That is the whole reason this module returns ``f_n`` rather than the damped tensor: the
Pauli term here and the electrostatics/polarization terms that come later share one
implementation and differ only in sign and in which multipoles they contract.

Transcribed from ``pyCMM/cmm/short_range.py:6-39`` (the damping polynomials) and
``pyCMM/cmm/multipole.py:246-344`` (the interaction tensor). pyCMM's third damper,
``computeShortRangePolarizationDampFactors`` (``short_range.py:41-53``), is deliberately
*not* ported: polarization will use the standard Slater factors above.

Conventions
-----------
``dr_vec = pos[j] - pos[i]`` and a pair energy is ``m_j^T T m_i`` -- both matching pyCMM,
because getting either backwards silently flips the charge-dipole cross terms while
leaving charge-charge and dipole-dipole correct. The identities that pin it down:

    rank 0            E = f1 * q_i q_j / r
    charge-dipole     E = q_j (mu_i . dr) / r^3 - q_i (mu_j . dr) / r^3   (damped by f3)
    dipole-dipole     E = [(mu_i . mu_j) - 3 (mu_i . rhat)(mu_j . rhat)] / r^3

Units are **atomic units throughout**: ``r`` in bohr, ``b`` in 1/bohr, charges in e,
dipoles in e*bohr, energies in Hartree. Callers convert once at their boundary.

Unlike :func:`rsfff.ff.damping.tang_toennies`, no small-``u`` series branch is needed:
``p_n(0) = 1`` for every n, so ``f_n(0) = 1`` exactly and the polynomials are evaluated
without cancellation. The ``1/r`` prefactor still diverges as ``r -> 0``, which is why the
pair list is built with ``loop=False``.
"""

from __future__ import annotations

import math

import torch

#: Multipole polytensor width per rank: 1 (charge), 4 (charge+dipole), 10 (+quadrupole).
_POLYTENSOR_WIDTH = {0: 1, 1: 4, 2: 10}


#: Quadrupole polytensor weights on ``(xx, xy, xz, yy, yz, zz)``. The 1/3 is the
#: quadrupole convention and the 2/3 on the off-diagonals accounts for storing only the
#: upper triangle of a symmetric tensor whose contraction runs over all nine index pairs.
#: From pyCMM's ``convertMultipolesToPolytensor`` (``cmm/multipole.py:194``), which is what
#: builds the multipoles the Pauli term there contracts (``cmm/force_field.py:903-918``).
_QUAD_POLY_WEIGHTS = (1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0)


def _check_rank(max_rank: int) -> int:
    max_rank = int(max_rank)
    if max_rank in (0, 1, 2):
        return max_rank
    raise ValueError(f"max_rank must be 0, 1 or 2, got {max_rank}")


def slater_two_center_damp(u: torch.Tensor, max_rank: int = 1) -> torch.Tensor:
    """Two-center Slater damping factors, stacked ``(2*max_rank + 1, ...)``.

    Returns ``f1`` for ``max_rank=0`` and ``(f1, f3, f5)`` for ``max_rank=1``, where the
    index into the leading axis is the multipole order that factor damps: ``f1`` scales
    ``1/r``, ``f3`` scales ``1/r^3``, ``f5`` scales ``1/r^5``.

    ``u = b_ij * r`` with ``b_ij`` the combined exponent (``sqrt(b_i b_j)`` in pyCMM's
    combination rule). This is the *equal-exponent* two-center form; the polynomials
    already assume ``b_i == b_j == b_ij``, which is what makes a single combined exponent
    the right argument.
    """
    max_rank = _check_rank(max_rank)
    u2 = u * u
    u3 = u2 * u
    exp_u = torch.exp(-u)
    p1 = 1.0 + 11.0 * u / 16.0 + 3.0 * u2 / 16.0 + u3 / 48.0
    if max_rank == 0:
        return torch.stack([p1 * exp_u], dim=0)
    u4 = u3 * u
    u5 = u4 * u
    p3 = 1.0 + u + u2 / 2.0 + 7.0 * u3 / 48.0 + u4 / 48.0
    # The u^0..u^4 head shared by p5, p7 and p9; they differ only from u^5 on.
    head = 1.0 + u + u2 / 2.0 + u3 / 6.0 + u4 / 24.0
    p5 = head + u5 / 144.0
    if max_rank == 1:
        return torch.stack([p1 * exp_u, p3 * exp_u, p5 * exp_u], dim=0)
    u6 = u5 * u
    p7 = head + u5 / 120.0 + u6 / 720.0
    p9 = p7 + u6 * u / 5040.0
    return torch.stack(
        [p1 * exp_u, p3 * exp_u, p5 * exp_u, p7 * exp_u, p9 * exp_u], dim=0
    )


def slater_one_center_damp(u: torch.Tensor, max_rank: int = 1) -> torch.Tensor:
    """One-center Slater damping factors, stacked like :func:`slater_two_center_damp`.

    The point-charge/Slater-density case: one site is a point (a nucleus, or a probe), the
    other a Slater 1s density, so ``u = b_i * r`` uses that one site's exponent and no
    combination rule applies. Used by the electrostatics nucleus-shell terms rather than by
    Pauli, but it belongs with its two-center sibling.
    """
    max_rank = _check_rank(max_rank)
    u2 = u * u
    exp_u = torch.exp(-u)
    p1 = 1.0 + u / 2.0
    if max_rank == 0:
        return torch.stack([p1 * exp_u], dim=0)
    p3 = 1.0 + u + u2 / 2.0
    p5 = p3 + u2 * u / 6.0
    if max_rank == 1:
        return torch.stack([p1 * exp_u, p3 * exp_u, p5 * exp_u], dim=0)
    u4 = u2 * u2
    # Note p9 builds on p5, not on p7 (pyCMM/cmm/short_range.py:17-18) -- transcribed
    # rather than "corrected", since it is the fitted model's actual functional form.
    p7 = p5 + u4 / 30.0
    p9 = p5 + u4 * 4.0 / 105.0 + u4 * u / 210.0
    return torch.stack(
        [p1 * exp_u, p3 * exp_u, p5 * exp_u, p7 * exp_u, p9 * exp_u], dim=0
    )


def damped_interaction_tensor(
    dr_vec: torch.Tensor,                  # (P, 3) bohr, = pos[j] - pos[i]
    damp: torch.Tensor | None,             # (2*max_rank+1, P) or None for the bare tensor
    r_inv: torch.Tensor | None = None,     # (P,) 1/|dr_vec|, reused if already computed
    max_rank: int = 1,
) -> torch.Tensor:
    """Multipole interaction tensor ``(P, K, K)``, ``K = 1``, ``4`` or ``10`` by rank.

    Each inverse power is scaled by its damping factor before the Cartesian derivatives are
    assembled -- ``1/r`` by ``damp[0]``, ``1/r^3`` by ``damp[1]``, and so on up to
    ``1/r^9`` by ``damp[4]`` -- which is what makes the result the *damped* tensor rather
    than the bare one scaled uniformly. Passing ``damp=None`` gives the undamped tensor
    (useful for tests and for the long-range electrostatics).

    The rank-2 block is the third and fourth derivatives of ``1/r``; the undamped tensor is
    checked against autograd derivatives of ``1/r`` in ``tests/test_ff_multipole.py``,
    which is what makes a 100-entry transcription safe.
    """
    max_rank = _check_rank(max_rank)
    if r_inv is None:
        r_inv = 1.0 / dr_vec.norm(dim=-1)
    if damp is not None:
        expected = 2 * max_rank + 1
        if damp.shape[0] != expected:
            raise ValueError(
                f"max_rank={max_rank} needs {expected} damping factors, got {damp.shape[0]}"
            )

    r_inv1 = r_inv if damp is None else r_inv * damp[0]
    if max_rank == 0:
        return r_inv1.reshape(-1, 1, 1)

    r_inv2 = r_inv * r_inv
    r_inv3 = r_inv2 * r_inv
    r_inv5 = r_inv3 * r_inv2
    if damp is not None:
        r_inv3 = r_inv3 * damp[1]
        r_inv5 = r_inv5 * damp[2]

    x, y, z = dr_vec[:, 0], dr_vec[:, 1], dr_vec[:, 2]
    x2, y2, z2 = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    # First derivatives of 1/r: t_a = d/da (1/r) = -a/r^3.
    tx, ty, tz = -x * r_inv3, -y * r_inv3, -z * r_inv3
    # Second derivatives: t_ab = 3 a b / r^5 - delta_ab / r^3.
    txx = 3.0 * x2 * r_inv5 - r_inv3
    txy = 3.0 * xy * r_inv5
    txz = 3.0 * xz * r_inv5
    tyy = 3.0 * y2 * r_inv5 - r_inv3
    tyz = 3.0 * yz * r_inv5
    tzz = 3.0 * z2 * r_inv5 - r_inv3

    if max_rank == 1:
        # Row/column order [q, mu_x, mu_y, mu_z]; energy is m_j^T T m_i, so the
        # charge-dipole blocks carry opposite signs (pyCMM/cmm/multipole.py:321-327).
        return torch.stack(
            (
                r_inv1, -tx, -ty, -tz,
                tx, -txx, -txy, -txz,
                ty, -txy, -tyy, -tyz,
                tz, -txz, -tyz, -tzz,
            ),
            dim=-1,
        ).reshape(-1, 4, 4)

    r_inv7 = r_inv5 * r_inv2
    r_inv9 = r_inv7 * r_inv2
    if damp is not None:
        r_inv7 = r_inv7 * damp[3]
        r_inv9 = r_inv9 * damp[4]

    # Third derivatives: t_abc = -15 abc/r^7 + 3(a d_bc + b d_ac + c d_ab)/r^5.
    txxx = -15.0 * x2 * x * r_inv7 + 9.0 * x * r_inv5
    txxy = -15.0 * x2 * y * r_inv7 + 3.0 * y * r_inv5
    txxz = -15.0 * x2 * z * r_inv7 + 3.0 * z * r_inv5
    tyyy = -15.0 * y2 * y * r_inv7 + 9.0 * y * r_inv5
    tyyx = -15.0 * y2 * x * r_inv7 + 3.0 * x * r_inv5
    tyyz = -15.0 * y2 * z * r_inv7 + 3.0 * z * r_inv5
    tzzz = -15.0 * z2 * z * r_inv7 + 9.0 * z * r_inv5
    tzzx = -15.0 * z2 * x * r_inv7 + 3.0 * x * r_inv5
    tzzy = -15.0 * z2 * y * r_inv7 + 3.0 * y * r_inv5
    txyz = -15.0 * x * yz * r_inv7

    # Fourth derivatives.
    txxxx = 105.0 * x2 * x2 * r_inv9 - 90.0 * x2 * r_inv7 + 9.0 * r_inv5
    txxxy = 105.0 * x2 * xy * r_inv9 - 45.0 * xy * r_inv7
    txxxz = 105.0 * x2 * xz * r_inv9 - 45.0 * xz * r_inv7
    txxyy = 105.0 * x2 * y2 * r_inv9 - 15.0 * (x2 + y2) * r_inv7 + 3.0 * r_inv5
    txxzz = 105.0 * x2 * z2 * r_inv9 - 15.0 * (x2 + z2) * r_inv7 + 3.0 * r_inv5
    txxyz = 105.0 * x2 * yz * r_inv9 - 15.0 * yz * r_inv7
    tyyyy = 105.0 * y2 * y2 * r_inv9 - 90.0 * y2 * r_inv7 + 9.0 * r_inv5
    tyyyx = 105.0 * y2 * xy * r_inv9 - 45.0 * xy * r_inv7
    tyyyz = 105.0 * y2 * yz * r_inv9 - 45.0 * yz * r_inv7
    tyyzz = 105.0 * y2 * z2 * r_inv9 - 15.0 * (y2 + z2) * r_inv7 + 3.0 * r_inv5
    tyyxz = 105.0 * y2 * xz * r_inv9 - 15.0 * xz * r_inv7
    tzzzz = 105.0 * z2 * z2 * r_inv9 - 90.0 * z2 * r_inv7 + 9.0 * r_inv5
    tzzzx = 105.0 * z2 * xz * r_inv9 - 45.0 * xz * r_inv7
    tzzzy = 105.0 * z2 * yz * r_inv9 - 45.0 * yz * r_inv7
    tzzxy = 105.0 * z2 * xy * r_inv9 - 15.0 * xy * r_inv7

    # Row/column order [q, mu_x, mu_y, mu_z, Q_xx, Q_xy, Q_xz, Q_yy, Q_yz, Q_zz],
    # transcribed from pyCMM/cmm/multipole.py:329-340.
    return torch.stack(
        (
            r_inv1, -tx, -ty, -tz, txx, txy, txz, tyy, tyz, tzz,
            tx, -txx, -txy, -txz, txxx, txxy, txxz, tyyx, txyz, tzzx,
            ty, -txy, -tyy, -tyz, txxy, tyyx, txyz, tyyy, tyyz, tzzy,
            tz, -txz, -tyz, -tzz, txxz, txyz, tzzx, tyyz, tzzy, tzzz,
            txx, -txxx, -txxy, -txxz, txxxx, txxxy, txxxz, txxyy, txxyz, txxzz,
            txy, -txxy, -tyyx, -txyz, txxxy, txxyy, txxyz, tyyyx, tyyxz, tzzxy,
            txz, -txxz, -txyz, -tzzx, txxxz, txxyz, txxzz, tyyxz, tzzxy, tzzzx,
            tyy, -tyyx, -tyyy, -tyyz, txxyy, tyyyx, tyyxz, tyyyy, tyyyz, tyyzz,
            tyz, -txyz, -tyyz, -tzzy, txxyz, tyyxz, tzzxy, tyyyz, tyyzz, tzzzy,
            tzz, -tzzx, -tzzy, -tzzz, txxzz, tzzxy, tzzzx, tyyzz, tzzzy, tzzzz,
        ),
        dim=-1,
    ).reshape(-1, 10, 10)


def multipole_pair_energy(
    m_i: torch.Tensor,        # (P, K) polytensor on atom i
    m_j: torch.Tensor,        # (P, K) polytensor on atom j
    tensor: torch.Tensor,     # (P, K, K)
) -> torch.Tensor:
    """``m_j^T T m_i`` per pair: ``(P,)``. Matches pyCMM's contraction order."""
    return torch.einsum("pa,pab,pb->p", m_j, tensor, m_i)


def build_polytensor(
    charges: torch.Tensor,                    # (N,)
    dipoles: torch.Tensor | None = None,      # (N, 3) or None
    quadrupoles: torch.Tensor | None = None,  # (N, 3, 3) Cartesian, or None
    *,
    max_rank: int = 1,
) -> torch.Tensor:
    """Stack per-atom multipoles into the ``(N, K)`` polytensor the tensor contracts.

    A ``None`` block yields zeros in its slots, so a lower-rank model is exactly the
    higher-rank model with the missing blocks zeroed -- the two agree to round-off rather
    than approximately, which is what makes ``max_rank`` a clean ablation.

    Quadrupoles arrive as full Cartesian ``(N, 3, 3)`` (traceless, symmetric) and are
    reduced to the six unique components with pyCMM's ``[1/3, 2/3, 2/3, 1/3, 2/3, 1/3]``
    weights; see :data:`_QUAD_POLY_WEIGHTS`.
    """
    max_rank = _check_rank(max_rank)
    if max_rank == 0:
        return charges.unsqueeze(-1)
    if dipoles is None:
        dipoles = charges.new_zeros(charges.shape[0], 3)
    blocks = [charges.unsqueeze(-1), dipoles]
    if max_rank >= 2:
        if quadrupoles is None:
            blocks.append(charges.new_zeros(charges.shape[0], 6))
        else:
            w = torch.tensor(
                _QUAD_POLY_WEIGHTS, dtype=quadrupoles.dtype, device=quadrupoles.device
            )
            unique = torch.stack(
                (
                    quadrupoles[..., 0, 0], quadrupoles[..., 0, 1], quadrupoles[..., 0, 2],
                    quadrupoles[..., 1, 1], quadrupoles[..., 1, 2], quadrupoles[..., 2, 2],
                ),
                dim=-1,
            )
            blocks.append(unique * w)
    return torch.cat(blocks, dim=-1)


# ---------------------------------------------------------------------------
# Quadrupoles: spherical <-> Cartesian
# ---------------------------------------------------------------------------

def spherical_to_cartesian_quadrupole(q_s: torch.Tensor) -> torch.Tensor:
    """``(..., 5)`` spherical ``(q20, q21c, q21s, q22c, q22s)`` -> ``(..., 3, 3)`` Cartesian.

    pyCMM's ``computeCartesianQuadrupoles`` (``cmm/multipole.py:197-219``). The result is
    symmetric and traceless by construction, which is the whole point of predicting the
    five spherical components rather than six Cartesian ones: a six-component prediction
    would carry a spurious isotropic degree of freedom that the interaction tensor's
    traceless contraction cannot see, i.e. an unconstrained gauge on the parameters.
    """
    if q_s.shape[-1] != 5:
        raise ValueError(f"expected 5 spherical components, got {tuple(q_s.shape)}")
    half_sqrt3 = math.sqrt(3.0) / 2.0
    q20, q21c, q21s, q22c, q22s = (q_s[..., i] for i in range(5))
    qxx = q22c * half_sqrt3 - q20 / 2.0
    qxy = q22s * half_sqrt3
    qxz = q21c * half_sqrt3
    qyy = -q22c * half_sqrt3 - q20 / 2.0
    qyz = q21s * half_sqrt3
    qzz = q20
    return torch.stack(
        (
            torch.stack((qxx, qxy, qxz), dim=-1),
            torch.stack((qxy, qyy, qyz), dim=-1),
            torch.stack((qxz, qyz, qzz), dim=-1),
        ),
        dim=-2,
    )


def cartesian_to_spherical_quadrupole(q_c: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`spherical_to_cartesian_quadrupole`, for tests and inspection."""
    half_sqrt3 = math.sqrt(3.0) / 2.0
    return torch.stack(
        (
            q_c[..., 2, 2],
            q_c[..., 0, 2] / half_sqrt3,
            q_c[..., 1, 2] / half_sqrt3,
            (q_c[..., 0, 0] - q_c[..., 1, 1]) / half_sqrt3 / 2.0,
            q_c[..., 0, 1] / half_sqrt3,
        ),
        dim=-1,
    )


def l2_rotation_generators(dtype=None, device=None) -> torch.Tensor:
    """The three ``l=2`` rotation generators in the spherical basis: ``(3, 5, 5)``.

    ``L[a]`` is the derivative of the ``l=2`` representation with respect to a rotation about
    Cartesian axis ``a``, acting on ``(q20, q21c, q21s, q22c, q22s)``. Each is real and
    antisymmetric, and ``L(n) = sum_a n_a L[a]`` for a unit ``n`` has eigenvalues
    ``0, +-i, +-2i`` -- so ``-L(n)^2`` is symmetric PSD with eigenvalues ``m^2 = 0, 1, 1, 4, 4``.
    That spectrum is what :class:`rsfff.mlip.response_heads.AxialQuadrupolePolarizabilityHead`
    builds its three spectral projectors from.

    **Derived, not transcribed.** A Cartesian symmetric tensor rotates as ``Theta -> R Theta
    R^T``, so an infinitesimal rotation gives ``dTheta = [A_a, Theta]`` with
    ``(A_a)_bc = -eps_abc``. Pushing that through the *existing*
    :func:`spherical_to_cartesian_quadrupole` and :func:`cartesian_to_spherical_quadrupole`
    means the generators cannot drift from the spherical convention the rest of the multipole
    code uses -- the same reason :func:`irrep2_to_spherical` is derived from the backend's own
    constant rather than hand-written. ``tests/test_ff_multipole.py`` checks
    ``expm(theta L(n))`` against a finite rotation round-tripped through Cartesian.
    """
    dtype = torch.float64 if dtype is None else dtype
    device = torch.device("cpu") if device is None else device
    # Levi-Civita, then A_a = -eps_a
    eps = torch.zeros(3, 3, 3, dtype=dtype, device=device)
    for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        eps[a, b, c] = 1.0
        eps[a, c, b] = -1.0
    a_gen = -eps                                                     # (3, 3, 3)

    basis = spherical_to_cartesian_quadrupole(
        torch.eye(5, dtype=dtype, device=device)
    )                                                                 # (5, 3, 3)
    out = torch.zeros(3, 5, 5, dtype=dtype, device=device)
    for a in range(3):
        commutator = a_gen[a] @ basis - basis @ a_gen[a]              # (5, 3, 3)
        # row m holds the spherical image of basis vector m, so transpose to get a matrix
        # acting on component vectors
        out[a] = cartesian_to_spherical_quadrupole(commutator).transpose(0, 1)
    return out


def irrep2_to_spherical(irrep6_to_voigt: torch.Tensor) -> torch.Tensor:
    """``(5, 5)`` map from the backend's lambda=2 slots to ``(q20, q21c, q21s, q22c, q22s)``.

    **Do not replace this with a hand-written permutation.** The equivariant backend's
    lambda=2 components are not a relabeling of the spherical components used here: e3nn's
    real spherical harmonics carry a permuted axis convention, so (measured) slot 2 is
    ``-0.4082 q20 - 0.7071 q22c`` and slot 4 is ``0.7071 q20 - 0.4082 q22c`` -- the m=0 and
    m=2c components *mix*. Labeling the head's raw outputs ``(q20, q21c, ...)`` therefore
    silently destroys equivariance while leaving every shape and magnitude plausible.

    Deriving the map from the backend's own tested ``irrep6_to_voigt`` constant instead
    means the two conventions can never drift apart. Applied as ``q_s = a2 @ C``.
    """
    # Cartesian (Voigt) image of each lambda=2 basis vector, and of each spherical one.
    v_irrep = irrep6_to_voigt[:, 1:].to(torch.float64)                   # (6_voigt, 5)
    eye = torch.eye(5, dtype=torch.float64)
    cart = spherical_to_cartesian_quadrupole(eye)                       # (5, 3, 3)
    v_sph = torch.stack(
        (
            cart[:, 0, 0], cart[:, 1, 1], cart[:, 2, 2],
            cart[:, 1, 2], cart[:, 0, 2], cart[:, 0, 1],
        ),
        dim=0,
    )                                                                    # (6_voigt, 5)
    # v_irrep[:, m] = sum_n C[m, n] v_sph[:, n]
    c = torch.linalg.lstsq(v_sph, v_irrep).solution.t().contiguous()     # (5, 5)
    residual = (v_sph @ c.t() - v_irrep).abs().max()
    if residual > 1e-10:
        raise RuntimeError(
            f"lambda=2 slots are not a linear combination of the spherical quadrupole "
            f"basis (residual {float(residual):.2e}); the backend's irrep6_to_voigt "
            f"convention has changed"
        )
    return c.to(torch.get_default_dtype())


__all__ = [
    "slater_two_center_damp",
    "slater_one_center_damp",
    "damped_interaction_tensor",
    "multipole_pair_energy",
    "build_polytensor",
    "spherical_to_cartesian_quadrupole",
    "cartesian_to_spherical_quadrupole",
    "irrep2_to_spherical",
]
