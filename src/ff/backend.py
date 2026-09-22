"""Which implementation evaluates the classical force-field terms: rsfff's torch code or torchff-lib.

rsfff owns the model -- features, the parameter network, the fragment bookkeeping, the EDA
routing -- and the *leaf* evaluations (a Tang-Toennies pair energy, a Morse bond, ...) are the
part that a fused CUDA kernel can replace. This module is the seam: every leaf the film model
evaluates goes through one function here, and each function has two paths that agree to
round-off, including second derivatives (``tests/backend/``):

``torch``    the implementations in :mod:`rsfff.ff.dispersion`, :mod:`rsfff.ff.film.bonded`,
             ... exactly as before this module existed. Runs anywhere.
``torchff``  :mod:`torchff.ffterms` from the ``external/torchff-lib`` submodule: CUDA kernels
             with per-pair output and double backward, so force training works through them.
             On a CPU tensor, or without the compiled extension, ``torchff.ffterms`` falls back
             to its own pure-torch reference formulas -- which are the rsfff formulas moved
             over, so the answer is the same either way.

Selection: ``RSFFF_FF_BACKEND=torch|torchff|auto`` (default ``auto``: torchff when it imports
and the positions are on CUDA, else torch), or :func:`set_backend` at runtime. The choice is
read per call, so a test can flip it.

Units at this seam are the model's: positions in **Angstrom** and parameters in atomic units
(``c6`` in Ha bohr^6, ``b`` in 1/bohr, ``r_eq`` in bohr, ...). The torchff kernels are
unit-agnostic, so this module converts positions to bohr once before calling them. Index
tensors are rsfff's ``(2, P)`` / ``(3, Na)`` layout; torchff wants ``(P, 2)``, transposed here.
"""

from __future__ import annotations

import os

import torch

from .units import BOHR_ANG

__all__ = ["active_backend", "set_backend", "tt_dispersion", "bonded_energy", "slater_elec_field",
           "slater_elec_pair_energy", "slater_pauli_pair_energy", "HAVE_TORCHFF"]

try:
    from torchff import ffterms as _ffterms

    HAVE_TORCHFF = True
except ImportError:  # submodule not installed
    _ffterms = None
    HAVE_TORCHFF = False

_VALID = ("auto", "torch", "torchff")
_BACKEND = os.environ.get("RSFFF_FF_BACKEND", "auto").strip().lower() or "auto"
if _BACKEND not in _VALID:
    raise ValueError(f"RSFFF_FF_BACKEND must be one of {_VALID}, got {_BACKEND!r}")


def set_backend(name: str) -> None:
    """``"auto"``, ``"torch"`` or ``"torchff"``; takes effect on the next call."""
    global _BACKEND
    name = name.strip().lower()
    if name not in _VALID:
        raise ValueError(f"backend must be one of {_VALID}, got {name!r}")
    if name == "torchff" and not HAVE_TORCHFF:
        raise RuntimeError(
            "torchff is not importable; install the submodule: "
            "TORCHFF_NO_CUDA=1 pip install -e external/torchff-lib (CPU) or see "
            "scripts/build_torchff_perlmutter.sh (GPU)"
        )
    _BACKEND = name


def active_backend(positions: torch.Tensor | None = None) -> str:
    """The backend a call on ``positions`` would use: ``"torch"`` or ``"torchff"``."""
    if _BACKEND == "auto":
        if HAVE_TORCHFF and positions is not None and positions.is_cuda:
            return "torchff"
        return "torch"
    if _BACKEND == "torchff" and not HAVE_TORCHFF:
        raise RuntimeError("RSFFF_FF_BACKEND=torchff but torchff is not importable")
    return _BACKEND


# --------------------------------------------------------------------------------------
# dispersion
# --------------------------------------------------------------------------------------

def tt_dispersion(
    positions_ang: torch.Tensor,     # (N, 3) Angstrom
    pair_index: torch.Tensor,        # (2, P)
    c6_ij: torch.Tensor,             # (P,) Ha bohr^6
    b_ij: torch.Tensor,              # (P,) 1/bohr
    *,
    r_ang: torch.Tensor | None = None,  # (P,) pair distances if the caller has them (torch path)
) -> torch.Tensor:
    """``(P,)`` Tang-Toennies damped C6 energies in Hartree."""
    if active_backend(positions_ang) == "torch":
        from .dispersion import tt_damped_c6_energy

        if r_ang is None:
            i, j = pair_index[0], pair_index[1]
            r_ang = (positions_ang[i] - positions_ang[j]).norm(dim=-1)
        return tt_damped_c6_energy(r_ang, c6_ij, b_ij)
    return _ffterms.tt_dispersion_pair_energy(
        positions_ang / BOHR_ANG, pair_index.t(), c6_ij, b_ij
    )


# --------------------------------------------------------------------------------------
# bonded
# --------------------------------------------------------------------------------------

def bonded_energy(
    positions_ang: torch.Tensor,
    topo,                               # rsfff.ff.film.bonded.BondedTopology
    params,                             # rsfff.ff.film.bonded.BondedParameters
    *,
    geometry: tuple[torch.Tensor, torch.Tensor] | None = None,  # topo.geometry(positions), torch path
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(per-bond (Nb,), per-angle (Na,))`` Hartree, co-membership weighted."""
    if active_backend(positions_ang) == "torch":
        r_bond, cos_t = topo.geometry(positions_ang) if geometry is None else geometry
        return params.energy(r_bond, cos_t, topo)
    coords = positions_ang / BOHR_ANG
    e_bond = topo.bond_weight * _ffterms.morse_bond_energy(
        coords, topo.bond_index.t(), params.r_eq, params.d, params.k
    )
    e_angle = topo.angle_weight * _ffterms.cosine_angle_energy(
        coords, topo.angle_index.t(), params.cos_theta_eq, params.k_theta
    )
    return e_bond, e_angle


# --------------------------------------------------------------------------------------
# Slater-penetrated multipole electrostatics
# --------------------------------------------------------------------------------------

def slater_elec_field(
    positions_ang: torch.Tensor,   # (N, 3) Angstrom
    pair_index: torch.Tensor,      # (2, P)
    b: torch.Tensor,               # (N,) 1/bohr
    gate: torch.Tensor,            # (P,) the elst gate
    m: torch.Tensor,               # (N, K) polytensor, a.u.
    m_nuc: torch.Tensor,           # (N, K) nuclear point charges
    *,
    reference: bool = False,       # pure-torch reference formulas (differentiable to any order)
) -> torch.Tensor:
    """``d/dm`` of the gated point + penetration energy: ``(N, K)``, the coupled-solve matvec.

    torchff only -- the torch path of the coupled solve keeps its precomputed ``(P, K, K)``
    tensors (:func:`rsfff.ff.coupled_solve._coupling_grad`), which is faster than rebuilding
    them per CG iteration in torch. The kernel rebuilds them per pair on the fly instead, so
    nothing of size ``P x K x K`` is ever materialised; its backward is the exact VJP with
    respect to positions, ``b``, ``gate``, ``m`` and ``m_nuc`` (first order), which is all the
    adjoint of :class:`rsfff.ff.coupled_solve._CoupledSolve` asks of it *inside the CG loop*.
    The adjoint's one recorded residual VJP under a force loss needs a second derivative, which
    the kernel does not have yet; ``reference=True`` returns torchff's pure-torch reference
    field for that call (same formulas, ordinary autograd, ``(P, K, K)`` tensors materialised).
    """
    if not HAVE_TORCHFF:
        raise RuntimeError("slater_elec_field needs torchff")
    from torchff import slaterelec

    return slaterelec.slater_elec_field(
        positions_ang / BOHR_ANG, pair_index.t(), b, gate, m, m_nuc,
        use_customized_ops=False if reference else None,
    )


def slater_elec_pair_energy(
    positions_ang: torch.Tensor,
    pair_index: torch.Tensor,
    b: torch.Tensor,
    gate: torch.Tensor | None,     # (P,) or None for the ungated sum
    m: torch.Tensor,
    m_nuc: torch.Tensor,
    *,
    dr_au: torch.Tensor | None = None,   # torch path: reuse the model's (P, 3) r_j - r_i in bohr
    r_au: torch.Tensor | None = None,    # torch path: reuse the model's (P,) distances in bohr
    tensors: tuple | None = None,        # torch path: precomputed slater_elec_tensors
) -> torch.Tensor:
    """``(P,)`` point + penetration energies in Hartree, gated if ``gate`` is given.

    ``m`` are the full multipoles and ``m_nuc`` the nuclear point charges; the shell
    ``m - m_nuc`` is formed inside. Double backward on both paths: the torch path is ordinary
    autograd through :func:`rsfff.ff.electrostatics.slater_elec_pair_energy`, the torchff path
    is the Energy/Grad kernel pair with the dual-number HVP (M4). The film model's elst channel
    and the coupled solve's pair energy at the converged multipoles both go through here.
    """
    if active_backend(positions_ang) == "torch":
        from .electrostatics import slater_elec_pair_energy as _torch_pair

        i, j = pair_index[0], pair_index[1]
        if dr_au is None:
            dr_au = (positions_ang[j] - positions_ang[i]) / BOHR_ANG
        if r_au is None:
            r_au = dr_au.norm(dim=-1)
        max_rank = {1: 0, 4: 1, 10: 2}[int(m.shape[1])]
        e_point, e_pen = _torch_pair(
            dr_au, r_au, m, m - m_nuc, m_nuc, b, pair_index, max_rank=max_rank, tensors=tensors
        )
        e = e_point + e_pen
        return e if gate is None else gate * e
    from torchff import slaterelec

    if gate is None:
        gate = torch.ones(pair_index.shape[1], dtype=positions_ang.dtype, device=positions_ang.device)
    return slaterelec.slater_elec_pair_energy(positions_ang / BOHR_ANG, pair_index.t(), b, gate, m, m_nuc)


# --------------------------------------------------------------------------------------
# Slater multipolar Pauli repulsion
# --------------------------------------------------------------------------------------

def slater_pauli_pair_energy(
    positions_ang: torch.Tensor,   # (N, 3) Angstrom
    pair_index: torch.Tensor,      # (2, P)
    poly_i: torch.Tensor,          # (P, K) Pauli polytensor gathered onto i
    poly_j: torch.Tensor,          # (P, K) Pauli polytensor gathered onto j
    b_ij: torch.Tensor,            # (P,) combined exponent, 1/bohr
    *,
    dr_au: torch.Tensor | None = None,   # torch path: reuse the model's (P, 3) r_j - r_i in bohr
    r_au: torch.Tensor | None = None,    # torch path: reuse the model's (P,) distances in bohr
) -> torch.Tensor:
    """``(P,)`` ``poly_j^T f_2c(b_ij r) T poly_i`` in Hartree; double backward on both paths (M3)."""
    if active_backend(positions_ang) == "torch":
        from .pauli import slater_pauli_pair_energy as _torch_pauli

        if dr_au is None:
            i, j = pair_index[0], pair_index[1]
            dr_au = (positions_ang[j] - positions_ang[i]) / BOHR_ANG
        if r_au is None:
            r_au = dr_au.norm(dim=-1)
        max_rank = {1: 0, 4: 1, 10: 2}[int(poly_i.shape[1])]
        return _torch_pauli(dr_au, r_au, poly_i, poly_j, b_ij, max_rank=max_rank)
    from torchff import slaterelec

    return slaterelec.slater_pauli_pair_energy(positions_ang / BOHR_ANG, pair_index.t(), b_ij, poly_i, poly_j)
