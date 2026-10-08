"""External electrostatic sources: probe point charges and uniform fields / field gradients.

Two things the model needs from an external environment, both linear in it:

* **the conjugate potential** ``ext_m = dE_ext/dM`` -- an ``(N, K)`` polytensor, so that
  ``E_ext = sum_i ext_m_i . M_i + c``. It enters the permanent-multipole electrostatics
  (first order) and the coupled solve's right-hand side (``CoupledSystem.ext_m``; the
  operator never changes). For a uniform source it is ``[phi, -E, -dE]`` in the polytensor's
  own slots -- the pyCMM ``[1/3, 2/3, ...]`` quadrupole weights make ``-1/3 Theta : grad E``
  exactly ``-sum_slots p_slot dE_slot``. For a probe charge it is the same object computed
  through :func:`rsfff.ff.electrostatics.slater_elec_tensors` with the probe as a bare
  nucleus, so the atom's shell sees it one-center Slater damped (penetration), exactly the
  way it sees another atom's nucleus.
* **the undamped point fields** ``(phi, E, grad E)`` at each atom, the inputs of the
  field-dependent bonded parameters (:mod:`rsfff.ff.film.fields`).

A uniform field is described by ``field`` F and ``field_gradient`` G (traceless symmetric part
used) about ``origin``: ``phi(r) = -F.(r - o) - 1/2 (r - o).G.(r - o)``, ``E = F + G (r - o)``.
Response properties are energy derivatives with respect to these two inputs
(:meth:`rsfff.ff.film.FilmModel.response_properties`).

Units: positions and origins in Angstrom; charges in e; F in Ha/(e bohr); G in Ha/(e bohr^2);
everything returned in atomic units.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from .electrostatics import slater_elec_tensors
from .multipole import build_polytensor
from .units import BOHR_ANG

__all__ = ["ExternalSources", "external_potential", "point_fields", "with_uniform"]


@dataclass
class ExternalSources:
    """Per-batch external sources. Any part may be ``None``."""

    charge_positions: torch.Tensor | None = None   # (M, 3) Angstrom
    charges: torch.Tensor | None = None            # (M,) e
    charge_batch: torch.Tensor | None = None       # (M,) frame of each probe
    field: torch.Tensor | None = None              # (B, 3) a.u.
    field_gradient: torch.Tensor | None = None     # (B, 3, 3) a.u.
    origin: torch.Tensor | None = None             # (B, 3) Angstrom; None = coordinate origin

    @property
    def has_charges(self) -> bool:
        return self.charges is not None and self.charges.numel() > 0

    def to(self, device) -> "ExternalSources":
        f = lambda t: None if t is None else t.to(device)  # noqa: E731
        return ExternalSources(*(f(getattr(self, k)) for k in (
            "charge_positions", "charges", "charge_batch", "field", "field_gradient", "origin"
        )))


def with_uniform(ext: ExternalSources | None, field=None, field_gradient=None) -> ExternalSources:
    """``ext`` with a uniform field / gradient *added* (the property-derivative handles)."""
    ext = ExternalSources() if ext is None else ext
    f = field if ext.field is None or field is None else ext.field + field
    g = (
        field_gradient if ext.field_gradient is None or field_gradient is None
        else ext.field_gradient + field_gradient
    )
    return replace(
        ext,
        field=ext.field if field is None else f,
        field_gradient=ext.field_gradient if field_gradient is None else g,
    )


def _traceless(g: torch.Tensor) -> torch.Tensor:
    g = 0.5 * (g + g.transpose(-1, -2))
    eye = torch.eye(3, dtype=g.dtype, device=g.device)
    return g - g.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None] / 3.0 * eye


def _rel_bohr(ext, positions, batch_idx):
    r = positions / BOHR_ANG
    if ext.origin is not None:
        r = r - ext.origin[batch_idx] / BOHR_ANG
    return r


def _probe_pairs(ext, batch_idx):
    """All (atom, probe) pairs in the same frame: ``(atom (P,), probe (P,))``."""
    same = batch_idx.unsqueeze(1) == ext.charge_batch.unsqueeze(0)      # (N, M)
    return same.nonzero(as_tuple=True)


def point_fields(
    ext: ExternalSources, positions: torch.Tensor, batch_idx: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Undamped ``(phi (N,), E (N, 3), grad E (N, 3, 3))`` at every atom, atomic units."""
    n = positions.shape[0]
    phi = positions.new_zeros(n)
    e = positions.new_zeros(n, 3)
    g = positions.new_zeros(n, 3, 3)
    if ext.field is not None or ext.field_gradient is not None:
        r = _rel_bohr(ext, positions, batch_idx)
        if ext.field is not None:
            f = ext.field[batch_idx]
            phi = phi - (f * r).sum(-1)
            e = e + f
        if ext.field_gradient is not None:
            gg = _traceless(ext.field_gradient)[batch_idx]
            gr = torch.einsum("nab,nb->na", gg, r)
            phi = phi - 0.5 * (r * gr).sum(-1)
            e = e + gr
            g = g + gg
    if ext.has_charges:
        ia, ik = _probe_pairs(ext, batch_idx)
        d = (positions[ia] - ext.charge_positions[ik]) / BOHR_ANG          # (P, 3) atom - probe
        r2 = (d * d).sum(-1)
        rr = torch.sqrt(r2)
        q = ext.charges[ik]
        inv3 = q / (r2 * rr)
        phi = phi.index_add(0, ia, q / rr)
        e = e.index_add(0, ia, inv3.unsqueeze(-1) * d)
        eye = torch.eye(3, dtype=d.dtype, device=d.device)
        gp = inv3[:, None, None] * (eye - 3.0 * d[:, :, None] * d[:, None, :] / r2[:, None, None])
        g = g.index_add(0, ia, gp)
    return phi, e, g


def _unique6(g: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (g[..., 0, 0], g[..., 0, 1], g[..., 0, 2], g[..., 1, 1], g[..., 1, 2], g[..., 2, 2]),
        dim=-1,
    )


def external_potential(
    ext: ExternalSources,
    positions: torch.Tensor,     # (N, 3) Angstrom
    batch_idx: torch.Tensor,     # (N,)
    n_systems: int,
    b: torch.Tensor,             # (N,) the atoms' Slater penetration exponents, 1/bohr
    z: torch.Tensor,             # (N,) the atoms' nuclear (core) charges
    max_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(ext_m (N, K), c (B,))`` with ``E_ext(M) = sum_i ext_m_i . M_i + c`` per frame."""
    n = positions.shape[0]
    k = 1 if max_rank == 0 else (4 if max_rank == 1 else 10)
    ext_m = positions.new_zeros(n, k)
    const = positions.new_zeros(int(n_systems))

    if ext.field is not None or ext.field_gradient is not None:
        r = _rel_bohr(ext, positions, batch_idx)
        phi = positions.new_zeros(n)
        e = positions.new_zeros(n, 3)
        if ext.field is not None:
            f = ext.field[batch_idx]
            phi = phi - (f * r).sum(-1)
            e = e + f
        blocks = [phi]
        if ext.field_gradient is not None:
            gg = _traceless(ext.field_gradient)[batch_idx]
            gr = torch.einsum("nab,nb->na", gg, r)
            phi = phi - 0.5 * (r * gr).sum(-1)
            e = e + gr
            blocks = [phi]
        if max_rank >= 1:
            blocks.append(-e)
        if max_rank >= 2:
            blocks.append(
                -_unique6(_traceless(ext.field_gradient)[batch_idx])
                if ext.field_gradient is not None else positions.new_zeros(n, 6)
            )
        ext_m = ext_m + torch.cat([blk.reshape(n, -1) for blk in blocks], dim=-1)

    if ext.has_charges:
        ia, ik = _probe_pairs(ext, batch_idx)
        m = ext.charges.shape[0]
        pos_all = torch.cat((positions, ext.charge_positions.to(positions.dtype)))
        b_all = torch.cat((b, b.new_ones(m)))
        pair_index = torch.stack((ia, n + ik))
        dr = (pos_all[pair_index[1]] - pos_all[pair_index[0]]) / BOHR_ANG
        r_au = dr.norm(dim=-1)
        t_point, _t_ss, t_1c_i, _t_1c_j = slater_elec_tensors(
            dr, r_au, b_all, pair_index, max_rank=max_rank
        )
        n_j = build_polytensor(ext.charges[ik].to(positions.dtype), None, None,
                               max_rank=max_rank)
        n_i = build_polytensor(z[ia], None, None, max_rank=max_rank)
        # E = n_j^T (T_pt + T_1c_i) m_i - n_j^T T_1c_i n_i   (probe = bare nucleus j)
        g = torch.einsum("pab,pa->pb", t_point + t_1c_i, n_j)
        ext_m = ext_m.index_add(0, ia, g)
        c = -torch.einsum("pa,pab,pb->p", n_j, t_1c_i, n_i)
        const = const.index_add(0, batch_idx[ia], c)
    return ext_m, const
