"""The state as a structured partition: ``C``, its co-membership, and the conditioning vector.

``docs/fff_film.md`` §3 in code. The primary state variable is the atom-to-fragment assignment

    C[i, f] in [0, 1],   sum_f C[i, f] = 1

one-hot in the nonreactive phase, fractional later. Everything downstream is built from
label-invariant reductions of it:

    P_ij  = sum_f C[i, f] C[j, f]        edge co-membership (never materialized densely)
    k_i   = sum_f C[i, f] r_f            local state key from fragment attributes
    u_i   = 1 - sum_f C[i, f]^2          mixing / assignment-uncertainty measure

``P`` and ``u`` are invariant under any permutation of the fragment columns, and ``k`` is
invariant because the attribute rows permute with them. That is the structural form of the
fragment-relabeling invariance the tests pin.

Relation to the v4 machinery: ``fragment_idx`` (the argmax assignment) is retained because the
channel enumeration, the per-fragment pooling and the pair bookkeeping all need a definite
grouping; at a one-hot ``C`` it *is* the assignment. ``soft_partition`` computed the same
``P_e`` from an ``(M, N)`` stack of candidate fragmentations plus mixture weights -- here the
mixing lives inside ``C`` itself, which is the form a learned router will eventually produce.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

__all__ = ["StateDescriptor"]


def _fragment_census(
    fragment_idx: torch.Tensor,  # (N,)
    species_idx: torch.Tensor,   # (N,)
    n_fragments: int,
    n_species: int,
) -> torch.Tensor:
    """``(F, n_species)`` element counts of each fragment.

    Counts rather than a composition one-hot, for the same reason
    :func:`rsfff.ff.partition.element_counts` gives: counts are what a fractional membership
    can average, and "half an extra hydrogen" is a real input to a continuous embedding.
    """
    out = torch.zeros(
        int(n_fragments), int(n_species),
        dtype=torch.get_default_dtype(), device=fragment_idx.device,
    )
    flat = fragment_idx * int(n_species) + species_idx
    return out.reshape(-1).index_add_(
        0, flat, torch.ones_like(flat, dtype=out.dtype)
    ).reshape(int(n_fragments), int(n_species))


@dataclass
class StateDescriptor:
    """One fragmentation state of one batch: the assignment matrix plus fragment attributes.

    ``C``               : (N, F) rows sum to 1; one-hot in Phase I.
    ``fragment_charge`` : (F,) formal charge Q_f.
    ``fragment_two_s``  : (F,) 2S_f.
    ``fragment_counts`` : (F, n_species) element census, continuous-capable.
    ``fragment_idx``    : (N,) the argmax assignment -- supplies the channel enumeration,
                          per-fragment pooling targets and the pair bookkeeping, all of which
                          need a definite grouping. At a one-hot ``C`` it is exact.
    ``fragment_to_batch``: (F,) frame id per fragment.
    ``n_fragments``     : F.
    """

    C: torch.Tensor
    fragment_charge: torch.Tensor
    fragment_two_s: torch.Tensor
    fragment_counts: torch.Tensor
    fragment_idx: torch.Tensor
    fragment_to_batch: torch.Tensor
    n_fragments: int

    @classmethod
    def from_batch(cls, batch, species_idx: torch.Tensor, n_species: int) -> "StateDescriptor":
        """The one-hot descriptor of a batch's definite fragmentation."""
        frag = batch.fragment_idx
        if frag is None:
            raise ValueError(
                "StateDescriptor.from_batch needs batch.fragment_idx; the extxyz must carry "
                "a fragment_idx column (a distance rule is never an acceptable substitute)"
            )
        n_frag = int(batch.n_fragments)
        dtype = torch.get_default_dtype()
        C = torch.zeros(frag.shape[0], n_frag, dtype=dtype, device=frag.device)
        C.scatter_(1, frag.reshape(-1, 1), 1.0)
        zeros = torch.zeros(n_frag, dtype=dtype, device=frag.device)
        charge = (
            zeros if batch.fragment_charge is None else batch.fragment_charge.to(dtype)
        )
        two_s = zeros if batch.fragment_two_s is None else batch.fragment_two_s.to(dtype)
        to_batch = (
            batch.fragment_to_batch
            if batch.fragment_to_batch is not None
            else torch.zeros(n_frag, dtype=torch.long, device=frag.device)
        )
        return cls(
            C=C,
            fragment_charge=charge,
            fragment_two_s=two_s,
            fragment_counts=_fragment_census(frag, species_idx, n_frag, n_species),
            fragment_idx=frag,
            fragment_to_batch=to_batch,
            n_fragments=n_frag,
        )

    @classmethod
    def blend(cls, a: "StateDescriptor", b: "StateDescriptor", lam: float) -> "StateDescriptor":
        """``C(lam) = (1 - lam) C_a + lam C_b`` -- the Stage-E continuous-state sweep.

        The two descriptors must index the *same* atoms and the same number of fragment
        columns; the fragment attributes blend linearly alongside the assignment, which is the
        input-mixing rule of :func:`rsfff.ff.partition.mixed_state` restated on ``C``. No
        intermediate EDA target exists along the path -- this is a diagnostic and a
        total-energy/force training interface, not a claim that every soft partition has a
        unique physical decomposition.

        ``fragment_idx`` is kept from ``a`` rather than recomputed by argmax, for two reasons
        that are the same reason: it is *bookkeeping* (channel enumeration, pooling targets),
        not physics, so it must stay fixed and sorted along a sweep -- an argmax would both
        break the grouped-atom layout the channel builders require and change the enumeration
        discontinuously mid-path. Everything physical reads ``C``.
        """
        if a.C.shape != b.C.shape:
            raise ValueError(
                f"blend needs matching assignment shapes, got {tuple(a.C.shape)} and "
                f"{tuple(b.C.shape)}; pad the candidate fragmentations to a shared column set"
            )
        t = float(lam)
        return cls(
            C=(1.0 - t) * a.C + t * b.C,
            fragment_charge=(1.0 - t) * a.fragment_charge + t * b.fragment_charge,
            fragment_two_s=(1.0 - t) * a.fragment_two_s + t * b.fragment_two_s,
            fragment_counts=(1.0 - t) * a.fragment_counts + t * b.fragment_counts,
            fragment_idx=a.fragment_idx,
            fragment_to_batch=a.fragment_to_batch,
            n_fragments=a.n_fragments,
        )

    def permute_fragments(self, perm: torch.Tensor) -> "StateDescriptor":
        """Relabel the fragments by ``perm`` (a permutation of ``[0, F)``).

        Physically a no-op: every consumer reads ``C`` through label-invariant reductions.
        Exists so the relabeling-invariance test states exactly what it varies.
        """
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.shape[0], device=perm.device)
        return replace(
            self,
            C=self.C[:, perm],
            fragment_charge=self.fragment_charge[perm],
            fragment_two_s=self.fragment_two_s[perm],
            fragment_counts=self.fragment_counts[perm],
            fragment_idx=inv[self.fragment_idx],
            fragment_to_batch=self.fragment_to_batch[perm],
        )

    # -- label-invariant reductions ------------------------------------------------------

    def edge_comembership(self, edge_index: torch.Tensor) -> torch.Tensor:
        """``(E,) P_e = sum_f C[i, f] C[j, f]`` on the given edges.

        Never materialize the dense ``N x N`` projector; every consumer has an index list
        (the SOAP edges, the pair list, the channel graph) and this is computed on it. At a
        one-hot ``C`` the result is an exact 0.0 or 1.0 -- the multiplications are by exact
        ones and zeros -- which is what keeps the isolated-fragment guarantee bitwise.
        """
        i, j = edge_index[0], edge_index[1]
        return (self.C[i] * self.C[j]).sum(dim=-1)

    def mixing_measure(self) -> torch.Tensor:
        """``(N,) u_i = 1 - sum_f C[i, f]^2``: zero at every vertex, positive when shared."""
        return 1.0 - (self.C * self.C).sum(dim=-1)

    def local_conditioning(self, state_embedding=None) -> torch.Tensor:
        """``(N, d_k + 1)`` the conditioning vector ``c_i = [k_i, u_i]``.

        ``k_i = sum_f C[i, f] r_f`` with ``r_f`` from the shared
        :class:`rsfff.ff.fragment_state.FragmentStateEmbedding` over the *physical* labels
        ``(Q_f, 2S_f, n_f)`` -- zero for a neutral singlet by that block's anchoring, so on
        water-only data the conditioning is exactly ``[0, ..., 0, u_i]`` and FiLM starts from
        (and, at one-hot assignments, stays at) the unconditioned network. ``state_embedding``
        may be ``None`` (or built with ``dim=0``), leaving ``c_i = [u_i]``.
        """
        u = self.mixing_measure().unsqueeze(-1)
        if state_embedding is None or state_embedding.net is None:
            return u
        r = state_embedding._embed(
            self.fragment_charge, self.fragment_two_s, self.fragment_counts
        )                                                        # (F, dim)
        return torch.cat((self.C @ r, u), dim=-1)
