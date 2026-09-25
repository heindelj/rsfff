"""Charge-dependent atomic reference energies and the formal-charge share behind them.

The film model references every fragment to its neutral atoms, ``sum_i E0[Z_i]``. For an ion
that puts the ionization energy into the bonds: against neutral atoms H3O+ is bound by
-0.14 Ha and OH- by -0.18, where water is bound by -0.37, so the third O-H of hydronium would
have to be "worth" +0.23 Ha -- a per-fragment constant hidden in a pair term. Referencing
each atom to *its own* charge state removes it::

    E0(Z, q) = E0(Z) + chi_Z q + 1/2 eta_Z q^2

with ``chi = (IP + EA)/2`` and ``eta = IP - EA`` from the atomic reference states, so the
quadratic runs exactly through ``E(Z, +1) = E0 + IP`` and ``E(Z, -1) = E0 - EA`` and is smooth
in between (the same form the SQE on-site energy has, and for the same reason). Against
``O+ + 3H`` hydronium is bound by -0.65 Ha, against ``O- + H`` hydroxide by -0.18: one O-H bond
is then worth ~-0.2 Ha whatever the fragment's charge, which is what a transferable pairing
term needs.

The charge each atom is referenced at is its **formal-charge share** -- the fragment's formal
charge divided among its heavy atoms (or, for a fragment without heavy atoms, among its
hydrogens with the opposite sign convention a proton/hydride needs). It is the same share the
valence capacity reads (:mod:`rsfff.ff.pairing.heads`); fractional shares (a Zundel cation as
one fragment) interpolate smoothly through the quadratic. When the fragment labels go, this
share is what the local electron count has to replace.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["ChargedAtomicReference", "formal_charge_share"]


def formal_charge_share(
    heavy: torch.Tensor,              # (N,) bool
    fragment_idx: torch.Tensor,       # (N,)
    fragment_charge: torch.Tensor,    # (F,)
    dtype=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(q_share (N,), has_heavy (N,))``: each atom's part of its fragment's formal charge.

    Heavy atoms split the charge equally; a fragment with no heavy atom splits it over its
    hydrogens. ``has_heavy`` tells the valence rule which sign convention applies.
    """
    dtype = dtype or torch.get_default_dtype()
    n_frag = int(fragment_charge.shape[0])
    heavy_f = heavy.to(dtype)
    ones = torch.ones_like(heavy_f)
    n_heavy = heavy_f.new_zeros(n_frag).index_add(0, fragment_idx, heavy_f)
    n_atoms = heavy_f.new_zeros(n_frag).index_add(0, fragment_idx, ones)
    q = fragment_charge.to(dtype)[fragment_idx]
    has_heavy = n_heavy[fragment_idx] > 0
    w = torch.where(
        has_heavy, heavy_f / n_heavy[fragment_idx].clamp(min=1.0), ones / n_atoms[fragment_idx]
    )
    return q * w, has_heavy


class ChargedAtomicReference(nn.Module):
    """``E0(Z, q)`` per atom from frozen per-species ``(E0, chi, eta)`` tables.

    With ``chi`` and ``eta`` absent (``None``) this is the film model's neutral reference.
    """

    def __init__(
        self,
        energies: torch.Tensor,             # (n_species,) neutral-atom energies, Hartree
        chi: torch.Tensor | None = None,    # (n_species,) (IP + EA) / 2
        eta: torch.Tensor | None = None,    # (n_species,) IP - EA
    ) -> None:
        super().__init__()
        self.register_buffer("energies", energies.clone())
        zeros = torch.zeros_like(energies)
        self.register_buffer("chi", zeros.clone() if chi is None else chi.to(energies.dtype).clone())
        self.register_buffer("eta", zeros.clone() if eta is None else eta.to(energies.dtype).clone())
        self.charged = chi is not None

    @classmethod
    def from_states(cls, energies: torch.Tensor, atomic_states=None) -> "ChargedAtomicReference":
        """From the ``reference_energies`` table and an optional ``AtomicStateReference``.

        Elements whose IP/EA could not be derived (NaN in the states file) are referenced
        neutrally: a wrong quadratic is worse than none.
        """
        if atomic_states is None:
            return cls(energies)
        chi = atomic_states.chi_mulliken.to(energies.dtype)
        eta = atomic_states.hardness.to(energies.dtype)
        bad = torch.isnan(chi) | torch.isnan(eta)
        chi = torch.where(bad, torch.zeros_like(chi), chi)
        eta = torch.where(bad, torch.zeros_like(eta), eta)
        return cls(energies, chi, eta)

    def forward(self, species_idx: torch.Tensor, q: torch.Tensor | None = None) -> torch.Tensor:
        """``(N,)`` reference energy of each atom at its formal-charge share ``q``."""
        e = self.energies[species_idx]
        if q is None:
            return e
        return e + self.chi[species_idx] * q + 0.5 * self.eta[species_idx] * q * q
