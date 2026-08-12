"""The 1-body term: atomic references, the monomer's own response energy, and a bond energy.

    E_1body(f) = sum_{i in f} E0[Z_i]  +  E_internal(f)  +  sum_{i<j in f} W(r_ij) dE_ij

Three pieces, in order of how much of them is learned:

``E0`` is the isolated-atom energy at the reference level of theory, frozen.

``E_internal`` comes from :class:`rsfff.ff.response.FragmentResponse` -- the SQE charge
sector plus the on-site dipole and quadrupole sectors, i.e. **the cost of arranging that
monomer's multipoles**. It is the same solve, and in a composite model literally the same
module instance, that the electrostatics term takes its multipoles from. The electrostatics
term computes this quantity and deliberately excludes it from the interaction energy; this
is where it was always going.

That sharing is what makes the response parameters identifiable at all. The interaction
energy constrains only combinations of the multipoles, and the fragment multipole targets
constrain the multipoles but not the response *energy* -- ``(C, chiquad) -> (lambda C,
chiquad / lambda)`` leaves ``Theta`` untouched while scaling ``E_quad`` by ``1/lambda``.
Putting ``E_internal`` into a per-fragment energy target is what distinguishes them. Note
the identification is soft rather than exact: the bond head below is also a smooth function
of the same geometry and can absorb some of what ``E_internal`` would otherwise have to
explain.

The pair sum is the **bond energy** -- physically the spin-pairing energy that turns a bag
of atoms into a molecule. ``E0`` is position-independent, so the force comes from
``E_internal`` and the bond head together.

Two properties follow structurally from the pair list being intra-fragment:

1. **Exactly one-body.** Every ``E^(k>=2)`` of the many-body expansion is zero to
   round-off, which ``tests/test_ff_onebody.py`` checks with ``mbe_decompose``. Moving a
   monomer changes nothing about any other. That is the whole reason the term exists in
   this shape.
2. **A per-fragment target.** ``fragment_energy`` is compared elementwise to
   ``batch.fragment_energy`` -- the ``Fragment Energies (Ha)`` a Q-Chem EDA job prints --
   so every monomer in a pentamer is its own training example.

**It is not a pair potential.** The head reads ``inv_feats``, the SOAP power spectrum, so
each pair's energy depends on the whole fragment; the H-O-H bend is representable. The pair
*decomposition* is there to make the term rigorously per-fragment, not to make it two-body.

Scale, because the defaults elsewhere are wrong here
----------------------------------------------------
``PairEnergyHead`` is built for intermolecular corrections: ``energy_scale=1e-3`` Ha and an
envelope over 4-5 Angstrom. A water molecule's two O-H bonds are about **-0.37 Ha** below
the free atoms, and the intramolecular distances are 0.96 (O-H) and 1.5 (H-H) Angstrom.
Left at the defaults the head starts 370 scales away from its target with its envelope
switched off over the entire bonding region. :class:`OneBodyModel` therefore takes its own
``bond_energy_scale`` (~0.2 Ha) and window (~2.5 to 4.0 Angstrom).

``E0`` is not identifiable from water-only data
-----------------------------------------------
Every fragment is H2O, so ``n_H = 2 n_O`` exactly and the data constrains only
``E0_O + 2 E0_H`` -- one number, not two. ``E0`` therefore has to come from real atomic
calculations at the matching level of theory and stay a frozen buffer; making it learnable
would fit that one combination and leave the split arbitrary, which is invisible here and
wrong the moment an ion-water system arrives.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..features.features import LambdaFeatures
from ..mlip.pair_heads import PairEnergyHead
from .pairs import intra_fragment_pairs
from .response import FragmentResponse, FragmentResponseOutput  # noqa: F401


@dataclass
class OneBodyEnvironment:
    """Post-solve electrostatic environment at each atom, atomic units. **Not yet used.**

    The slot exists from the start so wiring it in later is an addition rather than a
    rewrite. What it will carry:

    potential      : (N,)    Hartree / e
    field          : (N, 3)  Hartree / (e a0)
    field_gradient : (N, 5)  spherical, Hartree / (e a0^2)

    The constraint that decides how it may be consumed: :class:`OneBodyEnergy` must stay
    rotation-invariant, so ``field`` cannot be handed to the pair head raw. The available
    pair invariants are ``V_i``, ``V_j``, ``|F_i|``, ``|F_j|``, ``F_i . F_j``,
    ``F_i . rhat``, ``F_j . rhat`` and ``rhat^T (grad F)_i rhat``. Of these the ones
    contracted with ``rhat`` flip sign when ``i`` and ``j`` are swapped, so they must be
    symmetrized the same way the head already symmetrizes atomic features: use
    ``(F_i - F_j) . rhat`` (even, since both the difference and ``rhat`` flip) and
    ``|(F_i + F_j) . rhat|``. Feeding ``F_i . rhat`` directly would make the head's output
    depend on the arbitrary ordering of an undirected pair.

    Two ordering constraints for whoever wires this up: a composite model must call the
    electrostatics solve *before* the 1-body term, and the environment must be built from
    **inter-fragment** multipoles only -- an atom's own monomer's field is already in
    ``feats``, and including it would make the term stop being 1-body.
    """

    potential: torch.Tensor
    field: torch.Tensor
    field_gradient: torch.Tensor | None = None


def _fragment_to_batch(batch, n_frag: int) -> torch.Tensor:
    """Frame id of each fragment, derived if the batch does not carry it.

    ``MoleculeDataset.flat_batch`` supplies ``fragment_to_batch`` directly, but batches
    assembled by hand -- ``rsfff.ff.many_body.subset_batch``, which is what the many-body
    decomposition runs on -- do not, and deriving it is one scatter. Atoms are grouped by
    fragment and fragments nest inside frames, so the frame of a fragment is the frame of
    any of its atoms.
    """
    if batch.fragment_to_batch is not None:
        return batch.fragment_to_batch
    out = batch.batch_idx.new_zeros(n_frag)
    return out.scatter_(0, batch.fragment_idx, batch.batch_idx)


@dataclass
class OneBodyOutput:
    """Per-fragment and per-frame 1-body energies, plus the pair breakdown."""

    energy: torch.Tensor            # (B,)  per frame, the sum over its fragments
    fragment_energy: torch.Tensor   # (F,)  the training target
    energy_ref: torch.Tensor        # (F,)  sum_i E0[Z_i], the frozen part
    energy_internal: torch.Tensor   # (F,)  the response solve's internal energy
    energy_bond: torch.Tensor       # (F,)  sum_{i<j} W(r) dE_ij, the learned part
    pair_index: torch.Tensor        # (2, P)
    r: torch.Tensor                 # (P,) Angstrom
    e_pair: torch.Tensor            # (P,)
    #: The shared solve's full output, or None when no response module is attached. Carries
    #: the multipoles the fragment-multipole loss supervises.
    response: "FragmentResponseOutput | None" = None


class OneBodyEnergy(nn.Module):
    """Frozen atomic references plus a bond-energy pair head over intra-fragment pairs.

    ``forward(batch, feats)`` mirrors
    :meth:`rsfff.ff.electrostatics.SlaterElectrostatics.forward` deliberately: a composite
    model can then featurize once and hand the same ``feats`` to both terms.
    """

    def __init__(
        self,
        bond_head: PairEnergyHead,
        reference_energies: torch.Tensor,   # (n_species,) Hartree, ordered like neighbor_types
        response: FragmentResponse | None = None,
    ) -> None:
        super().__init__()
        self.bond_head = bond_head
        self.response = response
        self.register_buffer("reference_energies", reference_energies.clone())

    def forward(
        self,
        batch,
        feats: LambdaFeatures,
        *,
        response=None,
        env: OneBodyEnvironment | None = None,
    ) -> OneBodyOutput:
        if batch.fragment_idx is None:
            raise ValueError(
                "the 1-body term is defined per fragment but batch.fragment_idx is None; "
                "the extxyz needs a `fragment_idx` column"
            )
        if env is not None:
            raise NotImplementedError(
                "the electrostatic-environment hook is designed but not wired up; see "
                "OneBodyEnvironment for the invariants it has to be reduced to first"
            )
        positions = batch.positions
        frag = batch.fragment_idx
        n_frag = int(batch.n_fragments)

        e_atom = self.reference_energies[feats.species_idx]
        energy_ref = e_atom.new_zeros(n_frag).index_add_(0, frag, e_atom)

        # The response solve. `response` lets a composite model solve once and hand the same
        # result to the electrostatics term, so both read one set of multipoles.
        res = response
        if res is None and self.response is not None:
            res = self.response(batch, feats)
        energy_internal = (
            energy_ref.new_zeros(n_frag) if res is None else res.internal_energy
        )

        pair_index, r, pair_frag = intra_fragment_pairs(positions, frag)
        e_pair = self.bond_head(
            feats.inv_feats, feats.species_idx, positions, pair_index, r
        )
        energy_bond = e_pair.new_zeros(n_frag).index_add_(0, pair_frag, e_pair)

        fragment_energy = energy_ref + energy_internal + energy_bond
        energy = fragment_energy.new_zeros(int(batch.n_systems)).index_add_(
            0, _fragment_to_batch(batch, n_frag), fragment_energy
        )
        return OneBodyOutput(
            energy=energy,
            fragment_energy=fragment_energy,
            energy_ref=energy_ref,
            energy_internal=energy_internal,
            energy_bond=energy_bond,
            pair_index=pair_index,
            r=r,
            e_pair=e_pair,
            response=res,
        )


class OneBodyModel(nn.Module):
    """Featurizer grouped by fragment + :class:`OneBodyEnergy`.

    ``intra_fragment`` is not optional here the way it is for the other terms: a 1-body
    energy whose descriptors see neighboring monomers is not a 1-body energy, and the
    resulting term would be silently non-additive while still fitting well.
    """

    def __init__(self, featurizer, onebody: OneBodyEnergy) -> None:
        super().__init__()
        self.featurizer = featurizer
        self.onebody = onebody

    @property
    def reference_energies(self) -> torch.Tensor:
        return self.onebody.reference_energies

    def forward(self, batch) -> OneBodyOutput:
        return self.onebody(batch, self.featurizer(batch, batch.fragment_idx))


__all__ = ["OneBodyEnergy", "OneBodyEnvironment", "OneBodyModel", "OneBodyOutput"]
