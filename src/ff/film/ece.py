"""Per-bond edge-cluster-expansion (ECE) features for the bonded and term parameter heads.

The bonded heads (:class:`~rsfff.ff.film.bonded.BondedParameterHead`,
:class:`~rsfff.ff.film.terms.TermParameterHead`) read per-atom trunk latents, so a bond's
parameters see its environment only through two atom-centred SOAP vectors pooled
symmetrically (``z_i + z_j``, ``(z_i - z_j)^2``). That pooling cannot represent the bond-frame
quantities a bonded parameter actually depends on -- the dihedral harmonics ``cos m omega``
between the two ends, the angles on each side, which neighbour sits where -- except
indirectly. This module computes those directly, on the covalent graph, with
:class:`bb.features.EdgeClusterExpansion` (``docs/ece.md`` in bb): for every bond ``i - j``
the bonded neighbours of ``i`` (near) and of ``j`` (far) are expanded in the bond frame and
combined up to correlation order 2 (four-body: the bond, one neighbour on each side), which is
exactly the information a bonded term with 1-4 reach can use.

Per bond the two directed edges are merged symmetrically, so the feature is invariant under
``i <-> j`` with nothing lost to a sum alone::

    even:  g(i->j) + g(j->i),   (g(i->j) - g(j->i))^2
    odd :  o(i->j)^2 + o(j->i)^2,   o(i->j) o(j->i)       (pseudoscalars enter squared, so
                                                           every parameter stays parity-even)
    own :  Morse basis of r_ij                             (``own_bond``; already symmetric)

compressed elementwise with ``asinh`` (the products are up to fourth order in the densities
and span orders of magnitude between a terminal H-X bond and a C-C bond with six neighbours;
``asinh`` is smooth, monotone, linear near zero and logarithmic beyond, so no feature
dominates the first layer and nothing saturates), then embedded by a small MLP into ``dim``
channels. The heads consume that embedding:

==================  =====================================================================
term                bond-feature input (symmetric under the term's own reversal)
==================  =====================================================================
bond                ``b[ij]``
angle i-a-k         ``b[ia] + b[ak]``, ``(b[ia] - b[ak])^2``
improper c(l; j,k)  ``b[cl]``, ``b[cj] + b[ck]``, ``(b[cj] - b[ck])^2``
torsion a-b-c-d     ``[b[ab], b[bc], b[cd]]`` in the orbit sum (reversed: ``[cd, bc, ab]``)
bond/angle coupl.   ``[b[ia], b[ak]]`` in the orbit sum (reversed: legs swapped)
angle-angle         ``[b[as], b[ax], b[ay]]`` in the orbit sum (``x <-> y``)
==================  =====================================================================

**Isolated by construction.** The ECE runs on the covalent graph only: intramolecular, no
distance cutoff, nothing from another fragment. It therefore belongs to the isolated
parameters ``theta_0`` and cannot double-count the environment: the env branch of every head
keeps reading the joined latent behind the environment gate (exactly zero in isolation), and
its energy keeps landing in the induction channel. The same embedding is passed to both
branches; in the env branch it only tells the environment correction *which* bond it is
correcting.

**Own coordinates.** With ``own_bond=True`` the bond's parameters see ``r_ij`` and (through the
near/far ``m = 0`` channels) the adjacent angles; an angle's parameters see the angle itself
through its legs' densities. The parameters are then a function of the term's own coordinate,
which is what lets the bounded functional forms become exact for the 1-4 monomers, and also
what lets a "parameter" turn into a disguised energy. The geometry-independence regularizer
(``delta_iso``, ``bonding.bond_var``) and a stretch-scan hold-out are what keep it honest.

**Smoothness.** The invariants do not depend on the frame gauge (the twist about each bond),
and the radial functions are Morse ``exp(-a (r/r0 - 1))^n`` without a cutoff, so the features
are smooth in the geometry and stay bounded as a bond stretches.

**Reactive blends.** Each neighbour's contribution to a density is weighted by the
co-membership of the bond that reaches it (``BondedTopology.bond_weight``), so a fractional
bond fades out of its neighbours' features continuously, as its own terms do.

**Dissociation (``neighbour_decay``, default on).** The Morse radial basis contains the
constant ``y^0 = 1``, so without a decay a neighbour that has left would still contribute its
*direction* to the densities of the bonds next to it, and the remaining fragment's features
would never become those of the isolated fragment. Each neighbour's contribution is therefore
also weighted by ``y = exp(-a (r / r0 - 1))`` of the bond that reaches it (the basis' own Morse
variable, ``~1`` at a normal bond length): on a fixed diabatic topology, stretching a bond to
infinity turns every feature of the rest of the molecule into that fragment's own.

Besides the bond embedding (:meth:`embed`), :meth:`edge_features` exposes the raw
:class:`bb.features.EdgeFeatures` (with the SO(2)-covariant ``mblocks`` for equivariant bond
readouts) and :meth:`directed_invariants` the per-directed-edge invariants, for the ECE
electrical heads (:mod:`rsfff.ff.film.electrical`).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from bb.features import EdgeClusterExpansion
from bb.graph import bond_graph
from bb.nn import mlp

__all__ = ["ECEBondFeatures"]


class ECEBondFeatures(nn.Module):
    """``(Nb, dim)`` per-bond embedding of the ECE invariants of ``BondedTopology.bond_index``.

    ``species`` are the model's sorted neighbour types (``species_idx`` indexes them).
    Positions are in Angstrom (the model's geometry); the Morse radial reference lengths
    default to the sum of covalent radii.
    """

    def __init__(
        self,
        species: list[int],
        *,
        n_max: int = 4,
        l_max: int = 3,
        channels: int = 8,
        species_dim: int = 4,
        morse_a: float = 2.0,
        own_bond: bool = True,
        dim: int = 16,
        hidden: int = 32,
        depth: int = 1,
        backend: str = "e3nn",
        neighbour_decay: bool = True,
    ) -> None:
        super().__init__()
        self.species = [int(z) for z in species]
        self.own_bond = bool(own_bond)
        self.neighbour_decay = bool(neighbour_decay)
        self.ece = EdgeClusterExpansion(
            self.species, n_max=int(n_max), l_max=int(l_max), channels=int(channels),
            species_dim=int(species_dim), morse_a=float(morse_a), backend=backend,
        )
        n_in = 2 * self.ece.dim_0e + 2 * self.ece.dim_0o + (int(n_max) if self.own_bond else 0)
        self.n_invariants = n_in
        self.embed = mlp(n_in, int(hidden), int(depth), int(dim))
        self.dim = int(dim)

    def bond_decay(self, positions: torch.Tensor, species_idx: torch.Tensor, bonds) -> torch.Tensor:
        """``(Nb,)`` the radial basis' Morse variable ``y = exp(-a (r / r0 - 1))`` of each bond."""
        i, j = bonds[0], bonds[1]
        r = (positions[j] - positions[i]).norm(dim=-1)
        rad = self.ece.radial
        log_r0 = 0.5 * (rad.log_r0 + rad.log_r0.T)
        r0 = log_r0.exp()[species_idx[i], species_idx[j]].to(r.dtype)
        return torch.exp(-rad.a * (r / r0 - 1.0))

    def edge_features(self, positions: torch.Tensor, species_idx: torch.Tensor, topo):
        """The raw :class:`bb.features.EdgeFeatures` of ``topo.bond_index`` (``None`` without
        bonds). Directed edge ``e < Nb`` is bond ``e`` in its stored orientation, ``e + Nb`` the
        reverse."""
        bonds = topo.bond_index
        if int(bonds.shape[1]) == 0:
            return None
        g = bond_graph(bonds.t(), int(positions.shape[0]), check=False)
        weight = getattr(topo, "bond_weight", None)
        if self.neighbour_decay:
            decay = self.bond_decay(positions, species_idx, bonds)
            weight = decay if weight is None else weight.to(decay.dtype) * decay
        return self.ece(positions, species_idx, g, bond_weight=weight)

    @staticmethod
    def directed_invariants(f) -> torch.Tensor:
        """``(E, directed_dim)`` per directed edge: ``asinh`` of the even invariants, the
        squared pseudoscalars and the bond radial basis -- even under every improper operation."""
        return torch.asinh(torch.cat((f.inv0e, f.inv0o ** 2, f.bond_radial), dim=-1))

    def directed_dim(self) -> int:
        return self.ece.dim_0e + self.ece.dim_0o + self.ece.n_max

    def symmetric_invariants(self, f) -> torch.Tensor:
        """``(Nb, n_invariants)`` direction-symmetrized, ``asinh``-compressed ECE invariants."""
        nb = f.graph.n_bonds
        e_fwd, e_rev = f.inv0e[:nb], f.inv0e[nb:]
        o_fwd, o_rev = f.inv0o[:nb], f.inv0o[nb:]
        parts = [e_fwd + e_rev, (e_fwd - e_rev) ** 2, o_fwd ** 2 + o_rev ** 2, o_fwd * o_rev]
        if self.own_bond:
            parts.append(f.bond_radial[:nb])
        return torch.asinh(torch.cat(parts, dim=-1))

    def invariants(self, positions: torch.Tensor, species_idx: torch.Tensor, topo) -> torch.Tensor:
        f = self.edge_features(positions, species_idx, topo)
        if f is None:
            return positions.new_zeros(0, self.n_invariants)
        return self.symmetric_invariants(f)

    def embed_features(self, f, ref: torch.Tensor) -> torch.Tensor:
        """``(Nb, dim)`` bond embedding from precomputed :meth:`edge_features` (``f`` may be
        ``None``: no bonds; ``ref`` supplies dtype / device)."""
        if f is None:
            return ref.new_zeros(0, self.dim)
        return self.embed(self.symmetric_invariants(f))

    def forward(self, positions: torch.Tensor, species_idx: torch.Tensor, topo) -> torch.Tensor:
        return self.embed_features(self.edge_features(positions, species_idx, topo), positions)
