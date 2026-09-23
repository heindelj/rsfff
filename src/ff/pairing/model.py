"""The pairing model: the film assembly with a variational bond order in place of the topology.

    E_f     = sum_i E0[Z_i] + E_pair(theta_0)
            + sum_{intra-assigned ij} [ (1 - c_ij) sum_a gate_a E_a + c_ij gate_pauli E_pauli ]

    E_inter = sum_a sum_{inter-assigned ij} (1 - c_ij) gate_a E_a      -> eda_cls_elec / mod_pauli / disp

    E_ind   = [ coupled solve around the permanent multipoles, at theta ]
            - [ the same functional at zero response ]
            + [ E_pair(theta) - E_pair(theta_0) ]

    E_cross = c_ij gate_pauli E_pauli and the pairing energy on inter-assigned pairs
              -- exactly zero while nothing reacts

    E_total = sum_f E_f + sum_a E_inter^a + E_ind + E_cross

Every classical channel acts between two atoms exactly as much as the bond order says they
are *not* one fragment, ``(1 - c_ij)``; the Pauli repulsion is additionally on within a bond,
at full strength, as its repulsive wall. Summed over all pairs the weights are
``(1 - c) sum_a E_a + c E_pauli``, and the four buckets above are that sum split by the
assigned fragmentation, which only decides which label a pair is compared against.

Relative to :class:`rsfff.ff.film.FilmModel`, three things change and nothing else:

1. **The bonded potential.** Morse + cosine angle on the assigned covalent graph become the
   minimized pairing functional (:mod:`rsfff.ff.pairing.bond_order`) over every pair within
   ``pairing_cutoff``: ``-J p + kappa p^2 / 2`` plus the entropic barriers, with ``J`` the Pauli
   operator on the valence polytensor and ``p`` the solved bond order. Nothing in it is
   switched by distance; ``J`` is an exponentially decaying overlap and the cutoff taper sits
   where it is already ~1e-6 of a bond.
2. **The co-membership.** ``p_intra`` -- the film's ``P_ij = sum_f C_if C_jf`` from the
   assigned fragmentation -- is replaced by ``c_ij`` derived from the bond orders (1-2 and 1-3
   paths). It plays every role ``P_ij`` played: the intra/inter split of the classical
   channels, the induction gate ``gate_elst (1 - c)``, and the SQE conductances. On intact
   water ``c`` is one on the bonds and the 1-3 pair and zero elsewhere to ~1e-20, so the film
   accounting (and its EDA supervision) is recovered exactly; as a bond stretches, every
   intramolecular classical term turns on with ``1 - c`` and nothing else.
3. **The range separation.** With ``range_gate="bond_order"`` the per-channel Fermi switches
   are gone: every classical channel is evaluated with its cutoff taper only, and the bond
   order is the only thing that says "these two atoms are one fragment". The Pauli repulsion
   in particular is at full strength on a bond -- it is the bond's repulsive wall.
   ``range_gate="fermi"`` keeps the film's switches as an ablation.

The assigned ``fragment_idx`` survives as **bookkeeping** (which label a pair's energy is
compared against, per-fragment pooling), never as physics, and the ``E_cross`` bucket is what
keeps the total exact when the bond orders disagree with the assignment.

The vertex condition of the film model holds up to the barrier temperature: an isolated
fragment's ``c`` is ``1 - O(exp(-(J - kappa)/T))`` on its bonds, i.e. one to double precision
for any bond, so its induction is zero and its permanent multipoles are what its labels say.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ...mlip.switch import pairwise_switch
from ...mlip.sqe import sqe_solve
from ..damping import fermi_switch
from .. import backend as ff_backend
from ..backend import slater_elec_pair_energy, slater_pauli_pair_energy
from ..expert_model import ClassicalSpec
from ..film.model import DEFAULT_FILM_CLASSICAL, FilmOutput
from ..film.state import StateDescriptor
from ..fragment_state import FragmentStateEmbedding
from ..multipole import build_polytensor, spherical_to_cartesian_quadrupole
from ..pairs import intra_fragment_channels, union_channels, union_pairs
from ..polarization import LevelOutput, coupled_response
from ..response import ResponseParameters, fragment_polarizability
from ..units import BOHR_ANG
from .bond_order import (
    BondOrderSolution,
    comembership_from_bond_order,
    pairing_energy,
    solve_bond_order,
)
from .heads import PairingFamily
from .network import PairingParameterNetwork, PairingParameters

__all__ = ["PairingModel", "PairingOutput"]


@dataclass
class PairingOutput(FilmOutput):
    """:class:`FilmOutput` plus the bond-order state. ``p_intra`` *is* the co-membership ``c``."""

    sub_index: torch.Tensor | None = None      # (Pb,) candidate pairs in the pair list
    bond_order: torch.Tensor | None = None     # (Pb,) p at theta_0
    unpaired: torch.Tensor | None = None       # (N,) u at theta_0
    coupling: torch.Tensor | None = None       # (Pb,) J at theta_0, Hartree
    kappa_pair: torch.Tensor | None = None     # (Pb,) at theta_0
    energy_pairing: torch.Tensor | None = None  # (F,) per fragment, theta_0
    bo_solver: dict[str, tuple] | None = None  # name -> (n_iter, converged, residual)


def _lookup(keys_sorted: torch.Tensor, i: torch.Tensor, j: torch.Tensor, n_atoms: int):
    """Positions of the pairs ``(min, max)`` in a sorted key list; ``found`` marks the hits."""
    lo, hi = torch.minimum(i, j), torch.maximum(i, j)
    want = lo * n_atoms + hi
    n = keys_sorted.shape[0]
    pos = torch.searchsorted(keys_sorted, want).clamp(max=max(n - 1, 0))
    found = (keys_sorted[pos] == want) if n else torch.zeros_like(want, dtype=torch.bool)
    return pos, found


class PairingModel(nn.Module):
    """Projector + state conditioning + pairing parameter network + one evaluation.

    Args (beyond the film model's)
    ------------------------------
    pairing_cutoff   : Angstrom, the candidate radius for bond orders and SQE channels.
    pairing_taper    : Angstrom, the taper width on ``J`` below ``pairing_cutoff``.
    temperature      : Hartree, the entropic barrier scale ``T`` of the bond-order functional.
    range_gate       : ``"bond_order"`` (cutoff tapers only) or ``"fermi"`` (film switches).
    include_13       : whether the co-membership includes the two-step paths.
    """

    def __init__(
        self,
        projector,
        state_embedding: FragmentStateEmbedding,
        network: PairingParameterNetwork,
        range_heads: nn.Module,
        reference_energies: torch.Tensor,
        *,
        max_rank: int = 2,
        classical: dict[str, ClassicalSpec] | None = None,
        induction: bool = True,
        max_num_neighbors: int = 512,
        cg_rtol: float = 1.0e-9,
        cg_atol: float = 1.0e-12,
        cg_maxiter: int = 100,
        cg_check_every: int = 1,
        pairing_cutoff: float = 4.0,
        pairing_taper: float = 1.0,
        temperature: float = 0.002,
        range_gate: str = "bond_order",
        include_13: bool = True,
        bo_tol: float = 1.0e-10,
        bo_maxiter: int = 100,
    ) -> None:
        super().__init__()
        if range_gate not in ("bond_order", "fermi"):
            raise ValueError(f"range_gate must be 'bond_order' or 'fermi', got {range_gate!r}")
        if not pairing_cutoff > pairing_taper > 0.0:
            raise ValueError("need pairing_cutoff > pairing_taper > 0")
        self.projector = projector
        self.state_embedding = state_embedding
        self.network = network
        self.range_heads = range_heads
        self.classical = dict(classical or DEFAULT_FILM_CLASSICAL)
        self.max_rank = int(max_rank)
        self.induction = bool(induction)
        self.max_num_neighbors = int(max_num_neighbors)
        self.cutoff_max = max(max(c.cutoff for c in self.classical.values()), pairing_cutoff)
        self.cg = dict(rtol=float(cg_rtol), atol=float(cg_atol), maxiter=int(cg_maxiter),
                       check_every=int(cg_check_every))
        self.pairing_cutoff = float(pairing_cutoff)
        self.pairing_taper = float(pairing_taper)
        self.range_gate = str(range_gate)
        self.include_13 = bool(include_13)
        self.bo = dict(tol=float(bo_tol), maxiter=int(bo_maxiter))
        self.register_buffer("temperature", torch.tensor(float(temperature)))
        self.register_buffer("reference_energies", reference_energies.clone())

    # -- helpers -------------------------------------------------------------------------

    def _gates(self, r, species_idx, pair_index):
        """Per channel: cutoff taper, times the element-table Fermi switch under ``fermi``."""
        i, j = pair_index[0], pair_index[1]
        zero_width = r.new_zeros(species_idx.shape[0], 0)
        r0, alpha = self.range_heads(zero_width, species_idx)
        gate, r0_pair, log_prior, log_prior_pair = {}, {}, {}, {}
        for c, name in enumerate(self.range_heads.channel_names):
            spec = self.classical[name]
            taper = pairwise_switch(r, spec.cutoff - spec.taper_width, spec.cutoff)
            r0_ij = (0.5 * (r0[name][i].log() + r0[name][j].log())).exp()
            if self.range_gate == "fermi":
                gate[name] = fermi_switch(r, r0_ij, alpha[name]) * taper
            else:
                gate[name] = taper
            r0_pair[name] = r0_ij
            prior = self.range_heads.log_r0_prior[c][species_idx]
            log_prior[name] = prior
            log_prior_pair[name] = 0.5 * (prior[i] + prior[j])
        return gate, r0, r0_pair, alpha, log_prior, log_prior_pair

    @staticmethod
    def _route(theta0, theta, c):
        """Per-pair parameter mix by the co-membership: bonded pairs read ``theta_0``."""
        if theta0 is theta:
            return theta0
        w = c.reshape(-1, *([1] * (theta0.dim() - 1)))
        return w * theta0 + (1.0 - w) * theta

    def _coupling(self, fam: PairingFamily, positions, sub_pairs, dr_au, r_au, r_ang):
        """``(J (Pb,), kappa_ij (Pb,))`` on the candidate pairs from one parameter set."""
        i, j = sub_pairs[0], sub_pairs[1]
        quad_c = None if fam.quad_s is None else spherical_to_cartesian_quadrupole(fam.quad_s)
        poly = build_polytensor(fam.q, fam.mu, quad_c, max_rank=self.max_rank)
        b_ij = (0.5 * (fam.b[i].log() + fam.b[j].log())).exp()
        J = slater_pauli_pair_energy(
            positions, sub_pairs, poly[i], poly[j], b_ij, dr_au=dr_au, r_au=r_au
        )
        J = J * pairwise_switch(
            r_ang, self.pairing_cutoff - self.pairing_taper, self.pairing_cutoff
        )
        kappa = (0.5 * (fam.kappa[i].log() + fam.kappa[j].log())).exp()
        return J, kappa

    def _pairing(self, fam, positions, sub_pairs, dr_au, r_au, r_ang, batch_idx, n_sys):
        """One bond-order solve and its energy: ``(sol, J, kappa, e_pair (Pb,), e_atom (N,))``."""
        J, kappa = self._coupling(fam, positions, sub_pairs, dr_au, r_au, r_ang)
        sol = solve_bond_order(
            J, kappa, fam.valence, self.temperature, sub_pairs, batch_idx, n_sys, **self.bo
        )
        e_pair, e_atom = pairing_energy(J, kappa, self.temperature, sol.p, sol.u, fam.valence)
        return sol, J, kappa, e_pair, e_atom

    # -- forward -------------------------------------------------------------------------

    def forward(
        self,
        batch,
        state: StateDescriptor | None = None,
        *,
        with_polarizability: bool = False,
        with_induction: bool | None = None,
    ) -> PairingOutput:
        induction = self.induction if with_induction is None else bool(with_induction)
        positions = batch.positions
        n_atoms = int(positions.shape[0])
        species_idx = self.projector.species_index(batch.atomic_numbers)
        if state is None:
            state = StateDescriptor.from_batch(
                batch, species_idx, self.projector.featurizer.n_species
            )
        frag = state.fragment_idx
        n_frag = int(state.n_fragments)
        n_sys = int(batch.n_systems)
        f2b = state.fragment_to_batch
        batch_idx = batch.batch_idx

        # --- features, conditioning, parameters ------------------------------------------
        pf = self.projector(batch, state)
        c_cond = state.local_conditioning(self.state_embedding)
        # The SQE channel graph: every pair within the pairing radius (charge may flow
        # wherever a bond order says the atoms are one fragment) plus the assigned intra
        # enumeration, frame-grouped.
        ch_ind, chb_ind, ch_radius = union_channels(
            positions, batch_idx, frag, self.pairing_cutoff,
            max_num_neighbors=self.max_num_neighbors,
        )
        params: PairingParameters = self.network(pf, c_cond, state, positions, ch_ind)

        # --- one pair list ----------------------------------------------------------------
        pair_index, r, is_intra, pair_frag = union_pairs(
            positions, batch_idx, frag, self.cutoff_max,
            max_num_neighbors=self.max_num_neighbors,
        )
        i, j = pair_index[0], pair_index[1]
        dr_au = (positions[j] - positions[i]) / BOHR_ANG
        r_au = r / BOHR_ANG
        pair_batch = batch_idx[i]
        keys = i * n_atoms + j

        def pool_batch(x):
            return x.new_zeros(n_sys).index_add_(0, pair_batch, x)

        # --- the bond order at theta_0 ---------------------------------------------------
        sub_mask = r < self.pairing_cutoff
        sub_index = torch.nonzero(sub_mask, as_tuple=False).squeeze(-1)
        sub_pairs = pair_index[:, sub_index]
        sol0, J0, kappa0, e_pair0, e_atom0 = self._pairing(
            params.pairing0, positions, sub_pairs, dr_au[sub_index], r_au[sub_index],
            r[sub_index], batch_idx, n_sys,
        )
        c = comembership_from_bond_order(
            sol0.p, sub_index, pair_index, n_atoms, include_13=self.include_13
        )
        bo_solver = {"pairing0": (sol0.n_iter, sol0.converged, sol0.residual)}

        gate, r0, r0_pair, alpha, log_r0_prior, log_r0_prior_pair = self._gates(
            r, species_idx, pair_index
        )

        # --- classical channels -----------------------------------------------------------
        quad_c = (
            None if params.quad_perm is None
            else spherical_to_cartesian_quadrupole(params.quad_perm)
        )
        z0, b0 = params.response0.z, params.response0.b
        m_real = build_polytensor(
            params.q_perm, params.mu_perm, quad_c, max_rank=self.max_rank
        )
        m_nuc = build_polytensor(z0, None, None, max_rank=self.max_rank)
        e_elst = slater_elec_pair_energy(
            positions, pair_index, b0, None, m_real, m_nuc, dr_au=dr_au, r_au=r_au
        )

        spec_pauli = self.classical["pauli"]
        pq0, pb0, pmu0, pquad0 = params.pauli0
        pq, pb, pmu, pquad = params.pauli if spec_pauli.environment else params.pauli0
        poly0 = build_polytensor(
            pq0, pmu0,
            None if pquad0 is None else spherical_to_cartesian_quadrupole(pquad0),
            max_rank=self.max_rank,
        )
        poly = (
            poly0 if pq is pq0 else build_polytensor(
                pq, pmu,
                None if pquad is None else spherical_to_cartesian_quadrupole(pquad),
                max_rank=self.max_rank,
            )
        )
        poly_i = self._route(poly0[i], poly[i], c)
        poly_j = self._route(poly0[j], poly[j], c)
        b_p_i = self._route(pb0[i], pb[i], c)
        b_p_j = self._route(pb0[j], pb[j], c)
        e_pauli = slater_pauli_pair_energy(
            positions, pair_index, poly_i, poly_j,
            (0.5 * (b_p_i.log() + b_p_j.log())).exp(), dr_au=dr_au, r_au=r_au,
        )

        spec_disp = self.classical["disp"]
        c6_0, bd_0 = params.disp0
        c6_t, bd_t = params.disp if spec_disp.environment else params.disp0
        c6_i = self._route(c6_0[i], c6_t[i], c)
        c6_j = self._route(c6_0[j], c6_t[j], c)
        bd_i = self._route(bd_0[i], bd_t[i], c)
        bd_j = self._route(bd_0[j], bd_t[j], c)
        e_disp = ff_backend.tt_dispersion(
            positions, pair_index,
            (0.5 * (c6_i.log() + c6_j.log())).exp(),
            (0.5 * (bd_i.log() + bd_j.log())).exp(),
            r_ang=r,
        )

        e_pair = {
            "elst": gate["elst"] * e_elst,
            "pauli": gate["pauli"] * e_pauli,
            "disp": gate["disp"] * e_disp,
        }
        # Every channel is on between atoms exactly as much as the bond order says they are
        # *not* one fragment, ``(1 - c)``; the Pauli repulsion is additionally on *within* a
        # bond, at full strength, as its repulsive wall. The assignment only decides which
        # label a pair's energy is compared against.
        inter = ~is_intra
        interaction = {
            name: pool_batch(torch.where(inter, (1.0 - c) * value, torch.zeros_like(value)))
            for name, value in e_pair.items()
        }

        # --- fragment energies ------------------------------------------------------------
        e_class = (1.0 - c) * sum(e_pair.values()) + c * e_pair["pauli"]
        energy_intra = r.new_zeros(n_frag).index_add_(
            0, pair_frag[is_intra], e_class[is_intra]
        )
        sub_intra = is_intra[sub_index]
        sub_frag = pair_frag[sub_index]
        energy_pairing = (
            e_pair0.new_zeros(n_frag)
            .index_add_(0, sub_frag[sub_intra], e_pair0[sub_intra])
            .index_add_(0, frag, e_atom0)
        )
        e0 = self.reference_energies[species_idx]
        energy_ref = e0.new_zeros(n_frag).index_add_(0, frag, e0)
        fragment_energy = energy_ref + energy_pairing + energy_intra

        # What the assignment calls inter but the bond order calls bonded: zero until a
        # reaction, and the term that keeps the total an exact sum when one happens.
        cross = pool_batch(torch.where(inter, c * e_pair["pauli"], torch.zeros_like(c)))
        sub_batch = batch_idx[sub_pairs[0]]
        cross = cross.index_add(0, sub_batch[~sub_intra], e_pair0[~sub_intra])
        interaction["cross"] = cross

        # --- induction ---------------------------------------------------------------------
        level_ind = None
        energy_pairing_env = None
        solver: dict[str, tuple] = {}
        if induction:
            resp = params.response
            # SQE conductance by co-membership: charge flows along bonds, and along a
            # candidate channel exactly as much as the bond order says it is one.
            ch_pos, ch_found = _lookup(keys, ch_ind[0], ch_ind[1], n_atoms)
            c_chan = torch.where(ch_found, c[ch_pos], torch.zeros_like(c[ch_pos]))
            envelope = torch.where(
                ch_radius,
                pairwise_switch(
                    (positions[ch_ind[0]] - positions[ch_ind[1]]).norm(dim=-1),
                    self.pairing_cutoff - self.pairing_taper, self.pairing_cutoff,
                ),
                torch.ones_like(c_chan),
            )
            rp = ResponseParameters(
                chi=-resp.eta * params.q_perm,
                eta=resp.eta,
                q0=params.q_perm,
                compliance=resp.compliance * c_chan * envelope,
                chivec=None,
                alpha=resp.alpha,
                chiquad=None,
                cquad=None,
                z=resp.z,
                b=resp.b,
                mu0=params.mu_perm,
                quad0=params.quad_perm,
            )
            gate_ind = gate["elst"] * (1.0 - c)
            level_ind = coupled_response(
                rp, positions=positions, batch_idx=batch_idx, n_systems=n_sys,
                bond_index=ch_ind, bond_batch=chb_ind, pair_index=pair_index,
                gate=gate_ind, max_rank=self.max_rank, **self.cg,
            )
            solver["ind"] = (level_ind.n_iter, level_ind.converged, level_ind.pd_fail)

            e0_atom = -0.5 * rp.eta * params.q_perm * params.q_perm
            e0_internal = e0_atom.new_zeros(n_sys).index_add_(0, batch_idx, e0_atom)
            e0_ref = e0_internal + pool_batch(gate_ind * e_elst)

            # the pairing functional at theta: the bond's response to its surroundings
            sol_env, _, _, e_pair_env, e_atom_env = self._pairing(
                params.pairing, positions, sub_pairs, dr_au[sub_index], r_au[sub_index],
                r[sub_index], batch_idx, n_sys,
            )
            bo_solver["pairing"] = (sol_env.n_iter, sol_env.converged, sol_env.residual)
            d_pair = (
                (e_pair_env - e_pair0).new_zeros(n_sys)
                .index_add_(0, sub_batch, e_pair_env - e_pair0)
                .index_add_(0, batch_idx, e_atom_env - e_atom0)
            )
            energy_pairing_env = (
                e_pair_env.new_zeros(n_frag)
                .index_add_(0, sub_frag[sub_intra], e_pair_env[sub_intra])
                .index_add_(0, frag, e_atom_env)
            )
            interaction["induction"] = (level_ind.energy - e0_ref) + d_pair

        # --- polarizability (monomer anchors only) ------------------------------------------
        polarizability = None
        if with_polarizability:
            resp0 = params.response0
            ch_frag, chb_frag = intra_fragment_channels(frag)
            sol = sqe_solve(
                -resp0.eta * params.q_perm, resp0.eta, resp0.compliance, params.q_perm,
                positions, ch_frag, frag, chb_frag, n_frag,
                field=None, with_polarizability=True,
            )
            polarizability = fragment_polarizability(
                sol.alpha_flow, resp0.alpha, frag, n_frag
            )

        # --- assembly ------------------------------------------------------------------------
        energy = fragment_energy.new_zeros(n_sys).index_add_(0, f2b, fragment_energy)
        for value in interaction.values():
            energy = energy + value

        return PairingOutput(
            energy=energy,
            fragment_energy=fragment_energy,
            interaction=interaction,
            energy_ref=energy_ref,
            energy_bonded=energy_pairing,
            energy_intra=energy_intra,
            parameters=params,
            topology=None,
            pair_index=pair_index,
            r=r,
            is_intra=is_intra,
            pair_frag=pair_frag,
            p_intra=c,
            e_pair=e_pair,
            gate=gate,
            r0=r0,
            r0_pair=r0_pair,
            alpha=alpha,
            log_r0_prior=log_r0_prior,
            log_r0_prior_pair=log_r0_prior_pair,
            species_idx=species_idx,
            env_norm=pf.x_env.inv_feats.norm(dim=-1),
            a_env=pf.a_env,
            env_shift=params.env_shift(),
            conditioning=c_cond,
            polarizability=polarizability,
            level_ind=level_ind,
            energy_bonded_env=energy_pairing_env,
            solver=solver or None,
            sub_index=sub_index,
            bond_order=sol0.p,
            unpaired=sol0.u,
            coupling=J0,
            kappa_pair=kappa0,
            energy_pairing=energy_pairing,
            bo_solver=bo_solver,
        )
