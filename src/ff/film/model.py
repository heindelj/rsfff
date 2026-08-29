"""The film model: one projection, one conditioned parameter set, one potential evaluation.

The assembly parallels :class:`rsfff.ff.expert_model.FragmentExpertModel` with the frozen SQE
solve removed and the NN bond energy replaced by the physically-parameterized bonded terms::

    E_f     = sum_i E0[Z_i] + E_bonded(theta_0) + sum_intra gate(theta_0) E_class(theta_0)

    E_inter = sum_{inter, c} gate_c E_class^c          -> eda_cls_elec / mod_pauli / disp
              (elst at the PERMANENT multipoles: rigorously pairwise, like its label)

    E_ind   = [ coupled solve around the permanent multipoles, at theta ]
            - [ the same functional at zero response ]
            + sum_f [ E_bonded(theta) - E_bonded(theta_0) ]

    E_total = sum_f E_f + sum_c E_inter^c + E_ind

**The zero-response construction.** The coupled solve receives ``q0 = q_perm`` and
``chi = -eta * q_perm``, so the charge-sector drive ``chi + eta q0`` vanishes identically; the
dipole/quadrupole sectors run in the direct ``mu0``/``quad0`` form, whose drives vanish at the
permanent moments by construction. Every remaining source term comes through the gated
inter-fragment coupling (``gate_ind = gate_elst * (1 - P_ij)`` -- a C-projection, never a
distance rule), so an isolated fragment has an exactly zero right-hand side, an exactly zero
response, and ``E_ind == 0`` to machine precision. That is the vertex condition of
``docs/fff_film.md`` §5.1 realized in the solver rather than in a penalty, and it also means
``with_induction=False`` is an optimization rather than a correctness requirement for the
isolated streams (v4 needed it for both: its coupled level relaxed a lone fragment against its
own intramolecular field).

No ``eta``-slot or environment path reaches ``fragment_energy``: the bonded ``theta_0`` reads
the isolated latent, the permanent multipoles read the isolated latent by construction, and the
intra classical pairs read ``theta_0``. The v4 headline invariant survives with fewer moving
parts -- there is no ``energy_internal`` and no frozen-solve bookkeeping at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ...mlip.switch import pairwise_switch
from ...mlip.sqe import sqe_solve
from ..damping import fermi_switch
from ..dispersion import tt_damped_c6_energy
from ..electrostatics import slater_elec_pair_energy
from ..expert_model import ClassicalSpec
from ..fragment_state import FragmentStateEmbedding
from ..multipole import build_polytensor, spherical_to_cartesian_quadrupole
from ..pairs import intra_fragment_channels, union_channels, union_pairs
from ..pauli import slater_pauli_pair_energy
from ..polarization import LevelOutput, coupled_response
from ..response import ResponseParameters, fragment_polarizability
from ..units import BOHR_ANG
from .bonded import BondedTopology
from .network import ConditionedParameterNetwork, FilmParameters
from .projector import FragmentProjector
from .state import StateDescriptor

__all__ = ["DEFAULT_FILM_CLASSICAL", "FilmModel", "FilmOutput"]


#: ``elst`` reads the permanent multipoles (rigorously pairwise, like its label) and books no
#: environment difference anywhere -- the polarization is carried entirely by the coupled
#: solve. ``pauli``/``disp`` read ``theta`` on inter pairs as v4 does: their labels carry
#: genuine many-body content.
DEFAULT_FILM_CLASSICAL: dict[str, ClassicalSpec] = {
    "elst": ClassicalSpec(12.0, environment=False),
    "pauli": ClassicalSpec(7.0),
    "disp": ClassicalSpec(10.0),
}


@dataclass
class FilmOutput:
    """Per-frame totals, per-fragment energies, and the per-pair breakdown."""

    energy: torch.Tensor                      # (B,)
    fragment_energy: torch.Tensor             # (F,) -> batch.fragment_energy
    interaction: dict[str, torch.Tensor]      # elst / pauli / disp / induction, each (B,)
    energy_ref: torch.Tensor                  # (F,) sum_i E0[Z_i]
    energy_bonded: torch.Tensor               # (F,) Morse + angle at theta_0
    energy_intra: torch.Tensor                # (F,) intra classical pairs at theta_0
    parameters: FilmParameters                # every generated parameter, both evaluations
    topology: BondedTopology
    pair_index: torch.Tensor                  # (2, P)
    r: torch.Tensor                           # (P,) Angstrom
    is_intra: torch.Tensor                    # (P,) bool -- routing, never a gate
    pair_frag: torch.Tensor                   # (P,) fragment id, -1 for inter pairs
    p_intra: torch.Tensor                     # (P,) soft co-membership on the pair list
    e_pair: dict[str, torch.Tensor]           # (P,) gate * classical, per channel
    gate: dict[str, torch.Tensor]             # (P,) fermi * taper per channel
    r0: dict[str, torch.Tensor]               # (N,) per-atom, element table
    r0_pair: dict[str, torch.Tensor]          # (P,)
    alpha: dict[str, torch.Tensor]            # () per channel
    log_r0_prior: dict[str, torch.Tensor]     # (N,)
    log_r0_prior_pair: dict[str, torch.Tensor]  # (P,)
    species_idx: torch.Tensor                 # (N,)
    env_norm: torch.Tensor                    # (N,) ||x_env|| at lambda=0
    a_env: torch.Tensor                       # (N,) the smooth environment activity
    env_shift: dict[str, torch.Tensor]        # per-quantity |theta - theta_0|
    conditioning: torch.Tensor                # (N, d_c) c_i = [k_i, u_i]
    polarizability: torch.Tensor | None = None  # (F, 3, 3), monomer-anchor label units
    level_ind: "LevelOutput | None" = None
    energy_bonded_env: torch.Tensor | None = None  # (F,) bonded at theta (feeds induction)
    solver: dict[str, tuple] | None = None

    # Duck type for `rsfff.train.loss`: the multipole labels are frozen-monomer values, and
    # in this model the permanent multipoles *are* the frozen level.
    @property
    def charges(self) -> torch.Tensor:
        return self.parameters.q_perm

    @property
    def mu(self) -> torch.Tensor | None:
        return self.parameters.mu_perm

    @property
    def quad_s(self) -> torch.Tensor | None:
        return self.parameters.quad_perm


class FilmModel(nn.Module):
    """Projector + state conditioning + parameter network + one force-field evaluation.

    Args
    ----
    projector          : :class:`FragmentProjector` -- shared primitives + C-projection.
    state_embedding    : the ``(Q_f, 2S_f, n_f)`` block feeding the local state key ``k_i``.
    network            : :class:`ConditionedParameterNetwork` -- every generated parameter.
    range_heads        : element-only ``r0``/``alpha`` tables (reused verbatim from v4; the
                         v4 lesson stands: ``r0`` must not read the atom's description).
    reference_energies : (n_species,) isolated-atom energies, frozen buffer.
    """

    def __init__(
        self,
        projector: FragmentProjector,
        state_embedding: FragmentStateEmbedding,
        network: ConditionedParameterNetwork,
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
    ) -> None:
        super().__init__()
        self.projector = projector
        self.state_embedding = state_embedding
        self.network = network
        self.range_heads = range_heads
        self.classical = dict(classical or DEFAULT_FILM_CLASSICAL)
        self.max_rank = int(max_rank)
        self.induction = bool(induction)
        self.max_num_neighbors = int(max_num_neighbors)
        self.cutoff_max = max(c.cutoff for c in self.classical.values())
        self.cg = dict(rtol=float(cg_rtol), atol=float(cg_atol), maxiter=int(cg_maxiter))
        self.register_buffer("reference_energies", reference_energies.clone())

    # -- helpers -------------------------------------------------------------------------

    def _gates(self, r, species_idx, pair_index):
        """Element-table Fermi switch x compact taper, per channel."""
        i, j = pair_index[0], pair_index[1]
        zero_width = r.new_zeros(species_idx.shape[0], 0)
        r0, alpha = self.range_heads(zero_width, species_idx)
        gate, r0_pair, log_prior, log_prior_pair = {}, {}, {}, {}
        for c, name in enumerate(self.range_heads.channel_names):
            spec = self.classical[name]
            r0_ij = (0.5 * (r0[name][i].log() + r0[name][j].log())).exp()
            gate[name] = fermi_switch(r, r0_ij, alpha[name]) * pairwise_switch(
                r, spec.cutoff - spec.taper_width, spec.cutoff
            )
            r0_pair[name] = r0_ij
            prior = self.range_heads.log_r0_prior[c][species_idx]
            log_prior[name] = prior
            log_prior_pair[name] = 0.5 * (prior[i] + prior[j])
        return gate, r0, r0_pair, alpha, log_prior, log_prior_pair

    @staticmethod
    def _route(theta0, theta, p_intra):
        """Per-pair parameter mix: intra pairs read theta_0, inter read theta, smoothly in C.

        Linear interpolation by the pair co-membership -- the same convex rule
        ``soft_partition`` applies everywhere else. At a one-hot ``C`` it is a hard select.
        """
        if theta0 is theta:
            return theta0
        w = p_intra.reshape(-1, *([1] * (theta0.dim() - 1)))
        return w * theta0 + (1.0 - w) * theta

    # -- forward -------------------------------------------------------------------------

    def forward(
        self,
        batch,
        state: StateDescriptor | None = None,
        *,
        with_polarizability: bool = False,
        with_induction: bool | None = None,
    ) -> FilmOutput:
        induction = self.induction if with_induction is None else bool(with_induction)
        positions = batch.positions
        species_idx = self.projector.species_index(batch.atomic_numbers)
        if state is None:
            state = StateDescriptor.from_batch(
                batch, species_idx, self.projector.featurizer.n_species
            )
        frag = state.fragment_idx
        n_frag = int(state.n_fragments)
        n_sys = int(batch.n_systems)
        f2b = state.fragment_to_batch

        # --- features, conditioning, topology, parameters --------------------------------
        pf = self.projector(batch, state)
        c = state.local_conditioning(self.state_embedding)
        topo = BondedTopology.from_state(state, batch.atomic_numbers)
        # The SQE channel graph: complete intra-fragment enumeration, frame-grouped (charge
        # flows within fragments only; the coupled solve groups by frame).
        ch_ind, chb_ind, _ = union_channels(positions, batch.batch_idx, frag, 0.0)
        params = self.network(pf, c, state, topo, positions, ch_ind)

        # --- one pair list ----------------------------------------------------------------
        pair_index, r, is_intra, pair_frag = union_pairs(
            positions, batch.batch_idx, frag, self.cutoff_max,
            max_num_neighbors=self.max_num_neighbors,
        )
        i, j = pair_index[0], pair_index[1]
        dr_au = (positions[j] - positions[i]) / BOHR_ANG
        r_au = r / BOHR_ANG
        p_intra = state.edge_comembership(pair_index)
        pair_batch = batch.batch_idx[i]

        def pool_batch(x):
            return x.new_zeros(n_sys).index_add_(0, pair_batch, x)

        gate, r0, r0_pair, alpha, log_r0_prior, log_r0_prior_pair = self._gates(
            r, species_idx, pair_index
        )

        # --- classical channels -----------------------------------------------------------
        # elst at the PERMANENT multipoles with the isolated-evaluation penetration: a
        # function of the two fragments alone, exactly like its label.
        quad_c = (
            None if params.quad_perm is None
            else spherical_to_cartesian_quadrupole(params.quad_perm)
        )
        z0, b0 = params.response0.z, params.response0.b
        m_real = build_polytensor(
            params.q_perm, params.mu_perm, quad_c, max_rank=self.max_rank
        )
        m_shell = build_polytensor(
            params.q_perm - z0, params.mu_perm, quad_c, max_rank=self.max_rank
        )
        m_nuc = build_polytensor(z0, None, None, max_rank=self.max_rank)
        e_point, e_pen = slater_elec_pair_energy(
            dr_au, r_au, m_real, m_shell, m_nuc, b0, pair_index, max_rank=self.max_rank
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
        poly_i = self._route(poly0[i], poly[i], p_intra)
        poly_j = self._route(poly0[j], poly[j], p_intra)
        b_p_i = self._route(pb0[i], pb[i], p_intra)
        b_p_j = self._route(pb0[j], pb[j], p_intra)
        e_pauli = slater_pauli_pair_energy(
            dr_au, r_au, poly_i, poly_j,
            (0.5 * (b_p_i.log() + b_p_j.log())).exp(), max_rank=self.max_rank,
        )

        spec_disp = self.classical["disp"]
        c6_0, bd_0 = params.disp0
        c6_t, bd_t = params.disp if spec_disp.environment else params.disp0
        c6_i = self._route(c6_0[i], c6_t[i], p_intra)
        c6_j = self._route(c6_0[j], c6_t[j], p_intra)
        bd_i = self._route(bd_0[i], bd_t[i], p_intra)
        bd_j = self._route(bd_0[j], bd_t[j], p_intra)
        e_disp = tt_damped_c6_energy(
            r,
            (0.5 * (c6_i.log() + c6_j.log())).exp(),
            (0.5 * (bd_i.log() + bd_j.log())).exp(),
        )

        e_pair = {
            "elst": gate["elst"] * (e_point + e_pen),
            "pauli": gate["pauli"] * e_pauli,
            "disp": gate["disp"] * e_disp,
        }
        interaction = {
            name: pool_batch((1.0 - p_intra) * value) for name, value in e_pair.items()
        }

        # --- bonded -----------------------------------------------------------------------
        r_bond, cos_t = topo.geometry(positions)
        e_bond0, e_angle0 = params.bonded0.energy(r_bond, cos_t, topo)
        energy_bonded = (
            e_bond0.new_zeros(n_frag)
            .index_add_(0, topo.bond_frag, e_bond0)
            .index_add_(0, topo.angle_frag, e_angle0)
        )

        # --- fragment energies ------------------------------------------------------------
        # Intra classical pairs at theta_0. `pair_frag` is only valid where `is_intra`; a
        # soft assignment weighs the pair by its co-membership, exactly the accounting
        # `soft_partition` defines.
        e_intra_pair = (p_intra * sum(e_pair.values()))[is_intra]
        energy_intra = r.new_zeros(n_frag).index_add_(
            0, pair_frag[is_intra], e_intra_pair
        )
        e0 = self.reference_energies[species_idx]
        energy_ref = e0.new_zeros(n_frag).index_add_(0, frag, e0)
        fragment_energy = energy_ref + energy_bonded + energy_intra

        # --- induction ---------------------------------------------------------------------
        level_ind = None
        energy_bonded_env = None
        solver: dict[str, tuple] = {}
        if induction:
            resp = params.response
            rp = ResponseParameters(
                chi=-resp.eta * params.q_perm,
                eta=resp.eta,
                q0=params.q_perm,
                compliance=resp.compliance,
                chivec=None,
                alpha=resp.alpha,
                chiquad=None,
                cquad=None,
                z=resp.z,
                b=resp.b,
                mu0=params.mu_perm,
                quad0=params.quad_perm,
            )
            gate_ind = gate["elst"] * (1.0 - p_intra)
            level_ind = coupled_response(
                rp, positions=positions, batch_idx=batch.batch_idx, n_systems=n_sys,
                bond_index=ch_ind, bond_batch=chb_ind, pair_index=pair_index,
                gate=gate_ind, max_rank=self.max_rank, **self.cg,
            )
            solver["ind"] = (level_ind.n_iter, level_ind.converged, level_ind.pd_fail)

            # The same functional at zero response. The internal part collapses to
            # `chi q0 + 1/2 eta q0^2 = -1/2 eta q_perm^2` by the chi-trick; the pair part is
            # the gated inter-fragment elst at the permanent multipoles. (`rp.z/b` equal
            # `z0/b0` today -- the penetration heads carry no environment path -- so reusing
            # `e_pair["elst"]` here is exact; revisit if that ever changes.)
            e0_atom = -0.5 * rp.eta * params.q_perm * params.q_perm
            e0_internal = e0_atom.new_zeros(n_sys).index_add_(
                0, batch.batch_idx, e0_atom
            )
            e0_ref = e0_internal + pool_batch(gate_ind * (e_point + e_pen))

            e_bond_env, e_angle_env = params.bonded.energy(r_bond, cos_t, topo)
            energy_bonded_env = (
                e_bond_env.new_zeros(n_frag)
                .index_add_(0, topo.bond_frag, e_bond_env)
                .index_add_(0, topo.angle_frag, e_angle_env)
            )
            d_bond = energy_bonded_env - energy_bonded
            interaction["induction"] = (
                (level_ind.energy - e0_ref)
                + d_bond.new_zeros(n_sys).index_add_(0, f2b, d_bond)
            )

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

        return FilmOutput(
            energy=energy,
            fragment_energy=fragment_energy,
            interaction=interaction,
            energy_ref=energy_ref,
            energy_bonded=energy_bonded,
            energy_intra=energy_intra,
            parameters=params,
            topology=topo,
            pair_index=pair_index,
            r=r,
            is_intra=is_intra,
            pair_frag=pair_frag,
            p_intra=p_intra,
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
            conditioning=c,
            polarizability=polarizability,
            level_ind=level_ind,
            energy_bonded_env=energy_bonded_env,
            solver=solver or None,
        )
