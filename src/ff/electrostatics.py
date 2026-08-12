"""Classical electrostatics: a local SQE solve, Slater penetration, and a pair correction.

Unlike dispersion and Pauli, this term is **exactly two-body** -- and deliberately so.
Classical electrostatics between *frozen* monomer densities has no many-body content by
definition, and that is enforced structurally rather than hoped for:

1. the descriptor is grouped by ``fragment_idx``, so every atom's response parameters are a
   function of its own monomer alone;
2. the SQE solve is grouped by ``fragment_idx`` too, so charge cannot cross a fragment
   boundary even in principle;
3. the interaction is a sum over inter-fragment pairs of a function of those two monomers.

Every ``E^(k>=3)`` of the many-body expansion is therefore zero to round-off, which
``tests/test_ff_electrostatics.py`` checks rather than assumes.

The energy over inter-fragment pairs is

    E = sum_{i<j} [ S(r) T(r) ( E_point + E_pen )  +  W(r) dE_ij ]

    M      = [q, mu, Theta]      real multipoles, all three from the response solve:
                                     q      from chi/eta through the SQE charge solve
                                     mu     = -alpha chivec
                                     Theta  = -C chiquad          (max_rank 2 only)
    M_cp   = [q - Z, mu, Theta]  shell (electron) multipoles
    M_Z    = [Z, 0, ...]         point nucleus

    E_point = M_j^T    T(dr)                M_i          undamped, the 1/r tail
    E_pen   = M_cp_j^T [-f_2c(b_ij r) T]    M_cp_i       shell-shell
            + M_Z_j^T  [-f_1c(b_i   r) T]   M_cp_i       nucleus j sees shell i
            + M_cp_j^T [-f_1c(b_j   r) T]   M_Z_i        shell j sees nucleus i

    S(r) = fermi_switch(r; r0, alpha)   learned range separation, 0 short / 1 long
    T(r) = pairwise_switch(...)         neighbor-list taper, C2, exactly 0 at the cutoff
    W(r) = the pair head's own compact envelope

Penetration is **not** a small correction here: measured against ``eda_cls_elec`` with
pyCMM's fitted water multipoles, point multipoles alone reproduce only ~53% of the component
(MAE 13.7 kJ/mol on dimers), while adding penetration reaches MAE 3.25 and correlation 0.995.
That is why an effective nuclear charge ``Z`` is carried explicitly rather than folding the
short range into the neural correction.

**The SQE internal energy is computed and returned but never added to the interaction
energy.** The cost of spreading charge out within a monomer is 1-body, and this module fits
an interaction component; adding it would be a plausible-looking mistake, so
:class:`ElectrostaticsOutput` keeps it in a separate field with that name.

This is a **pre-training stage**. Once polarization enters, the response parameters become
functions of the whole environment, the two-body exactness above stops holding by design, and
the intramolecular response has to come from the full atomic-energy model instead.

Units
-----
================  ==============  ===========================================
quantity          unit            note
================  ==============  ===========================================
positions, r      Angstrom        model convention (rsfff.train.data)
cutoff, r0        Angstrom
q, Z              e
mu                e * bohr
Theta             e * bohr^2      Buckingham convention, traceless (rsfff.ff.multipole)
chi, eta          Hartree, Ha/e^2 the SQE functional's own units
b                 1 / bohr        Slater exponent
energy            Hartree
================  ==============  ===========================================

The single Angstrom -> bohr conversion happens once in :meth:`SlaterElectrostatics.forward`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..features.features import LambdaFeatures
from ..mlip.pair_heads import PairEnergyHead
from ..mlip.switch import pairwise_switch
from .damping import fermi_switch
from .multipole import (
    build_polytensor,
    damped_interaction_tensor,
    irrep2_to_spherical,
    multipole_pair_energy,
    slater_one_center_damp,
    slater_two_center_damp,
    spherical_to_cartesian_quadrupole,
)
from .pairs import inter_fragment_pairs
from .response import (
    DEFAULT_ELEC_PRIOR,
    DEFAULT_Q0_PRIOR,
    ElectrostaticParameterHeads,
    FragmentResponse,
    build_elec_priors,
    _inverse_softplus,
)
from .units import BOHR_ANG

#: The response parameterization and the per-fragment solve moved to
#: :mod:`rsfff.ff.response` when the 1-body term started sharing them; they are re-exported
#: here because this module's public surface predates that split.


@dataclass
class ElectrostaticsOutput:
    """Per-system energies, the solved multipoles, and the per-pair breakdown."""

    energy: torch.Tensor          # (B,) predicted interaction energy, Hartree
    energy_ff: torch.Tensor       # (B,) switched+tapered point + penetration
    energy_corr: torch.Tensor     # (B,) learned pair correction
    energy_point: torch.Tensor    # (B,) point-multipole part of energy_ff
    energy_pen: torch.Tensor      # (B,) penetration part of energy_ff
    #: SQE charge sector + atomic-dipole sector, per **fragment**. This is 1-body: the cost
    #: of polarizing a monomer internally. It is *never* part of ``energy``.
    internal_energy: torch.Tensor
    pair_index: torch.Tensor      # (2, P)
    r: torch.Tensor               # (P,) Angstrom
    e_pair_ff: torch.Tensor       # (P,)
    e_pair_corr: torch.Tensor     # (P,)
    charges: torch.Tensor         # (N,) e
    mu: torch.Tensor | None       # (N, 3) e*bohr
    quad_s: torch.Tensor | None   # (N, 5) spherical, e*bohr^2; -cquad * chiquad
    chiquad: torch.Tensor | None  # (N, 5) the rank-2 drive
    cquad: torch.Tensor | None    # (N,) isotropic quadrupole polarizability
    transfers: torch.Tensor       # (Nb,) split charges
    compliance: torch.Tensor      # (Nb,)
    chi: torch.Tensor             # (N,)
    eta: torch.Tensor             # (N,)
    z: torch.Tensor               # (N,) effective nuclear charge
    b: torch.Tensor               # (N,) Slater exponent
    r0: torch.Tensor              # () range-separation midpoint, Angstrom


class SlaterElectrostatics(nn.Module):
    """Point multipoles + Slater charge penetration over inter-fragment pairs.

    Args
    ----
    cutoff     : pair-list cutoff in Angstrom. 12.0 by default, far beyond the 7 used for
                 Pauli: this is the one term with a genuine ``1/r`` tail, and truncating it
                 early is a real error rather than a rounding one.
    r0_init    : Fermi midpoint. The switch gates the **whole** backbone (point *and*
                 penetration), turning it off at short range so the correction carries that
                 region -- the same arrangement as ``TTDispersion``. Because the switch tends
                 to 1 as ``r -> infinity`` the long-range asymptotics are untouched.
    """

    def __init__(
        self,
        response: FragmentResponse,
        correction: PairEnergyHead | None,
        *,
        cutoff: float = 12.0,
        taper_width: float = 1.0,
        r0_init: float = 1.5,
        alpha: float = 8.0,
        max_rank: int = 1,
        learn_r0: bool = True,
        max_num_neighbors: int = 512,
    ) -> None:
        super().__init__()
        if not cutoff > taper_width:
            raise ValueError(f"cutoff {cutoff} must exceed taper_width {taper_width}")
        if int(max_rank) != int(response.max_rank):
            # Both come from `ecfg.max_rank` in practice, but they are set independently, and
            # a mismatch is silent: heads at rank 1 with a rank-2 polytensor would just feed
            # zeros into the quadrupole block and fit slightly worse for no visible reason.
            raise ValueError(
                f"max_rank {max_rank} does not match the response heads' "
                f"{response.max_rank}; the heads decide which multipoles exist and this "
                f"decides which ones the interaction tensor carries"
            )
        self.response = response
        self.correction = correction
        self.cutoff = float(cutoff)
        self.taper_width = float(taper_width)
        self.alpha = float(alpha)
        self.max_rank = int(max_rank)
        self.max_num_neighbors = int(max_num_neighbors)
        self.r0_raw = nn.Parameter(
            torch.tensor(_inverse_softplus(r0_init)), requires_grad=learn_r0
        )

    @property
    def r0(self) -> torch.Tensor:
        """Range-separation midpoint in Angstrom (positive by construction)."""
        return torch.nn.functional.softplus(self.r0_raw)

    @property
    def params(self) -> ElectrostaticParameterHeads:
        """The response parameter heads (through the shared solve)."""
        return self.response.params

    def forward(
        self,
        batch,
        feats: LambdaFeatures,
        response=None,
    ) -> ElectrostaticsOutput:
        """``response`` lets a composite model solve once and hand the same result to the
        1-body term, so the two share one set of multipoles rather than two that drift."""
        positions = batch.positions
        res = self.response(batch, feats) if response is None else response
        q, mu, quad_s = res.charges, res.mu, res.quad_s
        z, b, internal = res.z, res.b, res.internal_energy

        pair_index, r = inter_fragment_pairs(
            positions, batch.batch_idx, self.cutoff,
            fragment_idx=batch.fragment_idx, max_num_neighbors=self.max_num_neighbors,
        )
        i, j = pair_index[0], pair_index[1]

        # The one Angstrom -> bohr conversion; everything below is atomic units.
        dr_au = (positions[j] - positions[i]) / BOHR_ANG
        r_au = r / BOHR_ANG
        r_inv = 1.0 / r_au

        quad_c = None if quad_s is None else spherical_to_cartesian_quadrupole(quad_s)
        m_real = build_polytensor(q, mu, quad_c, max_rank=self.max_rank)
        m_shell = build_polytensor(q - z, mu, quad_c, max_rank=self.max_rank)
        m_nuc = build_polytensor(z, None, None, max_rank=self.max_rank)

        # Point multipoles: the exact long-range interaction, undamped.
        t_point = damped_interaction_tensor(dr_au, None, r_inv, max_rank=self.max_rank)
        e_point = multipole_pair_energy(m_real[i], m_real[j], t_point)

        # Penetration. The leading minus is pyCMM's sign convention (rsfff.ff.multipole
        # returns the *overlap complement*), so the damped tensors are built from -f.
        b_ij = (0.5 * (b[i].log() + b[j].log())).exp()
        t_ss = damped_interaction_tensor(
            dr_au, -slater_two_center_damp(b_ij * r_au, self.max_rank), r_inv,
            max_rank=self.max_rank,
        )
        t_1c_i = damped_interaction_tensor(
            dr_au, -slater_one_center_damp(b[i] * r_au, self.max_rank), r_inv,
            max_rank=self.max_rank,
        )
        t_1c_j = damped_interaction_tensor(
            dr_au, -slater_one_center_damp(b[j] * r_au, self.max_rank), r_inv,
            max_rank=self.max_rank,
        )
        e_pen = (
            multipole_pair_energy(m_shell[i], m_shell[j], t_ss)
            + multipole_pair_energy(m_shell[i], m_nuc[j], t_1c_i)
            + multipole_pair_energy(m_nuc[i], m_shell[j], t_1c_j)
        )

        switch = fermi_switch(r, self.r0, self.alpha)
        taper = pairwise_switch(r, self.cutoff - self.taper_width, self.cutoff)
        gate = switch * taper
        e_pair_point = gate * e_point
        e_pair_pen = gate * e_pen
        e_pair_ff = e_pair_point + e_pair_pen

        if self.correction is not None:
            e_pair_corr = self.correction(
                feats.inv_feats, feats.species_idx, positions, pair_index, r
            )
        else:
            e_pair_corr = torch.zeros_like(e_pair_ff)

        n = batch.n_systems
        pair_batch = batch.batch_idx[i]

        def pool(x):
            return x.new_zeros(n).index_add_(0, pair_batch, x)

        energy_point, energy_pen = pool(e_pair_point), pool(e_pair_pen)
        energy_corr = pool(e_pair_corr)
        energy_ff = energy_point + energy_pen
        # `internal` is deliberately absent here: it is the 1-body cost of polarizing each
        # monomer, not part of the interaction energy this module predicts.
        return ElectrostaticsOutput(
            energy=energy_ff + energy_corr,
            energy_ff=energy_ff,
            energy_corr=energy_corr,
            energy_point=energy_point,
            energy_pen=energy_pen,
            internal_energy=internal,
            pair_index=pair_index,
            r=r,
            e_pair_ff=e_pair_ff,
            e_pair_corr=e_pair_corr,
            charges=q,
            mu=mu,
            quad_s=quad_s,
            chiquad=res.chiquad,
            cquad=res.cquad,
            transfers=res.transfers,
            compliance=res.compliance,
            chi=res.chi,
            eta=res.eta,
            z=z,
            b=b,
            r0=self.r0,
        )


class ElectrostaticsModel(nn.Module):
    """Featurizer + :class:`SlaterElectrostatics`, with intra-fragment features enforced.

    ``intra_fragment`` defaults to **True** and is the whole point: it makes the response
    parameters functions of their own monomer alone, hence the model exactly two-body.
    Turning it off requires ``allow_environment=True`` as well, because losing exactness
    would be invisible in the fit -- the model would simply look slightly better while
    silently ceasing to be classical electrostatics.
    """

    def __init__(
        self,
        featurizer,
        elec: SlaterElectrostatics,
        *,
        intra_fragment: bool = True,
        allow_environment: bool = False,
    ) -> None:
        super().__init__()
        if not intra_fragment and not allow_environment:
            raise ValueError(
                "intra_fragment=False makes the electrostatics environment-dependent and so "
                "no longer exactly two-body, which is the property this term exists to "
                "provide. Pass allow_environment=True if that is genuinely intended."
            )
        self.featurizer = featurizer
        self.elec = elec
        self.intra_fragment = bool(intra_fragment)

    @property
    def params(self) -> ElectrostaticParameterHeads:
        return self.elec.params

    @property
    def r0(self) -> torch.Tensor:
        return self.elec.r0

    def forward(self, batch) -> ElectrostaticsOutput:
        group = batch.fragment_idx if self.intra_fragment else None
        return self.elec(batch, self.featurizer(batch, group))


__all__ = [
    "DEFAULT_ELEC_PRIOR",
    "DEFAULT_Q0_PRIOR",
    "FragmentResponse",
    "build_elec_priors",
    "ElectrostaticParameterHeads",
    "ElectrostaticsOutput",
    "SlaterElectrostatics",
    "ElectrostaticsModel",
    "irrep2_to_spherical",
]
