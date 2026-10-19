"""``NonvariationalModel``: the film with the solve replaced. See ``docs/fff_nonvariational.md``."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..film.model import FilmModel
from ..film.network import FilmParameters
from ..polarization import LevelOutput, coupled_response, unrolled_response
from ..response import ResponseParameters

__all__ = ["NonvariationalModel", "diagnose"]


class NonvariationalModel(FilmModel):
    """The film model with a fixed-``K`` unrolled induction and learned mutual damping.

    Everything outside the induction level is inherited unchanged: features, conditioning,
    parameter network, classical channels, bonded terms, the CT bond response, the accounting.
    What changes:

    * :meth:`_response_parameters` adds the quadrupole polarizability ``cquad`` when the
      response heads emit one (``induced_quadrupoles``);
    * :meth:`_induction_level` runs :func:`rsfff.ff.polarization.unrolled_response` with
      ``n_iter`` iterations, the induced widths ``b_ind`` from the response heads, the
      fragment as the block of the charge solve, and the global iterate weights.

    Args (beyond :class:`FilmModel`)
    ----
    n_iter          : ``K``, the number of mutual-induction iterations after the exact
                      fragment-internal solve. ``0`` is direct induction only.
    iterate_weights : learn global weights ``c_k`` on the corrections ``x^(k) - x^(k-1)``,
                      ``k = 2..K`` (``c_1 = 1`` fixed, all initialised at 1).
    with_residual   : also report the physical residual ``max |A x_K + b|`` per frame
                      (one extra matvec; ``solver["ind_residual"]``).
    """

    def __init__(
        self, *args, n_iter: int = 3, iterate_weights: bool = True,
        with_residual: bool = False, **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if int(n_iter) < 0:
            raise ValueError(f"n_iter must be >= 0, got {n_iter}")
        self.n_iter = int(n_iter)
        self.with_residual = bool(with_residual)
        self.iterate_weights_raw = None
        if iterate_weights and self.n_iter >= 2:
            # c_k = 1 + raw_k, k = 2..K; zero-init so a fresh model is the plain iterate
            self.iterate_weights_raw = nn.Parameter(torch.zeros(self.n_iter - 1))

    @property
    def iterate_weights(self) -> torch.Tensor | None:
        """``(K,)`` weights with ``c_1 = 1`` fixed, or ``None`` for the plain iterate."""
        if self.iterate_weights_raw is None:
            return None
        one = self.iterate_weights_raw.new_ones(1)
        return torch.cat((one, 1.0 + self.iterate_weights_raw))

    def _response_parameters(self, params: FilmParameters) -> ResponseParameters:
        rp = super()._response_parameters(params)
        rp.cquad = params.response.cquad
        return rp

    def _induction_level(
        self, rp: ResponseParameters, params: FilmParameters, *, positions, batch, state,
        bond_index, bond_batch, pair_index, gate, solver: dict,
    ) -> LevelOutput:
        level = unrolled_response(
            rp, positions=positions, batch_idx=batch.batch_idx, n_systems=int(batch.n_systems),
            bond_index=bond_index, bond_batch=bond_batch, pair_index=pair_index,
            gate=gate, max_rank=self.max_rank,
            n_iter=self.n_iter, iterate_weights=self.iterate_weights,
            b_ind=params.response.b_ind,
            fragment_idx=state.fragment_idx, n_fragments=int(state.n_fragments),
            # force training differentiates the loop twice; the torchff field kernel has a
            # first-order VJP only, so in training mode the loop takes the reference field
            differentiable=self.training,
            with_residual=self.with_residual,
        )
        solver["ind"] = (level.n_iter, level.converged, level.pd_fail)
        if level.residual is not None:
            solver["ind_residual"] = level.residual
        return level

    def converged_level(
        self, rp: ResponseParameters, *, positions, batch, bond_index, bond_batch, pair_index,
        gate, **cg,
    ) -> LevelOutput:
        """The converged solve of the *same* functional -- the diagnostic reference."""
        return coupled_response(
            rp, positions=positions, batch_idx=batch.batch_idx, n_systems=int(batch.n_systems),
            bond_index=bond_index, bond_batch=bond_batch, pair_index=pair_index,
            gate=gate, max_rank=self.max_rank, **{**self.cg, **cg},
        )


@torch.no_grad()
def diagnose(model: NonvariationalModel, batch, state=None, **cg) -> dict[str, torch.Tensor]:
    """How far the unrolled state is from the minimum of its own functional, per frame.

    Runs the model once, then the converged solve at the same parameters, and reports the
    energy gap ``E(x_K) - E(x*)`` (Hartree, ``>= 0`` when the functional is PD), the max
    difference of the induced charges and dipoles, and the residual. Validation-time only.
    """
    from ..film.model import StateDescriptor
    from ..film.bonded import BondedTopology
    from ..pairs import union_channels, union_pairs

    was_training = model.training
    model.eval()
    model.with_residual, keep = True, model.with_residual
    try:
        out = model(batch, state)
    finally:
        model.with_residual = keep
        model.train(was_training)
    if state is None:
        species_idx = model.projector.species_index(batch.atomic_numbers)
        state = StateDescriptor.from_batch(batch, species_idx, model.projector.featurizer.n_species)
    frag = state.fragment_idx
    ch_ind, chb_ind, _ = union_channels(batch.positions, batch.batch_idx, frag, 0.0)
    rp = model._response_parameters(out.parameters)
    gate_ind = out.gate["elst"] * (1.0 - out.p_intra)
    ref = model.converged_level(
        rp, positions=batch.positions, batch=batch, bond_index=ch_ind, bond_batch=chb_ind,
        pair_index=out.pair_index, gate=gate_ind, **cg,
    )
    lev = out.level_ind
    res = {
        "energy_gap": lev.energy - ref.energy,
        "dq_max": (lev.charges - ref.charges).abs().amax(),
        "residual": out.solver["ind_residual"],
        "n_iter_cg": torch.tensor(ref.n_iter),
    }
    if lev.mu is not None:
        res["dmu_max"] = (lev.mu - ref.mu).abs().amax()
    return res
