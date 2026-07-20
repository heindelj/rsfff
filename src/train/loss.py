"""Losses for the total-energy MLIP and the charge-aware model.

Forces are the negative gradient of the predicted energy w.r.t. atomic positions,
obtained by autograd. During training we keep the graph (``create_graph=True``) so the
force term is itself differentiable and contributes to ``loss.backward()``.

The charge-aware losses add dipole, dipole-derivative, and polarizability terms.
Dipole derivatives and the field-route polarizability are first derivatives of the
attached SCF charges, so they are obtained by differentiating the model dipole
w.r.t. positions / the applied field (3 backward passes each, one per Cartesian
component of mu).
"""

from __future__ import annotations

import torch


def compute_forces(
    energy: torch.Tensor,
    positions: torch.Tensor,
    *,
    create_graph: bool,
    retain_graph: bool | None = None,
) -> torch.Tensor:
    """Forces ``F = -dE/dx``. ``positions`` must be the leaf the energy depends on.

    ``retain_graph`` defaults to ``create_graph``; pass ``True`` explicitly when
    further backward passes (dipole derivatives, polarizability) follow at eval.
    """
    (grad,) = torch.autograd.grad(
        energy.sum(),
        positions,
        create_graph=create_graph,
        retain_graph=create_graph if retain_graph is None else retain_graph,
    )
    return -grad


def energy_force_loss(
    energy_pred: torch.Tensor,   # (B,)
    forces_pred: torch.Tensor,   # (Ntot, 3)
    batch,
    *,
    energy_weight: float = 1.0,
    force_weight: float = 1.0,
):
    """Weighted per-atom-normalized MSE on energy and forces.

    The energy MSE is divided by the atom count per molecule so molecules of different
    sizes contribute comparably (and it matches the intensive force MSE scale). Returns
    ``(loss, energy_mae, force_mae)`` where the MAEs are plain per-molecule / per-atom
    means for logging.
    """
    counts = torch.bincount(batch.batch_idx, minlength=batch.n_systems).clamp(min=1)
    e_err = energy_pred - batch.energy
    energy_mse = (e_err.pow(2) / counts).mean()
    force_mse = (forces_pred - batch.forces).pow(2).sum(dim=-1).mean()

    loss = energy_weight * energy_mse + force_weight * force_mse
    energy_mae = e_err.abs().mean().detach()
    force_mae = (forces_pred - batch.forces).abs().mean().detach()
    return loss, energy_mae, force_mae


def compute_dipole_derivatives(
    mu: torch.Tensor,          # (M, 3) molecular dipoles
    positions: torch.Tensor,   # (N, 3)
    *,
    create_graph: bool,
) -> torch.Tensor:
    """``d mu_b / d R_{i,a}`` as (N, 3, 3) with layout ``[i, a, b]``.

    One backward pass per dipole component. Molecules are independent (block
    structure), so summing ``mu[:, b]`` over molecules before the backward is
    exact -- each atom only receives its own molecule's contribution.
    """
    cols = []
    for b in range(3):
        (g,) = torch.autograd.grad(
            mu[:, b].sum(), positions, create_graph=create_graph, retain_graph=True
        )
        cols.append(g)                                          # (N, 3) = d mu_b / dR
    return torch.stack(cols, dim=-1)                            # (N, 3, 3) [i, a, b]


def compute_polarizability(
    mu: torch.Tensor,          # (M, 3) molecular dipoles (must depend on `field`)
    field: torch.Tensor,       # (M, 3) applied-field tensor with requires_grad
    *,
    create_graph: bool,
    symmetrize: bool = True,
) -> torch.Tensor:
    """Field-route polarizability ``alpha = d mu / dF = -d2 E*/dF2``: (M, 3, 3).

    Valid because ``dE*/dF = -mu`` at the SCF stationary point (envelope theorem)
    and the attach step carries ``dq*/dF``. Exactly symmetric only at exact
    stationarity -- the asymmetry is a convergence diagnostic, so ``symmetrize``
    can be turned off to inspect it.
    """
    rows = []
    for b in range(3):
        (g,) = torch.autograd.grad(
            mu[:, b].sum(), field, create_graph=create_graph, retain_graph=True
        )
        rows.append(g)                                          # (M, 3) = d mu_b / dF
    alpha = torch.stack(rows, dim=1)                            # (M, 3, 3) [m, b, a]
    if symmetrize:
        alpha = 0.5 * (alpha + alpha.transpose(-1, -2))
    return alpha


def charge_model_loss(
    out,                       # ChargeAwareOutput
    batch,
    *,
    energy_weight: float = 1.0,
    force_weight: float = 1.0,
    dipole_weight: float = 0.0,
    dmu_dr_weight: float = 0.0,
    alpha_weight: float = 0.0,
    alpha_consistency_weight: float = 0.0,
    q_l2_weight: float = 0.0,
    create_graph: bool = False,
):
    """Weighted multi-target loss for the charge-aware model.

    Each term is computed only when its weight is > 0 *and* the batch carries the
    target; skipped terms are absent from the metrics dict. Derivative targets
    (forces, d mu/dR, alpha) trigger their own backward passes here, so call this
    once per minibatch. Returns ``(loss, metrics)`` with detached float metrics
    (``*_mae`` entries, plus SCF diagnostics).
    """
    counts = torch.bincount(batch.batch_idx, minlength=batch.n_systems).clamp(min=1)
    metrics: dict[str, float] = {}

    e_err = out.energy - batch.energy
    loss = energy_weight * (e_err.pow(2) / counts).mean()
    metrics["energy_mae"] = float(e_err.abs().mean())

    if force_weight > 0.0:
        # retain_graph: the dipole-derivative / polarizability terms below run
        # further backward passes over the same graph even at eval.
        forces = compute_forces(
            out.energy, batch.positions, create_graph=create_graph, retain_graph=True
        )
        f_err = forces - batch.forces
        loss = loss + force_weight * f_err.pow(2).sum(dim=-1).mean()
        metrics["force_mae"] = float(f_err.abs().mean())

    if dipole_weight > 0.0 and batch.dipole is not None:
        mu_err = out.dipole - batch.dipole
        loss = loss + dipole_weight * mu_err.pow(2).sum(dim=-1).mean()
        metrics["dipole_mae"] = float(mu_err.abs().mean())

    if dmu_dr_weight > 0.0 and batch.dipole_derivatives is not None:
        dmu = compute_dipole_derivatives(
            out.dipole, batch.positions, create_graph=create_graph
        )
        dmu_err = dmu - batch.dipole_derivatives
        loss = loss + dmu_dr_weight * dmu_err.pow(2).mean()
        metrics["dmu_dr_mae"] = float(dmu_err.abs().mean())

    if alpha_weight > 0.0 and batch.polarizability is not None:
        alpha = compute_polarizability(out.dipole, out.field, create_graph=create_graph)
        a_err = alpha - batch.polarizability
        loss = loss + alpha_weight * a_err.pow(2).mean()
        metrics["alpha_mae"] = float(a_err.abs().mean())
        if alpha_consistency_weight >= 0.0:
            cons = (out.alpha_head - alpha).detach()
            metrics["alpha_head_vs_field_mae"] = float(cons.abs().mean())
        if alpha_consistency_weight > 0.0:
            loss = loss + alpha_consistency_weight * (out.alpha_head - alpha).pow(2).mean()

    # Head-based alpha supervised directly against the target as well: cheap, and
    # it is the piece the future long-range stage consumes.
    if alpha_weight > 0.0 and batch.polarizability is not None:
        ah_err = out.alpha_head - batch.polarizability
        loss = loss + alpha_weight * ah_err.pow(2).mean()
        metrics["alpha_head_mae"] = float(ah_err.abs().mean())

    if q_l2_weight > 0.0:
        loss = loss + q_l2_weight * out.charges.pow(2).mean()
    metrics["q_abs_mean"] = float(out.charges.detach().abs().mean())

    if out.scf_info is not None:
        metrics["scf_iters"] = float(out.scf_info.n_iter)
        metrics["scf_residual"] = out.scf_info.residual
        metrics["scf_min_eig"] = out.scf_info.min_tangent_eig

    return loss, metrics


def isolated_species_loss(model, iso_batch, weight: float = 1.0):
    """Energy MSE over the isolated-species anchor systems (integer charges).

    ``iso_batch`` is the ragged Batch from ``data.load_isolated_species`` (absolute
    energies in Hartree). Single atoms are fully determined by the constraint
    (q = Q), and the fragments (OH, OH-, H2O...) run the normal SCF. Plain MSE --
    these are a handful of anchors that must be hit closely, hence typically a
    large ``weight``.
    """
    out = model(iso_batch)
    err = out.energy - iso_batch.energy
    return weight * err.pow(2).mean(), float(err.abs().max())
