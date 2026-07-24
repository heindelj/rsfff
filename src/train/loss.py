"""Losses for the total-energy MLIP and the MLIP+EEM model.

Forces are the negative gradient of the predicted energy w.r.t. atomic positions,
obtained by autograd. During training we keep the graph (``create_graph=True``) so the
force term is itself differentiable and contributes to ``loss.backward()``.

The EEM losses add dipole, dipole-derivative, and polarizability terms. The
molecular polarizability is *analytic* in the EEM model (``EEMOutput.alpha``), so
its loss is a plain MSE; dipole derivatives are obtained by differentiating the
model dipole w.r.t. positions (3 backward passes, one per component of mu).
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
    """Field-route polarizability ``alpha = d mu / dF`` by autograd: (M, 3, 3).

    For the closed-form EEM model this must equal the analytic ``EEMOutput.alpha``
    exactly; it is kept as an independent cross-check (tests/diagnostics), not
    for training -- the analytic route is cheaper.
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


def eem_model_loss(
    out,                       # EEMOutput
    batch,
    *,
    energy_weight: float = 1.0,
    force_weight: float = 1.0,
    dipole_weight: float = 0.0,
    dmu_dr_weight: float = 0.0,
    alpha_weight: float = 0.0,
    q_l2_weight: float = 0.0,
    create_graph: bool = False,
):
    """Weighted multi-target loss for the MLIP+EEM model.

    Each term is computed only when its weight is > 0 *and* the batch carries the
    target; skipped terms are absent from the metrics dict. Forces and d mu/dR
    trigger their own backward passes here, so call this once per minibatch; the
    polarizability is the model's analytic ``out.alpha`` (plain MSE, no extra
    backwards). Returns ``(loss, metrics)`` with detached float metrics.
    """
    counts = torch.bincount(batch.batch_idx, minlength=batch.n_systems).clamp(min=1)
    metrics: dict[str, float] = {}

    e_err = out.energy - batch.energy
    loss = energy_weight * (e_err.pow(2) / counts).mean()
    metrics["energy_mae"] = float(e_err.detach().abs().mean())

    if force_weight > 0.0:
        # retain_graph: the dipole-derivative term below runs further backward
        # passes over the same graph even at eval.
        forces = compute_forces(
            out.energy, batch.positions, create_graph=create_graph, retain_graph=True
        )
        f_err = forces - batch.forces
        loss = loss + force_weight * f_err.pow(2).sum(dim=-1).mean()
        metrics["force_mae"] = float(f_err.detach().abs().mean())

    if dipole_weight > 0.0 and batch.dipole is not None:
        mu_err = out.dipole - batch.dipole
        loss = loss + dipole_weight * mu_err.pow(2).sum(dim=-1).mean()
        metrics["dipole_mae"] = float(mu_err.detach().abs().mean())

    if dmu_dr_weight > 0.0 and batch.dipole_derivatives is not None:
        dmu = compute_dipole_derivatives(
            out.dipole, batch.positions, create_graph=create_graph
        )
        dmu_err = dmu - batch.dipole_derivatives
        loss = loss + dmu_dr_weight * dmu_err.pow(2).mean()
        metrics["dmu_dr_mae"] = float(dmu_err.detach().abs().mean())

    if alpha_weight > 0.0 and batch.polarizability is not None:
        a_err = out.alpha - batch.polarizability
        loss = loss + alpha_weight * a_err.pow(2).mean()
        metrics["alpha_mae"] = float(a_err.detach().abs().mean())

    if q_l2_weight > 0.0:
        loss = loss + q_l2_weight * out.charges.pow(2).mean()
    metrics["q_abs_mean"] = float(out.charges.detach().abs().mean())
    metrics["eta_min"] = float(out.eta.detach().min())

    return loss, metrics


def monomer_loss(out, batch, **kwargs):
    """Multi-target loss for the Phase-1 monomer stack.

    :class:`rsfff.mlip.MonomerOutput` exposes the same observables as ``EEMOutput``, so the
    term structure (energy / forces / dipole / dmu-dR / alpha / charge-L2) is shared with
    :func:`eem_model_loss`; this wrapper adds the split-charge diagnostics that only exist
    under SQE. Returns ``(loss, metrics)``.
    """
    loss, metrics = eem_model_loss(out, batch, **kwargs)
    if out.compliance.numel():
        metrics["s_mean"] = float(out.compliance.detach().mean())
        metrics["p_abs_max"] = float(out.transfers.detach().abs().max())
    return loss, metrics


def atomic_reference_loss(
    model,
    anchor_batch,
    states,
    *,
    energy_weight: float = 1.0,
    alpha_weight: float = 0.0,
    unbound_weight: float = 0.0,
):
    """Free-atom anchors: isolated-atom energy and polarizability at integer charge.

    These are the exact free-atom limit of the model (zero features, no channels, ``q = Q``),
    so they pin the per-element ``chi``/``eta``/``alpha`` rather than merely nudging them --
    see ``rsfff.mlip.monomer`` for why the limit is architectural.

    States flagged unbound at this level of theory (``AtomicStateReference.bound``) are scaled
    by ``unbound_weight``; the default of 0 drops them entirely. Their energies and
    polarizabilities are artifacts of whichever diffuse basis function the SCF fell into, so
    fitting them at full weight teaches the model basis-set noise.

    Returns ``(loss, metrics)`` with the max energy error over *bound* states for logging.
    """
    out = model(anchor_batch)
    w = torch.where(
        states.bound.to(out.energy.device),
        torch.ones_like(out.energy),
        torch.full_like(out.energy, float(unbound_weight)),
    )

    e_err = out.energy - anchor_batch.energy
    loss = energy_weight * (w * e_err.pow(2)).mean()
    metrics = {
        "atomic_e_max": float(e_err.detach()[states.bound].abs().max())
        if bool(states.bound.any()) else 0.0
    }

    if alpha_weight > 0.0 and anchor_batch.polarizability is not None:
        a_err = out.alpha - anchor_batch.polarizability
        loss = loss + alpha_weight * (w.view(-1, 1, 1) * a_err.pow(2)).mean()
        metrics["atomic_alpha_max"] = float(
            a_err.detach()[states.bound].abs().max()
        ) if bool(states.bound.any()) else 0.0

    return loss, metrics


def isolated_species_loss(model, iso_batch, weight: float = 1.0):
    """Energy MSE over the isolated-species anchor systems (integer charges).

    ``iso_batch`` is the ragged Batch from ``data.load_isolated_species`` (absolute
    energies in Hartree). Single atoms are fully determined by the constraint
    (q = Q) -- for the EEM model these anchors directly pin the per-element
    chi0/eta0 through E(Q) = E0 + E_MLIP(0) + chi0 Q + eta0 Q^2 / 2. Plain MSE --
    a handful of anchors that must be hit closely, hence typically a large
    ``weight``.
    """
    out = model(iso_batch)
    err = out.energy - iso_batch.energy
    return weight * err.pow(2).mean(), float(err.detach().abs().max())
