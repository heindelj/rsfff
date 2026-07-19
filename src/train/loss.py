"""Energy + force loss for the total-energy MLIP.

Forces are the negative gradient of the predicted energy w.r.t. atomic positions,
obtained by autograd. During training we keep the graph (``create_graph=True``) so the
force term is itself differentiable and contributes to ``loss.backward()``.
"""

from __future__ import annotations

import torch


def compute_forces(
    energy: torch.Tensor, positions: torch.Tensor, *, create_graph: bool
) -> torch.Tensor:
    """Forces ``F = -dE/dx``. ``positions`` must be the leaf the energy depends on."""
    (grad,) = torch.autograd.grad(
        energy.sum(),
        positions,
        create_graph=create_graph,
        retain_graph=create_graph,
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
