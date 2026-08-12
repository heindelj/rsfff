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

    if force_weight > 0.0 and batch.forces is not None:
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


def fragment_multipole_loss(
    out,
    batch,
    *,
    dipole_weight: float = 0.0,
    quadrupole_weight: float = 0.0,
    dipole_scale: float = 0.05,
    quadrupole_scale: float = 0.2,
):
    """Fit per-fragment molecular multipoles to the frozen-monomer reference values.

    ``out`` is an :class:`rsfff.ff.electrostatics.ElectrostaticsOutput`; the labels are
    ``batch.fragment_dipole`` / ``batch.fragment_second_moment``, written by
    ``scripts/parse_roundtrip.py`` from the isolated-fragment blocks of a Q-Chem EDA job.

    Prediction and label go through the *same* origin shift and Buckingham projection
    (:mod:`rsfff.ff.molecular_multipoles`), so a sign or factor error in either cancels
    rather than biasing the fit. The quadrupole comparison is over the five independent
    components of the traceless tensor -- the trace carries the electron-cloud spread,
    which no point-multipole model can represent, and is removed structurally rather than
    weighted down.

    Errors are divided by their scale *before* squaring, the invariant
    :mod:`rsfff.train.term_loop` is built on, so both weights are dimensionless and each
    term equals 1.0 at an error of one scale. Defaults: 0.05 e*a0 (~0.13 D, against
    monomer dipoles of ~0.75) and 0.2 e*a0^2 (against Buckingham components of order 1).

    **What this can and cannot do.** It constrains the *molecular* multipole only. Many
    different ``(q_i, mu_i, Theta_i)`` reproduce the same fragment dipole and quadrupole,
    so this makes the model's long-range field right without making the distributed
    multipoles themselves physical; that would need a DMA or MBIS target.

    Returns ``(dict[str, Tensor], dict[str, float])`` -- loss terms and metrics. Both are
    empty when the weights are zero or the labels are absent.
    """
    from ..ff.molecular_multipoles import fragment_multipoles, reference_multipoles

    terms: dict[str, torch.Tensor] = {}
    metrics: dict[str, float] = {}
    want_dipole = dipole_weight > 0.0 and batch.fragment_dipole is not None
    want_quad = quadrupole_weight > 0.0 and batch.fragment_second_moment is not None
    if not (want_dipole or want_quad):
        return terms, metrics

    quad_c = None
    if out.quad_s is not None:
        from ..ff.multipole import spherical_to_cartesian_quadrupole

        quad_c = spherical_to_cartesian_quadrupole(out.quad_s)

    args = (batch.fragment_idx, int(batch.n_fragments), batch.fragment_charge)
    pred_dipole, pred_theta = fragment_multipoles(
        out.charges, batch.positions, batch.atomic_numbers, *args, out.mu, quad_c
    )
    ref_dipole, ref_theta = reference_multipoles(
        batch.fragment_dipole, batch.fragment_second_moment,
        batch.positions, batch.atomic_numbers, *args,
    )

    if want_dipole:
        err = (pred_dipole - ref_dipole) / dipole_scale
        terms["dipole"] = dipole_weight * err.pow(2).sum(-1).mean()
        metrics["dip_mae"] = float(
            (pred_dipole - ref_dipole).detach().norm(dim=-1).mean()
        )
    if want_quad:
        err = (pred_theta - ref_theta) / quadrupole_scale
        # Frobenius norm over the 3x3 counts the six off-diagonal entries twice, which is
        # a uniform factor on a traceless symmetric tensor and so only rescales the weight.
        terms["quad"] = quadrupole_weight * err.pow(2).sum((-1, -2)).mean()
        metrics["quad_mae"] = float(
            (pred_theta - ref_theta).detach().abs().amax(dim=(-1, -2)).mean()
        )
    return terms, metrics


def onebody_fit(out, batch, cfg):
    """Fit term for the 1-body head: per-fragment energy.

    Signature and return match ``rsfff.train.term_loop.eda_component_fit``: the shared loop
    calls whichever it is given. The differences that make a separate one necessary are that
    the target is ``batch.fragment_energy`` -- one label per *fragment*, so a pentamer frame
    is five training examples -- and that ``OneBodyOutput`` has no ``energy_ff`` for the
    default's ``corr_share`` metric.

    Forces are deliberately *not* here: on a cluster frame the reference gradient is the sum
    over every term of the force field and cannot be attributed to this one. They enter
    through :func:`onebody_anchor_loss` on the isolated-monomer frames, where the QC gradient
    really is ``-dE_1body/dR``.
    """
    err = out.fragment_energy - batch.fragment_energy
    loss = (err / cfg.energy_scale).pow(2).mean()
    from ..ff.units import KJMOL_PER_HARTREE

    metrics = {
        "mae": float(err.detach().abs().mean()) * KJMOL_PER_HARTREE,
        "rmse": float(err.detach().pow(2).mean().sqrt()) * KJMOL_PER_HARTREE,
        "bond": float(out.energy_bond.detach().mean()),
    }
    return loss, metrics, batch.fragment_energy


def onebody_anchor_loss(model, anchor_batch, cfg, out=None):
    """Energy and force terms on the isolated-monomer anchor: ``(dict, dict)``.

    ``out`` is an already-computed :class:`rsfff.ff.onebody.OneBodyOutput` for
    ``anchor_batch``. The joint model passes one in so the anchor -- whose forward includes
    a backward pass for the forces -- is evaluated once per step rather than once here and
    again for the multipole terms.

    ``anchor_batch`` is the ragged Batch from ``data.load_monomer_batch`` -- standalone
    monomers, one fragment each, carrying analytic forces. Because the system *is* one
    fragment, ``out.energy`` and ``out.fragment_energy`` coincide and the reference gradient
    is exactly this term's, which is what makes a force fit meaningful here and nowhere else
    yet.

    ``anchor_batch.positions`` must already have ``requires_grad_(True)``; the caller sets it
    once, since the anchor is a fixed batch reused every step.

    Grad is forced on for the whole body regardless of the caller's context: a force is
    itself a backward pass, so it has to be computed even on an evaluation epoch where the
    surrounding loop has disabled autograd. ``create_graph`` is taken from the context as it
    was on entry, so a training step gets the second-order graph the optimizer needs and an
    evaluation step does not pay for one.
    """
    from ..ff.units import KJMOL_PER_HARTREE

    training = torch.is_grad_enabled()
    with torch.enable_grad():
        if out is None:
            out = model(anchor_batch)
        err = out.fragment_energy - anchor_batch.fragment_energy
        terms = {"anchor_e": (err / cfg.energy_scale).pow(2).mean()}
        metrics = {"a_mae": float(err.detach().abs().mean()) * KJMOL_PER_HARTREE}

        if cfg.force_weight > 0.0:
            if anchor_batch.forces is None:
                raise ValueError(
                    "onebody.force_weight > 0 but the monomer anchor carries no forces; the "
                    "files from scripts/parse_roundtrip.py do"
                )
            forces = compute_forces(
                out.energy, anchor_batch.positions, create_graph=training
            )
            f_err = (forces - anchor_batch.forces) / cfg.force_scale
            terms["anchor_f"] = cfg.force_weight * f_err.pow(2).sum(-1).mean()
            metrics["f_mae"] = float((forces - anchor_batch.forces).detach().abs().mean())
    return terms, metrics
