"""Training the mediator: the mixture stream and the charge-transfer prior.

``docs/fff_v2.md`` §8, *Training the mediator*. There is a label problem and it has to be stated
plainly: **a mixture has no ALMO-EDA label**. Every EDA channel is defined relative to a choice
of fragments, so ``eda_cls_elec`` for a 60/40 mixture of two decompositions is not a quantity
Q-Chem computed or could compute. What survives is what is fragmentation *invariant*:

============================  ===================================================
Supervises                    With
============================  ===================================================
the pure vertices             the four EDA channels, per fragmentation -- the
                              existing cluster stream, unchanged
the mixture                   ``E_total`` and forces, the only labels defined
                              where the membership is not one-hot
the mediator, as a prior      ``L_ct``, the weighted magnitude of induction
============================  ===================================================

This module owns the second and third rows. The first is untouched: the exploded
single-fragmentation dataset still trains every expert exactly as it did, and the mixture is a
term added on top rather than a replacement.

The charge-transfer prior
-------------------------
An inferior assignment announces itself by needing a large charge transfer -- it describes the
system as fragments that are wrong, and induction then has to move the electrons back. So::

    L_ct = lambda_ct * sum_groups sum_m  w_m * | E_ind^(m) |.detach()

**The detach is the safety property, not an optimization detail.** ``|E_ind|`` can be made small
two ways: pick the better assignment, or shrink the induction channel. The second is §8's flat
direction -- a second route into the ``E_bond`` / intra-classical degeneracy. Detached, the
penalty can *reweight* induction and can never *shrink* it, and the EDA induction label keeps
sole authority over its magnitude.

The magnitude has two sources and they are the same expression. Early, ``E_ind`` is a model
prediction and means nothing, so ``|eda_pol + eda_ct|`` is read off the vertex labels the corpus
carries; that makes ``L_ct`` a supervised ranking term and reuses exactly the quantity
``argmin |E_pol + E_ct|`` was measured on (398 of 399 frames, a 165-476 kJ/mol best-vs-second
gap). Later it falls back to the model's own detached induction, which needs no label and so
applies at geometries the corpus does not decompose at all.

What actually decides the mixture
---------------------------------
``L_ct`` is a *ranking* signal and the degree of mixing is a magnitude, so no ranking can set
it. Only the total-energy residual through the crossover can, and it can, precisely because
every decomposition of one geometry carries the *same* ``E_total`` while the model gives them
different answers. ``lambda_ct`` should therefore be small: it exists to keep the mediator out
of a bad basin early, not to decide it.
"""

from __future__ import annotations

import torch

from ..ff.mixture_model import mixture_forward

__all__ = ["MixtureStream"]


class MixtureStream:
    """A minibatch of mediated geometries per step, as a :func:`rsfff.train.term_loop.fit`
    penalty.

    Held as an object for the same reason :class:`rsfff.train.train_expert.IsolatedStreams` is:
    it owns a step counter, a generator and a device, and every term it produces is a penalty
    in ``fit``'s sense -- added to the loss and not derived from the cluster batch.
    """

    def __init__(
        self,
        model,
        mediator,
        groups,
        device,
        *,
        batch_size: int = 8,
        energy_weight: float = 1.0,
        force_weight: float = 1.0,
        energy_scale: float = 3.8093e-4,
        force_scale: float = 1.0e-3,
        ct_weight: float = 0.02,
        ct_scale: float = 3.8093e-4,
        force_every: int = 2,
        induction: bool = False,
        seed: int = 0,
    ) -> None:
        self.model = model
        self.mediator = mediator
        self.groups = list(groups)
        self.device = device
        self.batch_size = int(batch_size)
        self.energy_weight = float(energy_weight)
        self.force_weight = float(force_weight)
        self.energy_scale = float(energy_scale)
        self.force_scale = float(force_scale)
        self.ct_weight = float(ct_weight)
        self.ct_scale = float(ct_scale)
        self.force_every = max(int(force_every), 1)
        self.induction = bool(induction)
        self.generator = torch.Generator().manual_seed(int(seed))
        self._step = 0
        self._metrics: dict[str, float] = {}

    def _sample(self):
        if not self.groups:
            return []
        n = min(self.batch_size, len(self.groups))
        idx = torch.randperm(len(self.groups), generator=self.generator)[:n]
        return [self.groups[int(i)] for i in idx]

    def penalties(self, out, batch, cfg):
        """The loss terms this stream contributes. ``out``/``batch`` are the cluster
        stream's and go unread.

        Terms only, matching :meth:`rsfff.train.train_expert.IsolatedStreams.penalties`;
        metrics are stashed and collected by :meth:`diagnostics`, which is the same split that
        class uses and the reason its numbers reach the printed line at all.

        The mixture draws its own minibatch, because a mediated geometry is a different object
        from a cluster frame: it carries several fragmentations at once and cannot be packed
        into the ragged single-partition batch the rest of the model consumes.
        """
        self._metrics = {}
        if not self.groups or (self.energy_weight <= 0.0 and self.ct_weight <= 0.0):
            return {}
        self._step += 1
        want_forces = self.force_weight > 0.0 and (self._step % self.force_every == 0)

        e_err, f_err, ct, occ, splits = [], [], [], [], 0
        for group in self._sample():
            group = _to(group, self.device)
            if want_forces:
                group.positions.requires_grad_(True)
            out_mix = mixture_forward(
                self.model, group, self.mediator, with_induction=self.induction
            )
            if group.energy is not None:
                e_err.append((out_mix.energy - group.energy) / self.energy_scale)
            if want_forces and group.forces is not None:
                forces = -torch.autograd.grad(
                    out_mix.energy, group.positions, create_graph=True
                )[0]
                f_err.append(((forces - group.forces) / self.force_scale).flatten())

            # --- the shaping prior ---------------------------------------------------------
            magnitude = self._induction_magnitude(group, out_mix)
            if magnitude is not None:
                ct.append((out_mix.mediator.weights * magnitude).sum() / self.ct_scale)

            occupancy = float(out_mix.mediator.occupancy.detach())
            occ.append(occupancy)
            splits += int(occupancy > 0.05)

        terms: dict[str, torch.Tensor] = {}
        if e_err and self.energy_weight > 0.0:
            err = torch.stack(e_err)
            terms["mix_energy"] = self.energy_weight * err.pow(2).mean()
            self._metrics["mix_e_mae"] = float(
                err.detach().abs().mean() * self.energy_scale * 2625.4996
            )
        if f_err and self.force_weight > 0.0:
            err = torch.cat(f_err)
            terms["mix_force"] = self.force_weight * err.pow(2).mean()
            self._metrics["mix_f"] = float(
                err.detach().abs().mean() * self.force_scale
            )
        if ct and self.ct_weight > 0.0:
            terms["mix_ct"] = self.ct_weight * torch.stack(ct).mean()
            self._metrics["mix_ct"] = float(torch.stack(ct).detach().mean())
        if occ:
            self._metrics["pi_occ"] = sum(occ) / len(occ)
            self._metrics["pi_split"] = splits / len(occ)
        return terms

    def _induction_magnitude(self, group, out_mix):
        """``(M,)`` ``|E_ind|`` per decomposition, **detached**, or ``None``.

        The label where the corpus has one, the model's own induction otherwise. Detached
        either way: see the module docstring -- this term ranks assignments and must never be
        able to shrink the channel it ranks them by.
        """
        if group.vertex_induction_label is not None:
            return group.vertex_induction_label.detach()
        if out_mix.vertex_induction is None:
            return None
        return out_mix.vertex_induction.detach().abs()

    def diagnostics(self, out, batch, target) -> dict[str, float]:
        """Whatever the last :meth:`penalties` call measured. See its docstring for the split."""
        return dict(self._metrics)


def _to(group, device):
    """A :class:`rsfff.ff.mixture_model.MixtureGroup` moved to ``device``. Cheap; no copy on CPU."""
    from dataclasses import replace

    if group.positions.device == torch.device(device):
        return group
    move = lambda t: None if t is None else t.to(device)  # noqa: E731
    return replace(
        group,
        positions=move(group.positions),
        atomic_numbers=move(group.atomic_numbers),
        fragments=move(group.fragments),
        atom_charge=move(group.atom_charge),
        atom_two_s=move(group.atom_two_s),
        contested=move(group.contested),
        energy=move(group.energy),
        forces=move(group.forces),
        vertex_induction_label=move(group.vertex_induction_label),
    )
