"""An ASE calculator over the mediated model, with the bias and the wall folded in.

This is the first place in the repo where the fragment-expert model drives dynamics rather
than being scored against labels, and the whole of the "gradient of the routing weight" ask is
the four lines in :meth:`MediatedCalculator.calculate` that sum three energies and take one
backward. ``mixture_forward`` already returns ``w`` inside the graph; adding a function of it
to the total before ``autograd.grad`` is all that biasing on it requires.

Not built on ``benchmarks/scripts/benchmark_utils.RSFFFCalculator``. That one is pinned to the
archived v1 loader, infers fragments from a hardcoded ``O H H`` atom order, and predates the
mediator -- it is the right *shape* and none of the right code.
"""

from __future__ import annotations

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes

from ..ff.mediator import MediatorHead
from ..ff.mixture_model import intra_pairs_unsorted, mixture_forward
from ..mlip.reference_states import AtomicStateReference
from ..train.build_expert import build_expert_model
from ..train.data import load_reference_energies
from .assign import DEFAULT_BUMP, base_assignment, enumerate_group
from .bias import HarmonicBias
from .confine import flat_bottom_sphere

__all__ = ["MediatedCalculator", "load_mediated_model"]

#: The model works in Hartree and Angstrom; ASE wants eV and eV/Angstrom.
HARTREE_TO_EV = 27.211386245988


def load_mediated_model(checkpoint: str, *, device: str = "cpu"):
    """Rebuild a trained model **including** its mediator, and load strictly.

    ``build_expert_model`` does not create the mediator: ``train_expert._build_mixture_stream``
    attaches it after the fact as ``model.mediator``, so it is in the checkpoint's state dict
    but not in a freshly built model, and a strict load fails on the missing keys. Re-attaching
    it needs the slot widths, and those are read off a real emission rather than recomputed
    from the feature config -- the fragment slot carries the appended ``(Q, 2S, n)`` state
    block and is wider than ``feature_dims`` says.

    Returns ``(model, config, state)``.
    """
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = state["config"]
    torch.set_default_dtype(torch.float64 if cfg.dtype == "float64" else torch.float32)
    dtype = torch.get_default_dtype()

    nt = tuple(int(z) for z in state["neighbor_types"])
    e0 = load_reference_energies(cfg.data.reference_energies, nt).to(dtype)
    states = (
        AtomicStateReference.from_json(cfg.data.atomic_reference_states, nt, dtype=dtype)
        if cfg.data.atomic_reference_states
        else None
    )
    model = build_expert_model(cfg, nt, e0, states).to(dtype)

    # A minimal two-fragment probe, purely to measure the slot widths.
    probe_pos = torch.tensor(
        [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0],
         [2.8, 0.0, 0.0], [3.2, 0.9, 0.0], [3.2, -0.9, 0.0]], dtype=dtype
    )
    probe = enumerate_group(probe_pos, torch.tensor([8, 1, 1, 8, 1, 1]), 0).batch(0)
    em = model.emit(
        probe, probe.fragment_idx, bond_index=intra_pairs_unsorted(probe.fragment_idx)
    )
    p_frag = int(em.iso.inv_feats.shape[-1])
    p_env = int(em.joined.inv_feats.shape[-1]) - p_frag

    model.mediator = MediatorHead(
        p_frag, p_env,
        hidden=cfg.expert.mediator_hidden,
        depth=cfg.expert.mediator_depth,
        bump=dict(lo0=0.0, lo1=0.0,
                  hi1=cfg.expert.mediator_bump_hi1, hi0=cfg.expert.mediator_bump_hi0),
    ).to(dtype)
    model.load_state_dict(state["model_state"], strict=True)
    model.eval().to(device)
    return model, cfg, state


class MediatedCalculator(Calculator):
    """Energy and forces from the mediated model, plus an optional bias and spherical wall.

    ``results["energy"]`` is the **biased** total, because that is what has to drive the
    integrator; the unbiased model energy and every mediator diagnostic are stashed alongside
    it so the driver can log a step without paying for a second forward pass.

    Cost is set by ``M``, the number of live decompositions, since each one is its own
    featurization: measured 39 ms/step at ``M=3`` on 13 atoms without induction and 59 ms with
    it. ``results["n_decompositions"]`` is logged for exactly that reason.
    """

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(
        self,
        model,
        total_charge: int,
        *,
        bias: HarmonicBias | None = None,
        radius: float = 0.0,
        k_confine: float = 0.0,
        h_slack: float = 1.2,
        with_induction: bool = True,
        bump: dict | None = None,
        commit_threshold: float = 0.9,
    ) -> None:
        super().__init__()
        if not hasattr(model, "mediator"):
            raise ValueError(
                "the model has no `mediator` submodule; load it with `load_mediated_model`, "
                "which re-attaches the MediatorHead that `build_expert_model` does not create"
            )
        self.model = model
        self.total_charge = int(total_charge)
        self.bias = bias or HarmonicBias(k=0.0)
        self.radius, self.k_confine, self.h_slack = float(radius), float(k_confine), float(h_slack)
        self.with_induction = bool(with_induction)
        # Default to the *model's own* envelope, never a module constant: enumeration and
        # the head must agree about when a candidate is open, and a checkpoint trained with
        # a different `mediator_bump_hi1` would otherwise be pre-filtered against the wrong
        # one -- silently, since both envelopes look reasonable on their own.
        self.bump = dict(bump or getattr(model.mediator, "bump", None) or DEFAULT_BUMP)
        #: How decisively the mediator must prefer a different assignment before the held
        #: base is moved to it. Hysteresis, not a tie-break: rebuilding the base is the one
        #: genuinely discrete event in the trajectory, and doing it the instant the weights
        #: cross 0.5 would rebuild it repeatedly while the proton rattles in the well. At 0.9
        #: the candidate being promoted already carries almost all the weight, so the energy
        #: barely moves across the commit -- but it does move, so `n_commits` counts them.
        self.commit_threshold = float(commit_threshold)
        self._base: np.ndarray | None = None
        self.n_commits = 0
        self.dtype = next(model.parameters()).dtype

    def reset_base(self, atoms=None) -> None:
        """Forget the held assignment; the next call reseeds it from rank 0.

        Call this after moving atoms by anything other than dynamics -- loading a new
        geometry, restarting a trajectory -- so the base describes the structure in front of
        it rather than the one it last saw.
        """
        self._base = None
        self.n_commits = 0

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        pos = torch.tensor(
            self.atoms.get_positions(), dtype=self.dtype, requires_grad=True
        )
        z = torch.as_tensor(self.atoms.get_atomic_numbers(), dtype=torch.long)

        with torch.enable_grad():
            group = enumerate_group(
                pos, z, self.total_charge, bump=self.bump, base=self._base
            )
            out = mixture_forward(
                self.model, group, self.model.mediator, with_induction=self.with_induction
            )
            e_bias, cv = self.bias(out, group)
            e_conf = flat_bottom_sphere(
                pos, z, radius=self.radius, k=self.k_confine, h_slack=self.h_slack
            )
            total = out.energy + e_bias + e_conf
            (grad,) = torch.autograd.grad(total, pos)

        self._commit(group, out)

        energy = float(total.detach()) * HARTREE_TO_EV
        self.results["energy"] = energy
        self.results["free_energy"] = energy
        self.results["forces"] = -grad.detach().cpu().numpy() * HARTREE_TO_EV
        # Everything below is in Hartree/Angstrom, the model's own units -- these are
        # diagnostics for analysis, not quantities ASE consumes.
        self.results.update(
            energy_hartree=float(out.energy.detach()),
            bias_energy=float(e_bias.detach()),
            confine_energy=float(e_conf.detach()),
            collective_variable=float(cv.detach()),
            weights=out.mediator.weights.detach().cpu().numpy(),
            omega=out.mediator.omega.detach().cpu().numpy(),
            occupancy=float(out.mediator.occupancy.detach()),
            contested=group.contested.cpu().numpy(),
            fragment_idx=group.fragments.cpu().numpy(),
            fragment_charge=group.atom_charge.detach().cpu().numpy(),
            n_decompositions=int(group.fragments.shape[0]),
            n_commits=self.n_commits,
        )

    def _commit(self, group, out) -> None:
        """Move the held base onto the winning decomposition, once the mediator is sure.

        The base has to follow the chemistry or the enumeration goes stale: after a transfer
        the old base describes an ion that is no longer there, and the candidate that *is*
        the chemistry sits at the edge of its envelope instead of at the centre of a fresh
        one. Following it the moment the weights cross 0.5 would instead thrash, because the
        weights cross back and forth every time the proton rattles.
        """
        w = out.mediator.weights.detach()
        winner = int(torch.argmax(w))
        if self._base is None:
            self._base = group.fragments[0].cpu().numpy().copy()
        if winner != 0 and float(w[winner]) >= self.commit_threshold:
            self._base = group.fragments[winner].cpu().numpy().copy()
            self.n_commits += 1

    def unbiased_energy(self, atoms) -> float:
        """The model energy alone, in Hartree, with no bias and no wall. For checks."""
        self.calculate(atoms)
        return float(self.results["energy_hartree"])


def snapshot_info(results: dict) -> dict:
    """The mediator diagnostics of one step, flattened for an extxyz ``info`` block.

    Arrays are formatted the way :mod:`rsfff.qcgen.multifrag` writes them, so a trajectory
    written with these keys is one step away from the multi-fragmentation schema the labeling
    pipeline reads.
    """
    frag = np.asarray(results["fragment_idx"])
    return {
        "n_fragmentations": int(frag.shape[0]),
        "n_fragments": int(frag.max()) + 1,
        "mediator_weights": " ".join(f"{v:.6f}" for v in results["weights"]),
        "ambiguity": float(1.0 - (np.asarray(results["weights"]) ** 2).sum()),
        "collective_variable": float(results["collective_variable"]),
        "contested": " ".join(str(int(v)) for v in results["contested"]),
        "energy_hartree": float(results["energy_hartree"]),
        "bias_energy": float(results["bias_energy"]),
        "confine_energy": float(results["confine_energy"]),
    }
