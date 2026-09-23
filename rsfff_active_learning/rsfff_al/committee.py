"""A committee of film models evaluated together on a batch of replicas, in torch.

Everything the uncertainty-driven sampler needs comes out of one call::

    c = Committee.load("path/to/film_committee_100k")          # dir, committee.json or .pt list
    top = Topology.water(n_waters=8, device=c.device)
    ev = c.evaluate(positions, top)                              # positions (R, N, 3) Angstrom
    ev.energy          (K, R)       Hartree, every member
    ev.forces          (K, R, N, 3) Hartree/Angstrom, every member
    ev.mean_forces     (R, N, 3)    the committee-mean force: what the dynamics follows
    ev.sigma_energy    (R,)         std over members of the TOTAL energy, Hartree
    ev.grad_sigma      (R, N, 3)    d sigma_energy / dR, Hartree/Angstrom
    ev.sigma_forces    (R,)         DeePMD model_devi_f: max over atoms of the RMS deviation

The gradient of the energy uncertainty needs no second derivatives. With
``sigma^2 = sum_k (E_k - Ebar)^2 / (K - 1)`` and ``sum_k (E_k - Ebar) = 0``,

    d sigma / dR = sum_k (E_k - Ebar) dE_k/dR / ((K - 1) sigma)

so it is a weighted sum of the members' own gradients, which the dynamics computes anyway to
get the mean force. One forward + one first-order backward per member per step, exactly what
unbiased committee MD costs; the coupled induction solve is never differentiated twice.

The force uncertainty ``sigma_forces`` is reported (it is free) but is not differentiated:
its gradient is a Hessian-vector product through the induction CG solve, which
``rsfff.md.film_driver`` deliberately never trusts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from rsfff.md.film_driver import load_film_model
from rsfff.train.data import Batch

__all__ = ["Committee", "Evaluation", "Topology", "member_checkpoints"]

MANIFEST = "committee.json"
MASS = {1: 1.00784, 8: 15.999}


def member_checkpoints(path) -> list[Path]:
    """Checkpoint paths of a committee.

    ``path`` is a committee directory (with or without ``committee.json``), a
    ``committee.json``, a single ``.pt``, or a list of ``.pt``. A manifest written on another
    machine carries that machine's absolute paths; when one does not exist here, the member is
    looked up by its path relative to the committee directory (``member_NN/member_NN_*/best.pt``).
    """
    if isinstance(path, (list, tuple)):
        return [Path(p) for p in path]
    path = Path(path)
    if path.suffix == ".pt":
        return [path]
    manifest = path if path.name == MANIFEST else path / MANIFEST
    root = manifest.parent
    if manifest.exists():
        found = []
        for entry in json.loads(manifest.read_text())["members"]:
            ckpt = Path(entry["checkpoint"])
            if not ckpt.exists():
                name = f"member_{int(entry['member']):02d}"
                local = sorted(root.glob(f"{name}/{name}_full/best.pt")) or sorted(
                    root.glob(f"{name}/*/best.pt")) or sorted(root.glob(f"{name}/best.pt"))
                if not local:
                    raise FileNotFoundError(f"{ckpt} (from {manifest}) not found, and no "
                                            f"{name}/*/best.pt under {root}")
                ckpt = local[-1]
            found.append(ckpt)
        return found
    found = (sorted(root.glob("member_*/member_*_full/best.pt"))
             or sorted(root.glob("member_*/*/best.pt")) or sorted(root.glob("member_*/best.pt")))
    if not found:
        raise FileNotFoundError(f"no committee members under {root}")
    return found


@dataclass
class Topology:
    """Species and fixed fragmentation of one cluster, shared by every replica.

    The film model takes one fragmentation for the whole trajectory (it is never recomputed
    mid-flight), so it is fixed here and the geometry guards check that the waters it names
    stay intact.
    """

    atomic_numbers: torch.Tensor     # (N,) long
    fragment_idx: torch.Tensor       # (N,) long, non-decreasing
    masses: torch.Tensor             # (N,) amu

    @property
    def n_atoms(self) -> int:
        return int(self.atomic_numbers.numel())

    @property
    def n_fragments(self) -> int:
        return int(self.fragment_idx.max()) + 1

    @classmethod
    def from_arrays(cls, atomic_numbers, fragment_idx, *, device="cpu") -> "Topology":
        z = torch.as_tensor(np.asarray(atomic_numbers), dtype=torch.long, device=device)
        frag = torch.as_tensor(np.asarray(fragment_idx), dtype=torch.long, device=device)
        if torch.any(frag[1:] < frag[:-1]):
            raise ValueError("fragment_idx must be non-decreasing (atoms in fragment order)")
        m = torch.tensor([MASS[int(v)] for v in z.cpu()], dtype=torch.get_default_dtype(),
                         device=device)
        return cls(z, frag, m)

    @classmethod
    def water(cls, n_waters: int, *, device="cpu") -> "Topology":
        """``O H H O H H ...``, molecule k = atoms 3k..3k+2 (packmol's order)."""
        return cls.from_arrays([8, 1, 1] * n_waters, np.repeat(np.arange(n_waters), 3),
                               device=device)

    def batch(self, positions: torch.Tensor, n_replicas: int) -> Batch:
        """One ragged graph of ``n_replicas`` copies, fragments re-offset per replica, every
        tensor on the positions' device (``film_driver.make_batch`` builds on the CPU)."""
        dev, dtype = positions.device, positions.dtype
        n, nf = self.n_atoms, self.n_fragments
        rep = torch.arange(n_replicas, device=dev)
        return Batch(
            positions=positions.reshape(n_replicas * n, 3),
            atomic_numbers=self.atomic_numbers.repeat(n_replicas),
            batch_idx=rep.repeat_interleave(n),
            n_systems=n_replicas,
            energy=torch.zeros(n_replicas, dtype=dtype, device=dev),
            fragment_idx=self.fragment_idx.repeat(n_replicas) + nf * rep.repeat_interleave(n),
            fragment_charge=torch.zeros(n_replicas * nf, dtype=dtype, device=dev),
            fragment_two_s=torch.zeros(n_replicas * nf, dtype=dtype, device=dev),
            fragment_to_batch=rep.repeat_interleave(nf),
            n_fragments=n_replicas * nf,
        )


@dataclass
class Evaluation:
    energy: torch.Tensor          # (K, R) Hartree
    forces: torch.Tensor          # (K, R, N, 3) Hartree/Angstrom
    failed: torch.Tensor          # (R,) bool: some member could not evaluate this replica

    @property
    def n_members(self) -> int:
        return int(self.energy.shape[0])

    @property
    def mean_energy(self) -> torch.Tensor:
        return self.energy.mean(0)

    @property
    def mean_forces(self) -> torch.Tensor:
        return self.forces.mean(0)

    @property
    def sigma_energy(self) -> torch.Tensor:
        """Sample std (ddof=1) of the total energy over members, Hartree. Zero for K=1."""
        if self.n_members < 2:
            return torch.zeros_like(self.energy[0])
        return self.energy.std(0, unbiased=True)

    def grad_sigma(self, eps: float = 1e-12) -> torch.Tensor:
        """d sigma_energy / dR, Hartree/Angstrom, (R, N, 3). Zero where sigma ~ 0.

        ``sum_k (E_k - Ebar) dE_k/dR / ((K-1) sigma)`` with ``dE_k/dR = -F_k``.
        """
        k = self.n_members
        if k < 2:
            return torch.zeros_like(self.forces[0])
        dev = self.energy - self.energy.mean(0, keepdim=True)            # (K, R)
        sigma = self.sigma_energy                                          # (R,)
        grad = -(dev[:, :, None, None] * self.forces).sum(0)              # (R, N, 3)
        scale = torch.where(sigma > eps, 1.0 / ((k - 1) * sigma.clamp(min=eps)),
                            torch.zeros_like(sigma))
        return grad * scale[:, None, None]

    @property
    def atom_force_deviation(self) -> torch.Tensor:
        """Per-atom RMS distance of the members' forces from their mean, (R, N)."""
        d = self.forces - self.forces.mean(0, keepdim=True)
        return (d ** 2).sum(-1).mean(0).sqrt()

    @property
    def sigma_forces(self) -> torch.Tensor:
        """DeePMD ``model_devi_f``: the largest per-atom deviation, Hartree/Angstrom, (R,)."""
        return self.atom_force_deviation.max(-1).values


class Committee:
    """K film checkpoints, evaluated member by member on one batch of replicas."""

    def __init__(self, models, paths, *, device: str = "cpu", with_induction: bool = True):
        if not models:
            raise ValueError("a committee needs at least one member")
        self.models = models
        self.paths = [Path(p) for p in paths]
        self.device = torch.device(device)
        self.with_induction = with_induction
        self.n_evaluations = 0

    @classmethod
    def load(cls, path, *, device: str = "auto", with_induction: bool = True) -> "Committee":
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        paths = member_checkpoints(path)
        models = [load_film_model(p, device=device)[0] for p in paths]
        for m in models:
            for p in m.parameters():
                p.requires_grad_(False)
        return cls(models, paths, device=device, with_induction=with_induction)

    @property
    def n_members(self) -> int:
        return len(self.models)

    def describe(self) -> dict:
        return {"n_members": self.n_members, "device": str(self.device),
                "with_induction": self.with_induction,
                "checkpoints": [str(p.resolve()) for p in self.paths]}

    def _member(self, model, positions: torch.Tensor, top: Topology):
        r = positions.shape[0]
        x = positions.detach().reshape(-1, 3).clone().requires_grad_(True)
        out = model(top.batch(x, r), with_induction=self.with_induction)
        (grad,) = torch.autograd.grad(out.energy.sum(), x)
        return out.energy.detach(), (-grad).reshape(positions.shape)

    def evaluate(self, positions: torch.Tensor, top: Topology) -> Evaluation:
        """Energies and forces of every member on every replica.

        A replica that makes a member raise (the induction solve failing on a pathological
        geometry, say) does not take the batch down: the batch is retried one replica at a
        time and the offending replicas come back with NaN and ``failed=True``, which the
        dynamics treats like any other blow-up.
        """
        positions = positions.to(self.device, torch.get_default_dtype())
        r = positions.shape[0]
        energies, forces = [], []
        failed = torch.zeros(r, dtype=torch.bool, device=self.device)
        for model in self.models:
            try:
                e, f = self._member(model, positions, top)
            except Exception:
                e = torch.full((r,), float("nan"), device=self.device)
                f = torch.full_like(positions, float("nan"))
                for i in range(r):
                    try:
                        ei, fi = self._member(model, positions[i:i + 1], top)
                        e[i], f[i] = ei[0], fi[0]
                    except Exception:
                        failed[i] = True
            energies.append(e)
            forces.append(f)
        self.n_evaluations += r * self.n_members
        energy = torch.stack(energies)
        force = torch.stack(forces)
        failed |= ~torch.isfinite(energy).all(0)
        failed |= ~torch.isfinite(force).reshape(self.n_members, r, -1).all(-1).all(0)
        return Evaluation(energy, force, failed)
