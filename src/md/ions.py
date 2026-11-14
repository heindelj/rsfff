"""Dynamics on a cmm model (:mod:`rsfff.ff.cmm`): clusters (periodic cells once the model supports them).

:class:`CmmPotential` evaluates a batch of structures in one call of the batched precompute and the model --
energies, forces and the virial -dE/d(strain) from one backward pass through positions and cell (the strain is
applied to both, so the virial carries the Ewald reciprocal part and the dispersion tail). Periodic structures
use the PBC path of :meth:`rsfff.ff.cmm.CmmModel.precompute` (pair list with images within ``cutoff``,
short-range complements switched off over the last ``switch_width``, Ewald for the bare Coulomb part, the
matrix-free CG solve of the joint induction + charge-flux minimum warm-started from the previous call). :class:`CmmCalculator` is the ASE face
of it; ``rsfff/md/ipi_ions.py`` the i-PI driver.

Energies are interaction energies relative to the isolated ions (the model's ``e_int``), Hartree internally.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from ..ff.cmm import load_cmm

__all__ = ["CmmCalculator", "CmmPotential", "load_model"]

HARTREE_TO_EV = 27.211386245988


def load_model(checkpoint, params=None, device="cpu", table_cache=None):
    """The cmm model of a checkpoint, a legacy complete-model checkpoint or a run tarball
    (:func:`rsfff.ff.cmm.load_cmm`) on ``device``, its radial table built (kept on disk in ``table_cache``)."""
    import os
    if table_cache:
        os.environ.setdefault("RSFFF_TABLE_CACHE", str(table_cache))
    model = load_cmm(checkpoint, params=params)
    model.table()
    model.tab = model.tab.to(device)
    return model.to(device)


class CmmPotential:
    """Energies, forces and virials of a batch of structures (same or different atoms, clusters or periodic).

    ``cutoff`` / ``switch_width`` (Angstrom; ``cutoff`` None: the model's short-range reach plus the switch width),
    ``ewald_tol`` and ``dispersion_tail`` are the periodic path's settings. Periodic batches warm-start the solve from the
    previous call's moments when the atoms are the same (MD steps, or the same beads)."""

    def __init__(self, model, *, cutoff=None, switch_width=1.0, ewald_tol=1e-10, dispersion_tail=True, warm_start=True):
        self.model = model
        self.cutoff = None if cutoff is None else float(cutoff)
        self.switch_width, self.ewald_tol = float(switch_width), float(ewald_tol)
        self.dispersion_tail, self.warm_start = bool(dispersion_tail), bool(warm_start)
        self.device = next(model.parameters()).device
        self._x = None
        self._key = None
        self.last = {}

    @classmethod
    def from_checkpoint(cls, checkpoint, params=None, device="cpu", table_cache=None, **kw):
        return cls(load_model(checkpoint, params, device, table_cache), **kw)

    def evaluate(self, numbers, positions, cells=None, stress=True):
        """``numbers`` a list of (n_b,) atomic numbers, ``positions`` a list of (n_b, 3) Angstrom, ``cells`` None
        (clusters) or a list of (3, 3) Angstrom, rows the lattice vectors. Returns energies (B,) Hartree, forces
        (list of (n_b, 3) Hartree/Angstrom) and, with ``stress`` and cells, virials (B, 3, 3) Hartree = -dE/d eps,
        symmetrized."""
        dev, f64 = self.device, torch.float64
        B = len(positions)
        Z = torch.cat([torch.as_tensor(np.asarray(z), dtype=torch.long) for z in numbers]).to(dev)
        fr = torch.cat([torch.full((len(z),), b, dtype=torch.long) for b, z in enumerate(numbers)]).to(dev)
        pos = torch.as_tensor(np.concatenate([np.asarray(p, float) for p in positions]), dtype=f64, device=dev)
        pos = pos.requires_grad_(True)
        periodic = cells is not None
        leaves = [pos]
        kw = {}
        if periodic:
            eps = torch.zeros(B, 3, 3, dtype=f64, device=dev, requires_grad=stress)
            F = torch.eye(3, dtype=f64, device=dev) + eps
            cell = torch.as_tensor(np.stack([np.asarray(c, float) for c in cells]), dtype=f64, device=dev)
            p_in = torch.einsum("nk,nkl->nl", pos, F[fr])
            kw = dict(cell=torch.einsum("bkl,blm->bkm", cell, F), cutoff=self.cutoff, switch_width=self.switch_width,
                      ewald_tol=self.ewald_tol, dispersion_tail=self.dispersion_tail)
            if stress:
                leaves.append(eps)
        else:
            p_in = pos
        bt = self.model.precompute(p_in, Z, fr, **kw)
        key = (tuple(Z.tolist()), B)
        if periodic and self.warm_start and self._x is not None and key == self._key:
            bt["guess"] = self._x
        out = self.model(bt)
        E = out["e_int"]
        grads = torch.autograd.grad(E.sum(), leaves)
        if periodic:
            self._x, self._key = out.get("guess"), key
        self.last = {"cg_iters": out.get("cg_iters", 0), "terms": {k: out[k].detach().cpu().numpy() for k in
                     ("e_elst", "e_pauli", "e_disp", "e_pol", "e_ct")}}
        forces = (-grads[0]).detach().cpu().numpy()
        split = np.cumsum([len(z) for z in numbers])[:-1]
        vir = None
        if periodic and stress:
            g = grads[1].detach()
            vir = (-0.5 * (g + g.transpose(1, 2))).cpu().numpy()
        return E.detach().cpu().numpy(), np.split(forces, split), vir


class CmmCalculator(Calculator):
    """ASE calculator: eV, eV/Angstrom, eV/Angstrom^3. ``atoms.pbc`` all True is a periodic cell, all False a
    cluster (mixed is not supported)."""

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, potential: CmmPotential, **kwargs):
        super().__init__(**kwargs)
        self.potential = potential

    @classmethod
    def from_checkpoint(cls, checkpoint, params=None, device="cpu", table_cache=None, **kw):
        return cls(CmmPotential.from_checkpoint(checkpoint, params, device, table_cache, **kw))

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        pbc = np.asarray(self.atoms.pbc, bool)
        if pbc.any() and not pbc.all():
            raise NotImplementedError("CmmCalculator: mixed periodicity (pbc must be all True or all False)")
        periodic = bool(pbc.all())
        want_stress = "stress" in properties
        if want_stress and not periodic:
            raise NotImplementedError("stress needs a periodic cell")
        E, F, vir = self.potential.evaluate([self.atoms.numbers], [self.atoms.positions],
                                            [self.atoms.cell.array] if periodic else None, stress=want_stress)
        self.results["energy"] = self.results["free_energy"] = float(E[0]) * HARTREE_TO_EV
        self.results["forces"] = F[0] * HARTREE_TO_EV
        if want_stress:
            # ASE stress = (1/V) dE/d eps = -virial / V
            self.results["stress"] = full_3x3_to_voigt_6_stress(-vir[0] * HARTREE_TO_EV / self.atoms.get_volume())
