"""i-PI driver for the complete ion model: ``i-pi-py_driver -m custom -P <rsfff>/src/md/ipi_ions.py -o ...``.

Parameters (``-o`` key=value, comma-separated): ``template`` (an ASE-readable file with the atoms in i-PI's
order, e.g. the run's init.xyz), ``model`` (model.pt, its run directory or a tarball holding it), and optionally
``params`` (the ion table json; default: the one the run names, or the one in the tarball), ``device`` (cpu,
cuda), ``cutoff`` and ``switch_width`` (Angstrom), ``ewald_tol``, ``dispersion_tail`` (true / false),
``table_cache`` (a directory for the radial table) and ``periodic`` (default true: i-PI always sends a cell; set
false for a cluster in a large box -- the cell is then ignored).

Everything is evaluated in one model call per request: in batched mode (i-PI's batched sockets, a ring polymer's
beads) the structures form one batch. Units: i-PI's atomic units in and out (bohr, Hartree), the virial is
-dE/d(strain) in Hartree. rsfff (``src`` as package ``rsfff``) and bb must be importable by the driver's python.
"""
from __future__ import annotations

import json

import numpy as np
from ase.io import read
from ipi.pes.dummy import Dummy_driver

from rsfff.md.ions import CompletePotential

__DRIVER_NAME__ = "rsfff_ions"
__DRIVER_CLASS__ = "RsfffIonsDriver"

BOHR_ANG = 0.529177210903


def _bool(v):
    return v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "yes", "on")


class RsfffIonsDriver(Dummy_driver):
    def __init__(self, template, model, params=None, device="cpu", cutoff=9.0, switch_width=1.0, ewald_tol=1e-10,
                 dispersion_tail=True, table_cache=None, periodic=True, *args, **kwargs):
        self.template, self.model_path, self.params = template, model, params
        self.device, self.table_cache = str(device), table_cache
        self.settings = dict(cutoff=float(cutoff), switch_width=float(switch_width), ewald_tol=float(ewald_tol),
                             dispersion_tail=_bool(dispersion_tail))
        self.periodic = _bool(periodic)
        super().__init__(*args, **kwargs)

    def check_parameters(self):
        self.numbers = read(self.template).numbers
        self.potential = CompletePotential.from_checkpoint(self.model_path, self.params, self.device, self.table_cache,
                                                           **self.settings)

    def _evaluate(self, cells, positions):
        pos = [np.asarray(p, float).reshape(-1, 3) * BOHR_ANG for p in positions]
        cl = [np.asarray(c, float).T * BOHR_ANG for c in cells] if self.periodic else None     # i-PI: columns
        E, F, vir = self.potential.evaluate([self.numbers] * len(pos), pos, cl, stress=self.periodic)
        extras = json.dumps({"cg_iters": int(self.potential.last.get("cg_iters", 0))})
        out = []
        for b in range(len(pos)):
            v = vir[b] if vir is not None else np.zeros((3, 3))
            out.append((float(E[b]), F[b] * BOHR_ANG, v, extras))           # Hartree/A -> Hartree/bohr
        return out

    def compute_structure(self, cell, pos):
        return self._evaluate([cell], [pos])[0]

    def compute(self, cell, pos):
        if isinstance(cell, list):
            return self._evaluate(cell, pos)
        return self.compute_structure(cell, pos)
