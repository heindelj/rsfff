"""Reference-data generation via pyscf / gpu4pyscf.

A folder-staged, notebook-drivable pipeline that takes raw ``.xyz`` geometries
through geometry optimization, harmonic frequencies, finite-temperature Wigner
sampling, and per-structure property labeling, emitting extended-XYZ data whose
schema matches ``data/labels/*.extxyz``.

The compute core (:mod:`rsfff.qcgen.backend`, :mod:`rsfff.qcgen.compute`) prefers
``gpu4pyscf`` when CUDA is available and transparently falls back to plain
``pyscf`` on CPU, so the same pipeline runs on a Colab GPU or a local laptop.
"""

from . import backend, compute, extxyz, pipeline, wigner

__all__ = ["backend", "compute", "extxyz", "pipeline", "wigner"]
