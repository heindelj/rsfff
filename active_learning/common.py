"""Shared access to the Q-Chem job pool and the round-trip parser for the AL stages.

``qchem_roundtrip/scripts/qchem_roundtrip.py`` is a standalone stdlib script -- it has to run
under Perlmutter's system ``python3`` inside a worker allocation -- so it is loaded by path
rather than imported, the same way ``job_runner/active_learning/common.py`` loads ``hbq_jobs``
in the HBQ project.

``scripts/parse_roundtrip.py`` is loaded the same way, and it expects ``qcgen`` as a top-level
package (it inserts ``<repo>/src`` on ``sys.path``). Importing that package for real is not an
option here: ``qcgen/__init__.py`` pulls in the pyscf compute backend, and the environment that
runs an active-learning loop has no reason to have pyscf in it. :func:`_install_qcgen` therefore
builds a stand-in ``qcgen`` package out of the three files the parsers actually need
(``qchem_out``, ``qchem_eda``, ``qchem_force`` -- numpy and each other, nothing else), from the
installed ``rsfff.qcgen`` when there is one and from the checkout otherwise. One copy of the
eda/force merge then serves both, with no pyscf and no duplicated parsing code.

Three roots, because a loop run on scratch is not inside the checkout:

``REPO_ROOT``       ``$RSFFF_REPO``, else the directory holding this package
``ROUNDTRIP_ROOT``  ``$RSFFF_QCHEM_ROOT``, else ``<repo>/qchem_roundtrip``
``DEFAULT_CONFIG``  ``<roundtrip>/config.json``

    from common import ROUNDTRIP_ROOT, qchem_roundtrip, roundtrip_parser
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types
from pathlib import Path

AL_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("RSFFF_REPO") or AL_ROOT.parent).resolve()
ROUNDTRIP_ROOT = Path(
    os.environ.get("RSFFF_QCHEM_ROOT") or (REPO_ROOT / "qchem_roundtrip")
).resolve()
DEFAULT_CONFIG = ROUNDTRIP_ROOT / "config.json"

__all__ = ["AL_ROOT", "REPO_ROOT", "ROUNDTRIP_ROOT", "DEFAULT_CONFIG", "qchem_roundtrip",
           "roundtrip_parser"]


def _load_by_path(name: str, path: Path):
    existing = sys.modules.get(name)
    if existing is not None and getattr(existing, "__file__", None):
        if Path(existing.__file__).resolve() == path.resolve():
            return existing
    if not path.exists():
        raise FileNotFoundError(
            f"{name}: {path} does not exist. Point $RSFFF_REPO at an rsfff checkout "
            f"(and $RSFFF_QCHEM_ROOT at the job-pool directory if it lives elsewhere)."
        )
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve their annotations through sys.modules
    spec.loader.exec_module(module)
    return module


#: The Q-Chem output parsers, in dependency order. They import each other relatively and
#: otherwise need only numpy, which is what makes the stand-in package below safe.
QCGEN_PARSERS = ("qchem_out", "qchem_eda", "qchem_force")


def _qcgen_dir() -> Path:
    """Where ``qcgen``'s source files are, without importing the package (and its pyscf)."""
    try:
        spec = importlib.util.find_spec("rsfff.qcgen")
    except (ImportError, ValueError):
        spec = None
    if spec is not None and spec.submodule_search_locations:
        return Path(list(spec.submodule_search_locations)[0])
    return REPO_ROOT / "src" / "qcgen"


def _install_qcgen() -> None:
    """Register a ``qcgen`` package holding only the parsers, if one is not there already."""
    if "qcgen.qchem_eda" in sys.modules:
        return
    source = _qcgen_dir()
    package = sys.modules.get("qcgen")
    if package is None:
        package = types.ModuleType("qcgen")
        package.__path__ = [str(source)]  # so the submodules' relative imports resolve
        package.__doc__ = "Stand-in for rsfff.qcgen holding only the Q-Chem output parsers."
        sys.modules["qcgen"] = package
    for name in QCGEN_PARSERS:
        path = source / f"{name}.py"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found; install rsfff, or point $RSFFF_REPO at a checkout, so the "
                f"Q-Chem output parsers can be loaded"
            )
        spec = importlib.util.spec_from_file_location(f"qcgen.{name}", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"qcgen.{name}"] = module
        spec.loader.exec_module(module)
        setattr(package, name, module)


qchem_roundtrip = _load_by_path(
    "rsfff_qchem_roundtrip", ROUNDTRIP_ROOT / "scripts" / "qchem_roundtrip.py"
)

_parser = None


def roundtrip_parser():
    """``scripts/parse_roundtrip.py``, loaded on first use.

    Deferred because it pulls in numpy and the Q-Chem parsers, and building a loop (or reading
    ``--help``) should not need either.
    """
    global _parser
    if _parser is None:
        _install_qcgen()
        _parser = _load_by_path(
            "rsfff_parse_roundtrip", REPO_ROOT / "scripts" / "parse_roundtrip.py"
        )
    return _parser
