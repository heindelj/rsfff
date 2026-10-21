"""Backend selection: prefer ``gpu4pyscf`` (CUDA), fall back to plain ``pyscf``.

Every other module in :mod:`rsfff.qcgen` builds molecules and mean-field objects
through this layer so that the same code runs unchanged on a Colab GPU
(``gpu4pyscf``) or a local CPU (``pyscf``). Helpers here also normalize array
types: ``gpu4pyscf`` returns cupy arrays, so :func:`to_numpy` funnels everything
back to host numpy for downstream math and serialization.
"""

from __future__ import annotations

import numpy as np
import pyscf

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------
# gpu4pyscf is only importable where a CUDA toolkit + GPU are present. Importing
# it also fails loudly on machines without cupy, so we guard the whole thing.
try:  # pragma: no cover - exercised only on GPU hosts
    import gpu4pyscf  # noqa: F401
    from gpu4pyscf import dft as _gpu_dft

    HAVE_GPU = True
except Exception:  # noqa: BLE001 - any import failure means "no GPU backend"
    _gpu_dft = None
    HAVE_GPU = False


def backend_name() -> str:
    """Human-readable name of the active backend ("gpu4pyscf" or "pyscf")."""
    return "gpu4pyscf" if HAVE_GPU else "pyscf"


def as_backend_array(x):
    """Return ``x`` on the active backend's array type (cupy on GPU, else numpy).

    Used when injecting a host-computed operator (e.g. a finite-field dipole
    term) into a mean-field object whose matrices live on the GPU.
    """
    if HAVE_GPU:  # pragma: no cover - GPU-only path
        import cupy

        return cupy.asarray(x)
    return np.asarray(x)


def to_numpy(x):
    """Return ``x`` as a host numpy array, whatever backend produced it.

    ``gpu4pyscf`` returns cupy arrays; ``cupy.asnumpy`` moves them to the host.
    Plain numpy arrays (and scalars) pass through unchanged.
    """
    get = getattr(x, "get", None)  # cupy arrays expose .get()
    if get is not None and type(x).__module__.startswith("cupy"):
        return np.asarray(get())
    return np.asarray(x)


# ---------------------------------------------------------------------------
# Molecule / mean-field construction
# ---------------------------------------------------------------------------
#: elements for which the Karlsruhe def2 basis sets are defined with the def2-ECP (Rb and heavier): Q-Chem needs
#: ``ECP = def2-ecp`` for them, pyscf takes the same potentials by the basis name (``ecp="def2-tzvpd"``): the
#: Stuttgart/Koeln small-core sets, e.g. Rb ECP28MWB, Cs ECP46MWB, I ECP28MDF
ECP_FROM_Z = 37


def default_ecp(symbols, basis):
    """The ECP a def2 basis implies for these atoms (``None`` when no atom needs one or the basis is not def2)."""
    if not isinstance(basis, str) or not basis.lower().startswith("def2"):
        return None
    from pyscf.data.elements import charge as _z
    return basis if any(_z(s) >= ECP_FROM_Z for s in symbols) else None


def make_mol(symbols, coords_ang, charge=0, spin=0, basis="def2-svpd", verbose=0, ecp="auto"):
    """Build a :class:`pyscf.gto.Mole` in the *input* Cartesian frame.

    ``spin`` is pyscf's convention (Nalpha - Nbeta = 2S), so a doublet is
    ``spin=1``. Symmetry detection and reorientation are disabled: the molecule
    keeps the exact coordinates passed in, so gradients/forces evaluated later
    share the frame of the coordinates we serialize.

    Coordinates are in Angstrom. ``ecp="auto"`` assigns the def2-ECP to Rb and heavier atoms when the basis is a
    def2 set (:func:`default_ecp`); pass a name to force one or ``None`` for all-electron.
    """
    atom = [
        (sym, (float(x), float(y), float(z)))
        for sym, (x, y, z) in zip(symbols, np.asarray(coords_ang))
    ]
    if ecp == "auto":
        ecp = default_ecp(symbols, basis)
    kw = {} if ecp is None else {"ecp": ecp}
    mol = pyscf.M(
        atom=atom,
        unit="Angstrom",
        basis=basis,
        charge=int(charge),
        spin=int(spin),
        symmetry=False,
        verbose=verbose,
        **kw,
    )
    return mol


def make_mf(mol, xc, spin=0):
    """Return a DFT mean-field object for ``mol`` on the active backend.

    Restricted (RKS) for closed-shell (``spin == 0``), unrestricted (UKS)
    otherwise. The ``xc`` string is passed straight through, so non-local
    functionals such as ``wB97M-V`` carry their VV10 correlation automatically.
    """
    restricted = int(spin) == 0
    if HAVE_GPU:  # pragma: no cover - GPU-only path
        mf = _gpu_dft.RKS(mol, xc=xc) if restricted else _gpu_dft.UKS(mol, xc=xc)
    else:
        mf = mol.RKS(xc=xc) if restricted else mol.UKS(xc=xc)
    return mf
