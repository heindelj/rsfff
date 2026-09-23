"""Between easyAL frames and the cc_workers store: specs in, training frames out.

In: a selected frame (``labels = "force"`` or ``"force,eda"``) becomes a ``force`` spec and,
when asked, an ``eda2`` spec at wB97M-V/def2-TZVPD -- the level every existing rsfff water
label is at. Both are the same geometry, so they share a ``geometry_id``; the store keeps
whatever tags a request was made under (the AL lineage) on the job.

Out: :func:`training_frame` joins a ``force`` result and, optionally, the ``eda2`` result of
the same geometry into one frame of the schema ``scripts/parse_roundtrip.py`` wrote and
``rsfff.train.data`` reads::

    arrays  species pos mulliken_charges fragment_idx forces           (forces Ha/bohr)
    info    energy [eda_*  fragment_energies fragment_dipoles fragment_second_moments]
            dipole quadrupole octopole hexadecapole (full tensors, e*a0^n) multipole_format=tensor
            n_fragments charge multiplicity fragment_charges fragment_multiplicities
            method basis config_type sample_id source units=atomic

EDA terms go kJ/mol -> Hartree, multipoles Debye*A^(n-1) -> e*a0^n, with the same CODATA
constants as ``qcgen``. Coordinates, forces and multipoles stay in **Q-Chem's standard
orientation** (multipoles about its origin), exactly as before: the two jobs of one geometry
come back in the same orientation, which is checked, and the link to the sampled structure is
the geometry id and the tags, not the coordinates. A force-only frame carries no ``eda_*`` and
no ``fragment_*`` labels (the loader wants those all-or-nothing per file), but it does carry
the water fragmentation, which the model needs as input.
"""

from __future__ import annotations

import numpy as np

from cc_workers.common.chem.molecule import Molecule
from cc_workers.common.chem.units import KJMOL_PER_HARTREE
from cc_workers.workers.qchem.parse import expand_multipole, multipole_to_atomic_units
from cc_workers.workers.qchem.specify import WB97MV_DEF2TZVPD, eda2, force

__all__ = ["THEORY", "molecule_of", "specs_for", "training_frame", "config_type",
           "GEOM_ATOL", "ENERGY_ATOL", "EDA_KEY_ORDER"]

THEORY = WB97MV_DEF2TZVPD
MULTIPOLES = ("dipole", "quadrupole", "octopole", "hexadecapole")
EDA_KEY_ORDER = ("cls_elec", "mod_pauli", "disp", "pol", "ct", "prp", "frz", "int", "elec",
                 "pauli", "cls_pauli", "cls_disp")
GEOM_ATOL = 1e-8      # Angstrom: eda and force jobs of one geometry must share an orientation
ENERGY_ATOL = 1e-6    # Hartree: two independent SCFs of the same wavefunction

#: tags copied from a selected frame's info onto its jobs
LINEAGE = ("traj_id", "time_fs", "phase", "n_waters", "select_rank", "sigma_energy",
           "sigma_forces", "source", "fs_to_failure")


def molecule_of(frame: dict) -> Molecule:
    info, arr = frame["info"], frame["arrays"]
    n_frag = int(info["n_fragments"])
    return Molecule(
        symbols=tuple(arr["species"]),
        coords=tuple(tuple(float(v) for v in row) for row in arr["pos"]),
        charge=int(info.get("charge", 0)), multiplicity=int(info.get("multiplicity", 1)),
        fragment_idx=tuple(int(v) for v in arr["fragment_idx"]),
        fragment_charges=tuple(int(v) for v in np.atleast_1d(info.get("fragment_charges", [0] * n_frag))),
        fragment_multiplicities=tuple(int(v) for v in np.atleast_1d(
            info.get("fragment_multiplicities", [1] * n_frag))))


def specs_for(frame: dict, *, tags: dict | None = None, theory=THEORY) -> dict:
    """``{"force": spec}`` plus ``"eda2"`` when the frame's ``labels`` ask for it."""
    info = frame["info"]
    mol = molecule_of(frame)
    lineage = {k: info[k] for k in LINEAGE if k in info}
    lineage["fragment_idx"] = list(mol.fragment_idx)
    lineage.update(tags or {})
    out = {"force": force(mol, theory, tags=dict(lineage))}
    if "eda" in str(info.get("labels", "force")).split(","):
        out["eda2"] = eda2(mol, theory, tags=dict(lineage))
    return out


def config_type(symbols, fragment_idx, n_fragments: int) -> str:
    """``w<n>`` for n waters, else the per-fragment formulas (as parse_roundtrip)."""
    formulas = []
    for f in range(n_fragments):
        syms = sorted(s for s, i in zip(symbols, fragment_idx) if i == f)
        counts = {s: syms.count(s) for s in dict.fromkeys(syms)}
        formulas.append("".join(s + (str(c) if c > 1 else "") for s, c in counts.items()))
    return f"w{n_fragments}" if all(f == "H2O" for f in formulas) else "_".join(formulas)


def _tensor(values, name) -> list[float]:
    full = expand_multipole(np.asarray(values, float), name)
    return multipole_to_atomic_units(full, name).ravel().tolist()


def training_frame(force_res: dict, eda_res: dict | None = None, *, spec_record: dict,
                   extra_info: dict | None = None) -> dict:
    """One rsfff training frame from loaded result envelopes (``Job.load_result()``).

    ``spec_record`` is the force job's ``spec.json`` (charge, multiplicity, tags with the
    fragmentation). Raises ``ValueError`` when the two jobs disagree on atoms, orientation or
    energy, or when an EDA lacks its isolated-fragment blocks.
    """
    fa = force_res["arrays"]
    tags = spec_record.get("tags", {})
    mol = spec_record["spec"]["molecule"]
    symbols = list(force_res["data"]["symbols"])
    positions = np.asarray(fa["positions"], float)
    order = force_res["frame"].get("atom_order", list(range(len(symbols))))
    if list(order) != list(range(len(symbols))):
        raise ValueError("force job reordered atoms; not supported for training export")

    if eda_res is not None:
        ea = eda_res["arrays"]
        if list(eda_res["data"]["symbols"]) != symbols:
            raise ValueError("eda and force jobs disagree on the element list")
        delta = float(np.abs(np.asarray(ea["positions"]) - positions).max())
        if delta > GEOM_ATOL:
            raise ValueError(f"eda and force orientations differ by {delta:.3g} A")
        de = abs(float(eda_res["data"]["energy"]) - float(force_res["data"]["energy"]))
        if de > ENERGY_ATOL:
            raise ValueError(f"eda and force energies differ by {de:.3g} Ha")
        if not eda_res["data"].get("has_fragment_blocks"):
            raise ValueError("EDA has no isolated-fragment blocks (needs SCF_PRINT_FRGM)")
        fragment_idx = [int(v) for v in ea["fragment_idx"]]
        n_frag = int(eda_res["data"]["n_fragments"])
        source, moments, mulliken = eda_res, ea, ea["mulliken_charges"]
    else:
        given = tags.get("fragment_idx") or mol.get("fragment_idx")
        if not given:
            raise ValueError("force-only frame needs the fragmentation (tags['fragment_idx'])")
        fragment_idx = [int(v) for v in given]
        n_frag = max(fragment_idx) + 1
        source, moments, mulliken = force_res, fa, fa["mulliken_charges"]

    frag_charges = tags.get("fragment_charges") or [0] * n_frag
    frag_mults = tags.get("fragment_multiplicities") or [1] * n_frag
    info = {"energy": float(source["data"]["energy"])}
    if eda_res is not None:
        eda = eda_res["data"]["eda"]
        for name in EDA_KEY_ORDER:
            if name in eda:
                info[f"eda_{name}"] = float(eda[name]) / KJMOL_PER_HARTREE
        info["fragment_energies"] = [float(v) for v in ea["fragment_energies"]]
        info["fragment_dipoles"] = multipole_to_atomic_units(
            np.asarray(ea["fragment_dipole"], float), "dipole").ravel().tolist()
        info["fragment_second_moments"] = multipole_to_atomic_units(
            np.asarray(ea["fragment_quadrupole"], float), "quadrupole").ravel().tolist()
    for name in MULTIPOLES:
        if name in moments:
            info[name] = _tensor(moments[name], name)
    info.update(
        multipole_format="tensor", n_fragments=n_frag, charge=int(mol.get("charge", 0)),
        multiplicity=int(mol.get("multiplicity", 1)), fragment_charges=list(frag_charges),
        fragment_multiplicities=list(frag_mults),
        method=spec_record["spec"]["rem"]["METHOD"], basis=spec_record["spec"]["rem"]["BASIS"],
        config_type=config_type(symbols, fragment_idx, n_frag),
        sample_id=str(tags.get("traj_id", "")) + (f"@{tags['time_fs']}" if "time_fs" in tags else ""),
        source=f"store:{force_res['id']}" + (f"+{eda_res['id']}" if eda_res else ""),
        units="atomic", geometry_id=force_res["geometry_id"],
    )
    info.update(extra_info or {})
    arrays = {"species": symbols, "pos": positions,
              "mulliken_charges": np.asarray(mulliken, float),
              "fragment_idx": np.asarray(fragment_idx, int),
              "forces": np.asarray(fa["forces"], float)}
    return {"info": info, "arrays": arrays}
