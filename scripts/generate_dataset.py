"""End-to-end reference-data pipeline.

For a given reference molecule in ``data/mols`` this script:

  1. Optimizes the geometry at b3lyp/def2-svpd (charge handled per species).
  2. Runs an analytical frequency calculation to obtain the Hessian.
  3. Draws ``N_SAMPLES`` geometries from the finite-temperature Wigner
     distribution at ``TEMPERATURE`` K (via ``sample_wigner``).
  4. Labels every sampled geometry with energy, forces, dipole, quadrupole,
     polarizability, and dipole derivatives (via the psi4 example routines).
  5. Writes the results to ``data/labels/<name>.extxyz`` in extended-XYZ
     format: per-atom columns hold ``species pos forces``; everything else
     (energy, dipole, quadrupole, polarizability, dipole derivatives) lives on
     the header/comment line.

All quantities are stored in atomic units (energy: Hartree, forces:
Hartree/Bohr, dipole: e*a0, quadrupole: e*a0^2, polarizability: a0^3,
dipole derivatives: e (d mu / d R), positions: Angstrom).

Usage:
    python scripts/generate_dataset.py [molecule_name ...]

With no arguments it processes only h2o. Pass one or more stems (e.g. ``co2
nh3``) to process specific molecules, or ``all`` for every species.
"""

import os
import sys
import time

import numpy as np
import psi4

# Make the sibling example module importable regardless of CWD.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sample_wigner import sample_wigner  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO = os.path.dirname(HERE)
MOL_DIR = os.path.join(REPO, "data", "mols")
OUT_DIR = os.path.join(REPO, "data", "labels")

METHOD = "b3lyp"
BASIS = "def2-svpd"
N_SAMPLES = 500
TEMPERATURE = 500.0  # Kelvin

# Charge for each species (multiplicity is 1 for all -- every species has an
# even number of electrons, so the rhf reference is valid throughout).
CHARGES = {
    "h2o": 0,
    "ch4": 0,
    "co2": 0,
    "hf": 0,
    "nh3": 0,
    "h2so4": 0,
    "h3o+": 1,
    "hso4-": -1,
    "oh-": -1,
    "so42-": -2,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_progress_fh = None  # per-molecule progress log handle (set in process())


def log(msg):
    """Emit a timestamped line to stdout and the progress file, flushed."""
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _progress_fh is not None:
        _progress_fh.write(line + "\n")
        _progress_fh.flush()
        os.fsync(_progress_fh.fileno())


def read_xyz(path):
    """Return (symbols, coords[Angstrom]) from a plain .xyz file."""
    with open(path) as fh:
        lines = fh.read().splitlines()
    natoms = int(lines[0].split()[0])
    symbols, coords = [], []
    for line in lines[2 : 2 + natoms]:
        tok = line.split()
        symbols.append(tok[0])
        coords.append([float(tok[1]), float(tok[2]), float(tok[3])])
    return symbols, np.array(coords)


def charge_for(name):
    """Look up the formal charge, falling back to parsing the file stem."""
    if name in CHARGES:
        return CHARGES[name]
    # Fallback: trailing charge encoded in the stem, e.g. 'so42-' -> -2.
    stem = name
    sign = 0
    if stem.endswith("+"):
        sign = +1
        stem = stem[:-1]
    elif stem.endswith("-"):
        sign = -1
        stem = stem[:-1]
    else:
        return 0
    mag = ""
    while stem and stem[-1].isdigit():
        mag = stem[-1] + mag
        stem = stem[:-1]
    return sign * (int(mag) if mag else 1)


def build_geometry(symbols, coords, charge, mult=1, fix_frame=False):
    """Construct a Psi4 Molecule from symbols/coords with charge & mult.

    When ``fix_frame`` is True the input orientation is preserved (no
    center-of-mass shift, no reorientation) so that the returned forces share
    the same Cartesian frame as the coordinates we save.
    """
    lines = [f"{charge} {mult}"]
    for sym, (x, y, z) in zip(symbols, coords):
        lines.append(f"{sym} {x:.10f} {y:.10f} {z:.10f}")
    lines.append("units angstrom")
    lines.append("symmetry c1")
    if fix_frame:
        lines.append("no_com")
        lines.append("no_reorient")
    return psi4.geometry("\n".join(lines))


def fmt(arr):
    """Flatten a numeric array to a space-separated string."""
    return " ".join(f"{v:.10e}" for v in np.asarray(arr).ravel())


def write_frame(fh, symbols, coords, data, meta):
    """Append one extended-XYZ frame.

    Per-atom columns: species pos(3) forces(3).
    Header key=value pairs carry all remaining labels.
    """
    natoms = len(symbols)
    forces = np.asarray(data["forces"]).reshape(natoms, 3)

    header = (
        'Properties=species:S:1:pos:R:3:forces:R:3 '
        f'energy={data["energy"]:.12e} '
        f'dipole="{fmt(data["dipole"])}" '
        f'quadrupole="{fmt(data["quadrupole"])}" '
        f'polarizability="{fmt(data["polarizability"])}" '
        f'dipole_derivatives="{fmt(data["dipole_derivatives"])}" '
        f'charge={meta["charge"]} multiplicity={meta["mult"]} '
        f'method={METHOD} basis={BASIS} '
        f'config_type={meta["name"]} sample_index={meta["index"]} '
        f'temperature={TEMPERATURE} units=atomic'
    )

    fh.write(f"{natoms}\n{header}\n")
    for sym, (x, y, z), (fx, fy, fz) in zip(symbols, coords, forces):
        fh.write(
            f"{sym:<3} {x:18.10f} {y:18.10f} {z:18.10f} "
            f"{fx:20.12e} {fy:20.12e} {fz:20.12e}\n"
        )


# ---------------------------------------------------------------------------
# Reference-data labeling (validated against Psi4 1.11 analytical engines)
# ---------------------------------------------------------------------------
def compute_reference_data(natoms, method=METHOD, basis=BASIS, field_step=1e-3):
    """Label the currently active molecule with all requested properties.

    Returns energy (Hartree), forces (Hartree/Bohr), dipole (e*a0), quadrupole
    (e*a0^2), polarizability (a0^3), and dipole derivatives (e), all in atomic
    units. The active molecule must already carry the correct charge and be
    fixed in a definite Cartesian frame (no_com/no_reorient) so that forces and
    coordinates share an orientation.

    Design notes (Psi4 1.11):
      * ``properties(ref_wfn=...)`` cannot seed an SCF, so orbital reuse across
        property calls is unavailable; each stage runs its own SCF.
      * Static polarizability is taken from the analytical CPHF response
        (``DIPOLE_POLARIZABILITIES``) -- exact and correctly signed.
      * Dipole derivatives use the interchange theorem, d(mu_i)/d(R_a) =
        d(F_a)/d(E_i), via central finite differences of forces under a uniform
        field applied with ``perturb_dipole``. Psi4's perturbation convention
        flips the response sign, so the result is negated to match the
        analytical atomic polar tensor (verified to ~1e-6).
    """
    psi4.set_options(
        {
            "basis": basis,
            "reference": "rhf",
            "e_convergence": 1e-9,
            "d_convergence": 1e-9,
            "perturb_h": False,
            "perturb_dipole": [0.0, 0.0, 0.0],
        }
    )

    # 1. Multipoles + analytical polarizability (one SCF + CPHF).
    psi4.properties(
        method,
        properties=["DIPOLE", "QUADRUPOLE", "DIPOLE_POLARIZABILITIES"],
    )
    dipole = np.array(psi4.variable("SCF DIPOLE"))
    try:
        quadrupole = np.array(psi4.variable(f"{method.upper()} QUADRUPOLE"))
    except KeyError:
        quadrupole = np.array(psi4.variable("CURRENT QUADRUPOLE"))
    polarizability = np.array(
        [
            [psi4.variable(f"DIPOLE POLARIZABILITY {a}{b}") for b in "XYZ"]
            for a in "XYZ"
        ]
    )

    # 2. Energy and forces (one SCF + gradient).
    grad, wfn = psi4.gradient(method, return_wfn=True)
    forces = -np.array(grad)
    energy = wfn.energy()

    # 3. Dipole derivatives via finite field on the forces (6 SCF+gradient).
    dF_dfield = np.zeros((natoms, 3, 3))  # [atom, R_coord, field_coord]
    for axis in range(3):
        forces_pm = []
        for sign in (+1.0, -1.0):
            field = [0.0, 0.0, 0.0]
            field[axis] = sign * field_step
            psi4.set_options(
                {"perturb_h": True, "perturb_with": "dipole",
                 "perturb_dipole": field}
            )
            g_f = psi4.gradient(method)
            forces_pm.append(-np.array(g_f))
        dF_dfield[:, :, axis] = (forces_pm[0] - forces_pm[1]) / (
            2.0 * field_step
        )
    psi4.set_options({"perturb_h": False, "perturb_dipole": [0.0, 0.0, 0.0]})

    # Negate to match the analytical d(mu_i)/d(R_a) sign; index [atom, dR, dmu].
    dipole_derivatives = -dF_dfield

    return {
        "energy": energy,
        "forces": forces,
        "dipole": dipole,
        "quadrupole": quadrupole,
        "polarizability": polarizability,
        "dipole_derivatives": dipole_derivatives,
    }


# ---------------------------------------------------------------------------
# Per-molecule driver
# ---------------------------------------------------------------------------
def process(name):
    global _progress_fh

    xyz_path = os.path.join(MOL_DIR, f"{name}.xyz")
    if not os.path.isfile(xyz_path):
        print(f"[{name}] SKIP: {xyz_path} not found", flush=True)
        return

    charge = charge_for(name)
    symbols, coords = read_xyz(xyz_path)

    # Open a flushed, human-monitorable progress log for this molecule.
    progress_path = os.path.join(OUT_DIR, f"{name}.progress.log")
    _progress_fh = open(progress_path, "w", buffering=1)
    t_start = time.time()
    try:
        log(f"=== {name}  (charge={charge:+d}, mult=1, {len(symbols)} atoms) ===")
        log(f"progress log: {progress_path}")

        psi4.core.clean()
        psi4.core.clean_options()
        psi4.set_output_file(os.path.join(OUT_DIR, f"{name}_psi4.out"), False)
        psi4.set_options(
            {
                "basis": BASIS,
                "reference": "rhf",
                "e_convergence": 1e-8,
                "d_convergence": 1e-8,
            }
        )

        # 1. Optimize the reference geometry.
        build_geometry(symbols, coords, charge)
        log(f"[{name}] optimizing at {METHOD}/{BASIS} ...")
        psi4.optimize(METHOD)

        # 2. Analytical frequencies -> Hessian for the Wigner sampler.
        log(f"[{name}] frequency calculation ...")
        _, wfn = psi4.frequency(METHOD, return_wfn=True)

        # 3. Sample geometries from the finite-T Wigner distribution.
        log(f"[{name}] sampling {N_SAMPLES} geometries at {TEMPERATURE} K ...")
        geometries = sample_wigner(
            wfn, n_samples=N_SAMPLES, temperature=TEMPERATURE
        )
        opt_symbols = [
            wfn.molecule().symbol(i) for i in range(wfn.molecule().natom())
        ]

        # 4. Label each geometry and stream it to the extxyz file.
        out_path = os.path.join(OUT_DIR, f"{name}.extxyz")
        log(f"[{name}] labeling -> {out_path}")
        n_ok = 0
        with open(out_path, "w", buffering=1) as fh:
            for idx, geom in enumerate(geometries):
                psi4.core.clean()
                build_geometry(opt_symbols, geom, charge, fix_frame=True)
                try:
                    data = compute_reference_data(
                        len(opt_symbols), method=METHOD, basis=BASIS
                    )
                except Exception as exc:  # noqa: BLE001
                    log(f"[{name}] sample {idx}: FAILED ({exc})")
                    continue
                meta = {
                    "name": name,
                    "index": idx,
                    "charge": charge,
                    "mult": 1,
                }
                write_frame(fh, opt_symbols, geom, data, meta)
                fh.flush()
                os.fsync(fh.fileno())
                n_ok += 1
                if (idx + 1) % 10 == 0 or (idx + 1) == N_SAMPLES:
                    rate = (time.time() - t_start) / (idx + 1)
                    eta = rate * (N_SAMPLES - (idx + 1))
                    log(
                        f"[{name}]   labeled {idx + 1}/{N_SAMPLES} "
                        f"({rate:.1f}s/frame, ETA {eta/60:.1f} min)"
                    )

        log(
            f"[{name}] DONE: {n_ok}/{N_SAMPLES} frames -> {out_path} "
            f"(total {(time.time() - t_start)/60:.1f} min)"
        )
    finally:
        _progress_fh.close()
        _progress_fh = None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    psi4.set_memory("8 GB")
    psi4.set_num_threads(os.cpu_count() or 1)

    args = sys.argv[1:]
    if not args:
        names = ["h2o"]
    elif len(args) == 1 and args[0] == "all":
        names = sorted(CHARGES)
    else:
        names = args

    for name in names:
        process(name)


if __name__ == "__main__":
    main()
