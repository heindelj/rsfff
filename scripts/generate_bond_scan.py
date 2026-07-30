"""Diatomic bond-scan reference data (dissociation curves).

A diatomic has a single internal degree of freedom -- the bond length ``r`` -- so a deterministic
scan over ``r`` (repulsive wall -> well -> past the MLIP cutoff) is the *complete* internal
configuration space, unlike the near-equilibrium Wigner sampling of ``scripts/generate_dataset.py``.
This one dataset feeds all three consumers: the short-range MLIP (energy/forces), the parameter
function (dipole/quadrupole/polarizability along the bond), and the long-range diabat switching
(the reference curve each competing cover is validated against).

For each species this script walks an adaptive ``r``-grid, builds the collinear two-atom geometry
(heteronuclear puts the heavy atom at index 0 to match the library's [O, H] ordering), labels the
full property set with the *same* psi4 b3lyp/def2-svpd engines as the Wigner pipeline (reusing
``compute_reference_data``), and streams ``data/labels/<key>.extxyz`` whose ``config_type`` is the
explicit diabatic key (e.g. ``o2_q0_m3``) and whose header adds a ``bond_length`` tag.

**Closed-shell singlets** (H2, OH-, singlet O2) are run **broken-symmetry UKS** (`reference=uhf`
with `guess_mix`) so the energy can dissociate toward the sum of atoms rather than a spurious
ionic asymptote. Every point is independent -- scratch is cleaned and a fresh HOMO/LUMO-mixed SAD
guess is used. An earlier version followed one BS solution *inward* with orbital reuse
(`guess=read`, no clean between points), which gave lower, smoother dissociation curves; but a
single failed SCF in that no-clean chain corrupted psi4's DIIS state and wedged the process in an
infinite loop. Cleaning every point makes each SCF bounded by `maxiter` and the run wedge-proof.
The trade-offs: `guess_mix` can relapse to the closed shell at intermediate `r` (a mild spurious
hump), and the hardest dissociated-tail points may not converge and are skipped -- their limit is
anchored by the atomic reference states regardless. Open-shell states (triplet H2/O2, OH,
superoxide) are ordinary UKS.

Usage:
    python scripts/generate_bond_scan.py [key ...]      # default: all diatomic scans
"""

import os
import sys
import tempfile
import time

import numpy as np

# Isolate this run's psi4 scratch BEFORE importing psi4: two generators sharing the default
# scratch dir corrupt each other's SCF state (stale orbital/DIIS files -> spurious
# non-convergence). A per-process scratch makes concurrent runs safe. Overridable via PSI_SCRATCH.
os.environ.setdefault(
    "PSI_SCRATCH", tempfile.mkdtemp(prefix=f"psi4_bondscan_{os.getpid()}_")
)

import psi4  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from generate_dataset import (  # noqa: E402  (reuse the validated QC labeling)
    BASIS,
    METHOD,
    build_geometry,
    compute_reference_data,
    fmt,
)

REPO = os.path.dirname(HERE)
OUT_DIR = os.path.join(REPO, "data", "labels")

# key -> (elements, charge, multiplicity, broken_symmetry, r_min). key doubles as the config_type
# and the diabatic-library entry (see data/diabatic_states.yaml). Heavy atom first. r_min is the
# compressed end of the scan (Angstrom), set per bond so the grid starts on the repulsive wall
# rather than absurdly inside it (O-O eq ~1.2, O-H ~0.97, H-H ~0.74).
SCANS = {
    "h2_q0_m1":  (["H", "H"], 0, 1, True,  0.40),   # singlet H2 (BS-UKS)
    "h2_q0_m3":  (["H", "H"], 0, 3, False, 0.40),   # triplet H2 (repulsive)
    "oh_q0_m2":  (["O", "H"], 0, 2, False, 0.60),   # OH radical
    "oh_q-1_m1": (["O", "H"], -1, 1, True,  0.60),  # hydroxide (BS-UKS)
    "o2_q0_m3":  (["O", "O"], 0, 3, False, 0.80),   # O2 ground triplet
    "o2_q0_m1":  (["O", "O"], 0, 1, True,  0.80),   # singlet oxygen (BS-UKS)
    "o2_q-1_m2": (["O", "O"], -1, 2, False, 0.80),  # superoxide
}


def bond_grid(r_min=0.50):
    """Adaptive r-grid (Angstrom): fine through the well, coarser toward dissociation.

    Starts at ``r_min`` (the repulsive wall for this bond) and ends past the 5 A MLIP cutoff so the
    free-atom limit is captured in-data.
    """
    near = np.arange(r_min, 2.50, 0.05)         # repulsive wall + well
    far = np.arange(2.50, 6.001, 0.15)          # dissociation tail (beyond cutoff)
    return np.concatenate([near, far])


_progress_fh = None


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _progress_fh is not None:
        _progress_fh.write(line + "\n")
        _progress_fh.flush()
        os.fsync(_progress_fh.fileno())


def write_frame(fh, symbols, coords, data, meta):
    """One extended-XYZ frame: per-atom `species pos forces`; labels on the header line."""
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
        f'config_type={meta["key"]} sample_index={meta["index"]} '
        f'bond_length={meta["bond_length"]:.6f} units=atomic'
    )
    fh.write(f"{natoms}\n{header}\n")
    for sym, (x, y, z), (fx, fy, fz) in zip(symbols, coords, forces):
        fh.write(
            f"{sym:<3} {x:18.10f} {y:18.10f} {z:18.10f} "
            f"{fx:20.12e} {fy:20.12e} {fz:20.12e}\n"
        )


def _label_point(symbols, charge, mult, r, reference):
    """Build the collinear geometry at bond length ``r`` and label it. Returns the data dict."""
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, float(r)]])
    build_geometry(symbols, coords, charge, mult=mult, fix_frame=True)
    data = compute_reference_data(
        len(symbols), method=METHOD, basis=BASIS, reference=reference
    )
    return coords, data


def scan(key):
    global _progress_fh
    if key not in SCANS:
        print(f"[{key}] SKIP: not a known scan (have: {sorted(SCANS)})", flush=True)
        return
    symbols, charge, mult, bs, r_min = SCANS[key]
    reference = "uhf"                             # UKS for KS-DFT; broken-symmetry for the singlets
    grid = bond_grid(r_min)

    progress_path = os.path.join(OUT_DIR, f"{key}.progress.log")
    _progress_fh = open(progress_path, "w", buffering=1)
    out_path = os.path.join(OUT_DIR, f"{key}.extxyz")
    t_start = time.time()
    n_ok = 0
    try:
        log(f"=== {key}  ({'-'.join(symbols)} q={charge:+d} m={mult}, "
            f"{'BS-UKS' if bs else 'UKS'}, {len(grid)} points) ===")
        psi4.set_output_file(os.path.join(OUT_DIR, f"{key}_psi4.out"), False)
        # Robust, independent per point: clean scratch, then a fresh SAD guess (HOMO/LUMO-mixed for
        # the broken-symmetry singlets). No orbital reuse across points -- reuse (guess=read with
        # no clean) gave lower-energy dissociation curves but, once a stretched-tail SCF failed,
        # left psi4's DIIS state corrupt and the process wedged in an infinite loop. Cleaning every
        # point makes each SCF bounded by maxiter and the whole run wedge-proof; the price is that
        # guess_mix may relapse to the closed shell at intermediate r (a mild spurious hump) and
        # some hard dissociated-tail points simply fail to converge and are skipped (their limit is
        # anchored by the atomic reference states anyway).
        with open(out_path, "w", buffering=1) as fh:
            for idx, r in enumerate(grid):
                r = float(r)
                psi4.core.clean()
                psi4.set_options({
                    "basis": BASIS, "reference": reference,
                    "e_convergence": 1e-8, "d_convergence": 1e-8, "maxiter": 500,
                    "guess": "sad", "guess_mix": bool(bs),
                })
                try:
                    coords, data = _label_point(symbols, charge, mult, r, reference)
                except Exception as exc:  # noqa: BLE001
                    log(f"[{key}] r={r:.3f}: FAILED ({exc})")
                    continue
                meta = {"key": key, "index": idx, "charge": charge, "mult": mult,
                        "bond_length": r}
                write_frame(fh, symbols, coords, data, meta)
                fh.flush()
                os.fsync(fh.fileno())
                n_ok += 1
                if (idx + 1) % 8 == 0 or (idx + 1) == len(grid):
                    rate = (time.time() - t_start) / (idx + 1)
                    log(f"[{key}]   {idx + 1}/{len(grid)} r={r:.3f} "
                        f"E={data['energy']:.6f} ({rate:.1f}s/pt)")
        log(f"[{key}] DONE: {n_ok}/{len(grid)} points -> {out_path} "
            f"({(time.time() - t_start)/60:.1f} min)")
    finally:
        _progress_fh.close()
        _progress_fh = None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    psi4.set_memory("8 GB")
    psi4.set_num_threads(os.cpu_count() or 1)
    args = sys.argv[1:]
    keys = args if args else list(SCANS)
    for key in keys:
        scan(key)


if __name__ == "__main__":
    main()
