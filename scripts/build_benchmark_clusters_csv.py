"""Merge the published water-cluster benchmark table with this repo's film-model results.

    python scripts/build_benchmark_clusters_csv.py

Writes ``data/kristina_clusters/benchmark_clusters.csv``, one row per cluster, holding the
literature numbers alongside the two film checkpoints so the notebook
``notebooks/benchmark_clusters.ipynb`` only ever has to read a single file.

The literature block below is transcribed by hand from the published comparison tables
(q-AQUA / MB-Pol / CMM / wB97X-V / HIPPO binding energies, and the RMSD of each method's
relaxed geometry to the reference minimum). The 25 rows there appear in the *same order* as
``data/kristina_clusters/all_clusters.xyz`` -- that is the only thing tying the two halves
together, so the merge asserts that the water counts agree row by row rather than trusting it.

Transcription check, from the footers the published tables carry:

    q-AQUA MAE/n 0.040, CMM 0.100, HIPPO 0.194, and mean RMSD 0.026 / 0.046 / 0.017 for
    q-AQUA / MB-Pol / wB97X-V

all reproduce exactly from the values below, which is what validates the reference column.
Two footers do *not* reproduce: MB-Pol MAE/n comes out 0.109 against a published 0.156, and
wB97X-V 0.057 against 0.051 (CMM's mean RMSD lands at 0.033 against a published 0.032). Either
a value in those two columns is mis-transcribed or the published footer was computed over
something slightly different; treat the MB-Pol *column* as good enough to plot and the MB-Pol
*aggregate* as unverified.

The film columns come straight out of ``scripts/optimize_clusters.py``: the binding energy at
the model's own relaxed geometry (the same quantity the other methods report) and the all-atom
Kabsch RMSD to the reference minimum. Re-run that script for both checkpoints before re-running
this one whenever either model is retrained.
"""

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "kristina_clusters"

# n_waters, isomer,
#   E (kcal/mol): q-AQUA, MB-Pol, CMM, wB97X-V, HIPPO, reference
#   RMSD (Angstrom): q-AQUA, MB-Pol, CMM, wB97X-V
# HIPPO is blank for the clusters the published table leaves blank; the RMSD table has no
# HIPPO or reference column at all (the reference *is* the geometry RMSD is measured against).
LITERATURE = [
    (2,  "",               -4.97,   -4.96,   -4.93,   -5.00,   -4.96,   -4.99,   0.005, 0.008, 0.011, 0.005),
    (3,  "",              -15.73,  -15.69,  -15.03,  -15.77,  -15.77,  -15.77,  0.010, 0.014, 0.018, 0.008),
    (4,  "",              -27.35,  -27.12,  -26.91,  -27.75,  -26.69,  -27.39,  0.008, 0.024, 0.010, 0.006),
    (5,  "",              -35.71,  -35.94,  -35.63,  -36.51,  -34.58,  -35.90,  0.013, 0.059, 0.025, 0.008),
    (6,  "Prism",         -46.21,  -45.87,  -45.83,  -46.53,  -46.15,  -46.20,  0.010, 0.035, 0.023, 0.009),
    (6,  "Cage",          -45.94,  -45.51,  -45.29,  -46.30,  -45.39,  -45.90,  0.013, 0.027, 0.027, 0.018),
    (6,  "Book",          -45.21,  -45.19,  -45.10,  -45.95,  -44.25,  -45.40,  0.010, 0.029, 0.061, 0.009),
    (6,  "Ring",          -43.71,  -44.70,  -44.18,  -45.07,  -42.54,  -44.30,  0.013, 0.043, 0.014, 0.010),
    (7,  "",              -57.71,  -57.37,  -57.20,  -58.08,  None,    -57.40,  0.016, 0.041, 0.046, 0.025),
    (8,  "D2d",           -73.32,  -72.28,  -71.97,  -73.58,  -71.55,  -73.00,  0.006, 0.041, 0.018, 0.004),
    (8,  "S4",            -72.93,  -72.35,  -72.22,  -73.55,  -71.56,  -72.90,  0.007, 0.019, 0.017, 0.005),
    (9,  "D2dDD",         -82.87,  -81.67,  -81.49,  -83.00,  None,    -83.00,  0.089, 0.116, 0.044, 0.052),
    (10, "",              -94.72,  -93.07,  -93.02,  -94.50,  None,    -94.60,  0.012, 0.049, 0.025, 0.010),
    (11, "43'4",         -104.23, -102.17, -102.09, -103.77, -100.23, -104.60,  0.034, 0.065, 0.024, 0.017),
    (16, "Antiboat",     -164.87, -162.20, -163.38, -164.20, -159.63, -164.60,  0.023, 0.064, 0.032, 0.017),
    (16, "4444-a",       -163.10, -162.98, -163.20, -164.84, -161.84, -164.20,  0.039, 0.038, 0.034, 0.015),
    (16, "4444-b",       -162.54, -162.87, -163.15, -163.84, -161.56, -164.10,  0.040, 0.049, 0.031, 0.029),
    (16, "Boat a",       -164.53, -161.92, -162.89, -164.51, -159.36, -164.40,  0.023, 0.038, 0.032, 0.016),
    (16, "Boat b",       -164.31, -162.04, -163.37, -164.35, -159.43, -164.20,  0.028, 0.057, 0.060, 0.016),
    (17, "Sphere",       -177.56, -174.15, -175.45, -175.78, -170.68, -175.70,  0.039, 0.063, 0.039, 0.022),
    (20, "ES Prism",     -212.49, -210.20, -212.09, -211.98, None,    -214.20,  0.042, 0.056, 0.056, 0.024),
    (20, "FS Prism",     -210.63, -208.46, -209.22, -210.12, None,    -211.90,  0.047, 0.050, 0.033, 0.023),
    (20, "Fused Cubes",  -208.07, -208.56, -208.90, -209.90, None,    -210.60,  0.067, 0.050, 0.034, 0.029),
    (20, "Pentag. Dodec.", -199.79, -197.99, -198.14, -201.22, None, -200.80,   0.034, 0.066, 0.047, 0.018),
    (25, "Isomer 2",     -276.50, -266.04, -271.37, -272.02, None,    -276.30,  0.029, 0.049, 0.054, 0.023),
]

FIELDS = [
    "n_waters", "isomer", "repo_label",
    "energy_q_aqua", "energy_mbpol", "energy_cmm", "energy_wb97xv", "energy_hippo",
    "energy_film", "energy_film_large", "energy_reference",
    "rmsd_q_aqua", "rmsd_mbpol", "rmsd_cmm", "rmsd_wb97xv",
    "rmsd_film", "rmsd_film_large",
]


def read_optimizer_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def build_rows(small_path, large_path):
    small = read_optimizer_csv(small_path)
    large = read_optimizer_csv(large_path)
    if not len(small) == len(large) == len(LITERATURE):
        raise SystemExit(
            f"row-count mismatch: {len(small)} small, {len(large)} large, "
            f"{len(LITERATURE)} literature -- the tables are no longer the same cluster set"
        )

    rows = []
    for lit, s, l in zip(LITERATURE, small, large):
        n, isomer, qa, mbpol, cmm, wb97xv, hippo, ref, r_qa, r_mbpol, r_cmm, r_wb97xv = lit
        if not int(s["n_waters"]) == int(l["n_waters"]) == n:
            raise SystemExit(
                f"cluster order diverged at '{s['label']}': the optimizer CSVs have "
                f"{s['n_waters']}/{l['n_waters']} waters where the table has {n}"
            )
        rows.append({
            "n_waters": n,
            "isomer": isomer,
            "repo_label": s["label"],
            "energy_q_aqua": qa,
            "energy_mbpol": mbpol,
            "energy_cmm": cmm,
            "energy_wb97xv": wb97xv,
            "energy_hippo": "" if hippo is None else hippo,
            "energy_film": round(float(s["binding_energy_kcal_mol"]), 2),
            "energy_film_large": round(float(l["binding_energy_kcal_mol"]), 2),
            "energy_reference": ref,
            "rmsd_q_aqua": r_qa,
            "rmsd_mbpol": r_mbpol,
            "rmsd_cmm": r_cmm,
            "rmsd_wb97xv": r_wb97xv,
            "rmsd_film": round(float(s["rmsd_all_angstrom"]), 3),
            "rmsd_film_large": round(float(l["rmsd_all_angstrom"]), 3),
        })
    return rows


def summarize(rows):
    """MAE/n and mean RMSD per method -- the same footers the published tables carry."""
    lines = []
    for method in ("q_aqua", "mbpol", "cmm", "wb97xv", "hippo", "film", "film_large"):
        errors = [
            abs(float(r[f"energy_{method}"]) - float(r["energy_reference"])) / r["n_waters"]
            for r in rows if r[f"energy_{method}"] != ""
        ]
        mae = sum(errors) / len(errors)
        rmsd_key = f"rmsd_{method}"
        if rmsd_key in FIELDS:
            values = [float(r[rmsd_key]) for r in rows]
            tail = f"   mean RMSD {sum(values) / len(values):.3f} A"
        else:
            tail = ""
        lines.append(f"{method:>11}: MAE/n {mae:.3f} kcal/mol  (n={len(errors)}){tail}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--small-csv", type=Path, default=DATA / "all_clusters_film_opt.csv")
    parser.add_argument("--large-csv", type=Path, default=DATA / "all_clusters_film_large_opt.csv")
    parser.add_argument("--out", type=Path, default=DATA / "benchmark_clusters.csv")
    args = parser.parse_args()

    rows = build_rows(args.small_csv, args.large_csv)
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {args.out} ({len(rows)} rows)")
    print(summarize(rows))


if __name__ == "__main__":
    main()
