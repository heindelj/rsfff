import numpy as np
import psi4


def compute_mlip_reference_data_analytical(
    mol, method="b3lyp", basis="def2-SVP"
):
    """Computes energy, forces, dipoles, quadrupoles, polarizabilities, and dipole

    derivatives using Psi4's analytical engines with maximal orbital reuse.
    """
    psi4.set_options(
        {
            "basis": basis,
            "reference": "rhf",
            "e_convergence": 1e-8,
            "d_convergence": 1e-8,
        }
    )

    # 1. Analytical Frequency Calculation
    # Solves ground-state SCF once, then evaluates gradients, dipole derivatives, and Hessian
    e_scf, wfn = psi4.frequency(method, return_wfn=True)

    # Extract Forces: negative of the analytical gradient (in Hartree/Bohr)
    forces = -1.0 * np.array(wfn.gradient())

    # 2. Polarizability & Multipoles via Orbital Reuse
    # Passing ref_wfn avoids re-running the base SCF! Solves only the 3 electric field CPHF equations.
    psi4.properties(
        method,
        properties=["DIPOLE", "QUADRUPOLE", "POLARIZABILITY"],
        ref_wfn=wfn,
    )

    # Extract Dipole (convert from Debye to strict atomic units: e*Bohr)
    debye2au = 1.0 / psi4.constants.dipolebu2debye
    dipole = (
        np.array(
            [
                wfn.variable("CURRENT DIPOLE X"),
                wfn.variable("CURRENT DIPOLE Y"),
                wfn.variable("CURRENT DIPOLE Z"),
            ]
        )
        * debye2au
    )

    # Extract Traceless Quadrupole Tensor (in atomic units: e*Bohr^2)
    quadrupole = np.array(wfn.variable("QUADRUPOLE TENSOR"))  # Shape: (3, 3)

    # Extract Static Polarizability Tensor (in atomic units: Bohr^3)
    polarizability = np.array(
        wfn.variable("POLARIZABILITY TENSOR")
    )  # Shape: (3, 3)

    # Extract Dipole Derivatives / Atomic Axial Tensors (Shape: 3N x 3 in atomic units)
    dipole_grad = np.array(wfn.variable("CURRENT DIPOLE GRADIENT"))
    natoms = mol.natom()
    dipole_derivatives = dipole_grad.reshape((natoms, 3, 3))  # (atom, d/dR, mu)

    return {
        "energy": e_scf,
        "forces": forces,
        "dipole": dipole,
        "quadrupole": quadrupole,
        "polarizability": polarizability,
        "dipole_derivatives": dipole_derivatives,
    }

def compute_mlip_reference_data_fast_field(
    mol, method="b3lyp", basis="def2-SVP", field_step=1e-3
):
    """Bypasses the nuclear Hessian entirely by using central finite differences of

    forces and dipoles under uniform electric fields to get polarizabilities
    and dipole derivatives in exactly 7 gradient evaluations.
    """
    psi4.set_options(
        {
            "basis": basis,
            "reference": "rhf",
            "e_convergence": 1e-9,
            "d_convergence": 1e-9,
        }
    )

    # 1. Unperturbed Ground State & Forces
    e_0, wfn_0 = psi4.energy(method, return_wfn=True)
    grad_0 = psi4.gradient(method, ref_wfn=wfn_0)
    forces_0 = -1.0 * np.array(grad_0)

    # Extract unperturbed multipoles (free from unperturbed density)
    psi4.properties(method, properties=["DIPOLE", "QUADRUPOLE"], ref_wfn=wfn_0)
    debye2au = 1.0 / psi4.constants.dipolebu2debye
    dipole_0 = (
        np.array(
            [
                wfn_0.variable("CURRENT DIPOLE X"),
                wfn_0.variable("CURRENT DIPOLE Y"),
                wfn_0.variable("CURRENT DIPOLE Z"),
            ]
        )
        * debye2au
    )
    quadrupole_0 = np.array(wfn_0.variable("QUADRUPOLE TENSOR"))

    # 2. Electric Field Perturbations (+/- Fx, Fy, Fz)
    natoms = mol.natom()
    polarizability = np.zeros((3, 3))
    dipole_derivatives = np.zeros(
        (natoms, 3, 3)
    )  # Shape: (atom, R_coord, F_coord)

    field_axes = [
        [field_step, 0.0, 0.0],
        [0.0, field_step, 0.0],
        [0.0, 0.0, field_step],
    ]

    for axis_idx, f_vec in enumerate(field_axes):
        forces_step = []
        dipoles_step = []

        for sign in [+1.0, -1.0]:
            applied_field = [sign * x for x in f_vec]
            psi4.set_options({"e_field": applied_field})

            # Re-use unperturbed orbitals as starting guess -> converges in ~2 iterations!
            e_step, wfn_step = psi4.energy(method, return_wfn=True)
            grad_step = psi4.gradient(method, ref_wfn=wfn_step)
            forces_step.append(-1.0 * np.array(grad_step))

            psi4.properties(
                method, properties=["DIPOLE"], ref_wfn=wfn_step
            )
            dip_au = (
                np.array(
                    [
                        wfn_step.variable("CURRENT DIPOLE X"),
                        wfn_step.variable("CURRENT DIPOLE Y"),
                        wfn_step.variable("CURRENT DIPOLE Z"),
                    ]
                )
                * debye2au
            )
            dipoles_step.append(dip_au)

        # Central difference for Polarizability: d(mu)/d(F)
        polarizability[:, axis_idx] = (dipoles_step[0] - dipoles_step[1]) / (
            2.0 * field_step
        )

        # Central difference for Dipole Derivatives via Interchange Theorem: d(force)/d(F) = d(mu)/d(R)
        df_df = (forces_step[0] - forces_step[1]) / (2.0 * field_step)
        dipole_derivatives[:, :, axis_idx] = df_df

    # Reset electric field to zero for subsequent calculations
    psi4.set_options({"e_field": [0.0, 0.0, 0.0]})

    return {
        "energy": e_0,
        "forces": forces_0,
        "dipole": dipole_0,
        "quadrupole": quadrupole_0,
        "polarizability": polarizability,
        "dipole_derivatives": dipole_derivatives,
    }