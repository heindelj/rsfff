import numpy as np
import psi4


def sample_wigner(wfn, n_samples=100, temperature=298.15):
    """Samples atomic coordinates from the finite-temperature Wigner distribution

    using a completed Psi4 frequency calculation wavefunction.
    """
    # 1. Extract geometry, masses, and Hessian in atomic units
    geom_bohr = np.array(wfn.molecule().geometry())
    natoms = wfn.molecule().natom()

    masses_amu = np.array([wfn.molecule().mass(i) for i in range(natoms)])
    # Divide by au2amu to convert from AMU to electron masses (m_e)
    masses_au = masses_amu / psi4.constants.au2amu
    m_3n = np.repeat(masses_au, 3)
    inv_sqrt_m = 1.0 / np.sqrt(m_3n)

    hessian = np.array(wfn.hessian())

    # 2. Mass-weighted Hessian and normal mode decomposition
    mw_hessian = hessian * np.outer(inv_sqrt_m, inv_sqrt_m)
    evals, evecs = np.linalg.eigh(mw_hessian)

    # Filter out translational and rotational zero-modes (keep positive eigenvalues)
    vib_idx = np.where(evals > 1e-6)[0]
    omega = np.sqrt(evals[vib_idx])  # Angular frequencies w_k (a.u.)
    L = evecs[:, vib_idx]  # Normal mode eigenvectors

    # 3. Calculate temperature-dependent standard deviations
    # Convert k_B from J/K to Hartree/K
    kb_au = psi4.constants.kb / psi4.constants.hartree2J

    if temperature > 0.0:
        # Thermal scaling factor: sqrt(coth(hbar * omega / (2 * k_B * T)))
        # In atomic units, hbar = 1
        coth_val = 1.0 / np.tanh(omega / (2.0 * kb_au * temperature))
        scale_factor = np.sqrt(coth_val)
    else:
        # T = 0 K reduces to pure zero-point energy (ZPE)
        scale_factor = np.ones_like(omega)

    # Standard deviation for normal coordinates q_k at temperature T
    sigma_q = (1.0 / np.sqrt(2.0 * omega)) * scale_factor

    # 4. Sample configurations
    sampled_geometries_ang = []

    for _ in range(n_samples):
        # Sample normal coordinates from Gaussian N(0, sigma_q^2)
        q_sample = np.random.normal(loc=0.0, scale=sigma_q)

        # Transform back to Cartesian displacements: dx = M^{-1/2} * L * q
        dx = inv_sqrt_m * np.dot(L, q_sample)
        dx = dx.reshape((natoms, 3))

        # Add displacement to equilibrium geometry and convert to Angstroms
        geom_sample_bohr = geom_bohr + dx
        geom_sample_ang = geom_sample_bohr * psi4.constants.bohr2angstroms
        sampled_geometries_ang.append(geom_sample_ang)

    return np.array(sampled_geometries_ang)


# --- Example Usage & ML Dataset Export ---

if __name__ == "__main__":
    psi4.set_options({"basis": "def2-SVP", "reference": "rhf"})
    psi4.geometry("""
    O
    H 1 0.96
    H 1 0.96 2 104.5
    """)

    print("Running optimization and frequency calculation...")
    psi4.optimize("b3lyp")
    e_freq, wfn = psi4.frequency("b3lyp", return_wfn=True)

    # Generate 500 samples at room temperature (298.15 K)
    target_temp = 298.15
    geometries = sample_wigner(wfn, n_samples=500, temperature=target_temp)

    # Export directly to a multi-frame XYZ file for ML potential training
    symbols = [wfn.molecule().symbol(i) for i in range(wfn.molecule().natom())]
    output_filename = f"wigner_dataset_{int(target_temp)}K.xyz"

    with open(output_filename, "w") as f:
        for idx, geom in enumerate(geometries):
            f.write(f"{len(symbols)}\n")
            f.write(
                f"Sample {idx+1} | T={target_temp}K | Generated via Psi4 Wigner\n"
            )
            for sym, (x, y, z) in zip(symbols, geom):
                f.write(f"{sym:<4} {x:15.8f} {y:15.8f} {z:15.8f}\n")

    print(f"Successfully saved {len(geometries)} structures to {output_filename}")