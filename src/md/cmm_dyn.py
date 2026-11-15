"""Batched molecular dynamics on a cmm model: R independent replicas (same atoms or not), one model call per step.

Integrator: velocity Verlet with a Langevin (BAOAB-style O step) thermostat per replica, and optionally the
stochastic cell rescaling barostat of Bernetti & Bussi (J. Chem. Phys. 153, 114107 (2020)) -- isotropic, or
per-axis ("aniso": each diagonal cell length its own strain, the off-diagonal cell fixed; "z": x and y fixed,
z coupled -- the solid-liquid coexistence set-up). Units: ASE's (eV, Angstrom, amu, ASE time); temperatures K,
pressures GPa.

    sim = BatchMD(potential, atoms_list, dt_fs=2.0, T=[1100, 1150], gamma_ps=5.0, P_GPa=0.0, baro="iso")
    for blk in range(n): sim.run(100); print(sim.thermo())

``thermo()`` is per replica: potential energy (eV/atom, the model's interaction energy), kinetic temperature,
volume (A^3), pressure (GPa), cell lengths.
"""
from __future__ import annotations

import time

import numpy as np
from ase import units

HEV = 27.211386245988          # Hartree -> eV
EV_A3_TO_GPA = 160.21766208


class BatchMD:
    def __init__(self, potential, atoms_list, *, dt_fs=2.0, T=1000.0, gamma_ps=5.0, P_GPa=None, baro="iso",
                 tau_p_ps=1.0, compressibility_GPa=0.1, seed=0, init_velocities=True):
        self.pot = potential
        R = len(atoms_list)
        self.R = R
        self.Z = [a.numbers.copy() for a in atoms_list]
        self.X = [a.positions.copy() for a in atoms_list]
        self.C = [a.cell.array.copy() for a in atoms_list]
        self.M = [a.get_masses()[:, None] for a in atoms_list]
        self.T = np.broadcast_to(np.asarray(T, float), (R,)).copy()
        self.dt = dt_fs * units.fs
        self.gamma = gamma_ps * 1e-3 / units.fs
        self.P = None if P_GPa is None else np.broadcast_to(np.asarray(P_GPa, float), (R,)) / EV_A3_TO_GPA
        if baro not in ("iso", "aniso", "z"):
            raise ValueError(f"baro {baro!r}: iso | aniso | z")
        self.baro = baro
        self.tau_p = tau_p_ps * 1e3 * units.fs
        self.beta = compressibility_GPa * EV_A3_TO_GPA       # isothermal compressibility, 1/(eV/A^3)
        self.rng = np.random.default_rng(seed)
        if init_velocities:
            self.V = [self.rng.normal(size=x.shape) * np.sqrt(units.kB * t / m) for x, m, t in zip(self.X, self.M, self.T)]
            for v, m in zip(self.V, self.M):      # no centre-of-mass drift
                v -= (m * v).sum(0) / m.sum()
        else:
            self.V = [a.get_velocities().copy() for a in atoms_list]
        self.steps = 0
        self.wall = 0.0
        self.cg_iters = []
        self._forces()

    # -- forces --------------------------------------------------------------------------------------------------
    def _forces(self):
        E, F, vir = self.pot.evaluate(self.Z, self.X, self.C, stress=True)
        self.E = E * HEV
        self.F = [f * HEV for f in F]
        self.vir = vir * HEV                              # eV, -dE/d strain
        self.cg_iters.append(int(self.pot.last.get("cg_iters", 0)))

    def volume(self, r):
        return abs(np.linalg.det(self.C[r]))

    def pressure_tensor(self, r):
        v, m = self.V[r], self.M[r]
        kin = (m[:, :, None] * v[:, :, None] * v[:, None, :]).sum(0)
        return (kin + self.vir[r]) / self.volume(r)

    # -- integration ---------------------------------------------------------------------------------------------
    def _thermostat(self, h):
        c1 = np.exp(-self.gamma * h)
        for r in range(self.R):
            s = np.sqrt((1 - c1 ** 2) * units.kB * self.T[r] / self.M[r])
            self.V[r] = c1 * self.V[r] + s * self.rng.normal(size=self.V[r].shape)

    def _barostat(self, h):
        """Stochastic cell rescaling over a time h (Euler-Maruyama in the log strain), positions and velocities
        scaled with the cell."""
        for r in range(self.R):
            V = self.volume(r)
            kT = units.kB * self.T[r]
            Pt = self.pressure_tensor(r)
            if self.baro == "iso":
                P = np.trace(Pt) / 3
                de = (-self.beta / self.tau_p * (self.P[r] - P - kT / V) * h
                      + np.sqrt(2 * kT * self.beta * h / (V * self.tau_p)) * self.rng.normal())
                mu = np.full(3, np.exp(de / 3))
            else:
                axes = range(3) if self.baro == "aniso" else (2,)
                mu = np.ones(3)
                for k in axes:
                    de = (-self.beta / (3 * self.tau_p) * (self.P[r] - Pt[k, k] - kT / V) * h
                          + np.sqrt(2 * kT * self.beta * h / (3 * V * self.tau_p)) * self.rng.normal())
                    mu[k] = np.exp(de)
            # cells with orthogonal axes along x, y, z (the cubic supercells here): scale column k by mu[k]
            self.C[r] = self.C[r] * mu[None, :]
            self.X[r] = self.X[r] * mu[None, :]
            self.V[r] = self.V[r] / mu[None, :]

    def step(self):
        dt = self.dt
        for r in range(self.R):
            self.V[r] = self.V[r] + 0.5 * dt * self.F[r] / self.M[r]
        for r in range(self.R):
            self.X[r] = self.X[r] + 0.5 * dt * self.V[r]
        self._thermostat(dt)
        if self.P is not None:
            self._barostat(dt)
        for r in range(self.R):
            self.X[r] = self.X[r] + 0.5 * dt * self.V[r]
        self._forces()
        for r in range(self.R):
            self.V[r] = self.V[r] + 0.5 * dt * self.F[r] / self.M[r]
        self.steps += 1

    def run(self, n, callback=None, every=1):
        t0 = time.perf_counter()
        for i in range(n):
            self.step()
            if callback is not None and (i + 1) % every == 0:
                callback(self)
        self.wall += time.perf_counter() - t0

    # -- observables ---------------------------------------------------------------------------------------------
    def temperature(self, r):
        n = len(self.X[r])
        return float((self.M[r] * self.V[r] ** 2).sum() / (3 * (n - 1) * units.kB))

    def thermo(self):
        out = []
        for r in range(self.R):
            n = len(self.X[r])
            Pt = self.pressure_tensor(r)
            out.append({"epot_eV_atom": float(self.E[r]) / n, "T": self.temperature(r), "V": self.volume(r),
                        "P_GPa": float(np.trace(Pt) / 3 * EV_A3_TO_GPA),
                        "Pzz_GPa": float(Pt[2, 2] * EV_A3_TO_GPA),
                        "cell": np.diag(self.C[r]).tolist()})
        return out

    def atoms(self, r):
        from ase import Atoms
        a = Atoms(numbers=self.Z[r], positions=self.X[r], cell=self.C[r], pbc=True)
        a.set_velocities(self.V[r])
        return a


def partial_rdf(atoms, rmax=None, nbins=200):
    """{"A-B": g(r)} for each species pair, and the bin centres; minimum-image over an orthorhombic cell."""
    from ase.neighborlist import neighbor_list
    L = np.diag(atoms.cell.array)
    rmax = rmax or 0.5 * L.min() - 1e-6
    i, j, d = neighbor_list("ijd", atoms, rmax)
    edges = np.linspace(0, rmax, nbins + 1)
    shell = 4 / 3 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    V = atoms.get_volume()
    sym = np.array(atoms.get_chemical_symbols())
    species = sorted(set(sym), key=lambda s: atoms.numbers[sym == s][0])
    out = {}
    for a in species:
        for b in species:
            if species.index(b) < species.index(a):
                continue
            na, nb = (sym == a).sum(), (sym == b).sum()
            mask = (sym[i] == a) & (sym[j] == b)
            h, _ = np.histogram(d[mask], edges)
            out[f"{a}-{b}"] = h / (na * nb / V * shell)
    return 0.5 * (edges[1:] + edges[:-1]), out
