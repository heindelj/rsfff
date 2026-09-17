"""The two sampling stages: minimize with the model, then run dynamics from the minima.

Why two stages and not one
--------------------------
They answer different questions and fail differently. The minimizer asks *where are this
model's minima*, and a packmol structure that the model cannot relax is a statement about the
model worth recording. The dynamics asks *what does this model do at temperature around those
minima*, and its failure mode is a trajectory that heats up, breaks a molecule or evaporates
one. Splitting them gives each its own contract, its own hashes and its own metrics in
``stage.json``, and lets either be reset and redone without the other (``loop.reset(i,
"dynamics")``).

Running dynamics *from the minima* rather than from the packings is the point of the ordering:
a random packing has 100 kcal/mol of strain in it, and a thermostatted trajectory started there
spends its first picosecond dumping that into the bath, sampling nothing that the reference
method should ever be asked to label.

What is kept
------------
Everything the stage produces is a structure the *labeling* stage may be asked to run, so both
stages are conservative: a frame whose energy is not finite, whose forces have blown up, whose
temperature has run away, or whose waters are no longer intact molecules is dropped and
counted, not written. ``model_energy`` on a kept frame is the model's own number, wall
excluded -- it is a prediction to compare against, never a label.
"""

from __future__ import annotations

import time

import numpy as np
from ase import Atoms, units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

from easyal import Contract, Sample, new_frame
from rsfff.md.film_driver import optimize

from .model import HARTREE_TO_EV, FilmCalculator, LoadedFilmModel, fragment_or_none

__all__ = ["MinimizeSample", "DynamicsSample", "CARRIED", "STRUCTURE", "SAMPLED"]

#: Frame info a sampling stage passes through untouched.
CARRIED = ("charge", "multiplicity", "n_waters")

#: What a stage needs from its input: bare structures that know their charge state.
STRUCTURE = Contract(info=CARRIED)

#: What a sampling stage guarantees: the same, plus the model's energy for the frame.
SAMPLED = Contract(info=(*CARRIED, "model_energy"))


def _carry(frame: dict) -> dict:
    info = {k: frame["info"][k] for k in CARRIED}
    info["parent"] = frame["info"].get("source", "")
    return info


def _model(ctx) -> LoadedFilmModel:
    if ctx.model is None:
        raise RuntimeError(
            f"stage {ctx.stage.name!r} needs a model: pass initial_model=<checkpoint> to "
            f"ActiveLearning (iteration 0 has no trained model to fall back on)"
        )
    return LoadedFilmModel(ctx.model, device=ctx.params.get("device", "cpu"))


class MinimizeSample(Sample):
    """Relax every built structure on the model's own surface (L-BFGS-B, Cartesian).

    ``gtol``          convergence on max |dE/dR|, Hartree/Angstrom (default 1e-3). These
                      minima are trajectory starting points, not spectroscopy: a 300 K
                      trajectory carries |F|max around 0.07 Hartree/Angstrom, so 1e-3 is
                      already two orders of magnitude inside the thermal noise the structure
                      is about to be given. The driver's own default (1e-7) and the 1e-5 this
                      stage used first both spend most of their iterations below the level
                      anything downstream can distinguish. Tighten it only if these minima
                      are also going to a Hessian
    ``max_iter``      per restart (default 2000)
    ``max_restarts``  L-BFGS-B restarts from the current point (default 1). Restarts exist to
                      clear a stale limited-memory Hessian in the last order of magnitude of
                      convergence, which is below where this stage stops
    ``keep``          ``"converged"`` (default) or ``"all"``
    ``with_induction``  evaluate the coupled induction solve (default True)
    """

    name = "optimize"
    requires = STRUCTURE
    produces = SAMPLED

    def run(self, ctx):
        p = ctx.params
        model = _model(ctx)
        ctx.log(**{f"model_{k}": v for k, v in model.describe().items()})
        keep_all = p.get("keep", "converged") == "all"
        gtol = float(p.get("gtol", 1e-3))

        frames_in = ctx.read()
        out, n_broken, n_unconverged, iterations = [], 0, 0, []
        t0 = time.perf_counter()
        for i, frame in enumerate(frames_in):
            species = frame["arrays"]["species"]
            start = np.asarray(frame["arrays"]["pos"], dtype=float)
            if fragment_or_none(species, start) is None:
                n_broken += 1
                continue
            potential, _ = model.potential(species, start,
                                           with_induction=p.get("with_induction", True))
            e_start = float(potential.energy(start)[0])
            result = optimize(potential, start, gtol=gtol,
                              max_iter=int(p.get("max_iter", 2000)),
                              max_restarts=int(p.get("max_restarts", 1)))
            iterations.append(result.n_iterations)
            if not result.converged:
                n_unconverged += 1
                if not keep_all:
                    continue
            if fragment_or_none(species, result.positions) is None:
                n_broken += 1
                continue
            info = _carry(frame)
            n = int(info["n_waters"])
            info.update(
                source=f"{info['parent']}|min",
                model_energy=round(result.energy, 12),
                model_energy_per_water=round(result.energy / max(n, 1), 12),
                relaxation_energy=round(e_start - result.energy, 12),
                opt_max_force=round(result.max_force, 12),
                opt_rms_force=round(result.rms_force, 12),
                opt_converged=bool(result.converged),
                opt_iterations=int(result.n_iterations),
                opt_evaluations=int(result.n_evaluations),
            )
            out.append(new_frame(species, result.positions, info))
            if (i + 1) % 10 == 0:
                ctx.note(f"minimized {i + 1}/{len(frames_in)}")

        if not out:
            raise RuntimeError(
                f"every one of {len(frames_in)} structures was dropped "
                f"({n_unconverged} unconverged, {n_broken} not intact water)"
            )
        ctx.log(n_in=len(frames_in), n_out=len(out), n_unconverged=n_unconverged,
                n_broken=n_broken, gtol=gtol,
                median_iterations=int(np.median(iterations)) if iterations else 0,
                seconds=round(time.perf_counter() - t0, 2))
        return out


class DynamicsSample(Sample):
    """Langevin NVT from every input structure, inside a flat-bottom spherical wall.

    ``temperature_K``     thermostat temperature (default 300)
    ``timestep_fs``       0.5 fs by default: the model has real O-H stretches in it
    ``friction_per_fs``   Langevin friction, 1/fs (default 0.02)
    ``equilibrate_steps`` discarded before sampling starts (default 1000 = 500 fs, about
                          ten thermostat time constants; a cluster started from a minimum with
                          300 K velocities loses half of that to potential energy immediately
                          and needs the thermostat to put it back before anything is sampled)
    ``steps``             sampled steps per structure (default 2000)
    ``stride``            steps between kept frames (default 100)
    ``wall_margin``       A added to the starting cluster radius for the wall (default 2.0)
    ``wall_k``            wall force constant, Hartree/A^2 (default 0.5); 0 disables it
    ``max_temperature``   abort a trajectory above this (default 3x ``temperature_K``)
    ``max_force``         abort above this |F|max, eV/A (default 50)
    ``seed``              base RNG seed; trajectory j uses ``seed + 1000 * iteration + j``
    """

    name = "dynamics"
    requires = STRUCTURE
    produces = SAMPLED

    def run(self, ctx):
        p = ctx.params
        model = _model(ctx)
        ctx.log(**{f"model_{k}": v for k, v in model.describe().items()})

        temperature = float(p.get("temperature_K", 300.0))
        dt = float(p.get("timestep_fs", 0.5))
        friction = float(p.get("friction_per_fs", 0.02))
        equilibrate = int(p.get("equilibrate_steps", 1000))
        steps = int(p.get("steps", 2000))
        stride = int(p.get("stride", 100))
        wall_k = float(p.get("wall_k", 0.5))
        wall_margin = float(p.get("wall_margin", 2.0))
        t_ceiling = float(p.get("max_temperature", 3.0 * temperature))
        f_ceiling = float(p.get("max_force", 50.0))
        seed0 = int(p.get("seed", 20260917)) + 1000 * ctx.iteration

        frames_in = ctx.read()
        out, aborted, n_broken = [], [], 0
        t0 = time.perf_counter()
        for j, frame in enumerate(frames_in):
            species = list(frame["arrays"]["species"])
            start = np.asarray(frame["arrays"]["pos"], dtype=float)
            if fragment_or_none(species, start) is None:
                n_broken += 1
                continue
            potential, _ = model.potential(species, start,
                                           with_induction=p.get("with_induction", True))
            radius = float(np.linalg.norm(start - start.mean(axis=0), axis=1).max())
            radius = max(radius, float(frame["info"].get("cavity_radius", 0.0))) + wall_margin
            atoms = Atoms(symbols=species, positions=start)
            atoms.calc = FilmCalculator(potential, wall_radius=radius, wall_k=wall_k)

            rng = np.random.default_rng(seed0 + j)
            MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, rng=rng)
            Stationary(atoms)
            # fixcm=False: ASE's in-thermostat center-of-mass fix does not sample NVT
            # exactly and is deprecated, and its replacement -- a FixCom constraint -- goes
            # unstable on clusters this small (a 3-water run reached 20000 K in 300 steps).
            # Letting the center of mass random-walk costs nothing here: the potential and
            # the wall are both translation-invariant, the wall being measured from the
            # running center of mass, so nothing in the trajectory depends on where the
            # cluster is. Stationary() above only removes the initial drift.
            dyn = Langevin(atoms, timestep=dt * units.fs, temperature_K=temperature,
                           friction=friction / units.fs, rng=rng, fixcm=False)
            if equilibrate:
                dyn.run(equilibrate)

            info0 = _carry(frame)
            kept_here, reason = 0, ""
            for step in range(stride, steps + 1, stride):
                dyn.run(stride)
                results = atoms.calc.results
                energy = results["model_energy_hartree"]
                t_now = atoms.get_temperature()
                fmax = float(np.abs(atoms.get_forces()).max())
                if not np.isfinite(energy):
                    reason = "energy is not finite"
                elif t_now > t_ceiling:
                    reason = f"{t_now:.0f} K over the {t_ceiling:.0f} K ceiling"
                elif fmax > f_ceiling:
                    reason = f"|F|max {fmax:.1f} over {f_ceiling:.0f} eV/A"
                elif fragment_or_none(species, atoms.get_positions()) is None:
                    reason = "a water is no longer intact"
                if reason:
                    break
                info = dict(info0)
                info.update(
                    source=f"{info0['parent']}|md{equilibrate + step}",
                    model_energy=round(float(energy), 12),
                    model_energy_per_water=round(float(energy) / max(int(info0["n_waters"]), 1), 12),
                    wall_energy=round(float(results["wall_energy_hartree"]), 12),
                    temperature=round(float(t_now), 3),
                    time_fs=round((equilibrate + step) * dt, 3),
                    md_step=equilibrate + step,
                    wall_radius=round(radius, 4),
                    max_force_ev=round(fmax, 6),
                )
                out.append(new_frame(species, atoms.get_positions(), info))
                kept_here += 1
            if reason:
                aborted.append((info0["parent"], reason, kept_here))
            ctx.note(f"trajectory {j + 1}/{len(frames_in)} ({len(species) // 3} waters): "
                     f"{kept_here} frames" + (f", stopped: {reason}" if reason else ""))

        if not out:
            raise RuntimeError(f"no frames survived dynamics ({len(aborted)} trajectories "
                               f"aborted, {n_broken} inputs not intact water)")
        temperatures = [f["info"]["temperature"] for f in out]
        ctx.log(n_in=len(frames_in), n_out=len(out), n_aborted=len(aborted),
                n_broken_input=n_broken, temperature_K=temperature, timestep_fs=dt,
                steps=steps, stride=stride, equilibrate_steps=equilibrate,
                wall_k=wall_k, seed0=seed0,
                mean_temperature=round(float(np.mean(temperatures)), 1),
                seconds=round(time.perf_counter() - t0, 2))
        if aborted:
            ctx.note("aborted: " + "; ".join(f"{s} after {k} frames: {r}"
                                             for s, r, k in aborted[:8]))
        return out
