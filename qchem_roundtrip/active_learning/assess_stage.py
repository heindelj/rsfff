"""Assess stage: what the freshly trained committee does on the frames it just learned.

The loop this drives is **schedule-driven**, not threshold-driven: it walks up in cluster size
for as many iterations as the schedule has entries, and stops because the walk is finished.
So :meth:`converged` returns False unless a ``threshold`` is set, and the job here is to
produce the numbers that say whether the walk is working.

What it measures, on this iteration's labeled frames:

``fit_rmse_energy``   Hartree per atom, committee mean against the Q-Chem reference
``fit_rmse_forces``   Hartree/Angstrom, per component
``sigma_energy``      mean committee spread, the same quantity the selection ranked on
``sigma_forces``      mean committee spread, per atom
``sigma_drop``        this iteration's mean ``sigma_forces`` over the previous one's

``sigma_drop`` is the one to watch: these frames were chosen *because* the previous committee
disagreed about them, so after labeling and refitting the spread on them should have fallen.
A ratio near 1 means the new data did not teach the model anything it did not already have,
which usually means the selection is finding the same kind of structure over and over.

**These frames are in the training set**, so ``fit_rmse_*`` is a fit error, not a
generalization error, and it will flatter the model. The honest version needs the train stage
to record the exact holdout it used (``split_indices_grouped`` on the concatenated dataset)
and this stage to read it -- the shape ``job_runner/active_learning/assess_stage.py`` uses in
the HBQ project, where the committee's 95% prediction interval is measured on the structures
it was held out from, with coverage reported alongside so a committee that understates its own
uncertainty is visible. That is the next thing to build here.
"""

from __future__ import annotations

import numpy as np

from easyal import Assess, read_extxyz

from committee import Committee

__all__ = ["ValidationAssess"]

#: Hartree/bohr -> Hartree/Angstrom. The reference forces are per bohr (parse_roundtrip's
#: schema) and the model's are per Angstrom; comparing them without this is a silent 1.89.
BOHR = 0.529177210903


class ValidationAssess(Assess):
    """Score the new committee on what it just learned.

    ``threshold``  optional: mean ``sigma_forces`` (Hartree/Angstrom) to stop below
    ``patience``   consecutive iterations under the threshold before stopping (default 2)
    ``device``     where to evaluate the committee
    """

    def run(self, ctx):
        committee = Committee.load(ctx.input, device=self.params.get("device", "cpu"))
        frames = read_extxyz(ctx.output_of("label"))
        if not frames:
            raise ValueError("the label stage produced no frames to assess")

        errors_e, errors_f, sigma_e, sigma_f, skipped = [], [], [], [], 0
        for frame in frames:
            try:
                spread = committee.spread(frame)
                energies, forces = committee.predict(
                    frame["arrays"]["species"], frame["arrays"]["pos"])
            except Exception:
                skipped += 1
                continue
            n_atoms = len(frame["arrays"]["species"])
            sigma_e.append(spread.sigma_energy)
            sigma_f.append(spread.sigma_forces)
            errors_e.append((energies.mean() - float(frame["info"]["energy"])) / n_atoms)
            if "forces" in frame["arrays"]:
                # the reference is Hartree/bohr (parse_roundtrip's schema); the model is
                # Hartree/Angstrom, and comparing them without saying so would be a silent
                # factor of 1.89
                reference = np.asarray(frame["arrays"]["forces"], dtype=float) / BOHR
                errors_f.append((forces.mean(axis=0) - reference).reshape(-1))

        history = ctx.loop.history()
        previous = history[-1]["sigma_forces"] if history else None
        mean_sigma_f = float(np.mean(sigma_f)) if sigma_f else 0.0
        metrics = {
            "n_frames": len(frames) - skipped,
            "n_skipped": skipped,
            "n_members": committee.n_members,
            "fit_rmse_energy": float(np.sqrt(np.mean(np.square(errors_e)))) if errors_e else None,
            "fit_rmse_forces": (float(np.sqrt(np.mean(np.square(np.concatenate(errors_f)))))
                                if errors_f else None),
            "sigma_energy": float(np.mean(sigma_e)) if sigma_e else 0.0,
            "sigma_forces": mean_sigma_f,
            "sigma_drop": (round(mean_sigma_f / previous, 4)
                           if previous not in (None, 0) else None),
        }
        ctx.note(f"fit RMSE {metrics['fit_rmse_energy']:.2e} Ha/atom, "
                 f"{metrics['fit_rmse_forces']:.2e} Ha/A; mean committee sigma_forces "
                 f"{mean_sigma_f:.2e}"
                 if metrics["fit_rmse_energy"] is not None else
                 f"mean committee sigma_forces {mean_sigma_f:.2e}")
        return metrics

    def converged(self, history) -> bool:
        threshold = self.params.get("threshold")
        if threshold is None:
            return False          # the schedule decides when this loop is finished
        patience = int(self.params.get("patience", 2))
        if len(history) < patience:
            return False
        return all(h["sigma_forces"] < float(threshold) for h in history[-patience:])
