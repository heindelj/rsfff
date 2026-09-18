"""Label stage: Q-Chem EDA + force reference data for the sampled water clusters.

One call of :meth:`QChemLabel.run` does, in order:

1. writes this iteration's frames as one extxyz geometry file into the existing job pool,
   ``<roundtrip_root>/<calculation>/geoms/<prefix>_iterNNN.xyz``, once per calculation;
2. calls ``qchem_roundtrip.generate_inputs`` so the pool expands them into one ``.in`` per
   frame per calculation, exactly as ``scripts/qchem_roundtrip.py generate`` would;
3. the first time inputs are written, runs the ``submit`` shell commands (sync the bundle up,
   top up the worker pool -- or just top up the pool, when the loop is already on Perlmutter);
4. runs the ``sync`` shell commands and checks for results; with ``wait_seconds`` it keeps
   syncing and checking on a timer, and otherwise raises :class:`easyal.Pending` as soon as
   anything is still outstanding;
5. once every job is complete or failed, merges each frame's **eda** and **force** outputs
   into one labeled frame.

Every step is safe to repeat, so ``loop.run(...)`` is simply called again until the stage
completes. The workers are untouched: they claim the new inputs from the pool like any others.

Waiting, or not
---------------
The default (``wait_seconds=0``) is the batch behaviour: submit, look once, report
``Pending``, let whatever drives the loop call it again later. That is right for a queue,
where the jobs may sit for hours. It is the wrong shape for an interactive allocation, where
the workers are running in the same session and the whole point is to watch one command go all
the way through -- so ``wait_seconds`` polls instead, running the ``sync`` hooks each time, and
falls back to ``Pending`` when the budget runs out. Nothing else changes between the two.

Two jobs, one frame
-------------------
A training frame needs both the EDA decomposition and the analytic forces, and they are
separate Q-Chem runs. ``scripts/parse_roundtrip.build_merged_frame`` is what joins them, and
its cross-checks are the reason to want both: the two jobs are given the same geometry and
must come back in the same standard orientation (1e-8 A) with the same supersystem energy
(1e-6 Ha). A frame missing either job, or failing those checks, is dropped and counted;
``max_failed_fraction`` decides whether that stops the stage.

The merged geometry is Q-Chem's standard orientation, not the sampled one -- forces and
multipoles are in that frame, so the coordinates have to be. The link back to the sampled
structure is the stem, and it is checked: same species sequence, and the same sorted list of
interatomic distances, which is what survives the reorientation.

Output frames carry the schema ``scripts/parse_roundtrip.py`` writes (atomic units)::

    arrays  species pos forces mulliken_charges fragment_idx
    info    energy eda_* fragment_energies fragment_dipoles fragment_second_moments
            dipole quadrupole octopole hexadecapole n_fragments charge multiplicity
            fragment_charges fragment_multiplicities method basis config_type source

plus the loop's own lineage: ``sample_id`` (the sampled frame's ``source``), ``al_loop``,
``al_iteration``, ``al_stem``.
"""

from __future__ import annotations

import json
import math
import subprocess
import time
from pathlib import Path

from easyal import Contract, Label, Pending

from common import DEFAULT_CONFIG, ROUNDTRIP_ROOT, qchem_roundtrip, roundtrip_parser

__all__ = ["QCHEM_TRAINING", "QChemLabel"]

#: What a merged eda+force frame guarantees, and what the trainer consumes.
QCHEM_TRAINING = Contract(
    info=["energy", "charge", "multiplicity", "n_fragments",
          "eda_cls_elec", "eda_mod_pauli", "eda_disp", "eda_pol", "eda_ct"],
    arrays=["forces", "fragment_idx"],
)

#: What the sampler has to hand over: a structure that already knows its fragmentation, since
#: that is what an EDA input is built from.
SAMPLED_STRUCTURE = Contract(
    info=["charge", "multiplicity", "n_fragments", "fragment_charges",
          "fragment_multiplicities"],
    arrays=["fragment_idx"],
)


def _distance_spectrum(positions) -> list[float]:
    """Sorted interatomic distances -- what a rigid reorientation leaves alone."""
    out = []
    for i, a in enumerate(positions):
        for b in positions[i + 1:]:
            out.append(math.dist(a, b))
    out.sort()
    return out


class QChemLabel(Label):
    """Label sampled frames with Q-Chem EDA and force jobs from the round-trip pool.

    Parameters (all keyword, all recorded in ``stage.json``)

    ``roundtrip_root``      the job pool (default: ``$RSFFF_QCHEM_ROOT`` or
                            ``<repo>/qchem_roundtrip``)
    ``config``              its ``config.json`` (default: ``<roundtrip_root>/config.json``)
    ``calculations``        which calculations to run per frame (default ``("eda", "force")``;
                            both are needed for a merged frame)
    ``prefix``              geometry/input stem prefix (default ``al_<loop directory name>``)
    ``submit``              shell commands run once, in ``roundtrip_root``, after inputs appear
    ``sync``                shell commands run in ``roundtrip_root`` before every check
    ``max_failed_fraction`` fraction of frames allowed to fail or be dropped before the stage
                            raises instead of completing (default 0.0: any loss stops it)
    ``wait_seconds``        how long to keep polling for results before reporting ``Pending``
                            (default 0: look once). Use it when the workers are running in the
                            same allocation
    ``poll_seconds``        seconds between checks while waiting (default 30)
    ``geometry_tol``        tolerance on the interatomic-distance check, Angstrom (1e-4)
    ``strict``              treat a merge warning (an EDA or force consistency complaint) as a
                            drop rather than a note (default False)
    """

    requires = SAMPLED_STRUCTURE
    produces = QCHEM_TRAINING

    def __init__(self, **params):
        params.setdefault("roundtrip_root", str(ROUNDTRIP_ROOT))
        params.setdefault("config", None)
        params.setdefault("calculations", ("eda", "force"))
        params.setdefault("prefix", None)
        params.setdefault("submit", [])
        params.setdefault("sync", [])
        params.setdefault("max_failed_fraction", 0.0)
        params.setdefault("wait_seconds", 0.0)
        params.setdefault("poll_seconds", 30.0)
        params.setdefault("geometry_tol", 1e-4)
        params.setdefault("strict", False)
        super().__init__(**params)

    # --- paths and helpers ------------------------------------------------------------------

    @property
    def roundtrip_root(self) -> Path:
        return Path(self.params["roundtrip_root"]).resolve()

    def config_path(self) -> Path:
        cfg = self.params["config"]
        return (self.roundtrip_root / "config.json") if cfg is None else Path(cfg).resolve()

    def stem(self, ctx) -> str:
        prefix = self.params["prefix"] or f"al_{ctx.root.name}"
        return f"{prefix}_iter{ctx.iteration:03d}"

    def _shell(self, ctx, commands, what: str) -> None:
        for command in commands:
            ctx.note(f"{what}: {command}")
            subprocess.run(command, shell=True, check=True, cwd=str(self.roundtrip_root))

    @staticmethod
    def _geometry_text(frames) -> str:
        """The frames as extxyz the round-trip generator accepts, for both molecule modes.

        Written here rather than with ``easyal.write_extxyz`` because the generator parses the
        ``Properties`` types strictly: ``fragment_idx`` has to be ``I``, and the per-fragment
        charges and multiplicities have to be quoted integer lists.
        """
        out = []
        for frame in frames:
            species = list(frame["arrays"]["species"])
            positions = frame["arrays"]["pos"]
            fragment_idx = [int(v) for v in frame["arrays"]["fragment_idx"]]
            info = frame["info"]
            charges = " ".join(str(int(c)) for c in info["fragment_charges"])
            mults = " ".join(str(int(m)) for m in info["fragment_multiplicities"])
            header = (
                "Properties=species:S:1:pos:R:3:fragment_idx:I:1"
                f" charge={int(info['charge'])}"
                f" multiplicity={int(info['multiplicity'])}"
                f" n_fragments={int(info['n_fragments'])}"
                f' fragment_charges="{charges}"'
                f' fragment_multiplicities="{mults}"'
                f" sample_id={info.get('source', '')}"
            )
            out.append(f"{len(species)}\n{header}\n")
            for symbol, (x, y, z), f in zip(species, positions, fragment_idx):
                out.append(f"{symbol:<3} {float(x):18.10f} {float(y):18.10f} "
                           f"{float(z):18.10f} {f:4d}\n")
        return "".join(out)

    # --- the stage --------------------------------------------------------------------------

    def run(self, ctx):
        root = self.roundtrip_root
        calculations = tuple(self.params["calculations"])
        config_path = self.config_path()
        if not config_path.exists():
            raise FileNotFoundError(
                f"{config_path} does not exist, so there is no Q-Chem job pool to label "
                f"through. Point roundtrip_root= (or $RSFFF_QCHEM_ROOT) at the "
                f"qchem_roundtrip directory."
            )
        cfg = qchem_roundtrip.load_config(ctx.track("roundtrip_config", config_path))
        for name in calculations:
            if name not in cfg["calculations"]:
                raise ValueError(
                    f"calculation {name!r} is not in {self.config_path()} "
                    f"(have {sorted(cfg['calculations'])})"
                )
        qchem_roundtrip.ensure_layout(root, cfg)

        samples = ctx.read()
        if not samples:
            raise ValueError(f"{ctx.input}: no frames to label")
        stem = self.stem(ctx)
        text = self._geometry_text(samples)

        # 1. one geometry file per calculation; identical content, so the two calculations
        #    produce the same per-frame stems and join without a lookup table.
        for name in calculations:
            geom = root / name / "geoms" / f"{stem}.xyz"
            if geom.exists():
                if geom.read_text() != text:
                    raise RuntimeError(
                        f"{geom} exists with different content; it belongs to another sample "
                        f"set. Use a different prefix=, or move that file away."
                    )
            else:
                qchem_roundtrip.atomic_write(geom, text)
                ctx.note(f"wrote {len(samples)} geometries to {geom}")

        # 2. let the pool's own generator turn them into inputs (it skips ones that exist;
        #    it also regenerates every other geometry sitting in these calculation folders,
        #    which is what the worker does on every poll anyway)
        generated = qchem_roundtrip.generate_inputs(root, cfg, calculations=set(calculations))
        mine = {}
        for item in generated:
            if Path(item.geometry).name != f"{stem}.xyz":
                continue
            mine.setdefault(item.frame_index, {})[item.calculation] = Path(item.input_path)
        missing = [i for i in range(len(samples))
                   if sorted(mine.get(i, {})) != sorted(calculations)]
        if missing:
            raise RuntimeError(
                f"the generator did not produce every {'/'.join(calculations)} input for "
                f"{stem} (frames {missing[:5]}{' ...' if len(missing) > 5 else ''})"
            )
        n_new = sum(1 for item in generated
                    if Path(item.geometry).name == f"{stem}.xyz" and not item.skipped)
        if n_new:
            ctx.note(f"generated {n_new} Q-Chem inputs for {stem}")

        manifest = ctx.scratch / "jobs.json"
        if not manifest.exists():
            manifest.write_text(json.dumps({
                "roundtrip_root": str(root), "stem": stem,
                "calculations": list(calculations),
                "jobs": [{"frame": i, "inputs": {c: str(p) for c, p in per.items()}}
                         for i, per in sorted(mine.items())],
            }, indent=2) + "\n")

        # 3. submit once
        submitted = ctx.scratch / "submitted.json"
        if not submitted.exists():
            self._shell(ctx, list(self.params["submit"]), "submit")
            submitted.write_text(json.dumps(
                {"time": time.time(), "commands": list(self.params["submit"])}) + "\n")

        # 4./5. sync and look for results; with wait_seconds, keep looking
        wait_seconds = float(self.params["wait_seconds"])
        poll_seconds = max(float(self.params["poll_seconds"]), 1.0)
        deadline = time.monotonic() + wait_seconds
        first = True
        while True:
            self._shell(ctx, list(self.params["sync"]), "sync" if first else "sync (poll)")
            ready, failed, waiting = self._scan(root, calculations, mine, len(samples))
            if not waiting or time.monotonic() >= deadline:
                break
            first = False
            # printed, not noted: a long wait would otherwise fill stage.json with heartbeats
            print(f"[iter {ctx.iteration} {self.name}] {len(ready)}/{len(samples)} frames "
                  f"complete, {len(waiting)} waiting; checking again in {poll_seconds:.0f}s",
                  flush=True)
            time.sleep(poll_seconds)
        ctx.log(n_frames=len(samples), n_ready=len(ready), n_failed=len(failed),
                n_waiting=len(waiting), stem=stem, waited_seconds=round(wait_seconds, 1),
                n_jobs=len(samples) * len(calculations))
        if waiting:
            raise Pending(
                f"{len(ready)}/{len(samples)} frames complete, {len(failed)} failed, "
                f"{len(waiting)} waiting on Q-Chem ({'/'.join(calculations)} under "
                f"{root}); run again when the workers have finished"
            )

        # 6. merge
        parser = roundtrip_parser()
        frames, dropped = [], {f"frame{i:04d}": reason for i, reason in failed.items()}
        tol = float(self.params["geometry_tol"])
        for i, outputs in sorted(ready.items()):
            tag = f"frame{i:04d}"
            try:
                merged, warnings = parser.build_merged_frame(
                    str(outputs["eda"]), str(outputs["force"]), str(root)
                )
            except Exception as exc:             # a parser refusing this pair is a drop
                dropped[tag] = f"{type(exc).__name__}: {exc}"
                continue
            if warnings and self.params["strict"]:
                dropped[tag] = "; ".join(warnings)
                continue
            problem = self._check_correspondence(samples[i], merged, tol)
            if problem:
                dropped[tag] = problem
                continue
            if warnings:
                ctx.note(f"{tag}: {'; '.join(warnings)}")
            frames.append(self._as_frame(merged, samples[i], ctx,
                                         mine[i][calculations[0]].stem))

        ctx.log(n_labeled=len(frames), n_dropped=len(dropped))
        if dropped:
            (ctx.scratch / "dropped.json").write_text(json.dumps(dropped, indent=2) + "\n")
            ctx.note(f"dropped {len(dropped)} of {len(samples)} frames")
        if len(dropped) > float(self.params["max_failed_fraction"]) * len(samples):
            raise RuntimeError(
                f"{len(dropped)}/{len(samples)} frames failed or were dropped (allowed "
                f"fraction {self.params['max_failed_fraction']}); see "
                f"{ctx.scratch / 'dropped.json'}. Fix the jobs, clear their markers under "
                f"{root}/<calculation>/state/failed/, and run again."
            )
        if not frames:
            raise RuntimeError("no frames could be labeled")
        return frames

    # --- pieces -----------------------------------------------------------------------------

    def _scan(self, root: Path, calculations, mine: dict, n_frames: int):
        """``(ready, failed, waiting)`` over frames: a frame is ready only with every job."""
        ready, failed, waiting = {}, {}, []
        for i in range(n_frames):
            outputs, why = {}, []
            for name in calculations:
                state, output = self._job_state(root, name, mine[i][name])
                if state == "done":
                    outputs[name] = output
                else:
                    why.append(f"{name} {state}")
            if len(outputs) == len(calculations):
                ready[i] = outputs
            elif any(w.endswith("waiting") for w in why):
                waiting.append(i)
            else:
                failed[i] = ", ".join(why)
        return ready, failed, waiting

    @staticmethod
    def _job_state(root: Path, calculation: str, input_path: Path):
        """``("done", output) | ("failed", None) | ("waiting", None)`` for one pool job."""
        job_dir = input_path.parent.parent
        stem = input_path.stem
        output = job_dir / "outputs" / f"{stem}.out"
        if (job_dir / "state" / "done" / f"{stem}.json").exists() and output.exists():
            return "done", output
        if (job_dir / "state" / "failed" / f"{stem}.json").exists():
            return "failed", None
        return "waiting", None

    @staticmethod
    def _check_correspondence(sample, merged, tol: float) -> str | None:
        """Is this output really this sampled frame, after Q-Chem reoriented it?"""
        if list(merged.symbols) != list(sample["arrays"]["species"]):
            return "species sequence does not match the sampled frame"
        a = _distance_spectrum([[float(v) for v in row] for row in merged.positions])
        b = _distance_spectrum([[float(v) for v in row] for row in sample["arrays"]["pos"]])
        if len(a) != len(b):
            return "atom count does not match the sampled frame"
        worst = max((abs(x - y) for x, y in zip(a, b)), default=0.0)
        if worst > tol:
            return f"interatomic distances differ from the sampled frame by {worst:.2e} A"
        return None

    @staticmethod
    def _as_frame(merged, sample, ctx, job_stem: str) -> dict:
        """One merged Q-Chem frame as an easyal frame, with the loop's lineage on it."""
        parser = roundtrip_parser()
        info = {
            "energy": float(merged.energy),
            **{f"eda_{k}": float(merged.eda[k]) for k in parser.EDA_KEY_ORDER
               if k in merged.eda},
            "fragment_energies": [float(v) for v in merged.fragment_energies],
            "fragment_dipoles": [float(v) for v in merged.fragment_dipoles.reshape(-1)],
            "fragment_second_moments":
                [float(v) for v in merged.fragment_second_moments.reshape(-1)],
            **{name: [float(v) for v in merged.multipoles[name].reshape(-1)]
               for name in parser.MULTIPOLE_NAMES},
            "multipole_format": "tensor",
            "n_fragments": int(merged.n_fragments),
            "charge": int(merged.total_charge),
            "multiplicity": int(merged.multiplicity),
            "fragment_charges": [int(c) for c in merged.fragment_charges],
            "fragment_multiplicities": [int(m) for m in merged.fragment_mults],
            "method": str(merged.method),
            "basis": str(merged.basis),
            "config_type": parser.config_type(
                merged.symbols, merged.fragment_idx, merged.n_fragments
            ),
            "source": str(merged.source),
            "units": "atomic",
            # lineage: which packing, which minimum, which trajectory step this came from
            "sample_id": str(sample["info"].get("source", merged.sample_id)),
            "al_loop": str(ctx.root.name),
            "al_iteration": int(ctx.iteration),
            "al_stem": str(job_stem),
        }
        for key in ("n_waters", "temperature", "time_fs", "model_energy"):
            if key in sample["info"]:
                info[f"sampled_{key}" if key == "model_energy" else key] = sample["info"][key]
        return {
            "info": info,
            "arrays": {
                "species": list(merged.symbols),
                "pos": [[float(v) for v in row] for row in merged.positions],
                "forces": [[float(v) for v in row] for row in merged.forces],
                "mulliken_charges": [float(v) for v in merged.mulliken],
                "fragment_idx": [int(v) for v in merged.fragment_idx],
            },
        }
