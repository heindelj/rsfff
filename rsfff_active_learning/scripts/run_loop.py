#!/usr/bin/env python3
"""The active-learning loop: build -> dynamics -> select -> label -> train -> assess.

    python scripts/run_loop.py --config configs/water_udd.yaml            # run until pending
    python scripts/run_loop.py --config configs/water_udd.yaml --status
    python scripts/run_loop.py --root runs/loop_quick --quick             # tiny, CPU, own store

Rerunning is free: completed stages are skipped; a dynamics stage out of its time budget
continues from its checkpoints; the label stage checks the store; the train stage checks its
batch job. The return line is "pending" (call again later), "converged" (the size walk is
done) or "max_iterations".

The YAML (see configs/water_udd.yaml) has one block per stage; every key is a stage
parameter and ends up in that stage's stage.json. ``budget`` is shared by build and select.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
AL = HERE.parent
sys.path.insert(0, str(AL))

from easyal import ActiveLearning  # noqa: E402

from rsfff_al.assess import CommitteeAssess  # noqa: E402
from rsfff_al.label import QChemStoreLabel  # noqa: E402
from rsfff_al.stages import PackmolBuild, SelectUncertain, UncertaintyDynamics  # noqa: E402
from rsfff_al.train_stage import CommitteeTrain  # noqa: E402

QUICK = {
    "budget": {"schedule": [{"sizes": [3, 5], "labels_per_size": 4}], "pool_multiplier": 2,
               "candidates_per_trajectory": 4, "min_trajectories": 2},
    "dynamics": {"device": "cpu", "time_ps": 0.2, "warmup_fs": 50.0, "window_fs": 50.0,
                 "stride": 10},
    # labels from a film model instead of Q-Chem (scripts/fake_qchem.py), into the loop's own
    # store -- a dry run of every stage, not data
    "label": {"sync": [f"{sys.executable} {HERE / 'fake_qchem.py'} {{store}}"]},
    "train": {"mode": "local", "device": "cpu", "members": 2, "warm_epochs": 1,
              "set": ["train.epochs=1", "train.eval_every=1", "train.batch_size=8",
                      "data.holdout_fraction=0.5"]},
    "assess": {"device": "cpu"},
}


def _expand(value):
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, list):
        return [_expand(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    return value


def build_loop(cfg: dict, root=None) -> ActiveLearning:
    cfg = _expand(cfg)
    root = Path(root or cfg["root"])
    budget = dict(cfg.get("budget") or {})
    store = cfg.get("store") or str(root / "_store")
    committee = Path(cfg.get("committee") or AL / "committees" / "film_committee_100k")
    if not committee.is_absolute():
        committee = AL / committee
    label = {"store": store, **(cfg.get("label") or {})}
    train = {"store": store, **(cfg.get("train") or {})}
    return ActiveLearning(
        root,
        build=PackmolBuild(**budget),
        sample=[UncertaintyDynamics(**(cfg.get("dynamics") or {})), SelectUncertain(**budget)],
        label=QChemStoreLabel(**label),
        train=CommitteeTrain(**train),
        assess=CommitteeAssess(**{**(cfg.get("assess") or {}),
                                  **({"schedule": budget["schedule"]} if "schedule" in budget else {})}),
        initial_model=committee,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--root", type=Path, help="overrides the config's root")
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(a.config.read_text()) if a.config else {}
    if a.quick:
        cfg = {**QUICK, **cfg}
    if not (a.root or cfg.get("root")):
        ap.error("--root or a config with root:")
    loop = build_loop(cfg, a.root)
    if a.quick and not cfg.get("store"):
        store = loop.root / "_store"
        store.mkdir(parents=True, exist_ok=True)
        (store / "_fake_ok").touch()          # fake_qchem.py writes only into such a store
    if a.status:
        print(loop.status())
        return 0
    result = loop.run(max_iterations=a.iterations)
    print(loop.status())
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
