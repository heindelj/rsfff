"""The size walk: which cluster sizes each iteration samples, how many frames each labels, and
which of those also get an ALMO-EDA.

Defaults (edit here, or pass ``schedule=`` / ``eda_policy=`` to the stages):

    iter  sizes    labels/size  labeled   EDA fraction         EDA
    0     2-8         300        2100     1.0                  2100
    1     9-14        300        1800     1.0 (<=10), 0.5      1200
    2     15-20       250        1500     0.5 (15), 0.3         500
    3     21-29       200        1800     0.15                  270
    4     30-40       120        1320     0                       0
    5     41-52        70         840     0                       0
    6     53-64        55         660     0                       0
                               -------                        -----
                               10,020                         4,070

Every labeled frame gets forces (energy + gradient). EDA shrinks with size and stops at 30
waters: it is expensive, and the terms it anchors are short-ranged, so the small clusters
(plus the monomer set and the existing EDA corpus, which every fit keeps) pin them. Larger
clusters teach the many-body sum and the forces.

The number of trajectories per size follows from the label budget: ``pool_multiplier``
candidates per label, and ``candidates_per_trajectory`` candidates from one trajectory (one
per ``window_fs`` of production, so 10 ps / 100 fs = 100).
"""

from __future__ import annotations

import math

__all__ = ["DEFAULT_SCHEDULE", "eda_fraction", "entry_for", "plan_iteration", "totals"]

DEFAULT_SCHEDULE = [
    {"sizes": list(range(2, 9)), "labels_per_size": 300},
    {"sizes": list(range(9, 15)), "labels_per_size": 300},
    {"sizes": list(range(15, 21)), "labels_per_size": 250},
    {"sizes": list(range(21, 30)), "labels_per_size": 200},
    {"sizes": list(range(30, 41)), "labels_per_size": 120},
    {"sizes": list(range(41, 53)), "labels_per_size": 70},
    {"sizes": list(range(53, 65)), "labels_per_size": 55},
]

#: (upper size bound inclusive, EDA fraction); first match wins, beyond the last -> 0
DEFAULT_EDA_POLICY = [(10, 1.0), (15, 0.5), (20, 0.3), (29, 0.15)]


def eda_fraction(n_waters: int, policy=None) -> float:
    for upper, frac in (policy or DEFAULT_EDA_POLICY):
        if n_waters <= int(upper):
            return float(frac)
    return 0.0


def entry_for(iteration: int, schedule=None) -> dict:
    schedule = schedule or DEFAULT_SCHEDULE
    if iteration >= len(schedule):
        raise ValueError(f"the schedule has {len(schedule)} entries and this is iteration "
                         f"{iteration}: the size walk is finished")
    return schedule[iteration]


def plan_iteration(iteration: int, *, schedule=None, eda_policy=None, pool_multiplier=4.0,
                   candidates_per_trajectory=100, min_trajectories=2) -> list[dict]:
    """One row per size: labels, EDA labels, trajectories to run."""
    entry = entry_for(iteration, schedule)
    rows = []
    for n in entry["sizes"]:
        labels = int(entry["labels_per_size"])
        n_traj = max(int(min_trajectories),
                     math.ceil(labels * float(pool_multiplier) / float(candidates_per_trajectory)))
        rows.append({"n_waters": int(n), "labels": labels,
                     "eda": int(round(labels * eda_fraction(n, eda_policy))),
                     "trajectories": n_traj})
    return rows


def totals(schedule=None, eda_policy=None, **kw) -> dict:
    schedule = schedule or DEFAULT_SCHEDULE
    per_it = [plan_iteration(i, schedule=schedule, eda_policy=eda_policy, **kw)
              for i in range(len(schedule))]
    return {"labels": sum(r["labels"] for it in per_it for r in it),
            "eda": sum(r["eda"] for it in per_it for r in it),
            "trajectories": sum(r["trajectories"] for it in per_it for r in it),
            "per_iteration": [{"sizes": f"{it[0]['n_waters']}-{it[-1]['n_waters']}",
                               "labels": sum(r["labels"] for r in it),
                               "eda": sum(r["eda"] for r in it),
                               "trajectories": sum(r["trajectories"] for r in it)}
                              for it in per_it]}


if __name__ == "__main__":
    import json

    print(json.dumps(totals(), indent=1))
