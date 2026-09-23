#!/usr/bin/env python3
"""One table from a directory of udd.py runs (e.g. runs/calibration).

    python scripts/summarize_udd.py runs/calibration [--csv out.csv]

Per run: waters, replicas, bias mode, s/step, production frames, failures and retired
replicas, sigma_E per water (mHa, median and p95), sigma_F (mHa/A, median, p95), bias ratio,
mean temperature. Frames are production only; failures count every rewind.
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def row(d: Path) -> dict:
    s = json.loads((d / "summary.json").read_text())
    c = s["config"]
    reps = s["replica_status"]
    g = lambda k, q: (s.get(k) or {}).get(q)  # noqa: E731
    mode = c["bias_mode"] if c["bias_weight"] > 0 else "none"
    if mode == "raw":
        mode = f"raw k={c['bias_kappa']:g}"
    return {"run": d.name, "waters": s["waters"], "replicas": s["replicas"], "mode": mode,
            "s_per_step": s["s_per_step"], "frames": s["n_frames"],
            "failures": sum(len(r["failures"]) for r in reps),
            "retired": sum(r["status"] == "retired" for r in reps),
            "sigE_med": g("sigma_energy_per_water_mHa", "median"),
            "sigE_p95": g("sigma_energy_per_water_mHa", "p95"),
            "sigF_med": g("sigma_forces_mHa_per_A", "median"),
            "sigF_p95": g("sigma_forces_mHa_per_A", "p95"),
            "bias_ratio": g("bias_ratio", "median"), "T": g("temperature_K", "mean"),
            "reasons": sorted({x["reason"].split(" ")[0] + " " + x["reason"].split(" ")[1]
                               for r in reps for x in r["failures"]})}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--csv", type=Path)
    a = ap.parse_args(argv)
    rows = [row(d) for d in sorted(a.root.iterdir()) if (d / "summary.json").exists()]
    if not rows:
        print("no finished runs", file=sys.stderr)
        return 1
    fmt = lambda v: f"{v:.4g}" if isinstance(v, float) else str(v)  # noqa: E731
    keys = [k for k in rows[0] if k != "reasons"]
    print("  ".join(f"{k:>11s}" for k in keys))
    for r in rows:
        print("  ".join(f"{fmt(r[k]):>11s}" for k in keys) + ("  " + "; ".join(r["reasons"])
                                                             if r["reasons"] else ""))
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
