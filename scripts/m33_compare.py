"""Compare m33 ablation runs (and a reference run, e.g. m32) from their metrics.json / train.log.

    python scripts/m33_compare.py train/nacl/ionct/m33 [train/nacl/ionct/m32_al1/al1 ...] [--csv out.csv]

Every directory holding a metrics.json (the m33 root's subdirectories, plus any run directories given) is one row:
test per-term MAE (kJ/mol per ion), dimers (n=2) vs clusters (n>=3), whole held-out clusters, held-out forces
(energy MAE per ion, force rms in kJ/mol/A), the last logged held-out alpha error per ion (bohr^3) and the last
logged geometric curvature of the environment log-factors on fixed validation frames (val_curv, per A^4)."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

TERMS = ("pol", "ct", "pauli", "disp", "int")


def _mae(d, t):
    v = d.get(f"{t}_mae") if d else None
    return float("nan") if v is None else v


def _clusters(test):
    rows = [(k, v) for k, v in test.items() if k.startswith("n=") and int(k[2:]) >= 3 and v]
    out = {}
    for t in TERMS:
        num = sum(v.get(f"{t}_mae", 0.0) * v["n"] for _, v in rows if f"{t}_mae" in v)
        den = sum(v["n"] for _, v in rows if f"{t}_mae" in v)
        out[t] = num / den if den else float("nan")
    return out


def row(run: Path):
    m = json.loads((run / "metrics.json").read_text())
    te = m["metrics"]["test"]
    r = {"run": run.name if run.name not in ("al1",) else f"{run.parent.name}/{run.name}"}
    for t in TERMS:
        r[f"test_{t}"] = _mae(te["all"], t)
    for t in ("pol", "pauli", "disp"):
        r[f"dimer_{t}"] = _mae(te.get("n=2", {}), t)
    cl = _clusters(te)
    for t in TERMS:
        r[f"clust_{t}"] = cl[t]
    for t in ("pol", "pauli", "int"):
        r[f"full_{t}"] = _mae(te.get("full", {}), t)
    ft = m["metrics"].get("force_test", {})
    r["force_e_mae"] = ft.get("e_mae", float("nan"))
    r["force_f_rms"] = ft.get("f_rms", float("nan"))
    log = (run / "train.log").read_text() if (run / "train.log").exists() else ""
    a = re.findall(r"alpha/ion \|err\| pol\|total ([0-9.]+)", log)
    r["alpha_err"] = float(a[-1]) if a else float("nan")
    curv = [e["val_curv"] for e in m.get("log", []) if "val_curv" in e]
    r["curv"] = curv[-1] if curv else float("nan")
    args = m.get("args", {})
    r["env"] = (f"{args.get('env_mode')}/{args.get('env_latent')}/2b={','.join(args.get('env_two_body') or []) or '-'}"
                if args.get("env_channels") else "heads")
    if args.get("env_field") or args.get("env_extra"):
        r["env"] += "/x=" + ",".join((["E"] if args.get("env_field") else []) + list(args.get("env_extra") or []))
    r["ablate"] = ",".join(k for k in ("xpol_kappa_const", "ct_q_const", "ct_legacy_head", "env_ct_pen") if args.get(k))
    if args.get("w_curv"):
        r["ablate"] += f",curv={args['w_curv']:g}"
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    runs = []
    for d in map(Path, a.dirs):
        if (d / "metrics.json").exists():
            runs.append(d)
        else:
            runs += sorted(p.parent for p in d.glob("*/metrics.json"))
    rows = [row(r) for r in runs]
    if not rows:
        raise SystemExit("no metrics.json found")
    cols = list(rows[0])
    show = ["run", "env", "ablate", "test_pol", "test_ct", "test_pauli", "test_disp", "test_int", "dimer_pauli",
            "clust_pol", "clust_pauli", "clust_ct", "full_int", "force_e_mae", "force_f_rms", "alpha_err", "curv"]
    w = {c: max(len(c), 7) for c in show}
    w["run"] = max(w["run"], *(len(r["run"]) for r in rows))
    w["env"] = max(w["env"], *(len(r["env"]) for r in rows))
    w["ablate"] = max(w["ablate"], *(len(r["ablate"]) for r in rows))
    print(" ".join(c.rjust(w[c]) for c in show))
    for r in rows:
        print(" ".join((f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c])).rjust(w[c]) for c in show))
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=cols)
            wr.writeheader()
            wr.writerows(rows)


if __name__ == "__main__":
    main()
