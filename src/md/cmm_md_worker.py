"""Condensed-phase runs of a cmm model behind ChemLab's worker protocol: the ``md.cmm`` job.

    python -m rsfff.md.cmm_md_worker run WORKDIR

Inputs: ``in/model`` (a cmm checkpoint or run tarball, :func:`rsfff.ff.cmm.load_cmm`), ``in/settings``:
``task`` -- the program, a module of :mod:`rsfff.md` with a ``main(argv)`` (``cmm_bench``, ...) -- and ``args``,
its command line after the model (shell-quoted). The program writes ``--json out/result.json``; its stdout and
stderr go to ``out/md.log``. The report carries the task, the device and the result's ``summary`` (if any).
"""
from __future__ import annotations

import importlib
import json
import os
import shlex
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

TASKS = {"cmm_bench", "cmm_melt"}


def _settings(workdir: Path) -> dict:
    p = workdir / "in" / "settings"
    if not p.exists():
        return {}
    out = {}

    def flat(d):
        for k, v in d.items():
            if isinstance(v, dict):
                flat(v)
            else:
                out[k] = v

    flat(json.loads(p.read_text()))
    return out


def run_job(workdir) -> int:
    workdir = Path(workdir).resolve()
    out = workdir / "out"
    out.mkdir(exist_ok=True)
    s = _settings(workdir)
    task = str(s.get("task") or "cmm_bench")
    if task not in TASKS:
        raise ValueError(f"md.cmm: task {task!r} not in {sorted(TASKS)}")
    model = workdir / "in" / "model"
    if not model.exists():
        raise ValueError("md.cmm: no model input")
    # load_cmm tells a tarball by its suffix
    link = workdir / "model.tar"
    if not link.exists():
        link.symlink_to(model)
    argv = [str(link)] + shlex.split(str(s.get("args") or "")) + ["--json", str(out / "result.json")]
    dev = str(s.get("device") or "auto")
    if dev == "auto":
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    if "--device" not in argv:
        argv += ["--device", dev]
    cache = os.environ.get("RSFFF_CACHE_DIR")
    if cache and "--table-cache" not in argv:
        argv += ["--table-cache", cache]
    t0 = time.time()
    mod = importlib.import_module(f"rsfff.md.{task}")
    with open(out / "md.log", "w") as fh, redirect_stdout(fh), redirect_stderr(fh):
        print(f"{task} {' '.join(shlex.quote(a) for a in argv)}", flush=True)
        mod.main(argv)
    res = json.loads((out / "result.json").read_text()) if (out / "result.json").exists() else {}
    report = {"md.task": task, "md.device": dev, "md.wall_s": round(time.time() - t0, 1),
              "md.summary": res.get("summary") or {"rows": len(res.get("rows", []))}}
    (workdir / "report.json").write_text(json.dumps({"data": report}))
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] in (["run"], ["verify"]) and len(argv) == 2:
        try:
            return run_job(argv[1])
        except Exception as exc:  # the protocol: a nonzero exit and the reason on stderr
            traceback.print_exc()
            sys.stderr.write(f"rsfff.md.cmm_md_worker: {exc}\n")
            return 1
    sys.stderr.write(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
