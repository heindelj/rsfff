"""Saving and loading cmm models.

A cmm checkpoint is a dict ``{"format": "cmm", "config": CmmConfig.to_dict(), "state": state_dict, "symbols": [...],
"params": <species table path>, ...}``; :func:`load_cmm` also opens complete-model checkpoints of the converged
configuration (``legacy.load_legacy``) and run tarballs (``run.tar.gz`` holding run/model.pt and the table).
"""
from __future__ import annotations

import io
import tarfile
from pathlib import Path

import torch

from .config import CmmConfig

__all__ = ["save_cmm", "load_cmm", "resolve_params"]

REPO = Path(__file__).resolve().parents[3]


def save_cmm(path, model, *, params=None, **extra):
    """Write ``model`` (state on the CPU) with its config and species list; ``extra`` keys are stored alongside."""
    ck = {"format": "cmm", "config": model.config.to_dict(),
          "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
          "symbols": list(model.tab.symbols), "params": str(params) if params else None, **extra}
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(ck, tmp)
    tmp.replace(path)
    return path


def _open(path, map_location):
    """(checkpoint dict, {name: bytes} of the other files in a run tarball)."""
    p = Path(path)
    if p.suffixes[-2:] == [".tar", ".gz"] or p.suffix in (".tgz", ".tar"):
        files = {}
        with tarfile.open(p) as tar:
            for m in tar.getmembers():
                if m.isfile():
                    files[Path(m.name).name] = tar.extractfile(m).read()
        if "model.pt" not in files:
            raise FileNotFoundError(f"{path}: no model.pt in the tarball")
        return torch.load(io.BytesIO(files.pop("model.pt")), map_location=map_location, weights_only=False), files
    return torch.load(p, map_location=map_location, weights_only=False), {}


def resolve_params(name, near=None):
    """The species table: ``name`` itself, relative to the working directory or to ``near`` and its parents, or the
    file of that name in rsfff's params/."""
    if name is None:
        raise ValueError("no species table: pass params=")
    p = Path(name)
    if p.exists():
        return p
    cands = []
    if near is not None:
        for d in [Path(near).resolve().parent, *Path(near).resolve().parents]:
            cands += [d / p, d / p.name]
    cands.append(REPO / "params" / p.name)
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(f"species table {name!r} not found (tried {[str(c) for c in cands[:4]]} ...)")


def load_cmm(path, params=None, map_location="cpu", tab=None, chan=None):
    """A :class:`CmmModel` (eval mode) from a cmm or legacy complete-model checkpoint, or a run tarball.
    ``params``: the species table (default: the checkpoint's own, or the copy in the tarball)."""
    from .model import CmmModel
    from .params import ChannelTable, IonTable
    ck, files = _open(path, map_location)
    legacy = ck.get("format") != "cmm"
    pname = params or (ck.get("args", {}).get("params") if legacy else ck.get("params"))
    tmp, ppath = None, None
    if tab is None:
        if params is None and pname and Path(pname).name in files:          # the tarball's own copy of the table
            import tempfile
            tmp = tempfile.NamedTemporaryFile("wb", prefix="cmm_table_", suffix=".json", delete=False)
            tmp.write(files[Path(pname).name])
            tmp.close()
            ppath = Path(tmp.name)          # kept: model.params_path (the precompute's table is keyed by it)
        else:
            ppath = resolve_params(pname, near=path)
        symbols = None if legacy else ck.get("symbols")
        tab, chan = IonTable.load(ppath, symbols=symbols), ChannelTable.load(ppath)
    if legacy:
        from .legacy import load_legacy
        m = load_legacy(ck, tab, chan)
    else:
        m = CmmModel(tab, chan, CmmConfig.from_dict(ck["config"]))
        m.load_state_dict(ck["state"])
    if ppath is None and pname:
        try:
            ppath = resolve_params(pname, near=path)
        except FileNotFoundError:
            ppath = None
    m.params_path = ppath
    return m.eval()
