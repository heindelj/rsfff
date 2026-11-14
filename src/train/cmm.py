"""Train a cmm model (:class:`rsfff.ff.cmm.CmmModel`) from one config file.

    python -m rsfff.train.cmm CONFIG.toml [--set train.epochs=10 loss.w_force=0.05 ...]

The config has four tables, every key optional (the defaults are the complete.default run of 2026-11-13):

    [model]   CmmConfig: hidden, depth, latent, pairs, probes, film, species_dim, env_cap, ct_q_cap, env_gate
    [data]    params (species table), ct_sets, walk_eda, salts, extra_sets, force_sets, force_data, force_test,
              polar_sets, polar_test, polar_almo, exclude_names, ct_max_pair, max_ho_ratio, ct_dimers, test
    [loss]    w_pol, w_ct, w_pauli, w_disp, w_alpha, w_alpha_ct, w_force, w_fe, w_smooth, w_curv, w_env_mid
    [train]   out, epochs, batch, lr, seed, device, cache_dir, precompute_workers, force_batch, force_every,
              alpha_every, curv_every, curv_frames, curv_delta, smooth_sigma, env_mid_scale, env_sat, val_cap,
              log_every, init_from, hull_k

Relative data paths are taken relative to ``data.root`` (default: the config file's directory).

Loss: per-ion squared errors of the ALMO-EDA pol and ct (kJ/mol; ct only where it is a channel quantity, see
``cmm_data.load_frames``), mod_pauli and disp; the whole-cluster finite-field alpha (every ``alpha_every``-th step);
total interaction energies and forces of the force frames (every ``force_every``-th step); and three regularizers on
the environment factors: the readouts' sensitivity to input noise (w_smooth), their curvature along random
displacements (w_curv) and their size where the atom's overlaps are small (w_env_mid).

Outputs in ``train.out``: model.pt (the best validation state, written as it is found; :func:`rsfff.ff.cmm.load_cmm`),
metrics.json, predictions.csv, hull.pt (the training atoms' channel hull, ``rsfff.coverage``), config.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import torch

from ..ff.cmm import CmmConfig, CmmModel
from ..ff.cmm.io import load_cmm, save_cmm
from ..ff.ionct import ct as CT
from ..ff.ionct.model import collate
from ..ff.ionct.params import ATOMIC_NUMBER, IonTable
from . import cmm_data as D

KJ, KJ_ANG = D.KJ, D.KJ_ANG


# --------------------------------------------------------------------------------------------- configuration
@dataclass
class DataConfig:
    root: str = ""                    # base of the relative paths below (default: the config file's directory)
    params: str = "params/ions_salts8_cpksb.json"
    ct_sets: str = "ct_sets"
    walk_eda: str = "walk_eda"
    salts: tuple = ("lif", "kbr", "csi")
    extra_sets: tuple = ("ct/nacl-compressed.jsonl", "ct/al-nacl-r0.jsonl", "ct/al-nacl-r1.jsonl",
                         "ct/al-nacl-lin.jsonl", "ct/al-nacl-holes.jsonl")
    force_data: tuple = ("force",)
    force_sets: tuple = ("ct/al-nacl-r0.jsonl", "ct/al-nacl-r1.jsonl", "ct_sets/seed-nacl-cuts.jsonl",
                         "ct_sets/seed-lif-cuts.jsonl", "ct_sets/seed-kbr-cuts.jsonl", "ct_sets/seed-csi-cuts.jsonl")
    force_test: tuple = ("nacl11", "al-nacl-r1:nacl_bulk64_005", "al-nacl-r1:nacl_bulk_009", "al-nacl-r1:nacl_bulk_013")
    force_test_file: str = "force_test.txt"
    polar_sets: tuple = ("ct/al-nacl-r1.jsonl",)
    polar_test: tuple = ("al-nacl-r1:nacl_polar_002", "al-nacl-r1:nacl_polar_008", "al-nacl-r1:nacl_polar_012",
                         "al-nacl-r1:nacl_polar_018", "al-nacl-r1:nacl_polar_022")
    polar_almo: bool = False          # the ALMO alpha_MI / alpha_CT targets of the ct-polar sets
    polar_ct_max_pair: float = D.POLAR_CT_MAX_PAIR
    exclude_names: tuple = ("nacl_bulk_004", "nacl_bulk_016", "nacl_bulk64_004",
                            *[f"hole_{k:03d}" for k in range(49, 61)])
    ct_max_pair: float = 3.6
    max_ho_ratio: float = 1.4
    ct_dimers: str = "like"           # all | none | like: which two-ion frames keep their ct label
    test: tuple = D.DEFAULT_TEST      # test parents (the salts' 6_03, 8_07, 9_09 clusters are added)


@dataclass
class LossConfig:
    w_pol: float = 1.0
    w_ct: float = 1.0
    w_pauli: float = 1.0
    w_disp: float = 1.0
    w_alpha: float = 0.2
    w_alpha_ct: float = 1.0
    w_force: float = 0.1
    w_fe: float = 1.0
    w_smooth: float = 0.003
    w_curv: float = 2e-3
    w_env_mid: float = 0.1


@dataclass
class RunConfig:
    out: str = "runs/cmm"
    epochs: int = 500
    batch: int = 128
    lr: float = 1e-3
    seed: int = 0
    device: str = "cpu"
    cache_dir: str = ""
    precompute_workers: int = 0
    force_batch: int = 64
    force_every: int = 5
    alpha_every: int = 4
    curv_every: int = 4
    curv_frames: int = 8
    curv_delta: float = 0.02
    smooth_sigma: float = 0.05
    env_mid_scale: float = 5e-4
    env_sat: float = 1.2              # inputs saturate at env_sat x the training range
    val_cap: float = 20.0
    log_every: int = 25
    init_from: str = ""
    hull_k: int = 6


@dataclass
class TrainConfig:
    model: CmmConfig = field(default_factory=CmmConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: RunConfig = field(default_factory=RunConfig)

    def to_dict(self):
        return {"model": self.model.to_dict(), "data": _plain(asdict(self.data)), "loss": asdict(self.loss),
                "train": asdict(self.train)}

    @classmethod
    def from_dict(cls, d):
        def mk(c, v):
            names = {f.name for f in fields(c)}
            unknown = set(v or {}) - names
            if unknown:
                raise ValueError(f"unknown {c.__name__} keys: {sorted(unknown)}")
            return c(**{k: tuple(x) if isinstance(x, list) else x for k, x in (v or {}).items()})
        unknown = set(d) - {"model", "data", "loss", "train"}
        if unknown:
            raise ValueError(f"unknown config tables: {sorted(unknown)}")
        return cls(model=CmmConfig.from_dict(d.get("model") or {}), data=mk(DataConfig, d.get("data")),
                   loss=mk(LossConfig, d.get("loss")), train=mk(RunConfig, d.get("train")))


def _plain(x):
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x


def parse_value(val: str):
    """A ``--set`` / override value: JSON where it parses, a bracketed comma list ([a, b] with bare words, as a shell
    leaves it) as a list, else the string."""
    try:
        return json.loads(val)
    except json.JSONDecodeError:
        pass
    v = val.strip()
    if v.startswith("[") and v.endswith("]"):
        return [parse_value(x.strip()) for x in v[1:-1].split(",") if x.strip()]
    return val


def load_config(path=None, sets=()) -> tuple[TrainConfig, Path]:
    """The config of ``path`` (TOML, JSON or YAML; None: the defaults) with ``sets`` ("table.key=value", values parsed
    as JSON where they parse) applied. Returns (config, the directory relative data paths are taken from)."""
    d = {}
    base = Path.cwd()
    if path is not None:
        p = Path(path)
        base = p.resolve().parent
        if p.suffix == ".toml":
            import tomllib
            d = tomllib.loads(p.read_text())
        elif p.suffix in (".yaml", ".yml"):
            import yaml
            d = yaml.safe_load(p.read_text()) or {}
        else:
            d = json.loads(p.read_text())
    for s in sets:
        key, val = s.split("=", 1)
        tbl, k = key.split(".", 1)
        d.setdefault(tbl, {})[k] = parse_value(val)
    cfg = TrainConfig.from_dict(d)
    if cfg.data.root:
        base = (base / cfg.data.root).resolve()
    return cfg, base


# --------------------------------------------------------------------------------------------- model evaluation
def loss_terms(out, bt):
    """Per-frame per-ion errors (kJ/mol) of E_pol and E_CT (zero where E_CT is not a label)."""
    lab = bt["labels"]
    n = lab["n"]
    e_pol = (out["e_pol"] - lab["pol"]) * KJ / n
    e_ct = torch.where(lab["ct_mask"], (out["e_ct"] - lab["ct"]) * KJ / n, torch.zeros_like(n))
    return e_pol, e_ct


def term_error(e, bt, key):
    """Per-frame per-ion error (kJ/mol) of an energy term against its label (zero where there is none)."""
    lab = bt["labels"]
    y = lab[key]
    return torch.where(torch.isfinite(y), (e - torch.nan_to_num(y)) * KJ / lab["n"], torch.zeros_like(y))


def polar_fields(B):
    f = torch.zeros(7, 3, dtype=torch.float64)
    for k in range(3):
        f[1 + 2 * k, k], f[2 + 2 * k, k] = D.F_SET, -D.F_SET
    return f.repeat(B, 1)


def polar_batch(caches):
    return {"bt": collate([c for c in caches for _ in range(7)]), "field": polar_fields(len(caches)), "caches": caches}


def polar_alpha(model, pb):
    """(B, 3, 2) the model's [alpha_pol, alpha_CT] per axis (7-point field stencil)."""
    B = len(pb["caches"])
    o = model(pb["bt"], field=pb["field"])
    e = torch.stack([o["e_pol"], o["e_ct"]], -1).reshape(B, 7, 2)
    return -(e[:, 1::2] + e[:, 2::2] - 2 * e[:, :1]) / D.F_SET ** 2


def polar_loss(model, pb):
    """Per-ion alpha errors (bohr^3): the first slot against alpha_MI (ALMO targets) or the whole-cluster alpha
    (total targets, compared with alpha_pol + alpha_CT), the second against alpha_CT (ALMO targets only)."""
    caches = pb["caches"]
    a = polar_alpha(model, pb)
    t = torch.stack([torch.as_tensor(c.labels["alpha"]) for c in caches]).to(a.device)
    n = torch.tensor([c.species.shape[0] for c in caches], dtype=a.dtype, device=a.device)[:, None]
    ok = torch.tensor([c.labels["ct_ok"] for c in caches], device=a.device)[:, None]
    tot = torch.tensor([bool(c.labels.get("total", False)) for c in caches], device=a.device)[:, None]
    have = torch.isfinite(t[..., 0])
    t = torch.nan_to_num(t)
    z = torch.zeros_like(a[..., 0])
    first = torch.where(tot, a[..., 0] + a[..., 1], a[..., 0])
    return (torch.where(have, (first - t[..., 0]) / n, z),
            torch.where(have & ok & ~tot, (a[..., 1] - t[..., 1]) / n, z))


def frames_batch(frames, tab, table, chan, requires_grad=False, device="cpu"):
    """(positions, batched_precompute batch) of raw frames [(symbols, coords A, ...)]."""
    from ..ff.complete import batched_precompute
    pos = torch.tensor(np.concatenate([np.asarray(X, float) for _, X, *_ in frames]), dtype=torch.float64,
                       device=device, requires_grad=requires_grad)
    Z = torch.tensor([ATOMIC_NUMBER[x] for sym, *_ in frames for x in sym], device=device)
    fr = torch.cat([torch.full((len(sym),), k, dtype=torch.long) for k, (sym, *_) in enumerate(frames)]).to(device)
    return pos, batched_precompute(pos, Z, fr, tab, table, chan, inv_form="sq")


def force_terms(model, frames, tab, table, chan):
    """Energy (per ion, kJ/mol) and force (per component, kJ/mol/A) errors of force frames, graph kept."""
    dev = table.device
    pos, bt = frames_batch(frames, tab, table, chan, requires_grad=True, device=dev)
    out = model(bt)
    E = out["e_int"]
    F = -torch.autograd.grad(E.sum(), pos, create_graph=True)[0]
    e_ref = torch.tensor([lab["int"] for _, _, lab, _ in frames], dtype=torch.float64, device=dev)
    f_ref = torch.tensor(np.concatenate([lab["forces"] for _, _, lab, _ in frames]), dtype=torch.float64, device=dev)
    n = torch.tensor([len(sym) for sym, *_ in frames], dtype=torch.float64, device=dev)
    return (E - e_ref) * KJ / n, F * KJ - f_ref * KJ_ANG, out


def force_metrics(model, frames, tab, table, chan, bs=16):
    """Held-out energy and force errors: E/ion MAE and rms (kJ/mol), force rms, MAE and max per component
    (kJ/mol/A), and per file."""
    if not frames:
        return {}
    model.eval()
    es, fs, per = [], [], {}
    for k in range(0, len(frames), bs):
        chunk = frames[k:k + bs]
        e, f, _ = force_terms(model, chunk, tab, table, chan)
        e, f = e.detach().cpu(), f.detach().cpu()
        es.append(e)
        fs.append(f)
        off = 0
        for (sym, _, _, meta), ek in zip(chunk, e):
            n = len(sym)
            d = per.setdefault(meta["file"], {"e": [], "f": []})
            d["e"].append(abs(float(ek)))
            d["f"].append(float((f[off:off + n] ** 2).mean().sqrt()))
            off += n
    model.train()
    e, f = torch.cat(es), torch.cat(fs)
    out = {"n": len(frames), "e_mae": float(e.abs().mean()), "e_rms": float((e ** 2).mean().sqrt()),
           "f_rms": float((f ** 2).mean().sqrt()), "f_mae": float(f.abs().mean()), "f_max": float(f.abs().max())}
    out["per_file"] = {k: {"e_mae": float(np.mean(v["e"])), "f_rms": float(np.sqrt(np.mean(np.square(v["f"]))))}
                       for k, v in sorted(per.items())}
    return out


def curvature_penalty(model, caches, tab, table, chan, delta=0.02):
    """Mean squared second difference (per A^2) of every environment log-factor along one random per-atom
    displacement field (unit vectors x delta A) of the given frames."""
    from ..ff.complete import batched_precompute
    dev = table.device
    pos = torch.cat([c.pos for c in caches]).to(dev, torch.float64)
    Z = torch.tensor([ATOMIC_NUMBER[tab.symbols[int(s)]] for c in caches for s in c.species], device=dev)
    fr = torch.cat([torch.full((c.species.shape[0],), k, dtype=torch.long) for k, c in enumerate(caches)]).to(dev)
    xi = torch.randn_like(pos)
    xi = xi / xi.norm(dim=1, keepdim=True).clamp(min=1e-12)
    us = []
    for sgn in (0.0, 1.0, -1.0):
        bt = batched_precompute(pos + sgn * delta * xi, Z, fr, tab, table, chan, inv_form="sq")
        us.append(dict(model.environment(bt)))
    return sum((((us[1][k] - 2 * us[0][k] + us[2][k]) / delta ** 2) ** 2).mean() for k in us[0])


def input_matrices(model, caches, bs=128):
    """(N, 17) raw channels and (N, 15) raw gated probe invariants (or None) of every atom of the caches."""
    X, G = [], []
    dev = next(model.parameters()).device
    with torch.no_grad():
        for k in range(0, len(caches), bs):
            bt = D.to_device(collate(caches[k:k + bs]), dev)
            X.append(model.channels_raw(bt).cpu())
            if model.config.probes:
                G.append(model.gated_raw(bt).cpu())
    return torch.cat(X), (torch.cat(G) if G else None)


def set_input_scales(model, caches, env_sat):
    """The encoder inputs' training scale (std over atoms with any) and saturation (env_sat x the training max)."""
    X, G = input_matrices(model, caches)
    nz = X.abs().sum(1) > 0
    model.chan_scale.copy_(X[nz].std(0).clamp(min=1e-12).to(model.chan_scale))
    if env_sat > 0:
        model.chan_sat.copy_((env_sat * (X / model.chan_scale.cpu()).abs().max(0).values).to(model.chan_sat))
    if G is not None:
        nzg = G.abs().sum(1) > 0
        model.gate_scale.copy_(G[nzg].std(0).clamp(min=1e-12).to(model.gate_scale))
        if env_sat > 0:
            model.gate_sat.copy_((env_sat * (G / model.gate_scale.cpu()).abs().max(0).values).to(model.gate_sat))
    return X


def evaluate(model, caches, bs=64):
    rows = []
    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        for k in range(0, len(caches), bs):
            chunk = caches[k:k + bs]
            bt, _ = D.pack_to_device(collate(chunk), dev)
            out = {k_: (v.cpu() if torch.is_tensor(v) else v) for k_, v in model(bt).items()}
            for q, f in enumerate(chunk):
                lab = f.labels
                rows.append({**f.meta, "pol_ref": lab["pol"] * KJ, "ct_ref": lab["ct"] * KJ,
                             "pol": out["e_pol"][q].item() * KJ, "ct": out["e_ct"][q].item() * KJ,
                             "pauli_ref": lab.get("pauli", float("nan")) * KJ, "pauli": out["e_pauli"][q].item() * KJ,
                             "disp_ref": lab.get("disp", float("nan")) * KJ, "disp": out["e_disp"][q].item() * KJ,
                             "int_ref": lab.get("int", float("nan")) * KJ,
                             "int": (lab.get("elec", float("nan")) + out["e_pauli"][q].item() + out["e_disp"][q].item()
                                     + out["e_pol"][q].item() + out["e_ct"][q].item()) * KJ,
                             "ct_mask": bool(lab["ct_mask"])})
    model.train()
    return rows


def summarize(rows, cap=20.0):
    """MAEs (capped per frame at ``cap`` kJ/mol; *_raw uncapped) of pol, ct, Pauli, dispersion and the total
    interaction energy (cls_elec label + model terms), overall, for whole clusters / MBE subsets and per size."""
    def stats(sel):
        if not sel:
            return {}
        p = np.array([r["pol"] - r["pol_ref"] for r in sel])
        c = np.array([r["ct"] - r["ct_ref"] for r in sel if r["ct_mask"]])
        cr = np.array([r["ct_ref"] for r in sel if r["ct_mask"]])
        out = {"n": len(sel), "pol_mae": D._mae(p, cap), "pol_mae_raw": float(np.abs(p).mean()),
               "pol_rms": float(np.sqrt((p ** 2).mean()))}
        big = [np.abs(p)]
        for key in ("pauli", "disp", "int"):
            d = np.array([r[key] - r[f"{key}_ref"] for r in sel if (key != "int" or r["ct_mask"])
                          and np.isfinite(r.get(f"{key}_ref", np.nan)) and np.isfinite(r.get(key, np.nan))])
            if len(d):
                out[f"{key}_mae"], out[f"{key}_mae_raw"] = D._mae(d, cap), float(np.abs(d).mean())
                if key == "pauli":
                    big.append(np.abs(d))
        if len(c):
            big.append(np.abs(c))
            out.update(ct_mae=D._mae(c, cap), ct_mae_raw=float(np.abs(c).mean()), ct_rms=float(np.sqrt((c ** 2).mean())),
                       ct_rel=float(np.abs(c).sum() / max(np.abs(cr).sum(), 1e-12)))
        if cap > 0:
            b = np.concatenate(big)
            out["n_over_cap"], out["max_err"] = int((b > cap).sum()), float(b.max()) if len(b) else 0.0
        return out
    res = {"all": stats(rows)}
    for key in ("full", "subset"):
        res[key] = stats([r for r in rows if (r["full"] if key == "full" else not r["full"])])
    for n in sorted({r["n"] for r in rows}):
        res[f"n={n}"] = stats([r for r in rows if r["n"] == n])
    return res


def radial_table(tab, params_path, cache_dir=None):
    """The :class:`RadialTable` of the precompute for this species table, cached on disk by the table's hash."""
    import hashlib
    from ..ff.complete import RadialTable
    key = hashlib.sha1(Path(params_path).read_bytes() + b"|table-v2").hexdigest()[:16]
    path = Path(cache_dir) / f"table_{key}.pt" if cache_dir else None
    if path and path.exists():
        return torch.load(path, weights_only=False)
    t0 = time.time()
    table = RadialTable(tab.to("cpu"))
    print(f"radial table: {table.n_funcs} functions x {table.G} points in {time.time() - t0:.0f}s", flush=True)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(table, path)
    return table


def write_hull(out, X_train, X_held, model, k, meta):
    """out/hull.pt: the training atoms' convex hull in the encoder's channel space; never fails the run."""
    try:
        from ..coverage import INSIDE_TOL, FeatureHull
        from ..ff.complete.channels import CHANNEL_NAMES
        hull = FeatureHull.fit(X_train.numpy(), model.chan_scale.double().cpu().numpy(), k=k,
                               feature_names=list(CHANNEL_NAMES), verbose=True, meta={**meta, "space": "env_channels"})
        hull.save(out / "hull.pt")
        if len(X_held):
            d = hull.distance(X_held.numpy())
            print(f"hull: {100 * float((d > INSIDE_TOL).mean()):.1f} % of {len(d)} held-out atoms outside "
                  f"-> {out / 'hull.pt'}", flush=True)
    except Exception as e:      # a by-product: never costs the trained model
        print(f"hull: not written ({type(e).__name__}: {e})", flush=True)


# --------------------------------------------------------------------------------------------- the run
def run(cfg: TrainConfig, base: Path):
    import os
    dc, lc, rc = cfg.data, cfg.loss, cfg.train
    P = lambda p: (Path(p) if Path(p).is_absolute() else base / p)
    out = Path(rc.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=1))
    if rc.cache_dir:
        os.environ.setdefault("RSFFF_TABLE_CACHE", str(rc.cache_dir))
    torch.manual_seed(rc.seed)
    random.seed(rc.seed)
    params = P(dc.params)
    tab, chan = IonTable.load(params), CT.ChannelTable.load(params)
    from ..ff.complete import PROBE_NAMES, DensityEnvironment
    feat = DensityEnvironment(tab, probes=PROBE_NAMES, inv_form="sq")
    ct_dimers = {"all": True, "none": False, "like": "like"}[dc.ct_dimers]
    frames = D.load_frames(P(dc.ct_sets), P(dc.walk_eda) if dc.walk_eda else None, dc.ct_max_pair, dc.max_ho_ratio,
                           species=tab.symbols, ct_dimers=ct_dimers,
                           extra_sets=[f"seed-{x}-{k}" for x in dc.salts for k in ("scan", "mbe", "sat")],
                           extra_files=[P(f) for f in dc.extra_sets], exclude=dc.exclude_names)
    test = list(dc.test) + [f"{x}{n}" for x in dc.salts for n in ("6_03", "8_07", "9_09")]
    print(f"ct label kept for {sum(f[2]['ct_mask'] for f in frames)} of {len(frames)} frames", flush=True)
    tags = D.split(frames, test, seed=rc.seed)
    print(f"{len(frames)} frames: " + ", ".join(f"{t} {tags.count(t)}" for t in ("train", "val", "test")), flush=True)
    cache_path = None
    if rc.cache_dir:
        cache_path = Path(rc.cache_dir) / f"cache_{D.cache_key(params, 'density', True, frames, False)}.pt"
    caches = D.build_cache(frames, tab, chan, feat, cache_path or out / "cache.pt", lean=True,
                           workers=rc.precompute_workers, frontier=False)
    tr = [c for c, t in zip(caches, tags) if t == "train"]
    va = [c for c, t in zip(caches, tags) if t == "val"]
    te = [c for c, t in zip(caches, tags) if t == "test"]
    model = CmmModel(tab, chan, cfg.model)
    Xtr = set_input_scales(model, tr, rc.env_sat)
    if rc.init_from:
        src = load_cmm(rc.init_from, params=params)
        missing, unexpected = model.load_state_dict(src.state_dict(), strict=False)
        print(f"init from {rc.init_from}: {len(missing)} new, {len(unexpected)} dropped", flush=True)
    dev = torch.device(rc.device)
    polar_tr, polar_te, pb_tr, pb_all = [], [], None, None
    if lc.w_alpha > 0:
        almo = (D.polar_targets(P(dc.ct_sets), ct_dimers=ct_dimers is True,
                                files=("ct-polar", *[f"seed-{x}-polar" for x in dc.salts]),
                                ct_max_pair=dc.polar_ct_max_pair) if dc.polar_almo else [])
        total = D.total_polar_targets([P(f) for f in dc.polar_sets], species=tab.symbols, exclude=dc.exclude_names)
        from ..ff.ionct.model import precompute
        for pid, sym, X, tgt, ok, is_tot in ([(*t, False) for t in almo] + [(*t, False, True) for t in total]):
            lab = {"alpha": tgt, "ct_ok": ok, **({"total": True} if is_tot else {})}
            c = precompute(sym, X, tab, chan, feat, lab, {"polar_id": pid}, lean=True, frontier=False)
            (polar_te if pid in set(dc.polar_test) else polar_tr).append(c)
        print(f"alpha targets: {len(polar_tr)} train, {len(polar_te)} held out", flush=True)
        if polar_tr:
            pb_tr = D.to_device(polar_batch(polar_tr), dev)
            pb_all = D.to_device(polar_batch(polar_tr + polar_te), dev)
    if dev.type != "cpu":
        tab = tab.to(dev)
        model.tab = tab
        model.to(dev)
    force_tr, force_te = [], []
    table = None
    if dc.force_data or dc.force_sets:
        fsets = [P(f) for f in dc.force_sets]
        mono = {**D.force_set_monomers(fsets), **D.monomer_energies([P(dc.walk_eda)] if dc.walk_eda else [])}
        ff = D.load_force_frames([P(d) for d in dc.force_data], mono) + D.load_force_sets(fsets, mono, tab.symbols,
                                                                                            dc.exclude_names)
        held = set(dc.force_test)
        if dc.force_test_file and P(dc.force_test_file).exists():
            held |= {x.strip() for x in P(dc.force_test_file).read_text().split() if x.strip()}
        force_tr = [f for f in ff if f[3]["file"] not in held]
        force_te = [f for f in ff if f[3]["file"] in held]
        print(f"force frames: {len(force_tr)} train, {len(force_te)} held out", flush=True)
    if force_tr or lc.w_curv > 0:
        table = radial_table(tab.to("cpu"), params, rc.cache_dir or None).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=rc.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(rc.epochs, 1), eta_min=rc.lr * 0.02)
    best, log, curv_log, n_bad = math.inf, [], [], 0
    t0 = time.time()
    for ep in range(1, rc.epochs + 1):
        random.shuffle(tr)
        tot = 0.0
        chunks = [tr[k:k + rc.batch] for k in range(0, len(tr), rc.batch)]
        for step, bt in enumerate(D.prefetch_batches(chunks, dev)):
            o = model(bt)
            ep_, ec_ = loss_terms(o, bt)
            loss = (lc.w_pol * ep_ ** 2 + lc.w_ct * ec_ ** 2).mean()
            if lc.w_env_mid > 0:            # right after model(bt): the model's factors are this batch's
                loss = loss + lc.w_env_mid * model.env_mid_penalty(rc.env_mid_scale)
            if lc.w_pauli > 0:
                loss = loss + lc.w_pauli * (term_error(o["e_pauli"], bt, "pauli") ** 2).mean()
            if lc.w_disp > 0:
                loss = loss + lc.w_disp * (term_error(o["e_disp"], bt, "disp") ** 2).mean()
            if pb_tr is not None and step % rc.alpha_every == 0:
                ap_, ac_ = polar_loss(model, pb_tr)
                loss = loss + rc.alpha_every * lc.w_alpha * (ap_ ** 2 + lc.w_alpha_ct * ac_ ** 2).mean()
            if lc.w_smooth > 0:
                loss = loss + lc.w_smooth * model.smoothness_penalty(bt, rc.smooth_sigma)
            if lc.w_curv > 0 and step % rc.curv_every == 0:
                cp = curvature_penalty(model, random.sample(tr, min(rc.curv_frames, len(tr))), tab, table, chan,
                                       rc.curv_delta)
                loss = loss + rc.curv_every * lc.w_curv * cp
                curv_log.append(float(cp.detach()))
            if force_tr and lc.w_force > 0 and step % rc.force_every == 0:
                fb = random.sample(force_tr, min(rc.force_batch, len(force_tr)))
                try:
                    fe_, ff_, _ = force_terms(model, fb, tab, table, chan)
                    loss = loss + rc.force_every * (lc.w_fe * (fe_ ** 2).mean() + lc.w_force * (ff_ ** 2).mean())
                except torch.linalg.LinAlgError as err:
                    print(f"ep {ep} step {step}: force step skipped ({err})", flush=True)
            opt.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            if not (torch.isfinite(loss) and torch.isfinite(gn)):
                n_bad += 1
                print(f"ep {ep} step {step}: non-finite loss / gradient (|g| {float(gn):.3g}); step skipped", flush=True)
                opt.zero_grad()
                if n_bad > 50:
                    raise RuntimeError("more than 50 non-finite steps: giving up")
                continue
            opt.step()
            tot += loss.item() * len(bt["labels"]["n"])
        sched.step()
        if ep % rc.log_every == 0 or ep == rc.epochs:
            sv = summarize(evaluate(model, va), rc.val_cap)["all"]
            score = (sv.get("pol_mae", 0) + sv.get("ct_mae", 0) + (sv.get("pauli_mae", 0) if lc.w_pauli > 0 else 0)
                     + (sv.get("disp_mae", 0) if lc.w_disp > 0 else 0))
            if score < best:
                best = score
                save_cmm(out / "model.pt", model, params=params, train=cfg.to_dict(), epoch=ep, val_score=score)
            log.append({"epoch": ep, "train_loss": tot / max(len(tr), 1), **{f"val_{k}": v for k, v in sv.items()}})
            msg = ""
            if pb_all is not None:
                with torch.no_grad():
                    ap_, ac_ = polar_loss(model, pb_all)
                msg += f" | alpha/ion |err| {ap_.abs().mean():.2f} bohr^3"
            if "pauli_mae" in sv:
                msg += f" | pauli MAE {sv['pauli_mae']:.2f} disp MAE {sv.get('disp_mae', 0):.2f}"
            if force_te:
                fm = force_metrics(model, force_te, tab, table, chan)
                msg += f" | held-out forces: E/ion MAE {fm['e_mae']:.2f} kJ/mol, F rms {fm['f_rms']:.1f} kJ/mol/A"
                log[-1].update({f"force_{k}": v for k, v in fm.items()})
            if table is not None and va:
                st = torch.random.get_rng_state()
                torch.manual_seed(1234)
                with torch.no_grad():
                    cv = float(curvature_penalty(model, va[:16], tab, table, chan, rc.curv_delta))
                torch.random.set_rng_state(st)
                log[-1]["val_curv"] = cv
                msg += f" | curv {cv:.3g}" + (f" (train {np.mean(curv_log):.3g})" if curv_log else "")
                curv_log.clear()
            print(f"ep {ep:4d} loss {tot / max(len(tr), 1):.4f} | val pol MAE {sv.get('pol_mae', 0):.2f} ct MAE "
                  f"{sv.get('ct_mae', 0):.2f} kJ/mol{msg} [{time.time() - t0:.0f}s]", flush=True)
    if (out / "model.pt").exists():
        model.load_state_dict(load_cmm(out / "model.pt", params=params).state_dict())
    else:                                                       # epochs = 0: the model as initialized
        save_cmm(out / "model.pt", model, params=params, train=cfg.to_dict(), epoch=0)
    print("cmm model:\n" + model.summary(), flush=True)
    res = {}
    with open(out / "predictions.csv", "w", newline="") as fh:
        w = None
        for name, cs in (("train", tr), ("val", va), ("test", te)):
            rows = evaluate(model, cs)
            res[name] = summarize(rows, rc.val_cap)
            for r in rows:
                r["split"] = name
                if w is None:
                    w = csv.DictWriter(fh, fieldnames=list(r))
                    w.writeheader()
                w.writerow({k: r.get(k) for k in w.fieldnames})
    if force_tr or force_te:
        res["force_train"] = force_metrics(model, force_tr, tab, table, chan)
        res["force_test"] = force_metrics(model, force_te, tab, table, chan)
    json.dump({"metrics": res, "log": log, "config": cfg.to_dict()}, open(out / "metrics.json", "w"), indent=1)
    for name in ("train", "val", "test"):
        m = res[name]["all"]
        print(f"{name:5} pol MAE {m.get('pol_mae', 0):.2f}  ct MAE {m.get('ct_mae', 0):.2f}  pauli MAE "
              f"{m.get('pauli_mae', 0):.2f}  disp MAE {m.get('disp_mae', 0):.2f} kJ/mol", flush=True)
    if rc.hull_k > 0:
        Xh = input_matrices(model, va + te)[0].cpu() if (va or te) else torch.zeros(0, 17)
        write_hull(out, Xtr, Xh, model, rc.hull_k, {"run": str(out), "params": str(params), "n_train_frames": len(tr)})
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m rsfff.train.cmm", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default=None, help="TOML / JSON / YAML config (default: all defaults)")
    ap.add_argument("--set", nargs="*", default=[], metavar="TABLE.KEY=VALUE", help="overrides, e.g. train.epochs=10")
    ap.add_argument("--print-config", action="store_true", help="print the resolved config and exit")
    a = ap.parse_args(argv)
    cfg, base = load_config(a.config, a.set)
    if a.print_config:
        print(json.dumps(cfg.to_dict(), indent=1))
        return 0
    run(cfg, base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
