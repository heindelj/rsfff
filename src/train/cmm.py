"""Train a cmm model (:class:`rsfff.ff.cmm.CmmModel`) from one config file.

    python -m rsfff.train.cmm CONFIG.toml [--set train.epochs=10 loss.w_force=0.05 ...]

The config has four tables, every key optional (the defaults are the complete.default run of 2026-11-13):

    [model]   CmmConfig: hidden, depth, latent, pairs, probes, film, species_dim, env_cap, ct_q_cap, xpol_bound,
              disp_model, disp_confine, disp_table, hyperpol, hyper_ceiling, ct_model, ct_gap_overlap, ct_gap_w0,
              ct_gap_predict, ct_gap_converge, ct_gap_tol, ct_exclusion, ct_q_onesided, ct_q_headroom, ct_floor,
              ct_relay, env_gate, env_overlap, env_switch, ct_kernel, ct_cut, ct_overlap_table, ct_overlap_window
    [data]    params (species table), ct_sets, walk_eda, salts, extra_sets, force_sets, force_data, force_test,
              polar_sets, polar_test, polar_almo, exclude_names, ct_max_pair, max_ho_ratio, ct_dimers, ct_min_gap,
              pair_weight, dipole_charged, dip_min_gap,
              test
    [loss]    w_pol, w_ct, w_pauli, w_disp, w_frz, w_int, frz_elst, w_alpha, w_alpha_ct, w_force, w_fe, w_smooth, w_curv,
              w_env_mid, w_dip
    [train]   out, epochs, batch, lr, seed, device, cache_dir, force_batch, force_every, dip_every,
              alpha_every, curv_every, curv_frames, curv_delta, smooth_sigma, env_mid_scale, env_sat, val_cap,
              log_every, init_from, hull_k

Relative data paths are taken relative to ``data.root`` (default: the config file's directory).

Loss: per-ion squared errors of the ALMO-EDA pol and ct (kJ/mol; ct only where it is a channel quantity, see
``cmm_data.load_frames``), mod_pauli and disp; the whole-cluster polarizability (exact, from the model's linear
response; every ``alpha_every``-th step);
total interaction energies and forces of the force frames (every ``force_every``-th step); and three regularizers on
the environment factors: the readouts' sensitivity to input noise (w_smooth), their curvature along random
displacements (w_curv) and their size where the atom's overlaps are small (w_env_mid).

Training without the EDA's partition (``w_pol = w_ct = w_pauli = w_disp = 0``, ``w_frz`` and ``w_int`` > 0): the labelled
frames give only the frozen energy minus electrostatics, against E_Pauli + E_disp (``w_frz``; ``frz_elst = "model"``
takes the model's own frozen-density electrostatics off the EDA's frozen energy, so that this target and the total
energy are one consistent pair; ``"eda"`` takes the EDA's cls_elec off, i.e. mod_pauli + disp), and the total
interaction energy (``w_int``, with the model's own electrostatics); the force frames and the polarizabilities as
before. Every two-ion frame counts there, the unlike pairs through the ionic / radical crossing included (the ct
masks of ``data`` only decide where E_CT itself is a label). The EDA's pol, ct, Pauli and dispersion are still
written to predictions.csv and summarized: with these weights they are a test, not a fit. The best validation state
is chosen on the terms that are fitted.

Dipole moments (``w_dip`` > 0; every ``dip_every``-th step): the frames whose record has a total SCF dipole
(``cmm_data.dipole_label``: ``dipole`` in Debye in the frame of the coordinates, ``dipole_origin`` in Angstrom) give
the induced dipole, the total minus the ions' point charges', against the model's -dE/dF at zero field (its induced
dipoles plus the transferred charges times their positions; exact, by the same double backward as the force loss).
Squared error per ion in Debye, summed over the components, with the frames' weights; frames in the crossing region
(vacuum gap below ``data.dip_min_gap``) are left out. It is a label-free handle on
the split of E_pol + E_CT: the two enter the dipole in another proportion than the energy.

Every batch is precomputed from positions on the training device (:meth:`CmmModel.precompute`; no per-frame cache);
``train.cache_dir`` keeps the radial table on disk. Outputs in ``train.out``: model.pt (the best validation state,
written as it is found; :func:`rsfff.ff.cmm.load_cmm`), metrics.json, predictions.csv, hull.pt (the training atoms' channel hull, ``rsfff.coverage``), config.json.
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
from ..ff.cmm.io import load_cmm, save_cmm, save_table
from ..ff.cmm.params import ChannelTable, IonTable
from . import cmm_data as D

KJ, KJ_ANG = D.KJ, D.KJ_ANG


# --------------------------------------------------------------------------------------------- configuration
@dataclass
class DataConfig:
    root: str = ""                    # base of the relative paths below (default: the config file's directory)
    params: str = "params/ions_salts8_cpksb_dipresp.json"       # dipole parents from the response density
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
    ct_min_gap: float = 0.0           # Hartree; > 0 drops the ct label of frames whose vacuum gap is below it (0.04)
    dipole_charged: bool = True       # the dipole labels of charged frames too (about their dipole_origin); false:
                                      # neutral frames only
    dip_min_gap: float = 0.04         # dipole labels count only in frames whose vacuum gap (cmm_data.vacuum_gap) is
                                      # at least this (Hartree; 0: all): not in the crossing region, where the SCF
                                      # dipole is that of a transferred electron
    pair_weight: float = 1.5          # per-pair overlap switch (frontier kernel, ct_cut = "overlap"): a two-ion frame
                                      # counts fully while its pair's overlap is above pair_weight x s_on and not at
                                      # all from where the pair's switch starts
                                      # (cmm_data.set_pair_weights); 0: every frame counts
    test: tuple = D.DEFAULT_TEST      # test parents (the salts' 6_03, 8_07, 9_09 clusters are added)


@dataclass
class LossConfig:
    w_pol: float = 1.0
    w_ct: float = 1.0
    w_pauli: float = 1.0
    w_disp: float = 1.0
    w_frz: float = 0.0                # E_Pauli + E_disp against the frozen energy minus electrostatics (per ion)
    w_int: float = 0.0                # the total interaction energy of the labelled frames (per ion; own electrostatics)
    frz_elst: str = "model"           # the electrostatics taken off the frozen energy: "model" (the model's) | "eda" (cls_elec)
    w_alpha: float = 0.2
    w_alpha_ct: float = 1.0
    w_force: float = 0.1
    w_fe: float = 1.0
    w_smooth: float = 0.003
    w_curv: float = 2e-3
    w_env_mid: float = 0.1
    w_dip: float = 0.0                # the induced dipole of the frames with a dipole label (per ion, Debye)


@dataclass
class RunConfig:
    out: str = "runs/cmm"
    epochs: int = 500
    batch: int = 128
    lr: float = 1e-3
    seed: int = 0
    device: str = "cpu"
    cache_dir: str = ""
    force_batch: int = 64
    force_every: int = 5
    dip_every: int = 1                # the dipole loss on every dip_every-th step (it costs a second backward)
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
def wmean(x, w):
    """Mean over the batch of the frames' x weighted by their loss weights w (the plain mean when all are 1; frames
    of weight 0 contribute nothing and still count in the denominator, as a masked label does)."""
    return (x * (w if x.dim() == 1 else w[:, None])).mean()


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


def frozen_error(out, bt, elst="model"):
    """Per-frame per-ion error (kJ/mol) of E_Pauli + E_disp against the EDA's frozen energy minus electrostatics:
    mod_pauli + disp (``elst="eda"``: minus the EDA's own cls_elec), or cls_elec + mod_pauli + disp minus the model's
    frozen-density electrostatics (``"model"``). Zero where a label is missing."""
    lab = bt["labels"]
    y = lab["pauli"] + lab["disp"]
    if elst == "model":
        y = y + lab["elec"] - out["e_elst"].detach()
    elif elst != "eda":
        raise ValueError(f"frz_elst {elst!r}: 'model' or 'eda'")
    e = out["e_pauli"] + out["e_disp"]
    return torch.where(torch.isfinite(y), (e - torch.nan_to_num(y)) * KJ / lab["n"], torch.zeros_like(e))


def val_score(sv, lc) -> float:
    """The validation score that picks the saved state: the sum of the MAEs of the fitted energy terms. Without the
    frozen / total losses it is the EDA terms' (pol and ct always, Pauli and dispersion when weighted). With the
    dipole loss the dipole MAE is added, 0.1 Debye counting as 1 kJ/mol."""
    dip = DIP_SCORE * sv.get("dip_mae", 0) if lc.w_dip > 0 else 0.0
    if lc.w_frz <= 0 and lc.w_int <= 0:
        return (sv.get("pol_mae", 0) + sv.get("ct_mae", 0) + (sv.get("pauli_mae", 0) if lc.w_pauli > 0 else 0)
                + (sv.get("disp_mae", 0) if lc.w_disp > 0 else 0)) + dip
    terms = (("pol", lc.w_pol), ("ct", lc.w_ct), ("pauli", lc.w_pauli), ("disp", lc.w_disp), ("frz", lc.w_frz),
             ("etot", lc.w_int))
    return sum(sv.get(f"{k}_mae", 0) for k, w in terms if w > 0) + dip


DIP_SCORE = 10.0                      # val_score: kJ/mol per Debye of dipole MAE


def model_dipole(model, bt, create_graph=False):
    """(out, mu): the forward in a zero uniform field and mu = -dE_int/dF (B, 3) (e bohr), the model's induced dipole
    per frame: its induced dipoles plus the transferred charges times their positions (the permanent charges do not
    couple to the field in E_int). The energies of ``out`` are those of ``model(bt)``. ``create_graph``: keep the
    graph of mu for a loss on it."""
    with torch.enable_grad():
        F = torch.zeros(int(bt["B"]), 3, dtype=torch.float64, device=bt["pos"].device, requires_grad=True)
        out = model(bt, field=F)
        mu = -torch.autograd.grad(out["e_int"].sum(), F, create_graph=create_graph)[0]
    return out, mu


def dipole_error(mu, bt):
    """(B, 3) the per-ion error (Debye) of the model's induced dipole against the frames' dipole labels
    (``labels["dip"]``), zero where there is none."""
    lab = bt["labels"]
    y = lab["dip"]
    return torch.where(torch.isfinite(y), (mu - torch.nan_to_num(y)) * D.DEBYE / lab["n"][:, None], torch.zeros_like(y))


def polar_alpha(model, frames):
    """(B, 3, 2) the model's [alpha_pol, alpha_CT] per axis (the diagonal of its exact polarizabilities)."""
    _, bt = D.batch_of(frames, model)
    o = model(bt, alpha=True)
    ap = torch.diagonal(o["alpha_pol"], dim1=1, dim2=2)
    return torch.stack([ap, torch.diagonal(o["alpha"], dim1=1, dim2=2) - ap], -1)


def polar_loss(model, frames):
    """Per-ion alpha errors (bohr^3): the first slot against alpha_MI (ALMO targets) or the whole-cluster alpha
    (total targets, compared with alpha_pol + alpha_CT), the second against alpha_CT (ALMO targets only)."""
    a = polar_alpha(model, frames)
    t = torch.stack([torch.as_tensor(f.labels["alpha"]) for f in frames]).to(a.device)
    n = torch.tensor([f.n for f in frames], dtype=a.dtype, device=a.device)[:, None]
    ok = torch.tensor([f.labels["ct_ok"] for f in frames], device=a.device)[:, None]
    tot = torch.tensor([bool(f.labels.get("total", False)) for f in frames], device=a.device)[:, None]
    have = torch.isfinite(t[..., 0])
    t = torch.nan_to_num(t)
    z = torch.zeros_like(a[..., 0])
    first = torch.where(tot, a[..., 0] + a[..., 1], a[..., 0])
    sw = D.frame_weights(frames, a.device).sqrt()[:, None]            # the errors enter squared
    return (sw * torch.where(have, (first - t[..., 0]) / n, z),
            sw * torch.where(have & ok & ~tot, (a[..., 1] - t[..., 1]) / n, z))


def force_terms(model, frames):
    """Energy (per ion, kJ/mol) and force (per component, kJ/mol/A) errors of force frames, graph kept."""
    pos, bt = D.batch_of(frames, model, requires_grad=True)
    dev = pos.device
    out = model(bt)
    E = out["e_int"]
    F = -torch.autograd.grad(E.sum(), pos, create_graph=True)[0]
    e_ref = torch.tensor([f.labels["int"] for f in frames], dtype=torch.float64, device=dev)
    f_ref = torch.as_tensor(np.concatenate([f.labels["forces"] for f in frames]), dtype=torch.float64).to(dev)
    n = torch.tensor([f.n for f in frames], dtype=torch.float64, device=dev)
    sw = D.frame_weights(frames, dev).sqrt()                          # the errors enter squared
    return (sw * (E - e_ref) * KJ / n, sw.repeat_interleave(n.long())[:, None] * (F * KJ - f_ref * KJ_ANG), out)


def force_metrics(model, frames, bs=16):
    """Held-out energy and force errors: E/ion MAE and rms (kJ/mol), force rms, MAE and max per component
    (kJ/mol/A), and per file."""
    if not frames:
        return {}
    model.eval()
    es, fs, per = [], [], {}
    for k in range(0, len(frames), bs):
        chunk = frames[k:k + bs]
        e, f, _ = force_terms(model, chunk)
        e, f = e.detach().cpu(), f.detach().cpu()
        es.append(e)
        fs.append(f)
        off = 0
        for fk, ek in zip(chunk, e):
            n = fk.n
            d = per.setdefault(fk.meta["file"], {"e": [], "f": []})
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


def curvature_penalty(model, frames, delta=0.02):
    """Mean squared second difference (per A^2) of every environment log-factor along one random per-atom
    displacement field (unit vectors x delta A) of the given frames."""
    dev = next(model.parameters()).device
    pos = torch.cat([f.pos for f in frames]).to(dev)
    Z = torch.cat([f.Z for f in frames]).to(dev)
    fr = torch.cat([torch.full((f.n,), k, dtype=torch.long) for k, f in enumerate(frames)]).to(dev)
    xi = torch.randn_like(pos)
    xi = xi / xi.norm(dim=1, keepdim=True).clamp(min=1e-12)
    us = []
    for sgn in (0.0, 1.0, -1.0):
        with torch.no_grad():
            bt = model.precompute(pos + sgn * delta * xi, Z, fr)
        us.append(dict(model.environment(bt)))
    return sum((((us[1][k] - 2 * us[0][k] + us[2][k]) / delta ** 2) ** 2).mean() for k in us[0])


def input_matrices(model, frames, bs=128):
    """(N, 17) raw channels and (N, 15) raw gated probe invariants (or None) of every atom of the frames."""
    X, G = [], []
    for k in range(0, len(frames), bs):
        _, bt = D.batch_of(frames[k:k + bs], model)
        X.append(bt["chan"].cpu())
        if model.config.probes:
            G.append(bt["gated"].cpu())
    return torch.cat(X), (torch.cat(G) if G else None)


def set_input_scales(model, frames, env_sat):
    """The encoder inputs' training scale (std over atoms with any) and saturation (env_sat x the training max)."""
    X, G = input_matrices(model, frames)
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


def evaluate(model, frames, bs=64, frz_elst="model"):
    """One row per frame (kJ/mol, frame totals): the model's terms and their labels. ``int`` is the EDA's cls_elec +
    the model's other terms; ``etot`` the model's own total (its frozen-density electrostatics, ``elst``); ``frz``
    E_Pauli + E_disp against the frozen energy minus electrostatics (:func:`frozen_error`)."""
    rows = []
    model.eval()
    with torch.no_grad():
        for k in range(0, len(frames), bs):
            chunk = frames[k:k + bs]
            _, bt = D.batch_of(chunk, model)
            has_dip = "dip" in bt["labels"]
            out, mu = model_dipole(model, bt) if has_dip else (model(bt), None)
            out = {k_: (v.detach().cpu() if torch.is_tensor(v) else v) for k_, v in out.items()}
            if has_dip:
                mu, mu_ref = mu.cpu().numpy() * D.DEBYE, bt["labels"]["dip"].cpu().numpy() * D.DEBYE
            for q, f in enumerate(chunk):
                lab = f.labels
                nan = float("nan")
                dp = ({f"dip{c}": float(mu[q, a]) for a, c in enumerate("xyz")}
                      | {f"dip{c}_ref": float(mu_ref[q, a]) for a, c in enumerate("xyz")}
                      | {"dip_err": float(np.linalg.norm(mu[q] - mu_ref[q]))}) if has_dip else \
                    {k_: nan for k_ in ("dipx", "dipy", "dipz", "dipx_ref", "dipy_ref", "dipz_ref", "dip_err")}
                el, el_ref = out["e_elst"][q].item(), lab.get("elec", nan)
                frz_ref = lab.get("pauli", nan) + lab.get("disp", nan) + ((el_ref - el) if frz_elst == "model" else 0.0)
                rows.append({**f.meta, "pol_ref": lab["pol"] * KJ, "ct_ref": lab["ct"] * KJ,
                             "pol": out["e_pol"][q].item() * KJ, "ct": out["e_ct"][q].item() * KJ,
                             "pauli_ref": lab.get("pauli", float("nan")) * KJ, "pauli": out["e_pauli"][q].item() * KJ,
                             "disp_ref": lab.get("disp", float("nan")) * KJ, "disp": out["e_disp"][q].item() * KJ,
                             "int_ref": lab.get("int", float("nan")) * KJ,
                             "int": (lab.get("elec", float("nan")) + out["e_pauli"][q].item() + out["e_disp"][q].item()
                                     + out["e_pol"][q].item() + out["e_ct"][q].item()) * KJ,
                             "elst_ref": el_ref * KJ, "elst": el * KJ,
                             "frz_ref": frz_ref * KJ, "frz": (out["e_pauli"][q].item() + out["e_disp"][q].item()) * KJ,
                             "etot_ref": lab.get("int", nan) * KJ, "etot": out["e_int"][q].item() * KJ,
                             "ct_mask": bool(lab["ct_mask"]), "weight": float(lab.get("weight", 1.0)), **dp})
    model.train()
    return rows


def summarize(rows, cap=20.0):
    """MAEs (capped per frame at ``cap`` kJ/mol; *_raw uncapped) of pol, ct, Pauli, dispersion, the total
    interaction energy (``int``: cls_elec label + model terms, where E_CT is a label; ``etot``: the model's own total,
    every frame) and the frozen energy minus electrostatics (``frz``), overall, for whole clusters / MBE subsets and
    per size. The *_mae are weighted by the frames' loss weights (``weight``; frames of weight 0, the two-ion frames
    beyond their pair's switch, do not count: ``n_w`` is the sum of the weights), *_mae_raw and the rms are not."""
    def stats(sel):
        if not sel:
            return {}
        wt = lambda rs: np.array([r.get("weight", 1.0) for r in rs])       # noqa: E731
        p = np.array([r["pol"] - r["pol_ref"] for r in sel])
        cs = [r for r in sel if r["ct_mask"]]
        c = np.array([r["ct"] - r["ct_ref"] for r in cs])
        cr = np.array([r["ct_ref"] for r in cs])
        out = {"n": len(sel), "n_w": float(wt(sel).sum()), "pol_mae": D._mae(p, cap, wt(sel)),
               "pol_mae_raw": float(np.abs(p).mean()), "pol_rms": float(np.sqrt((p ** 2).mean()))}
        big = [np.abs(p)]
        for key in ("pauli", "disp", "int", "frz", "etot"):
            ks = [r for r in sel if (key != "int" or r["ct_mask"])
                  and np.isfinite(r.get(f"{key}_ref", np.nan)) and np.isfinite(r.get(key, np.nan))]
            d = np.array([r[key] - r[f"{key}_ref"] for r in ks])
            if len(d):
                out[f"{key}_mae"], out[f"{key}_mae_raw"] = D._mae(d, cap, wt(ks)), float(np.abs(d).mean())
                if key == "pauli":
                    big.append(np.abs(d))
        ds = [r for r in sel if np.isfinite(r.get("dip_err", np.nan))]
        if ds:                              # |model - reference| of the induced dipole per frame (Debye), not capped
            de = np.array([r["dip_err"] for r in ds])
            out.update(dip_n=len(ds), dip_mae=D._mae(de, 0.0, wt(ds)), dip_mae_raw=float(de.mean()), dip_max=float(de.max()))
        if len(c):
            big.append(np.abs(c))
            out.update(ct_mae=D._mae(c, cap, wt(cs)), ct_mae_raw=float(np.abs(c).mean()),
                       ct_rms=float(np.sqrt((c ** 2).mean())),
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


def write_hull(out, X_train, X_held, model, k, meta):
    """out/hull.pt: the training atoms' convex hull in the encoder's channel space; never fails the run."""
    try:
        from ..coverage import INSIDE_TOL, FeatureHull
        from ..ff.cmm.channels import CHANNEL_NAMES
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
    tab, chan = IonTable.load(params), ChannelTable.load(params)
    ct_dimers = {"all": True, "none": False, "like": "like"}[dc.ct_dimers]
    frames = D.load_frames(P(dc.ct_sets), P(dc.walk_eda) if dc.walk_eda else None, dc.ct_max_pair, dc.max_ho_ratio,
                           species=tab.symbols, ct_dimers=ct_dimers, ct_min_gap=dc.ct_min_gap, ions=D.gap_ions(tab),
                           extra_sets=[f"seed-{x}-{k}" for x in dc.salts for k in ("scan", "mbe", "sat")],
                           extra_files=[P(f) for f in dc.extra_sets], exclude=dc.exclude_names,
                           dipole_charged=dc.dipole_charged)
    test = list(dc.test) + [f"{x}{n}" for x in dc.salts for n in ("6_03", "8_07", "9_09")]
    print(f"ct label kept for {sum(f[2]['ct_mask'] for f in frames)} of {len(frames)} frames", flush=True)
    if dc.ct_min_gap > 0:
        print(f"  vacuum gap below {dc.ct_min_gap} Hartree in {sum(f[3]['vacuum_gap'] < dc.ct_min_gap for f in frames)} frames",
              flush=True)
    tags = D.split(frames, test, seed=rc.seed)
    print(f"{len(frames)} frames: " + ", ".join(f"{t} {tags.count(t)}" for t in ("train", "val", "test")), flush=True)
    frames = D.as_frames(frames)
    tr = [f for f, t in zip(frames, tags) if t == "train"]
    va = [f for f, t in zip(frames, tags) if t == "val"]
    te = [f for f, t in zip(frames, tags) if t == "test"]
    dev = torch.device(rc.device)
    model = CmmModel(tab, chan, cfg.model)
    if dev.type != "cpu":
        model.tab = tab = tab.to(dev)
        model.to(dev)
    Xtr = set_input_scales(model, tr, rc.env_sat)
    if rc.init_from:
        src = load_cmm(rc.init_from, params=params)
        missing, unexpected = model.load_state_dict(src.state_dict(), strict=False)
        print(f"init from {rc.init_from}: {len(missing)} new, {len(unexpected)} dropped", flush=True)
    polar_tr, polar_te = [], []
    if lc.w_alpha > 0:
        almo = (D.polar_targets(P(dc.ct_sets), ct_dimers=ct_dimers is True,
                                files=("ct-polar", *[f"seed-{x}-polar" for x in dc.salts]),
                                ct_max_pair=dc.polar_ct_max_pair) if dc.polar_almo else [])
        total = D.total_polar_targets([P(f) for f in dc.polar_sets], species=tab.symbols, exclude=dc.exclude_names)
        for pid, sym, X, tgt, ok, is_tot in ([(*t, False) for t in almo] + [(*t, False, True) for t in total]):
            lab = {"alpha": tgt, "ct_ok": ok, **({"total": True} if is_tot else {})}
            c = D.as_frame(sym, X, lab, {"polar_id": pid})
            (polar_te if pid in set(dc.polar_test) else polar_tr).append(c)
        print(f"alpha targets: {len(polar_tr)} train, {len(polar_te)} held out", flush=True)
    force_tr, force_te = [], []
    if dc.force_data or dc.force_sets:
        fsets = [P(f) for f in dc.force_sets]
        mono = {**D.force_set_monomers(fsets), **D.monomer_energies([P(dc.walk_eda)] if dc.walk_eda else [])}
        ff = D.as_frames(D.load_force_frames([P(d) for d in dc.force_data], mono)
                         + D.load_force_sets(fsets, mono, tab.symbols, dc.exclude_names))
        held = set(dc.force_test)
        if dc.force_test_file and P(dc.force_test_file).exists():
            held |= {x.strip() for x in P(dc.force_test_file).read_text().split() if x.strip()}
        force_tr = [f for f in ff if f.meta["file"] not in held]
        force_te = [f for f in ff if f.meta["file"] in held]
        print(f"force frames: {len(force_tr)} train, {len(force_te)} held out", flush=True)
    # the radial table: every species pair of the data, built (or taken from the disk cache) now and shipped as
    # table.pt with the run
    t1 = time.time()
    allf = tr + va + te + force_tr + force_te + polar_tr + polar_te
    used = torch.unique(torch.cat([tab.species_index(f.Z.to(tab.Z.device)).cpu() for f in allf]))
    model.table().ensure_species(used)
    save_table(out / "table.pt", model)
    print(f"radial table: {len(used) ** 2} species pairs x {model.table().n_funcs} functions x {model.table().G} "
          f"points [{time.time() - t1:.0f}s]", flush=True)
    n_off = D.mask_dipoles(tr + va + te, dc.dip_min_gap)
    dr = D.dipole_report(tr + va + te, tab)
    if dr["n"] or lc.w_dip > 0:
        print(f"dipole labels: {dr['n']} of {len(tr) + len(va) + len(te)} frames ({dr['pairs']} two-ion; {n_off} more left "
              f"out, vacuum gap below {dc.dip_min_gap:g} Ha); induced dipole "
              f"per ion up to {dr['per_ion']:.2f} D; largest off-axis component of a pair {dr['off_axis']:.3f} D"
              + ("  <-- not zero: check the frame and the origin of the dipoles" if dr["off_axis"] > 0.02 else "")
              + ("" if dc.dipole_charged else
                 f"; {sum(bool(f[3].get('dipole_dropped')) for f in frames)} charged frames' dipoles dropped"),
              flush=True)
        if lc.w_dip > 0 and not dr["n"]:
            raise ValueError("w_dip > 0 but no record of the sets has a dipole (cmm_data.dipole_label)")
    if model.overlap_cut and dc.pair_weight > 0:
        pw = D.set_pair_weights(model, allf, dc.pair_weight)
        print(f"pair weights: {pw['pairs']} two-ion frames, {pw['reduced']} with a reduced weight, {pw['dropped']} "
              f"beyond their pair's switch (weight 0)", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=rc.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(rc.epochs, 1), eta_min=rc.lr * 0.02)
    best, log, curv_log, n_bad = math.inf, [], [], 0
    t0 = time.time()
    for ep in range(1, rc.epochs + 1):
        random.shuffle(tr)
        tot = 0.0
        chunks = [tr[k:k + rc.batch] for k in range(0, len(tr), rc.batch)]
        for step, chunk in enumerate(chunks):
            _, bt = D.batch_of(chunk, model)
            dip = lc.w_dip > 0 and "dip" in bt["labels"] and step % rc.dip_every == 0
            o, mu = model_dipole(model, bt, create_graph=True) if dip else (model(bt), None)
            ep_, ec_ = loss_terms(o, bt)
            fw = bt["labels"]["weight"]
            loss = wmean(lc.w_pol * ep_ ** 2 + lc.w_ct * ec_ ** 2, fw)
            if dip:
                loss = loss + rc.dip_every * lc.w_dip * wmean((dipole_error(mu, bt) ** 2).sum(1), fw)
            if lc.w_env_mid > 0:            # right after model(bt): the model's factors are this batch's
                loss = loss + lc.w_env_mid * model.env_mid_penalty(rc.env_mid_scale)
            if lc.w_pauli > 0:
                loss = loss + lc.w_pauli * wmean(term_error(o["e_pauli"], bt, "pauli") ** 2, fw)
            if lc.w_disp > 0:
                loss = loss + lc.w_disp * wmean(term_error(o["e_disp"], bt, "disp") ** 2, fw)
            if lc.w_frz > 0:
                loss = loss + lc.w_frz * wmean(frozen_error(o, bt, lc.frz_elst) ** 2, fw)
            if lc.w_int > 0:
                loss = loss + lc.w_int * wmean(term_error(o["e_int"], bt, "int") ** 2, fw)
            if polar_tr and step % rc.alpha_every == 0:
                ap_, ac_ = polar_loss(model, polar_tr)
                loss = loss + rc.alpha_every * lc.w_alpha * (ap_ ** 2 + lc.w_alpha_ct * ac_ ** 2).mean()
            if lc.w_smooth > 0:
                loss = loss + lc.w_smooth * model.smoothness_penalty(bt, rc.smooth_sigma)
            if lc.w_curv > 0 and step % rc.curv_every == 0:
                cp = curvature_penalty(model, random.sample(tr, min(rc.curv_frames, len(tr))), rc.curv_delta)
                loss = loss + rc.curv_every * lc.w_curv * cp
                curv_log.append(float(cp.detach()))
            if force_tr and lc.w_force > 0 and step % rc.force_every == 0:
                fb = random.sample(force_tr, min(rc.force_batch, len(force_tr)))
                try:
                    fe_, ff_, _ = force_terms(model, fb)
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
            sv = summarize(evaluate(model, va, frz_elst=lc.frz_elst), rc.val_cap)["all"]
            score = val_score(sv, lc)
            if score < best:
                best = score
                save_cmm(out / "model.pt", model, params=params, train=cfg.to_dict(), epoch=ep, val_score=score)
            log.append({"epoch": ep, "train_loss": tot / max(len(tr), 1), **{f"val_{k}": v for k, v in sv.items()}})
            msg = ""
            if polar_tr:
                with torch.no_grad():
                    ap_, ac_ = polar_loss(model, polar_tr + polar_te)
                msg += f" | alpha/ion |err| {ap_.abs().mean():.2f} bohr^3"
            if "pauli_mae" in sv:
                msg += f" | pauli MAE {sv['pauli_mae']:.2f} disp MAE {sv.get('disp_mae', 0):.2f}"
            if (lc.w_frz > 0 or lc.w_int > 0) and "frz_mae" in sv:
                msg += f" | frozen - elst MAE {sv['frz_mae']:.2f} total MAE {sv.get('etot_mae', 0):.2f}"
            if "dip_mae" in sv:
                msg += f" | dipole MAE {sv['dip_mae']:.3f} D"
            if force_te:
                fm = force_metrics(model, force_te)
                msg += f" | held-out forces: E/ion MAE {fm['e_mae']:.2f} kJ/mol, F rms {fm['f_rms']:.1f} kJ/mol/A"
                log[-1].update({f"force_{k}": v for k, v in fm.items()})
            if va and (force_tr or lc.w_curv > 0):
                st = torch.random.get_rng_state()
                torch.manual_seed(1234)
                with torch.no_grad():
                    cv = float(curvature_penalty(model, va[:16], rc.curv_delta))
                torch.random.set_rng_state(st)
                log[-1]["val_curv"] = cv
                msg += f" | curv {cv:.3g}" + (f" (train {np.mean(curv_log):.3g})" if curv_log else "")
                curv_log.clear()
            print(f"ep {ep:4d} loss {tot / max(len(tr), 1):.4f} | val pol MAE {sv.get('pol_mae', 0):.2f} ct MAE "
                  f"{sv.get('ct_mae', 0):.2f} kJ/mol{msg} [{time.time() - t0:.0f}s]", flush=True)
            if model.gap:                   # the CT charges and the acceptor overlap shift can trade off: watch both
                with torch.no_grad():
                    kov, q0 = model.overlap_strength(), model.ct_ln_q0.exp()
                rows = {x: {"kov": float(kov[k]), "qD0": float(q0[k, 0]), "qA0": float(q0[k, 1])}
                        for k, x in enumerate(model.tab.symbols)}
                log[-1]["gap_params"] = rows
                print("       gap " + "  ".join(f"{x} kov {r['kov']:.2f} qD0 {r['qD0']:.3f} qA0 {r['qA0']:.3f}"
                                               for x, r in rows.items()), flush=True)
    if (out / "model.pt").exists():
        model.load_state_dict(load_cmm(out / "model.pt", params=params).state_dict())
    else:                                                       # epochs = 0: the model as initialized
        save_cmm(out / "model.pt", model, params=params, train=cfg.to_dict(), epoch=0)
    print("cmm model:\n" + model.summary(), flush=True)
    res = {}
    with open(out / "predictions.csv", "w", newline="") as fh:
        w = None
        for name, cs in (("train", tr), ("val", va), ("test", te)):
            rows = evaluate(model, cs, frz_elst=lc.frz_elst)
            res[name] = summarize(rows, rc.val_cap)
            for r in rows:
                r["split"] = name
                if w is None:
                    w = csv.DictWriter(fh, fieldnames=list(r))
                    w.writeheader()
                w.writerow({k: r.get(k) for k in w.fieldnames})
    if force_tr or force_te:
        res["force_train"] = force_metrics(model, force_tr)
        res["force_test"] = force_metrics(model, force_te)
    json.dump({"metrics": res, "log": log, "config": cfg.to_dict()}, open(out / "metrics.json", "w"), indent=1)
    for name in ("train", "val", "test"):
        m = res[name]["all"]
        print(f"{name:5} pol MAE {m.get('pol_mae', 0):.2f}  ct MAE {m.get('ct_mae', 0):.2f}  pauli MAE "
              f"{m.get('pauli_mae', 0):.2f}  disp MAE {m.get('disp_mae', 0):.2f}  frozen - elst MAE "
              f"{m.get('frz_mae', 0):.2f}  total MAE {m.get('etot_mae', 0):.2f} kJ/mol"
              + (f"  dipole MAE {m['dip_mae']:.3f} D ({m['dip_n']} frames)" if "dip_mae" in m else ""), flush=True)
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
