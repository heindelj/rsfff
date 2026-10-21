"""Train the ion CT force field's learned quantities -- confinement and the overlap gap -- on ALMO-EDA pol and ct.

    python -m rsfff.train.train_ionct --ct-sets CT_SETS [--walk-eda DIR] [--out runs/ionct_v1] [--epochs 300]

Data (wB97M-V/def2-TZVPD ALMO-EDA2, every frame with eda pol and ct):

* ct_model's folded sets (CT_SETS = ct_model/data/ct_sets): ct-walk (MBE-EDA of the NaCl walk clusters: every pair,
  triple and the whole cluster), ct-mbe (MBE-EDA of 20 walk clusters of 4-9 ions), ct-sat (Cl/Na-centred shells),
  ct-scan (pair scans and trimers);
* the walk's own EDA frames (``--walk-eda``, research/rsfff train/nacl/data/eda/*.xyz: random dimers and the
  walk clusters).

Frames are deduplicated by geometry and grouped by parent cluster (an MBE set with all its subsets, one scan, one
shell series), and the split is by parent so a test cluster's pairs and triples are never trained on. The default
test parents are four Na5Cl5 walk clusters and one MBE cluster of each size.

Na-Cl pairs farther apart than ``--ct-max-pair`` (3.6 A) keep their pol label but not their ct label: the isolated
dimer's ionic state mixes with the covalent one there (E_CT / sum PCT climbs past 1.5), which is not the CT the model
describes; in clusters that state is pushed away.

Loss: mean over frames of the per-ion errors of E_pol and E_CT (kJ/mol), squared. Outputs in --out: model.pt,
metrics.json, predictions.csv.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from ..ff.ionct import IonTable
from ..ff.ionct import ct as CT
from ..ff.ionct.model import FrameCache, IonCTModel, collate, precompute

KJ = 2625.4996
DEFAULT_TEST = ("walk_nacl10_111", "walk_nacl10_112", "walk_nacl10_113", "walk_nacl10_114",
                "nacl4_03", "nacl5_07", "nacl6_11", "nacl8_15", "nacl9_19")

# ---- finite-field polarisabilities (ct-polar) ----------------------------------------------------------------------

F_SET = 0.002
#: ct-polar geometries held out of the alpha loss (the rest train it)
POLAR_TEST = ("pair_Na-Cl_r2.250", "pair_Na-Cl_r2.750", "pair_Na-Cl_r3.250", "pair_Cl-Cl_r3.500", "pair_Cl-Cl_r4.500",
              "pair_Cl-Cl_r5.500", "shell_tet4_r2.800", "trimer_Na-Cl-Na_r2.800_a90", "trimer_Na-Cl-Na_r2.800_a120",
              "trimer_Na-Cl-Na_r2.800_a150", "trimer_Na-Cl-Na_r2.800_a180")
#: Na-Cl pairs beyond this (A) carry ionic-covalent mixing in ALMO's alpha_CT: no CT target there
POLAR_CT_MAX_PAIR = 3.6


def polar_targets(ct_sets):
    """[(polar_id, symbols, coords A, target (3, 2) [alpha_MI, alpha_CT] per axis bohr^3 (nan: axis not computed),
    ct_ok)] from the
    ct-polar finite-field EDAs: alpha = -d2E/dF2 by central differences at F_SET; MI = full - ct."""
    import collections
    g = collections.defaultdict(dict)
    p = Path(ct_sets) / "ct-polar.jsonl"
    if not p.exists():
        return []
    for line in open(p):
        r = json.loads(line)
        t = r.get("tags") or {}
        if r.get("stage") != "label" or "polar_id" not in t or "eda" not in r:
            continue
        f = t.get("field") or ""
        key = ("", 0.0) if not f else (f.split(":")[0], round(float(f.split(":")[1]), 6))
        g[t["polar_id"]][key] = r
    out = []
    for pid, d in sorted(g.items()):
        if ("", 0.0) not in d:
            continue
        r0 = d[("", 0.0)]
        full = lambda r: r["energy"]
        ct = lambda r: r["eda"]["ct"] / KJ
        tgt = []
        for ax in "xyz":
            if (ax, F_SET) not in d or (ax, -F_SET) not in d:      # axis not computed (pairs: one perpendicular)
                tgt.append([float("nan"), float("nan")])
                continue
            rp, rm = d[(ax, F_SET)], d[(ax, -F_SET)]
            af = -(full(rp) + full(rm) - 2 * full(r0)) / F_SET ** 2
            ac = -(ct(rp) + ct(rm) - 2 * ct(r0)) / F_SET ** 2
            tgt.append([af - ac, ac])
        X = np.asarray(r0["coords"], float)
        ct_ok = not (pid.startswith("pair_Na-Cl") and float(pid.split("_r")[1]) > POLAR_CT_MAX_PAIR)
        out.append((pid, r0["symbols"], X, torch.tensor(tgt, dtype=torch.float64), ct_ok))
    return out


def polar_fields(B):
    """The 7 fields of the stencil (0, +-x, +-y, +-z) for each of B geometries: (7B, 3), geometry-major."""
    f = torch.zeros(7, 3, dtype=torch.float64)
    for k in range(3):
        f[1 + 2 * k, k], f[2 + 2 * k, k] = F_SET, -F_SET
    return f.repeat(B, 1)


def polar_batch(caches):
    """The stencil batch of the polar caches (each geometry 7 times), collated once."""
    return {"bt": collate([c for c in caches for _ in range(7)]), "field": polar_fields(len(caches)),
            "caches": caches}


def polar_alpha(model, pb):
    """Model alpha (B, 3, 2) [pol, CT] per axis for a polar_batch, network outputs at zero field."""
    B = len(pb["caches"])
    o = model(pb["bt"], field=pb["field"])
    e = torch.stack([o["e_pol"], o["e_ct"]], -1).reshape(B, 7, 2)
    return -(e[:, 1::2] + e[:, 2::2] - 2 * e[:, :1]) / F_SET ** 2


def polar_loss(model, pb):
    """Per-ion alpha errors (bohr^3): (B, 3) for alpha_pol vs ALMO alpha_MI, (B, 3) for alpha_CT (0 where masked)."""
    caches = pb["caches"]
    a = polar_alpha(model, pb)
    t = torch.stack([torch.as_tensor(c.labels["alpha"]) for c in caches]).to(a.device)
    n = torch.tensor([c.species.shape[0] for c in caches], dtype=a.dtype, device=a.device)[:, None]
    ok = torch.tensor([c.labels["ct_ok"] for c in caches], device=a.device)[:, None]
    have = torch.isfinite(t[..., 0])
    t = torch.nan_to_num(t)
    z = torch.zeros_like(a[..., 0])
    return (torch.where(have, (a[..., 0] - t[..., 0]) / n, z),
            torch.where(have & ok, (a[..., 1] - t[..., 1]) / n, z))


def featurizer_for(a, tab):
    """The environment featurizer a run was trained with (train_ionct --env)."""
    if a.get("env", "soap") == "density":
        from rsfff.ff.complete import PROBE_NAMES, DensityEnvironment
        return DensityEnvironment(tab, probes=a.get("env_probes") or PROBE_NAMES)
    from bb.features import SOAPFeaturizer
    return SOAPFeaturizer(cutoff=6.0, n_max=6, l_max=3, neighbor_types=(11, 17), selected_lambdas=(0,)).double()


def _geom_key(symbols, coords):
    """Rigid-motion invariant key: species counts and the sorted, rounded interatomic distances by species pair."""
    a = np.asarray(coords, float)
    d = np.linalg.norm(a[:, None] - a[None], axis=-1)
    i, j = np.triu_indices(len(symbols), 1)
    parts = sorted(f"{min(symbols[p], symbols[q])}{max(symbols[p], symbols[q])}{d[p, q]:.3f}" for p, q in zip(i, j))
    return hashlib.md5(("".join(sorted(symbols)) + "|" + ",".join(parts)).encode()).hexdigest()


def _ho_ratio(r):
    """E_CT / (sum of the ALMO-CTA channel energies, complement included); NaN without CTA or if not attractive."""
    c = r.get("cta") or {}
    if not c.get("E"):
        return float("nan")
    n = len(r["symbols"])
    E = np.nan_to_num(np.array([[np.nan if v is None else v for v in x] for x in c["E"]], float))[:n, :n]
    pct = E.sum() - np.trace(E) + sum(v or 0.0 for v in (c.get("E_to_complement") or []))
    return r["eda"]["ct"] / pct if pct < 0 else float("inf")


def _channel_sums(r, ratio):
    """ALMO-CTA channel energies (Hartree) summed per channel type, scaled by E_CT / sum PCT (how the pair
    functions were calibrated); None without CTA."""
    from ..ff.ionct.model import CHANNEL_TYPES
    c = r.get("cta") or {}
    if not c.get("E") or not np.isfinite(ratio):
        return None
    sym = r["symbols"]
    n = len(sym)
    E = np.nan_to_num(np.array([[np.nan if v is None else v for v in x] for x in c["E"]], float))[:n, :n]
    out = [0.0] * len(CHANNEL_TYPES)
    for i in range(n):
        for j in range(n):
            if i != j:
                out[CHANNEL_TYPES.index(f"{sym[i]}>{sym[j]}")] += E[i, j] * ratio / KJ
    return out


def load_frames(ct_sets: str, walk_eda: str | None = None, ct_max_pair: float = 3.6, max_ho_ratio: float = 1.4):
    """[(symbols, coords A, labels, meta)], deduplicated.

    The ct label is kept only where ALMO's E_CT is a channel quantity: Na-Cl pairs within ``ct_max_pair`` and, for
    every frame with CTA, E_CT / sum PCT <= ``max_ho_ratio``. Above that the higher-order relaxation dominates --
    the ionic-covalent state mixing of a Cl- far from its Na+ partners, which cluster *fragments* (an MBE pair or
    triple without its environment) show and whole clusters do not (E_CT / sum PCT = 1.1-1.2 for every walk
    cluster; up to 4-10 for long-range Na-Cl pairs and Na2Cl+ triples). Whole clusters of 4 or more ions keep their
    label whatever the ratio (1.11-1.30, one at 1.51); above ~1.4 the fragments' CT per ion jumps from ~-3 to
    -5..-6 kJ/mol (e.g. a Cl- 5.5 A from a Na2 2+ pair, E_CT -70 kJ/mol at ratio 1.58)."""
    out, seen = [], set()

    def add(symbols, coords, pol, ct, meta, ho=float("nan"), chan=None, pauli=float("nan"), extra=None):
        key = _geom_key(symbols, coords)
        if key in seen:
            return
        seen.add(key)
        n = len(symbols)
        ct_mask = True
        if n == 2 and sorted(symbols) == ["Cl", "Na"]:
            ct_mask = float(np.linalg.norm(np.subtract(coords[0], coords[1]))) <= ct_max_pair
        whole = bool(meta.get("full")) and n >= 4
        if (np.isfinite(ho) or ho == float("inf")) and not whole:
            ct_mask = ct_mask and ho <= max_ho_ratio
        out.append((list(symbols), np.asarray(coords, float),
                    {"pol": pol / KJ, "ct": ct / KJ, "ct_mask": ct_mask, "chan": chan if ct_mask else None,
                     "pauli": pauli / KJ, **{k: v / KJ for k, v in (extra or {}).items()}},
                    {**meta, "n": n, "ho_ratio": ho}))

    for wf in ("ct-walk", "ct-mbe", "ct-sat", "ct-scan"):
        p = Path(ct_sets) / f"{wf}.jsonl"
        if not p.exists():
            continue
        for line in open(p):
            r = json.loads(line)
            if r.get("stage") != "label" or "eda" not in r or not r.get("symbols") or len(r["symbols"]) < 2:
                continue
            if not set(r["symbols"]) <= {"Na", "Cl"} or (r.get("tags") or {}).get("field"):
                continue
            t = r.get("tags") or {}
            if "mbe_set" in t:
                parent = t["mbe_set"]
            elif t.get("scan") == "pair":
                parent = f"scan_{t.get('scan_pair')}"
            else:
                parent = f"{wf}_{t.get('source', 'x')}"
            add(r["symbols"], r["coords"], r["eda"]["pol"], r["eda"]["ct"],
                {"workflow": wf, "parent": parent, "order": int(t.get("mbe_order", len(r["symbols"]))),
                 "full": t.get("mbe_order") == t.get("mbe_n") if "mbe_set" in t else True,
                 "cta_E": (r.get("cta") or {}).get("E")}, _ho_ratio(r), _channel_sums(r, _ho_ratio(r)),
                r["eda"].get("mod_pauli", float("nan")),
                {k: r["eda"].get(e, float("nan")) for k, e in (("disp", "disp"), ("elec", "cls_elec"), ("int", "int"))})
    if walk_eda:
        from ase.io import read
        for f in sorted(Path(walk_eda).glob("*.xyz")):
            for k, a in enumerate(read(f, ":")):
                sym = a.get_chemical_symbols()
                if "eda_ct" not in a.info:
                    continue
                parent = f"walkeda_{f.stem}_{k}" if len(sym) == 2 else f"walkeda_{f.stem}_{k}"
                add(sym, a.positions, a.info["eda_pol"] * KJ, a.info["eda_ct"] * KJ,
                    {"workflow": "walk-eda", "parent": parent, "order": len(sym), "full": True},
                    pauli=a.info.get("eda_mod_pauli", float("nan")) * KJ,
                    extra={k: a.info.get(e, float("nan")) * KJ for k, e in (("disp", "eda_disp"), ("elec", "eda_cls_elec"),
                                                                             ("int", "eda_int"))})
    return out


def split(frames, test_parents, val_frac=0.1, seed=0):
    parents = sorted({m["parent"] for *_, m in frames})
    rest = [p for p in parents if p not in set(test_parents)]
    rng = random.Random(seed)
    rng.shuffle(rest)
    val = set(rest[: max(1, int(val_frac * len(rest)))])
    tag = lambda p: "test" if p in set(test_parents) else "val" if p in val else "train"
    return [tag(m["parent"]) for *_, m in frames]


_PP = {}


def _pp_init(tab, chan, featurizer, lean, frontier=False):
    torch.set_num_threads(1)
    _PP.update(tab=tab, chan=chan, featurizer=featurizer, lean=lean, frontier=frontier)


def _pp_one(item):
    """One frame's cache, serialized with torch.save (bytes): torch's own pickling of tensors between processes
    passes shared-memory file descriptors and runs out of them on big batches ("received 0 items of ancdata")."""
    import io
    k, sym, X = item
    buf = io.BytesIO()
    torch.save(precompute(sym, X, _PP["tab"], _PP["chan"], _PP["featurizer"], lean=_PP["lean"],
                          frontier=_PP["frontier"]), buf)
    return k, buf.getvalue()


CACHE_VERSION = 2     # 2: equivariant environment blocks on the frame caches


def cache_key(params_path, env, lean, frames, frontier=False):
    """A short hash of what the precompute depends on: the parameter table's contents, the environment, lean (and
    whether the lean cache keeps the frontier potentials), the cache layout version and the frames' geometries
    (species + coordinates), in order."""
    import hashlib
    h = hashlib.sha1()
    h.update(Path(params_path).read_bytes())
    h.update(f"|{env}|lean={int(lean)}|frontier={int(frontier)}|v{CACHE_VERSION}|{len(frames)}|".encode())
    for sym, X, *_ in frames:
        h.update("".join(sym).encode())
        h.update(np.asarray(X, dtype=np.float64).round(6).tobytes())
    return h.hexdigest()[:16]


def build_cache(frames, tab, chan, featurizer, path=None, lean=False, workers=0, frontier=False):
    """The per-frame precompute (:func:`rsfff.ff.ionct.model.precompute`), from ``path`` when it holds a complete
    cache for these frames, else computed -- in ``workers`` processes when > 0 (the frames are independent; each
    worker runs single-threaded, so use the node's cores) -- and saved there. Labels and masks always come from
    the data, never from the cache."""
    if path and Path(path).exists():
        caches = torch.load(path, weights_only=False)
        need = ("ps_index",) if lean else ("ps_index", "EVD", "ch_EP", "ch_J")
        complete = all(getattr(caches[0], k, None) is not None for k in need) if caches else False
        if len(caches) == len(frames) and complete:   # geometry is cached; labels and masks always come from the data
            for c, (sym, X, lab, meta) in zip(caches, frames):
                assert len(sym) == c.species.shape[0]
                c.labels, c.meta = dict(lab), dict(meta)
            return caches
    t0 = time.time()
    caches = [None] * len(frames)
    if workers and workers > 1 and len(frames) > 2 * workers:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        items = [(k, sym, X) for k, (sym, X, *_rest) in enumerate(frames)]
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), initializer=_pp_init,
                                 initargs=(tab, chan, featurizer, lean, frontier)) as ex:
            done = 0
            import io
            for k, blob in ex.map(_pp_one, items, chunksize=8):
                caches[k] = torch.load(io.BytesIO(blob), weights_only=False)
                done += 1
                if done % 500 == 0:
                    print(f"  precomputed {done}/{len(frames)} [{time.time() - t0:.0f}s, {workers} workers]", flush=True)
    else:
        for k, (sym, X, *_rest) in enumerate(frames):
            caches[k] = precompute(sym, X, tab, chan, featurizer, lean=lean, frontier=frontier)
            if (k + 1) % 200 == 0:
                print(f"  precomputed {k + 1}/{len(frames)} [{time.time() - t0:.0f}s]", flush=True)
    for c, (sym, X, lab, meta) in zip(caches, frames):
        c.labels, c.meta = dict(lab), dict(meta)
    print(f"  precompute done: {len(frames)} frames in {time.time() - t0:.0f}s (lean={lean})", flush=True)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(caches, path)
    return caches


FORCE_TEST = ("nacl14", "nacl16", "nacl18", "nacl20")
KJ_ANG = KJ / 0.529177210903          # Hartree/bohr -> kJ/mol/A


def monomer_energies(eda_dirs):
    """{symbol: isolated-ion SCF energy (Hartree)} from the ``fragment_energies`` of EDA extxyz files."""
    from ase.io import read
    out = {}
    for d in eda_dirs:
        for f in sorted(Path(d).glob("*.xyz")):
            for a in read(f, ":"):
                fe = a.info.get("fragment_energies")
                if fe is None:
                    continue
                fi = a.arrays["fragment_idx"]
                for k, e in enumerate(np.asarray(fe, float).reshape(-1)):
                    out.setdefault(a.get_chemical_symbols()[int(np.argmax(fi == k))], float(e))
                if len(out) >= 2:
                    return out
    return out


def load_force_frames(dirs, mono, species=("Na", "Cl")):
    """[(symbols, coords A, {'int': Hartree, 'forces': (N, 3) Hartree/bohr}, meta)] from extxyz files with total
    energies and forces (atomic units); the interaction energy is E - sum of the isolated-ion energies."""
    from ase.io import read
    out = []
    for d in dirs:
        for f in sorted(Path(d).glob("*.xyz")):
            for k, a in enumerate(read(f, ":")):
                sym = a.get_chemical_symbols()
                if not set(sym) <= set(species) or a.calc is None or "forces" not in a.calc.results:
                    continue
                e_int = float(a.get_potential_energy()) - sum(mono[x] for x in sym)
                out.append((sym, np.asarray(a.positions, float), {"int": e_int, "forces": np.asarray(a.get_forces(), float)},
                            {"workflow": "force", "parent": f.stem, "file": f.stem, "index": k, "n": len(sym)}))
    return out


def radial_table(tab, params_path, cache_dir=None):
    """The complete model's :class:`RadialTable` for this parameter table, cached on disk by the table's hash."""
    import hashlib
    from ..ff.complete import RadialTable
    key = hashlib.sha1(Path(params_path).read_bytes()).hexdigest()[:16]
    path = Path(cache_dir) / f"table_{key}.pt" if cache_dir else None
    if path and path.exists():
        return torch.load(path, weights_only=False)
    t0 = time.time()
    table = RadialTable(tab)
    print(f"radial table: {table.n_funcs} functions x {table.G} points in {time.time() - t0:.0f}s", flush=True)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(table, path)
    return table


def to_device(obj, device):
    """Recursively move the tensors of a batch dict / list to ``device`` (no-op on CPU)."""
    if device is None or str(device) == "cpu":
        return obj
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device(v, device) for v in obj)
    return obj


def force_terms(model, frames, tab, table, chan):
    """Energy (per ion, kJ/mol) and force (per component, kJ/mol/A) errors of a list of force frames through the
    batched differentiable precompute: (B,), (sum N, 3). Forces are -dE/dx with the graph kept for training."""
    from ..ff.complete import batched_precompute
    Zmap = {"Na": 11, "Cl": 17}
    dev = table.device
    pos = torch.tensor(np.concatenate([X for _, X, *_ in frames]), dtype=torch.float64, device=dev, requires_grad=True)
    Z = torch.tensor([Zmap[x] for sym, *_ in frames for x in sym], device=dev)
    fr = torch.cat([torch.full((len(sym),), k, dtype=torch.long) for k, (sym, *_) in enumerate(frames)]).to(dev)
    bt = batched_precompute(pos, Z, fr, tab, table, chan)
    out = model(bt)
    E = out["e_int"]
    F = -torch.autograd.grad(E.sum(), pos, create_graph=True)[0]
    e_ref = torch.tensor([lab["int"] for _, _, lab, _ in frames], dtype=torch.float64, device=dev)
    f_ref = torch.tensor(np.concatenate([lab["forces"] for _, _, lab, _ in frames]), dtype=torch.float64, device=dev)
    n = torch.tensor([len(sym) for sym, *_ in frames], dtype=torch.float64, device=dev)
    return (E - e_ref) * KJ / n, F * KJ - f_ref * KJ_ANG, out      # model forces are Hartree/A, labels Hartree/bohr


def force_metrics(model, frames, tab, table, chan, bs=16):
    """Held-out energy and force errors over force frames: E/ion MAE and rms (kJ/mol), force rms and MAE per
    component (kJ/mol/A), max |dF|, and the same per file."""
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


def build_model(a: dict, tab, chan, n_feat, feat_scale=None, feature_names=None):
    """The model of a run from its arguments (train_ionct's, as saved in model.pt): the ion CT model or, with
    ``model == "complete"``, the complete model on the environment heads."""
    g = a.get
    if g("model", "ionct") == "complete":
        from ..ff.complete import CompleteModel
        return CompleteModel(tab, chan, n_feat, feature_names=feature_names, hidden=g("hidden", 64),
                             feat_scale=feat_scale, c_range=tuple(g("c_range", (0.05, 1.25))), confine=g("confine"),
                             pseudo=g("pseudo", False), tie_c=g("tie_c", False), species_emb=g("species_emb", False),
                             aniso=not g("no_aniso", False), ct_gap=not g("no_ct_gap", False),
                             dphi_scale=g("dphi_scale", 0.05), frozen=g("frozen", False),
                             conf_form=g("conf_form", "tanh"), drive_overlap=g("drive_overlap", False),
                             hyper=g("hyper", False), pauli_moments=g("pauli_moments", False))
    return IonCTModel(tab, chan, n_feat, hidden=g("hidden", 64), wkb=g("wkb", "smooth+floor"), feat_scale=feat_scale,
                      c_range=tuple(g("c_range", (0.05, 1.25))), confine=g("confine"), pseudo=g("pseudo", False),
                      joint=g("joint", False), ct_pol=g("ct_pol", False), gap_form=g("gap_form", "pair"),
                      tie_c=g("tie_c", False), coupling_resp=g("coupling_resp", "barrier"),
                      ct_form=g("ct_form", "channels"), pol_form=g("pol_form", "confine"), pauli_nn=g("pauli_nn", False))


def loss_terms(out, bt):
    """Per-frame per-ion errors (kJ/mol): E_pol, E_CT and the per-type channel sums (B, 4; zero where no label)."""
    lab = bt["labels"]
    n = lab["n"]
    e_pol = (out["e_pol"] - lab["pol"]) * KJ / n
    e_ct = torch.where(lab["ct_mask"], (out["e_ct"] - lab["ct"]) * KJ / n, torch.zeros_like(n))
    ch = lab["chan"]
    e_ch = torch.where(torch.isfinite(ch), (out["e_type"] - torch.nan_to_num(ch)) * KJ / n[:, None], torch.zeros_like(ch))
    return e_pol, e_ct, e_ch


def pauli_term(out, bt):
    """Per-frame per-ion error (kJ/mol) of the frozen Pauli energy against ALMO mod_pauli (zero where no label)."""
    return _term(out["e_pauli"], bt, "pauli")


def disp_term(out, bt):
    return _term(out["e_disp"], bt, "disp")


def int_term(out, bt):
    """Total interaction energy: the cls_elec label (fixed electrostatics, Stage A parity) + the model's Pauli,
    dispersion, polarization and CT, against ALMO int."""
    lab = bt["labels"]
    e = lab["elec"] + out["e_pauli"] + out["e_disp"] + out["e_pol"] + out["e_ct"]
    return torch.where(lab["ct_mask"], _term(e, bt, "int"), torch.zeros_like(e))     # no CT label: no total either


def _term(e, bt, key):
    lab = bt["labels"]
    y = lab[key]
    return torch.where(torch.isfinite(y), (e - torch.nan_to_num(y)) * KJ / lab["n"], torch.zeros_like(y))


def evaluate(model, caches, bs=64):
    rows = []
    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        for k in range(0, len(caches), bs):
            chunk = caches[k:k + bs]
            bt = to_device(collate(chunk), dev)
            out = {k_: (v.cpu() if torch.is_tensor(v) else v) for k_, v in model(bt).items()}
            for q, f in enumerate(chunk):
                rows.append({**{k: v for k, v in f.meta.items() if k != "cta_E"}, "pol_ref": f.labels["pol"] * KJ, "ct_ref": f.labels["ct"] * KJ,
                             "pol": out["e_pol"][q].item() * KJ, "ct": out["e_ct"][q].item() * KJ,
                             "pauli_ref": f.labels.get("pauli", float("nan")) * KJ, "pauli": out["e_pauli"][q].item() * KJ,
                             "disp_ref": f.labels.get("disp", float("nan")) * KJ,
                             "disp": out["e_disp"][q].item() * KJ if "e_disp" in out else float("nan"),
                             "int_ref": f.labels.get("int", float("nan")) * KJ,
                             "int": (f.labels.get("elec", float("nan")) + out["e_pauli"][q].item() + out.get("e_disp", out["e_pauli"])[q].item()
                                     + out["e_pol"][q].item() + out["e_ct"][q].item()) * KJ,
                             "ct_mask": bool(f.labels["ct_mask"])})
    model.train()
    return rows


def summarize(rows):
    def stats(sel):
        if not sel:
            return {}
        p = np.array([r["pol"] - r["pol_ref"] for r in sel])
        c = np.array([r["ct"] - r["ct_ref"] for r in sel if r["ct_mask"]])
        cr = np.array([r["ct_ref"] for r in sel if r["ct_mask"]])
        out = {"n": len(sel), "pol_mae": float(np.abs(p).mean()), "pol_rms": float(np.sqrt((p ** 2).mean()))}
        pa = np.array([r["pauli"] - r["pauli_ref"] for r in sel if np.isfinite(r.get("pauli_ref", np.nan))])
        if len(pa) and any(r["pauli"] != 0 for r in sel):
            out["pauli_mae"] = float(np.abs(pa).mean())
        if any(r["pauli"] != 0 for r in sel):          # a model with frozen terms: dispersion and total interaction
            for key in ("disp", "int"):
                d = np.array([r[key] - r[f"{key}_ref"] for r in sel if (key != "int" or r["ct_mask"])
                              and np.isfinite(r.get(f"{key}_ref", np.nan)) and np.isfinite(r.get(key, np.nan))])
                if len(d):
                    out[f"{key}_mae"] = float(np.abs(d).mean())
        if len(c):
            out.update(ct_mae=float(np.abs(c).mean()), ct_rms=float(np.sqrt((c ** 2).mean())),
                       ct_rel=float(np.abs(c).sum() / max(np.abs(cr).sum(), 1e-12)))
        return out
    res = {"all": stats(rows)}
    for key in ("full", "subset"):
        res[key] = stats([r for r in rows if (r["full"] if key == "full" else not r["full"])])
    for n in sorted({r["n"] for r in rows}):
        res[f"n={n}"] = stats([r for r in rows if r["n"] == n])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct-sets", required=True)
    ap.add_argument("--walk-eda", default=None)
    ap.add_argument("--params", default=str(Path(__file__).resolve().parents[2] / "params" / "ions_wb97mv_tzvpd.json"))
    ap.add_argument("--out", default="runs/ionct")
    ap.add_argument("--cache", default=None, help="the precompute cache file (default: <out>/cache.pt, or --cache-dir)")
    ap.add_argument("--cache-dir", default=None,
                    help="directory of keyed caches, cache_<hash>.pt with the hash of (params, env, lean, geometries): "
                         "runs sharing them share the precompute")
    ap.add_argument("--precompute-workers", type=int, default=0,
                    help="processes for the precompute (0: this process; e.g. the node's cores)")
    ap.add_argument("--full-precompute", action="store_true",
                    help="compute the channel CT model's tensors too (the default is lean when --ct-form direct)")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--wkb", default="smooth+floor", help="two_sided | none | smooth[:a_-,a_+], each optionally +floor")
    ap.add_argument("--ct-max-pair", type=float, default=3.6)
    ap.add_argument("--max-ho-ratio", type=float, default=1.4, help="E_CT / sum PCT above which ct is not a label")
    ap.add_argument("--w-pol", type=float, default=1.0)
    ap.add_argument("--w-ct", type=float, default=1.0)
    ap.add_argument("--w-chan", type=float, default=0.0, help="weight of the per-type ALMO-CTA channel sums (0: E_CT only)")
    ap.add_argument("--c-range", nargs=2, type=float, default=(0.05, 1.25), help="bounds of the confinement scales")
    ap.add_argument("--confine", nargs="*", default=None, help="species the network may confine (default: all)")
    ap.add_argument("--test", nargs="*", default=list(DEFAULT_TEST))
    ap.add_argument("--pseudo", action="store_true", help="learned Pauli pseudo-potential drive on the induction")
    ap.add_argument("--joint", action="store_true", help="joint variational pol + CT solve (CT feeds back on the induction)")
    ap.add_argument("--ct-pol", action="store_true", help="CT-state polarizability: second-order Stark shift of the gaps")
    ap.add_argument("--gap-form", default="pair", choices=("pair", "density"),
                    help="'density' (R9): gap I_D - A_A + potentials - J, coupling from frontier overlap; no pair calibration")
    ap.add_argument("--tie-c", action="store_true", help="c_C = c_alpha^(5/3) (a scaled ion) instead of a free c_C")
    ap.add_argument("--ct-form", default="channels", choices=("channels", "direct"),
                    help="'direct': CMM-style short-range CT from environment-dependent donor/acceptor CT charges")
    ap.add_argument("--pol-form", default="confine", choices=("confine", "pauli"),
                    help="response shaping: learned confinement (+ --pseudo) or the Pauli kernel on induced moments")
    ap.add_argument("--pauli-nn", action="store_true", help="network rescales each ion's Pauli monopole")
    ap.add_argument("--w-pauli", type=float, default=0.0, help="weight of the frozen Pauli energy vs ALMO mod_pauli")
    ap.add_argument("--coupling-resp", default="barrier", choices=("barrier", "frontier"),
                    help="density coupling's environment/field response: 'frontier' = polarizable frontier densities "
                         "(orbital polarizabilities tied to the CT-state Stark shift)")
    ap.add_argument("--w-alpha-ct", type=float, default=1.0,
                    help="relative weight of alpha_CT within the alpha loss (0: alpha_CT is validation only)")
    ap.add_argument("--w-smooth", type=float, default=0.0, help="feature-noise smoothness penalty on c, Delta^ov, A")
    ap.add_argument("--smooth-sigma", type=float, default=0.05)
    ap.add_argument("--alpha-every", type=int, default=1, help="evaluate the alpha loss every k-th step (weight x k)")
    ap.add_argument("--holdout-scans", action="store_true",
                    help="all ct-scan pair frames, and the ct-polar pair geometries, go to the test split")
    ap.add_argument("--w-alpha", type=float, default=0.0, help="weight of the ct-polar alpha loss (bohr^3 per ion)")
    ap.add_argument("--polar-test", nargs="*", default=list(POLAR_TEST))
    ap.add_argument("--env", default="soap", choices=("soap", "density"),
                    help="per-ion environment features: SOAP invariants (bb), or the C0 field/overlap projections onto "
                         "the ion's own densities (rsfff.ff.complete.DensityEnvironment; complete model plan)")
    ap.add_argument("--env-probes", nargs="*", default=None, help="density env: which probes (default all)")
    ap.add_argument("--model", default="ionct", choices=("ionct", "complete"),
                    help="complete: environment heads, anisotropic alpha, CT gap factor (direct CT, density env)")
    ap.add_argument("--species-emb", action="store_true", help="complete: species embedding on the heads' input")
    ap.add_argument("--no-aniso", action="store_true", help="complete: isotropic polarizability")
    ap.add_argument("--no-ct-gap", action="store_true", help="complete: no environment gap factor on the CT")
    ap.add_argument("--dphi-scale", type=float, default=0.05, help="complete: scale (Eh/e) of the CT gap potential")
    ap.add_argument("--frozen", action="store_true", help="complete: frozen Pauli (env qP) and dispersion (env C6) terms")
    ap.add_argument("--pauli-moments", action="store_true",
                    help="complete + frozen: induced Pauli dipole / quadrupole from the overlap gradients (polytensor Pauli kernel)")
    ap.add_argument("--hyper", action="store_true",
                    help="complete: field-induced polarizability enhancement alpha (1 + g |V[dip]^(1)|^2) (hyperpolarizability)")
    ap.add_argument("--drive-overlap", action="store_true",
                    help="complete: the exchange-polarization drive strength carries the pair overlap of the densities")
    ap.add_argument("--conf-form", default="tanh", choices=("tanh", "mono"),
                    help="complete: confinement form -- tanh (c_range-bounded, either sign) or mono (stiffening only, "
                         "c <= 1, overlap-gated)")
    ap.add_argument("--force-data", nargs="*", default=None,
                    help="complete + frozen: extxyz dirs with total energies and forces (atomic units); trains total "
                         "interaction energies and forces through the tabulated, differentiable precompute")
    ap.add_argument("--w-force", type=float, default=0.0, help="weight of the force loss (kJ/mol/A per component)")
    ap.add_argument("--w-fe", type=float, default=1.0, help="weight of the force frames' total energy (kJ/mol per ion)")
    ap.add_argument("--force-every", type=int, default=1, help="force step every k-th energy step (weight x k)")
    ap.add_argument("--force-batch", type=int, default=16)
    ap.add_argument("--force-test", nargs="*", default=list(FORCE_TEST), help="force files (stems) held out")
    ap.add_argument("--init-from", default=None, help="model.pt of a previous run to start from (same architecture)")
    ap.add_argument("--device", default="cpu", help="cpu | cuda | cuda:N (complete model; batches are moved per step)")
    ap.add_argument("--w-disp", type=float, default=0.0, help="weight of the dispersion energy vs ALMO disp")
    ap.add_argument("--w-int", type=float, default=0.0,
                    help="weight of the total interaction energy (cls_elec label + pauli + disp + pol + ct) vs ALMO int")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tab, chan = IonTable.load(a.params), CT.ChannelTable.load(a.params)
    feat = featurizer_for(vars(a), tab)
    if a.env == "density":
        print(f"environment: density projections, {feat.n_feat} invariants", flush=True)
    frames = load_frames(a.ct_sets, a.walk_eda, a.ct_max_pair, a.max_ho_ratio)
    print(f"ct label kept for {sum(f[2]['ct_mask'] for f in frames)} of {len(frames)} frames", flush=True)
    tags = split(frames, a.test, seed=a.seed)
    if a.holdout_scans:
        tags = ["test" if (m["workflow"] == "ct-scan" and m["n"] == 2) else t for t, (*_, m) in zip(tags, frames)]
        print(f"held out {sum(m['workflow'] == 'ct-scan' and m['n'] == 2 for *_, m in frames)} pair-scan frames", flush=True)
    print(f"{len(frames)} frames: " + ", ".join(f"{t} {tags.count(t)}" for t in ("train", "val", "test")), flush=True)
    if a.model == "complete":
        if a.ct_form != "direct" or a.env != "density":
            raise SystemExit("--model complete needs --ct-form direct and --env density")
    lean = (a.ct_form == "direct") and not a.full_precompute
    frontier = a.model == "complete" and not a.no_ct_gap
    cache_path = a.cache
    if cache_path is None and a.cache_dir:
        cache_path = Path(a.cache_dir) / f"cache_{cache_key(a.params, a.env, lean, frames, frontier)}.pt"
        print(f"cache: {cache_path}", flush=True)
    caches = build_cache(frames, tab, chan, feat, cache_path or out / "cache.pt", lean=lean, workers=a.precompute_workers,
                         frontier=frontier)
    tr = [c for c, t in zip(caches, tags) if t == "train"]
    va = [c for c, t in zip(caches, tags) if t == "val"]
    te = [c for c, t in zip(caches, tags) if t == "test"]
    F = torch.cat([c.feats for c in tr])
    nz = F.abs().sum(1) > 0
    scale = F[nz].std(0).clamp(min=1e-6)
    model = build_model(vars(a), tab, chan, F.shape[1], scale, getattr(feat, "feature_names", None))
    dev = torch.device(a.device)
    if dev.type != "cpu":
        tab = tab.to(dev)
        model.tab = tab
        model.to(dev)
        print(f"device: {dev} ({torch.cuda.get_device_name(dev) if dev.type == 'cuda' else ''})", flush=True)
    polar_tr, polar_te = [], []
    if a.w_alpha > 0:
        for pid, sym, X, tgt, ok in polar_targets(a.ct_sets):
            c = precompute(sym, X, tab, chan, feat, {"alpha": tgt, "ct_ok": ok}, {"polar_id": pid}, lean=lean,
                           frontier=frontier)
            held = pid in set(a.polar_test) or (a.holdout_scans and pid.startswith("pair_"))
            (polar_te if held else polar_tr).append(c)
        print(f"alpha targets: {len(polar_tr)} train, {len(polar_te)} held out", flush=True)
        pb_tr, pb_all = to_device(polar_batch(polar_tr), dev), to_device(polar_batch(polar_tr + polar_te), dev)
    if a.init_from:
        ck = torch.load(a.init_from, weights_only=False)
        missing, unexpected = model.load_state_dict(ck["state"], strict=False)
        print(f"init from {a.init_from}: {len(missing)} new parameters, {len(unexpected)} dropped", flush=True)
    force_tr, force_te, table = [], [], None
    if a.force_data:
        if not (a.model == "complete" and a.frozen):
            raise SystemExit("--force-data needs --model complete --frozen (the model must own every term)")
        mono = monomer_energies([a.walk_eda] if a.walk_eda else [])
        if len(mono) < 2:
            raise SystemExit("monomer energies: --walk-eda files with fragment_energies are needed")
        print(f"monomer energies (Hartree): {mono}", flush=True)
        ff = load_force_frames(a.force_data, mono)
        held = set(a.force_test)
        force_tr = [f for f in ff if f[3]["file"] not in held]
        force_te = [f for f in ff if f[3]["file"] in held]
        print(f"force frames: {len(force_tr)} train, {len(force_te)} held out ({', '.join(sorted(held))})", flush=True)
        table = radial_table(tab, a.params, a.cache_dir).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=a.lr * 0.02)
    best, best_state, log = math.inf, None, []
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        random.shuffle(tr)
        tot = 0.0
        for step, k in enumerate(range(0, len(tr), a.batch)):
            bt = to_device(collate(tr[k:k + a.batch]), dev)
            o = model(bt)
            ep_, ec_, ech_ = loss_terms(o, bt)
            loss = (a.w_pol * ep_ ** 2 + a.w_ct * ec_ ** 2 + a.w_chan * (ech_ ** 2).sum(1)).mean()
            if a.w_pauli > 0:
                loss = loss + a.w_pauli * (pauli_term(o, bt) ** 2).mean()
            if a.w_disp > 0:
                loss = loss + a.w_disp * (disp_term(o, bt) ** 2).mean()
            if a.w_int > 0:
                loss = loss + a.w_int * (int_term(o, bt) ** 2).mean()
            if polar_tr and step % a.alpha_every == 0:      # the stencil batch is the costly part: every k-th step, k x weight
                ap_, ac_ = polar_loss(model, pb_tr)
                loss = loss + a.alpha_every * a.w_alpha * (ap_ ** 2 + a.w_alpha_ct * ac_ ** 2).mean()
            if a.w_smooth > 0:
                loss = loss + a.w_smooth * model.smoothness_penalty(bt, a.smooth_sigma)
            if force_tr and a.w_force > 0 and step % a.force_every == 0:
                fb = random.sample(force_tr, min(a.force_batch, len(force_tr)))
                fe_, ff_, _ = force_terms(model, fb, tab, table, chan)
                loss = loss + a.force_every * (a.w_fe * (fe_ ** 2).mean() + a.w_force * (ff_ ** 2).mean())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            tot += loss.item() * len(bt["labels"]["n"])
        sched.step()
        if ep % a.log_every == 0 or ep == a.epochs:
            sv = summarize(evaluate(model, va))["all"]
            score = (sv.get("pol_mae", 0) + sv.get("ct_mae", 0) + (sv.get("pauli_mae", 0) if a.w_pauli > 0 else 0)
                     + (sv.get("disp_mae", 0) if a.w_disp > 0 else 0))
            if score < best:
                best, best_state = score, {k: v.detach().clone() for k, v in model.state_dict().items()}
            log.append({"epoch": ep, "train_loss": tot / len(tr), **{f"val_{k}": v for k, v in sv.items()}})
            msg = ""
            if polar_tr:
                with torch.no_grad():
                    ap_, ac_ = polar_loss(model, pb_all)
                msg = f" | alpha/ion |err| pol {ap_.abs().mean():.2f} ct {ac_.abs().mean():.2f} bohr^3"
            if "pauli_mae" in sv:
                msg = f" pauli MAE {sv['pauli_mae']:.2f}" + (f" disp MAE {sv['disp_mae']:.2f}" if "disp_mae" in sv else "") + msg
            if force_te:
                fm = force_metrics(model, force_te, tab, table, chan)
                msg += f" | held-out forces: E/ion MAE {fm['e_mae']:.2f} kJ/mol, F rms {fm['f_rms']:.1f} kJ/mol/A"
                log[-1].update({f"force_{k}": v for k, v in fm.items()})
            print(f"ep {ep:4d} loss {tot / len(tr):.4f} | val pol MAE {sv.get('pol_mae', 0):.2f} ct MAE "
                  f"{sv.get('ct_mae', 0):.2f} kJ/mol{msg} [{time.time() - t0:.0f}s]", flush=True)
    model.load_state_dict(best_state)
    if model.gap_form == "density":
        from ..ff.ionct.model import CHANNEL_TYPES as _CT
        print("coupling ln w / gamma / lambda: " + "; ".join(
            f"{t} {model.lnw[k]:.2f} / {model.gamma[k]:.3f} / {model.lam[k]:.3f}" for k, t in enumerate(_CT))
            + f" | A_A: {dict(zip(tab.symbols, [round(v, 4) for v in model.acceptor_affinity().tolist()]))}", flush=True)
    if getattr(model, "pol_form", "confine") == "pauli":
        import torch.nn.functional as Fn
        kap = Fn.softplus(model.pauli_kappa_raw)
        s0 = Fn.softplus(model.pauli_s0_raw)
        s2 = s0[:, 0] * (-1.5 + 4.5 * torch.sigmoid(model.pauli_s2_raw))
        print("Pauli kernel: " + "; ".join(
            f"{x} qP {model.pauli_lnq[k].exp():.3f} bP {model.pauli_lnb[k].exp():.3f} kappa_d {kap[k, 0]:.3f} "
            f"kappa_q {kap[k, 1]:.3f} s0d {s0[k, 0]:.3f} s2d {s2[k]:.3f} s0q {s0[k, 1]:.3f}"
            for k, x in enumerate(tab.symbols)), flush=True)
    if getattr(model, "ct_form", "channels") == "direct":
        bD, bA = model.ct_exponents()
        print("direct CT kernel exponents (1/bohr): donor " + ", ".join(f"{x} {v:.3f}" for x, v in zip(tab.symbols, bD.tolist()))
              + "; acceptor " + ", ".join(f"{x} {v:.3f}" for x, v in zip(tab.symbols, bA.tolist())), flush=True)
    if getattr(model, "coupling_resp", "barrier") == "frontier":
        oa, od = model.orbital_polarizabilities()
        print("frontier orbital polarizabilities (bohr^3): acceptor " +
              ", ".join(f"{x} {v:.1f}" for x, v in zip(tab.symbols, oa.tolist())) + "; donor " +
              ", ".join(f"{x} {v:.1f}" for x, v in zip(tab.symbols, od.tolist())), flush=True)
    if model.ct_pol:
        aa, ad = model.astar()
        print("CT-state excess polarizability (bohr^3): acceptor " +
              ", ".join(f"{x} {v:.1f}" for x, v in zip(tab.symbols, aa.tolist())) + "; donor " +
              ", ".join(f"{x} {v:.1f}" for x, v in zip(tab.symbols, ad.tolist())), flush=True)
    if hasattr(model, "summary"):
        print("complete model: " + json.dumps(model.summary()), flush=True)
    torch.save({"state": {k: v.cpu() for k, v in model.state_dict().items()}, "args": vars(a), "n_feat": F.shape[1],
                "feature_names": getattr(feat, "feature_names", None)}, out / "model.pt")
    res = {}
    with open(out / "predictions.csv", "w", newline="") as fh:
        w = None
        for name, cs in (("train", tr), ("val", va), ("test", te)):
            rows = evaluate(model, cs)
            res[name] = summarize(rows)
            for r in rows:
                r["split"] = name
                if w is None:
                    w = csv.DictWriter(fh, fieldnames=list(r))
                    w.writeheader()
                w.writerow({k: r.get(k) for k in w.fieldnames})
    if force_tr or force_te:
        res["force_train"] = force_metrics(model, force_tr, tab, table, chan)
        res["force_test"] = force_metrics(model, force_te, tab, table, chan)
        for name in ("force_train", "force_test"):
            print(f"{name}: " + json.dumps(res[name]), flush=True)
    json.dump({"metrics": res, "log": log, "args": vars(a)}, open(out / "metrics.json", "w"), indent=1)
    for name in ("train", "val", "test"):
        m = res[name]["all"]
        extra = "".join(f"  {k} MAE {m[f'{k}_mae']:.2f}" for k in ("pauli", "disp", "int") if f"{k}_mae" in m)
        print(f"{name:5} pol MAE {m.get('pol_mae', 0):.2f}  ct MAE {m.get('ct_mae', 0):.2f}{extra} kJ/mol "
              f"(ct rel {m.get('ct_rel', 0):.3f}); full clusters: {res[name].get('full', {})}")


if __name__ == "__main__":
    main()
