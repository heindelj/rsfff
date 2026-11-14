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
import os
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


# Shannon ionic radii (A, 6-coordinate): scale the Na-Cl dimer CT cut to the other salts
IONIC_RADIUS = {"Li": 0.76, "Na": 1.02, "K": 1.38, "Rb": 1.52, "Cs": 1.67, "F": 1.33, "Cl": 1.81, "Br": 1.96, "I": 2.20}


def dimer_ct_cut(symbols, na_cl_cut):
    """The distance below which an unlike ion pair's ALMO E_CT is a label: ``na_cl_cut`` for Na-Cl (3.6 A by
    default), scaled by the sum of the two ionic radii for the other salts (KBr 4.24, CsI 4.89, LiF 2.67 A at 3.6).
    Beyond it the vacuum dimer's ionic and covalent configurations mix and E_CT is not a channel quantity."""
    a, b = symbols
    return na_cl_cut * (IONIC_RADIUS[a] + IONIC_RADIUS[b]) / (IONIC_RADIUS["Na"] + IONIC_RADIUS["Cl"])


def polar_targets(ct_sets, ct_dimers=True, files=("ct-polar",), min_ct=None, ct_max_pair=POLAR_CT_MAX_PAIR):
    """[(polar_id, symbols, coords A, target (3, 2) [alpha_MI, alpha_CT] per axis bohr^3 (nan: axis not computed),
    ct_ok)] from the
    ct-polar finite-field EDAs: alpha = -d2E/dF2 by central differences at F_SET; MI = full - ct.
    ``files``: the polar sets to read (ct-polar, seed-<salt>-polar); a seed set's ids carry its salt,
    "<salt>:<polar_id>" (its shell ids name no species).
    ``min_ct``: drop a geometry whose alpha_CT per ion falls below it on any axis (bohr^3): a negative alpha_CT means
    the field switched the EDA's CT partition (LiF near contact: CT -157, MI +163 per ion, full ~5), so MI is wrong too.
    ``ct_max_pair``: an unlike pair's alpha_CT is a target only within ``dimer_ct_cut(pair, ct_max_pair)`` (its alpha_MI
    always is): beyond it the field mixes the dimer's ionic and covalent configurations (ALMO alpha_CT keeps rising)."""
    import collections
    g = collections.defaultdict(dict)
    for name in files:
        p = Path(ct_sets) / f"{name}.jsonl"
        if not p.exists():
            continue
        prefix = f"{name.split('-')[1]}:" if name.startswith("seed-") else ""
        for line in open(p):
            r = json.loads(line)
            t = r.get("tags") or {}
            if r.get("stage") != "label" or "polar_id" not in t or "eda" not in r:
                continue
            f = t.get("field") or ""
            key = ("", 0.0) if not f else (f.split(":")[0], round(float(f.split(":")[1]), 6))
            g[prefix + t["polar_id"]][key] = r
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
        if min_ct is not None and min((c for _, c in tgt if c == c), default=0.0) / len(r0["symbols"]) < min_ct:
            continue
        base = pid.split(":")[-1]
        if not ct_dimers:                   # no alpha_CT target on any pair (dimer CT needs a dielectric)
            ct_ok = len(r0["symbols"]) > 2
        else:
            ab = base.split("_")[1].split("-") if base.startswith("pair_") else []
            ct_ok = not (len(ab) == 2 and ab[0] != ab[1]
                         and float(base.split("_r")[1]) > dimer_ct_cut(ab, ct_max_pair))
        out.append((pid, r0["symbols"], X, torch.tensor(tgt, dtype=torch.float64), ct_ok))
    return out


def total_polar_targets(files, species=("Na", "Cl"), exclude=()):
    """[(polar_id "<set>:<name>", symbols, coords A, target (3, 2) [alpha_total, nan] per axis bohr^3)] from labelled
    sets' finite-field polarizability jobs (ct_model rows with a "polarizability" tensor, in the input frame): the
    diagonal of the whole-cluster SCF polarizability. Trained against the model's alpha_pol + alpha_CT."""
    out = []
    for f in files:
        stem = Path(f).stem
        for line in open(f):
            r = json.loads(line)
            t = r.get("tags") or {}
            if r.get("stage") != "label" or r.get("polarizability") is None or not r.get("symbols"):
                continue
            if not set(r["symbols"]) <= set(species) or t.get("name") in set(exclude):
                continue
            d = np.diag(np.asarray(r["polarizability"], float))
            tgt = torch.tensor([[v, float("nan")] for v in d], dtype=torch.float64)
            out.append((f"{stem}:{t.get('name')}", r["symbols"], np.asarray(r["coords"], float), tgt))
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
    tot = torch.tensor([bool(c.labels.get("total", False)) for c in caches], device=a.device)[:, None]
    have = torch.isfinite(t[..., 0])
    t = torch.nan_to_num(t)
    z = torch.zeros_like(a[..., 0])
    # a total target (whole-cluster SCF alpha, no MI / CT split) is compared with alpha_pol + alpha_CT, in the first slot
    first = torch.where(tot, a[..., 0] + a[..., 1], a[..., 0])
    return (torch.where(have, (first - t[..., 0]) / n, z),
            torch.where(have & ok & ~tot, (a[..., 1] - t[..., 1]) / n, z))


def _species_floats(spec):
    """'Na=6,Cl=0.1' -> {'Na': 6.0, 'Cl': 0.1} (None or '' -> None)."""
    if not spec:
        return None
    return {k.strip(): float(v) for k, v in (t.split("=") for t in spec.split(",") if t.strip())}


def featurizer_for(a, tab):
    """The environment featurizer a run was trained with (train_ionct --env)."""
    if a.get("env", "soap") == "density":
        from rsfff.ff.complete import PROBE_NAMES, DensityEnvironment
        # runs from before --inv-form (2026-10-23) were trained on the magnitudes
        return DensityEnvironment(tab, probes=a.get("env_probes") or PROBE_NAMES, inv_form=a.get("inv_form") or "norm")
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


def _channel_sums(r, ratio, types=None):
    """ALMO-CTA channel energies (Hartree) summed per channel type, scaled by E_CT / sum PCT (how the pair
    functions were calibrated); None without CTA."""
    from ..ff.ionct.model import CHANNEL_TYPES
    CHANNEL_TYPES = types or CHANNEL_TYPES
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


def load_frames(ct_sets: str, walk_eda: str | None = None, ct_max_pair: float = 3.6, max_ho_ratio: float = 1.4,
                species=("Na", "Cl"), types=None, ct_dimers=True, extra_sets=(), extra_files=(), exclude=()):
    """[(symbols, coords A, labels, meta)], deduplicated.

    The ct label is kept only where ALMO's E_CT is a channel quantity: unlike pairs within ``dimer_ct_cut`` (``ct_max_pair`` for Na-Cl) and, for
    every frame with CTA, E_CT / sum PCT <= ``max_ho_ratio``. Above that the higher-order relaxation dominates --
    the ionic-covalent state mixing of a Cl- far from its Na+ partners, which cluster *fragments* (an MBE pair or
    triple without its environment) show and whole clusters do not (E_CT / sum PCT = 1.1-1.2 for every walk
    cluster; up to 4-10 for long-range Na-Cl pairs and Na2Cl+ triples). Whole clusters of 4 or more ions keep their
    label whatever the ratio (1.11-1.30, one at 1.51); above ~1.4 the fragments' CT per ion jumps from ~-3 to
    -5..-6 kJ/mol (e.g. a Cl- 5.5 A from a Na2 2+ pair, E_CT -70 kJ/mol at ratio 1.58).

    ``ct_dimers="like"`` keeps it on like-ion pairs only (Cl-Cl, Na-Na) and drops every unlike pair's.
    ``ct_dimers=False`` drops the ct label of every two-ion frame instead of cutting unlike pairs at ``ct_max_pair``:
    in vacuum the dimer's ionic configuration is not the ground state at long range, so dimer CT needs a dielectric
    (out of scope); cluster CT keeps its label. ``extra_sets``: more ct_model sets of the same kinds
    (seed-<salt>-scan / -mbe / -sat), read like the ct-* ones. ``extra_files``: labelled sets anywhere (a ct_model
    data/ct_sets/<instance>.jsonl, e.g. nacl-compressed), every frame its own parent (by its name tag), so the
    train / val split takes them one by one."""
    out, seen = [], set()

    def add(symbols, coords, pol, ct, meta, ho=float("nan"), chan=None, pauli=float("nan"), extra=None):
        key = _geom_key(symbols, coords)
        if key in seen:
            return
        seen.add(key)
        n = len(symbols)
        ct_mask = True
        if n == 2 and (not ct_dimers or (ct_dimers == "like" and symbols[0] != symbols[1])):
            ct_mask = False
        elif n == 2 and symbols[0] != symbols[1]:
            ct_mask = float(np.linalg.norm(np.subtract(coords[0], coords[1]))) <= dimer_ct_cut(symbols, ct_max_pair)
        whole = bool(meta.get("full")) and n >= 4
        if (np.isfinite(ho) or ho == float("inf")) and not whole:
            ct_mask = ct_mask and ho <= max_ho_ratio
        out.append((list(symbols), np.asarray(coords, float),
                    {"pol": pol / KJ, "ct": ct / KJ, "ct_mask": ct_mask, "chan": chan if ct_mask else None,
                     "pauli": pauli / KJ, **{k: v / KJ for k, v in (extra or {}).items()}},
                    {**meta, "n": n, "ho_ratio": ho}))

    sets_ = [(wf, Path(ct_sets) / f"{wf}.jsonl", False) for wf in ("ct-walk", "ct-mbe", "ct-sat", "ct-scan", *extra_sets)]
    sets_ += [(Path(f).stem, Path(f), True) for f in extra_files]
    for wf, p, per_frame in sets_:
        if not p.exists():
            if per_frame:
                raise FileNotFoundError(p)
            continue
        for line in open(p):
            r = json.loads(line)
            if r.get("stage") != "label" or "eda" not in r or not r.get("symbols") or len(r["symbols"]) < 2:
                continue
            if not set(r["symbols"]) <= set(species) or (r.get("tags") or {}).get("field"):
                continue
            t = r.get("tags") or {}
            if per_frame and t.get("name") in set(exclude):
                continue
            if "mbe_set" in t:
                parent = t["mbe_set"]
            elif per_frame:
                parent = f"{wf}_{t.get('name') or len(out)}"
            elif t.get("scan") == "pair":
                parent = f"scan_{t.get('scan_pair')}"
            else:
                parent = f"{wf}_{t.get('source', 'x')}"
            add(r["symbols"], r["coords"], r["eda"]["pol"], r["eda"]["ct"],
                {"workflow": wf, "parent": parent, "order": int(t.get("mbe_order", len(r["symbols"]))),
                 "full": t.get("mbe_order") == t.get("mbe_n") if "mbe_set" in t else True,
                 "cta_E": (r.get("cta") or {}).get("E")}, _ho_ratio(r), _channel_sums(r, _ho_ratio(r), types),
                r["eda"].get("mod_pauli", float("nan")),
                {k: r["eda"].get(e, float("nan")) for k, e in (("disp", "disp"), ("elec", "cls_elec"), ("int", "int"))})
    if walk_eda:
        from ase.io import read
        for f in sorted(Path(walk_eda).glob("*.xyz")):
            for k, a in enumerate(read(f, ":")):
                sym = a.get_chemical_symbols()
                if "eda_ct" not in a.info or not set(sym) <= set(species):
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


CACHE_VERSION = 4     # 2: equivariant environment blocks; 3: per-shell pseudo basis; 4: induced Pauli block A_pp


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
            refeat = getattr(featurizer, "inv_form", None) is not None and all(c.env for c in caches)
            if refeat:                  # the features in the requested invariant form, from the cached blocks
                from ..ff.complete import env_invariants
            for c, (sym, X, lab, meta) in zip(caches, frames):
                assert len(sym) == c.species.shape[0]
                c.labels, c.meta = dict(lab), dict(meta)
                if refeat:
                    c.feats = env_invariants(c.env, featurizer.probes, featurizer.inv_form).to(c.feats.dtype)
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


def force_set_monomers(files):
    """{symbol: isolated-ion energy (Hartree)} from the single-ion force rows of labelled sets (the seed-<salt>-cuts
    sets carry one per ion, at the same level as the frames: Na, Cl agree with the walk EDA's to 1e-9 Eh)."""
    out = {}
    for f in files:
        for line in open(f):
            r = json.loads(line)
            if r.get("stage") == "label" and len(r.get("symbols") or ()) == 1 and r.get("energy") is not None:
                out.setdefault(r["symbols"][0], float(r["energy"]))
    return out


def load_force_sets(files, mono, species=("Na", "Cl"), exclude=()):
    """load_force_frames from labelled sets (ct_model rows with "forces", Hartree / bohr in the input frame, and the
    total "energy", Hartree); the frame's "file" is "<set>:<name>" (--force-test may name one)."""
    out = []
    for f in files:
        stem = Path(f).stem
        for line in open(f):
            r = json.loads(line)
            t = r.get("tags") or {}
            if r.get("stage") != "label" or r.get("forces") is None or r.get("energy") is None:
                continue
            sym = r["symbols"]
            name = t.get("name") or t.get("cut_id") or r["job"][:12]       # seed-<salt>-cuts frames carry a cut_id
            if len(sym) < 2 or not set(sym) <= set(species) or name in set(exclude):
                continue
            key = f"{stem}:{name}"
            if not set(sym) <= set(mono):
                raise SystemExit(f"monomer energies: none for {sorted(set(sym) - set(mono))} ({key})")
            out.append((sym, np.asarray(r["coords"], float),
                        {"int": float(r["energy"]) - sum(mono[x] for x in sym), "forces": np.asarray(r["forces"], float)},
                        {"workflow": stem, "parent": key, "file": key, "index": 0, "n": len(sym)}))
    return out


def radial_table(tab, params_path, cache_dir=None):
    """The complete model's :class:`RadialTable` for this parameter table, cached on disk by the table's hash."""
    import hashlib
    from ..ff.complete import RadialTable
    key = hashlib.sha1(Path(params_path).read_bytes() + b"|table-v2").hexdigest()[:16]     # v2: per-shell ps rows
    path = Path(cache_dir) / f"table_{key}.pt" if cache_dir else None
    if path and path.exists():
        return torch.load(path, weights_only=False)
    t0 = time.time()
    table = RadialTable(tab.to("cpu"))     # built on the CPU (its grids are CPU tensors); the caller moves it
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


def pack_to_device(obj, device, stream=None):
    """``obj`` (a batch dict / list tree) with its tensors on ``device`` in one host-to-device copy per dtype: the
    tensors are packed into a pinned buffer, copied (non-blocking, on ``stream`` if given) and come back as views of
    the device buffer. Returns (tree, device buffers). On CPU: (obj, [])."""
    if device is None or str(device) == "cpu":
        return obj, []
    from torch.utils._pytree import tree_flatten, tree_unflatten
    leaves, spec = tree_flatten(obj)
    groups = {}
    for k, t in enumerate(leaves):
        if torch.is_tensor(t):
            groups.setdefault(t.dtype, []).append(k)
    out, bufs = list(leaves), []
    for dt, ks in groups.items():
        host = torch.empty(sum(leaves[k].numel() for k in ks), dtype=dt, pin_memory=True)
        torch.cat([leaves[k].reshape(-1) for k in ks], out=host)
        with torch.cuda.stream(stream) if stream is not None else _nullctx():
            buf = host.to(device, non_blocking=True)
        bufs.append(buf)
        o = 0
        for k in ks:
            m = leaves[k].numel()
            out[k] = buf[o:o + m].view(leaves[k].shape)
            o += m
    return tree_unflatten(out, spec), bufs


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def prefetch_batches(chunks, device, depth=2):
    """Yield ``collate(chunk)`` on ``device`` for each chunk. On CUDA a background thread collates up to ``depth``
    batches ahead and copies each with :func:`pack_to_device` on a side stream; the consumer's stream waits on the
    copy. On CPU: collate inline."""
    if device is None or str(device) == "cpu" or not torch.cuda.is_available():
        for c in chunks:
            yield collate(c)
        return
    import queue
    import threading
    q, stop = queue.Queue(maxsize=depth), threading.Event()
    side = torch.cuda.Stream(device=device)

    def work():
        try:
            for c in chunks:
                if stop.is_set():
                    return
                bt, bufs = pack_to_device(collate(c), device, side)
                ev = torch.cuda.Event()
                ev.record(side)
                item = (bt, bufs, ev)
                while not stop.is_set():
                    try:
                        q.put(item, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        except BaseException as e:                     # re-raised in the consumer
            q.put((e, None, None))
    th = threading.Thread(target=work, daemon=True, name="prefetch_batches")
    th.start()
    try:
        for _ in range(len(chunks)):
            bt, bufs, ev = q.get()
            if isinstance(bt, BaseException):
                raise bt
            cur = torch.cuda.current_stream(device)
            cur.wait_event(ev)
            for b in bufs:                              # allocated on the side stream, used on this one
                b.record_stream(cur)
            yield bt
    finally:
        stop.set()


def force_terms(model, frames, tab, table, chan):
    """Energy (per ion, kJ/mol) and force (per component, kJ/mol/A) errors of a list of force frames through the
    batched differentiable precompute: (B,), (sum N, 3). Forces are -dE/dx with the graph kept for training."""
    from ..ff.complete import batched_precompute
    from ..ff.ionct.params import ATOMIC_NUMBER as Zmap
    dev = table.device
    pos = torch.tensor(np.concatenate([X for _, X, *_ in frames]), dtype=torch.float64, device=dev, requires_grad=True)
    Z = torch.tensor([Zmap[x] for sym, *_ in frames for x in sym], device=dev)
    fr = torch.cat([torch.full((len(sym),), k, dtype=torch.long) for k, (sym, *_) in enumerate(frames)]).to(dev)
    bt = batched_precompute(pos, Z, fr, tab, table, chan, inv_form=model.inv_form)
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


def resolve_exchange(a):
    """Fix ``a.exchange`` (and the ``pseudo`` / ``xpol`` flags build_model reads) from the arguments: an explicit
    --exchange wins, then --pseudo / --xpol; auto is xpol for the complete model with the frozen terms."""
    if a.exchange == "auto":
        if a.pseudo and a.xpol:
            raise SystemExit("--pseudo and --xpol exclude each other")
        a.exchange = ("pseudo" if a.pseudo else "xpol" if (a.xpol or (a.model == "complete" and a.frozen))
                      else "none")
    elif (a.pseudo and a.exchange != "pseudo") or (a.xpol and a.exchange != "xpol"):
        raise SystemExit(f"--exchange {a.exchange} contradicts --pseudo / --xpol")
    if a.exchange == "xpol" and not (a.model == "complete" and a.frozen):
        raise SystemExit("--exchange xpol needs --model complete --frozen (the pair Pauli energies)")
    a.pseudo, a.xpol = a.exchange == "pseudo", a.exchange == "xpol"
    print(f"exchange polarization: {a.exchange}", flush=True)


def _env_skip(a):
    """The per-readout linear skip of the shared environment: off by default since 2026-11-11 (--env-skip turns it
    back on); runs from before carry no ``env_skip`` and kept it unless they passed --env-no-skip."""
    if a.get("env_skip") is not None:
        return bool(a["env_skip"])
    return not a.get("env_no_skip", False)


def _translate_f(path, symbols, key="f"):
    from ..ff.complete.contract import load_translate_f
    return load_translate_f(path, symbols, key)


def _contract_A(path, symbols):
    from ..ff.complete.contract import load_contract_A
    return load_contract_A(path, symbols)


def _wself(path, symbols, tab):
    """(S, 4) [kV, kT, w, s1] of the bounded field energy from a json {ion: {kV, kT, w, s1}} (free-ion point-charge
    fits). An ion without an entry gets its table values: kT = 2 b / a, gamma = 12 w a^2 / s1 at s1 = 0.01 a, kV = 0."""
    d = json.loads(Path(path).read_text()) if path else {}
    d = d.get("ions", d)
    rows = []
    for k, x in enumerate(symbols):
        a_, g_, b_ = float(tab.alpha[k]), float(tab.hyper_gamma[k]), float(tab.hyper_b[k])
        s1 = 0.01 * a_
        e = d.get(x) or {"kV": 0.0, "kT": 2 * b_ / a_, "w": g_ * s1 / (12 * a_ * a_), "s1": s1}
        rows.append([float(e["kV"]), float(e["kT"]), float(e["w"]), float(e["s1"])])
    return torch.tensor(rows, dtype=torch.float64)


def build_model(a: dict, tab, chan, n_feat, feat_scale=None, feature_names=None):
    """The model of a run from its arguments (train_ionct's, as saved in model.pt): the ion CT model or, with
    ``model == "complete"``, the complete model on the environment heads."""
    g = a.get
    if g("model", "ionct") == "complete":
        from ..ff.complete import CompleteModel
        flux = g("ct_model", "direct") in ("flux", "twostate")   # the joint solve replaces the gap factor / saturation / Stark
        return CompleteModel(tab, chan, n_feat, feature_names=feature_names, hidden=g("hidden", 64),
                             feat_scale=feat_scale, c_range=tuple(g("c_range", (0.05, 1.25))), confine=g("confine"),
                             pseudo=g("pseudo", False), tie_c=g("tie_c", False), species_emb=g("species_emb", False),
                             aniso=not g("no_aniso", False), ct_gap=not g("no_ct_gap", False),
                             dphi_scale=g("dphi_scale", 0.05), frozen=g("frozen", False),
                             pauli_moments=g("pauli_moments", False),
                             pauli_form=g("pauli_form", "slater"), ct_stark=g("ct_stark", False) and not flux,
                             pauli_ii=g("pauli_ii", False),
                             ct_sc=g("ct_sc", 0), env_cap=g("env_cap", 0.0), env_log=g("env_log", 0.0),
                             ct_sat=g("ct_sat", False) and not flux, ct_q_cap=g("ct_q_cap", 0.0), gap_cap=g("gap_cap", 4.0),
                             ct_sat_ion=g("ct_sat_ion", False) and not flux, ct_w_fixed=g("ct_w_fixed", False) and not flux, xpol=g("xpol", False),
                             ct_model=g("ct_model", "direct"), flux_solver=g("flux_solver", "atomic"),
                             ct_outer=g("ct_outer", 1), ct_outer_tol=g("ct_outer_tol", 1e-10),
                             ct_gap_min=g("ct_gap_min", 0.03), ct_gap_learn=g("ct_gap_learn", False),
                             head_frontier=g("head_frontier", None),
                             gap_shift=g("gap_shift", "acceptor"),
                             xpol_perp=bool(g("xpol_perp", False)) and bool(g("xpol", False)),   # absent: before it was the default
                             xpol_kperp_init=_species_floats(g("xpol_kperp_init", None)),
                             xpol_kappa_head=not g("xpol_kappa_const", False), xpol_kappa_fixed=g("xpol_kappa_fixed", None), sr_features=g("sr_features", False),
                             env_species=g("env_species", "none"), env_species_emb=g("env_species_emb", "descriptor"),
                             env_species_dim=g("env_species_dim", 8), env_extra_latent=g("env_extra_latent", False),
                             tie_disp_range=g("tie_disp_range", False),
                             xpol_perp_weight=g("xpol_perp_weight", "pauli"), xpol_kappa_env=g("xpol_kappa_env", "overlap"),   # absent: a checkpoint from before scalefree
                             env_channels=g("env_channels", False), env_latent=g("env_latent", 4),
                             env_mode=g("env_mode", "shared"),
                             env_two_body=tuple(g("env_two_body", ("pauli", "disp", "ct")) or ()),
                             ct_q_head=not g("ct_q_const", False), pin_a0=g("pin_a0", False),
                             hyperpol=g("hyperpol", False), hyperpol_form=g("hyperpol_form", "oneshot"),   # absent: before the fix
                             hyperpol_b=g("hyperpol_b", False),     # absent: legendre runs before 2026-11-11 (gamma only)
                             xpol_form=g("xpol_form", "lam"), env_skip=_env_skip(a),
                             xpol_weights=g("xpol_weights", False), xpol_w1=g("xpol_w1", "rank"),
                             ct_b_pin=g("ct_b_pin", "none"),     # absent: checkpoints from before the pinned default
                             env_two_latent=g("env_two_latent", False), env_pairs=tuple(g("env_pairs", None) or ("ww", "ws", "ss")),
                             wself=(_wself(g("wself_params", None), tab.symbols, tab) if g("hyperpol_form", "oneshot") == "bounded" else None),
                             wself_caps=tuple(g("wself_caps", (0.5, 3.0, 1.5))), wself_learn=g("wself_learn", False),
                             breath_form=g("breath_form", "shape"),
                             xpol_dipole=g("xpol_dipole", "parent"), xpol_trans_pi=g("xpol_trans_pi", "zero"),
                             translate_f=_translate_f(g("translate_cpks", None), tab.symbols) if g("xpol_dipole", "parent") == "translate" else None,
                             xpol_quad=g("xpol_quad", "parent"),
                             strain_f=_translate_f(g("translate_cpks", None), tab.symbols, "f2") if g("xpol_quad", "parent") == "strain" else None,
                             contract_A=_contract_A(g("contract_cpks", None), tab.symbols) if g("breath_form", "shape") == "contract" else None,
                             env_extra=tuple(g("env_extra", ()) or ()),
                             env_extra_terms=tuple(g("env_extra_terms", ("ct",)) or ()),
                             env_field=g("env_field", False), ct_legacy_head=g("ct_legacy_head", False),
                             env_ct_pen=g("env_ct_pen", False),
                             env_ct_gated=g("env_ct_gated", False), env_gate=tuple(g("env_gate", None) or (1e-7, 1e-5)),
                             env_extra_capacity=g("env_extra_capacity", "small"))   # absent: a run from before "full"
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
            bt, _ = pack_to_device(collate(chunk), dev)
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


#: validation errors are capped at this (kJ/mol) per frame in the logged / selection MAEs (``--val-cap``; 0 = no cap):
#: one extreme frame (a squeezed pair with a 1700 kJ/mol Pauli energy) otherwise moves a ~200-frame MAE by 0.5
VAL_CAP = 20.0


def _mae(d, cap):
    d = np.abs(d)
    return float((np.minimum(d, cap) if cap > 0 else d).mean())


def summarize(rows, cap=None):
    cap = VAL_CAP if cap is None else cap

    def stats(sel):
        if not sel:
            return {}
        p = np.array([r["pol"] - r["pol_ref"] for r in sel])
        c = np.array([r["ct"] - r["ct_ref"] for r in sel if r["ct_mask"]])
        cr = np.array([r["ct_ref"] for r in sel if r["ct_mask"]])
        out = {"n": len(sel), "pol_mae": _mae(p, cap), "pol_mae_raw": float(np.abs(p).mean()),
               "pol_rms": float(np.sqrt((p ** 2).mean()))}
        big = [np.abs(p)]
        pa = np.array([r["pauli"] - r["pauli_ref"] for r in sel if np.isfinite(r.get("pauli_ref", np.nan))])
        if len(pa) and any(r["pauli"] != 0 for r in sel):
            out["pauli_mae"], out["pauli_mae_raw"] = _mae(pa, cap), float(np.abs(pa).mean())
            big.append(np.abs(pa))
        if any(r["pauli"] != 0 for r in sel):          # a model with frozen terms: dispersion and total interaction
            for key in ("disp", "int"):
                d = np.array([r[key] - r[f"{key}_ref"] for r in sel if (key != "int" or r["ct_mask"])
                              and np.isfinite(r.get(f"{key}_ref", np.nan)) and np.isfinite(r.get(key, np.nan))])
                if len(d):
                    out[f"{key}_mae"], out[f"{key}_mae_raw"] = _mae(d, cap), float(np.abs(d).mean())
        if len(c):
            big.append(np.abs(c))
            out.update(ct_mae=_mae(c, cap), ct_mae_raw=float(np.abs(c).mean()), ct_rms=float(np.sqrt((c ** 2).mean())),
                       ct_rel=float(np.abs(c).sum() / max(np.abs(cr).sum(), 1e-12)))
        if cap > 0:                       # pol / Pauli / CT errors above the cap (counted per term) and the worst one
            b = np.concatenate(big)
            out["n_over_cap"], out["max_err"] = int((b > cap).sum()), float(b.max()) if len(b) else 0.0
        return out
    res = {"all": stats(rows)}
    for key in ("full", "subset"):
        res[key] = stats([r for r in rows if (r["full"] if key == "full" else not r["full"])])
    for n in sorted({r["n"] for r in rows}):
        res[f"n={n}"] = stats([r for r in rows if r["n"] == n])
    return res


def main():
    global VAL_CAP
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
    ap.add_argument("--salts", nargs="*", default=[],
                    help="ct_model seed sets to add, by salt (lif lii kbr csi ...): seed-<salt>-{scan,mbe,sat,polar}; "
                         "each salt's <salt>6_03, <salt>8_07, <salt>9_09 clusters and tet4 shells go to the test split")
    ap.add_argument("--ct-no-dimers", action="store_true",
                    help="no ct (or alpha_CT) label on any two-ion frame, in place of the --ct-max-pair cut")
    ap.add_argument("--extra-sets", nargs="*", default=[],
                    help="more labelled sets (jsonl paths, e.g. ct_model data/ct_sets/nacl-compressed.jsonl), per-frame parents")
    ap.add_argument("--env-cap", type=float, default=0.0,
                    help="bound the log environment factors (CT charges, Pauli, C6, exclusion, breathing) to +-cap")
    ap.add_argument("--env-sat", type=float, default=0.0,
                    help="heads read m tanh(I / m), m = this x max |I| over the training atoms (per invariant)")
    ap.add_argument("--ct-q-cap", type=float, default=0.0,
                    help="bound on the CT charges' log environment factors (0: --env-cap)")
    ap.add_argument("--gap-cap", type=float, default=4.0, help="bound on the CT gap / Stark exponent")
    ap.add_argument("--ct-sat", action="store_true",
                    help="two-state CT channels: the environment lowers the gap, |E_c| <= the coupling W_c")
    ap.add_argument("--ct-w-fixed", action="store_true",
                    help="with --ct-sat: the CT coupling from the isolated ions only; the environment acts through the gap")
    ap.add_argument("--ct-sat-ion", action="store_true",
                    help="with --ct-sat: saturate per donor and per acceptor (all of an ion's channels at once)")
    ap.add_argument("--env-log", type=float, default=0.0,
                    help="heads read sign(I) log(1 + |I| / k) of the scaled invariants (k = this; 0 = linear)")
    ap.add_argument("--polar-min-ct", type=float, default=None,
                    help="drop alpha targets whose alpha_CT per ion is below this on any axis (bohr^3; e.g. -1)")
    ap.add_argument("--polar-ct-max-pair", type=float, default=POLAR_CT_MAX_PAIR,
                    help="Na-Cl distance (A; scaled by the ionic radii for other salts) beyond which an unlike pair's "
                         "alpha_CT is not a target (its alpha_MI still is); independent of --ct-max-pair for E_CT")
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
    ap.add_argument("--w-pauli-prior", type=float, default=0.0,
                    help="complete + pauli-form shells: pull of the Pauli charges / exponent scales toward the frozen "
                         "fit's values in the ion table (squared log deviation; 0: free)")
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
    ap.add_argument("--inv-form", default="sq", choices=("sq", "norm"),
                    help="density env: the vector / tensor invariants as squared norms |x|^2 (smooth where x = 0, e.g. "
                         "at lattice sites) or as magnitudes |x| (the form before 2026-10-23: a cone at x = 0)")
    ap.add_argument("--model", default="ionct", choices=("ionct", "complete"),
                    help="complete: environment heads, anisotropic alpha, CT gap factor (direct CT, density env)")
    ap.add_argument("--species-emb", action="store_true", help="complete: species embedding on the heads' input")
    ap.add_argument("--no-aniso", action="store_true", help="complete: isotropic polarizability")
    ap.add_argument("--no-ct-gap", action="store_true", help="complete: no environment gap factor on the CT")
    ap.add_argument("--dphi-scale", type=float, default=0.05, help="complete: scale (Eh/e) of the CT gap potential")
    ap.add_argument("--frozen", action="store_true", help="complete: frozen Pauli (env qP) and dispersion (env C6) terms")
    ap.add_argument("--pauli-moments", action="store_true",
                    help="complete + frozen: induced Pauli dipole / quadrupole from the overlap gradients (polytensor Pauli kernel)")
    ap.add_argument("--pauli-form", default="slater", choices=("slater", "shells"),
                    help="complete: Pauli exclusion density = one Slater per species, or the ion's own permanent shells "
                         "with per-shell weights (and the drive basis per shell)")
    ap.add_argument("--ct-stark", action="store_true",
                    help="complete: second-order Stark lowering of the CT gap by the fields at donor and acceptor "
                         "(the CT state's extra polarizability; gives alpha_CT perpendicular to a pair)")
    ap.add_argument("--pauli-ii", action="store_true",
                    help="complete: induced-induced Pauli kernel in the induction solve (kappa_i kappa_j A_pp)")
    ap.add_argument("--conf-nohead", action="store_true",
                    help="ion CT model only (the complete model has no confinement head)")
    ap.add_argument("--ct-sc", type=int, default=0,
                    help="complete: self-consistent CT, N fixed-point iterations of (CT charges -> induction -> CT)")
    ap.add_argument("--hyper", action="store_true",
                    help="ion CT model only (the complete model has no confinement head; see --hyperpol)")
    ap.add_argument("--conf-form", default="tanh", choices=("tanh", "mono", "sq"),
                    help="ion CT model only (the complete model has no confinement head: confinement is exchange "
                         "polarization's)")
    ap.add_argument("--force-data", nargs="*", default=None,
                    help="complete + frozen: extxyz dirs with total energies and forces (atomic units); trains total "
                         "interaction energies and forces through the tabulated, differentiable precompute")
    ap.add_argument("--w-force", type=float, default=0.0, help="weight of the force loss (kJ/mol/A per component)")
    ap.add_argument("--w-fe", type=float, default=1.0, help="weight of the force frames' total energy (kJ/mol per ion)")
    ap.add_argument("--force-every", type=int, default=1, help="force step every k-th energy step (weight x k)")
    ap.add_argument("--force-batch", type=int, default=16)
    ap.add_argument("--force-test", nargs="*", default=list(FORCE_TEST),
                    help="force files (stems; '<set>:<name>' for --force-sets frames) held out")
    ap.add_argument("--force-sets", nargs="*", default=[],
                    help="labelled sets (ct_model jsonl) whose force jobs are force frames (total energy + forces)")
    ap.add_argument("--polar-sets", nargs="*", default=[],
                    help="labelled sets (ct_model jsonl) whose finite-field polarizability jobs are alpha targets: the "
                         "whole-cluster alpha (diagonal), against the model's alpha_pol + alpha_CT")
    ap.add_argument("--polar-almo", default=True, action=argparse.BooleanOptionalAction,
                    help="the ALMO finite-field alpha_MI / alpha_CT targets (ct-polar, seed-<salt>-polar)")
    ap.add_argument("--train-subset", default=None,
                    help="JSON {eda: [frame geometry keys (_geom_key)], force: ['<file>#<index>']}: train on these "
                         "frames only; the val / test split, held-out force frames and alpha targets are unchanged")
    ap.add_argument("--exclude-names", nargs="*", default=[],
                    help="frames (name tag) of --extra-sets / --force-sets / --polar-sets left out entirely")
    ap.add_argument("--exchange", default="auto", choices=("auto", "xpol", "pseudo", "none"),
                    help="the exchange-polarization drive: xpol = the quadratic functional (pair Pauli energies as a PSD "
                         "quadratic in the induced moments and breathing amplitudes, one solve with the induction; the "
                         "breathing amplitude joins the solve), pseudo = the learned pseudo-potential drive + separate "
                         "breathing term, none. auto: xpol for --model complete --frozen, else pseudo with --pseudo, "
                         "else none. The confinement head (--confine, --conf-form) is independent of it")
    ap.add_argument("--xpol", action="store_true", help="same as --exchange xpol")
    ap.add_argument("--flux-solver", default="atomic", choices=("atomic", "channel"),
                    help="flux CT: atomic = channels eliminated, an N x N solve in atomic charges (default); "
                         "channel = the original Schur complement over the channels (reference; identical "
                         "results, cost ~ C^3 with C ~ N^2 channels)")
    ap.add_argument("--ct-outer", type=int, default=1,
                    help="twostate CT: outer (Newton) steps; 1 = linear response about the isolated two-state minimum, "
                         "> 1 = up to that many steps towards the exact minimum")
    ap.add_argument("--ct-outer-tol", type=float, default=1e-10, help="twostate CT: stop when max |dq| is below this")
    ap.add_argument("--ct-gap-min", type=float, default=0.03,
                    help="twostate CT: smooth floor of the channel gap (Hartree), see rsfff.ff.complete.twostate")
    ap.add_argument("--ct-gap-learn", action="store_true",
                    help="twostate CT: train per-species shifts of the table I, A (default: frozen at the table values)")
    ap.add_argument("--ct-model", default="direct", choices=("direct", "gap", "flux", "twostate"),
                    help="complete: direct = environment-dependent donor/acceptor CT charges (+ the multiplicative gap "
                         "factor / two-state saturation); gap = fixed per-species couplings W and the environment only "
                         "in an additive gap I - A + phiD(i) - phiA(j) + overlap shifts of the frontier levels; flux = "
                         "explicit split charges per channel in the xpol solve, compliance from the direct-CT energy "
                         "(needs xpol; turns off the gap factor, --ct-sat/--ct-sat-ion/--ct-stark/--ct-w-fixed; no "
                         "frontier potentials in the cache); twostate = two-state channels (population from the direct-CT energy and "
                         "a gap I - A + permanent potential difference - electron-hole attraction) coupled to the xpol "
                         "solve, see rsfff.ff.complete.twostate")
    ap.add_argument("--gap-shift", default="acceptor", choices=("acceptor", "both"),
                    help="--ct-model gap: overlap shift of the acceptor level only (gap opens with overlap) or of both "
                         "frontier levels")
    ap.add_argument("--head-frontier", default=None, action=argparse.BooleanOptionalAction,
                    help="complete: the donor / acceptor probe blocks as head inputs (default: yes for --ct-model "
                         "direct, no for gap)")
    ap.add_argument("--xpol-kperp-init", default=None,
                    help="--xpol-perp: starting p per species, e.g. 'Na=6,Cl=0.1' (k_perp,ij = (p_i + p_j)/2; default Na 6, Cl 0.1, others 0.1). "
                         "Pair E_pol does not see k_perp (no field across a bond), so it is learned from alpha and "
                         "asymmetric clusters only and moves slowly from a small start")
    ap.add_argument("--xpol-perp-weight", default="pauli", choices=("pauli", "sqrt_overlap"),
                    help="--xpol-perp: the pair weight of the perpendicular stiffening: the Pauli energy E0_ij, or "
                         "E_s sqrt(O_ij / O_s) (half its decay rate; equal at Na-Cl 2.6 A)")
    ap.add_argument("--xpol-kappa-env", default="scalefree", choices=("scalefree", "overlap"),
                    help="xpol: the kappa head's inputs: the 3 scale-free coordination descriptors (n_eff - 1, eta1, "
                         "eta2; exactly constant along any pair, so kappa = kappa0 on every pair; default), or the 18 "
                         "overlap invariants (R dependent: the head can learn distance switches)")
    ap.add_argument("--val-cap", type=float, default=VAL_CAP,
                    help="cap (kJ/mol) on each frame's error in the logged / checkpoint-selection validation MAEs "
                         "(the over-cap count and the largest error are logged; 0 = plain MAE)")
    ap.add_argument("--env-species", default="none", choices=("none", "film", "concatenate"),
                    help="--env-channels: condition the shared environment's encoders (latent trunk, two-body branches) "
                         "on the centre atom's species embedding; the readouts stay shared")
    ap.add_argument("--env-species-emb", default="descriptor", choices=("descriptor", "table"),
                    help="--env-species: the embedding from the free ion's physical descriptors (an MLP; transfers to "
                         "unseen ions) or a learned vector per species")
    ap.add_argument("--env-species-dim", type=int, default=8, help="--env-species: embedding width")
    ap.add_argument("--tie-disp-range", action="store_true",
                    help="complete (shells Pauli): the Tang-Toennies damping exponent of each species = its Pauli range "
                         "(outermost shell exponent x the learned Pauli scale); the pair still uses sqrt(b_i b_j)")
    ap.add_argument("--w-env-mid", type=float, default=0.0,
                    help="--env-channels: weight of the low-overlap penalty on the environment log-factors, "
                         "sum_p m u_p^2 with m = 1 / (1 + (S / --env-mid-scale)^2), S the atom's total overlap")
    ap.add_argument("--env-mid-scale", type=float, default=5e-4,
                    help="--w-env-mid: the total overlap S^w (bohr^-3) at which the penalty weight is 1/2 "
                         "(Na-Cl ~3.4 A, Cl-Cl ~3.9 A; contact ~2e-3 or more)")
    ap.add_argument("--ct-dimers", default=None, choices=("all", "none", "like"),
                    help="CT labels on two-ion frames: all (unlike pairs cut at --ct-max-pair), none (= --ct-no-dimers) "
                         "or like (like-ion pairs only: Cl-Cl, Na-Na scans)")
    ap.add_argument("--env-extra-latent", action="store_true",
                    help="--env-channels: the extra inputs (--env-extra / --env-field / --env-ct-pen / --env-ct-gated) "
                         "join the three-body channels as the shared trunk's input -- one latent for every readout -- "
                         "instead of a side branch for the readouts of --env-extra-terms")
    ap.add_argument("--sr-features", action="store_true",
                    help="complete model: drop the long-range invariants (|V1|^2, |V2|^2, V1.S1 of every probe: full "
                         "Coulomb field / field gradient) from the heads on the density invariants (confinement, "
                         "--ct-legacy-head); V0_pen and the overlaps stay")
    ap.add_argument("--xpol-kappa-fixed", type=float, default=None, metavar="K",
                    help="xpol: kappa = K on every species and component, not learned and no environment head (K = 1: "
                         "the tabulated overlap change d_ij / O_ij unscaled)")
    ap.add_argument("--xpol-kappa-const", action="store_true",
                    help="xpol: kappa = kappa0 per species and component, no environment head (the exchange term's R "
                         "dependence is then the Pauli energy and the tabulated overlap derivatives alone)")
    ap.add_argument("--xpol-perp", default=True, action=argparse.BooleanOptionalAction,
                    help="xpol (default on; --no-xpol-perp for the k_par term alone): the perpendicular exchange stiffening E0 k_perp (|v_i|^2 |z_i|^2 - (v_i.z_i)^2) per "
                         "pair side and component block (k_perp >= 0 per species): confinement across the overlap "
                         "direction, in place of a confinement head")
    ap.add_argument("--init-from", default=None, help="model.pt of a previous run to start from (same architecture)")
    ap.add_argument("--device", default="cpu", help="cpu | cuda | cuda:N (complete model; batches are moved per step)")
    ap.add_argument("--w-disp", type=float, default=0.0, help="weight of the dispersion energy vs ALMO disp")
    ap.add_argument("--w-int", type=float, default=0.0,
                    help="weight of the total interaction energy (cls_elec label + pauli + disp + pol + ct) vs ALMO int")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--env-channels", action="store_true",
                    help="complete (m33): every environment log-factor (Pauli weight, C6, xpol kappa per component, "
                         "CT q_D / q_A) from one shared environment on the 17 two- and three-body channels of the "
                         "overlap- and electrostatics-weighted neighbour densities (src/ff/complete/channels.py) in "
                         "place of head_frozen / head_xpol / head_ctq")
    ap.add_argument("--env-latent", type=int, default=4, help="--env-channels: latent width of the trunk(s)")
    ap.add_argument("--env-mode", default="shared", choices=("shared", "per_term"),
                    help="--env-channels: one trunk for every parameter (shared) or one trunk, with its own latent, "
                         "per term (Pauli, dispersion, kappa, CT)")
    ap.add_argument("--env-two-body", nargs="*", default=["pauli", "disp", "ct"],
                    help="--env-channels: the terms whose readouts see the two-body channels S^w, S^s (dimer-level "
                         "corrections); the others are exactly constant on every dimer. Any of pauli disp kappa ct")
    ap.add_argument("--env-two-latent", action="store_true",
                    help="--env-channels: the two-body channels S^w, S^s join the shared trunk's input and no readout has "
                         "a two-body branch (--env-two-body is ignored): one latent for the dimer-level and the many-body "
                         "dependence of every parameter")
    ap.add_argument("--env-pairs", nargs="+", choices=("ww", "ws", "ss", "Sw", "Ss", "Tww", "Tws", "Tss"),
                    default=["ww"],
                    help="--env-channels: the channels the environment reads -- families ww: plain overlap weights (and "
                         "S^w); ws, ss: dressed by the pair electrostatics (and S^s); or exact groups Sw, Ss (the two-body "
                         "sums) and Tww, Tws, Tss (one three-body family each), combined by union. The feature-reduction "
                         "ablations. Default ww (S^w + T^ww): the dressed blocks did not help in the b128 runs")
    ap.add_argument("--env-extra", nargs="*", default=[],
                    help="--env-channels: density-environment invariants (feature names, e.g. acceptor:S0 donor:S0) "
                         "given as an extra branch to the readouts of --env-extra-terms")
    ap.add_argument("--env-extra-terms", nargs="*", default=["ct"],
                    help="terms whose readouts get --env-extra / --env-field")
    ap.add_argument("--env-field", action="store_true",
                    help="--env-channels: the permanent electrostatic field and field gradient at each site "
                         "(|E|^2, |grad E|^2 of the induction drive) as an extra branch for --env-extra-terms (default ct)")
    ap.add_argument("--env-extra-capacity", default="full", choices=("full", "small"),
                    help="--env-channels: size of the extra branch's MLP (full = hidden x depth, as a head; small = "
                         "16 x 1, the first m33 runs)")
    ap.add_argument("--env-ct-gated", action="store_true",
                    help="--env-channels: the diffuse probes' (donor, acceptor, breath) invariants up to l = 2 with the "
                         "full Coulomb fields switched off smoothly with each probe's overlap with the neighbour "
                         "(--env-gate): no long-range contributions in the features; an extra branch for "
                         "--env-extra-terms (default ct)")
    ap.add_argument("--env-gate", type=float, nargs=2, default=None, metavar=("ON", "FULL"),
                    help="--env-ct-gated: the probe overlap at which a neighbour's field starts to count and counts "
                         "fully (smoothstep in log10; default 1e-7 1e-5, reach ~9 A for NaCl)")
    ap.add_argument("--env-ct-pen", action="store_true",
                    help="--env-channels: the diffuse probes' (donor, acceptor, breath) invariants up to l = 2 with "
                         "penetration-only Coulomb blocks as an extra branch for --env-extra-terms (default ct)")
    ap.add_argument("--w-curv", type=float, default=0.0,
                    help="geometric smoothness prior: weight of the mean squared second difference (per A^2) of every "
                         "environment log-factor along random per-atom displacements (--env-channels)")
    ap.add_argument("--curv-every", type=int, default=4, help="--w-curv: every k-th step (weight x k)")
    ap.add_argument("--curv-frames", type=int, default=8, help="--w-curv: training frames per evaluation")
    ap.add_argument("--curv-delta", type=float, default=0.02, help="--w-curv: displacement (A)")
    ap.add_argument("--ct-legacy-head", action="store_true",
                    help="--env-channels: CT q_D, q_A from the m32 head on the 42 density invariants (hybrid)")
    ap.add_argument("--ct-q-const", action="store_true",
                    help="--env-channels: CT q_D, q_A = per-species constants (no environment readout)")
    ap.add_argument("--hyperpol", action="store_true",
                    help="complete: the free-ion dipole hyperpolarizabilities (ion table hyperpol: gamma, field-gradient "
                         "b) added to each atom's polarizability at its permanent field / gradient, one-shot, no "
                         "learned parameters")
    ap.add_argument("--wself-params", default=None,
                    help="--hyperpol-form bounded: json {ion: {kV, kT, w, s1}} of the bounded field energy (free-ion "
                         "point-charge fits); ions without an entry use the table's gamma and b")
    ap.add_argument("--wself-caps", type=float, nargs=3, default=(0.5, 3.0, 1.5), metavar=("SOFT", "STIFF", "T"),
                    help="--hyperpol-form bounded: saturation of the penetration exponent (softening, stiffening) and of "
                         "the gradient exponent's norm")
    ap.add_argument("--wself-learn", action="store_true", help="--hyperpol-form bounded: train the constants (default: frozen)")
    ap.add_argument("--hyperpol-form", choices=("legendre", "oneshot", "bounded"), default="legendre",
                    help="complete --hyperpol: legendre (gamma and the field-gradient b, see --no-hyperpol-b; the ion's internal energy expanded about its polarized "
                         "state: alpha(E_p) on site, a field offset and a constant -- exact energy and moment for an "
                         "isolated ion) or oneshot (gamma + field-gradient b inside the linear solve: 6x the "
                         "hyperpolarizable energy; reproduces checkpoints from 2026-11-09)")
    ap.add_argument("--xpol-dipole", choices=("parent", "translate"), default="parent",
                    help="--xpol-form gram: the dipole channel's exchange from the CPKS parent (vW Gram) or from the rigid "
                         "translation of the shells (exact drive and along-bond second order; --translate-cpks)")
    ap.add_argument("--translate-cpks", default=None,
                    help="--xpol-dipole translate: the CPKS split of the induced dipole between the shells "
                         "(scripts/ion_translate_cpks.py output); stored in the checkpoint")
    ap.add_argument("--xpol-quad", choices=("parent", "strain"), default="parent",
                    help="--xpol-form gram: the quadrupole channel's exchange from the CPKS parent (vW Gram) or from the "
                         "volume-preserving strain of the shells (exact drive and second order; the CPKS l = 2 split "
                         "'f2' of --translate-cpks)")
    ap.add_argument("--xpol-trans-pi", choices=("zero", "exact"), default="zero",
                    help="--xpol-dipole translate: the perpendicular second order (zero, or the rigid shift's exact "
                         "softening S'/R)")
    ap.add_argument("--breath-form", choices=("shape", "contract"), default="shape",
                    help="complete --xpol: the l = 0 response as the fitted breathing shape (shape) or as the contraction "
                         "of the permanent shells (contract: two modes per breathing ion, CPKS stiffness from "
                         "--contract-cpks, Coulomb drive and inter-ion coupling; theory/parameter_free_xpol.tex)")
    ap.add_argument("--contract-cpks", default=None,
                    help="--breath-form contract: the CPKS compliance of the contraction modes "
                         "(scripts/ion_contract_cpks.py output); stored in the checkpoint, not needed to load it")
    ap.add_argument("--env-skip", action="store_true",
                    help="--env-channels: add back the per-readout linear skip a_p . T on the three-body channels (off by "
                         "default since 2026-11-11: every parameter's environment dependence beyond the two-body branch "
                         "goes through the shared conditioned latent; widen --env-latent if needed)")
    ap.add_argument("--env-no-skip", action="store_true", help=argparse.SUPPRESS)   # the default now; kept for old scripts
    ap.add_argument("--no-hyperpol-b", dest="hyperpol_b", action="store_false",
                    help="--hyperpol-form legendre: gamma only (default: gamma and the field-gradient b term)")
    ap.add_argument("--xpol-form", choices=("lam", "gram"), default="lam",
                    help="complete --xpol: lam (fitted kappa + head, lam u^2, k_perp) or gram (parameter-free: kappa = 1/2 "
                         "fixed and the on-site Gram blocks of the perturbed von Weizsaecker orbital in the neighbours' "
                         "densities; theory/parameter_free_xpol.tex)")
    ap.add_argument("--ct-b-pin", choices=("none", "all", "donor", "donor-scale", "donor-shell", "acceptor-float"), default="acceptor-float",
                    help="complete: the direct-CT exponents from the ion table (acceptor: outer electrostatic shell for "
                         "anions, Fukui for cations; cation donor: its shell). The anion donor: all = its outer shell, "
                         "donor = learned freely, donor-scale = outer shell x a learned factor >= 1 (started at the Fukui ratio), "
                         "donor-shell = the same factor started at the shell; acceptor-float = as all, with each cation acceptor "
                         "a bounded scale (0.61-1.65 x) around its Fukui exponent; none = all four learned "
                         "(the default before 2026-11-11). Default acceptor-float (since 2026-11-13): same energies as all, forces 0.2-0.9 kJ/mol/A better, "
                         "Na+ settles at 0.85 x Fukui")
    ap.add_argument("--xpol-weights", action="store_true",
                    help="--xpol-form gram: free the two magnitudes -- W1 = 2 kappa on the drive (per species and rank, "
                         "started at 1/2) and W2 on each atom's on-site Gram blocks (per species, started at 1), both "
                         "with a readout of the shared latent (--xpol-kappa-const: per-species constants only)")
    ap.add_argument("--xpol-w1", choices=("rank", "shared"), default="rank",
                    help="--xpol-weights: one drive weight per rank (dipole, quadrupole, breathing) or one shared by the three")
    ap.add_argument("--pin-a0", action="store_true",
                    help="complete: the breathing stiffness a0 stays at the ion table's monomer value (breath_lna frozen)")
    ap.add_argument("--hull-k", type=int, default=6,
                    help="write RUN/hull.pt: the training atoms' convex hull in this many whitened PCA "
                         "components of the invariants (rsfff.coverage; 0 = off)")
    a = ap.parse_args()
    if a.cache_dir:                      # the model's lazily built pair tables share the precompute cache's directory
        os.environ.setdefault("RSFFF_TABLE_CACHE", str(a.cache_dir))
    VAL_CAP = a.val_cap
    resolve_exchange(a)
    # CT labels on dimers: --ct-dimers (all / none / like) or the older --ct-no-dimers
    ct_dimers_mode = {"all": True, "none": False, "like": "like", None: not a.ct_no_dimers}[a.ct_dimers]
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tab, chan = IonTable.load(a.params), CT.ChannelTable.load(a.params)
    feat = featurizer_for(vars(a), tab)
    if a.env == "density":
        print(f"environment: density projections, {feat.n_feat} invariants", flush=True)
    from ..ff.ionct.model import channel_types
    frames = load_frames(a.ct_sets, a.walk_eda, a.ct_max_pair, a.max_ho_ratio, species=tab.symbols,
                         types=channel_types(tab.symbols), ct_dimers=ct_dimers_mode,
                         extra_sets=[f"seed-{x}-{k}" for x in a.salts for k in ("scan", "mbe", "sat")],
                         extra_files=a.extra_sets, exclude=a.exclude_names)
    a.test = list(a.test) + [f"{x}{n}" for x in a.salts for n in ("6_03", "8_07", "9_09")]
    print(f"ct label kept for {sum(f[2]['ct_mask'] for f in frames)} of {len(frames)} frames", flush=True)
    tags = split(frames, a.test, seed=a.seed)
    if a.holdout_scans:
        tags = ["test" if (m["workflow"].endswith("scan") and m["n"] == 2) else t for t, (*_, m) in zip(tags, frames)]
        print(f"held out {sum(m['workflow'].endswith('scan') and m['n'] == 2 for *_, m in frames)} pair-scan frames", flush=True)
    print(f"{len(frames)} frames: " + ", ".join(f"{t} {tags.count(t)}" for t in ("train", "val", "test")), flush=True)
    if a.model == "complete":
        if a.ct_form != "direct" or a.env != "density":
            raise SystemExit("--model complete needs --ct-form direct and --env density")
    lean = (a.ct_form == "direct") and not a.full_precompute
    if a.ct_model in ("flux", "twostate"):
        if a.model != "complete" or not a.xpol:
            raise SystemExit(f"--ct-model {a.ct_model} needs --model complete with xpol")
        off = [f for f in ("ct_sat", "ct_sat_ion", "ct_stark", "ct_w_fixed") if getattr(a, f, False)]
        if off:
            print(f"--ct-model {a.ct_model}: ignoring {', '.join('--' + f.replace('_', '-') for f in off)}", flush=True)
        a.no_ct_gap = True
    frontier = a.model == "complete" and not a.no_ct_gap
    cache_path = a.cache
    if cache_path is None and a.cache_dir:
        cache_path = Path(a.cache_dir) / f"cache_{cache_key(a.params, a.env, lean, frames, frontier)}.pt"
        print(f"cache: {cache_path}", flush=True)
    caches = build_cache(frames, tab, chan, feat, cache_path or out / "cache.pt", lean=lean, workers=a.precompute_workers,
                         frontier=frontier)
    tr = [c for c, t in zip(caches, tags) if t == "train"]
    if a.train_subset:                  # data-efficiency runs: train on these frames only (val / test unchanged)
        sub = json.load(open(a.train_subset))
        keep = set(sub.get("eda") or ())
        n0 = len(tr)
        tr = [c for c, t, f in zip(caches, tags, frames) if t == "train" and _geom_key(f[0], f[1]) in keep]
        print(f"train subset {a.train_subset}: {len(tr)} of {n0} training frames", flush=True)
    va = [c for c, t in zip(caches, tags) if t == "val"]
    te = [c for c, t in zip(caches, tags) if t == "test"]
    F = torch.cat([c.feats for c in tr])
    nz = F.abs().sum(1) > 0
    scale = F[nz].std(0).clamp(min=1e-6)
    model = build_model(vars(a), tab, chan, F.shape[1], scale, getattr(feat, "feature_names", None))
    Xc = None
    if getattr(model, "env", None) is not None:      # the channels' training scale (an init checkpoint overrides it)
        Xc = channel_matrix(model, tr)
        nzc = Xc.abs().sum(1) > 0
        model.chan_scale.copy_(Xc[nzc].std(0).clamp(min=1e-12))
        if model.env_field:
            Fv = model.field_invariants(torch.cat([c.L for c in tr]).double())
            nzf = Fv.abs().sum(1) > 0
            model.field_scale.copy_(Fv[nzf].std(0).clamp(min=1e-12))
        if model.env_ct_pen:
            Pv = pen_matrix(model, tr)
            nzp = Pv.abs().sum(1) > 0
            model.pen_scale.copy_(Pv[nzp].std(0).clamp(min=1e-12))
        if model.env_ct_gated:
            Gv = gate_matrix(model, tr)
            nzg = Gv.abs().sum(1) > 0
            model.gate_scale.copy_(Gv[nzg].std(0).clamp(min=1e-12))
            print(f"gated diffuse-probe fields: gate {model.env_gate[0]:g} -> {model.env_gate[1]:g} (probe overlap), "
                  f"reach {model.gate_reach():.2f} A", flush=True)
        print(f"env channels: {Xc.shape[1]} ({model.env.mode} trunk, latent {model.env.latent}; two-body for "
              f"{[p for p in model.env.names if model.env.two_body[p]]}), {int(nzc.sum())} training atoms with pairs",
              flush=True)
    dev = torch.device(a.device)
    polar_tr, polar_te = [], []
    if a.w_alpha > 0:
        almo = polar_targets(a.ct_sets, ct_dimers=ct_dimers_mode is True,
                             files=("ct-polar", *[f"seed-{x}-polar" for x in a.salts]),
                             min_ct=a.polar_min_ct, ct_max_pair=a.polar_ct_max_pair) if a.polar_almo else []
        total = total_polar_targets(a.polar_sets, species=tab.symbols, exclude=a.exclude_names)
        for pid, sym, X, tgt, ok, is_tot in ([(*t, False) for t in almo] + [(*t, False, True) for t in total]):
            lab = {"alpha": tgt, "ct_ok": ok, **({"total": True} if is_tot else {})}
            c = precompute(sym, X, tab, chan, feat, lab, {"polar_id": pid}, lean=lean, frontier=frontier)
            base = pid.split(":")[-1]
            held = (pid in set(a.polar_test) or (a.holdout_scans and base.startswith("pair_"))
                    or (":" in pid and base.startswith("shell_tet4")))
            (polar_te if held else polar_tr).append(c)
        if total:
            print(f"alpha targets: {len(almo)} ALMO alpha_MI / alpha_CT, {len(total)} whole-cluster totals "
                  f"({', '.join(a.polar_sets)})", flush=True)
        print(f"alpha targets: {len(polar_tr)} train, {len(polar_te)} held out", flush=True)
        pb_tr, pb_all = to_device(polar_batch(polar_tr), dev), to_device(polar_batch(polar_tr + polar_te), dev)
    # to the device only now: the alpha targets' precompute above, like the frames', runs on the CPU tensors
    if dev.type != "cpu":
        tab = tab.to(dev)
        model.tab = tab
        model.to(dev)
        print(f"device: {dev} ({torch.cuda.get_device_name(dev) if dev.type == 'cuda' else ''})", flush=True)
    if a.init_from:
        ck = torch.load(a.init_from, weights_only=False)
        missing, unexpected = model.load_state_dict(ck["state"], strict=False)
        print(f"init from {a.init_from}: {len(missing)} new parameters, {len(unexpected)} dropped", flush=True)
    if a.env_sat > 0 and hasattr(model, "feat_sat"):    # this run's training range, after any init
        m = a.env_sat * (F[nz] / scale).abs().max(0).values
        model.feat_sat.copy_(m.to(model.feat_sat))
        if Xc is not None:
            mc = a.env_sat * (Xc / model.chan_scale.cpu()).abs().max(0).values
            model.chan_sat.copy_(mc.to(model.chan_sat))
            if model.env_field:
                Fv = model.field_invariants(torch.cat([c.L for c in tr]).double())
                model.field_sat.copy_((a.env_sat * (Fv / model.field_scale.cpu()).abs().max(0).values).to(model.field_sat))
            if model.env_ct_pen:
                Pv = pen_matrix(model, tr)
                model.pen_sat.copy_((a.env_sat * (Pv / model.pen_scale.cpu()).abs().max(0).values).to(model.pen_sat))
            if model.env_ct_gated:
                Gv = gate_matrix(model, tr)
                model.gate_sat.copy_((a.env_sat * (Gv / model.gate_scale.cpu()).abs().max(0).values).to(model.gate_sat))
        print(f"env-sat: heads saturate at {a.env_sat} x the training range (median m {float(m.median()):.2f})",
              flush=True)
    force_tr, force_te, table = [], [], None
    if a.force_data or a.force_sets:
        if not (a.model == "complete" and a.frozen):
            raise SystemExit("--force-data needs --model complete --frozen (the model must own every term)")
        mono = {**force_set_monomers(a.force_sets), **monomer_energies([a.walk_eda] if a.walk_eda else [])}
        if len(mono) < 2:
            raise SystemExit("monomer energies: --walk-eda files with fragment_energies are needed")
        print(f"monomer energies (Hartree): {mono}", flush=True)
        ff = load_force_frames(a.force_data or [], mono) + load_force_sets(a.force_sets, mono, tab.symbols,
                                                                          a.exclude_names)
        held = set(a.force_test)
        force_tr = [f for f in ff if f[3]["file"] not in held]
        if a.train_subset:
            fk = set(json.load(open(a.train_subset)).get("force") or ())
            n0 = len(force_tr)
            force_tr = [f for f in force_tr if f"{f[3]['file']}#{f[3]['index']}" in fk]
            print(f"train subset: {len(force_tr)} of {n0} training force frames", flush=True)
        force_te = [f for f in ff if f[3]["file"] in held]
        print(f"force frames: {len(force_tr)} train, {len(force_te)} held out ({', '.join(sorted(held))})", flush=True)
        table = radial_table(tab, a.params, a.cache_dir).to(dev)
    if a.w_curv > 0:
        if getattr(model, "env", None) is None:
            raise SystemExit("--w-curv needs --env-channels")
        if table is None:
            table = radial_table(tab, a.params, a.cache_dir).to(dev)
    curv_log = []
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=a.lr * 0.02)
    best, best_state, log = math.inf, None, []
    n_sing = 0
    n_nonfin = 0
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        random.shuffle(tr)
        tot = 0.0
        batches = prefetch_batches([tr[k:k + a.batch] for k in range(0, len(tr), a.batch)], dev)
        for step, k in enumerate(range(0, len(tr), a.batch)):
            bt = next(batches)
            o = model(bt)
            ep_, ec_, ech_ = loss_terms(o, bt)
            loss = (a.w_pol * ep_ ** 2 + a.w_ct * ec_ ** 2 + a.w_chan * (ech_ ** 2).sum(1)).mean()
            parts = [("energy", loss.detach())]
            if a.w_env_mid > 0:              # right after model(bt): the readouts on the model are this batch's
                loss = loss + a.w_env_mid * model.env_mid_penalty(a.env_mid_scale)
                parts.append(("env_mid", loss.detach()))
            if a.w_pauli > 0:
                loss = loss + a.w_pauli * (pauli_term(o, bt) ** 2).mean()
                parts.append(("pauli", loss.detach()))
            if a.w_disp > 0:
                loss = loss + a.w_disp * (disp_term(o, bt) ** 2).mean()
                parts.append(("disp", loss.detach()))
            if a.w_pauli_prior > 0:
                loss = loss + a.w_pauli_prior * model.pauli_prior()
                parts.append(("pauli_prior", loss.detach()))
            if a.w_int > 0:
                loss = loss + a.w_int * (int_term(o, bt) ** 2).mean()
                parts.append(("int", loss.detach()))
            if polar_tr and step % a.alpha_every == 0:      # the stencil batch is the costly part: every k-th step, k x weight
                ap_, ac_ = polar_loss(model, pb_tr)
                loss = loss + a.alpha_every * a.w_alpha * (ap_ ** 2 + a.w_alpha_ct * ac_ ** 2).mean()
                parts.append(("alpha", loss.detach()))
            if a.w_smooth > 0:
                loss = loss + a.w_smooth * model.smoothness_penalty(bt, a.smooth_sigma)
                parts.append(("smooth", loss.detach()))
            if a.w_curv > 0 and step % a.curv_every == 0:
                cp = curvature_penalty(model, random.sample(tr, min(a.curv_frames, len(tr))), tab, table, chan,
                                       a.curv_delta)
                loss = loss + a.curv_every * a.w_curv * cp
                parts.append(("curv", loss.detach()))
                curv_log.append(float(cp.detach()))
            if force_tr and a.w_force > 0 and step % a.force_every == 0:
                fb = random.sample(force_tr, min(a.force_batch, len(force_tr)))
                try:
                    fe_, ff_, _ = force_terms(model, fb, tab, table, chan)
                    loss = loss + a.force_every * (a.w_fe * (fe_ ** 2).mean() + a.w_force * (ff_ ** 2).mean())
                    parts.append(("force", loss.detach()))
                except torch.linalg.LinAlgError as err:     # a singular response matrix: keep the evidence, skip
                    n_sing += 1
                    dump = out / f"singular_{n_sing:03d}.pt"
                    if n_sing <= 5:
                        torch.save({"state": {k_: v.detach().cpu() for k_, v in model.state_dict().items()},
                                    "frames": fb, "error": str(err), "epoch": ep, "step": step}, dump)
                    print(f"ep {ep} step {step}: force step skipped ({err}); "
                          f"frames {[m_['file'] for *_, m_ in fb]}" + (f" -> {dump}" if n_sing <= 5 else ""), flush=True)
            opt.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            if not (torch.isfinite(loss) and torch.isfinite(gn)):
                # a non-finite loss or gradient: keep the evidence (first 5), skip the step so the parameters survive
                n_nonfin += 1
                bad = next((nm for nm, v in parts if not torch.isfinite(v)), "gradient only")
                names = [f.meta.get("parent", f.meta.get("file", "?")) for f in tr[k:k + a.batch]]
                if n_nonfin <= 5:
                    torch.save({"state": {k_: v.detach().cpu() for k_, v in model.state_dict().items()},
                                "batch": names, "parts": [(nm, float(v)) for nm, v in parts], "epoch": ep,
                                "step": step, "grad_norm": float(gn)}, out / f"nonfinite_{n_nonfin:03d}.pt")
                print(f"ep {ep} step {step}: non-finite loss/gradient (first bad term: {bad}, |g| {float(gn):.3g}); "
                      f"step skipped", flush=True)
                opt.zero_grad()
                if n_nonfin > 50:
                    raise RuntimeError("more than 50 non-finite steps: giving up")
                continue
            opt.step()
            tot += loss.item() * len(bt["labels"]["n"])
        sched.step()
        if ep % a.log_every == 0 or ep == a.epochs:
            sv = summarize(evaluate(model, va))["all"]
            score = (sv.get("pol_mae", 0) + sv.get("ct_mae", 0) + (sv.get("pauli_mae", 0) if a.w_pauli > 0 else 0)
                     + (sv.get("disp_mae", 0) if a.w_disp > 0 else 0))
            if score < best:
                best, best_state = score, {k: v.detach().clone() for k, v in model.state_dict().items()}
                # the best model so far, written as it is found (model.pt, replaced atomically; the final save below
                # is the same state): a running or interrupted job can be analysed
                tmp = out / "model.pt.tmp"
                torch.save({"state": {k: v.cpu() for k, v in best_state.items()}, "args": vars(a), "n_feat": F.shape[1],
                            "feature_names": getattr(feat, "feature_names", None), "epoch": ep, "val_score": score}, tmp)
                tmp.replace(out / "model.pt")
            log.append({"epoch": ep, "train_loss": tot / len(tr), **{f"val_{k}": v for k, v in sv.items()}})
            msg = ""
            if polar_tr:
                with torch.no_grad():
                    ap_, ac_ = polar_loss(model, pb_all)
                has_tot = any(c.labels.get("total") for c in pb_all["caches"])
                msg = (f" | alpha/ion |err| {'pol|total' if has_tot else 'pol'} {ap_.abs().mean():.2f} "
                       f"ct {ac_.abs().mean():.2f} bohr^3")
            if "pauli_mae" in sv:
                msg = f" pauli MAE {sv['pauli_mae']:.2f}" + (f" disp MAE {sv['disp_mae']:.2f}" if "disp_mae" in sv else "") + msg
            if force_te:
                fm = force_metrics(model, force_te, tab, table, chan)
                msg += f" | held-out forces: E/ion MAE {fm['e_mae']:.2f} kJ/mol, F rms {fm['f_rms']:.1f} kJ/mol/A"
                log[-1].update({f"force_{k}": v for k, v in fm.items()})
            if getattr(model, "env", None) is not None and table is not None and va:
                # the parameters' geometric curvature on fixed validation frames and displacements (a diagnostic
                # whether or not --w-curv trains it): mean squared second difference per A^2 of the log-factors
                st = torch.random.get_rng_state()
                torch.manual_seed(1234)
                with torch.no_grad():
                    cv = float(curvature_penalty(model, va[:16], tab, table, chan, a.curv_delta))
                torch.random.set_rng_state(st)
                log[-1]["val_curv"] = cv
                msg += f" | curv {cv:.3g}" + (f" (train {np.mean(curv_log):.3g})" if curv_log else "")
                curv_log.clear()
            print(f"ep {ep:4d} loss {tot / len(tr):.4f} | val pol MAE {sv.get('pol_mae', 0):.2f} ct MAE "
                  f"{sv.get('ct_mae', 0):.2f} kJ/mol{msg}"
                  + (f" | val >{VAL_CAP:g}: {sv['n_over_cap']} (max {sv['max_err']:.0f})" if "n_over_cap" in sv else "")
                  + f" [{time.time() - t0:.0f}s]", flush=True)
    if best_state is not None:          # --epochs 0: evaluate the (--init-from) model as is
        model.load_state_dict(best_state)
    if model.gap_form == "density":
        print("coupling ln w / gamma / lambda: " + "; ".join(
            f"{t} {model.lnw[k]:.2f} / {model.gamma[k]:.3f} / {model.lam[k]:.3f}" for k, t in enumerate(model.ctypes))
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
    if a.hull_k > 0 and a.env == "density":
        write_hull(a, out, feat, F, scale, tr, va + te, force_tr, force_te,
                   model=model if getattr(model, "env", None) is not None else None)


def pen_matrix(model, caches, bs=128):
    """(N, 15) raw diffuse-probe penetration invariants (``env_ct_pen``) of every atom of the caches."""
    out, dev = [], next(model.parameters()).device
    with torch.no_grad():
        for k in range(0, len(caches), bs):
            out.append(model.diffuse_pen_invariants(to_device(collate(caches[k:k + bs]), dev)).cpu())
    return torch.cat(out)


def gate_matrix(model, caches, bs=128):
    """(N, 15) raw overlap-gated diffuse-probe invariants (``env_ct_gated``) of every atom of the caches."""
    out, dev = [], next(model.parameters()).device
    with torch.no_grad():
        for k in range(0, len(caches), bs):
            out.append(model.diffuse_gated_invariants(to_device(collate(caches[k:k + bs]), dev)).cpu())
    return torch.cat(out)


def curvature_penalty(model, caches, tab, table, chan, delta=0.02):
    """Mean squared second difference (per A^2) of every environment log-factor of the shared environment along one
    random per-atom displacement field (unit vectors x delta A) of the given frames: the geometric smoothness prior
    (a fold in the features shows up as curvature of the parameters in R, which a feature-space prior cannot see)."""
    from ..ff.complete import batched_precompute
    from ..ff.ionct.params import ATOMIC_NUMBER as Zmap
    dev = table.device
    pos = torch.cat([c.pos for c in caches]).to(dev, torch.float64)
    Z = torch.tensor([Zmap[tab.symbols[int(s)]] for c in caches for s in c.species], device=dev)
    fr = torch.cat([torch.full((c.species.shape[0],), k, dtype=torch.long) for k, c in enumerate(caches)]).to(dev)
    xi = torch.randn_like(pos)
    xi = xi / xi.norm(dim=1, keepdim=True).clamp(min=1e-12)
    us = []
    for sgn in (0.0, 1.0, -1.0):
        bt = batched_precompute(pos + sgn * delta * xi, Z, fr, tab, table, chan, inv_form=model.inv_form)
        model.env_prepare(bt, bt["species"])
        us.append(model._env_u)
    pen = 0.0
    for k in us[0]:
        pen = pen + (((us[1][k] - 2 * us[0][k] + us[2][k]) / delta ** 2) ** 2).mean()
    return pen


def channel_matrix(model, caches, bs=256):
    """(N, 17) raw m33 environment channels of every atom of ``caches`` (FrameCache) or of raw frames
    (symbols, coords, ...), batched per frame."""
    from ase.data import atomic_numbers
    out = []
    model_dev = next(model.parameters()).device
    with torch.no_grad():
        for k in range(0, len(caches), bs):
            chunk = caches[k:k + bs]
            pos, sp, fr = [], [], []
            for q, c in enumerate(chunk):
                if hasattr(c, "species"):
                    s_, x_ = c.species, c.pos
                else:
                    sym, X = c[0], c[1]
                    s_ = model.tab.species_index(torch.tensor([atomic_numbers[x] for x in sym]))
                    x_ = torch.as_tensor(X, dtype=torch.float64)
                sp.append(s_.to(model_dev))
                pos.append(x_.to(model_dev, torch.float64))
                fr.append(torch.full((len(s_),), q, dtype=torch.long, device=model_dev))
            out.append(model.env_raw(torch.cat(pos), torch.cat(sp), torch.cat(fr)).cpu())
    return torch.cat(out) if out else torch.zeros(0, 17, dtype=torch.float64)


def write_hull(a, out, feat, F, scale, tr, held, force_tr, force_te, model=None):
    """RUN/hull.pt: the convex hull of the training atoms' invariants (EDA frames + force frames) in the network's
    input space, and the fraction of held-out atoms outside it. With the m33 environment (``model`` given) the space
    is that of the 17 channels the heads read. Never fails the run."""
    try:
        from ase.data import atomic_numbers
        from ..coverage import INSIDE_TOL, FeatureHull
        if model is not None:
            Xtr = channel_matrix(model, list(tr) + list(force_tr)).numpy()
            hull = FeatureHull.fit(Xtr, model.chan_scale.double().cpu().numpy(), k=a.hull_k,
                                   feature_names=list(model.channel_names), verbose=True,
                                   meta={"run": str(out), "params": str(a.params), "n_train_frames": len(tr),
                                         "n_force_frames": len(force_tr), "space": "env_channels"})
            hull.save(out / "hull.pt")
            Xh = channel_matrix(model, list(held) + list(force_te)).numpy()
            if len(Xh):
                d = hull.distance(Xh)
                print(f"hull: {100 * float((d > INSIDE_TOL).mean()):.1f} % of {len(d)} held-out atoms outside "
                      f"-> {out / 'hull.pt'}", flush=True)
            return

        def ff_feats(frames):
            return [feat(torch.as_tensor(X, dtype=torch.float64), torch.tensor([atomic_numbers[x] for x in sym]),
                         torch.zeros(len(sym), dtype=torch.long)).inv_feats.double().numpy() for sym, X, *_ in frames]

        Ftr = np.concatenate([F.double().cpu().numpy(), *ff_feats(force_tr)])
        hull = FeatureHull.fit(Ftr, scale.double().cpu().numpy(), k=a.hull_k,
                               feature_names=getattr(feat, "feature_names", None), verbose=True,
                               meta={"run": str(out), "params": str(a.params), "n_train_frames": len(tr),
                                     "n_force_frames": len(force_tr)})
        hull.save(out / "hull.pt")
        Fh = [c.feats.double().cpu().numpy() for c in held] + ff_feats(force_te)
        if Fh:
            d = hull.distance(np.concatenate(Fh))
            print(f"hull: {100 * float((d > INSIDE_TOL).mean()):.1f} % of {len(d)} held-out atoms outside "
                  f"-> {out / 'hull.pt'}", flush=True)
    except Exception as e:      # the hull is a by-product; a failure here must not cost the trained model
        print(f"hull: not written ({type(e).__name__}: {e})", flush=True)


if __name__ == "__main__":
    main()
