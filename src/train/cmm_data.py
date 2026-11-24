"""Data of the cmm trainer: the labelled ALMO-EDA sets, the force frames, the polarizability targets, and the
batches the model takes (positions -> :meth:`rsfff.ff.cmm.CmmModel.precompute`, on the model's device).

Moved from ``train_ionct`` (tag complete-explore-final). The ALMO-CTA channel sums are no longer labels; the CTA
data still decides where a frame's E_CT is a channel quantity (``max_ho_ratio``).
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..ff.cmm.params import ATOMIC_NUMBER

KJ = 2625.4996

DEFAULT_TEST = ("walk_nacl10_111", "walk_nacl10_112", "walk_nacl10_113", "walk_nacl10_114",
                "nacl4_03", "nacl5_07", "nacl6_11", "nacl8_15", "nacl9_19")

F_SET = 0.002

#: ct-polar geometries held out of the alpha loss (the rest train it)
POLAR_TEST = ("pair_Na-Cl_r2.250", "pair_Na-Cl_r2.750", "pair_Na-Cl_r3.250", "pair_Cl-Cl_r3.500", "pair_Cl-Cl_r4.500",
              "pair_Cl-Cl_r5.500", "shell_tet4_r2.800", "trimer_Na-Cl-Na_r2.800_a90", "trimer_Na-Cl-Na_r2.800_a120",
              "trimer_Na-Cl-Na_r2.800_a150", "trimer_Na-Cl-Na_r2.800_a180")

#: Na-Cl pairs beyond this (A) carry ionic-covalent mixing in ALMO's alpha_CT: no CT target there
POLAR_CT_MAX_PAIR = 3.6

# Shannon ionic radii (A, 6-coordinate): scale the Na-Cl dimer CT cut to the other salts
IONIC_RADIUS = {"Li": 0.76, "Na": 1.02, "K": 1.38, "Rb": 1.52, "Cs": 1.67, "F": 1.33, "Cl": 1.81, "Br": 1.96, "I": 2.20}

FORCE_TEST = ("nacl14", "nacl16", "nacl18", "nacl20")

BOHR = 0.529177210903
KJ_ANG = KJ / BOHR                    # Hartree/bohr -> kJ/mol/A
DEBYE = 2.541746473                   # e bohr -> Debye


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


def vacuum_gap(symbols, coords, ions) -> float:
    """The smallest vacuum gap (Hartree) to moving an electron from an anion i to a cation j of the frame, with the
    other ions as point charges: I[i] - A[j] + phi(i) - phi(j) - 1 / r_ij. ``ions``: {symbol: (charge, I, A)}
    (Hartree; :func:`gap_ions`). Near or below zero the all-ionic state is not the vacuum ground state (the
    ionic / radical crossing), and the EDA's CT there is state mixing, not a channel quantity. inf without such a pair."""
    X = np.asarray(coords, float) / 0.529177210903
    q = np.array([ions[a][0] for a in symbols])
    don, acc = np.nonzero(q < 0)[0], np.nonzero(q > 0)[0]
    if not len(don) or not len(acc):
        return float("inf")
    d = np.linalg.norm(X[:, None] - X[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    phi = (q[None, :] / d).sum(1)
    I = np.array([ions[symbols[i]][1] for i in don])
    A = np.array([ions[symbols[j]][2] for j in acc])
    return float((I[:, None] - A[None, :] + phi[don][:, None] - phi[acc][None, :] - 1.0 / d[np.ix_(don, acc)]).min())


def gap_ions(tab) -> dict:
    """{symbol: (charge, I_donor, A_acceptor)} of a species table, for :func:`vacuum_gap`."""
    return {a: (float(tab.charge[k]), float(tab.I_donor[k]), float(tab.A_acceptor[k])) for k, a in enumerate(tab.symbols)}


def load_frames(ct_sets: str, walk_eda: str | None = None, ct_max_pair: float = 3.6, max_ho_ratio: float = 1.4,
                species=("Na", "Cl"), ct_dimers=True, extra_sets=(), extra_files=(), exclude=(), ct_min_gap=0.0,
                ions=None, dipole_charged=True):
    """[(symbols, coords A, labels, meta)], deduplicated.

    Dipole labels (:func:`dipole_label`) are kept on every frame by default. A charged system's dipole depends on its
    origin, which the records carry (``dipole_origin``), and :func:`induced_dipole` takes the ions' point charges off
    about that origin, so the label is origin-independent. ``dipole_charged=False`` keeps them on neutral frames only
    (the dipole of every frame with a net charge dropped, ``meta["dipole_dropped"]``). The charge is the record's
    ``charge`` (else the sum of its ``fragment_charges``, else of the ions' formal charges).

    ``ct_min_gap`` > 0 (Hartree, with ``ions`` = :func:`gap_ions` of the species table) drops the ct label of every
    frame whose :func:`vacuum_gap` is below it: a geometric criterion for the ionic / radical crossing region that
    needs no ALMO-CTA (0.04 drops the anion-rich fragments with a distant cation and the long unlike pairs). It adds
    to the two criteria below; ``max_ho_ratio=inf``, ``ct_dimers=True`` and a large ``ct_max_pair`` leave it alone.

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

    def add(symbols, coords, pol, ct, meta, ho=float("nan"), pauli=float("nan"), extra=None, dipole=None, charge=None):
        key = _geom_key(symbols, coords)
        if key in seen:
            return
        seen.add(key)
        n = len(symbols)
        if dipole and not dipole_charged and _net_charge(symbols, charge) != 0:
            dipole, meta = None, {**meta, "dipole_dropped": True}
        ct_mask = True
        if n == 2 and (not ct_dimers or (ct_dimers == "like" and symbols[0] != symbols[1])):
            ct_mask = False
        elif n == 2 and symbols[0] != symbols[1]:
            ct_mask = float(np.linalg.norm(np.subtract(coords[0], coords[1]))) <= dimer_ct_cut(symbols, ct_max_pair)
        whole = bool(meta.get("full")) and n >= 4
        if (np.isfinite(ho) or ho == float("inf")) and not whole:
            ct_mask = ct_mask and ho <= max_ho_ratio
        gap = vacuum_gap(symbols, coords, ions) if ions is not None else float("nan")
        if ct_min_gap > 0:
            if ions is None:
                raise ValueError("ct_min_gap needs ions (gap_ions(tab))")
            ct_mask = ct_mask and gap >= ct_min_gap
        out.append((list(symbols), np.asarray(coords, float),
                    {"pol": pol / KJ, "ct": ct / KJ, "ct_mask": ct_mask,
                     "pauli": pauli / KJ, **{k: v / KJ for k, v in (extra or {}).items()}, **(dipole or {})},
                    {**meta, "n": n, "ho_ratio": ho, "vacuum_gap": gap}))

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
                 "full": t.get("mbe_order") == t.get("mbe_n") if "mbe_set" in t else True}, _ho_ratio(r),
                r["eda"].get("mod_pauli", float("nan")),
                {k: r["eda"].get(e, float("nan")) for k, e in (("disp", "disp"), ("elec", "cls_elec"), ("int", "int"))},
                dipole_label(r.get("dipole"), r.get("dipole_origin")),
                r["charge"] if r.get("charge") is not None else
                (sum(r["fragment_charges"]) if r.get("fragment_charges") else None))
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
                                                                             ("int", "eda_int"))},
                    dipole=dipole_label(_extxyz_dipole(a), a.info.get("dipole_origin"),
                                        "au" if a.info.get("units") == "atomic" else "debye"),
                    charge=a.info.get("charge"))
    return out


FORMAL_CHARGE = {"Li": 1, "Na": 1, "K": 1, "Rb": 1, "Cs": 1, "F": -1, "Cl": -1, "Br": -1, "I": -1}


def _extxyz_dipole(a):
    """An extxyz frame's ``dipole``: ase's reader moves it from the comment line onto the frame's calculator."""
    if a.info.get("dipole") is not None:
        return a.info["dipole"]
    return (getattr(a.calc, "results", None) or {}).get("dipole")


def _net_charge(symbols, charge=None):
    """The frame's net charge: ``charge`` when given, else the sum of the ions' formal charges (None if a species has
    none -- treated as charged, so its dipole is dropped unless charged frames are asked for)."""
    if charge is not None:
        return int(round(float(charge)))
    if not set(symbols) <= set(FORMAL_CHARGE):
        return None
    return sum(FORMAL_CHARGE[s] for s in symbols)


def dipole_label(dipole, origin=None, unit="debye") -> dict:
    """The labels of a record's total SCF dipole moment, {} without one. Convention of the sets: ``dipole`` = three
    components in Debye in the frame of the record's coordinates, taken about ``dipole_origin`` (Angstrom; the
    coordinate origin when absent -- the dipole of a charged system depends on it). ``unit="au"`` for a dipole in
    e bohr (the extxyz frames of rsfff_al.store_io, ``units=atomic``: Q-Chem's standard orientation, about its origin,
    the frame of their own positions). Stored as ``dipole`` (e bohr) and ``dipole_origin`` (Angstrom); the loss uses
    :func:`induced_dipole`."""
    if dipole is None:
        return {}
    if unit not in ("debye", "au"):
        raise ValueError(f"dipole unit {unit!r}: 'debye' or 'au'")
    mu = np.asarray(dipole, float).reshape(-1) / (1.0 if unit == "au" else DEBYE)
    o = np.zeros(3) if origin is None else np.asarray(origin, float).reshape(-1)
    if mu.shape != (3,) or o.shape != (3,) or not np.isfinite(mu).all():
        raise ValueError(f"dipole {dipole!r} (origin {origin!r}): three finite components each")
    return {"dipole": mu, "dipole_origin": o}


def induced_dipole(f, tab) -> np.ndarray:
    """(3,) e bohr: the frame's dipole label minus the dipole of its ions' point charges about the same origin,
    mu - sum_i Q_i (r_i - o) -- independent of the origin, and what the model's -dE/dF is (the induced dipoles plus
    the transferred charges times their positions). nan without a label."""
    if "dipole" not in f.labels or f.labels.get("dipole_off", False):
        return np.full(3, np.nan)
    Q = tab.charge[tab.species_index(f.Z.to(tab.Z.device))].cpu().numpy()
    return f.labels["dipole"] - (Q[:, None] * (f.pos.numpy() - f.labels["dipole_origin"])).sum(0) / BOHR


def mask_dipoles(frames, min_gap: float) -> int:
    """Take the dipole labels of the frames whose :func:`vacuum_gap` (``meta["vacuum_gap"]``) is below ``min_gap``
    (Hartree) out of the loss and the metrics, in place (``labels["dipole_off"]``); the number taken out. Near and
    beyond the ionic / radical crossing the SCF state has moved a large part of an electron, and its dipole (errors of
    1 - 2 D per ion in a model that is fine elsewhere) would be most of a squared dipole loss. Frames without an
    anion - cation pair (gap inf) and frames without a gap keep theirs."""
    n = 0
    for f in frames:
        if "dipole" not in f.labels:
            continue
        g = f.meta.get("vacuum_gap")
        off = min_gap > 0 and g is not None and np.isfinite(g) and g < min_gap
        f.labels["dipole_off"] = bool(off)
        n += bool(off)
    return n


def dipole_report(frames, tab) -> dict:
    """Checks of the dipole labels that need no reference: {"n": frames with one, "pairs": two-ion frames among
    them, "off_axis": the largest component of a pair's induced dipole perpendicular to its axis (Debye; zero by
    symmetry, so anything else is a dipole given in another frame or about another origin), "per_ion": the largest
    |induced dipole| per ion (Debye)}."""
    out = {"n": 0, "pairs": 0, "off_axis": 0.0, "per_ion": 0.0}
    for f in frames:
        if "dipole" not in f.labels or f.labels.get("dipole_off", False):
            continue
        mu = induced_dipole(f, tab) * DEBYE
        out["n"] += 1
        out["per_ion"] = max(out["per_ion"], float(np.linalg.norm(mu)) / f.n)
        if f.n == 2:
            ax = (f.pos[1] - f.pos[0]).numpy()
            ax = ax / np.linalg.norm(ax)
            out["pairs"] += 1
            out["off_axis"] = max(out["off_axis"], float(np.linalg.norm(mu - (mu @ ax) * ax)))
    return out


def elec_labels(ct_sets, walk_eda):
    """{geometry key: ALMO cls_elec (kJ/mol)} of the ct-* sets and the walk EDAs (the frozen electrostatics labels
    the shell fits use)."""
    out = {}
    for wf in ("ct-walk", "ct-mbe", "ct-sat", "ct-scan"):
        p = Path(ct_sets) / f"{wf}.jsonl"
        for line in open(p) if p.exists() else []:
            r = json.loads(line)
            if r.get("stage") == "label" and "eda" in r and r.get("symbols") and not (r.get("tags") or {}).get("field"):
                out.setdefault(_geom_key(r["symbols"], r["coords"]), r["eda"]["cls_elec"])
    if walk_eda:
        from ase.io import read
        for f in sorted(Path(walk_eda).glob("*.xyz")):
            for a in read(f, ":"):
                if "eda_cls_elec" in a.info:
                    out.setdefault(_geom_key(a.get_chemical_symbols(), a.positions), a.info["eda_cls_elec"] * KJ)
    return out


def split(frames, test_parents, val_frac=0.1, seed=0):
    parents = sorted({m["parent"] for *_, m in frames})
    rest = [p for p in parents if p not in set(test_parents)]
    rng = random.Random(seed)
    rng.shuffle(rest)
    val = set(rest[: max(1, int(val_frac * len(rest)))])
    tag = lambda p: "test" if p in set(test_parents) else "val" if p in val else "train"
    return [tag(m["parent"]) for *_, m in frames]


@dataclass
class Frame:
    """A labelled structure: symbols, positions (N, 3) Angstrom, atomic numbers, labels (Hartree, masks), meta."""
    symbols: tuple
    pos: torch.Tensor
    Z: torch.Tensor
    labels: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @property
    def n(self):
        return len(self.symbols)


def as_frame(sym, X, labels=None, meta=None) -> Frame:
    return Frame(tuple(sym), torch.as_tensor(np.asarray(X, float), dtype=torch.float64),
                 torch.tensor([ATOMIC_NUMBER[x] for x in sym]), dict(labels or {}), dict(meta or {}))


def as_frames(rows) -> list[Frame]:
    """Frames of [(symbols, coords A, labels, meta)] rows (the loaders' format)."""
    return [as_frame(sym, X, lab, meta) for sym, X, lab, meta in rows]


LABEL_KEYS = ("pol", "ct", "pauli", "disp", "elec", "int")


def batch_of(frames: list[Frame], model, requires_grad=False):
    """(positions, batch) of frames on the model's device: the precompute of the concatenated positions (a graph
    from the positions when ``requires_grad``, else none), with the energy labels and masks attached."""
    dev = next(model.parameters()).device
    pos = torch.cat([f.pos for f in frames]).to(dev).requires_grad_(requires_grad)
    Z = torch.cat([f.Z for f in frames]).to(dev)
    fr = torch.cat([torch.full((f.n,), k, dtype=torch.long) for k, f in enumerate(frames)]).to(dev)
    lab = {k: torch.tensor([float(f.labels.get(k, float("nan"))) for f in frames], dtype=torch.float64, device=dev)
           for k in LABEL_KEYS}
    lab["ct_mask"] = torch.tensor([bool(f.labels.get("ct_mask", True)) for f in frames], device=dev)
    lab["weight"] = frame_weights(frames, dev)
    if any("dipole" in f.labels and not f.labels.get("dipole_off", False) for f in frames):
        # (B, 3) the induced dipoles (e bohr), nan without a label
        lab["dip"] = torch.as_tensor(np.stack([induced_dipole(f, model.tab) for f in frames]), dtype=torch.float64,
                                     device=dev)
    lab["n"] = torch.tensor([float(f.n) for f in frames], dtype=torch.float64, device=dev)
    with torch.set_grad_enabled(requires_grad):
        bt = model.precompute(pos, Z, fr, labels=lab)
    return pos, bt


def frame_weights(frames, device=None):
    """(B,) the frames' loss weights (``labels["weight"]``, 1 where there is none)."""
    return torch.tensor([float(f.labels.get("weight", 1.0)) for f in frames], dtype=torch.float64, device=device)


def set_pair_weights(model, frames, ratio=1.5) -> dict:
    """Loss weights of the two-ion frames for a model whose CT channels end with each pair's own overlap switch (the
    frontier kernel, ``ct_cut = "overlap"``), in place (``labels["weight"]``).

    A pair's coupling is complete inside R_on, where its switch starts (the outermost distance at which the overlap
    is s_on, ``ct_overlap_window``); between R_on and R_off the model has only part of the pair's charge transfer,
    beyond none, while the reference has all of it. So the frame counts fully while the overlap is at least
    ``ratio`` x s_on (R <= R_w), not at all from R_on on, and with a C-infinity step in between: every frame with
    weight has its full coupling, and no fitted quantity is asked to stand in for the truncated part. Only channels
    out of an anion count: a cation's donor is a core shell more than 0.8 Hartree down, its channels carry almost
    no charge (0.04 - 0.17 kJ/mol in the cation pairs of the salts data) and their overlap is inside the window at
    every distance, so cation-cation pairs keep their weight. Of an anion-anion pair's two directions the one whose
    switch starts further out decides. No labels are used. Frames of more than two ions and models without the
    kernel are left alone. Returns {"pairs", "reduced", "dropped"} counts (pairs: those with an anion)."""
    from ..ff.cmm.channels import switch_step
    from ..ff.cmm.ct_overlap import switch_radii
    out = {"pairs": 0, "reduced": 0, "dropped": 0}
    if not getattr(model, "overlap_cut", False) or ratio <= 0:
        return out
    if ratio <= 1:
        raise ValueError(f"pair weight ratio {ratio}: > 1 (the weight ends where the switch starts), or 0 for none")
    sym = list(model.tab.symbols)
    s_on = model.config.ct_overlap_window[0]
    grid, coef = model.ctk_grid.cpu().numpy(), model.ctk_coef.cpu().numpy()
    r = grid[0] + grid[1] * np.arange(coef.shape[2])
    memo = {}

    def radii(a, b):                    # (R_w, R_on) of the direction a -> b
        if (a, b) not in memo:
            on = float(model.ctk_ron[a, b])
            memo[a, b] = (min(switch_radii(coef[a, b, :, 0], r, ratio * s_on, s_on)[0], on), on)
        return memo[a, b]
    for f in frames:
        if f.n != 2:
            continue
        a, b = sym.index(f.symbols[0]), sym.index(f.symbols[1])
        dirs = [radii(d, c) for d, c in ((a, b), (b, a)) if bool(model.ct_anion[d])]
        if not dirs:
            continue
        rw, on = max(dirs, key=lambda v: v[1])
        R = float((f.pos[0] - f.pos[1]).norm())
        w = 1.0 if R <= rw else 0.0 if R >= on else float(1 - switch_step(torch.tensor((R - rw) / (on - rw)), "smooth"))
        f.labels["weight"] = w * float(f.labels.get("weight", 1.0))
        out["pairs"] += 1
        out["reduced"] += 0 < w < 1
        out["dropped"] += w == 0
    return out


def set_pair_gap_weights(frames, window, anions=("F", "Cl", "Br", "I")) -> dict:
    """Loss weights of the two-ion frames by their vacuum gap (``meta["vacuum_gap"]``, Hartree), in place: full weight
    at ``window[0]`` and above, none at ``window[1]`` and below, a C-infinity step in between; multiplies the weight
    the frame has. Towards the ionic / radical crossing the gas-phase reference of a pair mixes in the radical state
    (a long charge-transfer tail that the model's frontier coupling does not have and the condensed phase does not
    show); those frames are left out so that no fitted quantity stands in for it. Only pairs with an anion; frames of
    more than two ions are left alone. Returns {"pairs", "reduced", "dropped"} counts."""
    from ..ff.cmm.channels import switch_step
    out = {"pairs": 0, "reduced": 0, "dropped": 0}
    g_on, g_off = (float(v) for v in window)
    if not g_on > g_off >= 0:
        raise ValueError(f"pair gap window {tuple(window)}: needs g_on > g_off >= 0 (Hartree)")
    for f in frames:
        g = f.meta.get("vacuum_gap")
        if f.n != 2 or g is None or not np.isfinite(g) or not set(f.symbols) & set(anions):
            continue
        w = 1.0 if g >= g_on else 0.0 if g <= g_off else float(switch_step(torch.tensor((g - g_off) / (g_on - g_off)), "smooth"))
        f.labels["weight"] = w * float(f.labels.get("weight", 1.0))
        out["pairs"] += 1
        out["reduced"] += 0 < w < 1
        out["dropped"] += w == 0
    return out


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


def _mae(d, cap, w=None):
    """Mean |d|, each capped at ``cap`` (> 0); with weights w, their weighted mean (0 if they are all zero)."""
    d = np.abs(d)
    d = np.minimum(d, cap) if cap > 0 else d
    if w is None:
        return float(d.mean())
    w = np.asarray(w, float)
    return float((w * d).sum() / w.sum()) if w.sum() > 0 else 0.0
