"""Analysis helpers for a trained dispersion model (used by notebooks/eda_rsff_plotting).

Kept out of the notebook so the numbers are testable and the notebook stays about the
figures. Everything here is read-only with respect to the checkpoint.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rsfff.ff.many_body import mbe_dataset  # noqa: E402
from rsfff.ff.units import KJMOL_PER_HARTREE  # noqa: E402
from rsfff.train.config import load_config  # noqa: E402
from rsfff.train.data import load_extxyz  # noqa: E402
from rsfff.train.train_dispersion import build_dispersion_model  # noqa: E402

CLUSTERS = ("w2", "w3", "w4", "w5")
DATA_TEMPLATE = "data/eda_data/{tag}_wb97xv_qzvppd.xyz"


def load_model(checkpoint: str, config: str):
    """Rebuild the model from its config and load a checkpoint's weights."""
    cfg = load_config(config)
    torch.set_default_dtype(torch.float64 if cfg.dtype == "float64" else torch.float32)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_dispersion_model(cfg.features, cfg.dispersion, state["neighbor_types"])
    model.load_state_dict(state["model_state"])
    model.eval()
    return model, cfg, state


def load_clusters(tags=CLUSTERS, dtype=torch.float64):
    return {tag: load_extxyz(DATA_TEMPLATE.format(tag=tag), dtype=dtype) for tag in tags}


@dataclass
class Predictions:
    """Per-frame model output alongside the reference, all in kJ/mol."""

    tag: str
    pred: np.ndarray            # (n,) total predicted dispersion
    ref: np.ndarray             # (n,) eda_disp
    ff: np.ndarray              # (n,) force-field part
    corr: np.ndarray            # (n,) correction part
    n_fragments: np.ndarray     # (n,)

    @property
    def error(self) -> np.ndarray:
        return self.pred - self.ref

    @property
    def mae(self) -> float:
        return float(np.abs(self.error).mean())

    @property
    def rmse(self) -> float:
        return float(np.sqrt((self.error ** 2).mean()))

    @property
    def corr_fraction(self) -> np.ndarray:
        """|E_corr| / (|E_ff| + |E_corr|): 0 = pure force field, 1 = pure network."""
        return np.abs(self.corr) / np.maximum(np.abs(self.ff) + np.abs(self.corr), 1e-30)

    @property
    def r2(self) -> float:
        return float(np.corrcoef(self.pred, self.ref)[0, 1] ** 2)


def fragments_per_system(batch) -> torch.Tensor:
    """``(n_systems,)`` count of fragments in each frame of a ragged batch.

    ``fragment_idx`` is batch-global, so every fragment id belongs to exactly one system;
    map ids to systems and count.
    """
    frag_to_system = torch.zeros(batch.n_fragments, dtype=torch.long)
    frag_to_system.scatter_(0, batch.fragment_idx, batch.batch_idx)
    return torch.bincount(frag_to_system, minlength=batch.n_systems)


@torch.no_grad()
def predict(model, dataset, tag: str, *, batch_size: int = 128, limit=None) -> Predictions:
    n = len(dataset) if limit is None else min(limit, len(dataset))
    pred, ref, ff, corr, nfrag = [], [], [], [], []
    for start in range(0, n, batch_size):
        batch = dataset.flat_batch(range(start, min(start + batch_size, n)))
        out = model(batch)
        pred.append(out.energy.numpy())
        ref.append(batch.eda["disp"].numpy())
        ff.append(out.energy_ff.numpy())
        corr.append(out.energy_corr.numpy())
        nfrag.append(fragments_per_system(batch).numpy())

    k = KJMOL_PER_HARTREE
    return Predictions(
        tag=tag,
        pred=np.concatenate(pred) * k,
        ref=np.concatenate(ref) * k,
        ff=np.concatenate(ff) * k,
        corr=np.concatenate(corr) * k,
        n_fragments=np.concatenate(nfrag),
    )


def nearest_inter_fragment(batch) -> torch.Tensor:
    """``(N,)`` distance from each atom to the closest atom in a *different* fragment.

    The natural coordinate for "how engaged is this atom in the intermolecular
    interaction": for a water hydrogen it separates H-bond donors (~1.8 A to the acceptor
    oxygen) from free OH hydrogens (>2.5 A), which is the distinction the descriptors
    appear to be picking up on.
    """
    same_frame = batch.batch_idx[:, None] == batch.batch_idx[None, :]
    same_frag = batch.fragment_idx[:, None] == batch.fragment_idx[None, :]
    d = torch.cdist(batch.positions, batch.positions)
    d = d.masked_fill(~same_frame | same_frag, float("inf"))
    return d.min(dim=1).values


@torch.no_grad()
def atom_parameters(model, dataset, *, n_frames: int = 400, batch_size: int = 64):
    """Effective per-atom ``(C6, b)`` over many frames, with the per-species priors.

    Returns ``(records, priors)`` where ``records`` is a dict of flat arrays
    (``Z``, ``c6``, ``b``, ``n_fragments``, ``contact``) and ``priors`` maps Z -> (C6, b).
    """
    z, c6, b, nf, contact = [], [], [], [], []
    n = min(n_frames, len(dataset))
    for start in range(0, n, batch_size):
        batch = dataset.flat_batch(range(start, min(start + batch_size, n)))
        feats = model.featurizer(batch, batch.fragment_idx if model.intra_fragment else None)
        c6_i, b_i = model.params(feats.inv_feats, feats.species_idx)
        z.append(batch.atomic_numbers.numpy())
        c6.append(c6_i.numpy())
        b.append(b_i.numpy())
        # broadcast the frame's fragment count onto each of its atoms
        per_system = fragments_per_system(batch)
        nf.append(per_system[batch.batch_idx].numpy())
        contact.append(nearest_inter_fragment(batch).numpy())

    lut = model.featurizer._species_lut
    types = [int(zz) for zz in range(len(lut)) if lut[zz] >= 0]
    priors = {
        int(zz): (
            float(model.params.log_c6_prior[lut[zz]].exp()),
            float(model.params.log_b_prior[lut[zz]].exp()),
        )
        for zz in types
    }
    return (
        {
            "Z": np.concatenate(z),
            "c6": np.concatenate(c6),
            "b": np.concatenate(b),
            "n_fragments": np.concatenate(nf),
            "contact": np.concatenate(contact),
        },
        priors,
    )


def strongest_frames(dataset, n: int, *, key: str = "int") -> np.ndarray:
    """Indices of the ``n`` most strongly interacting frames (most negative ``eda[key]``)."""
    values = dataset.flat_batch(range(len(dataset))).eda[key].numpy()
    return np.argsort(values)[:n]


def many_body_table(model, datasets, per_cluster: int, *, key: str = "int",
                    tags=("w3", "w4", "w5"), progress_every: int = 0):
    """MBE of the strongest ``per_cluster`` frames of each cluster size, in kJ/mol.

    Returns a dict of flat arrays: ``tag``, ``n_fragments``, ``total``, ``two_body``,
    ``many_body``, ``e3``/``e4``/``e5``, and the ff/corr split of the many-body part.
    """
    rows: dict[str, list] = {k: [] for k in (
        "tag", "n_fragments", "total", "two_body", "many_body",
        "e3", "e4", "e5", "mb_ff", "mb_corr", "ref",
    )}
    for tag in tags:
        ds = datasets[tag]
        idx = strongest_frames(ds, per_cluster, key=key)
        res = mbe_dataset(model, ds, idx, progress_every=progress_every)
        ref = ds.flat_batch(idx).eda["disp"].numpy() * KJMOL_PER_HARTREE
        k = KJMOL_PER_HARTREE
        n = len(idx)
        rows["tag"] += [tag] * n
        rows["n_fragments"] += res.n_fragments.tolist()
        rows["total"] += (res.total.numpy() * k).tolist()
        rows["two_body"] += (res.two_body.numpy() * k).tolist()
        rows["many_body"] += (res.many_body.numpy() * k).tolist()
        rows["ref"] += ref.tolist()
        for order, name in ((3, "e3"), (4, "e4"), (5, "e5")):
            vals = res.by_order.get(order)
            rows[name] += (
                (vals.numpy() * k).tolist() if vals is not None else [0.0] * n
            )
        for comp, name in (("ff", "mb_ff"), ("corr", "mb_corr")):
            c = res.components[comp]
            rows[name] += ((c.total - c.by_order[2]).numpy() * k).tolist()
    return {k: np.asarray(v) for k, v in rows.items()}
