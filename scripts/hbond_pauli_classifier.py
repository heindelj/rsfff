"""Does the model's Pauli repulsion charge know what a hydrogen bond is?

Two things are computed and then put in one figure:

**Panel A.** Every inter-fragment ``O...H`` pair in a set of water clusters is labelled by
the Kumar/Schmidt/Skinner ``r``-``psi`` definition (:mod:`rsfff.ff.hbond`) and, independently,
by a straight dividing line in the ``(r, q_O)`` plane -- ``r`` the intermolecular H...O
distance and ``q_O`` the acceptor oxygen's *Pauli repulsion charge*, the environment-aware
monopole the film model emits for its Slater repulsion. The line is fitted to agree with the
``r``-``psi`` label as well as a line can; what the panel reports is how well that is,
because ``q_O`` was never trained on anything resembling a hydrogen-bond label. It is fitted
to ALMO-EDA energy components alone.

The signed distance from that line is the classifier score, and the two histograms are the
score split by the ``r``-``psi`` label.

**Panel B.** The monomer molecular polarizability against its Q-Chem label, as the three
sorted eigenvalues plus the isotropic mean. Eigenvalues and not tensor components, because
the reference geometries sit in arbitrary orientations: a component-wise comparison would be
reporting the sampling of Euler angles, not the response.

**Panel C.** Predicted-vs-reference correlation for each ALMO-EDA component the model fits
(frozen electrostatics, Pauli repulsion, dispersion, induction = pol + CT).

Also computed, and written to the ``.npz`` although no longer plotted, is the 2-body /
many-body split of those channels for one large cluster by direct subtraction:

    E^(2) = sum_{i<j} E(dimer ij),    E^(>2) = E(cluster) - E^(2)

which is the ``k=2`` term of the many-body expansion in :mod:`rsfff.ff.many_body` and the
whole remainder lumped together. The full Moebius inversion in that module needs ``2^N``
subsets and is unreachable past ``N ~ 20``; this needs ``N(N-1)/2 + 1`` evaluations.

Outputs ``notebooks/figures/hbond_pauli_manybody.{pdf,png}`` and a ``.npz`` of everything
plotted.
"""

from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

from rsfff.ff.hbond import dimer_geometry, hbond_labels, hbonds_per_molecule
from rsfff.ff.many_body import subset_batch
from rsfff.train.build_film import build_film_model
from rsfff.train.data import load_extxyz, load_reference_energies
from rsfff.ff.units import BOHR_ANG, KJMOL_PER_HARTREE

#: Thermally sampled small clusters, many frames each -- these supply the *non*-hydrogen-
#: bonded population. The optimized large clusters below are almost entirely bonded, so a
#: set made only of them would make the classification trivially easy.
SMALL = [f"data/wb97mv_tzvpd/w{n}_wb97mv_tzvpd.xyz" for n in (2, 3, 4, 5)]
LARGE_DIR = Path("data/wb97mv_tzvpd_large")

#: The ALMO-EDA components the model has a channel for, mapped to the ``batch.eda`` keys
#: that make up their label. ``induction`` is the one that is a sum: the model solves
#: polarization and charge transfer together, so ``pol`` and ``ct`` are only separable in
#: the reference, not in the prediction.
EDA_TERMS: dict[str, tuple[str, ...]] = {
    "elst": ("cls_elec",),
    "pauli": ("mod_pauli",),
    "disp": ("disp",),
    "induction": ("pol", "ct"),
}


def load_film_checkpoint(path: str, device: str = "cpu"):
    """Rebuild the film model from the checkpoint's own embedded config.

    The config travels with the weights, stage overrides already applied; reading a YAML
    instead builds a *different* model than the weights came from and the failure looks
    like a bad fit rather than a mismatch.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = state["config"]
    torch.set_default_dtype(torch.float64 if cfg.dtype == "float64" else torch.float32)
    neighbor_types = tuple(int(z) for z in state["neighbor_types"])
    ref = load_reference_energies(cfg.data.reference_energies, neighbor_types).to(
        torch.get_default_dtype()
    )
    model = build_film_model(cfg.features, cfg.film, neighbor_types, ref)
    model.load_state_dict(state["model_state"], strict=True)
    model.eval().to(device)
    return model, cfg, state


def collect_pairs(model, datasets, *, r_max: float, device: str, batch_size: int):
    """One row per inter-fragment ``O...H`` candidate, over every frame of every dataset.

    The model is run per batch of frames and ``q_O`` read off
    ``out.parameters.pauli[0]`` -- the environment-aware Pauli monopole, in ``e``. Frame-local
    atom indices from :func:`dimer_geometry` are shifted by the batch's own atom offsets, so
    the charge attached to a row is the one the model computed for *that* acceptor oxygen in
    *that* frame.
    """
    columns: dict[str, list] = {k: [] for k in
                                ("r", "psi", "alpha", "beta", "gamma", "R", "q_o", "q_o_iso",
                                 "q_h", "hb_occ", "hb_pmf", "size", "frame", "acceptor_uid",
                                 "donor_uid")}
    n_by_size: dict[int, list[float]] = {}
    # Predicted vs reference interaction energies, harvested from the same forward pass the
    # pair rows come from -- running the model twice over the same frames would be the only
    # way these two halves of the figure could ever disagree.
    energies: dict[str, list] = {f"{k}_{w}": [] for k in EDA_TERMS for w in ("pred", "ref")}
    energies["n_frag"] = []
    # A running counter so an (acceptor oxygen, frame) pair has one id across the whole run:
    # `q_O` is a per-atom quantity shared by every pair that oxygen appears in, and grouping
    # on it is the only way to see what it can and cannot resolve.
    uid_base = 0

    for tag, (ds, frames) in datasets.items():
        for start in range(0, len(frames), batch_size):
            indices = list(frames[start:start + batch_size])
            batch = ds.flat_batch(indices).to(device)
            with torch.no_grad():
                out = model(batch)
            if batch.eda is not None:
                for name, components in EDA_TERMS.items():
                    if name not in out.interaction:
                        continue
                    if any(c not in batch.eda for c in components):
                        continue
                    energies[f"{name}_pred"].append(
                        out.interaction[name].detach().cpu().numpy() * KJMOL_PER_HARTREE
                    )
                    energies[f"{name}_ref"].append(
                        sum(batch.eda[c] for c in components).cpu().numpy()
                        * KJMOL_PER_HARTREE
                    )
                energies["n_frag"].append(
                    np.bincount(batch.fragment_to_batch.cpu().numpy(),
                                minlength=batch.n_systems)
                )

            q = out.parameters.pauli[0].detach().cpu().numpy()
            q_iso = out.parameters.pauli0[0].detach().cpu().numpy()
            b_idx = batch.batch_idx.cpu().numpy()
            pos = batch.positions.detach().cpu().numpy()
            z = batch.atomic_numbers.cpu().numpy()
            frag = batch.fragment_idx.cpu().numpy()

            for local, frame in enumerate(indices):
                sel = np.flatnonzero(b_idx == local)
                offset = int(sel[0])
                f = frag[sel] - frag[sel].min()
                g = dimer_geometry(pos[sel], z[sel], f, r_max=r_max)
                n_frag = int(f.max()) + 1
                n_by_size.setdefault(n_frag, []).append(
                    hbonds_per_molecule(g, n_frag, definition="occupancy")
                )
                if len(g) == 0:
                    continue
                acc = g["acceptor_o"] + offset
                don_h = g["h"] + offset
                for key in ("r", "psi", "alpha", "beta", "gamma", "R"):
                    columns[key].append(g[key])
                columns["q_o"].append(q[acc])
                columns["q_o_iso"].append(q_iso[acc])
                columns["q_h"].append(q[don_h])
                columns["hb_occ"].append(hbond_labels(g, definition="occupancy"))
                columns["hb_pmf"].append(hbond_labels(g, definition="pmf"))
                columns["size"].append(np.full(len(g), n_frag))
                columns["frame"].append(np.full(len(g), frame))
                # `acc`/`don_h` are already batch-global; `g["acceptor_o"]` is frame-local
                # and would collide between frames sharing a batch.
                columns["acceptor_uid"].append(uid_base + acc)
                columns["donor_uid"].append(uid_base + don_h)
            uid_base += int(b_idx.size)
        print(f"  {tag}: {len(frames)} frames, {sum(len(c) for c in columns['r'])} pairs so far",
              flush=True)

    pairs = {k: np.concatenate(v) for k, v in columns.items()}
    energy = {f"e_{k}": np.concatenate(v) for k, v in energies.items() if v}
    return pairs, energy, n_by_size


def _best_threshold(score, label):
    """``(accuracy, cut)`` for the best "bonded when ``score > cut``" threshold.

    A single sort plus prefix sums, so sweeping a family of scores stays cheap.
    """
    order = np.argsort(score, kind="stable")
    lab = np.asarray(label, bool)[order]
    n, n_true = lab.size, int(lab.sum())
    # cut between element k-1 and k: predicted-bonded = the n-k elements at or above k
    cum_true = np.concatenate(([0], np.cumsum(lab)))        # (n+1,) true labels below k
    fn = cum_true                                            # bonded but predicted unbonded
    tn = np.arange(n + 1) - fn
    tp = n_true - fn
    acc = (tp + tn) / n
    k = int(np.argmax(acc))
    s_sorted = score[order]
    cut = float(s_sorted[0] - 1.0) if k == 0 else float(
        s_sorted[k - 1] if k == n else 0.5 * (s_sorted[k - 1] + s_sorted[k])
    )
    return float(acc[k]), cut


def fit_dividing_line(r, q, label, *, n_angle: int = 721):
    """The best straight dividing line in the ``(r, q_O)`` plane, by accuracy.

    The line is parameterized by its normal direction rather than as ``q = a*r + b``, which
    matters: the slope-intercept family cannot express a *vertical* line, and a vertical line
    here is exactly the null hypothesis -- a pure distance cutoff that ignores ``q_O``
    entirely. Leaving it out of the search would have let the fit look like it was using the
    Pauli charge when it had no way to report that it was not.

    ``r`` and ``q`` are standardized first so the angle sweep is uniform over genuinely
    different lines instead of crowding near whichever axis has the larger units.

    Returns ``(direction, accuracy)`` where ``direction`` is
    ``(w_r, w_q, cut, r_mean, r_std, q_mean, q_std)``: bonded when
    ``w_r * r_hat + w_q * q_hat > cut``.
    """
    r = np.asarray(r, float)
    q = np.asarray(q, float)
    label = np.asarray(label, bool)
    r_mean, r_std = r.mean(), r.std()
    q_mean, q_std = q.mean(), q.std()
    r_hat, q_hat = (r - r_mean) / r_std, (q - q_mean) / q_std

    best = (None, -1.0)
    for theta in np.linspace(0.0, 2.0 * np.pi, n_angle, endpoint=False):
        w_r, w_q = float(np.cos(theta)), float(np.sin(theta))
        acc, cut = _best_threshold(w_r * r_hat + w_q * q_hat, label)
        if acc > best[1]:
            best = ((w_r, w_q, cut, r_mean, r_std, q_mean, q_std), acc)
    return best


def line_score(direction, r, q):
    """Signed distance from the dividing line, in units of the standardized coordinates.

    Positive on the hydrogen-bonded side. This is the scalar the histograms are drawn over:
    one number per O...H pair that collapses the 2D cut, exactly as the paper collapses its
    own 2D contour onto the occupancy axis of its Fig. 10.
    """
    w_r, w_q, cut, r_mean, r_std, q_mean, q_std = direction
    return w_r * (np.asarray(r, float) - r_mean) / r_std \
        + w_q * (np.asarray(q, float) - q_mean) / q_std - cut


def classification_metrics(pred, label):
    pred, label = np.asarray(pred, bool), np.asarray(label, bool)
    tp = int((pred & label).sum())
    tn = int((~pred & ~label).sum())
    fp = int((pred & ~label).sum())
    fn = int((~pred & label).sum())
    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return {
        "accuracy": (tp + tn) / len(label),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "mcc": (tp * tn - fp * fn) / denom if denom > 0 else 0.0,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


@torch.no_grad()
def collect_polarizability(model, dataset, *, device: str, batch_size: int = 64):
    """``(pred, ref)`` molecular polarizability tensors, ``(M, 3, 3)`` each.

    Only the dedicated monomer file carries this label, and only monomers may carry it: a
    cluster's polarizability is not the sum of its fragments' isolated values, so
    :func:`rsfff.train.loss.fragment_polarizability_loss` refuses a multi-fragment frame
    outright. Both tensors are in the label's own units, ``e^2 Angstrom^2 / Ha``; the caller
    converts.
    """
    pred, ref = [], []
    for start in range(0, len(dataset), batch_size):
        batch = dataset.flat_batch(
            list(range(start, min(start + batch_size, len(dataset))))
        ).to(device)
        if batch.polarizability is None:
            raise ValueError("this dataset carries no polarizability label")
        out = model(batch, with_polarizability=True)
        pred.append(out.polarizability.detach().cpu().numpy())
        ref.append(batch.polarizability.cpu().numpy())
    return np.concatenate(pred), np.concatenate(ref)


@torch.no_grad()
def two_body_split(model, batch, *, channels, device, dimer_chunk: int = 64):
    """``{channel: (total, two_body)}`` in Hartree for a single cluster frame.

    ``total`` is the channel on the whole cluster; ``two_body`` is the sum of the same
    channel over every isolated dimer. The difference is everything the expansion carries
    beyond pairwise -- for a model whose parameters are emitted from environment-aware
    descriptors, that remainder is nonzero even for channels (dispersion, repulsion) whose
    functional form is a plain pair sum.

    Dimers are evaluated in chunks rather than one batch so the memory stays bounded at
    ``N ~ 25`` (300 dimers); the chunking cannot change the result because
    :func:`~rsfff.ff.many_body.subset_batch` makes every subset an independent system.
    """
    n_frag = int(batch.fragment_idx.max()) + 1
    full = model(batch)
    totals = {c: float(full.interaction[c][0]) for c in channels}

    pairs = list(itertools.combinations(range(n_frag), 2))
    two_body = {c: 0.0 for c in channels}
    for start in range(0, len(pairs), dimer_chunk):
        chunk = pairs[start:start + dimer_chunk]
        sub = subset_batch(
            batch.positions, batch.atomic_numbers, batch.fragment_idx, chunk,
            fragment_charge=batch.fragment_charge, fragment_two_s=batch.fragment_two_s,
        ).to(device)
        out = model(sub)
        for c in channels:
            two_body[c] += float(out.interaction[c].sum())
    return {c: (totals[c], two_body[c]) for c in channels}, n_frag, len(pairs)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="checkpoints/water_film_large/best.pt")
    p.add_argument("--device", default="cpu")
    p.add_argument("--r-max", type=float, default=3.0,
                   help="candidate window for O...H pairs (A); not a hydrogen-bond criterion")
    p.add_argument("--small-frames", type=int, default=500,
                   help="frames kept per thermally-sampled w2-w5 file")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--mbe-cluster", default="w22",
                   help="which large cluster carries panel B (>20 molecules)")
    p.add_argument("--mbe-frame", type=int, default=0)
    p.add_argument("--out", default="notebooks/figures/hbond_pauli_manybody")
    args = p.parse_args()

    model, cfg, state = load_film_checkpoint(args.checkpoint, args.device)
    print(f"checkpoint {args.checkpoint}  epoch {state.get('epoch')}  dtype {cfg.dtype}")

    dtype = torch.get_default_dtype()
    datasets: dict[str, tuple] = {}
    for path in SMALL:
        ds = load_extxyz(path, dtype=dtype)
        keep = min(len(ds), args.small_frames)
        # Even stride, not the first N: the sampled files are ordered, and a prefix would
        # take one stretch of configuration space rather than the whole set.
        frames = np.unique(np.linspace(0, len(ds) - 1, keep).round().astype(int)).tolist()
        datasets[Path(path).stem] = (ds, frames)
    for path in sorted(LARGE_DIR.glob("*.xyz")):
        ds = load_extxyz(str(path), dtype=dtype)
        datasets[path.stem.split("_")[0] + "-opt"] = (ds, list(range(len(ds))))

    print("collecting pairs and interaction energies ...", flush=True)
    data, energy, n_by_size = collect_pairs(
        model, datasets, r_max=args.r_max, device=args.device, batch_size=args.batch_size
    )
    print(f"{len(data['r'])} candidate O...H pairs; "
          f"{int(data['hb_occ'].sum())} hydrogen bonded by r-psi (occupancy)")

    direction, acc2d = fit_dividing_line(data["r"], data["q_o"], data["hb_occ"])
    score = line_score(direction, data["r"], data["q_o"])
    pred = score > 0
    metrics = classification_metrics(pred, data["hb_occ"])
    metrics_pmf = classification_metrics(pred, data["hb_pmf"])
    agree = float((data["hb_occ"] == data["hb_pmf"]).mean())

    # The two ablations that decide what this figure is allowed to claim. `r`-only is the
    # null hypothesis -- the r-psi occupancy map decays with a 0.343 A length scale, so it is
    # very nearly a distance criterion, and a classifier that merely rediscovers that has
    # learned nothing about hydrogen bonding from the Pauli charge.
    acc_r, cut_r = _best_threshold(-data["r"], data["hb_occ"])
    acc_q, cut_q = _best_threshold(-data["q_o"], data["hb_occ"])
    baseline = max(data["hb_occ"].mean(), 1.0 - data["hb_occ"].mean())

    # Per-acceptor-oxygen: `q_O` is one number per oxygen, shared by every pair that oxygen
    # takes part in, so as a *pair* classifier it is structurally capped. What it can resolve
    # is how many hydrogen bonds that oxygen accepts.
    uid = data["acceptor_uid"]
    order = np.argsort(uid, kind="stable")
    uid_sorted = uid[order]
    bounds = np.flatnonzero(np.diff(uid_sorted)) + 1
    groups = np.split(order, bounds)
    n_accepted = np.array([int(data["hb_occ"][g].sum()) for g in groups])
    q_per_o = np.array([float(data["q_o"][g[0]]) for g in groups])
    acc_counts = {k: q_per_o[n_accepted == k] for k in sorted(set(n_accepted.tolist()))}

    print(f"dividing line (standardized): {direction[0]:+.3f} r_hat {direction[1]:+.3f} q_hat "
          f"> {direction[2]:+.3f}")
    print(f"  majority-class baseline      {baseline:.4f}")
    print(f"  q_O alone                    {acc_q:.4f}")
    print(f"  r alone                      {acc_r:.4f}   (r < {-cut_r:.3f} A)")
    print(f"  (r, q_O) dividing line       {acc2d:.4f}")
    print(f"  vs r-psi occupancy: {metrics}")
    print(f"  vs r-psi PMF:       {metrics_pmf}")
    print(f"  the two r-psi cutoffs agree with each other on {agree:.1%} of pairs")
    print("  acceptor oxygens by hydrogen bonds accepted:")
    for k, v in acc_counts.items():
        print(f"    {k} accepted: n={v.size:5d}  q_O = {v.mean():.4f} +- {v.std():.4f} e")

    # --- panel C: how well each EDA channel is reproduced -------------------------------
    print(f"EDA channels over {energy['e_n_frag'].size} frames (kJ/mol):")
    for name in EDA_TERMS:
        pr, rf = energy[f"e_{name}_pred"], energy[f"e_{name}_ref"]
        mae = float(np.abs(pr - rf).mean())
        r2 = 1.0 - float(((pr - rf) ** 2).sum() / ((rf - rf.mean()) ** 2).sum())
        print(f"  {name:>10s}: MAE {mae:7.3f}   R2 {r2:.5f}   range [{rf.min():.1f}, {rf.max():.1f}]")

    # --- panel B: monomer polarizability ------------------------------------------------
    print(f"polarizability on {cfg.data.monomer_path} ...", flush=True)
    monomers = load_extxyz(cfg.data.monomer_path, dtype=dtype)
    alpha_pred, alpha_ref = collect_polarizability(
        model, monomers, device=args.device, batch_size=args.batch_size
    )
    # e^2 Ang^2 / Ha -> a0^3, the unit polarizabilities are quoted in.
    to_au = 1.0 / (BOHR_ANG * BOHR_ANG)
    eig_pred = np.linalg.eigvalsh(alpha_pred)[:, ::-1] * to_au
    eig_ref = np.linalg.eigvalsh(alpha_ref)[:, ::-1] * to_au
    iso_pred, iso_ref = eig_pred.mean(axis=1), eig_ref.mean(axis=1)
    for k, name in enumerate(("alpha_1", "alpha_2", "alpha_3")):
        print(f"  {name}: model {eig_pred[:, k].mean():6.3f}  QM {eig_ref[:, k].mean():6.3f}"
              f"  MAE {np.abs(eig_pred[:, k] - eig_ref[:, k]).mean():.4f} a.u.")
    print(f"  isotropic: model {iso_pred.mean():6.3f}  QM {iso_ref.mean():6.3f}"
          f"  MAE {np.abs(iso_pred - iso_ref).mean():.4f} a.u.")

    # --- the many-body split (kept in the npz, no longer plotted) -----------------------
    mbe_path = LARGE_DIR / f"{args.mbe_cluster}_wb97mv_tzvpd.xyz"
    mbe_ds = load_extxyz(str(mbe_path), dtype=dtype)
    frame = mbe_ds.flat_batch([args.mbe_frame]).to(args.device)
    channels = ("induction", "disp", "pauli")
    print(f"many-body split on {args.mbe_cluster} frame {args.mbe_frame} ...", flush=True)
    split, n_frag, n_dimers = two_body_split(
        model, frame, channels=channels, device=args.device
    )
    for c, (tot, two) in split.items():
        print(f"  {c:>10s}: total {tot * KJMOL_PER_HARTREE:10.2f}  "
              f"2-body {two * KJMOL_PER_HARTREE:10.2f}  "
              f"many-body {(tot - two) * KJMOL_PER_HARTREE:8.2f} kJ/mol "
              f"({100 * (tot - two) / tot:+.1f}%)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out.with_suffix(".npz"),
        line_direction=np.array(direction),
        acc_2d=acc2d, acc_r=acc_r, acc_q=acc_q, cut_r=-cut_r, cut_q=-cut_q,
        baseline=baseline, score=score,
        group_n_accepted=n_accepted, group_q_o=q_per_o,
        n_by_size=np.array([[k, np.mean(v)] for k, v in sorted(n_by_size.items())]),
        mbe_channels=np.array(channels),
        mbe_total=np.array([split[c][0] for c in channels]),
        mbe_two_body=np.array([split[c][1] for c in channels]),
        mbe_n_fragments=n_frag, mbe_n_dimers=n_dimers, mbe_cluster=args.mbe_cluster,
        metrics=np.array([metrics[k] for k in ("accuracy", "precision", "recall", "mcc")]),
        metrics_pmf=np.array([metrics_pmf[k] for k in ("accuracy", "precision", "recall", "mcc")]),
        cutoff_agreement=agree,
        eda_terms=np.array(list(EDA_TERMS)),
        alpha_eig_pred=eig_pred, alpha_eig_ref=eig_ref,
        alpha_iso_pred=iso_pred, alpha_iso_ref=iso_ref,
        **energy,
        **data,
    )
    print(f"wrote {out.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
