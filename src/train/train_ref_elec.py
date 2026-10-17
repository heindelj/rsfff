"""Train :class:`rsfff.ff.ref_elec.RefElecModel` on reference electrostatics (rsfff.qcgen.elst_db / pol_ref).

    python -m rsfff.train.train_ref_elec DB_ROOT [--epochs 200] [--out ckpt.pt]

Targets per monomer, all from the monomer's own density (no dimer calculations):

* permanent: the potential and field on the scaled vdW grid (``db/grids`` for molecules, ``db/grids_inner`` for
  monatomic ions, so their shells are physical);
* induced, where ``db/pol/<name>.npz`` exists: d phi/dF, dE/dF and d phi/dG, dE/dG on the response grid, and the
  molecular polarizability.

Each vdW shell is weighted equally relative to its reference RMS (the per-geometry fits' loss), so the numbers
logged are relative RMS errors and directly comparable to ``pol_fits_v2.json`` / ``joint_fits.json``.
Validation holds out every fifth conformer of each molecule.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..ff.ref_elec import RefElecConfig, RefElecModel, flow_kernel, make_batch, split_params
from ..qcgen import elst_db as Db
from ..qcgen import elst_fit as EF
from ..qcgen import pol_fit as PF

DTYPE = torch.float64


def _rel_weights(scale, ref, n_comp=1):
    sc = np.round(np.asarray(scale, float), 2)
    u = np.unique(sc)
    w = np.zeros(len(sc))
    for s in u:
        k = sc == s
        w[k] = 1.0 / (k.sum() * np.mean(ref[k] ** 2) * n_comp * len(u))
    return torch.as_tensor(w, dtype=DTYPE)


def load_monomers(root, names=None, inner_for_ions=True, use_gradient=True):
    db = Db.MonomerDB(root)
    out = []
    for n in names or list(db.index):
        m = db.get(n)
        charge = db.index[n]["charge"]
        atom = len(m.symbols) == 1
        gk = dict(scales=EF.INNER_SCALES, cache="grids_inner") if (atom and inner_for_ions) else {}
        g = Db.monomer_grid(db, n, **gk)
        rec = {"name": n, "numbers": [EF._Z(s) for s in m.symbols], "coords": m.coords, "charge": charge,
               "species": db.index[n].get("species", n), "ion": atom,
               "grid": {"points": torch.as_tensor(g["points"], dtype=DTYPE), "scale": np.asarray(g["scale"]),
                        "phi": torch.as_tensor(g["phi"], dtype=DTYPE), "field": torch.as_tensor(g["field"], dtype=DTYPE),
                        "wp": _rel_weights(g["scale"], g["phi"]), "wE": _rel_weights(g["scale"], g["field"], 3)}}
        pol = db.root / "pol" / f"{n}.npz"
        if pol.exists():
            r = dict(np.load(pol))
            X = torch.as_tensor(m.coords, dtype=DTYPE)
            sets = []
            for tag, cols, pr, er in PF._column_sets(r, X, np.ones(len(r["points"]), bool), use_gradient):
                if np.sqrt(np.mean(pr ** 2)) < 1e-10:
                    continue
                sets.append((tag, cols, torch.as_tensor(pr, dtype=DTYPE), torch.as_tensor(er, dtype=DTYPE),
                             _rel_weights(r["scale"], pr), _rel_weights(r["scale"], er, 3)))
            rec["resp"] = {"points": torch.as_tensor(r["points"], dtype=DTYPE), "scale": np.asarray(r["scale"]),
                           "sets": sets, "alpha": torch.as_tensor(r["alpha"], dtype=DTYPE)}
        out.append(rec)
    return out


def permanent_potential_field(X, sp, P, max_rank=2):
    """phi (P,), E (P, 3) of the permanent model (point multipoles + multi-shell one-centre penetration), from
    the damped interaction tensors directly (no autograd field): the same numbers as elst_fit.potential_field."""
    from ..ff.multipole import damped_interaction_tensor, slater_one_center_damp
    N, Pn = X.shape[0], P.shape[0]
    dr = (P[:, None, :] - X[None, :, :]).reshape(-1, 3)
    r = dr.norm(dim=-1)
    rinv = 1.0 / r
    m, m_cp, _ = sp.polytensors(max_rank)
    b, sw = sp.shells()
    f = sum(sw[:, s].repeat(Pn) * slater_one_center_damp(b[:, s].repeat(Pn) * r, max_rank) for s in range(b.shape[1]))
    T0 = damped_interaction_tensor(dr, None, rinv, max_rank=max_rank)
    T1 = damped_interaction_tensor(dr, -f, rinv, max_rank=max_rank)
    mi, mc = m.repeat(Pn, 1), m_cp.repeat(Pn, 1)
    phi = (T0[:, 0, :] * mi).sum(-1) + (T1[:, 0, :] * mc).sum(-1)
    E = -(torch.einsum("pak,pk->pa", T0[:, 1:4, :], mi) + torch.einsum("pak,pk->pa", T1[:, 1:4, :], mc))
    return phi.reshape(Pn, N).sum(1), E.reshape(Pn, N, 3).sum(1)


def monomer_losses(p, m, n_points=None, rng=None, metrics=False):
    """(permanent loss, response loss, alpha loss) for one monomer's parameters ``p``; with ``metrics`` also
    per-scale relative errors."""
    X = torch.as_tensor(m["coords"], dtype=DTYPE)
    g = m["grid"]
    idx = slice(None)
    if n_points and len(g["phi"]) > n_points:
        idx = torch.as_tensor(rng.choice(len(g["phi"]), n_points, replace=False))
    sp = EF.SiteParams(p["q"], p["mu"], p["theta"], p["Z"], p["b"], None, p["sw"])
    phi, E = permanent_potential_field(X, sp, g["points"][idx])
    # the weights make each vdW shell contribute its relative MSE / n_shells (they do not sum to one); a random
    # subset of points is rescaled to the full grid so the expectation is unchanged
    frac = len(g["phi"]) / len(phi)
    Lp = frac * ((g["wp"][idx] * (phi - g["phi"][idx]) ** 2).sum()
                 + (g["wE"][idx] * ((E - g["field"][idx]) ** 2).sum(1)).sum())
    met = {}
    if metrics:
        sc = g["scale"]
        dphi = (phi - g["phi"]).detach().numpy()
        for s in (0.6, 1.0, 2.0):
            k = np.isclose(sc, s)
            if k.any():
                met[f"perm@{s}"] = float(np.sqrt(np.mean(dphi[k] ** 2) / np.mean(g["phi"].numpy()[k] ** 2)))
    Lr = torch.zeros((), dtype=DTYPE)
    La = torch.zeros((), dtype=DTYPE)
    if "resp" in m:
        r = m["resp"]
        n = len(m["numbers"])
        K = flow_kernel(p["eta"], p["compliance"], p["bond_index"], n)
        T = PF.kernels(X, p["b_ind"], p["w_dip"], r["points"], 2, w_quad=p["w_quad"])
        for tag, (Ec, gc, ph), pr, er, wp, wE in r["sets"]:
            mu = torch.einsum("nab,cnb->cna", p["alpha"], Ec)
            th = p["cquad"][None, :, None] * gc
            dq = -(ph @ K.T)
            phi_m, E_m = PF.moments_potential_field(T, dq, mu, spherical_theta(th))
            Lr = Lr + (wp[:, None] * (phi_m - pr) ** 2).sum() / 3 + (wE[:, None, None] * (E_m - er) ** 2).sum() / 3
            if metrics:
                d = (phi_m - pr).detach().numpy()
                for s in (0.6, 1.0, 2.0):
                    k = np.isclose(r["scale"], s)
                    if k.any():
                        met[f"{tag}@{s}"] = float(np.sqrt(np.mean(d[k] ** 2) / np.mean(pr.numpy()[k] ** 2)))
        a_mol = p["alpha"].sum(0) + X.T @ K @ X
        La = ((a_mol - r["alpha"]) ** 2).sum() / (r["alpha"] ** 2).sum()
        if metrics:
            met["alpha_rel"] = float(La.sqrt())
            met["alpha_iso"] = float(torch.trace(a_mol) / 3)
            met["alpha_ref"] = float(torch.trace(r["alpha"]) / 3)
    return Lp, Lr, La, met


def spherical_theta(th5):
    from ..ff.multipole import spherical_to_cartesian_quadrupole
    return spherical_to_cartesian_quadrupole(th5)


def _to64(p):
    return {k: (v.to(DTYPE) if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in p.items()}


def evaluate(model, data, batch_size=16):
    model.eval()
    rows = []
    for i in range(0, len(data), batch_size):
        chunk = data[i:i + batch_size]
        batch = make_batch(chunk, dtype=DTYPE)
        with torch.enable_grad():
            parts = split_params(_to64(model(batch)), batch)
            for p, m in zip(parts, chunk):
                p = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in p.items()}
                Lp, Lr, La, met = monomer_losses(p, m, metrics=True)
                rows.append({"name": m["name"], "species": m["species"], **met})
    model.train()
    return rows


def summarize(rows):
    out = {}
    for r in rows:
        grp = r["species"]
        for k, v in r.items():
            if k in ("name", "species"):
                continue
            out.setdefault(grp, {}).setdefault(k, []).append(v)
    return {g: {k: float(np.median(v)) for k, v in d.items()} for g, d in out.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--points", type=int, default=800)
    ap.add_argument("--w-resp", type=float, default=1.0)
    ap.add_argument("--w-alpha", type=float, default=1.0)
    ap.add_argument("--out", default="ref_elec.pt")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--priors", default=None, help="JSON with z_init/b_init from per-geometry fits")
    ap.add_argument("--free-ratio", default="", help="elements whose b_ind/b_perm is learned, e.g. 9,17,35")
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed)
    torch.set_default_dtype(DTYPE)
    rng = np.random.default_rng(a.seed)
    data = load_monomers(a.root)
    def _conf(n):
        tail = n.rsplit("_", 1)[-1]
        return int(tail) if "_" in n and tail.isdigit() else None
    val = [m for m in data if _conf(m["name"]) is not None and _conf(m["name"]) % 15 == 0]
    train = [m for m in data if m not in val]
    cfg = RefElecConfig()
    if a.priors:
        pr = json.load(open(a.priors))
        cfg.z_init = {int(k): v for k, v in pr.get("z_init", {}).items()}
        cfg.b_init = {int(k): v for k, v in pr.get("b_init", {}).items()}
    if a.free_ratio:
        cfg.free_ratio = tuple(int(z) for z in a.free_ratio.split(","))
    model = RefElecModel(cfg).to(DTYPE)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    print(f"{len(train)} train / {len(val)} val monomers; {sum(p.numel() for p in model.parameters())} parameters", flush=True)
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        perm = rng.permutation(len(train))
        tot = np.zeros(3)
        for i in range(0, len(perm), a.batch):
            chunk = [train[j] for j in perm[i:i + a.batch]]
            batch = make_batch(chunk, dtype=DTYPE)
            parts = split_params(_to64(model(batch)), batch)
            L = 0.0
            for p, m in zip(parts, chunk):
                Lp, Lr, La, _ = monomer_losses(p, m, n_points=a.points, rng=rng)
                L = L + Lp + a.w_resp * Lr + a.w_alpha * La
                tot += [float(Lp), float(Lr), float(La)]
            opt.zero_grad()
            (L / len(chunk)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
        sched.step()
        if ep % a.log_every == 0 or ep == 1 or ep == a.epochs:
            sv = summarize(evaluate(model, val))
            fmt = lambda d: " ".join(f"{k} {v:.2e}" for k, v in d.items())
            print(f"ep {ep:4d} train Lp {tot[0] / len(train):.3e} Lr {tot[1]:.3e} La {tot[2]:.3e} | "
                  + " | ".join(f"{g}: {fmt(d)}" for g, d in sv.items()) + f" [{time.time() - t0:.0f}s]", flush=True)
            torch.save({"cfg": vars(cfg), "state": model.state_dict()}, a.out)
    rows = evaluate(model, data)
    json.dump({"all": rows, "summary_val": summarize([r for r in rows if r["name"] in {m["name"] for m in val}]),
               "summary_train": summarize([r for r in rows if r["name"] not in {m["name"] for m in val}])},
              open(Path(a.out).with_suffix(".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
