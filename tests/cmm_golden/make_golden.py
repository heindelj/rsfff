"""Golden reference of the complete model as trained in the complete.* runs (tag complete-explore-final).

Runs only at that tag (the legacy model and train_ionct are gone from the cmm code): ``git worktree add /tmp/golden
complete-explore-final`` and run it there; the references it wrote are tests/cmm_golden/golden_*.pt.

Builds the legacy ``CompleteModel`` with the arguments of ``complete_worker`` (DEFAULTS of 2026-11-13), puts every
trainable parameter at a random non-trivial value (readouts included, which start at zero), sets the input scales and
saturations from the frames themselves, and records per frame: every energy term, the forces, the joint moments, the
CT charges, the atomic charge changes, the eight environment log-factors and the finite-field polarizability.

    python tests/cmm_golden/make_golden.py PARAMS.json OUT.pt [--film]

``--film`` conditions the encoder on the species embedding (FiLM, needs the real ``bb``); without it the variant has no
species conditioning, so it runs where only a stub of ``bb`` is importable. The reference this writes is what the cmm
package must reproduce (tests/test_cmm_golden.py), after ``rsfff.ff.cmm.legacy.convert_state``.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import torch

from rsfff.ff.complete import DensityEnvironment, PROBE_NAMES, batched_precompute
from rsfff.ff.ionct import ct as CT
from rsfff.ff.ionct.params import ATOMIC_NUMBER, IonTable
from rsfff.train.train_ionct import build_model, radial_table

F_SET = 2e-3


def worker_args(film: bool) -> dict:
    """The train_ionct arguments build_model reads, as complete_worker passes them (2026-11-13 defaults)."""
    return dict(model="complete", frozen=True, xpol=True, pseudo=False, exchange="xpol", ct_model="flux",
                flux_solver="atomic", pauli_form="shells", no_aniso=True, pin_a0=True, tie_disp_range=True,
                tie_c=True, confine=[], hidden=64, env="density", inv_form="sq",
                env_cap=3.0, env_sat=1.2, ct_q_cap=1.5, gap_cap=2.0,
                xpol_form="gram", xpol_weights=True, xpol_w1="rank", hyperpol=True, hyperpol_form="legendre",
                hyperpol_b=True, ct_b_pin="acceptor-float",
                env_channels=True, env_latent=6, env_mode="shared", env_two_latent=True, env_pairs=["ww"],
                env_two_body=["pauli", "disp", "ct"], env_ct_gated=True, env_extra_latent=True, env_skip=False,
                env_extra_capacity="full", sr_features=True, ct_dimers="like",
                env_species="film" if film else "none", env_species_emb="descriptor", env_species_dim=8,
                xpol_kappa_env="scalefree", xpol_perp=False)


def frames():
    """(name, symbols, positions Angstrom): pairs, cages, rock-salt fragments, mixed salts."""
    g = torch.Generator().manual_seed(7)
    out = [("nacl_dimer", ["Na", "Cl"], [[0, 0, 0], [2.75, 0, 0]]),
           ("clcl_dimer", ["Cl", "Cl"], [[0, 0, 0], [3.7, 0.1, 0]]),
           ("nana_dimer", ["Na", "Na"], [[0, 0, 0], [3.4, 0, 0.2]])]
    # Cl- in a tetrahedral cage of Na+
    t = torch.tensor([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=torch.float64) / math.sqrt(3) * 2.7
    out.append(("cl_tet4", ["Cl"] + ["Na"] * 4, [[0, 0, 0]] + t.tolist()))

    def cube(a, b, d, n=2, jitter=0.12):
        sym, pos = [], []
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    sym.append(a if (i + j + k) % 2 == 0 else b)
                    pos.append([i * d, j * d, k * d])
        X = torch.tensor(pos, dtype=torch.float64)
        X = X + jitter * torch.randn(X.shape, generator=g, dtype=torch.float64)
        return sym, X.tolist()

    out.append(("nacl8", *cube("Na", "Cl", 2.8)))
    out.append(("lif8", *cube("Li", "F", 2.0)))
    out.append(("lif_dimer", ["Li", "F"], [[0, 0, 0], [1.6, 0.05, 0]]))
    s, X = cube("Na", "Cl", 2.85, n=3, jitter=0.15)                 # 27 ions, net charge +1
    out.append(("nacl27", s, X))
    s1, X1 = cube("Na", "Cl", 2.8)
    s2, X2 = cube("Li", "F", 2.0)
    X2 = (torch.tensor(X2) + torch.tensor([5.6, 0.3, 0.2])).tolist()
    out.append(("nacl_lif", s1 + s2, X1 + X2))
    return out


def batch(fr, tab, table, chan, inv_form="sq", requires_grad=True):
    pos = torch.tensor([x for _, _, X in fr for x in X], dtype=torch.float64, requires_grad=requires_grad)
    Z = torch.tensor([ATOMIC_NUMBER[x] for _, s, _ in fr for x in s])
    idx = torch.cat([torch.full((len(s),), k, dtype=torch.long) for k, (_, s, _) in enumerate(fr)])
    return pos, batched_precompute(pos, Z, idx, tab, table, chan, inv_form=inv_form)


def polar_fields(B):
    f = torch.zeros(7, 3, dtype=torch.float64)
    for k in range(3):
        f[1 + 2 * k, k], f[2 + 2 * k, k] = F_SET, -F_SET
    return f.repeat(B, 1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("params")
    ap.add_argument("out")
    ap.add_argument("--film", action="store_true")
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed)
    tab, chan = IonTable.load(a.params), CT.ChannelTable.load(a.params)
    feat = DensityEnvironment(tab, probes=PROBE_NAMES, inv_form="sq")
    table = radial_table(tab, a.params, os.environ.get("RSFFF_TABLE_CACHE"))
    args = worker_args(a.film)
    fr = frames()
    # feature scales from the frames (the trainer takes them from its training atoms)
    pos, bt = batch(fr, tab, table, chan, requires_grad=False)
    Fe = bt["feats"]
    nz = Fe.abs().sum(1) > 0
    scale = Fe[nz].std(0).clamp(min=1e-6)
    model = build_model(args, tab, chan, Fe.shape[1], scale, list(feat.feature_names))
    with torch.no_grad():
        Xc = model.env_raw_batch(bt)
        nzc = Xc.abs().sum(1) > 0
        model.chan_scale.copy_(Xc[nzc].std(0).clamp(min=1e-12))
        Gv = model.diffuse_gated_invariants(bt)
        nzg = Gv.abs().sum(1) > 0
        model.gate_scale.copy_(Gv[nzg].std(0).clamp(min=1e-12))
        model.feat_sat.copy_(1.2 * (Fe[nz] / scale).abs().max(0).values)
        model.chan_sat.copy_(1.2 * (Xc / model.chan_scale).abs().max(0).values)
        model.gate_sat.copy_(1.2 * (Gv / model.gate_scale).abs().max(0).values)
        # every trainable parameter off its start: small noise on the per-species constants, O(0.3) readouts
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("env.W."):
                p.copy_(0.3 * torch.randn_like(p))
            elif name.startswith("env.") or name.startswith("species_emb."):
                p.add_(0.05 * torch.randn_like(p))
            else:
                p.add_(0.1 * torch.randn_like(p))
    model.eval()
    pos, bt = batch(fr, tab, table, chan, model.inv_form)
    out = model(bt)
    E = out["e_int"]
    F = -torch.autograd.grad(E.sum(), pos)[0]
    u = {k: v.detach().clone() for k, v in model._env_u.items()}
    # finite-field polarizability (the trainer's 7-point stencil), per frame [pol, CT] x axis
    rep = [f for f in fr for _ in range(7)]
    with torch.no_grad():
        _, btp = batch(rep, tab, table, chan, model.inv_form, requires_grad=False)
        op = model(btp, field=polar_fields(len(fr)))
        e = torch.stack([op["e_pol"], op["e_ct"]], -1).reshape(len(fr), 7, 2)
        alpha = -(e[:, 1::2] + e[:, 2::2] - 2 * e[:, :1]) / F_SET ** 2
    keys = ("e_elst", "e_pauli", "e_disp", "e_pol", "e_xpol", "e_ct", "e_int", "x", "x_pol", "dq", "q_ct")
    ref = {k: out[k].detach().clone() for k in keys}
    ref.update(forces=F.detach(), u=u, alpha=alpha, frozen_params=[t.detach().clone() for t in out["frozen_params"]])
    torch.save({"args": args, "state": model.state_dict(), "n_feat": Fe.shape[1], "feature_names": list(feat.feature_names),
                "feat_scale": scale, "frames": fr, "ref": ref, "film": a.film}, a.out)
    print(f"wrote {a.out}: {len(fr)} frames, e_int {E.detach().tolist()}")


if __name__ == "__main__":
    sys.exit(main())
