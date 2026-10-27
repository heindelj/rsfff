"""l-decomposition of the Cl- response in Na_n shells with the Na+ replaced by
  PC : +1 point charges (pure Coulomb, bare-charge penetration)
  FD : frozen Na+ densities, nuclei + exact AO electron density, Coulomb only (no exchange / Pauli)
Exact split E2 = sum_l E_l: E_l = 1/2 int phi_src * Phi[drho_l], Phi[drho_l] the (pure-l, smooth) potential of the
l-projected induced charge density about Cl, integrated against the source charges (point nuclei exactly,
Na+ electron density on its own grid). No singular projection.
Shapes: pair, a90, a109, a120, a180 (two Na+ at that angle), tri3 (planar), sq4, tet4, oct6; R = Cl-Na in Angstrom.
Also: the Na+ share (CPKS of one Na+ in the Cl- + other Na+, PC and FD; every site is symmetry-equivalent).
Monomers from the elst_db (Clm, Nap), ELST_DB env var (default research/rsfff data/elst_db).
Usage: ion_cage_response.py OUT.json shape:R [shape:R ...]     e.g. oct6:2.6 tet4:2.6 pair:2.6
Pairs with ion_cage_scfmi.py (ALMO-style SCF-MI with frozen/relaxed cations) and ion_cage_channels.py (fit)."""
import json, os, sys, time
from dataclasses import replace
import numpy as np
from pyscf import dft
from pyscf.dft import radi, LebedevGrid
from scipy.special import sph_harm_y
from scipy.interpolate import CubicSpline

from rsfff.qcgen import elst_ref as E
from rsfff.qcgen import pol_ref as PR

BOHR = E.BOHR
KJ = 2625.4996
LMAX = 8

# ---------------------------------------------------------------- geometries (Angstrom, Cl at origin)
def shell(shape, R):
    s = {
        "pair": [[1, 0, 0]],
        "a90": [[1, 0, 0], [0, 1, 0]],
        "a109": [[1, 1, 1], [1, -1, -1]],
        "a120": [[1, 0, 0], [-0.5, np.sqrt(3) / 2, 0]],
        "a180": [[1, 0, 0], [-1, 0, 0]],
        "tri3": [[1, 0, 0], [-0.5, np.sqrt(3) / 2, 0], [-0.5, -np.sqrt(3) / 2, 0]],
        "sq4": [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]],
        "tet4": [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
        "oct6": [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
    }[shape]
    v = np.asarray(s, float)
    return R * v / np.linalg.norm(v, axis=1, keepdims=True)


# ---------------------------------------------------------------- monomers
DB = os.environ.get("ELST_DB", os.path.expanduser("~/research/rsfff/data/elst_db"))
cl = E.Monomer.load(os.path.join(DB, "monomers", "Clm.npz"))
na = E.Monomer.load(os.path.join(DB, "monomers", "Nap.npz"))
pcl = cl.place()
assert np.allclose(pcl.coords, 0)
mol = pcl.mol
mf_cl = PR.isolated(pcl)
pna0 = na.place()
mf_na = PR.isolated(pna0)
dm_na = PR._total_dm(mf_na)
na_c0 = pna0.coords[0].copy()

# Na+ electron charge density on its own grid (translated per site)
gna = dft.gen_grid.Grids(pna0.mol); gna.level = 5; gna.build()
rho_na_e = -dft.numint.eval_rho(pna0.mol, dft.numint.eval_ao(pna0.mol, gna.coords), dm_na)   # charge density
na_grid_rel = gna.coords - na_c0
na_w = gna.weights

# ---------------------------------------------------------------- Cl-centred product grid for drho
rr, wr = radi.treutler(250, 17)
ang = LebedevGrid.MakeAngularGrid(590)
uv, wa = ang[:, :3], 4 * np.pi * ang[:, 3]
pts = (rr[:, None, None] * uv[None]).reshape(-1, 3)
ao_cl = dft.numint.eval_ao(mol, pts)


def real_Y(l, u):
    """(2l+1, N) real orthonormal spherical harmonics at unit vectors u."""
    th = np.arccos(np.clip(u[:, 2], -1, 1)); ph = np.arctan2(u[:, 1], u[:, 0])
    out = [sph_harm_y(l, 0, th, ph).real]
    for m in range(1, l + 1):
        y = sph_harm_y(l, m, th, ph)
        out += [np.sqrt(2) * y.real, np.sqrt(2) * y.imag]
    return np.array(out)


Yang = [real_Y(l, uv) for l in range(LMAX + 1)]
rfine = np.geomspace(1e-5, 40.0, 40000)


def phi_l_funcs(ddm):
    """For each l: function p -> Phi_l(p), the potential of the l-part of the induced *charge* density."""
    rho = -dft.numint.eval_rho(mol, ao_cl, ddm).reshape(len(rr), len(uv))
    funcs, resid = [], rho.copy()
    for l in range(LMAX + 1):
        c = (rho * wa[None]) @ Yang[l].T                       # (NR, 2l+1)
        resid -= c @ Yang[l]
        cs = CubicSpline(rr, c, axis=0, extrapolate=False)
        cf = np.nan_to_num(cs(rfine))
        cf[rfine > rr.max()] = 0.0
        cf[rfine < rr.min()] = c[0] if l == 0 else 0.0
        fa = cf * rfine[:, None] ** (l + 2)
        fb = cf * rfine[:, None] ** (1 - l)
        dr = np.diff(rfine)[:, None]
        A = np.concatenate([np.zeros((1, 2 * l + 1)), np.cumsum(0.5 * (fa[1:] + fa[:-1]) * dr, 0)])
        Bc = np.concatenate([np.zeros((1, 2 * l + 1)), np.cumsum(0.5 * (fb[1:] + fb[:-1]) * dr, 0)])
        B = Bc[-1] - Bc

        def f(p, l=l, A=A, B=B):
            p = np.atleast_2d(p); r = np.linalg.norm(p, axis=1)
            Y = real_Y(l, p / r[:, None])
            Ai = np.stack([np.interp(r, rfine, A[:, m]) for m in range(2 * l + 1)])
            Bi = np.stack([np.interp(r, rfine, B[:, m]) for m in range(2 * l + 1)])
            return 4 * np.pi / (2 * l + 1) * np.sum(Y * (r ** (-l - 1) * Ai + r ** l * Bi), 0)
        funcs.append(f)
    # fraction of drho not captured by l <= LMAX (should be ~0: the basis is band-limited)
    w3 = (wr[:, None] * rr[:, None] ** 2 * wa[None])
    leak = np.sqrt(np.sum(w3 * resid ** 2) / np.sum(w3 * rho ** 2))
    return funcs, leak


def E_by_l(ddm, sites_b, kind):
    funcs, leak = phi_l_funcs(ddm)
    out = []
    for f in funcs:
        if kind == "PC":
            e = float(np.sum(f(sites_b)))
        else:
            e = 11.0 * float(np.sum(f(sites_b)))
            for s in sites_b:
                e += float(na_w @ (rho_na_e * f(na_grid_rel + s)))
        out.append(0.5 * e * KJ)
    return out, leak


def environment(sites_b, kind):
    if kind == "PC":
        return PR.Environment(charge_coords=sites_b, charges=np.ones(len(sites_b)))
    parts = []
    for s in sites_b:
        parts.append(replace(na.place(np.eye(3), s - na_c0), dm=dm_na))
    return PR.Environment(partners=parts)


def run(shape, R):
    sites_b = shell(shape, R) / BOHR
    rec = {"shape": shape, "R": R, "n": len(sites_b)}
    for kind in ("PC", "FD"):
        t = time.time()
        env = environment(sites_b, kind)
        res = PR.polarize(pcl, env, mf0=mf_cl, linear=True)
        El_lin, leak = E_by_l(res["ddm_lin"], sites_b, kind)
        El_scf, _ = E_by_l(res["ddm"], sites_b, kind)
        rec[kind] = {"e_scf": res["e_pol"] * KJ, "e_lin": res["e_pol_lin"] * KJ, "El_lin": El_lin, "El_scf_half": El_scf,
                     "dr2_lin": float(np.trace(res["dQ_lin"])), "dr2_scf": float(np.trace(res["dQ"])), "leak": float(leak)}
        print(f"{shape:5s} R {R:.2f} {kind}: E_pol scf {rec[kind]['e_scf']:8.2f} lin {rec[kind]['e_lin']:8.2f} "
              f"(sum_l {sum(El_lin):8.2f}, leak {leak:.1e}) | " + " ".join(f"l{l} {e:7.2f}" for l, e in enumerate(El_lin) if l <= 6)
              + f" | d<r2> lin {rec[kind]['dr2_lin']:+.3f} scf {rec[kind]['dr2_scf']:+.3f}  [{time.time() - t:.0f}s]", flush=True)
    # cation share: Na+ at site 0 (all sites are symmetry-equivalent in every shape here) in Cl- + the other Na+
    p0 = replace(na.place(np.eye(3), sites_b[0] - na_c0), dm=dm_na)
    mf0 = PR.isolated(p0)
    others = sites_b[1:]
    for kind in ("PC", "FD"):
        if kind == "PC":
            env = PR.Environment(charge_coords=np.vstack([[0.0, 0.0, 0.0], others]), charges=np.r_[-1.0, np.ones(len(others))])
        else:
            env = PR.Environment(partners=[replace(pcl, dm=PR._total_dm(mf_cl))] + [replace(na.place(np.eye(3), s - na_c0), dm=dm_na) for s in others])
        h1 = env.h1(p0.mol)
        d = PR.linear_response(mf0, h1)
        rec[kind]["na_lin_per_site"] = 0.5 * float(np.einsum("ij,ji->", h1, d)) * KJ
        rec[kind]["na_lin_total"] = rec[kind]["na_lin_per_site"] * len(sites_b)
    print(f"{shape:5s} R {R:.2f} Na+ lin total: PC {rec['PC']['na_lin_total']:7.2f}  FD {rec['FD']['na_lin_total']:7.2f}", flush=True)
    return rec


if __name__ == "__main__":
    out = sys.argv[1]
    cfgs = [(c.split(":")[0], float(c.split(":")[1])) for c in sys.argv[2:]]
    recs = []
    for shape, R in cfgs:
        recs.append(run(shape, R))
        json.dump(recs, open(out, "w"), indent=1)
