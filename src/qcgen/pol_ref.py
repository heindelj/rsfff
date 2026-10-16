"""Reference polarization: monomer density response to external fields, point charges and frozen partners.

The electrostatic reference (:mod:`elst_ref`, :mod:`elst_db`) freezes every monomer density. This module
relaxes it. A monomer is re-solved (same functional and basis as its database entry) in an external
electrostatic environment and everything a polarization model is fitted to is recorded as a difference
*with minus without* the environment:

* the induced density  delta_rho = rho[env] - rho[0]  (AO matrix, electrons; the nuclei do not move);
* its potential, field (and optionally field gradient) at the monomer's grid points -- the same scaled
  vdW surfaces as the permanent electrostatics (``elst_db.monomer_grid``), so permanent and induced fits
  share points;
* the induced dipole and second moment (origin: the centre of nuclear charge);
* energies: E[env], the isolated E0, the first-order (frozen) interaction E1 = <rho0|v_env> + nuclei, and
  the polarization energy E_pol = E[env] - E0 - E1 (<= 0 for a variational SCF);
* the environment at the nuclei (potential, field, field gradient) -- what a model's local response sees.

Environments (:class:`Environment`, combinable): a uniform field, point charges, and frozen partner
densities (placed ``elst_ref.Placed`` monomers: nuclei + exact AO electron density, Coulomb only -- no
Pauli repulsion, so the short-range response is the purely electrostatic one; the missing
exchange-polarization is exactly what a classical model also lacks). ``mutual=True`` freeze-thaws a
partner pair to self-consistency (classical mutual induction, still without exchange).

Linear response is available for any environment from coupled-perturbed KS (:func:`linear_response`), so
each record carries both the full SCF response and its linear part: their difference is the
hyperpolarization the linear model cannot represent. :func:`field_response` gives the uniform-field
response densities d rho / d F_x (the distributed polarizability) on the grid: the cleanest target for
polarizabilities and induced-density shapes.

Notes. The CPKS kernel is pyscf's ``gen_response``: it omits the VV10 (nlc) second derivative, so for
wB97M-V the linear response differs slightly from finite-field SCF (tested). Closed-shell (RKS) only for
the linear response; full SCF works for any spin.

Conventions: atomic units; an electron in potential phi has energy -phi, so the one-electron operator of an
environment is h1 = -phi_env(r) (uniform field F: phi = -F.r, h1 = +F.r -> the induced dipole is along F).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field as dfield, replace
from pathlib import Path

import numpy as np
from pyscf import df, gto, lib
from pyscf.scf import cphf
from pyscf.scf import jk as _jk

from . import elst_ref as E

BOHR = E.BOHR


# ---------------------------------------------------------------------------
# environments
# ---------------------------------------------------------------------------
@dataclass
class Environment:
    """External electrostatic environment: uniform field + point charges + frozen partner densities."""

    field: np.ndarray | None = None              # (3,) au
    charge_coords: np.ndarray | None = None      # (K, 3) Bohr
    charges: np.ndarray | None = None            # (K,) e
    partners: list = dfield(default_factory=list)   # elst_ref.Placed (nuclei + AO density)

    def is_empty(self):
        return self.field is None and self.charges is None and not self.partners

    def h1(self, mol: gto.Mole) -> np.ndarray:
        """One-electron operator -phi_env(r) in mol's AO basis."""
        h = np.zeros((mol.nao, mol.nao))
        if self.field is not None:
            with mol.with_common_orig((0.0, 0.0, 0.0)):
                h += np.einsum("x,xij->ij", np.asarray(self.field, float), mol.intor("int1e_r", comp=3))
        if self.charges is not None:
            j3 = df.incore.aux_e2(mol, gto.fakemol_for_charges(np.asarray(self.charge_coords, float)),
                                  intor="int3c2e", aosym="s1")
            h -= np.einsum("ijk,k->ij", j3, np.asarray(self.charges, float))
        for p in self.partners:
            j3 = df.incore.aux_e2(mol, gto.fakemol_for_charges(p.coords), intor="int3c2e", aosym="s1")
            h -= np.einsum("ijk,k->ij", j3, p.charges)                        # partner nuclei
            h += _jk.get_jk((mol, mol, p.mol, p.mol), p.dm, scripts="ijkl,lk->ij", aosym="s4")   # partner electrons
        return h

    def potential(self, points) -> np.ndarray:
        pts = np.asarray(points, float).reshape(-1, 3)
        phi = np.zeros(len(pts))
        if self.field is not None:
            phi -= pts @ np.asarray(self.field, float)
        if self.charges is not None:
            r = np.linalg.norm(pts[:, None] - np.asarray(self.charge_coords)[None], axis=2)
            phi += (np.asarray(self.charges)[None] / r).sum(1)
        if self.partners:
            phi += E.potential(self.partners, pts)
        return phi

    def field_at(self, points) -> np.ndarray:
        pts = np.asarray(points, float).reshape(-1, 3)
        out = np.zeros((len(pts), 3))
        if self.field is not None:
            out += np.asarray(self.field, float)
        if self.charges is not None:
            d = pts[:, None] - np.asarray(self.charge_coords)[None]
            r = np.linalg.norm(d, axis=2)
            out += (np.asarray(self.charges)[None, :, None] * d / r[..., None] ** 3).sum(1)
        if self.partners:
            out += E.field(self.partners, pts)
        return out

    def field_gradient_at(self, points) -> np.ndarray:
        """G_ab = dE_a/dr_b (P, 3, 3)."""
        pts = np.asarray(points, float).reshape(-1, 3)
        out = np.zeros((len(pts), 3, 3))
        if self.charges is not None:
            d = pts[:, None] - np.asarray(self.charge_coords)[None]
            r = np.linalg.norm(d, axis=2)
            dd = 3 * d[..., :, None] * d[..., None, :] - (r ** 2)[..., None, None] * np.eye(3)
            out -= (np.asarray(self.charges)[None, :, None, None] * dd / r[..., None, None] ** 5).sum(1)
        if self.partners:
            out += E.field_gradient(self.partners, pts)
        return out

    def nuclear(self, mol: gto.Mole) -> float:
        """Interaction of mol's nuclei with the environment."""
        return float(mol.atom_charges() @ self.potential(mol.atom_coords()))

    def spec(self) -> dict:
        """JSON-able description (partners by monomer name and placement)."""
        out = {}
        if self.field is not None:
            out["field"] = np.asarray(self.field, float).tolist()
        if self.charges is not None:
            out["charges"] = np.asarray(self.charges, float).tolist()
            out["charge_coords"] = np.asarray(self.charge_coords, float).tolist()
        if self.partners:
            out["partners"] = [{"name": p.monomer.meta.get("name"), "rot": p.rot.tolist(), "shift": p.shift.tolist()}
                               for p in self.partners]
        return out


# ---------------------------------------------------------------------------
# SCF in an environment, linear response
# ---------------------------------------------------------------------------
def _mf(placed: E.Placed, env: Environment | None, xc: str):
    from .backend import make_mf
    mol = placed.mol
    mf = make_mf(mol, xc, spin=placed.monomer.spin)
    if env is not None and not env.is_empty():
        h = mf.get_hcore() + env.h1(mol)
        e_nuc = mol.energy_nuc() + env.nuclear(mol)
        mf.get_hcore = lambda *a, **k: h
        mf.energy_nuc = lambda *a, **k: e_nuc
    return mf


def scf(placed: E.Placed, env: Environment | None = None, xc: str | None = None, dm0=None, conv_tol=1e-10,
        max_cycle=100):
    """Converged mean field of the placed monomer in ``env`` (None: isolated). Returns the pyscf object;
    ``mf.e_tot`` includes the environment's interaction with the nuclei."""
    xc = xc or placed.monomer.meta.get("xc", "wb97m-v")
    mf = _mf(placed, env, xc)
    mf.conv_tol = conv_tol
    mf.max_cycle = max_cycle
    mf.kernel(dm0=placed.dm if dm0 is None else dm0)
    if not mf.converged:
        mf = mf.newton()
        mf.kernel(mf.make_rdm1())
        if not mf.converged:
            raise RuntimeError("embedded SCF did not converge")
    return mf


def _total_dm(mf):
    dm = np.asarray(mf.make_rdm1())
    return dm[0] + dm[1] if dm.ndim == 3 else dm


def linear_response(mf, V, tol=1e-9, max_cycle=60) -> np.ndarray:
    """First-order density d dm / d lambda of the closed-shell KS state for H = H0 + lambda V.

    ``V``: (nao, nao) or (n, nao, nao) one-electron operators (e.g. ``Environment.h1``). Coupled-perturbed
    KS with the full (hybrid, range-separated) response kernel except VV10. Returns the AO response
    density, same leading shape as ``V``."""
    V = np.asarray(V, float)
    single = V.ndim == 2
    V = V[None] if single else V
    C, occ, e = mf.mo_coeff, mf.mo_occ, mf.mo_energy
    if np.asarray(C).ndim != 2:
        raise NotImplementedError("linear_response is closed-shell (RKS) only")
    o, v = C[:, occ > 0], C[:, occ == 0]
    nv, no = v.shape[1], o.shape[1]
    h1 = lib.einsum("pa,xpq,qi->xai", v, V, o)
    vresp = mf.gen_response(hermi=1)

    def dm_of(x):
        x = x.reshape(-1, nv, no)
        d = 2.0 * lib.einsum("pa,xai,qi->xpq", v, x, o)
        return d + d.transpose(0, 2, 1)

    def fvind(x):
        return lib.einsum("pa,xpq,qi->xai", v, vresp(dm_of(x)), o).ravel()

    mo1 = cphf.solve(fvind, e, occ, h1, None, max_cycle=max_cycle, tol=tol)[0]
    d = dm_of(mo1)
    return d[0] if single else d


# ---------------------------------------------------------------------------
# induced observables
# ---------------------------------------------------------------------------
def _density_only(placed: E.Placed, ddm) -> E.Placed:
    """A Placed whose 'density' is the induced electron density alone (no nuclei)."""
    return replace(placed, dm=np.asarray(ddm), charges=np.zeros_like(placed.charges))


def induced_potential(placed, ddm, points):
    return E.potential(_density_only(placed, ddm), points)


def induced_field(placed, ddm, points):
    return E.field(_density_only(placed, ddm), points)


def induced_field_gradient(placed, ddm, points):
    return E.field_gradient(_density_only(placed, ddm), points)


def charge_center(mol):
    Z = mol.atom_charges()
    return Z @ mol.atom_coords() / Z.sum()


def induced_moments(mol, ddm, origin=None):
    """(delta mu (3,), delta second moment (3, 3)) of the induced electron density (charge -1 per electron)
    about ``origin`` (default: centre of nuclear charge). The second moment is Cartesian, not traceless."""
    o = charge_center(mol) if origin is None else np.asarray(origin, float)
    with mol.with_common_orig(o):
        r = mol.intor("int1e_r", comp=3)
        rr = mol.intor("int1e_rr", comp=9).reshape(3, 3, mol.nao, mol.nao)
    return -np.einsum("xij,ji->x", r, ddm), -np.einsum("xyij,ji->xy", rr, ddm)


# ---------------------------------------------------------------------------
# one monomer, one environment
# ---------------------------------------------------------------------------
_ISO = {}


def isolated(placed: E.Placed, xc=None, key=None):
    """Isolated SCF of a placed monomer (cached by ``key``)."""
    if key is not None and key in _ISO:
        return _ISO[key]
    mf = scf(placed, None, xc)
    if key is not None:
        _ISO[key] = mf
    return mf


def polarize(placed: E.Placed, env: Environment, points=None, xc=None, linear=True, gradients=False,
             mf0=None, conv_tol=1e-10) -> dict:
    """Full-SCF (and linear) response of ``placed`` to ``env``.

    Returns a dict: ``ddm`` (induced AO density), energies ``e``, ``e0``, ``e1``, ``e_pol``, ``dmu``,
    ``dQ`` (origin ``origin``), the environment at the nuclei ``env_phi``/``env_field``/``env_grad``, and at
    ``points`` (if given) ``dphi``, ``dE`` (and ``dG``). With ``linear`` the same from CPKS: ``ddm_lin``,
    ``e_pol_lin`` (= 1/2 <ddm_lin|h1>, the second-order energy), ``dmu_lin``, ``dphi_lin``, ``dE_lin``."""
    mol = placed.mol
    mf0 = mf0 or isolated(placed, xc)
    dm0 = _total_dm(mf0)
    mf = scf(placed, env, xc, dm0=dm0, conv_tol=conv_tol)
    ddm = _total_dm(mf) - dm0
    h1 = env.h1(mol)
    e1 = float(np.einsum("ij,ji->", h1, dm0)) + env.nuclear(mol)
    X = mol.atom_coords()
    out = {"ddm": ddm, "e": float(mf.e_tot), "e0": float(mf0.e_tot), "e1": e1,
           "e_pol": float(mf.e_tot - mf0.e_tot - e1), "origin": charge_center(mol),
           "env_phi": env.potential(X), "env_field": env.field_at(X), "env_grad": env.field_gradient_at(X)}
    out["dmu"], out["dQ"] = induced_moments(mol, ddm)
    if points is not None:
        out["dphi"] = induced_potential(placed, ddm, points)
        out["dE"] = induced_field(placed, ddm, points)
        if gradients:
            out["dG"] = induced_field_gradient(placed, ddm, points)
    if linear:
        dl = linear_response(mf0, h1)
        out["ddm_lin"] = dl
        out["e_pol_lin"] = 0.5 * float(np.einsum("ij,ji->", h1, dl))
        out["dmu_lin"], out["dQ_lin"] = induced_moments(mol, dl)
        if points is not None:
            out["dphi_lin"] = induced_potential(placed, dl, points)
            out["dE_lin"] = induced_field(placed, dl, points)
    return out


def polarize_pair(pa: E.Placed, pb: E.Placed, mutual=True, xc=None, max_iter=30, tol=1e-8, mf_a=None, mf_b=None):
    """Electrostatic (no-exchange) polarization of two monomers by each other. Non-mutual: each in the
    other's isolated density. Mutual: freeze-and-thaw to self-consistency. Returns (dm_a, dm_b, info)."""
    mf_a = mf_a or isolated(pa, xc)
    mf_b = mf_b or isolated(pb, xc)
    da, db = _total_dm(mf_a), _total_dm(mf_b)
    a0, b0 = replace(pa, dm=da), replace(pb, dm=db)
    da_new = _total_dm(scf(pa, Environment(partners=[b0]), xc, dm0=da))
    db_new = _total_dm(scf(pb, Environment(partners=[a0]), xc, dm0=db))
    it, last = 1, None
    while mutual and it < max_iter:
        A, B = replace(pa, dm=da_new), replace(pb, dm=db_new)
        mfa = scf(pa, Environment(partners=[B]), xc, dm0=da_new)
        da_new = _total_dm(mfa)
        A = replace(pa, dm=da_new)
        mfb = scf(pb, Environment(partners=[A]), xc, dm0=db_new)
        db_new = _total_dm(mfb)
        # total electrostatic-embedding energy of the pair, for convergence
        etot = mfa.e_tot + mfb.e_tot - E.pair_energy(A, replace(pb, dm=db_new), exact_ee=True)["total"]
        it += 1
        if last is not None and abs(etot - last) < tol:
            break
        last = etot
    return da_new, db_new, {"iterations": it, "mutual": mutual}


# ---------------------------------------------------------------------------
# database: uniform-field response and environment records
# ---------------------------------------------------------------------------
def field_response(db, name, grid_kw=None, gradients=False, overwrite=False, cache="pol", method="auto",
                   step=1e-3) -> dict:
    """Uniform-field linear response of a database monomer: d dm / d F_x, the polarizability
    alpha_xy = d mu_x / d F_y, d(second moment)/dF, and on the monomer grid d phi / d F (P, 3), d E / d F
    (P, 3, 3) [, d G / d F (P, 3, 3, 3)]. Cached in ``db/<cache>/<name>.npz``.

    ``method``: "cpks" (closed shell), "finite" (central differences of field-perturbed SCF, +-``step`` au;
    any spin, includes every kernel term), "auto" (cpks for closed shells, finite otherwise)."""
    from . import elst_db as D
    out = db.root / cache / f"{name}.npz"
    if out.exists() and not overwrite:
        return dict(np.load(out))
    out.parent.mkdir(exist_ok=True)
    m = db.get(name)
    p = m.place()
    g = D.monomer_grid(db, name, **(grid_kw or {}))
    mf0 = isolated(p, m.meta.get("xc"), key=name)
    if method == "auto":
        method = "cpks" if m.spin == 0 else "finite"
    if method == "cpks":
        with p.mol.with_common_orig((0.0, 0.0, 0.0)):
            V = p.mol.intor("int1e_r", comp=3)            # h1 = F . r
        d1 = linear_response(mf0, V)                      # (3, nao, nao): d dm / d F_x
    else:
        d0 = _total_dm(mf0)
        d1 = []
        for x in range(3):
            Fv = np.zeros(3)
            Fv[x] = step
            dp = _total_dm(scf(p, Environment(field=Fv), m.meta.get("xc"), dm0=mf0.make_rdm1(), conv_tol=1e-12))
            dm = _total_dm(scf(p, Environment(field=-Fv), m.meta.get("xc"), dm0=mf0.make_rdm1(), conv_tol=1e-12))
            d1.append((dp - dm) / (2 * step))
        d1 = np.stack(d1)
        del d0
    mom = [induced_moments(p.mol, d1[x]) for x in range(3)]
    rec = {"ddm_dF": d1, "alpha": np.stack([mu for mu, _ in mom], 1),     # alpha[x, y] = d mu_x / d F_y
           "dQ_dF": np.stack([Q for _, Q in mom], 2), "origin": charge_center(p.mol),
           "points": g["points"], "scale": g["scale"], "coords": p.coords, "method": method,
           "dphi_dF": np.stack([induced_potential(p, d1[x], g["points"]) for x in range(3)], 1),
           "dE_dF": np.stack([induced_field(p, d1[x], g["points"]) for x in range(3)], 2)}
    if gradients:
        rec["dG_dF"] = np.stack([induced_field_gradient(p, d1[x], g["points"]) for x in range(3)], 3)
    np.savez(out, **rec)
    return rec


def _write_record(folder: Path, rec: dict, meta: dict):
    folder.mkdir(parents=True, exist_ok=True)
    idx = folder / "index.jsonl"
    k = sum(1 for _ in open(idx)) if idx.exists() else 0
    np.savez(folder / f"{k:05d}.npz", **{a: np.asarray(b) for a, b in rec.items()})
    with open(idx, "a") as f:
        f.write(json.dumps({"id": k, **meta}) + "\n")
    return k


def _summary(rec):
    s = {a: rec[a] for a in ("e", "e0", "e1", "e_pol", "e_pol_lin") if a in rec}
    s["dmu"] = np.asarray(rec["dmu"]).tolist()
    if "dmu_lin" in rec:
        s["dmu_lin"] = np.asarray(rec["dmu_lin"]).tolist()
    return s


def sample_probe_envs(db, n, tag, names=None, cutoff_ang=(2.5, 5.0), n_charges=(1, 4), total_charge=(-1.0, 0.0, 1.0),
                      charge_dist="uniform", spread=0.5, rng=None, grid_kw=None, gradients=False, linear=True, log=print):
    """Polarize random database monomers by probe point charges on their outer ``cutoff`` shell
    (``probe_charges.place_charges``). Records in ``db/pol_env/<tag>/`` (npz + index.jsonl)."""
    from . import elst_db as D
    from .probe_charges import place_charges
    rng = np.random.default_rng(rng)
    names = names or list(db.index)
    folder = db.root / "pol_env" / tag
    for k in range(n):
        name = names[rng.integers(len(names))]
        m = db.get(name)
        p = m.place()
        cut = rng.uniform(*cutoff_ang)
        nq = int(rng.integers(n_charges[0], n_charges[1] + 1))
        qt = float(rng.choice(total_charge))
        pos, q = place_charges(p.coords * BOHR, cutoff=cut, n_charges=nq, total_charge=qt, rng=rng,
                               charge_dist=charge_dist, spread=spread)
        env = Environment(charge_coords=np.asarray(pos) / BOHR, charges=np.asarray(q))
        g = D.monomer_grid(db, name, **(grid_kw or {}))
        rec = polarize(p, env, g["points"], m.meta.get("xc"), linear, gradients, mf0=isolated(p, key=name))
        meta = {"name": name, "kind": "charges", "cutoff_ang": cut, "env": env.spec(), **_summary(rec)}
        _write_record(folder, rec, meta)
        if log:
            log(f"{tag} {k + 1}/{n} {name}: {nq} charges (total {qt:+.0f}) at {cut:.2f} A, "
                f"E_pol {rec['e_pol'] * 627.5095:.3f} (lin {rec.get('e_pol_lin', np.nan) * 627.5095:.3f}) kcal/mol")
    return folder


def sample_partner_envs(db, n, tag, names_a=None, names_b=None, contact_ang=(1.6, 5.0), mutual=False, rng=None,
                        grid_kw=None, gradients=False, linear=True, log=print):
    """Polarize a database monomer A by a frozen partner B placed at a random closest contact (A in its
    stored frame, so A's grid applies). ``mutual``: B is first relaxed in A by freeze-and-thaw, then A's
    response to that B is recorded. Records in ``db/pol_env/<tag>/``."""
    from . import elst_db as D
    rng = np.random.default_rng(rng)
    names_a = names_a or list(db.index)
    names_b = names_b or list(db.index)
    folder = db.root / "pol_env" / tag
    for k in range(n):
        na, nb = names_a[rng.integers(len(names_a))], names_b[rng.integers(len(names_b))]
        A, B = db.get(na), db.get(nb)
        A.meta.setdefault("name", na)
        B.meta.setdefault("name", nb)
        contact = rng.uniform(*contact_ang) / BOHR
        _, pb = D.place_at_contact(A, B, contact, rng, rot_a=np.eye(3))
        pa = A.place()
        pb = B.place(pb.rot, pb.shift + A.coords.mean(0))     # into A's stored frame
        if mutual:      # (the rotated stored density is B's isolated SCF density to ~1e-7; relax it here)
            _, dmb, _ = polarize_pair(pa, pb, mutual=True, mf_a=isolated(pa, key=na), mf_b=isolated(pb))
            pb = replace(pb, dm=dmb)
        g = D.monomer_grid(db, na, **(grid_kw or {}))
        rec = polarize(pa, Environment(partners=[pb]), g["points"], A.meta.get("xc"), linear, gradients,
                       mf0=isolated(pa, key=na))
        rec["partner_dm"] = pb.dm
        meta = {"name": na, "kind": "partner", "partner": nb, "mutual": mutual, "contact_ang": contact * BOHR,
                "env": {"partners": [{"name": nb, "rot": pb.rot.tolist(), "shift": pb.shift.tolist()}]}, **_summary(rec)}
        _write_record(folder, rec, meta)
        if log:
            log(f"{tag} {k + 1}/{n} {na} + {nb} at {contact * BOHR:.2f} A: E_pol {rec['e_pol'] * 627.5095:.3f} "
                f"(lin {rec.get('e_pol_lin', np.nan) * 627.5095:.3f}) kcal/mol")
    return folder


def load_records(db, tag):
    """(index rows, loader) for ``db/pol_env/<tag>``: ``loader(i)`` returns record i as a dict."""
    folder = db.root / "pol_env" / tag
    rows = [json.loads(l) for l in open(folder / "index.jsonl")]
    return rows, (lambda i: dict(np.load(folder / f"{i:05d}.npz")))
