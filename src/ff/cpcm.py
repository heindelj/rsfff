"""C-PCM (and COSMO) for the force field, built to be the same continuum as pyscf's ``solvent.pcm`` / Q-Chem's PCM.

The cavity is the union of atom-centred spheres (Bondi radii x 1.2, H = 1.1 A -- pyscf's "modified Bondi" table,
overridable per element), each sphere carrying a Lebedev grid whose points are switched off smoothly inside the
other spheres (SWIG: Lange & Herbert, J. Chem. Phys. 133, 244111 (2010)); every surviving point i carries a Gaussian
charge of exponent zeta_i^2 = (xi(n) / (R_I sqrt(w_i)))^2. The apparent surface charges minimize

    G(q) = q . V + 1/(2 f) q^T S q      ->  q = -f S^-1 V,   G = 1/2 q . V = -f/2 V^T S^-1 V

with S_ij = erf(zeta_ij r_ij) / r_ij, S_ii = zeta_i sqrt(2/pi) / F_i, and f = (eps-1)/eps (C-PCM) or
(eps-1)/(eps+1/2) (COSMO). V_i is the solute's potential *smeared by the point's Gaussian* -- exactly how pyscf
evaluates the nuclear and electronic potentials on the surface (int2c2e / int3c2e against a Gaussian aux function) --
so a point charge contributes q erf(zeta r)/r and a Slater shell its closed-form smeared potential
(:func:`slater_shell_surface_potential`). Everything is torch, float64, and differentiable in the atom positions
(the switching functions, the moving grid points and the radii path), so forces come from autograd.

The pieces a force-field model needs: :func:`build_cavity`, :func:`s_matrix`, :func:`solve` (q and G for a given
surface potential), :func:`surface_potential_charges` / :func:`slater_shell_surface_potential` for the potentials of
its charge model, and :func:`joint_blocks` for adding the surface charges as one more block of an induction solve
(the joint Hessian [[A, J^T], [J, S/f]] is symmetric positive definite). :func:`to_pyscf_mm` exports the charges as
pyscf ``qmmm`` Gaussian charges (radii = 1/zeta) for the embedding round trip.

The pyscf oracle tests (tests/test_ff_cpcm.py) compare the surface (points, exponents, switching), S, and the solve
with pyscf's on identical input to ~1e-12.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field as dfield

import numpy as np
import torch

from ._lebedev import angular_grid

BOHR_ANG = 0.52917721092
SQRT_PI = math.sqrt(math.pi)

#: Bondi van der Waals radii (Angstrom) by atomic number, pyscf.data.radii.VDW with H = 1.1 ("modified Bondi",
#: the pyscf / Q-Chem PCM default; index 0 unused)
BONDI_ANG = [2.0, 1.1, 1.4, 1.82, 1.53, 1.92, 1.7, 1.55, 1.52, 1.47, 1.54, 2.27, 1.73, 1.84, 2.1, 1.8, 1.8, 1.75,
             1.88, 2.75, 2.31, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.63, 1.4, 1.39, 1.87, 2.11, 1.85, 1.9, 1.85, 2.02,
             3.03, 2.49, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.63, 1.72, 1.58, 1.93, 2.17, 2.06, 2.06, 1.98, 2.16,
             3.43, 2.49, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
             2.0, 2.0, 2.0, 1.75, 1.66, 1.55, 1.96, 2.02, 2.07, 1.97, 2.02, 2.2, 3.48, 2.83, 2.0, 2.0, 2.0, 1.86,
             2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]

#: Gaussian width constant xi(n) per Lebedev point count (Table II of J. Chem. Phys. 122, 194110 (2005); pyscf's XI)
XI = {6: 4.84566077868, 14: 4.86458714334, 26: 4.85478226219, 38: 4.90105812685, 50: 4.89250673295,
      86: 4.89741372580, 110: 4.90101060987, 146: 4.89825187392, 170: 4.90685517725, 194: 4.90337644248,
      302: 4.90498088169, 350: 4.86879474832, 434: 4.90567349080, 590: 4.90624071359, 770: 4.90656435779,
      974: 4.90685167998, 1202: 4.90704098216}

_SYMBOL_Z = {s: i for i, s in enumerate(
    ["X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K",
     "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr",
     "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba"])}

__all__ = ["CPCMSpec", "Cavity", "f_epsilon", "sphere_radii", "build_cavity", "s_matrix", "solve", "energy",
           "surface_potential_charges", "slater_shell_surface_potential", "joint_blocks", "to_pyscf_mm",
           "BONDI_ANG", "XI"]


@dataclass
class CPCMSpec:
    """The continuum. ``radii``: {symbol: final sphere radius in Angstrom} overriding the scaled table."""
    eps: float = 78.3553
    method: str = "C-PCM"          # C-PCM | COSMO
    ng: int = 302                  # Lebedev points per sphere (pyscf lebedev_order 29)
    vdw_scale: float = 1.2
    radii: dict = dfield(default_factory=dict)
    switching: str = "SWIG"        # SWIG | ISWIG
    drop: float = 1e-16            # points with w F < drop are removed (pyscf's threshold)

    @property
    def f(self):
        return f_epsilon(self.eps, self.method)

    def to_pyscf(self):
        """The matching :class:`rsfff.qcgen.almo.Continuum`."""
        from ..qcgen.almo import Continuum
        order = {50: 11, 110: 17, 194: 23, 302: 29, 434: 35, 590: 41}[self.ng]
        return Continuum(eps=self.eps, method=self.method, lebedev_order=order, vdw_scale=self.vdw_scale,
                         radii=dict(self.radii), switching=self.switching)


def f_epsilon(eps, method="C-PCM"):
    if eps is None or eps == 1.0:
        return 0.0
    if math.isinf(eps):
        return 1.0
    m = method.upper().replace("-", "")
    if m == "CPCM":
        return (eps - 1.0) / eps
    if m == "COSMO":
        return (eps - 1.0) / (eps + 0.5)
    raise ValueError(f"unknown continuum method {method!r} (C-PCM | COSMO)")


def sphere_radii(symbols, spec: CPCMSpec, dtype=torch.float64, device=None):
    """Per-atom sphere radii (bohr): vdw_scale x modified Bondi, or the override in spec.radii."""
    out = []
    for s in symbols:
        if s in spec.radii:
            out.append(float(spec.radii[s]) / BOHR_ANG)
        else:
            z = _SYMBOL_Z[s] if isinstance(s, str) else int(s)
            out.append(spec.vdw_scale * BONDI_ANG[z] / BOHR_ANG)
    return torch.tensor(out, dtype=dtype, device=device)


def _switch_h(x):
    """pyscf's switch_h: x^3 (10 - 15 x + 6 x^2) clamped to [0, 1]."""
    xc = x.clamp(0.0, 1.0)
    return xc ** 3 * (10.0 - 15.0 * xc + 6.0 * xc ** 2)


@dataclass
class Cavity:
    coords: torch.Tensor      # (P, 3) bohr, differentiable in the atom positions
    zeta: torch.Tensor        # (P,) Gaussian exponents (zeta_i; the charge is exp(-zeta_i^2 r^2))
    F: torch.Tensor           # (P,) switching functions
    w: torch.Tensor           # (P,) Lebedev weights x 4 pi
    R: torch.Tensor           # (P,) sphere radius of the point's atom
    atom: torch.Tensor        # (P,) atom index
    nvec: torch.Tensor        # (P, 3) outward unit normals
    radii: torch.Tensor       # (N,) sphere radii
    spec: CPCMSpec

    @property
    def area(self):
        return self.w * self.R ** 2 * self.F

    @property
    def npoints(self):
        return self.coords.shape[0]

    def slices_by_atom(self):
        out, p0 = [], 0
        for a in range(len(self.radii)):
            n = int((self.atom == a).sum())
            out.append((p0, p0 + n))
            p0 += n
        return out


def build_cavity(pos_bohr, radii, spec: CPCMSpec) -> Cavity:
    """The SWIG/ISWIG surface of the spheres (pos_bohr (N, 3), radii (N,) bohr), point for point pyscf's
    ``gen_surface`` (same Lebedev grid, same switching, same drop threshold, same point order)."""
    pos = torch.as_tensor(pos_bohr)
    N = pos.shape[0]
    ng = spec.ng
    ang = torch.as_tensor(angular_grid(ng), dtype=pos.dtype, device=pos.device)
    u, w = ang[:, :3], ang[:, 3] * 4.0 * math.pi
    xi_n = XI[ng]
    swig = spec.switching.upper() == "SWIG"
    if swig:
        R_sw = radii * math.sqrt(14.0 / ng)
        alpha = 0.5 + radii / R_sw - ((radii / R_sw) ** 2 - 1.0 / 28) ** 0.5
        R_in = radii - alpha * R_sw
    coords, zeta, F, ws, Rs, atoms, nv = [], [], [], [], [], [], []
    for a in range(N):
        r = radii[a]
        grid = r * u + pos[a]                                            # (ng, 3)
        d = (grid[:, None, :] - pos[None, :, :]).norm(dim=-1)            # (ng, N)
        xi = xi_n / (r * w.sqrt())
        if swig:
            x = (d - R_in[None, :]) / R_sw[None, :]
            x = torch.where(x < 1e-8, torch.zeros_like(x), x)
            f = _switch_h(x)
        else:
            f = 1 - 0.5 * (torch.erf(xi[:, None] * (radii[None, :] - d)) + torch.erf(xi[:, None] * (radii[None, :] + d)))
            f = torch.where(f < 1e-8, torch.zeros_like(f), f)
        f = torch.where(torch.arange(N, device=pos.device)[None, :] == a, torch.ones_like(f), f)
        swf = f.prod(dim=1)
        keep = (w * swf).detach() > spec.drop
        coords.append(grid[keep])
        zeta.append(xi[keep])
        F.append(swf[keep])
        ws.append(w[keep])
        Rs.append(r.expand(int(keep.sum())))
        atoms.append(torch.full((int(keep.sum()),), a, dtype=torch.long, device=pos.device))
        nv.append(u[keep])
    return Cavity(torch.cat(coords), torch.cat(zeta), torch.cat(F), torch.cat(ws), torch.cat(Rs), torch.cat(atoms),
                  torch.cat(nv), radii, spec)


def s_matrix(cav: Cavity):
    """S_ij = erf(zeta_ij r_ij)/r_ij, zeta_ij = zeta_i zeta_j / sqrt(zeta_i^2 + zeta_j^2); S_ii = zeta_i sqrt(2/pi)/F_i."""
    z = cav.zeta
    zij = z[:, None] * z[None, :] / (z[:, None] ** 2 + z[None, :] ** 2).sqrt()
    d = cav.coords[:, None, :] - cav.coords[None, :, :]
    eye = torch.eye(cav.npoints, dtype=torch.bool, device=z.device)
    r = torch.where(eye, torch.ones_like(zij), (d * d).sum(-1).clamp_min(1e-300).sqrt())
    S = torch.erf(zij * r) / r
    return torch.where(eye, torch.diag(z * math.sqrt(2.0 / math.pi) / cav.F), S)


def solve(S, V, f):
    """Equilibrated charges q = -f S^-1 V (V (P,) or (B, P)) and G = 1/2 q . V."""
    q = -f * torch.linalg.solve(S, V.unsqueeze(-1)).squeeze(-1) if f else torch.zeros_like(V)
    return q, 0.5 * (q * V).sum(-1)


def energy(S, V, f):
    return solve(S, V, f)[1]


# ---------------------------------------------------------------------------
# smeared potentials on the surface
# ---------------------------------------------------------------------------
def surface_potential_charges(cav: Cavity, pos_bohr, q):
    """Potential of point charges q at pos_bohr (M, 3), smeared by each point's Gaussian: sum_a q_a erf(zeta_i r)/r."""
    d = (cav.coords[:, None, :] - torch.as_tensor(pos_bohr)[None, :, :]).norm(dim=-1)   # (P, M)
    return (torch.erf(cav.zeta[:, None] * d) / d * torch.as_tensor(q)[None, :]).sum(1)


def _exp_erfcx(x, s):
    """exp(-s) erfcx(x) evaluated stably for x of either sign (s >= 0, x^2 - s moderate)."""
    pos = x >= 0
    xa = x.abs()
    e = torch.special.erfcx(xa)
    return torch.where(pos, torch.exp(-s) * e, 2.0 * torch.exp(x * x - s) - torch.exp(-s) * e)


def slater_shell_surface_potential(cav: Cavity, centers_bohr, n, b):
    """Gaussian-smeared potential at the surface points of Slater 1s shells rho_t(r) = n_t b_t^3/(8 pi) exp(-b_t r)
    (electron count n_t; the potential of the *charge* -n_t e is returned, i.e. the sign of an electron density):
    centers (M, 3), n (M,), b (M,). Closed form:

        <phi>(R) = (zeta/(sqrt(pi) R)) int_0^inf phi(r) r [e^{-zeta^2 (r-R)^2} - e^{-zeta^2 (r+R)^2}] dr,
        phi(r) = (1 - (1 + b r/2) e^{-b r}) / r,

    = erf(zeta R)/R - (zeta/(sqrt(pi) R)) [ I0(R) - I0(-R) + b/2 (I1(R) - I1(-R)) ] with
    I0(a) = sqrt(pi)/(2 zeta) e^{-zeta^2 R^2} erfcx(b/(2 zeta) - zeta a),  I1(a) = (a - b/(2 zeta^2)) I0(a) + e^{-zeta^2 R^2}/(2 zeta^2)."""
    c = torch.as_tensor(centers_bohr)
    n, b = torch.as_tensor(n), torch.as_tensor(b)
    R = (cav.coords[:, None, :] - c[None, :, :]).norm(dim=-1)              # (P, M)
    z = cav.zeta[:, None]
    s = z * z * R * R
    xp, xm = b[None, :] / (2 * z) - z * R, b[None, :] / (2 * z) + z * R
    I0p = SQRT_PI / (2 * z) * _exp_erfcx(xp, s)
    I0m = SQRT_PI / (2 * z) * _exp_erfcx(xm, s)
    e = torch.exp(-s) / (2 * z * z)
    I1p = (R - b[None, :] / (2 * z * z)) * I0p + e
    I1m = (-R - b[None, :] / (2 * z * z)) * I0m + e
    phi = torch.erf(z * R) / R - z / (SQRT_PI * R) * (I0p - I0m + 0.5 * b[None, :] * (I1p - I1m))
    return -(n[None, :] * phi).sum(1)


# ---------------------------------------------------------------------------
# joint solve with an induction model, export
# ---------------------------------------------------------------------------
def joint_blocks(A, L, J, V0, S, f):
    """Blocks of the joint minimization of  1/2 x^T A x + L . x + q . (V0 + J x) + 1/(2f) q^T S q  over the induced
    moments x (A (n, n) SPD, drive L (n,)) and the surface charges q (J (P, n): the moments' surface potential,
    V0 (P,): the permanent potential). Returns (H, g) with H = [[A, J^T], [J, S/f]], g = [L, V0]: the solution is
    y = -H^-1 g, the energy 1/2 g . y. With f = 0 (gas phase) only the induction block is meaningful."""
    n, P = A.shape[-1], S.shape[-1]
    H = A.new_zeros(*A.shape[:-2], n + P, n + P)
    H[..., :n, :n] = A
    H[..., :n, n:] = J.transpose(-1, -2)
    H[..., n:, :n] = J
    H[..., n:, n:] = S / f
    g = torch.cat([L, V0], -1)
    return H, g


def to_pyscf_mm(cav: Cavity, q):
    """(coords bohr, charges, radii bohr) for ``pyscf.qmmm.mm_charge(mf, coords, charges, radii, unit="Bohr")``:
    a Gaussian charge exp(-r^2/radius^2) has radius 1/zeta."""
    return (cav.coords.detach().cpu().numpy(), torch.as_tensor(q).detach().cpu().numpy(),
            (1.0 / cav.zeta).detach().cpu().numpy())
