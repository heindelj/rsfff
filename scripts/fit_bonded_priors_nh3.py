"""Fit the NH3 bonded priors (N-H Morse k, H-N-H angle, N improper) with constant parameters.

The film model's bonded terms start from per-type tables (``rsfff.ff.film.bonded``); for NH3
those tables come from this fit. It answers one question first -- can the cosine angles
alone describe the umbrella? -- and then fits with the out-of-plane improper:

    targets  HNH 106.7 deg; harmonic w1 3506, w2 1022 (umbrella, A1), w3 3577, w4 1691
             (E bend) cm^-1; classical inversion barrier 5.9 kcal/mol (at the relaxed
             planar bond length)
    fixed    r_e = 1.012 A, D = TAE_e(NH3)/3 = 0.158 Ha (the well-referenced Morse carries
             the atomization energy)
    free     Morse k, theta_eq, k_theta [, c_chi, k_chi]

Result (the numbers in the bonded tables):

    angles only   k .4504  theta_eq 107.60  k_theta .1538
                  -> HNH 107.6, w2 1158, w4 1611, barrier 5.65: the umbrella is 13% stiff and
                     the E bend 5% soft -- one (theta_eq, k_theta) cannot set both
    + improper    k .4502  theta_eq 100.85  k_theta .1919  c_chi -0.505  k_chi .0077
                  -> HNH 106.67, w2 1022, w4 1691, barrier 5.90: all four bend targets

The improper lands in its planarizing regime (c_chi < 0) working against a more pyramidal
theta_eq -- the geometry is a balance of the two, which is what frees the umbrella curvature
from the E bend. The residual is the stretch splitting (w1/w3), which a Morse sum without
bond-bond coupling cannot produce with constant parameters; the network's geometry-dependent
parameters are expected to carry it.

These are literature (CCSD(T)-quality) targets, not wB97M-V; the priors only have to start the
fit in the right basin. Rerun with the wB97M-V harmonic frequencies to refine them.

    python scripts/fit_bonded_priors_nh3.py
"""

from __future__ import annotations

import math

import numpy as np
import scipy.optimize as so
import torch

from rsfff.ff.film.bonded import cosine_angle_energy, morse_energy, wilson_improper_energy
from rsfff.ff.units import BOHR_ANG

torch.set_default_dtype(torch.float64)

AMU = 1822.888486                 # electron masses per amu
HA_TO_CM = 219474.6313
HA_TO_KCAL = 627.5095
MASS = torch.tensor([14.003074, 1.007825, 1.007825, 1.007825]).repeat_interleave(3) * AMU

R_E = 1.012 / BOHR_ANG            # bohr
D_NH = 0.158                      # Ha
TARGET = dict(hnh=106.7, w2=1022.0, w4=1691.0, w1=3506.0, w3=3577.0, barrier=5.9)
SIGMA = dict(hnh=0.5, w2=30.0, w4=30.0, w1=30.0, w3=30.0, barrier=0.3)

LEGS = ((0, 1, 2), (1, 2, 0), (2, 0, 1))


def nh3(r_bohr: float, hnh_deg: float) -> torch.Tensor:
    """C3v NH3 in bohr, N at the origin."""
    c = math.cos(math.radians(hnh_deg))
    cb2 = min(max((c + 0.5) / 1.5, 0.0), 1.0)
    cb, sb = math.sqrt(cb2), math.sqrt(1.0 - cb2)
    h = [[r_bohr * sb * math.cos(2 * math.pi * k / 3),
          r_bohr * sb * math.sin(2 * math.pi * k / 3), -r_bohr * cb] for k in range(3)]
    return torch.tensor([[0.0, 0.0, 0.0]] + h)


def energy(x: torch.Tensor, p: dict) -> torch.Tensor:
    """Bonded NH3 with the film model's own functional forms (atomic units)."""
    b = x[1:] - x[0]
    r = b.norm(dim=-1)
    one = torch.ones(3)
    e = morse_energy(r, R_E * one, D_NH * one, p["k"] * one).sum()
    u = b / r.unsqueeze(-1)
    cos_t = torch.stack([u[0] @ u[1], u[1] @ u[2], u[0] @ u[2]])
    e = e + cosine_angle_energy(cos_t, math.cos(p["theta_eq"]) * one, p["k_theta"] * one).sum()
    if p.get("k_chi", 0.0) > 0.0:
        s2 = []
        for leg, j, k in LEGS:
            n = torch.cross(u[j], u[k], dim=0)
            s2.append((u[leg] @ n) ** 2 / (n @ n))
        s2 = torch.stack(s2)
        e = e + wilson_improper_energy(s2, p["c_chi"] * one, p["k_chi"] * one).sum()
    return e


def analyze(p: dict) -> dict:
    x = nh3(R_E, 106.7).clone().requires_grad_(True)
    opt = torch.optim.LBFGS([x], max_iter=300, tolerance_grad=1e-11,
                            tolerance_change=1e-15, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        e = energy(x, p)
        e.backward()
        return e

    opt.step(closure)
    xe = x.detach()
    hess = torch.autograd.functional.hessian(lambda y: energy(y.view(4, 3), p), xe.reshape(-1))
    mw = hess / torch.sqrt(MASS[:, None] * MASS[None, :])
    w = sorted((torch.linalg.eigvalsh(mw).clamp(min=0).sqrt() * HA_TO_CM).tolist())[-6:]
    b = xe[1:] - xe[0]
    hnh = math.degrees(math.acos(float(b[0] @ b[1] / (b[0].norm() * b[1].norm()))))
    planar = min(float(energy(nh3(float(r), 120.0), p))
                 for r in torch.linspace(0.95, 1.05, 101) / BOHR_ANG)
    return dict(r=float(b[0].norm()) * BOHR_ANG, hnh=hnh, w1=w[3], w2=w[0], w3=w[4], w4=w[1],
                barrier=(planar - float(energy(xe, p))) * HA_TO_KCAL)


def residuals(p: dict) -> np.ndarray:
    a = analyze(p)
    return np.array([(a[k] - TARGET[k]) / SIGMA[k] for k in TARGET])


def fit(with_improper: bool):
    if with_improper:
        names = ["k", "theta_eq", "k_theta", "c_chi", "k_chi"]
        x0, lo, hi = [0.45, math.radians(100.0), 0.2, 0.3, 0.05], \
            [0.2, math.radians(80), 0.01, -0.999, 0.0], [0.8, math.radians(130), 2.0, 0.999, 20.0]
    else:
        names = ["k", "theta_eq", "k_theta"]
        x0, lo, hi = [0.45, math.radians(107.0), 0.17], \
            [0.2, math.radians(80), 0.01], [0.8, math.radians(130), 2.0]
    sol = so.least_squares(lambda v: residuals(dict(zip(names, v))), x0,
                           bounds=(lo, hi), diff_step=1e-4)
    return dict(zip(names, sol.x)), sol.cost


def main() -> None:
    for with_improper in (False, True):
        p, cost = fit(with_improper)
        a = analyze(p)
        label = "angles + improper" if with_improper else "angles only"
        print(f"{label}: cost {cost:.3f}")
        print("   " + "  ".join(
            f"{k} {math.degrees(v):.3f}" if k == "theta_eq" else f"{k} {v:.4f}"
            for k, v in p.items()))
        print(f"   r {a['r']:.4f} A  HNH {a['hnh']:.2f}  w1 {a['w1']:.0f}  w2 {a['w2']:.0f}  "
              f"w3 {a['w3']:.0f}  w4 {a['w4']:.0f}  barrier {a['barrier']:.2f} kcal/mol")


if __name__ == "__main__":
    main()
