"""Breathing as the contraction of the permanent shells (theory/parameter_free_xpol.tex, step 1).

Each shell t of an ion is a von Weizsaecker orbital phi_t = sqrt(n_t s_t) (s_t the normalized Slater 1s density with
exponent b_t). Its l = 0 response is a change of exponent, eps_t = d ln b_t, with the electron-density mode

    m_t(r) = n_t b_t d s_t / d b_t = n_t (3 - b_t r) s_t(r)                        (zero monopole: no net charge)

-- the ``contraction`` modes, one per shell, weighted by the shell's occupation. An ion with one shell gets its
contraction and the second-order mode m_2 = n (b d/db)^2 s = n ((3 - b r)^2 - b r) s, so every breathing ion carries two
l = 0 response variables. Their stiffness H0 = A^-1 (2 x 2 per species) is fitted to the free ion's CPKS l = 0
response to shell probes (scripts/ion_contract_cpks.py), not learned. Species without an l = 0 response (Li+, Na+:
a0 = 0 in the ion table) carry none.

:class:`ContractTable` tabulates, per ordered species pair (i <- j) and modes t, u of i (and j):

    ct:pot:{t}     int m_t(r) Phi_j(|r - R|) d^3r      Phi_j the potential of j's frozen charges (nucleus Z_j and shells);
                                                       the Coulomb drive is -sum_j ct:pot (electrons)
    ct:ovl:{t}     int m_t(r) rho_j(|r - R|) d^3r      the change of the density overlap O_ij (the exchange drive)
    ct:J:{t}{u}    int int m_t,i(r) m_u,j(r') / |r - r' + R|    the Coulomb coupling of the two ions' modes
    ct:ovl2:{t}    1/2 d^2 O_t / d eps_t^2 = 1/2 int n_t (b d/db)^2 s_t rho_j    the exact second order of the overlap
                   under the shell's exponent scaling (mode t a pure scaling: both modes of a two-shell ion, mode 0 of
                   a one-shell ion; 0 for the one-shell ion's second mode, whose second order the Gram term keeps)

all short-ranged (the modes carry no charge), by the spherical convolution
f(R) = (2 pi / R) int r m(r) [G(r + R) - G(|r - R|)] dr with G(s) = int_0^s s' g(s') ds'.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from ..ionct import kernels as K
from .tabulate import RadialTable

__all__ = ["ContractTable", "TranslateTable", "contraction_modes", "load_contract_A", "load_translate_f", "N_CONTRACT"]

BOHR = K.BOHR_ANG
N_CONTRACT = 2
_RQ_STEP, _RQ_END = 0.01, 30.0          # bohr: the r quadrature
_S_STEP, _S_END = 0.0005, 60.0          # bohr: the antiderivative grid


def _s1(b, r):
    return b ** 3 / (8 * math.pi) * np.exp(-b * r)


def breathing_species(tab):
    """(S,) bool: the ions with an l = 0 response (a0 > 0 in the ion table)."""
    a0 = getattr(tab, "a0", None)
    return [bool(a0 is not None and float(a0[s]) > 0) for s in range(len(tab.symbols))]


def contraction_modes(tab, s, r):
    """The two l = 0 electron-density modes (numpy arrays at r, bohr) of species s per unit amplitude: per-shell
    contractions for a two-shell ion, contraction and its second order for a one-shell ion; zeros if s does not
    breathe."""
    r = np.asarray(r, dtype=np.float64)
    if not breathing_species(tab)[s]:
        return [np.zeros_like(r), np.zeros_like(r)]
    sh = [(float(tab.shell_n[s, t]), float(tab.shell_b[s, t])) for t in range(tab.shell_n.shape[1])
          if float(tab.shell_n[s, t]) > 0]
    if len(sh) >= 2:
        return [n * (3 - b * r) * _s1(b, r) for n, b in sh[:2]]
    n, b = sh[0]
    x = 3 - b * r
    return [n * x * _s1(b, r), n * (x * x - b * r) * _s1(b, r)]


def scaling_second_order(tab, s, r):
    """The second-order densities n_t (b d/db)^2 s_t of the pure-scaling modes of species s (zeros elsewhere)."""
    r = np.asarray(r, dtype=np.float64)
    z = [np.zeros_like(r), np.zeros_like(r)]
    if not breathing_species(tab)[s]:
        return z
    sh = [(float(tab.shell_n[s, t]), float(tab.shell_b[s, t])) for t in range(tab.shell_n.shape[1])
          if float(tab.shell_n[s, t]) > 0]
    for t, (n, b) in enumerate(sh[:2]):
        x = 3 - b * r
        z[t] = n * (x * x - b * r) * _s1(b, r)
    return z


def scaling_mask(tab):
    """(S, 2) 1 where contraction mode t of the species is a pure exponent scaling of one shell."""
    out = []
    for s in range(len(tab.symbols)):
        nsh = sum(1 for t in range(tab.shell_n.shape[1]) if float(tab.shell_n[s, t]) > 0)
        br = breathing_species(tab)[s]
        out.append([float(br), float(br and nsh >= 2)])
    return torch.tensor(out, dtype=torch.float64)


def load_contract_A(path, symbols):
    """(S, 2, 2) the CPKS compliance A = H0^-1 per species from ``path`` (zeros where absent / not breathing)."""
    A = torch.zeros(len(symbols), N_CONTRACT, N_CONTRACT, dtype=torch.float64)
    if path is None or not Path(path).exists():
        return A
    d = json.loads(Path(path).read_text())
    d = d.get("ions", d)
    for k, s in enumerate(symbols):
        if s in d and d[s].get("A") is not None:
            A[k] = torch.tensor(d[s]["A"], dtype=torch.float64)
    return A


def _antiderivative(sg):
    """G(s) = int_0^s sg(s') ds' on the fine grid (trapezoid), as a function for np.interp."""
    s = np.arange(0.0, _S_END + _S_STEP / 2, _S_STEP)
    y = sg(s)
    G = np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * _S_STEP)])
    return lambda x: np.interp(x, s, G)


def _phi_frozen(tab, s_):
    """s * Phi_j(s): the potential of species s_'s frozen charges (pseudo-nucleus Z and shells) times s."""
    Z = float(tab.Z[s_])
    sh = [(float(tab.shell_n[s_, t]), float(tab.shell_b[s_, t])) for t in range(tab.shell_n.shape[1])
          if float(tab.shell_n[s_, t]) > 0]

    def f(s):
        out = np.full_like(s, Z)
        for n, b in sh:
            out = out - n * (1 - np.exp(-b * s) * (1 + 0.5 * b * s))
        return out
    return f


def _rho_s(tab, s_):
    sh = [(float(tab.shell_n[s_, t]), float(tab.shell_b[s_, t])) for t in range(tab.shell_n.shape[1])
          if float(tab.shell_n[s_, t]) > 0]
    return lambda s: s * sum(n * _s1(b, s) for n, b in sh)


def _mode_potential(m_of_r):
    """s * phi_m(s) for a spherical density m (callable of r): Q_in(s) + s * int_s^inf 4 pi r m dr."""
    s = np.arange(0.0, _S_END + _S_STEP / 2, _S_STEP)
    m = m_of_r(s)
    qin = np.concatenate([[0.0], np.cumsum(0.5 * ((4 * math.pi * s * s * m)[1:] + (4 * math.pi * s * s * m)[:-1]) * _S_STEP)])
    w = 4 * math.pi * s * m
    tail = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * _S_STEP)])
    out_ = tail[-1] - tail
    y = qin + s * out_
    return lambda x: np.interp(x, s, y)


class ContractTable(RadialTable):
    """The contraction modes' drives and couplings (module docstring), per ordered species pair, with first and
    second R-derivatives (Angstrom) by fourth-order finite differences."""

    def __init__(self, tab, r_max=10.0, **kw):
        self._G = {}
        with torch.no_grad():
            super().__init__(tab, probes=(), r_max=r_max, **kw)
        del self._G

    def _conv(self, m_i, key, sg):
        """(G + 4,) f(R) = (2 pi / R) int r m_i(r) [G(r + R) - G(|r - R|)] dr on the padded R grid (bohr)."""
        if key not in self._G:
            self._G[key] = _antiderivative(sg)
        G = self._G[key]
        h = self.h
        Rg = np.concatenate([float(self.r[0]) - h * np.arange(2, 0, -1), self.r.numpy(),
                             float(self.r[-1]) + h * np.arange(1, 3)]) / BOHR
        r = np.arange(_RQ_STEP / 2, _RQ_END, _RQ_STEP)
        w = r * m_i(r) * _RQ_STEP
        out = np.empty(Rg.shape[0])
        for k in range(0, Rg.shape[0], 64):
            R = Rg[k:k + 64, None]
            out[k:k + 64] = (2 * math.pi / R[:, 0]) * ((G(r[None] + R) - G(np.abs(r[None] - R))) * w[None]).sum(1)
        return out

    def _functions(self, si, sj):
        tab = self.tab
        mi = [lambda r, t=t: contraction_modes(tab, si, r)[t] for t in range(N_CONTRACT)]
        mj = [lambda r, t=t: contraction_modes(tab, sj, r)[t] for t in range(N_CONTRACT)]
        m2 = [lambda r, t=t: 0.5 * scaling_second_order(tab, si, r)[t] for t in range(N_CONTRACT)]
        out = {}
        for t in range(N_CONTRACT):
            out[f"ct:pot:{t}"] = self._conv(mi[t], ("phi", sj), _phi_frozen(tab, sj))
            out[f"ct:ovl:{t}"] = self._conv(mi[t], ("rho", sj), _rho_s(tab, sj))
            out[f"ct:ovl2:{t}"] = self._conv(m2[t], ("rho", sj), _rho_s(tab, sj))
            for u in range(N_CONTRACT):
                out[f"ct:J:{t}{u}"] = self._conv(mi[t], ("mode", sj, u), _mode_potential(mj[u]))
        if not self.names:
            self.names = list(out)
            self.index = {k: n for n, k in enumerate(self.names)}
        h = self.h
        res = []
        for k in self.names:
            f = torch.tensor(out[k], dtype=self.r.dtype)
            d1 = (f[:-4] - 8 * f[1:-3] + 8 * f[3:-1] - f[4:]) / (12 * h)
            d2 = (-f[:-4] + 16 * f[1:-3] - 30 * f[2:-2] + 16 * f[3:-1] - f[4:]) / (12 * h * h)
            res.append(torch.stack([f[2:-2], d1, d2], -1))
        return res


# ---------------------------------------------------------------------------------------------- dipole: shell translation
def _fd1(f, h):
    """Fourth-order first derivative of a uniformly sampled array (trims two points each side)."""
    return (f[:-4] - 8 * f[1:-3] + 8 * f[3:-1] - f[4:]) / (12 * h)


class TranslateTable(RadialTable):
    """The rigid translation of each permanent shell (the l = 1 deformation; theory/parameter_free_xpol.tex): per
    ordered species pair (i <- j) and shell t of i (zeros where i has no shell t),

        tr:S:{t}     S_t(R) = <s_t | rho_j(. - R)>      the normalized shell's overlap with j's electron density
        tr:S1:{t}    dS_t/dR      (bohr^-4)
        tr:S2:{t}    d^2S_t/dR^2  (bohr^-5)

    Translating shell t by delta changes the overlap to n_t S_t(|R - delta|) exactly: drive n_t S_t' and second order
    n_t [S_t'' R^R^ + (S_t'/R)(1 - R^R^)]."""

    def __init__(self, tab, r_max=10.0, **kw):
        self._G = {}
        with torch.no_grad():
            super().__init__(tab, probes=(), r_max=r_max, **kw)
        del self._G

    def _conv6(self, m_i, key, sg):
        if key not in self._G:
            self._G[key] = _antiderivative(sg)
        G = self._G[key]
        h = self.h
        Rg = np.concatenate([float(self.r[0]) - h * np.arange(6, 0, -1), self.r.numpy(),
                             float(self.r[-1]) + h * np.arange(1, 7)]) / BOHR
        r = np.arange(_RQ_STEP / 2, _RQ_END, _RQ_STEP)
        w = r * m_i(r) * _RQ_STEP
        out = np.empty(Rg.shape[0])
        for k in range(0, Rg.shape[0], 64):
            R = Rg[k:k + 64, None]
            out[k:k + 64] = (2 * math.pi / R[:, 0]) * ((G(r[None] + R) - G(np.abs(r[None] - R))) * w[None]).sum(1)
        return out

    def _functions(self, si, sj):
        tab = self.tab
        hb = self.h / BOHR
        sh = [(float(tab.shell_n[si, t]), float(tab.shell_b[si, t])) for t in range(tab.shell_n.shape[1])
              if float(tab.shell_n[si, t]) > 0]
        out = {}
        for t in range(2):
            if t < len(sh):
                b = sh[t][1]
                S = self._conv6(lambda r, b=b: _s1(b, r), ("rho", sj), _rho_s(tab, sj))       # pad 6
            else:
                S = np.zeros(self.G + 12)
            S1 = _fd1(S, hb)                                                                    # pad 4
            S2 = _fd1(S1, hb)                                                                   # pad 2
            out[f"tr:S:{t}"], out[f"tr:S1:{t}"], out[f"tr:S2:{t}"] = S[4:-4], S1[2:-2], S2
        if not self.names:
            self.names = list(out)
            self.index = {k: n for n, k in enumerate(self.names)}
        h = self.h
        res = []
        for k in self.names:
            f = torch.tensor(out[k], dtype=self.r.dtype)
            d1 = (f[:-4] - 8 * f[1:-3] + 8 * f[3:-1] - f[4:]) / (12 * h)
            d2 = (-f[:-4] + 16 * f[1:-3] - 30 * f[2:-2] + 16 * f[3:-1] - f[4:]) / (12 * h * h)
            res.append(torch.stack([f[2:-2], d1, d2], -1))
        return res


def load_translate_f(path, symbols, key="f"):
    """(S, 2) the free-ion field partition f_t of the induced dipole between the shells (translation CPKS fit;
    scripts/ion_translate_cpks.py); one-shell ions and absent entries: (1, 0)."""
    f = torch.zeros(len(symbols), 2, dtype=torch.float64)
    f[:, 0] = 1.0
    if path is None or not Path(path).exists():
        return f
    d = json.loads(Path(path).read_text())
    d = d.get("ions", d)
    for k, s in enumerate(symbols):
        if s in d and d[s].get(key) is not None:
            f[k] = torch.tensor(d[s][key], dtype=torch.float64)
    return f
