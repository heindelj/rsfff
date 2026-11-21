"""The one radial table of the cmm precompute. Its pairs are keyed by the ions' parameter values and
:attr:`CmmTable.VERSION` (:meth:`rsfff.ff.cmm.tabulate.RadialTable.pair_key`): bump the version whenever an integral
of this module, ``tabulate.py`` or ``gram.py`` changes."""
from __future__ import annotations

import torch

from .gram import GramTable
from .params import IonTable
from .tabulate import BOHR, PROBES, RadialTable, XpolTable, _expand, shape_self_coulomb
from ..multipole import shaped_one_center_multi, shaped_two_center_damp

__all__ = ["CmmTable", "cmm_table"]


class CmmTable(GramTable):
    """Every radial function of the cmm precompute on one grid, per ordered species pair: the
    :class:`RadialTable` rows (permanent drive, induced-induced and charge-slot couplings, the diffuse probes' drive
    complements and shell overlaps), the exchange-polarization rows of :class:`XpolTable` and the Gram rows of
    :class:`rsfff.ff.cmm.gram.GramTable`; 0.3 .. 12 A at 0.01 A. ``jself`` (S,): the Coulomb self-energy of each
    species' charge-slot shape."""

    VERSION = 2                 # 2: the overlaps on the finer table k-grid (kernels.table_ovl_grid)
    SITE_VERSION = 2

    def __init__(self, tab: IonTable, r_max=12.0, h=0.01, site=False, **kw):
        self.site = bool(site)
        if self.site and not bool(torch.isfinite(tab.pauli_site_s).all()):
            raise ValueError("CmmTable(site=True): the species table has no site potentials (ions[s]['pauli_site'])")
        self.site_T = int(tab.shell_n.shape[1]) if self.site else 0
        super().__init__(tab, r_max=r_max, breath="shape", probes=PROBES, h=h, **kw)
        self.jself = shape_self_coulomb(self.tab)

    # ---- the site-potential rows (``site=True``) ---------------------------------------------------------------
    # Per ordered pair (i <- j) and slot t of j's site potential v_j = sum_t a_jt y_t, y_t(r) = e^{-beta r}(1 / r +
    # beta / 2), beta = pauli_site_s[j] shell_b[j, t] (unit amplitude; the model holds a):
    #   xs:{d1,q2,b0}:{t}   int dr r^2 u(r) y_t^(L)(r; R): the radial part of <dn_i | y_t> for i's dipole, quadrupole
    #                       and breathing response (the drive); xs:ovl:{t} the same of i's density, <rho_i | y_t>
    #   gs:{ab}{L}:{t}      the Gram rows of i's response with y_t in place of j's density
    # all on the Gram quadrature (GramTable._drive_rows): drive^2 <= 4 ovl Gram holds on the discrete measure.
    #   xs:{don,acc}:{t}    <f_i | y_t>: one electron in i's normalized donor / acceptor frontier shape in y_t (closed
    #                       form: the potential of a unit point charge minus that of a normalized 1s of exponent beta)
    SITE_DRIVE = ("d1", "q2", "b0", "ovl")
    SITE_FRONT = ("don", "acc")

    def _site_beta(self, s):
        return [float(self.tab.pauli_site_s[s] * self.tab.shell_b[s, t]) if float(self.tab.shell_n[s, t]) > 0 else 0.0
                for t in range(self.site_T)]

    def _site_names(self):
        out = []
        for t in range(self.site_T):
            out += [f"xs:{k}:{t}" for k in self.SITE_DRIVE + self.SITE_FRONT]
            out += [f"{k.replace('gram:', 'gs:')}:{t}" for k in self._gram_names()]
        return out

    def _key_extra(self):
        return super()._key_extra() + (f"|site{self.SITE_VERSION}" if self.site else "")

    def pair_key(self, si: int, sj: int) -> str:
        key = super().pair_key(si, sj)
        if not self.site:
            return key
        import hashlib
        return hashlib.sha256(f"{key}|{self._site_beta(sj)!r}".encode()).hexdigest()[:32]      # (the frontier shapes are in the base key)

    def _site_front_values(self, si, sj, r):
        tab, rb, G = self.tab, r / BOHR, r.shape[0]
        one = torch.zeros(G, 4, dtype=r.dtype)
        one[:, 0] = 1.0
        out = {}
        for t, beta in enumerate(self._site_beta(sj)):
            for name, role in zip(self.SITE_FRONT, ("donor", "acceptor")):
                if beta <= 0:
                    out[f"xs:{name}:{t}"] = torch.zeros(G, dtype=r.dtype)
                    continue
                bp, wp = _expand(tab.front_b[role][si], tab.front_w[role][si], G)
                c = shaped_two_center_damp(bp, torch.full((G,), beta, dtype=r.dtype), wp, one, rb, 2) \
                    - shaped_one_center_multi(bp, wp, rb, 2)
                out[f"xs:{name}:{t}"] = c[0] / rb
        return out

    def _site_rows(self, si, sj):
        d = self.FD_STEP
        with torch.no_grad():
            o = self._site_front_values(si, sj, torch.cat([self.r + k * d for k in (-2, -1, 0, 1, 2)]))
        res = {}
        for k, f in o.items():
            f = f.reshape(5, -1)
            d1 = (f[0] - 8 * f[1] + 8 * f[3] - f[4]) / (12 * d)
            d2 = (-f[0] + 16 * f[1] - 30 * f[2] + 16 * f[3] - f[4]) / (12 * d * d)
            res[k] = torch.stack([f[2], d1, d2], -1)
        for t, beta in enumerate(self._site_beta(sj)):
            partner = ("site", beta)
            res.update({f"xs:{k}:{t}": v for k, v in self._drive_rows(si, sj, partner).items()})
            res.update({f"{k.replace('gram:', 'gs:')}:{t}": v for k, v in self._gram_rows(si, sj, partner).items()})
        return res

    def _names(self):
        return RadialTable._names(self) + XpolTable._xpol_names() + self._gram_names() + self._site_names()

    def _functions(self, si, sj):
        out = RadialTable._functions(self, si, sj)
        out.update(XpolTable._xpol_rows(self, si, sj))
        out.update(GramTable._gram_rows(self, si, sj))
        if self.site:
            out.update(self._site_rows(si, sj))
        return out

    def to(self, device):
        super().to(device)
        self.jself = self.jself.to(device)
        return self


def cmm_table(tab: IonTable, store=None, site=False) -> CmmTable:
    """The :class:`CmmTable` of a species table; its pairs come from ``store``, the disk cache or are built on
    first use. ``site``: with the site-potential rows (the model's ``xpol_form = "site"``)."""
    return CmmTable(tab, store=store, site=site)
