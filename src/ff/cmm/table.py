"""The one radial table of the cmm precompute. Its pairs are keyed by the ions' parameter values and
:attr:`CmmTable.VERSION` (:meth:`rsfff.ff.cmm.tabulate.RadialTable.pair_key`): bump the version whenever an integral
of this module, ``tabulate.py`` or ``gram.py`` changes."""
from __future__ import annotations

from .gram import GramTable
from .params import IonTable
from .tabulate import PROBES, RadialTable, XpolTable, shape_self_coulomb

__all__ = ["CmmTable", "cmm_table"]


class CmmTable(GramTable):
    """Every radial function of the cmm precompute on one grid, per ordered species pair: the
    :class:`RadialTable` rows (permanent drive, induced-induced and charge-slot couplings, the diffuse probes' drive
    complements and shell overlaps), the exchange-polarization rows of :class:`XpolTable` and the Gram rows of
    :class:`rsfff.ff.cmm.gram.GramTable`; 0.3 .. 12 A at 0.01 A. ``jself`` (S,): the Coulomb self-energy of each
    species' charge-slot shape."""

    VERSION = 1

    def __init__(self, tab: IonTable, r_max=12.0, h=0.01, **kw):
        super().__init__(tab, r_max=r_max, breath="shape", probes=PROBES, h=h, **kw)
        self.jself = shape_self_coulomb(self.tab)

    def _names(self):
        return RadialTable._names(self) + XpolTable._xpol_names() + self._gram_names()

    def _functions(self, si, sj):
        out = RadialTable._functions(self, si, sj)
        out.update(XpolTable._xpol_rows(self, si, sj))
        out.update(GramTable._gram_rows(self, si, sj))
        return out

    def to(self, device):
        super().to(device)
        self.jself = self.jself.to(device)
        return self


def cmm_table(tab: IonTable, store=None) -> CmmTable:
    """The :class:`CmmTable` of a species table; its pairs come from ``store``, the disk cache or are built on
    first use."""
    return CmmTable(tab, store=store)
