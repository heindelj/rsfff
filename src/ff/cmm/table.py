"""The one radial table of the cmm precompute (its own module: the disk cache of the table is keyed by the source of
the table code -- this module, ``tabulate.py`` and ``gram.py`` -- so edits of the precompute keep the cached tables)."""
from __future__ import annotations

from .gram import GramTable
from .params import IonTable
from .tabulate import PROBES, RadialTable, XpolTable, cached_table, shape_self_coulomb

__all__ = ["CmmTable", "cmm_table"]


class CmmTable(GramTable):
    """Every radial function of the cmm precompute on one grid, per ordered species pair: the
    :class:`RadialTable` rows (permanent drive, induced-induced and charge-slot couplings, the diffuse probes' drive
    complements and shell overlaps), the exchange-polarization rows of :class:`XpolTable` and the Gram rows of
    :class:`rsfff.ff.cmm.gram.GramTable`; 0.3 .. 12 A at 0.01 A. ``jself`` (S,): the Coulomb self-energy of each
    species' charge-slot shape."""

    def __init__(self, tab: IonTable, r_max=12.0, h=0.01, **kw):
        super().__init__(tab, r_max=r_max, breath="shape", probes=PROBES, h=h, **kw)
        self.jself = shape_self_coulomb(tab)

    def _functions(self, si, sj):
        out = RadialTable._functions(self, si, sj)
        out.update(XpolTable._xpol_rows(self, si, sj))
        out.update(GramTable._gram_rows(self, si, sj))
        return out

    def to(self, device):
        super().to(device)
        self.jself = self.jself.to(device)
        return self


def cmm_table(tab: IonTable) -> CmmTable:
    """The :class:`CmmTable` of a species table, kept on disk under ``RSFFF_TABLE_CACHE`` (:func:`cached_table`)."""
    return cached_table(CmmTable, tab)
