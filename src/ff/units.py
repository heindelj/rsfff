"""Unit constants for the force-field terms.

The training pipeline works in **mixed Hartree/Angstrom** units (see the table in
``rsfff.train.data``): positions are Angstrom, energies Hartree. Force-field
*parameters*, by contrast, are stored in atomic units, because that is how they
are quoted and fit in the literature and in pyCMM -- a C6 of ``35.83`` is
recognizable, ``0.7869 Ha*Ang^6`` is not.

Each FF term converts at exactly one place, on the way in.
"""

from __future__ import annotations

#: Angstrom per bohr (CODATA 2018). Matches the private ``_BOHR_ANG`` in
#: ``rsfff.mlip.mixture``; that duplicate should eventually import from here.
BOHR_ANG = 0.52917721067

#: kJ/mol per Hartree, for reporting energies at a human-readable scale.
KJMOL_PER_HARTREE = 2625.4996394798254
