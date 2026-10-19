"""The non-variational polarization model: ``docs/fff_nonvariational.md``.

:class:`NonvariationalModel` is the film model with the converged coupled solve replaced by a
fixed number of unrolled iterations whose mutual-induction operator carries learned
induced-density widths (:mod:`rsfff.ff.unrolled_solve`). Selected by
``film.model: nonvariational``.
"""

from .model import NonvariationalModel, diagnose

__all__ = ["NonvariationalModel", "diagnose"]
