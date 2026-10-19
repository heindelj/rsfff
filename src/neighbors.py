"""Moved to bb: this module *is* :mod:`bb.graph.neighbors`.

``rsfff.neighbors`` is an alias kept for scripts (``rsfff.neighbors.BACKEND``,
``build_radius_graph``, ...). It is the same module object, so ``set_strict``,
``CAP_EVENTS`` and monkeypatching behave exactly as before. The ``RSFFF_STRICT_NEIGHBORS`` /
``RSFFF_NEIGHBOR_CAP_CHECK`` environment variables are still read, alongside the ``BB_*``
names.
"""

import sys

from bb.graph import neighbors as _neighbors

sys.modules[__name__] = _neighbors
