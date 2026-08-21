"""**Frozen snapshot of the v1 unified pair model. Do not edit.**

This package exists for exactly one reason: ``checkpoints/water_staged/best.pt`` is a trained,
benchmarked model, and the live tree has since moved to the fragment-expert architecture of
``docs/fff_v2.md``. The two are not state-dict compatible -- v1 carries an ``EnvironmentResidual``
and an ``AtomicStateEnergy`` with free-atom anchoring and a per-species offset, none of which
survive into v2 -- so keeping that checkpoint runnable means keeping the code that built it.

What is here is a verbatim copy of ``src/ff/unified.py`` and ``src/ff/atomic_energy.py`` as of the
commit that trained the checkpoint, with two changes and no others:

* import depth (``..x`` -> ``...x``, ``.x`` -> ``..x``), because the files moved one level down;
* ``build.py``, which is ``rsfff.train.train_unified.build_unified_model`` lifted verbatim.

Everything else it needs -- the classical forms, the response solve, the parameter heads, the pair
list -- it imports from the **live** tree. That is safe by construction rather than by luck: the
two-slot refactor was made additive, so every parameter head builds bit-identical modules under its
``p_env = 0`` default. ``tests/test_v1_checkpoint.py`` is what proves it, by pinning this
checkpoint's energies. If that test ever fails, a "backward compatible" change was not.

Usage::

    from rsfff.ff.v1 import build_unified_model
"""

from .build import build_unified_model
from .loader import load_v1_checkpoint
from .unified import ClassicalSpec, UnifiedOutput, UnifiedPairModel

__all__ = [
    "ClassicalSpec",
    "UnifiedOutput",
    "UnifiedPairModel",
    "build_unified_model",
    "load_v1_checkpoint",
]
