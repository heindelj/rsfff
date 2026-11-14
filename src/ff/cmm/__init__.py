"""cmm: the polarizable charge-transfer force field with one learned environment latent (the converged complete model).

    model.py      CmmModel: the energy terms, the joint induction + charge-flux solve
    encoder.py    the one-latent encoder (channels + gated probes -> environment factors)
    config.py     CmmConfig: the architecture choices
    params.py     IonTable (the fixed per-species physics), ChannelTable (CT channels: cutoff, taper)
    kernels.py    shaped-density electrostatics and induction operators
    tabulate.py   the pair tables and batched_precompute (positions -> batch, differentiable; clusters and PBC)
    batch.py      per-frame caches of the training sets and their collation
    environment.py, channels.py   the encoder's inputs; elst.py, gram.py, periodic.py, species.py: term pieces
    io.py         save_cmm / load_cmm; legacy.py: complete-model checkpoints
    frozen_fit.py, solvent.py     the frozen-level fit for new ion tables; the C-PCM terms (not wired into the model)

See :mod:`rsfff.ff.cmm.model` for the functional form.
"""
from .config import CmmConfig  # noqa: F401
from .params import ATOMIC_NUMBER, ChannelTable, IonTable  # noqa: F401
from .environment import (PROBE_NAMES, DensityEnvironment, EnvFeatures, env_invariants, inv_form_of,  # noqa: F401
                          invariant_names, shaped_overlap_derivs)
from .elst import permanent_electrostatics  # noqa: F401
from .tabulate import RadialTable, batched_precompute  # noqa: F401
from .encoder import READOUTS, Encoder  # noqa: F401
from .model import CmmModel  # noqa: F401
from .io import load_cmm, save_cmm  # noqa: F401
from .legacy import convert_state, load_legacy  # noqa: F401
