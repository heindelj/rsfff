"""cmm: the polarizable charge-transfer force field with one learned environment latent (the converged complete model).

    model.py      CmmModel: the energy terms, the joint induction + charge-flux solve
    encoder.py    the one-latent encoder (channels + gated probes -> environment factors)
    config.py     CmmConfig: the architecture choices
    params.py     IonTable (the fixed per-species physics), ChannelTable (CT channels: cutoff, taper)
    kernels.py    shaped-density electrostatics and induction operators
    precompute.py CmmTable (the one radial table) and batched_precompute (positions -> batch, differentiable)
    tabulate.py   the radial-table machinery (quintic Hermite lookup, damped-tensor bases); batch.py: channel types
    environment.py, channels.py   the encoder's inputs; elst.py, gram.py, periodic.py, species.py: term pieces
    io.py         save_cmm / load_cmm; legacy.py: complete-model checkpoints
    frozen_fit.py                 the frozen-level fit for new ion tables

See :mod:`rsfff.ff.cmm.model` for the functional form.
"""
from .config import CmmConfig  # noqa: F401
from .params import ATOMIC_NUMBER, ChannelTable, IonTable  # noqa: F401
from .environment import (PROBE_NAMES, DensityEnvironment, EnvFeatures, env_invariants, inv_form_of,  # noqa: F401
                          invariant_names, shaped_overlap_derivs)
from .elst import permanent_electrostatics  # noqa: F401
from .tabulate import RadialTable  # noqa: F401
from .precompute import CmmTable, batched_precompute, cmm_table  # noqa: F401
from .encoder import READOUTS, Encoder  # noqa: F401
from .model import CmmModel  # noqa: F401
from .io import load_cmm, save_cmm  # noqa: F401
from .legacy import convert_state, load_legacy  # noqa: F401
