"""Per-atom prediction heads for the MLIP.

The energy head is a plain invariant MLP: it consumes the lambda=0 SOAP invariants
(``inv_feats``) concatenated with a learned per-species embedding and maps them to a
single scalar energy per atom. Rotation/translation invariance comes for free because
``inv_feats`` are already O3 invariants; the per-molecule energy is the sum of the
per-atom energies (done by the model wrapper, not here).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def mlp(in_dim: int, hidden: int, depth: int, out_dim: int) -> nn.Sequential:
    """Simple SiLU MLP: ``depth`` hidden layers of width ``hidden``."""
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.SiLU()]
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


def exempt_from_weight_decay(module: nn.Module) -> nn.Module:
    """Mark ``module`` so :func:`rsfff.train.term_loop.parameter_groups` gives it no decay.

    The two callers are :func:`zero_init_readout`, for a block whose readout starts at zero, and
    the equivariant heads, which flag *themselves* rather than only their gate. The distinction
    matters and is the reason this is a separate function -- see the ``equiv_reduce`` entry in
    :func:`zero_init_readout`'s docstring.
    """
    # Read by `rsfff.train.term_loop.parameter_groups`, which walks `model.modules()`.
    module.no_weight_decay = True
    return module


def zero_init_readout(module: nn.Module, *, bias: bool = True) -> nn.Module:
    """Zero a block's readout **and** exempt the block from weight decay. Both, always.

    Zero-initializing the last layer is how nearly every head here starts at its per-species
    prior: the residual it predicts begins at exactly zero, so turning a feature dependence on
    starts from the validated feature-free model instead of a different one.

    That device has a failure mode, and it is not obvious enough to leave to each caller. With
    the readout at zero, **every layer behind it has a gradient proportional to zero** -- not
    small, exactly zero, on the first step. Weight decay is then the only force acting on those
    layers and it wins uncontested; by the time the readout has grown enough to send a real
    gradient back, the hidden layers it would send it to have been flattened, and the block is
    left as a constant. Once flat it stays flat: the readout's own gradient carries the hidden
    activations, which no longer vary with the input.

    Measured across a 24-epoch staged fit (first-layer weight norm, init -> frozen -> pol ->
    ct), the blocks that lost the race and what each one silently stopped doing::

        cquad_axis_head.axis.gate_mlp  4.64 -> 2e-05 -> 3e-10 -> 4e-22   quadrupole anisotropy
        disp_params.c6_mlp             4.60 -> 3e-04 -> 3e-09 -> 7e-18   environment-quenched C6
        compliance_head.net            4.63 -> 0.16  -> 8e-05 -> 7e-05   per-channel compliance
        pauli_params.q_mlp             4.58 -> 0.59  -> 0.031 -> 1e-04   Pauli monopole
        environment.inv_mlp            4.62 -> 6e-07 -> 1e-13 -> 4e-20   all many-body content

    None of it raised anything. ``C6`` simply became a per-species constant again, the
    dispersion silently reverted to rigorously two-body, the compliance head gave every channel
    the same answer, and the anisotropic quadrupole axis this model was extended to carry could
    not be predicted at all. The blocks that survived (``chi_mlp``, the equivariant gates, the
    correction trunk) did so only because their readouts grow fast enough to beat the decay --
    it is a race, and nothing about the design says which blocks should win it.

    So the two go together. What replaces weight decay is the penalty that names the quantity it
    wants small -- ``env_weight`` on ``||h_env - h_frag||``, ``r0_spread_weight`` on the range
    separation -- rather than the raw weights of a block that is meant to start at zero.

    **A zero-init readout also poisons its non-exempt siblings, and that is a worse failure than
    the race above.** Every equivariant head here is a zero-init invariant ``gate_mlp`` times a
    learned channel reduction ``equiv_reduce`` of the lambda=1/2 features. Flagging only the
    ``gate_mlp`` -- which is what this function used to do, since it is handed the MLP and not
    the head -- leaves ``equiv_reduce`` decaying, and ``dL/d(equiv_reduce) ∝ gate == 0`` at
    initialization. It is the same "no gradient, decay uncontested" story, but the two
    parameters are now each other's only gradient path, so it is a **deadlock** rather than a
    race: ``equiv_reduce -> 0`` makes the head's output identically zero, which zeroes the
    gradient into ``gate_mlp``, which holds the gate at zero, which keeps ``equiv_reduce``'s
    gradient at zero. Nothing recovers.

    Measured on the same staged fit (``equiv_reduce`` norm, init ~5.66 -> frozen -> pol -> ct)::

        cquad_axis_head.axis.equiv_reduce  0.0    0.0    0.0     deadlocked before epoch 1
        response.alpha_head.equiv_reduce   0.712  0.970  0.176   polarizability anisotropy
        pauli dipole_head.equiv_reduce     0.719  0.840  0.286   Pauli dipole anisotropy
        pauli quadrupole_head.equiv_reduce 1.074  0.997  0.475
        response.chivec_head.equiv_reduce  1.373  1.250  0.830
        response.chiquad_head.equiv_reduce 1.646  1.737  1.061

    ``cquad_axis_head`` lost outright: its ``gate_mlp`` readout was still *exactly* zero after
    210 epochs, so ``anisotropic_cquad`` -- the whole reason that head exists -- never did
    anything, and the fitted Buckingham quadrupole's out-of-plane component was a constant. The
    others merely bled. The fix is :func:`exempt_from_weight_decay` on the **head**, which the
    equivariant heads now call on themselves; this function stays the only place a readout is
    zeroed.

    ``bias=False`` for a readout whose bias carries the block's initial value (a compliance
    head starting at ``s_init``); the exemption is unconditional either way.
    """
    last = module[-1] if isinstance(module, nn.Sequential) else module
    with torch.no_grad():
        last.weight.zero_()
        if bias:
            last.bias.zero_()
    return exempt_from_weight_decay(module)


class AtomicEnergyHead(nn.Module):
    """Per-atom scalar energy from invariant features + a learned species embedding.

    Args
    ----
    p0        : width of the invariant feature vector (``featurizer.feature_dims[0]``).
    n_species : number of distinct atomic species (embedding table size).
    emb_dim   : per-species embedding dimension.
    hidden    : MLP hidden width.
    depth     : number of hidden layers.

    Forward
    -------
    inv_feats   : (N, p0) lambda=0 invariants.
    species_idx : (N,) long, values in [0, n_species).
    returns     : (N,) per-atom energy (unreferenced; the model adds E0 offsets).
    """

    def __init__(
        self,
        p0: int,
        n_species: int,
        *,
        emb_dim: int = 16,
        hidden: int = 64,
        depth: int = 2,
    ) -> None:
        super().__init__()
        self.species_emb = nn.Embedding(n_species, emb_dim)
        # Zero-init readout: the model starts at the per-species E0 reference because the
        # residual it predicts begins at exactly 0. See `zero_init_readout` for why that also
        # means this block must not see weight decay.
        self.mlp = zero_init_readout(mlp(p0 + emb_dim, hidden, depth, 1))

    def forward(self, inv_feats: torch.Tensor, species_idx: torch.Tensor) -> torch.Tensor:
        emb = self.species_emb(species_idx)                      # (N, emb_dim)
        x = torch.cat((inv_feats, emb), dim=-1)                  # (N, p0 + emb_dim)
        return self.mlp(x).squeeze(-1)                           # (N,)
