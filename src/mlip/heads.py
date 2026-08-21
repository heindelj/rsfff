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


def exempt_parameter(param: torch.Tensor) -> torch.Tensor:
    """Exempt **one parameter** from weight decay, rather than a whole module.

    :func:`exempt_from_weight_decay` flags a module and takes everything inside it with it,
    which is the right granularity when the whole block starts at zero. The two-slot layers
    below need finer: ``w_frag`` is normally initialized, has live gradients from the first
    step, and should keep decaying like any other weight -- only ``w_env`` is exempt.

    The flag rides on the parameter object, which survives ``.to()``, ``.cuda()`` and
    ``load_state_dict`` (all of which write into ``.data`` rather than rebinding), so setting it
    once at construction is enough. It is deliberately *not* in the state dict: it is a property
    of how the model was built, and a checkpoint that disagreed with the code would be worse
    than no record at all.
    """
    param._no_weight_decay = True
    return param


def mark_env_slot(param: torch.Tensor) -> torch.Tensor:
    """Tag a parameter as belonging to the **environment slot**, and exempt it from decay.

    This is what makes "the environment sector" a set of tensors rather than an argument about
    which module reads which stream (``docs/fff_v2.md`` §4). :func:`env_parameters` and
    :func:`core_parameters` are the readers; the penalty, the per-quantity diagnostics and the
    "switch the environment off entirely" ablation all key off it.

    **Why exempt from decay.** Not for the zero-init reason in :func:`zero_init_readout` -- that
    failure needs a gradient that is *identically* zero, and ``dL/dw_env = dL/d(pre-act) (x) eta``
    is generically nonzero from the first step, because the fragment half of the layer is live.
    The reason is that decay on ``w_env`` is a second, unnamed prior pulling toward "no
    environment dependence", competing with ``L_env``, which is the *named* one and is
    calibrated in units of the parameter it acts on. Two knobs for one decision, one of them
    invisible, is how ``env_norm`` reached an exact zero the last time.
    """
    param._env_slot = True
    return exempt_parameter(param)


def env_parameters(module: nn.Module):
    """Every environment-slot parameter in ``module``, in ``named_parameters`` order."""
    for name, p in module.named_parameters():
        if getattr(p, "_env_slot", False):
            yield name, p


def core_parameters(module: nn.Module):
    """Every parameter that is **not** environment-slot: the isolated-fragment sector.

    ``theta_0 = P(h, 0)`` is a function of exactly these. Freezing them is the environment-only
    ablation of ``docs/fff_v2.md`` §4 -- kept because it answers "how much can the environment
    sector explain on its own", not because the training schedule needs it.
    """
    for name, p in module.named_parameters():
        if not getattr(p, "_env_slot", False):
            yield name, p


class TwoSlotLinear(nn.Module):
    """``W_frag . h + W_env . eta + W_tail . t + b``: the first layer of every parameterizer.

    The input layout is the one every head in this package already builds::

        x = cat(( inv_feats , species_emb ))        and inv_feats = cat(( h , eta ))
          = [  h  |  eta  |  emb  ]
             p_frag  p_env  p_tail

    so the environment block sits in the **middle**, and the species embedding is part of the
    *fragment* description -- an atom's element is a property of the atom, not of what is near
    it. That is why this takes a tail width rather than assuming the environment is last: the
    alternative was reordering the concatenation inside a dozen heads, which is exactly the kind
    of convention that has to be enforced by hand and eventually is not.

    **A narrow input means the isolated fragment.** Handed an ``x`` of width
    ``p_frag + p_tail`` -- what :meth:`rsfff.ff.slots.SlotFeatures.isolated` produces once the
    embedding is appended -- this drops the environment term and returns the isolated evaluation
    ``theta_0``. It is bit-identical to passing a zeroed environment block (it adds an exact
    ``0.0`` either way), allocates nothing on a path taken twice per forward, and makes
    ``theta_0`` visibly a function of ``h`` alone rather than of a zero the caller has to
    remember to supply.

    The weight is assembled as a small ``cat`` of the three blocks and applied in one matmul.
    The cat is on the ``(out, p)`` parameter, never on the ``(N, p)`` batch, so it is free next
    to the matmul it feeds -- and it is what lets ``w_env`` be a *separate parameter*, which is
    the whole point: :func:`mark_env_slot` tags it, so the environment sector is a set of
    tensors that :func:`env_parameters`, the ``L_env`` penalty and the ablation can all name.

    ``w_env`` is **zero-initialized**, so a freshly built model is bit-identical to one with no
    environment slot at all and every fit starts from the environment-free answer.
    """

    def __init__(
        self,
        p_frag: int,
        p_env: int,
        out_dim: int,
        *,
        p_tail: int = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if p_frag <= 0 or p_env <= 0:
            raise ValueError(
                f"TwoSlotLinear needs both slots, got p_frag={p_frag}, p_env={p_env}; "
                f"for a single-slot layer build nn.Linear (see two_slot_mlp)"
            )
        self.p_frag, self.p_env, self.p_tail = int(p_frag), int(p_env), int(p_tail)
        # `nn.Linear`'s own initialization over the full fragment width, then split, so a
        # two-slot model and a one-slot model start from the same distribution on the columns
        # they share -- fan_in is what sets the scale and it must not change with `p_env`.
        ref = nn.Linear(self.p_frag + self.p_tail, out_dim, bias=bias)
        w = ref.weight.detach().clone()
        self.w_frag = nn.Parameter(w[:, : self.p_frag].contiguous())
        self.w_tail = (
            nn.Parameter(w[:, self.p_frag :].contiguous()) if self.p_tail else None
        )
        self.bias = nn.Parameter(ref.bias.detach().clone()) if bias else None
        self.w_env = mark_env_slot(nn.Parameter(torch.zeros(out_dim, self.p_env)))

    def weight(self, *, isolated: bool) -> torch.Tensor:
        """``(out, p)`` the assembled weight, with or without the environment columns."""
        blocks = [self.w_frag] if isolated else [self.w_frag, self.w_env]
        if self.w_tail is not None:
            blocks.append(self.w_tail)
        return blocks[0] if len(blocks) == 1 else torch.cat(blocks, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        narrow = self.p_frag + self.p_tail
        if x.shape[-1] == narrow:                            # the isolated evaluation
            return torch.nn.functional.linear(x, self.weight(isolated=True), self.bias)
        if x.shape[-1] != narrow + self.p_env:
            raise ValueError(
                f"TwoSlotLinear got input width {x.shape[-1]}; expected {narrow} (isolated) "
                f"or {narrow + self.p_env} (joined), for p_frag={self.p_frag}, "
                f"p_env={self.p_env}, p_tail={self.p_tail}"
            )
        return torch.nn.functional.linear(x, self.weight(isolated=False), self.bias)

    def extra_repr(self) -> str:
        return (
            f"p_frag={self.p_frag}, p_env={self.p_env}, p_tail={self.p_tail}, "
            f"out={self.w_frag.shape[0]}"
        )


def two_slot_mlp(
    p_frag: int,
    p_env: int,
    hidden: int,
    depth: int,
    out_dim: int,
    *,
    p_tail: int = 0,
) -> nn.Sequential:
    """:func:`mlp` with the two slots split in its **first layer only**.

    Only the first layer needs the split: past it the slots have been mixed on purpose, and what
    remains to be named is which weights the environment enters *through*.

    ``p_tail`` is the width of anything concatenated after the features -- in practice the
    species embedding. See :class:`TwoSlotLinear` for why it is not simply folded into
    ``p_frag``.

    ``p_env = 0`` returns exactly ``mlp(p_frag + p_tail, hidden, depth, out_dim)``: the same
    modules, the same parameter names, the same state dict. That is not a convenience.
    ``rsfff.ff.v1`` imports these heads from the live tree so ``checkpoints/water_staged/best.pt``
    keeps loading, and ``tests/test_v1_checkpoint.py`` fails the moment it stops being true.
    """
    if not p_env:
        return mlp(p_frag + p_tail, hidden, depth, out_dim)
    layers: list[nn.Module] = [
        TwoSlotLinear(p_frag, p_env, hidden, p_tail=p_tail),
        nn.SiLU(),
    ]
    d = hidden
    for _ in range(depth - 1):
        layers += [nn.Linear(d, hidden), nn.SiLU()]
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


def slot_reduce(
    frag_weight: torch.Tensor,
    env_weight: torch.Tensor | None,
    n_channels: int,
) -> torch.Tensor:
    """The channel-reduction matrix for a ``(N, m, p)`` equivariant feature block.

    The equivariant heads contract the lambda=1/2 features against a learned ``(p, k)``
    reduction. Under two slots ``p = p_frag + p_env``, and the two halves are separate
    parameters for the same reason the linear layer's are: only the environment half is tagged
    and exempt. ``n_channels`` is the width of the block actually being contracted, which
    selects the isolated (``p_frag``) or joined (``p_frag + p_env``) form -- the same
    narrow-input convention :class:`TwoSlotLinear` uses.
    """
    if env_weight is None or n_channels == frag_weight.shape[0]:
        return frag_weight
    if n_channels != frag_weight.shape[0] + env_weight.shape[0]:
        raise ValueError(
            f"slot_reduce got a feature block of width {n_channels}; expected "
            f"{frag_weight.shape[0]} (isolated) or "
            f"{frag_weight.shape[0] + env_weight.shape[0]} (joined)"
        )
    return torch.cat((frag_weight, env_weight), dim=0)


def env_reduce_parameter(p_env: int | None, n_channels: int) -> nn.Parameter | None:
    """The zero-initialized environment half of an equivariant reduction, or ``None``.

    ``None`` when there is no environment slot, which is what keeps the head's parameter names
    -- and hence a v1 checkpoint -- unchanged.
    """
    if not p_env:
        return None
    return mark_env_slot(nn.Parameter(torch.zeros(int(p_env), int(n_channels))))


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
