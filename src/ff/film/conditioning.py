"""FiLM conditioning: the state modulates how the shared feature basis is used.

``docs/fff_film.md`` §5.2-5.4 in code. For hidden layer ``l``:

    a^l          = W_l h^l + b_l
    (dgamma, beta) = G_l(c_i)                       zero-initialized generator
    h^{l+1}      = SiLU((1 + dgamma) * a^l + beta)

The generators' readouts start at zero, so a fresh model is an ordinary shared MLP and state
specialization develops during training -- and, by :func:`rsfff.mlip.heads.zero_init_readout`,
the generators are exempt from weight decay, because a zero readout otherwise starves every
layer behind it of gradient and decay flattens them uncontested (the measured failure in that
docstring).

On water-only data the conditioning vector is ``[0, ..., 0, u_i]`` with ``u_i = 0`` at every
one-hot assignment (:meth:`rsfff.ff.film.state.StateDescriptor.local_conditioning`), so FiLM
is exercised as software but cannot be trained as chemistry -- ``docs/fff_film.md`` §8.3 says
this is expected, not a failure.

``conditioning_mode`` selects the mechanism:

    "none"        : the conditioning vector is ignored (the ablation baseline);
    "concatenate" : ``c_i`` is appended to the trunk input;
    "film"        : the mechanism above (the default);
    "low_rank"    : reserved -- a generated low-rank weight update (LoRA-style); the mode
                    string is accepted at the config boundary so ablation configs can name it,
                    but constructing a trunk with it raises until an ablation shows FiLM has
                    the features but not the state-dependent mixing (§5.4).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ...mlip.heads import mlp, zero_init_readout

__all__ = ["CONDITIONING_MODES", "ConditionedTrunk", "FiLMGenerator", "FiLMLayer"]

CONDITIONING_MODES = ("none", "concatenate", "film", "low_rank")


class FiLMGenerator(nn.Module):
    """``c -> (dgamma, beta)``, both zero at initialization."""

    def __init__(self, d_c: int, hidden: int, depth: int, width: int) -> None:
        super().__init__()
        self.width = int(width)
        self.net = zero_init_readout(mlp(int(d_c), int(hidden), int(depth), 2 * self.width))

    def forward(self, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(c)
        return out[..., : self.width], out[..., self.width :]


class FiLMLayer(nn.Module):
    """One affine layer with feature-wise modulation: ``SiLU((1 + dgamma) * a + beta)``."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(int(in_dim), int(out_dim))
        self.act = nn.SiLU()

    def forward(
        self,
        x: torch.Tensor,
        modulation: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        a = self.linear(x)
        if modulation is not None:
            dgamma, beta = modulation
            a = (1.0 + dgamma) * a + beta
        return self.act(a)


class ConditionedTrunk(nn.Module):
    """The shared parameter trunk: ``depth`` hidden layers, conditioned per ``mode``.

    Output width is ``hidden`` -- the trunk ends on an activation, and the per-family heads
    own their readouts. The same instance serves both evaluations of the two-pass convention
    (isolated and joined); it holds no state about which it is running.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        depth: int,
        *,
        d_c: int,
        mode: str = "film",
        film_hidden: int = 32,
        film_depth: int = 1,
    ) -> None:
        super().__init__()
        if mode not in CONDITIONING_MODES:
            raise ValueError(
                f"conditioning_mode {mode!r} is not one of {CONDITIONING_MODES}"
            )
        if mode == "low_rank":
            raise NotImplementedError(
                "conditioning_mode='low_rank' is a reserved hook (docs/fff_film.md §5.4); "
                "add it only when an ablation shows FiLM has adequate features but cannot "
                "represent the required state-dependent combinations"
            )
        if depth < 1:
            raise ValueError(f"the trunk needs at least one layer, got depth={depth}")
        self.mode = mode
        self.d_c = int(d_c)
        self.hidden = int(hidden)

        first_in = int(in_dim) + (self.d_c if mode == "concatenate" else 0)
        dims = [first_in] + [self.hidden] * int(depth)
        self.layers = nn.ModuleList(
            FiLMLayer(dims[k], dims[k + 1]) for k in range(int(depth))
        )
        self.generators = (
            nn.ModuleList(
                FiLMGenerator(self.d_c, film_hidden, film_depth, self.hidden)
                for _ in range(int(depth))
            )
            if mode == "film"
            else None
        )

    @property
    def out_dim(self) -> int:
        return self.hidden

    def forward(self, x: torch.Tensor, c: torch.Tensor | None) -> torch.Tensor:
        if self.mode == "concatenate":
            if c is None:
                raise ValueError("conditioning_mode='concatenate' needs a conditioning vector")
            x = torch.cat((x, c), dim=-1)
        for k, layer in enumerate(self.layers):
            modulation = None
            if self.generators is not None:
                if c is None:
                    raise ValueError("conditioning_mode='film' needs a conditioning vector")
                modulation = self.generators[k](c)
            x = layer(x, modulation)
        return x

    def film_scales(self, c: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Per-layer ``(||dgamma||, ||beta||)`` row norms -- the §9.4 diagnostic. Empty
        unless ``mode == 'film'``."""
        if self.generators is None:
            return []
        out = []
        for gen in self.generators:
            dgamma, beta = gen(c)
            out.append((dgamma.norm(dim=-1), beta.norm(dim=-1)))
        return out
