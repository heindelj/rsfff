"""The conditioned parameter network: feature blocks in, one set of parameters out.

``docs/fff_film.md`` §5 in code. The structured blocks keep their identity through separate
initial linear maps into a common width::

    x_iso    = E_in(x_in)
    x_joined = E_in(x_in) + E_env(x_env) + E_cross(x_cross)

``E_env`` and ``E_cross`` are **bias-free and zero-initialized**, and their weights are
tagged with :func:`rsfff.mlip.heads.mark_env_slot` -- the environment sector of this model is
those two matrices, nameable by :func:`rsfff.mlip.heads.env_parameters` exactly as v4's
``w_env`` columns were. Bias-free is what makes the vertex exact: an isolated fragment has
``x_env == 0`` and ``x_cross == 0``, so ``x_joined == x_iso`` bitwise and every parameter's
env-dressed evaluation *is* its isolated one. Zero-init is what makes a fresh model
environment-free everywhere, so every fit starts from the validated isolated answer.

One shared trunk (:class:`ConditionedTrunk`) runs on both blocks; each parameter family then
owns a light adapter layer with its **own** FiLM generator (§5.2: separate modulators per
family, shared hidden trunk). All generators are zero-initialized, so the initial model is an
ordinary shared network.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ...mlip.heads import mark_env_slot
from .bonded import BondedParameterHead, BondedParameters, BondedTopology
from .conditioning import ConditionedTrunk, FiLMGenerator, FiLMLayer
from .heads import FilmResponseHeads, ResponseFamily
from .permanent import PermanentMultipoleHeads
from .projector import ProjectedFeatures
from .state import StateDescriptor

__all__ = ["ConditionedParameterNetwork", "FilmParameters", "FAMILIES"]

FAMILIES = ("bonded", "permanent", "response", "pauli", "disp")


def _log_shift(theta: torch.Tensor, theta0: torch.Tensor) -> torch.Tensor:
    return (theta.log() - theta0.log()).abs()


@dataclass
class FilmParameters:
    """Every generated parameter, at both evaluations where an environment may enter.

    ``*0`` fields are the isolated evaluations ``theta_0``; the unstarred ones are the
    env-dressed ``theta``. The permanent multipoles have no starred twin because they have no
    env-dressed evaluation at all -- that is the strict separation.
    """

    bonded: BondedParameters
    bonded0: BondedParameters
    q_perm: torch.Tensor
    mu_perm: torch.Tensor | None
    quad_perm: torch.Tensor | None
    response: ResponseFamily
    response0: ResponseFamily
    pauli: tuple      # (q, b, mu, quad_s)
    pauli0: tuple
    disp: tuple       # (c6, b)
    disp0: tuple
    gate: torch.Tensor            # (N,) the environment gate g(a_env)

    def env_shift(self) -> dict[str, torch.Tensor]:
        """Per-quantity ``|theta - theta_0|`` (log-space for positives): the L_env inputs."""
        out = {
            "bond_r_eq": _log_shift(self.bonded.r_eq, self.bonded0.r_eq),
            "bond_d": _log_shift(self.bonded.d, self.bonded0.d),
            "bond_k": _log_shift(self.bonded.k, self.bonded0.k),
            "angle_cos_eq": (self.bonded.cos_theta_eq - self.bonded0.cos_theta_eq).abs(),
            "angle_k": _log_shift(self.bonded.k_theta, self.bonded0.k_theta),
            "eta": _log_shift(self.response.eta, self.response0.eta),
            "compliance": _log_shift(
                self.response.compliance.clamp(min=1e-12),
                self.response0.compliance.clamp(min=1e-12),
            ),
            "pauli_q": _log_shift(self.pauli[0], self.pauli0[0]),
            "pauli_b": _log_shift(self.pauli[1], self.pauli0[1]),
            "c6": _log_shift(self.disp[0], self.disp0[0]),
            "b_disp": _log_shift(self.disp[1], self.disp0[1]),
        }
        if self.response.alpha is not None:
            out["alpha"] = (self.response.alpha - self.response0.alpha).flatten(1).norm(dim=-1)
        if self.pauli[2] is not None:
            out["pauli_mu"] = (self.pauli[2] - self.pauli0[2]).norm(dim=-1)
        if self.pauli[3] is not None:
            out["pauli_quad"] = (self.pauli[3] - self.pauli0[3]).norm(dim=-1)
        return out


class ConditionedParameterNetwork(nn.Module):
    """Block embedders + shared trunk + per-family adapters + the family heads."""

    def __init__(
        self,
        *,
        p_in: int,
        p_cross: int,
        d_c: int,
        bonded_head: BondedParameterHead,
        permanent_heads: PermanentMultipoleHeads,
        response_heads: FilmResponseHeads,
        pauli_heads: nn.Module,
        disp_heads: nn.Module,
        block_dim: int = 64,
        hidden: int = 128,
        depth: int = 2,
        conditioning_mode: str = "film",
        film_hidden: int = 32,
        film_depth: int = 1,
        gate_a0: float = 0.5,
    ) -> None:
        super().__init__()
        self.embed_in = nn.Linear(int(p_in), int(block_dim))
        self.embed_env = nn.Linear(int(p_in), int(block_dim), bias=False)
        self.embed_cross = nn.Linear(int(p_cross), int(block_dim), bias=False)
        with torch.no_grad():
            self.embed_env.weight.zero_()
            self.embed_cross.weight.zero_()
        mark_env_slot(self.embed_env.weight)
        mark_env_slot(self.embed_cross.weight)

        self.trunk = ConditionedTrunk(
            block_dim, hidden, depth,
            d_c=d_c, mode=conditioning_mode,
            film_hidden=film_hidden, film_depth=film_depth,
        )
        self.adapters = nn.ModuleDict(
            {name: FiLMLayer(hidden, hidden) for name in FAMILIES}
        )
        self.family_generators = (
            nn.ModuleDict(
                {name: FiLMGenerator(d_c, film_hidden, film_depth, hidden) for name in FAMILIES}
            )
            if conditioning_mode == "film"
            else None
        )
        self.gate_a0 = float(gate_a0)

        self.bonded_head = bonded_head
        self.permanent_heads = permanent_heads
        self.response_heads = response_heads
        self.pauli_heads = pauli_heads
        self.disp_heads = disp_heads

    @property
    def latent_dim(self) -> int:
        return self.trunk.out_dim

    def gate(self, a_env: torch.Tensor) -> torch.Tensor:
        """``g(a) = a^2 / (a^2 + a0^2)``: smooth, in [0, 1), exactly zero at ``a = 0``."""
        a2 = a_env * a_env
        return a2 / (a2 + self.gate_a0 ** 2)

    def _family_latents(
        self, x: torch.Tensor, c: torch.Tensor | None
    ) -> dict[str, torch.Tensor]:
        z = self.trunk(x, c)
        out = {}
        for name in FAMILIES:
            modulation = None
            if self.family_generators is not None:
                modulation = self.family_generators[name](c)
            out[name] = self.adapters[name](z, modulation)
        return out

    def forward(
        self,
        pf: ProjectedFeatures,
        c: torch.Tensor | None,
        state: StateDescriptor,
        topo: BondedTopology,
        positions: torch.Tensor,
        bond_index: torch.Tensor,
    ) -> FilmParameters:
        species_idx = pf.x_in.species_idx
        x_iso = self.embed_in(pf.x_in.inv_feats)
        if bool(pf.a_env.any()):
            x_joined = (
                x_iso + self.embed_env(pf.x_env.inv_feats) + self.embed_cross(pf.cross_inv)
            )
            z_iso = self._family_latents(x_iso, c)
            z_joined = self._family_latents(x_joined, c)
        else:
            # No environment anywhere in the batch: the joined block *is* the isolated one
            # (x_env and x_cross are exact zeros), so run the trunk once and alias.
            z_iso = self._family_latents(x_iso, c)
            z_joined = z_iso
        gate = self.gate(pf.a_env)

        bonded = self.bonded_head(
            z_iso["bonded"], z_joined["bonded"], gate, species_idx, topo
        )
        bonded0 = self.bonded_head(z_iso["bonded"], None, None, species_idx, topo)

        q_perm, mu_perm, quad_perm = self.permanent_heads(
            z_iso["permanent"], pf.x_in, state
        )

        response = self.response_heads(
            z_joined["response"], pf.x_in, species_idx, positions, bond_index
        )
        response0 = self.response_heads(
            z_iso["response"], pf.x_in, species_idx, positions, bond_index
        )

        pauli = self.pauli_heads(
            z_joined["pauli"], species_idx, pf.x_in.vec_feats, pf.x_in.equiv_feats
        )
        pauli0 = self.pauli_heads(
            z_iso["pauli"], species_idx, pf.x_in.vec_feats, pf.x_in.equiv_feats
        )
        disp = self.disp_heads(z_joined["disp"], species_idx)
        disp0 = self.disp_heads(z_iso["disp"], species_idx)

        return FilmParameters(
            bonded=bonded, bonded0=bonded0,
            q_perm=q_perm, mu_perm=mu_perm, quad_perm=quad_perm,
            response=response, response0=response0,
            pauli=pauli, pauli0=pauli0,
            disp=disp, disp0=disp0,
            gate=gate,
        )
