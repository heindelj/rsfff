"""Checkpoints of the complete model (train_ionct ``--model complete``, tag complete-explore-final) as cmm models.

Only the configuration the complete.* runs converged on converts: frozen shells Pauli with the tied dispersion range,
pinned a0, no anisotropic alpha, Gram exchange polarization with the learned W1 / W2, Legendre hyperpolarization with
the field-gradient term, flux CT (pinned exponents: ``all`` or ``acceptor-float``), one shared latent (two-body
channels and the gated probes in the trunk, no skip). Anything else raises: reproduce those runs from the tag.

    model = load_legacy("run/model.pt", tab, chan)        # or: config, state = convert_state(ck["state"], ck["args"])
"""
from __future__ import annotations

import math

import torch

from .config import CmmConfig
from .encoder import READOUTS

__all__ = ["convert_args", "convert_state", "load_legacy"]

#: legacy readout names, in the cmm readout order
_LEGACY_READOUTS = ("pauli", "disp", "kappa_dip", "kappa_quad", "kappa_breath", "gram", "qD", "qA")

#: argument -> required legacy value (missing arguments take train_ionct's build_model defaults)
_REQUIRED = {
    "model": ("complete", "complete"), "frozen": (False, True), "xpol": (False, True), "pseudo": (False, False),
    "ct_model": ("direct", "flux"), "flux_solver": ("atomic", "atomic"), "pauli_form": ("slater", "shells"),
    "no_aniso": (False, True), "pin_a0": (False, True), "tie_disp_range": (False, True),
    "xpol_form": ("lam", "gram"), "xpol_weights": (False, True), "xpol_w1": ("rank", "rank"),
    "hyperpol": (False, True), "hyperpol_form": ("oneshot", "legendre"), "hyperpol_b": (False, True),
    "env_channels": (False, True), "env_mode": ("shared", "shared"), "env_two_latent": (False, True),
    "pauli_moments": (False, False), "pauli_ii": (False, False), "ct_legacy_head": (False, False),
    "ct_q_const": (False, False), "xpol_kappa_const": (False, False), "xpol_kappa_fixed": (None, None),
    "env_field": (False, False), "env_ct_pen": (False, False), "breath_form": ("shape", "shape"),
    "xpol_dipole": ("parent", "parent"), "xpol_quad": ("parent", "parent"), "env_log": (0.0, 0.0),
}


def convert_args(a: dict) -> CmmConfig:
    """The :class:`CmmConfig` of a legacy run's arguments (train_ionct's, as saved in model.pt); raises on a run
    outside the converged configuration."""
    bad = []
    for k, (default, want) in _REQUIRED.items():
        v = a.get(k, default)
        if v != want:
            bad.append(f"{k}={v!r} (needs {want!r})")
    if a.get("ct_b_pin", "none") not in ("all", "acceptor-float"):
        bad.append(f"ct_b_pin={a.get('ct_b_pin')!r} (needs 'all' or 'acceptor-float')")
    skip = a["env_skip"] if a.get("env_skip") is not None else not a.get("env_no_skip", False)
    if skip:
        bad.append("env_skip (the per-readout linear skip)")
    probes = bool(a.get("env_ct_gated", False))
    if probes and not a.get("env_extra_latent", False):
        bad.append("env_ct_gated without env_extra_latent (the probes as a side branch)")
    if tuple(a.get("env_extra", ()) or ()):
        bad.append("env_extra")
    if a.get("env_species", "none") not in ("none", "film") or (a.get("env_species") == "film"
                                                               and a.get("env_species_emb", "descriptor") != "descriptor"):
        bad.append(f"env_species={a.get('env_species')!r} / {a.get('env_species_emb')!r} (needs none or film + descriptor)")
    if bad:
        raise ValueError("not a converged complete-model run; reproduce it from tag complete-explore-final: "
                         + ", ".join(bad))
    film = a.get("env_species", "none") == "film"
    return CmmConfig(hidden=int(a.get("hidden", 64)), depth=2, latent=int(a.get("env_latent", 4)),
                     pairs=tuple(a.get("env_pairs") or ("ww", "ws", "ss")), probes=probes, film=film,
                     species_dim=int(a.get("env_species_dim", 8)), env_cap=float(a.get("env_cap", 0.0)),
                     ct_q_cap=float(a.get("ct_q_cap", 0.0)) or float(a.get("env_cap", 0.0)),
                     env_gate=tuple(a.get("env_gate") or (1e-7, 1e-5)))


def convert_state(state: dict, args: dict):
    """(CmmConfig, cmm state dict) of a legacy checkpoint's state and arguments."""
    cfg = convert_args(args)
    if cfg.env_cap <= 0:
        raise ValueError("legacy run without env_cap: the cmm factors are always bounded")
    lna = state.get("breath_lna")
    if lna is not None and float(lna.abs().max()) > 0:
        raise ValueError("breath_lna is not zero: the breathing stiffness was not pinned")
    out = {
        "pauli_ln_w": state["pauli_lnw"], "pauli_ln_bscale": state["pauli_lnbs"], "disp_ln_c6": state["disp_lnc60"],
        "ln_w1": state["xpol_lnk"] + math.log(2.0), "ln_w2": state["xpol_lng2"], "ct_ln_q0": state["ct_lnq0"],
        "ct_ln_v0": state["flux_lnv"], "ct_eta_raw": state["flux_eta"],
        "ct_acc_raw": state.get("ct_fA_raw", torch.zeros_like(state["flux_lnv"])),
        "chan_scale": state["chan_scale"], "chan_sat": state["chan_sat"],
    }
    if cfg.probes:
        out["gate_scale"], out["gate_sat"] = state["gate_scale"], state["gate_sat"]
    else:
        out["gate_scale"] = out["gate_sat"] = state["chan_scale"].new_zeros(0)
    pre = "env.trunks.shared."
    for k, v in state.items():
        if k.startswith(pre):
            rest = k[len(pre):]
            out[("encoder.trunk." if cfg.film else "encoder.trunk.net.") + rest] = v
        elif k.startswith("species_emb."):
            out["species." + k[len("species_emb."):]] = v
    out["encoder.readout.weight"] = torch.cat([state[f"env.W.{p}.weight"] for p in _LEGACY_READOUTS], 0)
    assert len(_LEGACY_READOUTS) == len(READOUTS)
    return cfg, out


def load_legacy(path, tab, chan, map_location="cpu"):
    """A :class:`CmmModel` from a legacy model.pt (or a dict with ``state`` and ``args``)."""
    from .model import CmmModel
    ck = path if isinstance(path, dict) else torch.load(path, map_location=map_location, weights_only=False)
    cfg, state = convert_state(ck["state"], ck["args"])
    m = CmmModel(tab, chan, cfg)
    m.load_state_dict(state)
    return m
