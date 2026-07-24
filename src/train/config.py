"""YAML config -> typed dataclasses for the MLIP training pipeline.

Top-level YAML blocks: ``features:`` (SOAP featurizer), ``mlip:`` (MLP head),
``data:`` (dataset + split), ``train:`` (optimizer / loss weights). Parsing mirrors the
``load_config`` pattern in the reference repo: ``yaml.safe_load`` then nested
``.get(key, default)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import yaml


@dataclass
class FeaturesConfig:
    cutoff: float = 5.0
    n_max: int = 6
    l_max: int = 4
    selected_lambdas: Sequence[int] = (0, 2)   # featurizer requires 2; head uses only inv
    backend: str = "e3nn"
    density_channels: int | None = 8


@dataclass
class MLIPConfig:
    emb_dim: int = 16
    hidden: int = 64
    depth: int = 2


@dataclass
class DataConfig:
    path: str | list[str] = "data/labels/h2o.extxyz"
    reference_energies: str = "data/atomic_references.json"
    isolated_species: str | None = None   # extxyz of integer-charge anchor systems
    diabatic_states: str | None = None    # YAML fragment-state library (channel graphs)
    atomic_reference_states: str | None = None  # isolated-atom E/alpha grid at integer charge
    holdout_fraction: float = 0.1
    seed: int = 0


@dataclass
class EEMConfig:
    """EEM parameter-function heads (chi, eta, chivec, alpha) on the features."""

    emb_dim: int = 16               # species embedding width for the parameter heads
    hidden: int = 64                # parameter-head MLP width
    depth: int = 2
    equiv_channels: int = 32        # channel reduction of the chivec / alpha heads
    eta_init: float = 0.5           # initial per-element hardness, Ha/e^2
    eta_floor: float = 0.05         # hard lower bound on eta (keeps charges bounded)
    psd_floor: float = 1.0e-4       # minimum eigenvalue of the atomic alphas


@dataclass
class MonomerConfig:
    """Phase-1 monomer stack: reference embedding, state-decorated density, 1-body heads."""

    emb_dim: int = 32               # reference-embedding width (also the head conditioning slot)
    weight_channels: int = 8        # Kw learned density channels; feature width ~ (Kw*n_max)^2
    hidden: int = 64                # head MLP width
    depth: int = 2
    equiv_channels: int = 32        # channel reduction of the chivec / alpha heads


@dataclass
class SQEConfig:
    """Split-charge equilibration: channel compliance head and on-site hardness."""

    s_init: float = 0.5             # initial channel compliance (e^2/Ha)
    s_floor: float = 0.0            # lower bound on compliance; 0 keeps closure exact
    n_radial: int = 8               # Bessel functions in the pair compliance head
    eta_init: float = 0.5           # fallback per-element hardness when no IP/EA is available
    eta_floor: float = 0.05         # keeps the charge problem strongly convex
    psd_floor: float = 1.0e-4       # minimum eigenvalue of the atomic alphas


@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 32
    learning_rate: float = 1.0e-3
    weight_decay: float = 0.0
    grad_clip: float | None = 5.0
    energy_weight: float = 1.0
    force_weight: float = 1.0
    dipole_weight: float = 0.0
    dmu_dr_weight: float = 0.0
    dmu_dr_every: int = 1           # apply the (expensive) dmu/dR term every k steps
    alpha_weight: float = 0.0
    iso_weight: float = 0.0         # isolated-species integer-charge anchors
    atomic_ref_weight: float = 0.0  # isolated-atom energy anchors at integer charge
    free_alpha_weight: float = 0.0  # isolated-atom polarizability anchors
    unbound_weight: float = 0.0     # relative weight of anion states that are unbound at this
                                    # level of theory (H-, O2-); 0 drops them from the anchors
    q_l2_weight: float = 0.0
    eval_every: int = 10


@dataclass
class Config:
    run_name: str = "run"
    device: str = "auto"       # auto -> cuda > mps > cpu
    dtype: str = "float32"     # float32 | float64
    checkpoint_root: str = "checkpoints"
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    mlip: MLIPConfig = field(default_factory=MLIPConfig)
    eem: EEMConfig = field(default_factory=EEMConfig)
    monomer: MonomerConfig = field(default_factory=MonomerConfig)
    sqe: SQEConfig = field(default_factory=SQEConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _from_block(cls, block: dict):
    """Build a config dataclass from a YAML block, keeping each field's declared type."""
    kwargs = {}
    for name, f in cls.__dataclass_fields__.items():
        if name not in block:
            continue
        value = block[name]
        kwargs[name] = f.type(value) if f.type in (int, float, str, bool) else value
    return cls(**kwargs)


def load_config(path) -> Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}

    feat = raw.get("features", {}) or {}
    mlip = raw.get("mlip", {}) or {}
    eem = raw.get("eem", {}) or {}
    data = raw.get("data", {}) or {}
    train = raw.get("train", {}) or {}

    features_cfg = FeaturesConfig(
        cutoff=float(feat.get("cutoff", FeaturesConfig.cutoff)),
        n_max=int(feat.get("n_max", FeaturesConfig.n_max)),
        l_max=int(feat.get("l_max", FeaturesConfig.l_max)),
        selected_lambdas=tuple(feat.get("selected_lambdas", (0, 2))),
        backend=str(feat.get("backend", FeaturesConfig.backend)),
        density_channels=feat.get("density_channels", FeaturesConfig.density_channels),
    )
    mlip_cfg = MLIPConfig(
        emb_dim=int(mlip.get("emb_dim", MLIPConfig.emb_dim)),
        hidden=int(mlip.get("hidden", MLIPConfig.hidden)),
        depth=int(mlip.get("depth", MLIPConfig.depth)),
    )
    raw_path = data.get("path", DataConfig.path)
    data_cfg = DataConfig(
        path=[str(p) for p in raw_path] if isinstance(raw_path, list) else str(raw_path),
        reference_energies=str(
            data.get("reference_energies", DataConfig.reference_energies)
        ),
        isolated_species=(
            str(data["isolated_species"]) if data.get("isolated_species") else None
        ),
        diabatic_states=(
            str(data["diabatic_states"]) if data.get("diabatic_states") else None
        ),
        atomic_reference_states=(
            str(data["atomic_reference_states"])
            if data.get("atomic_reference_states") else None
        ),
        holdout_fraction=float(data.get("holdout_fraction", DataConfig.holdout_fraction)),
        seed=int(data.get("seed", DataConfig.seed)),
    )
    eem_cfg = EEMConfig(
        emb_dim=int(eem.get("emb_dim", EEMConfig.emb_dim)),
        hidden=int(eem.get("hidden", EEMConfig.hidden)),
        depth=int(eem.get("depth", EEMConfig.depth)),
        equiv_channels=int(eem.get("equiv_channels", EEMConfig.equiv_channels)),
        eta_init=float(eem.get("eta_init", EEMConfig.eta_init)),
        eta_floor=float(eem.get("eta_floor", EEMConfig.eta_floor)),
        psd_floor=float(eem.get("psd_floor", EEMConfig.psd_floor)),
    )
    monomer_cfg = _from_block(MonomerConfig, raw.get("monomer", {}) or {})
    sqe_cfg = _from_block(SQEConfig, raw.get("sqe", {}) or {})
    train_cfg = TrainConfig(
        epochs=int(train.get("epochs", TrainConfig.epochs)),
        batch_size=int(train.get("batch_size", TrainConfig.batch_size)),
        learning_rate=float(train.get("learning_rate", TrainConfig.learning_rate)),
        weight_decay=float(train.get("weight_decay", TrainConfig.weight_decay)),
        grad_clip=train.get("grad_clip", TrainConfig.grad_clip),
        energy_weight=float(train.get("energy_weight", TrainConfig.energy_weight)),
        force_weight=float(train.get("force_weight", TrainConfig.force_weight)),
        dipole_weight=float(train.get("dipole_weight", TrainConfig.dipole_weight)),
        dmu_dr_weight=float(train.get("dmu_dr_weight", TrainConfig.dmu_dr_weight)),
        dmu_dr_every=int(train.get("dmu_dr_every", TrainConfig.dmu_dr_every)),
        alpha_weight=float(train.get("alpha_weight", TrainConfig.alpha_weight)),
        iso_weight=float(train.get("iso_weight", TrainConfig.iso_weight)),
        atomic_ref_weight=float(
            train.get("atomic_ref_weight", TrainConfig.atomic_ref_weight)
        ),
        free_alpha_weight=float(
            train.get("free_alpha_weight", TrainConfig.free_alpha_weight)
        ),
        unbound_weight=float(train.get("unbound_weight", TrainConfig.unbound_weight)),
        q_l2_weight=float(train.get("q_l2_weight", TrainConfig.q_l2_weight)),
        eval_every=int(train.get("eval_every", TrainConfig.eval_every)),
    )
    return Config(
        run_name=str(raw.get("run_name", "run")),
        device=str(raw.get("device", "auto")),
        dtype=str(raw.get("dtype", "float32")),
        checkpoint_root=str(raw.get("checkpoint_root", "checkpoints")),
        features=features_cfg,
        mlip=mlip_cfg,
        eem=eem_cfg,
        monomer=monomer_cfg,
        sqe=sqe_cfg,
        data=data_cfg,
        train=train_cfg,
    )
