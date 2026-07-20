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
    holdout_fraction: float = 0.1
    seed: int = 0


@dataclass
class ChargeConfig:
    """Charge-aware model: embedding, density weights, heads, and the SCF."""

    emb_dim: int = 16               # charge-aware embedding width
    q_hidden: int = 16              # hidden width of the q FiLM encoder
    weight_hidden: int = 32         # hidden width of the density-weight net
    charge_channels: int = 4        # Kw: density channels from w(z_j)
    hidden: int = 64                # energy head MLP
    depth: int = 2
    alpha_hidden: int = 64          # response heads (alpha / atomic dipole)
    alpha_depth: int = 2
    alpha_equiv_channels: int = 32
    alpha_positive_isotropic: bool = True
    dipole_head: bool = False       # atomic dipole head (needs lambda=1 features)
    eta_init: float = 0.5           # initial per-element hardness, Ha/e^2
    scf_max_iter: int = 30
    scf_tol: float = 1.0e-9
    scf_damping: float = 0.0
    attach_steps: int = 1           # 2 for exact 2nd-order training grads (alpha, dmu/dR)
    hessian_in_graph: bool = True
    freeze_charges: bool = False    # ablation: pin q at Q/n, skip the SCF


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
    alpha_consistency_weight: float = 0.0
    iso_weight: float = 0.0         # isolated-species integer-charge anchors
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
    charge: ChargeConfig = field(default_factory=ChargeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def load_config(path) -> Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}

    feat = raw.get("features", {}) or {}
    mlip = raw.get("mlip", {}) or {}
    charge = raw.get("charge", {}) or {}
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
        holdout_fraction=float(data.get("holdout_fraction", DataConfig.holdout_fraction)),
        seed=int(data.get("seed", DataConfig.seed)),
    )
    charge_cfg = ChargeConfig(
        emb_dim=int(charge.get("emb_dim", ChargeConfig.emb_dim)),
        q_hidden=int(charge.get("q_hidden", ChargeConfig.q_hidden)),
        weight_hidden=int(charge.get("weight_hidden", ChargeConfig.weight_hidden)),
        charge_channels=int(charge.get("charge_channels", ChargeConfig.charge_channels)),
        hidden=int(charge.get("hidden", ChargeConfig.hidden)),
        depth=int(charge.get("depth", ChargeConfig.depth)),
        alpha_hidden=int(charge.get("alpha_hidden", ChargeConfig.alpha_hidden)),
        alpha_depth=int(charge.get("alpha_depth", ChargeConfig.alpha_depth)),
        alpha_equiv_channels=int(
            charge.get("alpha_equiv_channels", ChargeConfig.alpha_equiv_channels)
        ),
        alpha_positive_isotropic=bool(
            charge.get("alpha_positive_isotropic", ChargeConfig.alpha_positive_isotropic)
        ),
        dipole_head=bool(charge.get("dipole_head", ChargeConfig.dipole_head)),
        eta_init=float(charge.get("eta_init", ChargeConfig.eta_init)),
        scf_max_iter=int(charge.get("scf_max_iter", ChargeConfig.scf_max_iter)),
        scf_tol=float(charge.get("scf_tol", ChargeConfig.scf_tol)),
        scf_damping=float(charge.get("scf_damping", ChargeConfig.scf_damping)),
        attach_steps=int(charge.get("attach_steps", ChargeConfig.attach_steps)),
        hessian_in_graph=bool(charge.get("hessian_in_graph", ChargeConfig.hessian_in_graph)),
        freeze_charges=bool(charge.get("freeze_charges", ChargeConfig.freeze_charges)),
    )
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
        alpha_consistency_weight=float(
            train.get("alpha_consistency_weight", TrainConfig.alpha_consistency_weight)
        ),
        iso_weight=float(train.get("iso_weight", TrainConfig.iso_weight)),
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
        charge=charge_cfg,
        data=data_cfg,
        train=train_cfg,
    )
