"""pyscf-all: energy + forces + Hessians + multipoles + response, stretch scans held out."""
import sys
from pathlib import Path

VARIANTS = {"latent": "", "ece": "  bond_features: ece\n  ece_latents: true\n"}


def write(name, epochs=100, tag="", fields=True, hessian=1.0, mult=1.0, probe=1.0):
    root = Path(__file__).resolve().parent
    run = f"{name}{tag}"
    cfg = f"""run_name: {run}
device: cpu
dtype: float64
checkpoint_root: {root}/ckpt
data:
  path: [{root}/data/train.xyz]
  reference_energies: {root}/data/ref.json
  holdout_fraction: 0.1
  seed: 0
bonding:
  streams:
    - {{name: probe, path: {root}/data/probe.xyz, weight: {probe}, batch_size: 24}}
  energy_weight: 1.0
  force_weight: 1.0
  hessian_weight: {hessian}
  hessian_probes: 1
  hessian_scale: 0.02
  bond_var_weight: 1.0e-3
  dipole_weight: {mult}
  quadrupole_weight: {mult}
  polarizability_weight: {mult}
features: {{cutoff: 5.0, n_max: 4, l_max: 3, selected_lambdas: [0, 1, 2], backend: e3nn, density_channels: 8}}
film:
  nonbonded: exclusions
  exclude_through: 4
  impropers: true
  atom_typing: degree
  torsions: true
  couplings: true
  torsion_nmax: 3
  field_features: {str(fields).lower()}
  induction: true
{VARIANTS[name]}  ece_n_max: 4
  ece_l_max: 3
  ece_channels: 8
  ece_dim: 32
  ece_hidden: 64
  hidden: 64
  block_dim: 32
  depth: 2
  film_hidden: 16
  max_rank: 2
  emb_dim: 16
  head_hidden: 32
  head_depth: 1
  equiv_channels: 16
  bonded_hidden: 64
  bonded_depth: 2
  term_hidden: 48
  term_depth: 1
train:
  epochs: {epochs}
  batch_size: 24
  learning_rate: 2.0e-3
  lr_schedule: cosine
  lr_final_factor: 0.02
  weight_decay: 0.0
  grad_clip: 5.0
  eval_every: 5
  seed: 0
"""
    path = root / f"{run}.yaml"
    path.write_text(cfg)
    return path


if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[2:])
    print(write(sys.argv[1], epochs=int(kw.get("epochs", 100)), tag=kw.get("tag", ""),
                fields=bool(int(kw.get("fields", 1)))))
