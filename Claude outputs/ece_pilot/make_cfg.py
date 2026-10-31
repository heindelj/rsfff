"""Pilot configs: latent-only vs ECE-fed bonded heads on the PySCF pilot (stretch scans held out)."""
import sys
from pathlib import Path

VARIANTS = {
    "latent": "",
    "ece": "  bond_features: ece\n  ece_latents: true\n",
    "ece_only": "  bond_features: ece\n  ece_latents: false\n",
}


def write(name, epochs=60, hessian=1.0, bond_var=1e-3, seed=0, tag="", big=False):
    root = Path(__file__).resolve().parent
    run = f"{name}{tag}"
    cfg = f"""run_name: {run}
device: cpu
dtype: float64
checkpoint_root: {root}/ckpt
data:
  path: [{root}/data/train.xyz]
  reference_energies: {root}/data/ref_b3lyp_def2svpd.json
  holdout_fraction: 0.1
  seed: 0
bonding:
  energy_weight: 1.0
  force_weight: 1.0
  hessian_weight: {hessian}
  hessian_probes: 2
  hessian_scale: 0.02
  bond_var_weight: {bond_var}
  dipole_weight: 0.0
  quadrupole_weight: 0.0
  polarizability_weight: 0.0
features: {{cutoff: 5.0, n_max: 4, l_max: 3, selected_lambdas: [0, 1, 2], backend: e3nn, density_channels: 8}}
film:
  nonbonded: exclusions
  exclude_through: 4
  impropers: true
  atom_typing: degree
  torsions: true
  couplings: true
  torsion_nmax: 3
  induction: false
{VARIANTS[name]}  ece_n_max: 4
  ece_l_max: 3
  ece_channels: 8
  ece_dim: 16
  ece_hidden: 32
  hidden: 48
  block_dim: 32
  depth: 2
  film_hidden: 16
  head_hidden: 24
  head_depth: 1
  equiv_channels: 8
  bonded_hidden: 32
  bonded_depth: 1
  term_hidden: 24
  term_depth: 1
train:
  epochs: {epochs}
  batch_size: 16
  learning_rate: 2.0e-3
  lr_schedule: cosine
  lr_final_factor: 0.02
  weight_decay: 0.0
  grad_clip: 5.0
  eval_every: 5
  seed: {seed}
"""
    if big:
        for a, b in (("bonded_hidden: 32", "bonded_hidden: 64"), ("term_hidden: 24", "term_hidden: 48"),
                     ("ece_dim: 16", "ece_dim: 32"), ("ece_hidden: 32", "ece_hidden: 64"),
                     ("bonded_depth: 1", "bonded_depth: 2")):
            cfg = cfg.replace(a, b)
    path = root / f"{run}.yaml"
    path.write_text(cfg)
    return path


if __name__ == "__main__":
    name = sys.argv[1]
    kw = dict(a.split("=") for a in sys.argv[2:])
    print(write(name, epochs=int(kw.get("epochs", 60)), hessian=float(kw.get("hessian", 1.0)),
                bond_var=float(kw.get("bond_var", 1e-3)), tag=kw.get("tag", ""),
                big=bool(int(kw.get("big", 0)))))
