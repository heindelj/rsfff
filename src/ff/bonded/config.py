"""What a bonded model is made of, as one config: which force-field terms exist and how big the network is.

    cfg = BondedConfig.load("configs/bonded_h2o_ions.toml")

The file has three model tables (every key optional; the defaults are the dataclass's). Any other table (``[data]``,
``[loss]``, ``[train]``) belongs to the trainer and is kept untouched in ``cfg.extra``.

    [terms]      which terms are present
      bonded = ["morse", "angle", "bond_bond", "bond_angle", "angle_angle", "opposite_bond_angle"]
               also: "improper", "torsion". A term with no instance in a molecule (an angle in OH-) is simply absent
               there. Forms: rsfff.ff.film.bonded / rsfff.ff.film.terms.
      charges = "sqe"            "sqe": the permanent charges are the zero-field solution of the split-charge model
                                 that also gives the charge-flow response (one eta, one compliance per bond);
                                 "flow": emitted bond flows, no charge-flow response; "none": Q / n on every atom
      dipoles, quadrupoles       permanent atomic dipoles / second moments
      polarizability = "anisotropic"     atomic dipole polarizabilities: "none" | "isotropic" | "anisotropic"
      quadrupole_polarizability  isotropic atomic quadrupole polarizabilities

    [tables]     the per-type constants the network modulates
      init = "params/bonded_monomers_wb97mv_tzvpd.json"     a constant-parameter fit ("" = built-in priors)
      learn = true               false: the tables stay at their initial values

    [encoder]    one latent per directed bond, decoded linearly into every parameter
      latent, hidden, depth      latent width; trunk width and hidden layers
      param_cap                  bound on |ln factor| of every modulated parameter (and on the additive shifts)
      r_cap                      the same bound for the bond lengths r_eq (tighter: e^0.2 is +-20 %)
      odd = "symmetrize"         how the pseudoscalar edge features enter: "symmetrize" | "square" | "drop"
      compress = "asinh"         elementwise compression of the edge invariants: "asinh" | "none"
      n_max, l_max, channels, species_dim, morse_a     bb.features.EdgeClusterExpansion
      conditioning = "film"      how the bond type and fragment state enter the trunk: "film" | "concatenate" | "none"
      condition_on = ["elements", "degrees", "charge", "multiplicity"]
      condition_dim              width of the conditioning vector

The model is built from it by :class:`rsfff.ff.bonded.BondedModel`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

__all__ = ["BONDED_TERMS", "BondedConfig"]

#: bonded term -> the terms whose coordinates and stiffnesses it is built from
BONDED_TERMS = {
    "morse": (),
    "angle": (),
    "improper": (),
    "torsion": (),
    "bond_bond": ("morse",),
    "bond_angle": ("morse", "angle"),
    "angle_angle": ("angle",),
    "opposite_bond_angle": ("morse", "angle"),
}
_CHARGES = ("sqe", "flow", "none")
_ALPHA = ("none", "isotropic", "anisotropic")
_CONDITIONING = ("film", "concatenate", "none")
_CONDITION_ON = ("elements", "degrees", "charge", "multiplicity")
_TABLES = {"terms": ("bonded", "charges", "dipoles", "quadrupoles", "polarizability", "quadrupole_polarizability"),
           "tables": ("init", "learn"),
           "encoder": ("latent", "hidden", "depth", "param_cap", "r_cap", "odd", "compress", "n_max", "l_max", "channels",
                       "species_dim", "morse_a", "conditioning", "condition_on", "condition_dim")}
_RENAME = {("tables", "init"): "table_init", ("tables", "learn"): "table_learn"}


@dataclass
class BondedConfig:
    # [terms]
    bonded: tuple = ("morse", "angle", "bond_bond", "bond_angle", "angle_angle", "opposite_bond_angle")
    charges: str = "sqe"
    dipoles: bool = True
    quadrupoles: bool = True
    polarizability: str = "anisotropic"
    quadrupole_polarizability: bool = True
    # [tables]
    table_init: str = ""
    table_learn: bool = True
    # [encoder]
    latent: int = 8
    hidden: int = 64
    depth: int = 2
    param_cap: float = 1.0
    r_cap: float = 0.2
    odd: str = "symmetrize"
    compress: str = "asinh"
    n_max: int = 6
    l_max: int = 3
    channels: int = 16
    species_dim: int = 4
    morse_a: float = 2.0
    conditioning: str = "film"
    condition_on: tuple = _CONDITION_ON
    condition_dim: int = 8
    #: the trainer's tables ([data], [loss], [train], ...), untouched
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        self.bonded, self.condition_on = tuple(self.bonded), tuple(self.condition_on)
        bad = [t for t in self.bonded if t not in BONDED_TERMS]
        if bad:
            raise ValueError(f"terms.bonded: unknown {bad}; choose from {sorted(BONDED_TERMS)}")
        if len(set(self.bonded)) != len(self.bonded):
            raise ValueError(f"terms.bonded lists a term twice: {self.bonded}")
        for t in self.bonded:
            missing = [d for d in BONDED_TERMS[t] if d not in self.bonded]
            if missing:
                raise ValueError(f"terms.bonded: {t!r} is built on {missing}, which are not listed")
        for name, value, allowed in (("terms.charges", self.charges, _CHARGES),
                                     ("terms.polarizability", self.polarizability, _ALPHA),
                                     ("encoder.conditioning", self.conditioning, _CONDITIONING),
                                     ("encoder.odd", self.odd, ("symmetrize", "square", "drop")),
                                     ("encoder.compress", self.compress, ("asinh", "none"))):
            if value not in allowed:
                raise ValueError(f"{name} = {value!r}: one of {allowed}")
        bad = [c for c in self.condition_on if c not in _CONDITION_ON]
        if bad:
            raise ValueError(f"encoder.condition_on: unknown {bad}; choose from {_CONDITION_ON}")
        for name in ("latent", "hidden", "depth", "n_max", "channels", "species_dim", "condition_dim"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"encoder.{name} must be >= 1")
        if self.l_max < 0 or self.param_cap <= 0 or self.r_cap <= 0 or self.morse_a <= 0:
            raise ValueError("encoder: l_max >= 0, param_cap > 0, r_cap > 0 and morse_a > 0 are required")
        if "torsion" in self.bonded and self.l_max < 1:
            raise ValueError("a torsion needs l_max >= 1 (the edge features' dihedral harmonics)")

    # ---- what the model is made of ----
    def has(self, term: str) -> bool:
        if term not in BONDED_TERMS:
            raise KeyError(f"{term!r} is not a bonded term ({sorted(BONDED_TERMS)})")
        return term in self.bonded

    @property
    def charge_flow_response(self) -> bool:
        """Charge moves along bonds in a field: only the split-charge model has it."""
        return self.charges == "sqe"

    @property
    def responds(self) -> bool:
        return self.charge_flow_response or self.polarizability != "none" or self.quadrupole_polarizability

    # ---- files ----
    @classmethod
    def from_dict(cls, d: dict) -> "BondedConfig":
        """From the nested tables of a config file. Unknown keys of a model table are an error."""
        kw, extra = {}, {}
        for table, body in (d or {}).items():
            if table not in _TABLES:
                extra[table] = body
                continue
            unknown = [k for k in body if k not in _TABLES[table]]
            if unknown:
                raise ValueError(f"[{table}]: unknown keys {unknown}; known: {list(_TABLES[table])}")
            for k, v in body.items():
                kw[_RENAME.get((table, k), k)] = v
        return cls(**kw, extra=extra)

    @classmethod
    def load(cls, path: str | Path) -> "BondedConfig":
        import tomllib
        return cls.from_dict(tomllib.loads(Path(path).read_text()))

    def to_dict(self) -> dict:
        """The nested tables (what :meth:`from_dict` reads; what a checkpoint stores)."""
        flat = asdict(self)
        out = {t: {k: flat[_RENAME.get((t, k), k)] for k in keys} for t, keys in _TABLES.items()}
        out["terms"]["bonded"], out["encoder"]["condition_on"] = list(self.bonded), list(self.condition_on)
        return {**out, **self.extra}
