"""The architecture choices of a cmm model, the only ones it has.

Everything else that the complete model's exploration made optional is fixed in :class:`rsfff.ff.cmm.CmmModel`:
frozen Pauli (permanent shells, one environment factor per atom) and Tang-Toennies C6 dispersion with the Pauli range;
induction with the free-ion dipole, quadrupole and breathing (pinned a0) responses and the Legendre hyperpolarization
(gamma and the field-gradient b); Gram exchange polarization with the learned drive and on-site weights W1, W2;
charge-flux CT with the pinned exponents (cation acceptors floating within e^+-0.5 of Fukui), solved jointly with the
induction, with three optional stabilizers (all off by default): CT charges that the environment can only lower
(``ct_q_onesided``), a stiffness floor on every channel (``ct_floor``) and a hardness on each site's throughput
(``ct_relay``). ``ct_model = "gap"`` replaces the flux by two-state channels with an explicit gap (the model module's
docstring); the three stabilizers belong to the flux and are not used there (its charges are always one-sided); the encoder: one latent from the overlap-weighted channels and the gated diffuse probes, decoded linearly
into every environment factor.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

__all__ = ["CmmConfig", "PAIR_GROUPS"]

#: channel groups the encoder can read: the two-body sums S^w (plain overlap weights) and S^s (signed by the pair's
#: frozen electrostatics), and the three-body families T^ww, T^ws, T^ss (L = 0..4). Family tokens are unions.
PAIR_GROUPS = {"ww": ("S^w", "ww"), "ws": ("S^s", "ws"), "ss": ("S^s", "ss"), "Sw": ("S^w",), "Ss": ("S^s",),
               "Tww": ("ww",), "Tws": ("ws",), "Tss": ("ss",)}


@dataclass
class CmmConfig:
    hidden: int = 64                  # encoder trunk width
    depth: int = 2                    # encoder trunk hidden layers
    latent: int = 6                   # width of the shared latent
    pairs: tuple = ("ww",)            # channel groups read (PAIR_GROUPS)
    probes: bool = True               # the gated diffuse-probe invariants (donor, acceptor, breath) as encoder inputs
    film: bool = True                 # FiLM conditioning of the trunk on the species embedding
    species_dim: int = 8              # species embedding width
    species_hidden: int = 16          # species embedding MLP width (descriptor -> embedding)
    env_cap: float = 3.0              # |ln factor| bound of the Pauli, C6, W1, W2 environment factors
    ct_q_cap: float = 1.5             # |ln factor| bound of the CT charges' environment factors
    ct_model: str = "flux"            # "flux": the pair-calibrated charge flux; "gap": two-state channels with an explicit gap
    ct_gap_overlap: str = "pauli"     # gap model, the acceptor level's overlap shift: "pauli" (c x the acceptor probe's
                                      # overlap with each neighbour x that pair's Pauli energy per unit density
                                      # overlap) or "probe" (kappa x the acceptor probe's total overlap)
    ct_gap_w0: float = 0.005          # gap model, the least width W0 (Hartree) of a channel's crossing: the two-state form
                                      # with W_b^2 = W^2 + W0^2 and amplitude W^2 / W_b^2 (same W^2 / Delta at a large
                                      # gap; response <= W^2 / (4 W_b^3), transfer <= W^2 / W_b^2). 0: the exact form
    ct_q_onesided: bool = False       # the CT charges' environment factors can only lower them: ln factor <= 0
    ct_floor: float = 0.0             # channel stiffness floor b_min (Hartree): b_c = V0_c^2 / (2|e_c|) + b_min
    ct_relay: float = 0.0             # throughput hardness as a fraction kappa of the slot hardness: tau_i = kappa eta_i
    env_gate: tuple = (1e-7, 1e-5)    # probe overlap at which a neighbour's point-charge field starts / fully counts
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        self.pairs = tuple(self.pairs)
        self.env_gate = tuple(float(x) for x in self.env_gate)
        bad = set(self.pairs) - set(PAIR_GROUPS)
        if bad or not self.pairs:
            raise ValueError(f"pairs {self.pairs!r}: a non-empty subset of {sorted(PAIR_GROUPS)}")
        if not 0 < self.env_gate[0] < self.env_gate[1]:
            raise ValueError(f"env_gate {self.env_gate}: needs 0 < on < full")
        if self.ct_model not in ("flux", "gap"):
            raise ValueError(f"ct_model {self.ct_model!r}: 'flux' or 'gap'")
        if self.ct_gap_overlap not in ("probe", "pauli"):
            raise ValueError(f"ct_gap_overlap {self.ct_gap_overlap!r}: 'probe' or 'pauli'")
        self.ct_floor, self.ct_relay, self.ct_gap_w0 = float(self.ct_floor), float(self.ct_relay), float(self.ct_gap_w0)
        if self.ct_floor < 0 or self.ct_relay < 0 or self.ct_gap_w0 < 0:
            raise ValueError(f"ct_floor {self.ct_floor}, ct_relay {self.ct_relay} and ct_gap_w0 {self.ct_gap_w0} must be >= 0")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pairs"], d["env_gate"] = list(self.pairs), list(self.env_gate)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CmmConfig":
        known = {k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__}
        return cls(**known)
