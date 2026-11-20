"""The fixed physical parameters of the monatomic ions (``scripts/ion_params.py`` -> ``params/*.json``) as tensors.

Every per-ion quantity is indexed by a species index ``s`` (the order of ``IonTable.symbols``):

    Z (S,)                      point nuclear charge
    shell_n, shell_b (S, M)     electrons and Slater 1s exponent of each permanent shell (zero-padded, n = 0)
    alpha, C (S,)               free-ion dipole and quadrupole polarizabilities (Theta = C g, pol_fit convention)
    b_dip, w_dip (S, T), (S, T, 4)   induced-dipole parent: a multi-term shape, sum_t sum_k w[t, k] N_k r^k e^{-b[t] r}
                                (one term per permanent shell when fitted by scripts/ion_parents.py; the legacy
                                single-exponent parent b_ind / w_dip is T = 1); padded terms have w = 0
    b_quad, w_quad              induced-quadrupole parent, same layout
    b_breath, w_breath, a0      l = 0 (breathing) response shape and its polarizability (a0 = 0: no such term)
    hyper_gamma, hyper_b (S,)   free-ion dipole hyperpolarizabilities (ions[s]['hyperpol'] = {gamma, b}; 0 if absent):
                                alpha_zz = alpha0 + gamma/2 E^2 + b G_zz, alpha_xx = alpha0 + gamma/6 E^2 - b/2 G_zz
                                (G_zz = 2q/R^3 for a charge q at R on z; Buckingham B = -2b)
    front_b[role], front_w[role] (S,), (S, 4)   frontier densities, role in ('donor', 'acceptor')
    I_donor (S,)                electron-removal energy of the free ion as a CT donor (Hartree)
    A_acceptor (S,)             electron-attachment energy as an acceptor (Hartree; NaN where unbound, e.g. Cl-)
    pauli_w, pauli_s (S, M), (S,)   the frozen fit's Pauli charge per permanent shell and exponent scale (the shells
                                form of the CP Pauli kernel; ions[s]['pauli'] = {w, s}); NaN where the ion has none
    pauli_site_a, pauli_site_s (S, M), (S,)   the site-potential form of the Pauli term (config.pauli_form = "site";
                                ions[s]['pauli_site'] = {a, s}): amplitude per permanent shell and exponent scale of
                                v_s(r) = sum_t a[t] e^{-beta_t r} (1 / r + beta_t / 2), beta_t = s shell_b[t]; NaN if absent
    disp_b (S,)                 starting value of the dispersion damping exponent (ions[s]['disp_range'], bohr^-1) for the
                                site form, whose potential scale is not a density range; NaN if absent

Units: atomic.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

ATOMIC_NUMBER = {"Li": 3, "F": 9, "Na": 11, "Cl": 17, "K": 19, "Br": 35, "Rb": 37, "I": 53, "Cs": 55}
N_SHAPE = 4                     # r^0..r^3, what rsfff.ff.multipole's shaped kernels carry
ROLES = ("donor", "acceptor")
DEFAULT_PARAMS = Path(__file__).resolve().parents[3] / "params" / "ions_wb97mv_tzvpd.json"


def _onehot(k: int) -> list[float]:
    w = [0.0] * N_SHAPE
    w[int(k)] = 1.0
    return w


def _pad(w):
    w = list(w)[:N_SHAPE]
    return w + [0.0] * (N_SHAPE - len(w))


@dataclass
class IonTable:
    symbols: tuple
    numbers: torch.Tensor
    charge: torch.Tensor
    Z: torch.Tensor
    shell_n: torch.Tensor
    shell_b: torch.Tensor
    alpha: torch.Tensor
    C: torch.Tensor
    b_dip: torch.Tensor
    w_dip: torch.Tensor
    b_quad: torch.Tensor
    w_quad: torch.Tensor
    b_breath: torch.Tensor
    w_breath: torch.Tensor
    a0: torch.Tensor
    front_b: dict
    front_w: dict
    I_donor: torch.Tensor
    A_acceptor: torch.Tensor
    pauli_w: torch.Tensor | None = None
    pauli_s: torch.Tensor | None = None
    pauli_site_a: torch.Tensor | None = None
    pauli_site_s: torch.Tensor | None = None
    disp_b: torch.Tensor | None = None
    hyper_gamma: torch.Tensor | None = None
    hyper_b: torch.Tensor | None = None

    @classmethod
    def load(cls, path: str | Path = DEFAULT_PARAMS, symbols=None, dtype=torch.float64) -> "IonTable":
        ions = json.load(open(path))["ions"]
        symbols = tuple(symbols or ions)
        M = max(len(ions[s]["shells"]) for s in symbols)
        t = lambda x: torch.tensor(x, dtype=dtype)
        sh_n = [[sh["n"] for sh in ions[s]["shells"]] + [0.0] * (M - len(ions[s]["shells"])) for s in symbols]
        sh_b = [[sh["b"] for sh in ions[s]["shells"]] + [1.0] * (M - len(ions[s]["shells"])) for s in symbols]
        def parent(key, legacy_b, legacy_w):
            """[(b, w)] terms per species: ions[s]['parents'][key] (a list of {b, w}) or the legacy single term."""
            terms = []
            for s in symbols:
                pr = (ions[s].get("parents") or {}).get(key)
                if pr:
                    terms.append([(float(q["b"]), _pad(q["w"])) for q in pr])
                elif legacy_b in ions[s]:
                    terms.append([(float(ions[s][legacy_b]), _pad(ions[s][legacy_w]))])
                else:
                    terms.append([(1.0, [0.0] * N_SHAPE)])
            T = max(len(x) for x in terms)
            b = [[q[0] for q in x] + [1.0] * (T - len(x)) for x in terms]
            w = [[q[1] for q in x] + [[0.0] * N_SHAPE] * (T - len(x)) for x in terms]
            return t(b), t(w)

        b_dip, w_dip = parent("dip", "b_ind", "w_dip")
        b_quad, w_quad = parent("quad", "b_quad", "w_quad")
        b_breath, w_breath = parent("breath", "-", "-")
        a0 = t([float(((ions[s].get("parents") or {}).get("a0")) or 0.0) for s in symbols])
        hyper_gamma = t([float((ions[s].get("hyperpol") or {}).get("gamma") or 0.0) for s in symbols])
        hyper_b = t([float((ions[s].get("hyperpol") or {}).get("b") or 0.0) for s in symbols])
        front_b = {r: t([ions[s]["frontier"][r]["b"] for s in symbols]) for r in ROLES}
        front_w = {r: t([_onehot(ions[s]["frontier"][r]["k"]) for s in symbols]) for r in ROLES}
        nan = float("nan")
        pw = [list((ions[s].get("pauli") or {}).get("w") or [nan] * len(ions[s]["shells"])) for s in symbols]
        if any(len(w) != len(ions[s]["shells"]) for w, s in zip(pw, symbols)):
            raise ValueError("ions[s]['pauli']['w'] needs one charge per permanent shell")
        pauli_w = t([w + [nan] * (M - len(w)) for w in pw])
        pauli_s = t([float((ions[s].get("pauli") or {}).get("s", nan)) for s in symbols])
        pa = [list((ions[s].get("pauli_site") or {}).get("a") or [nan] * len(ions[s]["shells"])) for s in symbols]
        if any(len(a) != len(ions[s]["shells"]) for a, s in zip(pa, symbols)):
            raise ValueError("ions[s]['pauli_site']['a'] needs one amplitude per permanent shell")
        pauli_site_a = t([a + [nan] * (M - len(a)) for a in pa])
        pauli_site_s = t([float((ions[s].get("pauli_site") or {}).get("s", nan)) for s in symbols])
        disp_b = t([float(ions[s].get("disp_range") or nan) for s in symbols])
        return cls(
            symbols=symbols, numbers=torch.tensor([ATOMIC_NUMBER[s] for s in symbols]),
            charge=t([ions[s]["charge"] for s in symbols]), Z=t([ions[s]["Z"] for s in symbols]),
            shell_n=t(sh_n), shell_b=t(sh_b), alpha=t([ions[s]["alpha"] for s in symbols]),
            C=t([ions[s]["C"] or 0.0 for s in symbols]),
            b_dip=b_dip, w_dip=w_dip, b_quad=b_quad, w_quad=w_quad, b_breath=b_breath, w_breath=w_breath, a0=a0,
            front_b=front_b, front_w=front_w, I_donor=t([ions[s]["I_donor"] for s in symbols]),
            A_acceptor=t([float("nan") if ions[s]["A_acceptor"] is None else ions[s]["A_acceptor"] for s in symbols]),
            pauli_w=pauli_w, pauli_s=pauli_s, pauli_site_a=pauli_site_a, pauli_site_s=pauli_site_s, disp_b=disp_b,
            hyper_gamma=hyper_gamma, hyper_b=hyper_b,
        )

    def to(self, device) -> "IonTable":
        """A copy with every tensor on ``device``."""
        import dataclasses
        mv = lambda v: (v.to(device) if torch.is_tensor(v) else {k: x.to(device) for k, x in v.items()} if isinstance(v, dict) else v)
        return dataclasses.replace(self, **{f.name: mv(getattr(self, f.name)) for f in dataclasses.fields(self)})

    def species_index(self, atomic_numbers: torch.Tensor) -> torch.Tensor:
        dev = self.numbers.device
        lut = torch.full((int(self.numbers.max()) + 1,), -1, dtype=torch.long, device=dev)
        lut[self.numbers] = torch.arange(len(self.symbols), device=dev)
        s = lut[atomic_numbers.long().to(dev)]
        if (s < 0).any():
            raise ValueError(f"atomic numbers outside the ion table {self.symbols}")
        return s


# ---- CT channels ----------------------------------------------------------------------------------------------

@dataclass
class ChannelTable:
    """Which directed pairs 'D>A' are CT channels, the channel cutoff (Angstrom) and the width of its taper.

    Every ordered species pair is a channel type; a table's ``channels`` section may set ``cutoff`` and ``taper``
    (its calibrated pair functions, from the channel CT model, are not read any more). A type's value in ``types``
    may be its own cutoff in Angstrom (at most ``cutoff``; None: ``cutoff``): the model's frontier kernel sets each
    pair's R_off there, so that channels beyond their pair's cutoff are not listed."""

    types: dict = field(default_factory=dict)
    cutoff: float = 8.0
    taper: float = 1.0
    meta: dict = field(default_factory=dict)

    def type_cutoff(self, name: str) -> float:
        v = self.types.get(name)
        return self.cutoff if v is None else min(float(v), self.cutoff)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_PARAMS, dtype=torch.float64) -> "ChannelTable":
        d = json.load(open(path))
        ch = d.get("channels") or {}
        return cls.direct(tuple(d["ions"]), cutoff=float(ch.get("cutoff", 8.0)), taper=float(ch.get("taper", 1.0)))

    @classmethod
    def direct(cls, symbols, cutoff=8.0, taper=1.0):
        return cls(types={f"{a}>{b}": None for a in symbols for b in symbols}, cutoff=cutoff, taper=taper,
                   meta={"form": "direct"})


def channel_taper(r, r_off, width):
    """1 below r_off - width, 0 beyond r_off, a quintic smoothstep in between."""
    x = ((r - (r_off - width)) / width).clamp(0.0, 1.0)
    return 1.0 - x ** 3 * (10.0 - 15.0 * x + 6.0 * x * x)
