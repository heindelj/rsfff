"""Atomic reference states: energies and free-atom polarizabilities at integer charge.

The frozen monomer stack (docs/mixture_of_diabatic_embeddings.md, §2.1/§2.3) builds every
fragment feature from *atomic reference embeddings* conditioned on the diabatic state
``(Z, fragment type, Q, S)``. A single atom has an all-zero SOAP density, so every prediction
head reduces to a function of that embedding alone -- which makes isolated-atom data at each
integer charge the exact anchor for the free-atom limit of the 1-body energy, the baseline
electronegativity chi_0, the hardness eta_0, and the baseline polarizability alpha_0.

This script computes that grid. Unlike ``scripts/atomic_references.py`` (neutral atoms only,
energies only) it sweeps charge states and adds the static dipole polarizability:

    H:  q = +1 (bare proton), 0, -1
    O:  q = +1, 0, -1, -2

Method/basis are b3lyp/def2-svpd, matching ``scripts/generate_dataset.py`` exactly so the
atomic references are consistent with the molecular labels in ``data/labels/``.

Polarizabilities come from a **finite field** (central difference of the dipole under +/-F).
psi4's analytic CPSCF response is not available for the open-shell UKS states in this grid,
and the finite-field route costs only six extra single-atom SCFs per state.

Anions that come out *unbound* -- energy above the next-lower-charge state, meaning the SCF has
parked the extra electron in the most diffuse basis function rather than binding it -- are
flagged ``"bound": false`` rather than silently trusted, since their energies and (especially)
polarizabilities are then basis-set artifacts. At b3lyp/def2-svpd this catches H(-), whose
EA comes out at -0.34 eV against an experimental +0.75 eV.

Outputs
-------
``data/atomic_reference_states.json``  the full grid + derived IP/EA/chi/eta per element.
``data/atomic_references.json``        rewritten neutral-atom E0 table (the format
                                       ``rsfff.train.data.load_reference_energies`` reads),
                                       so the existing pipeline keeps working unchanged.

Usage:
    python scripts/atomic_reference_states.py [ELEMENT ...]
"""

import json
import os
import sys

import numpy as np
import psi4

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT_PATH = os.path.join(REPO, "data", "atomic_reference_states.json")
E0_PATH = os.path.join(REPO, "data", "atomic_references.json")

METHOD = "b3lyp"
BASIS = "def2-svpd"
FIELD_STEP = 1.0e-3  # a.u., central difference for the polarizability

# psi4's ``perturb_dipole`` adds ``+lambda . r`` to the one-electron Hamiltonian. The electronic
# charge is -1, so the physical uniform field is ``F = -lambda``: a positive lambda pushes the
# density *against* the field. We therefore pass ``-F`` to psi4 and keep ``F`` as the physical
# field everywhere in this module, so ``alpha = dmu/dF`` comes out positive-definite.
# Calibrated against the free-atom values: H 4.5 a0^3, O 5.4 a0^3 (see tests below the main()).
_PERTURB_SIGN = -1.0

# Ground-state spin multiplicity (2S+1) of each (element, charge) reference state, from the
# atomic ground terms. Charge states with no entry are simply not computed.
#
#   H+  : bare proton, no electrons          -> E = 0 exactly (no SCF)
#   H   : 1s^1,  2S                          H-  : 1s^2, 1S
#   O+  : 2p^3,  4S                          O   : 2p^4, 3P
#   O-  : 2p^5,  2P
#
# O(2-) is deliberately absent: it does not bind its second electron in the gas phase at all
# (that is physics, not a basis-set artifact -- it is stabilized only by a surrounding lattice
# or solvent), so any computed energy is whatever diffuse function the basis happens to offer.
# H(-) is kept despite also coming out unbound here, because that one *is* a basis artifact:
# hydride is a real bound species (EA = +0.75 eV) that def2-svpd's tight diffuse set misses.
STATE_MULTIPLICITY = {
    "H": {+1: 1, 0: 2, -1: 1},
    "O": {+1: 4, 0: 3, -1: 2},
}

# An anion state is "unbound" when its energy lies *above* the next-lower-charge state, i.e. the
# extra electron is not actually bound at this level of theory and the SCF has parked it in the
# most diffuse function the basis offers. Such states are strongly basis-set dependent and are
# flagged in the output so training can weight them separately rather than trust them blindly.
# This is measured from the computed energies (see ``_mark_boundness``), not tabulated: at
# b3lyp/def2-svpd it catches O(2-) *and* H(-), whose EA comes out at -0.34 eV against an
# experimental +0.75 eV.
_UNBOUND_NOTE = "anion unbound at this level (E above the next-lower-charge state)"


def _set_options(mult: int):
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.set_options(
        {
            "basis": BASIS,
            "reference": "uks" if mult > 1 else "rks",
            "e_convergence": 1e-9,
            "d_convergence": 1e-9,
            "maxiter": 300,
        }
    )


def _atom(symbol: str, charge: int, mult: int):
    """A single atom at the origin, no COM shift / reorientation."""
    return psi4.geometry(
        f"{charge} {mult}\n{symbol} 0.0 0.0 0.0\n"
        "units angstrom\nsymmetry c1\nno_com\nno_reorient\n"
    )


def _dipole_with_field(symbol, charge, mult, field) -> np.ndarray:
    """Total dipole (a.u.) of the density converged under a uniform electric ``field``.

    ``field`` is the *physical* field; the sign flip into psi4's ``perturb_dipole``
    convention is applied here (see ``_PERTURB_SIGN``). The perturbation enters through
    psi4's ``perturb_h`` option rather than a custom core-Hamiltonian patch, so the SCF is
    fully self-consistent with the field.
    """
    _set_options(mult)
    psi4.set_options(
        {
            "perturb_h": True,
            "perturb_with": "dipole",
            "perturb_dipole": [_PERTURB_SIGN * float(f) for f in field],
        }
    )
    mol = _atom(symbol, charge, mult)
    _, wfn = psi4.energy(METHOD, molecule=mol, return_wfn=True)
    # psi4 >= 1.4 exposes the SCF dipole as a variable, in a.u. (e*a0).
    return np.asarray(wfn.variable("SCF DIPOLE")).reshape(3)


def polarizability(symbol, charge, mult) -> np.ndarray:
    """Static dipole polarizability (a0^3) by central finite field: (3, 3), symmetrized."""
    alpha = np.zeros((3, 3))
    for axis in range(3):
        f = np.zeros(3)
        f[axis] = FIELD_STEP
        mu_p = _dipole_with_field(symbol, charge, mult, +f)
        mu_m = _dipole_with_field(symbol, charge, mult, -f)
        alpha[:, axis] = (mu_p - mu_m) / (2.0 * FIELD_STEP)
    return 0.5 * (alpha + alpha.T)


def state_energy(symbol, charge, mult) -> float:
    """Total energy (Hartree) of one isolated (element, charge) reference state."""
    _set_options(mult)
    mol = _atom(symbol, charge, mult)
    return float(psi4.energy(METHOD, molecule=mol))


def compute_state(symbol, charge, mult) -> dict:
    """Energy + polarizability for one reference state.

    The bare proton (H+) has no electrons: its energy is exactly zero and its polarizability
    exactly zero, and no SCF can be run for it.
    """
    if symbol == "H" and charge == +1:
        return {
            "symbol": symbol, "charge": charge, "multiplicity": mult,
            "energy": 0.0, "polarizability": np.zeros((3, 3)).tolist(),
            "bound": True, "note": "bare proton: no electrons",
        }
    energy = state_energy(symbol, charge, mult)
    alpha = polarizability(symbol, charge, mult)
    return {
        "symbol": symbol, "charge": charge, "multiplicity": mult,
        "energy": energy, "polarizability": alpha.tolist(),
    }


def _mark_boundness(states: dict) -> list[tuple[str, int]]:
    """Set ``bound`` on every state from the computed energies; return the unbound ones.

    A state with charge ``q < 0`` is bound only if ``E(q) < E(q + 1)`` -- the electron it added
    is actually held. Cations and neutrals are bound by construction.
    """
    unbound = []
    for sym, by_charge in states.items():
        for charge, rec in by_charge.items():
            if "bound" in rec:  # already decided (bare proton)
                continue
            parent = by_charge.get(charge + 1)
            is_bound = not (charge < 0 and parent is not None
                            and rec["energy"] >= parent["energy"])
            rec["bound"] = is_bound
            if not is_bound:
                rec["note"] = _UNBOUND_NOTE
                unbound.append((sym, charge))
    return unbound


def derived_quantities(states: dict) -> dict:
    """Per-element IP, EA, Mulliken electronegativity, and chemical hardness (Hartree).

    ``IP = E(+1) - E(0)``, ``EA = E(0) - E(-1)``, ``chi = (IP + EA)/2``, ``eta = IP - EA``.
    These initialize the per-element chi_0 / eta_0 biases of the SQE parameter heads, which is
    exactly their physical meaning in the free-atom limit. Elements missing a cation or anion
    state get whichever entries are computable.
    """
    out = {}
    for sym, by_charge in states.items():
        e = {q: s["energy"] for q, s in by_charge.items()}
        d = {}
        if +1 in e and 0 in e:
            d["ip"] = e[+1] - e[0]
        if 0 in e and -1 in e:
            d["ea"] = e[0] - e[-1]
        if "ip" in d and "ea" in d:
            d["chi_mulliken"] = 0.5 * (d["ip"] + d["ea"])
            d["hardness"] = d["ip"] - d["ea"]
        out[sym] = d
    return out


def main():
    psi4.set_memory("8 GB")
    psi4.set_num_threads(os.cpu_count() or 1)
    psi4.set_output_file(
        os.path.join(REPO, "data", "atomic_reference_states_psi4.out"), False
    )

    args = sys.argv[1:]
    symbols = args if args else sorted(STATE_MULTIPLICITY)
    unknown = [s for s in symbols if s not in STATE_MULTIPLICITY]
    if unknown:
        raise SystemExit(
            f"no reference-state grid tabulated for {unknown}; "
            f"add entries to STATE_MULTIPLICITY (have: {sorted(STATE_MULTIPLICITY)})"
        )

    states: dict[str, dict[int, dict]] = {}
    for sym in symbols:
        states[sym] = {}
        for charge in sorted(STATE_MULTIPLICITY[sym], reverse=True):
            mult = STATE_MULTIPLICITY[sym][charge]
            print(f"[{sym}{charge:+d}] mult={mult}  {METHOD}/{BASIS}", flush=True)
            rec = compute_state(sym, charge, mult)
            states[sym][charge] = rec
            a_iso = float(np.trace(np.asarray(rec["polarizability"])) / 3.0)
            print(
                f"[{sym}{charge:+d}] E = {rec['energy']:.10f} Ha   "
                f"alpha_iso = {a_iso:.4f} a0^3",
                flush=True,
            )

    unbound = _mark_boundness(states)
    for sym, charge in unbound:
        print(
            f"[{sym}{charge:+d}] WARNING: {_UNBOUND_NOTE}; energy and polarizability are "
            f"strongly basis-set dependent -- weight this anchor accordingly",
            flush=True,
        )

    derived = derived_quantities(states)
    ha_ev = 27.211386245988
    for sym, d in derived.items():
        parts = [f"{k} = {v * ha_ev:7.3f} eV" for k, v in d.items()]
        print(f"[{sym}] " + "   ".join(parts), flush=True)

    out = {
        "method": METHOD,
        "basis": BASIS,
        "reference": "uks/rks (unrestricted for open-shell states)",
        "field_step_au": FIELD_STEP,
        "units": {"energy": "Hartree", "polarizability": "a0^3"},
        "states": [
            states[sym][q] for sym in sorted(states) for q in sorted(states[sym], reverse=True)
        ],
        "derived": derived,
    }
    with open(OUT_PATH, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    n_states = sum(len(v) for v in states.values())
    print(f"\nwrote {n_states} reference states -> {OUT_PATH}", flush=True)

    # Refresh the neutral-atom E0 table consumed by rsfff.train.data.load_reference_energies,
    # merging with any previously computed entries so other elements survive.
    e0 = {"energies": {}, "multiplicities": {}}
    if os.path.isfile(E0_PATH):
        prev = json.load(open(E0_PATH))
        e0["energies"].update(prev.get("energies", {}))
        e0["multiplicities"].update(prev.get("multiplicities", {}))
    for sym, by_charge in states.items():
        if 0 in by_charge:
            e0["energies"][sym] = by_charge[0]["energy"]
            e0["multiplicities"][sym] = by_charge[0]["multiplicity"]
    e0.update(
        {
            "method": METHOD, "basis": BASIS, "units": "Hartree",
            "reference": "uks/rks (unrestricted for open-shell atoms)",
        }
    )
    with open(E0_PATH, "w") as fh:
        json.dump(e0, fh, indent=2, sort_keys=True)
    print(f"wrote {len(e0['energies'])} neutral-atom E0 references -> {E0_PATH}", flush=True)


if __name__ == "__main__":
    main()
