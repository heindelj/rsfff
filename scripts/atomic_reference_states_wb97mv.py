"""Atomic reference states at wB97M-V/def2-TZVPD, via :mod:`rsfff.qcgen`.

The wB97M-V analogue of ``scripts/atomic_reference_states.py`` (b3lyp/def2-svpd via psi4),
standing in the same relation to it as ``scripts/isolated_species_wb97mv.py`` does to
``scripts/isolated_species.py``: same grid, same conventions, different level of theory and a
pyscf/gpu4pyscf backend shared with the rest of the reference-data pipeline.

**Why this exists.** ``data/atomic_reference_states.json`` says ``b3lyp``/``def2-svpd`` in its
own header, while every molecular label the unified model fits -- the Q-Chem ALMO-EDA clusters
and the monomer anchor alike -- is wB97M-V/def2-TZVPD. Those states are not decoration: they
seed the per-element ``chi_0``/``eta_0`` biases of the SQE charge solve
(:meth:`rsfff.mlip.reference_states.AtomicStateReference.head_bias_init`), and their free-atom
polarizabilities are the *exact* anchor for the on-site polarizability head -- a lone atom has
an all-zero density, so every head reduces to a function of its species embedding alone and the
anchor pins it rather than nudging it. An anchor at the wrong level of theory pins the wrong
number.

Grid, unchanged from the psi4 script (see ``STATE_MULTIPLICITY`` there for the atomic terms):

    H:  q = +1 (bare proton), 0, -1
    O:  q = +1, 0, -1, -2

Polarizabilities come from a **finite field**, requested explicitly rather than left to the
automatic fallback in :func:`rsfff.qcgen.compute.compute_reference_data`. Every state here
except the bare proton and O(2-) is open shell, and UKS + a non-local correlation functional is
exactly the combination whose analytic CPSCF response is unavailable -- so the fallback would
fire for most of the grid and not for the rest, silently mixing two methods within one table.
``response_method`` is recorded per state either way.

Anions whose energy lies *above* the next-lower-charge state are flagged ``bound: false``
rather than trusted: the SCF has parked the extra electron in the most diffuse basis function
it can find, so the energy and (especially) the polarizability are basis-set artifacts.
``rsfff.train.loss.atomic_reference_loss`` drops them by default.

Measured output of this grid, and two things in it are worth knowing before using a charged
state as an anchor:

    state   E (Ha)          alpha_iso (a0^3)    bound
    H+       0.0000000000     0.000             yes (exact, no electrons)
    H        -0.4941110651    5.053             yes
    H-       -0.4960244991   20.586             yes  <- newly bound; see below
    O+      -74.5710803944    2.586             yes
    O       -75.0780656004    5.529             yes
    O-      -75.1327158469   18.576             yes
    O2-     -74.8962191667   46.488             no

    H:  IP 13.445 eV   EA 0.052 eV     O:  IP 13.796 eV   EA 1.487 eV

* **H(-) binds here and did not at b3lyp/def2-svpd**, where its EA came out at -0.34 eV. The
  extra diffuse functions in def2-TZVPD are enough to flip the sign, so the ``bound`` flag now
  reads ``true`` -- but the EA is +0.052 eV against an experimental +0.754 eV, so the state is
  bound by a factor of fourteen too little and its polarizability is still mostly a property
  of the most diffuse function available. The ``bound`` test catches a catastrophically
  unbound anion, not a badly described one; treat H(-) accordingly.
* The neutral polarizabilities are 5.05 (H) and 5.53 (O) a0^3 against exact/reference values
  of 4.50 and ~5.4. O is close; H is 12% high, which is ordinary self-interaction error for a
  one-electron atom and is *the right anchor anyway* -- the model is fitting wB97M-V data, so
  its free-atom limit should be wB97M-V's free atom, not nature's.

Writes ``data/atomic_reference_states_wb97mv_tzvpd.json``. Unlike the psi4 script this does
**not** rewrite ``data/atomic_references.json``: the unified configs point at
``data/atomic_references_wb97mv_tzvpd.json``, whose H and O values already agree with the
Q-Chem jobs that produced the labels to ~1e-8 Ha, and silently replacing a table that agrees
with the labels would be a regression. The neutral energies are printed beside it as a check
instead.

Usage:
    python scripts/atomic_reference_states_wb97mv.py [ELEMENT ...]
"""

import json
import os
import sys

import numpy as np

from rsfff.qcgen.backend import backend_name
from rsfff.qcgen.compute import compute_reference_data

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT_PATH = os.path.join(REPO, "data", "atomic_reference_states_wb97mv_tzvpd.json")
#: Compared against, never written. See the module docstring.
E0_REFERENCE = os.path.join(REPO, "data", "atomic_references_wb97mv_tzvpd.json")

METHOD = "wB97M-V"
BASIS = "def2-tzvpd"
FIELD_STEP = 1.0e-3  # a.u., central difference for the polarizability

# Ground-state spin multiplicity (2S+1) of each (element, charge), from the atomic ground
# terms -- identical to the psi4 script, because these are properties of the atom and not of
# the functional:
#
#   H+  : bare proton, no electrons          -> E = 0 exactly (no SCF)
#   H   : 1s^1,  2S                          H-  : 1s^2, 1S
#   O+  : 2p^3,  4S                          O   : 2p^4, 3P
#   O-  : 2p^5,  2P                          O2- : 2p^6, 1S
STATE_MULTIPLICITY = {
    "H": {+1: 1, 0: 2, -1: 1},
    "O": {+1: 4, 0: 3, -1: 2, -2: 1},
}

_UNBOUND_NOTE = "anion unbound at this level (E above the next-lower-charge state)"

HA_EV = 27.211386245988
#: e^2 a0^2 / Ha per a0^3 -- the polarizability unit the JSON records.
A0_CUBED = 1.0


def compute_state(symbol: str, charge: int, mult: int) -> dict:
    """Energy + static polarizability for one reference state, at the origin.

    The bare proton has no electrons: no SCF can be run for it, and both its energy and its
    polarizability are exactly zero rather than approximately so.
    """
    if symbol == "H" and charge == +1:
        return {
            "symbol": symbol, "charge": charge, "multiplicity": mult,
            "energy": 0.0, "polarizability": np.zeros((3, 3)).tolist(),
            "bound": True, "response_method": "exact",
            "note": "bare proton: no electrons",
        }
    data = compute_reference_data(
        [symbol], np.zeros((1, 3)), charge, mult - 1, METHOD, BASIS,
        response="finite-difference", field_step=FIELD_STEP,
    )
    return {
        "symbol": symbol, "charge": charge, "multiplicity": mult,
        "energy": float(data["energy"]),
        "polarizability": np.asarray(data["polarizability"]).tolist(),
        "response_method": data["response_method"],
    }


def mark_boundness(states: dict) -> list[tuple[str, int]]:
    """Set ``bound`` on every state from the computed energies; return the unbound ones.

    A state with charge ``q < 0`` is bound only if ``E(q) < E(q + 1)`` -- the electron it added
    is actually held. Cations and neutrals are bound by construction. Measured rather than
    tabulated, because which anions bind is a property of the level of theory and this script
    exists precisely because the level of theory changed.
    """
    unbound = []
    for sym, by_charge in states.items():
        for charge, rec in by_charge.items():
            if "bound" in rec:  # already decided (bare proton)
                continue
            parent = by_charge.get(charge + 1)
            is_bound = not (
                charge < 0 and parent is not None and rec["energy"] >= parent["energy"]
            )
            rec["bound"] = is_bound
            if not is_bound:
                rec["note"] = _UNBOUND_NOTE
                unbound.append((sym, charge))
    return unbound


def derived_quantities(states: dict) -> dict:
    """Per-element IP, EA, Mulliken electronegativity and chemical hardness, in Hartree.

    ``IP = E(+1) - E(0)``, ``EA = E(0) - E(-1)``, ``chi = (IP + EA)/2``, ``eta = IP - EA``:
    the free-atom limits of the SQE ``chi`` and ``eta``, which is exactly why they seed those
    heads' per-element biases.
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


def _check_against_label_e0(states: dict) -> None:
    """Print the neutral energies beside the E0 table the training configs actually load.

    A disagreement here is worth knowing about but is not this script's to fix: the table it
    compares against was produced to match the Q-Chem jobs that produced the *labels*, and the
    model adds it to its own prediction, so the labels and that table have to agree with each
    other -- not with this grid.
    """
    if not os.path.isfile(E0_REFERENCE):
        print(f"\n{E0_REFERENCE} not present; skipping the E0 cross-check", flush=True)
        return
    ref = json.load(open(E0_REFERENCE)).get("energies", {})
    print(f"\nneutral-atom energies vs {os.path.basename(E0_REFERENCE)}:", flush=True)
    for sym, by_charge in sorted(states.items()):
        if 0 not in by_charge or sym not in ref:
            continue
        here, there = by_charge[0]["energy"], float(ref[sym])
        print(
            f"  {sym:>2s}  this grid {here:18.10f}   E0 table {there:18.10f}   "
            f"delta {(here - there) * HA_EV * 96.485:+.4f} kJ/mol",
            flush=True,
        )


def main() -> None:
    args = sys.argv[1:]
    symbols = args if args else sorted(STATE_MULTIPLICITY)
    unknown = [s for s in symbols if s not in STATE_MULTIPLICITY]
    if unknown:
        raise SystemExit(
            f"no reference-state grid tabulated for {unknown}; "
            f"add entries to STATE_MULTIPLICITY (have: {sorted(STATE_MULTIPLICITY)})"
        )

    print(f"{METHOD}/{BASIS} on {backend_name()}", flush=True)
    states: dict[str, dict[int, dict]] = {}
    for sym in symbols:
        states[sym] = {}
        for charge in sorted(STATE_MULTIPLICITY[sym], reverse=True):
            mult = STATE_MULTIPLICITY[sym][charge]
            print(f"[{sym}{charge:+d}] mult={mult}", flush=True)
            rec = compute_state(sym, charge, mult)
            states[sym][charge] = rec
            a_iso = float(np.trace(np.asarray(rec["polarizability"])) / 3.0)
            print(
                f"[{sym}{charge:+d}] E = {rec['energy']:.10f} Ha   "
                f"alpha_iso = {a_iso:.4f} a0^3   ({rec['response_method']})",
                flush=True,
            )

    for sym, charge in mark_boundness(states):
        print(
            f"[{sym}{charge:+d}] WARNING: {_UNBOUND_NOTE}; energy and polarizability are "
            f"strongly basis-set dependent -- weight this anchor accordingly",
            flush=True,
        )

    derived = derived_quantities(states)
    for sym, d in derived.items():
        parts = [f"{k} = {v * HA_EV:7.3f} eV" for k, v in d.items()]
        print(f"[{sym}] " + "   ".join(parts), flush=True)

    out = {
        "method": METHOD,
        "basis": BASIS,
        "backend": backend_name(),
        "reference": "uks/rks (unrestricted for open-shell states)",
        "field_step_au": FIELD_STEP,
        "response": "finite-difference (forced; UKS + VV10 has no analytic CPSCF path)",
        "units": {"energy": "Hartree", "polarizability": "a0^3"},
        "states": [
            states[sym][q]
            for sym in sorted(states) for q in sorted(states[sym], reverse=True)
        ],
        "derived": derived,
    }
    with open(OUT_PATH, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    n_states = sum(len(v) for v in states.values())
    print(f"\nwrote {n_states} reference states -> {OUT_PATH}", flush=True)

    _check_against_label_e0(states)


if __name__ == "__main__":
    main()
