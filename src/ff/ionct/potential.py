"""The complete ion potential: every ALMO-EDA term, differentiable in the positions.

    E_int = E_elec[perm] + E_Pauli[frozen] + E_disp + E_pol + E_CT
            (cls_elec)     (mod_pauli)        (disp)   (pol)   (ct)

* E_elec: the fixed shaped densities (``kernels.permanent_pair_energy``), no parameters learned;
* E_pol, E_CT: a trained :class:`IonCTModel`;
* E_Pauli: a separately fit :class:`IonPauli` if given, else the CT model's own frozen Pauli (``pol_form="pauli"``);
* E_disp: a trained :class:`IonDispersion`.

Ions are atoms, so E_int differs from the total energy by a constant (the free-ion energies) and the forces are
-dE_int/dX. Positions in Angstrom; energies Hartree; forces Hartree/bohr.
"""
from __future__ import annotations

from pathlib import Path

import torch

from . import ct as CT
from . import kernels as K
from .dispersion import IonDispersion
from .frozen_pauli import IonPauli
from .model import IonCTModel, collate, precompute
from .params import IonTable

ZMAP = {"Na": 11, "Cl": 17}


def soap_featurizer():
    from bb.features import SOAPFeaturizer
    return SOAPFeaturizer(cutoff=6.0, n_max=6, l_max=3, neighbor_types=(11, 17), selected_lambdas=(0,)).double()


def load_ct_model(run, params=None):
    ck = torch.load(Path(run) / "model.pt", weights_only=False)
    a = ck["args"]
    tab, chan = IonTable.load(params or a["params"]), CT.ChannelTable.load(params or a["params"])
    m = IonCTModel(tab, chan, ck["n_feat"], hidden=a["hidden"], wkb=a["wkb"],
                   c_range=tuple(a.get("c_range", (0.05, 1.25))), confine=a.get("confine"),
                   pseudo=a.get("pseudo", False), joint=a.get("joint", False), ct_pol=a.get("ct_pol", False),
                   gap_form=a.get("gap_form", "pair"), tie_c=a.get("tie_c", False),
                   coupling_resp=a.get("coupling_resp", "barrier"), ct_form=a.get("ct_form", "channels"),
                   pol_form=a.get("pol_form", "confine"), pauli_nn=a.get("pauli_nn", False))
    m.load_state_dict(ck["state"])
    m.eval()
    return m


class IonPotential(torch.nn.Module):
    def __init__(self, ct_model: IonCTModel, disp: IonDispersion, pauli: IonPauli | None = None, featurizer=None):
        super().__init__()
        self.ct, self.disp, self.pauli = ct_model, disp, pauli
        if pauli is None and ct_model.pol_form != "pauli":
            raise ValueError("the CT model has no frozen Pauli term: pass an IonPauli")
        self.tab, self.chan = ct_model.tab, ct_model.chan
        self.feat = featurizer or soap_featurizer()

    @classmethod
    def from_runs(cls, ct_run, disp_path, pauli_path=None):
        return cls(load_ct_model(ct_run), IonDispersion.load(disp_path),
                   IonPauli.load(pauli_path) if pauli_path else None)

    def terms_cached(self, caches):
        """Per-term energies (B,) for precomputed frames (fast, no position gradients)."""
        bt = collate(caches)
        out = self.ct(bt)
        sp, fr = bt["species"], bt["frame"]
        pairs = torch.cat([torch.triu_indices(c.species.shape[0], c.species.shape[0], 1) + o
                           for c, o in zip(caches, torch.cumsum(torch.tensor([0] + [c.species.shape[0] for c in caches[:-1]]), 0))], 1)
        e_el = bt["pos"].new_zeros(bt["B"]).index_add(0, fr[pairs[0]], K.permanent_pair_energy(bt["pos"], sp, pairs, self.tab))
        e_disp = self.disp(bt["pos"], sp, fr, bt["B"], pairs, bt["feats"])
        e_p = out["e_pauli"] if self.pauli is None else self.pauli(bt["pos"], sp, fr, bt["B"], pairs, bt["feats"])
        return self._pack(e_el, out, e_disp, e_p)

    @staticmethod
    def _pack(e_el, out, e_disp, e_pauli):
        t = {"elec": e_el, "pauli": e_pauli, "disp": e_disp, "pol": out["e_pol"], "ct": out["e_ct"]}
        t["int"] = sum(t.values())
        return t

    def terms(self, symbols, pos_ang):
        """Per-term energies (scalars) of one frame; differentiable in pos_ang (N, 3) Angstrom."""
        c = precompute(symbols, pos_ang, self.tab, self.chan, self.feat, differentiable=True)
        bt = collate([c])
        out = self.ct(bt)
        n = len(symbols)
        pairs = torch.triu_indices(n, n, 1)
        sp = c.species
        e_el = K.permanent_pair_energy(pos_ang, sp, pairs, self.tab).sum()[None]
        b0 = torch.zeros(n, dtype=torch.long)
        e_disp = self.disp(pos_ang, sp, b0, 1, pairs, c.feats)
        e_p = out["e_pauli"] if self.pauli is None else self.pauli(pos_ang, sp, b0, 1, pairs, c.feats)
        return {k: v[0] for k, v in self._pack(e_el, out, e_disp, e_p).items()}

    def energy_forces(self, symbols, pos_ang):
        """(E_int, forces (N, 3) Hartree/bohr, per-term energies)."""
        X = torch.as_tensor(pos_ang, dtype=torch.float64).clone().requires_grad_(True)
        t = self.terms(symbols, X)
        g, = torch.autograd.grad(t["int"], X)
        return t["int"].detach(), -g * K.BOHR_ANG, {k: v.detach() for k, v in t.items()}
