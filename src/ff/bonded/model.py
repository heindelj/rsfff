"""The bonded model: the terms of :mod:`rsfff.ff.film.bonded` / :mod:`rsfff.ff.film.terms` with parameters decoded
linearly from one latent per directed bond.

    x_e = edge cluster expansion of bond i -> j           (bb.features.EdgeClusterExpansion)
    c_e = embed(descriptors and degrees of i and j, the molecule's charge and multiplicity)
    z_e = encoder(x_e; c_e)                               (bb.heads.EdgeLatentEncoder; latent ~ 8)
    p   = table[type] modulated by a bounded linear readout of the latents of the term's edges

=====================  ==============================================  ==========================================
term                   latents                                         parameters
=====================  ==============================================  ==========================================
bond i-j               z[i->j] + z[j->i]                               r_eq e^{cap_r(u)}, D e^{cap(u)}, k e^{cap(u)}
angle i-a-k            z[a->i] + z[a->k]                               tanh(atanh cos_eq + cap(u)), k_theta e^{cap(u)}
bond-bond              z[a->i] + z[a->k]                               rho_bb
bond-angle (per leg)   z[a->leg], z[a->other]   (two rows)             rho_ba
angle-angle, opposite  z[a->s], z[a->x] + z[a->y]   (two rows)         rho_aa, rho_bo
improper (per leg)     z[c->leg], z[c->j] + z[c->k]   (two rows)       c_chi + cap(u), k_chi e^{cap(u)}
=====================  ==============================================  ==========================================

cap(u) = c tanh(u / c) with c = ``param_cap`` (``r_cap`` for bond lengths); rho = 0.95 tanh(raw + cap(u)). Every
readout row starts at zero, so a fresh model is the constant-parameter model of its tables. The energy is

    E = sum_atoms e0[type] + sum of the listed terms                                 (Hartree)

**Electrical head** (``[terms]``: charges, dipoles, quadrupoles, polarizability, quadrupole_polarizability). The
parameters of the no-Coulomb split-charge response model (:mod:`rsfff.ff.bonded.response`) come from the same latents:

=====================  ==============================================  ==========================================
parameter              latents                                         form
=====================  ==============================================  ==========================================
dchi of bond i-j       z[i->j] - z[j->i]                               chi[t_j] - chi[t_i] + cap(u)     (sqe)
compliance s of i-j    z[i->j] + z[j->i]                               s e^{cap(u)}                     (sqe)
hardness eta_i         sum_j z[i->j]                                   eta e^{cap(u)}                   (sqe)
bond flow i -> j       z[i->j] - z[j->i]                               flow + cap(u)                    (flow)
dipole mu_i (1o)       per edge i->j: gates = rows . z, on the         sum over i's edges
second moment (2e)       edge's m-channels (bb.heads.EdgeGateReadout)  trace: m2 + cap(u) from sum_j z[i->j]
polarizability alpha   sum_j z[i->j] (scale), 2e gates (shape)         alpha (e^{cap(u)} 1 + T)^2, positive definite
c_quad                 sum_j z[i->j]                                   c e^{cap(u)}
=====================  ==============================================  ==========================================

``charges = "sqe"``: the permanent charges are the zero-field solution, q = Q / n + B p*(dchi, eta, s), and the same
eta and s give the charge-flow polarizability. The response sector's own zero-field energy is not part of E: the
energy surface is the bonded terms alone, and the model's energy in a field F and gradient G is
E + ``solution.energy(F, G)``. Moments are about the coordinate origin, in atomic units.
"""
from __future__ import annotations

import torch
from torch import nn

from bb.features import EdgeClusterExpansion
from bb.graph import find_edges
from bb.heads import EdgeGateReadout, EdgeLatentEncoder, LinearDecoder
from bb.o3.cartesian import irrep1_to_vector, irrep2_to_traceless_matrix

from ..film.bonded import BondedParameters, cosine_angle_energy, improper_energy, morse_energy
from ..film.terms import RHO_MAX, TermParameters, term_energies
from ..units import BOHR_ANG
from .config import BondedConfig
from .response import ResponseParameters, ResponseSolution, response
from .tables import APEX, ATOM, BondedTables

__all__ = ["BondedModel"]

#: Z -> (Pauling electronegativity / 4, covalent radius [A], valence electrons / 8, period / 4)
ELEMENT_DESCRIPTORS = {
    1: (0.550, 0.31, 0.125, 0.25), 6: (0.638, 0.76, 0.500, 0.50), 7: (0.760, 0.71, 0.625, 0.50),
    8: (0.860, 0.66, 0.750, 0.50), 9: (0.995, 0.57, 0.875, 0.50), 15: (0.548, 1.07, 0.625, 0.75),
    16: (0.645, 1.05, 0.750, 0.75), 17: (0.790, 1.02, 0.875, 0.75),
}
_A = {n: i for i, n in enumerate(APEX)}
_T = {n: i for i, n in enumerate(ATOM)}


class BondedModel(nn.Module):
    """``forward(batch) -> {"energy": (M,), "parameters": {...}}`` for a :class:`rsfff.ff.bonded.Batch`."""

    def __init__(self, config: BondedConfig, atom_types, table_root: str = ""):
        super().__init__()
        self.config = cfg = config
        if cfg.has("torsion"):
            raise NotImplementedError("the torsion term is not wired into BondedModel yet")
        self.atom_types = [(int(z), int(d)) for z, d in atom_types]
        elements = sorted({z for z, _ in self.atom_types})
        missing = [z for z in elements if z not in ELEMENT_DESCRIPTORS]
        if missing:
            raise ValueError(f"no element descriptors for Z = {missing}")
        init = cfg.table_init and str((__import__("pathlib").Path(table_root) / cfg.table_init))
        self.tables = BondedTables(self.atom_types, init=init, learn=cfg.table_learn)

        elut = torch.full((120,), -1, dtype=torch.long)
        elut[torch.tensor(elements)] = torch.arange(len(elements))
        self.register_buffer("elut", elut, persistent=False)
        desc = torch.zeros(120, 4, dtype=torch.float64)
        for z in elements:
            desc[z] = torch.tensor(ELEMENT_DESCRIPTORS[z])
        self.register_buffer("desc", desc, persistent=False)

        self.ece = EdgeClusterExpansion(elements, n_max=cfg.n_max, l_max=cfg.l_max, channels=cfg.channels,
                                        species_dim=cfg.species_dim, morse_a=cfg.morse_a)
        on = cfg.condition_on
        self.n_cond = 2 * (4 * ("elements" in on) + ("degrees" in on)) + ("charge" in on) + ("multiplicity" in on)
        use_c = cfg.conditioning != "none" and self.n_cond > 0
        self.cond = nn.Sequential(nn.Linear(self.n_cond, 16), nn.SiLU(), nn.Linear(16, cfg.condition_dim)) if use_c else None
        self.encoder = EdgeLatentEncoder.from_ece(
            self.ece, latent=cfg.latent, hidden=cfg.hidden, depth=cfg.depth, d_c=cfg.condition_dim,
            conditioning=cfg.conditioning if use_c else "none", odd=cfg.odd, compress=cfg.compress)

        L = cfg.latent
        self.dec = nn.ModuleDict()
        if cfg.has("morse"):
            self.dec["bond"] = LinearDecoder(L, 3)
        if cfg.has("angle"):
            self.dec["angle"] = LinearDecoder(L, 2)
        if cfg.has("bond_bond"):
            self.dec["bb"] = LinearDecoder(L, 1)
        if cfg.has("bond_angle"):
            self.dec["ba"] = LinearDecoder(L, 1, slots=2)
        if cfg.has("angle_angle") or cfg.has("opposite_bond_angle"):
            self.dec["pair"] = LinearDecoder(L, 2, slots=2)
        if cfg.has("improper"):
            self.dec["improper"] = LinearDecoder(L, 2, slots=2)

        # ---- electrical head ----
        self.aniso = cfg.polarizability == "anisotropic"
        self.has_electrical = bool(cfg.charges != "none" or cfg.dipoles or cfg.quadrupoles
                                   or cfg.polarizability != "none" or cfg.quadrupole_polarizability)
        if cfg.charges != "none":
            self.dec["link_anti"] = LinearDecoder(L, 1)          # dchi (sqe) or the bond flow (flow)
        if cfg.charges == "sqe":
            self.dec["link_sym"] = LinearDecoder(L, 1)           # ln s
        self.atom_rows = [n for n, on in (("eta", cfg.charges == "sqe"), ("alpha", cfg.polarizability != "none"),
                                          ("c_quad", cfg.quadrupole_polarizability), ("m2_trace", cfg.quadrupoles)) if on]
        if self.atom_rows:
            self.dec["atom"] = LinearDecoder(L, len(self.atom_rows))
        self.n2 = int(cfg.quadrupoles) + int(self.aniso)
        irreps = " + ".join(x for x in ("1x1o" if cfg.dipoles else "", f"{self.n2}x2e" if self.n2 else "") if x)
        self.gate_readout = None
        if irreps:
            if cfg.l_max < (2 if self.n2 else 1):
                raise ValueError("atomic quadrupoles / anisotropic polarizabilities need l_max >= 2, dipoles l_max >= 1")
            self.gate_readout = EdgeGateReadout(self.ece, irreps)
            self.dec["gates"] = LinearDecoder(L, self.gate_readout.n_gates)
        self.double()

    # ---- pieces ----------------------------------------------------------------------------------------------
    def conditioning(self, batch) -> torch.Tensor | None:
        """(E, condition_dim) for every directed edge, or None."""
        if self.cond is None:
            return None
        src, dst = batch.graph.edge_index
        deg, on, parts = batch.topo.degree.to(batch.positions.dtype), self.config.condition_on, []
        for a in (src, dst):
            if "elements" in on:
                parts.append(self.desc[batch.z[a]])
            if "degrees" in on:
                parts.append(deg[a, None] / 4.0)
        mol = batch.mol_idx[src]
        if "charge" in on:
            parts.append(batch.charge[mol, None])
        if "multiplicity" in on:
            parts.append(batch.multiplicity[mol, None] - 1.0)
        return self.cond(torch.cat(parts, dim=-1))

    def edge_features(self, batch, positions=None):
        pos = batch.positions if positions is None else positions
        return self.ece(pos, self.elut[batch.z], batch.graph)

    def latents(self, batch, positions=None) -> torch.Tensor:
        return self.encoder(self.edge_features(batch, positions), self.conditioning(batch))

    def _encode(self, batch, pos, constant: bool):
        """(edge features, latents); ``constant``: zero latents, i.e. the tables alone."""
        ef = self.edge_features(batch, pos)
        z = pos.new_zeros(batch.graph.n_edges, self.config.latent) if constant else self.encoder(ef, self.conditioning(batch))
        return ef, z

    @torch.no_grad()
    def set_statistics(self, batch) -> None:
        """Standardize the edge features on this batch (the training frames)."""
        self.encoder.set_statistics(self.edge_features(batch))

    def _cap(self, u, c=None):
        c = self.config.param_cap if c is None else c
        return c * torch.tanh(u / c)

    def parameters_of(self, batch, z) -> tuple[BondedParameters, TermParameters]:
        """The emitted parameters of every term instance of the batch (atomic units)."""
        cfg, topo, g, tab = self.config, batch.topo, batch.graph, self.tables
        t = tab.type_index(batch.z, topo.degree)
        edge = lambda a, b: z[find_edges(g, a, b)]
        zero = lambda n, *s: z.new_zeros(n, *s)

        bi, bj = topo.bond_index
        row = tab.bond_rows(t[bi], t[bj])
        u = self.dec["bond"](edge(bi, bj) + edge(bj, bi)) if "bond" in self.dec else zero(bi.numel(), 3)
        r_eq = torch.exp(row[:, 0] + self._cap(u[:, 0], cfg.r_cap))
        d, k = torch.exp(row[:, 1] + self._cap(u[:, 1])), torch.exp(row[:, 2] + self._cap(u[:, 2]))

        i, a, kk = topo.angle_index
        na = a.numel()
        arow = tab.apex_rows(t[a])
        zi, zk = (edge(a, i), edge(a, kk)) if na else (zero(0, cfg.latent), zero(0, cfg.latent))
        u = self.dec["angle"](zi + zk) if "angle" in self.dec else zero(na, 2)
        cos_eq = torch.tanh(arow[:, _A["cos_theta_eq"]] + self._cap(u[:, 0]))
        k_theta = torch.exp(arow[:, _A["k_theta"]] + self._cap(u[:, 1]))
        bp = BondedParameters(r_eq=r_eq, d=d, k=k, cos_theta_eq=cos_eq, k_theta=k_theta)
        tp = TermParameters()
        rho = lambda raw, u: RHO_MAX * torch.tanh(raw + self._cap(u))
        if cfg.has("bond_bond") or cfg.has("bond_angle"):
            tp.bb_rho = rho(arow[:, _A["rho_bb"]], self.dec["bb"](zi + zk)[:, 0]) if cfg.has("bond_bond") else zero(na)
            if cfg.has("bond_angle"):
                raw = arow[:, _A["rho_ba"]]
                tp.ba_rho = torch.stack((rho(raw, self.dec["ba"](zi, zk)[:, 0]), rho(raw, self.dec["ba"](zk, zi)[:, 0])), -1)
            else:
                tp.ba_rho = zero(na, 2)
        if "pair" in self.dec and topo.n_angle_pairs:
            pa, ps, px, py = topo.angle_pair_index
            prow = tab.apex_rows(t[pa])
            u = self.dec["pair"](edge(pa, ps), edge(pa, px) + edge(pa, py))
            if cfg.has("angle_angle"):
                tp.aa_rho = rho(prow[:, _A["rho_aa"]], u[:, 0])
            if cfg.has("opposite_bond_angle"):
                tp.bo_rho = rho(prow[:, _A["rho_bo"]], u[:, 1])
        if "improper" in self.dec and topo.n_impropers:
            c, leg, j, l = topo.improper_index
            irow = tab.apex_rows(t[c])
            u = self.dec["improper"](edge(c, leg), edge(c, j) + edge(c, l))
            bp.c_chi = irow[:, _A["c_chi"]] + self._cap(u[:, 0])
            bp.k_chi = torch.exp(irow[:, _A["k_chi"]] + self._cap(u[:, 1]))
        return bp, tp

    # ---- energy ----------------------------------------------------------------------------------------------
    def bonded_energy(self, batch, positions=None, constant: bool = False, z=None):
        """((M,) bonded energy without the atomic references, parameters). ``constant``: the tables alone."""
        pos = batch.positions if positions is None else positions
        cfg, topo = self.config, batch.topo
        if z is None:
            z = self._encode(batch, pos, constant)[1]
        bp, tp = self.parameters_of(batch, z)
        e = pos.new_zeros(batch.n_mol)
        r, cos_t = topo.geometry(pos)
        if cfg.has("morse"):
            e = e.index_add(0, topo.bond_frag, morse_energy(r, bp.r_eq, bp.d, bp.k))
        if cfg.has("angle") and cos_t.numel():
            e = e.index_add(0, topo.angle_frag, cosine_angle_energy(cos_t, bp.cos_theta_eq, bp.k_theta))
        for term, frag in term_energies(pos, topo, bp, tp, geometry=(r, cos_t)):
            e = e.index_add(0, frag, term)
        imp = improper_energy(pos, topo, bp)
        if imp is not None:
            e = e.index_add(0, topo.improper_frag, imp)
        return e, (bp, tp)

    def reference_energy(self, batch) -> torch.Tensor:
        t = self.tables.type_index(batch.z, batch.topo.degree)
        return batch.positions.new_zeros(batch.n_mol).index_add(0, batch.mol_idx, self.tables.e0[t])

    # ---- electrical ------------------------------------------------------------------------------------------
    def response_parameters(self, batch, ef, z) -> ResponseParameters:
        """The split-charge / atomic multipole / polarizability parameters of the batch (atomic units)."""
        cfg, topo, g, tab = self.config, batch.topo, batch.graph, self.tables
        N, dt = batch.z.numel(), z.dtype
        t = tab.type_index(batch.z, topo.degree)
        src = g.edge_index[0]
        atom_sum = lambda v: v.new_zeros((N,) + v.shape[1:]).index_add(0, src, v)
        n_at = torch.bincount(batch.mol_idx, minlength=batch.n_mol).to(dt)
        p = ResponseParameters(q0=(batch.charge / n_at)[batch.mol_idx])
        arow = tab.atom[t]
        u = self.dec["atom"](atom_sum(z)) if self.atom_rows else None
        col = lambda name: self._cap(u[:, self.atom_rows.index(name)])

        if cfg.charges != "none":
            bi, bj = topo.bond_index                         # response's B: -1 on bi, +1 on bj
            zf, zr = z[find_edges(g, bi, bj)], z[find_edges(g, bj, bi)]
            ln_s, flow0 = tab.link_rows(t[bi], t[bj])
            anti = self._cap(self.dec["link_anti"](zf - zr)[:, 0])
            if cfg.charges == "sqe":
                p.dchi = arow[bj, _T["chi"]] - arow[bi, _T["chi"]] + anti
                p.eta = torch.exp(arow[:, _T["eta"]] + col("eta"))
                p.compliance = torch.exp(ln_s + self._cap(self.dec["link_sym"](zf + zr)[:, 0]))
            else:
                flow = flow0 + anti
                p.q0 = p.q0.index_add(0, bi, -flow).index_add(0, bj, flow)

        # gates as a mean over the edge's m-channels (1 / n_cov), not a sum: one optimizer step then moves an output
        # by as much as it moves a scalar readout, whatever the number of channels
        y = (self.gate_readout.edge_values(ef, self.dec["gates"](z) / self.gate_readout.n_cov)
             if self.gate_readout is not None else {})
        eye = torch.eye(3, dtype=dt, device=z.device)
        if cfg.dipoles:
            p.mu0 = irrep1_to_vector(atom_sum(y["1o"][:, :, 0]), self.ece.backend)
        two = (lambda k: irrep2_to_traceless_matrix(atom_sum(y["2e"][:, :, k]), self.ece.backend)) if self.n2 else None
        if cfg.polarizability != "none":
            a = torch.exp(col("alpha"))[:, None, None] * eye
            if self.aniso:
                a = a + two(0)
                a = a @ a                                    # positive semidefinite, equivariant
            else:
                a = a * a
            p.alpha = torch.exp(arow[:, _T["alpha"]])[:, None, None] * a
        if cfg.quadrupoles:
            tr = arow[:, _T["m2_trace"]] + col("m2_trace")
            p.m2 = (tr / 3.0)[:, None, None] * eye + two(self.n2 - 1)
        if cfg.quadrupole_polarizability:
            p.c_quad = torch.exp(arow[:, _T["c_quad"]] + col("c_quad"))
        return p

    def electrical(self, batch, positions=None, constant: bool = False, ef=None, z=None) -> ResponseSolution:
        """Permanent charges and moments and the generalized polarizability (:func:`response.response`)."""
        if not self.has_electrical:
            raise ValueError("this model has no electrical terms ([terms] of its config)")
        pos = batch.positions if positions is None else positions
        if z is None:
            ef, z = self._encode(batch, pos, constant)
        p = self.response_parameters(batch, ef, z)
        return response(p, pos / BOHR_ANG, batch.topo.bond_index.t(), batch.mol_idx, batch.topo.bond_frag, batch.n_mol)

    def forward(self, batch, positions=None, constant: bool = False, electrical: bool | None = None) -> dict:
        """``energy`` (M,) Hartree and the emitted bonded parameters; with the electrical head (default: when the
        config has one) also ``charges`` (N,), ``dipole`` (M, 3), ``second_moment`` and ``polarizability``
        (M, 3, 3) and the full ``solution``."""
        pos = batch.positions if positions is None else positions
        ef, z = self._encode(batch, pos, constant)
        e, (bp, tp) = self.bonded_energy(batch, pos, constant, z=z)
        out = {"energy": e + self.reference_energy(batch), "bonded": bp, "terms": tp}
        if self.has_electrical if electrical is None else electrical:
            sol = self.electrical(batch, pos, constant, ef=ef, z=z)
            out.update(solution=sol, charges=sol.charges, dipole=sol.dipole, second_moment=sol.second_moment,
                       polarizability=sol.polarizability)
        return out

    def energy_forces(self, batch, create_graph: bool = False, constant: bool = False):
        """(energy (M,) Hartree, forces (N, 3) Hartree/bohr)."""
        pos = batch.positions.detach().clone().requires_grad_(True)
        e = self.forward(batch, pos, constant, electrical=False)["energy"]
        (g,) = torch.autograd.grad(e.sum(), pos, create_graph=create_graph)
        return e, -g * BOHR_ANG

    @torch.no_grad()
    def fit_reference(self, batch) -> None:
        """Least-squares atomic reference energies e0 against ``batch.energy`` minus the present bonded energy."""
        t = self.tables.type_index(batch.z, batch.topo.degree)
        T = len(self.atom_types)
        comp = batch.positions.new_zeros(batch.n_mol, T).index_put((batch.mol_idx, t), torch.ones_like(batch.positions[:, 0]), accumulate=True)
        with torch.enable_grad():
            e = self.bonded_energy(batch)[0].detach()
        self.tables.e0.copy_(torch.linalg.lstsq(comp, (batch.energy - e)[:, None]).solution[:, 0])
