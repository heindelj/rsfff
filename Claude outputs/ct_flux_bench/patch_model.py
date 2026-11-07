"""Make the atomic-space flux solve the default in rsfff (model.py, train_ionct.py); keep the channel-space solve as
flux_operator_channel / flux_solve_channel. Usage: python patch_model.py <rsfff root>"""
import sys
from pathlib import Path

root = Path(sys.argv[1])
mp = root / "src/ff/complete/model.py"
s = mp.read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:80], s.count(old))
    s = s.replace(old, new)


# -- module docstring: how the joint minimum is solved --
rep("""Everything else -- the field, other channels (shared atoms through H), the induced moments and through them xpol --
enters linearly, and the joint minimum is one solve: with A = M + H_xpol and W = A^-1 C (extra right-hand sides of the
xpol solve), S = diag(b) + G^T H G - C^T W (the Schur complement, solved in q_c / s_c with s_c = sqrt(2|e_c|) / V0_c, which
keeps vanishing channels at the identity), b~ = -V + dphi_ext + C^T z_pol:

    q* = -S^-1 b~,   E_CT = E_joint - E_pol = 1/2 b~ . q*,   z* = z_pol - W q*
""", """Everything else -- the field, other channels (shared atoms through H), the induced moments and through them xpol --
enters linearly, and the joint minimum is one solve. The channels couple to each other and to everything else only
through the atomic charge changes Q = G q (the only per-channel pieces are b_c and V_c), so they are eliminated exactly
(``flux_solver="atomic"``, the default): stationarity in q gives

    q = w (V - G^T psi),   w_c = 1 / b_c = s_c^2,   psi = H Q + C^T z + phi_ext (the potential on the charges)
    Q = u - K psi,         K = G diag(w) G^T (an N x N weighted graph Laplacian),   u = G w V

and with A = M + H_xpol, W = A^-1 C (N extra right-hand sides of the xpol solve, C the atomic charges' coupling) and
S = H - C^T W (N x N) the joint minimum is

    (1 + K S) Q = u - K (phi_ext + C^T z_pol),   z* = z_pol - W Q,   E_CT = E_joint - E_pol = 1/2 b~ . q*,
    b~ = -V + G^T (phi_ext + C^T z_pol)

1 + K S is nonsingular (K PSD, S PD) and a vanishing channel (w_c -> 0) drops out of K and u, so the cost is the xpol
LU, N right-hand sides and an N x N solve, plus O(C) scatters: independent of the channel count. ``flux_solver="channel"``
keeps the original channel-space form for reference (identical to ~1e-14): W = A^-1 G-mapped couplings with one
right-hand side per channel and the Schur complement S = diag(b) + G^T H G - C^T W over the channels (solved in
q_c / s_c with s_c = sqrt(2|e_c|) / V0_c), q* = -S^-1 b~ -- (D N)^2 C + C^2 D N + C^3, with C = N (N - 1) for a cluster
inside the 8 A CT cutoff (5434 channels for 125 ions: 3.0 s vs 0.39 s per CPU forward).
""")

# -- constructor --
rep('xpol_perp_weight="pauli", xpol_kappa_env="scalefree", flux_v0_init=FLUX_V0_INIT):',
    'xpol_perp_weight="pauli", xpol_kappa_env="scalefree", flux_v0_init=FLUX_V0_INIT,\n'
    '                 flux_solver="atomic"):')
rep("""        if self.ct_model not in ("direct", "gap", "flux"):
            raise ValueError(f"ct_model {ct_model!r}")
""", """        if self.ct_model not in ("direct", "gap", "flux"):
            raise ValueError(f"ct_model {ct_model!r}")
        self.flux_solver = str(flux_solver)       # flux CT: "atomic" (N x N, default) or "channel" (reference)
        if self.flux_solver not in ("atomic", "channel"):
            raise ValueError(f"flux_solver {flux_solver!r}")
""")

# -- the flux operators and solves --
a = s.index("    # -- charge-flux CT --\n")
b = s.index("    @staticmethod\n    def flux_reverse(i, j, n):")
c = s.index("    def flux_solve(self, bt, Cz, Hq, s, V, G, ch, z_pol, W, phi_ext):")
d = s.index("    def xpol_solve_periodic(")
old_op = s[a:b]
reverse = s[b:c]
old_solve = s[c:d]
assert "Hq = torch.einsum" in old_op and "Sm = torch.eye(Cmax" in old_solve

new = '''    # -- charge-flux CT --
    def flux_operator(self, bt, sp, q, D, quadrupoles=True):
        """The charge block of the flux CT for clusters, in the form ``flux_solver`` selects: a tuple
        (C, H, s, V, G, ch, e_c) whose first element is the extra right-hand sides of the xpol solve and which
        :meth:`flux_solve` takes back. See :meth:`flux_operator_atomic` and :meth:`flux_operator_channel`."""
        if self.flux_solver == "channel":
            return self.flux_operator_channel(bt, sp, q, D, quadrupoles)
        return self.flux_operator_atomic(bt, sp, q, D, quadrupoles)

    def flux_solve(self, bt, Cz, Hq, s, V, G, ch, z_pol, W, phi_ext):
        """The joint minimum, given the xpol solution z_pol (B, nmax D) and W = (A + H)^-1 C for the operator of
        :meth:`flux_operator`: per frame E_CT (B,), per channel its share 1/2 b~_c q_c (C,) and charge q_c (C,), the
        joint moments z (B, nmax D) and the atomic charge changes (B, nmax)."""
        if self.flux_solver == "channel":
            return self.flux_solve_channel(bt, Cz, Hq, s, V, G, ch, z_pol, W, phi_ext)
        return self.flux_solve_atomic(bt, Cz, Hq, s, V, G, ch, z_pol, W, phi_ext)

    def _flux_atoms(self, bt, sp, D, quadrupoles=True):
        """Atomic charge slots: C (B, nmax, D, nmax) the coupling of a unit charge at k to i's site variables
        ([l, a, k]: energy Q_k C[l, a, k] z_l,a) and H (B, nmax, nmax) the slots' Coulomb with the on-site hardness
        (shape self-energy + eta) on the diagonal; padded atoms have zero rows."""
        B, nmax = bt["B"], bt["nmax"]
        fr, loc = self._local(bt)
        if self._flux_table is None:
            self._flux_table = FluxTable(self.tab)
        tbl = self._flux_table
        if tbl.device != bt["pos"].device:
            tbl.to(bt["pos"].device)
        X = bt["pos"] / K.BOHR_ANG
        pairs = K.all_pairs(bt["frame"])
        dp = torch.cat([pairs, pairs.flip(0)], 1)
        a_, b_ = dp
        Kp, Jp = flux_pair_terms(tbl, sp, dp, X[b_] - X[a_])
        Kd = X.new_zeros(B, nmax, nmax, 8).index_put((fr[a_], loc[a_], loc[b_]), Kp)    # [l, k]: on l of a charge at k
        eta = self.flux_jself[sp] + nn.functional.softplus(self.flux_eta[sp])
        Jd = X.new_zeros(B, nmax, nmax).index_put((fr[a_], loc[a_], loc[b_]), Jp).index_put((fr, loc, loc), eta)
        Ca = Kd.permute(0, 1, 3, 2)                                                     # (B, nmax, 8, nmax)
        if not quadrupoles:
            Ca = torch.cat([Ca[:, :, :3], torch.zeros_like(Ca[:, :, 3:])], 2)
        if D > 8:
            Ca = torch.cat([Ca, Ca.new_zeros(B, nmax, D - 8, nmax)], 2)
        return Ca, Jd

    def _flux_drives(self, bt, sp, q, h):
        """Per channel, given h_c = (G^T H G)_cc (C,): the kernel energy e_c <= 0, the drive V_c (with the pair
        factor t_p) and the variable scale s_c = sqrt(2 |e_c|) / V0_c = 1 / sqrt(b_c)."""
        i, j = bt["ch_index"]
        bD, bA = self.ct_exponents()
        rb = bt["ch_r"] / K.BOHR_ANG
        f = slater_two_center_damp_unequal(bD[sp[i]], bA[sp[j]], rb, 0)[0]
        e_c = -q[i, 0] * q[j, 1] * f / rb * torch.exp(bt["ch_ltaper"])          # <= 0
        ae = (-e_c).clamp(min=0.0)
        v0 = torch.exp(self.flux_lnv)
        V0 = v0[sp[i]] + v0[sp[j]]
        # the pair's two directions c = i -> j and c' = j -> i see the same h and are coupled by -h (opposite charge
        # patterns): their drives V_c = t_p sqrt(V0_c^2 + 2 |e_c| h) with the one factor t_p that makes the pair's own
        # 2x2 minimum exactly e_c + e_c'. In the compliances c = 1 / (b + h):
        #     t^2 = S (1 - h^2 c c') / (S + 2 h sqrt(|e| c |e'| c')),   S = |e| + |e'|
        Vh2 = V0 * V0 + 2 * ae * h
        cc = 2 * ae / Vh2
        rev = self.flux_reverse(i, j, sp.shape[0])
        has = rev >= 0
        rv = rev.clamp(min=0)
        ae2 = torch.where(has, ae[rv], torch.zeros_like(ae))
        cc2 = torch.where(has, cc[rv], torch.zeros_like(cc))
        S_ = ae + ae2
        t2 = S_ * (1 - h * h * cc * cc2) / (S_ + 2 * h * torch.sqrt(ae * cc * ae2 * cc2 + 1e-300) + 1e-300)
        t2 = torch.where(has, t2, torch.ones_like(t2))
        Vc = torch.sqrt(t2 * Vh2)
        sc = torch.sqrt(2 * ae + 1e-30) / V0
        return e_c, Vc, sc

    def flux_operator_atomic(self, bt, sp, q, D, quadrupoles=True):
        """The flux block in atomic-charge space. Returns C (B, nmax D, nmax) the atomic charges' coupling to the
        induced moments, H (B, nmax, nmax), s (C,) and V (C,) per channel, None (no charge-pattern matrix), the
        channels' (frame, donor slot, acceptor slot) and their kernel energies e_c (C,)."""
        B, nmax = bt["B"], bt["nmax"]
        fr, loc = self._local(bt)
        Ca, Jd = self._flux_atoms(bt, sp, D, quadrupoles)
        i, j = bt["ch_index"]
        cf = bt["ch_frame"]
        li, lj = loc[i], loc[j]
        h = Jd[cf, li, li] + Jd[cf, lj, lj] - Jd[cf, li, lj] - Jd[cf, lj, li]     # (G^T H G)_cc by gather
        e_c, Vc, sc = self._flux_drives(bt, sp, q, h)
        return Ca.reshape(B, nmax * D, nmax), Jd, sc, Vc, None, (cf, li, lj), e_c

    def flux_solve_atomic(self, bt, Ca, Ha, s, V, _G, ch, z_pol, W, phi_ext):
        """The joint minimum with the channels eliminated (module docstring): K = G diag(s^2) G^T, u = G s^2 V,
        (1 + K S) Q = u - K psi0 with S = H - C^T W, psi0 = phi_ext + C^T z_pol; q = s^2 (V - G^T psi)."""
        cf, li, lj = ch
        B, N = Ha.shape[0], Ha.shape[1]
        fr, loc = self._local(bt)
        w = s * s                                                      # compliance 1 / b_c
        Kl = Ha.new_zeros(B, N, N)
        Kl = Kl.index_put((cf, li, li), w, accumulate=True).index_put((cf, lj, lj), w, accumulate=True)
        Kl = Kl.index_put((cf, li, lj), -w, accumulate=True).index_put((cf, lj, li), -w, accumulate=True)
        wV = w * V
        u = Ha.new_zeros(B, N).index_put((cf, li), wV, accumulate=True).index_put((cf, lj), -wV, accumulate=True)
        psi0 = Ha.new_zeros(B, N).index_put((fr, loc), phi_ext) + torch.einsum("bnk,bn->bk", Ca, z_pol)
        S = Ha - Ca.transpose(1, 2) @ W                                # (B, N, N)
        M = torch.eye(N, dtype=Ha.dtype, device=Ha.device) + Kl @ S
        Q = torch.linalg.solve(M, (u - (Kl @ psi0[..., None])[..., 0])[..., None])[..., 0]
        psi = psi0 + (S @ Q[..., None])[..., 0]
        qc = w * (V - (psi[cf, li] - psi[cf, lj]))
        e_ch = 0.5 * (-V + psi0[cf, li] - psi0[cf, lj]) * qc          # 1/2 b~_c q_c
        z = z_pol - (W @ Q[..., None])[..., 0]
        return Ha.new_zeros(B).index_add(0, cf, e_ch), e_ch, qc, z, Q

    def flux_operator_channel(self, bt, sp, q, D, quadrupoles=True):
        """The charge block of the flux CT in channel space (reference form, ``flux_solver="channel"``). Returns C
        (B, nmax D, Cmax) the charges' coupling to the induced moments (energy q^T C^T z), Hq = G^T H G (B, Cmax,
        Cmax), s (B, Cmax) the variable scales sqrt(2 |e_c|) / V0_c, V (B, Cmax) the drives, G (B, nmax, Cmax) the
        charge pattern, the channels' (frame, slot) and their kernel energies e_c (C,); padded channel slots have
        s = 0, V = 0."""
        B, nmax = bt["B"], bt["nmax"]
        fr, loc = self._local(bt)
        Ca, Jd = self._flux_atoms(bt, sp, D, quadrupoles)
        i, j = bt["ch_index"]
        cf = bt["ch_frame"]
        nC = i.shape[0]
        cnt = torch.bincount(cf, minlength=B)
        Cmax = int(cnt.max()) if nC else 0
        order = torch.argsort(cf, stable=True)
        slot = torch.empty_like(cf)
        slot[order] = torch.arange(nC, device=cf.device) - (torch.cumsum(cnt, 0) - cnt)[cf[order]]
        one = Jd.new_ones(nC)
        G = Jd.new_zeros(B, nmax, Cmax).index_put((cf, loc[i], slot), one, accumulate=True)
        G = G.index_put((cf, loc[j], slot), -one, accumulate=True)
        Hq = torch.einsum("bkc,bkl,bld->bcd", G, Jd, G)
        Cq = torch.einsum("blak,bkc->blac", Ca, G)                         # (B, nmax, D, Cmax)
        e_c, Vc, sc = self._flux_drives(bt, sp, q, Hq[cf, slot, slot])
        s_pad = Jd.new_zeros(B, Cmax).index_put((cf, slot), sc)
        V_pad = Jd.new_zeros(B, Cmax).index_put((cf, slot), Vc)
        return Cq.reshape(B, nmax * D, Cmax), Hq, s_pad, V_pad, G, (cf, slot), e_c

'''
new_solve = old_solve.replace("    def flux_solve(self, bt, Cz, Hq, s, V, G, ch, z_pol, W, phi_ext):\n        \"\"\"The joint minimum",
                              "    def flux_solve_channel(self, bt, Cz, Hq, s, V, G, ch, z_pol, W, phi_ext):\n        \"\"\"Reference form (``flux_solver=\"channel\"``): the joint minimum", 1)
assert new_solve != old_solve
s = s[:a] + new + reverse + new_solve + s[d:]
mp.write_text(s)

# -- train_ionct: the flag and build_model --
tp = root / "src/train/train_ionct.py"
t = tp.read_text()
o1 = 'ct_model=g("ct_model", "direct"), head_frontier=g("head_frontier", None),'
assert t.count(o1) == 1
t = t.replace(o1, 'ct_model=g("ct_model", "direct"), flux_solver=g("flux_solver", "atomic"),\n'
                  '                             head_frontier=g("head_frontier", None),')
o2 = '    ap.add_argument("--ct-model", default="direct", choices=("direct", "gap", "flux"),'
assert t.count(o2) == 1
t = t.replace(o2, '    ap.add_argument("--flux-solver", default="atomic", choices=("atomic", "channel"),\n'
                  '                    help="flux CT: atomic = channels eliminated, an N x N solve in atomic charges (default); "\n'
                  '                         "channel = the original Schur complement over the channels (reference; identical "\n'
                  '                         "results, cost ~ C^3 with C ~ N^2 channels)")\n' + o2)
tp.write_text(t)
print("patched", mp, tp)
