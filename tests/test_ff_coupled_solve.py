"""The coupled response solve: is the operator the gradient of the energy it reports?

Everything else here rests on :func:`test_grad_state_is_the_gradient_of_the_energy`. The
solver's matrix-vector product is *derived* from the energy expression
(``A x = grad_E(x) - grad_E(0)``), so that test is what pins the whole construction: if the
interaction tensor were contracted on the wrong index, or the change of variables
``mu = alpha u`` were transposed, the solve and the reported energy would still agree with each
other and both be wrong. Checking the gradient against autograd of an *independently written*
energy is the only place that cannot happen.

The dense oracle then checks the CG itself -- convergence, and the per-frame scalars that make
a frame's answer independent of which others shared its minibatch.
"""

from dataclasses import replace

import pytest
import torch

from rsfff.ff.coupled_solve import (
    CoupledSystem,
    _grad_state,
    _spherical_to_poly_map,
    coupled_energy,
    coupled_solve,
    coupled_solve_dense,
    multipoles_from_state,
    pcg,
    zero_state,
)
from rsfff.ff.damping import fermi_switch
from rsfff.ff.multipole import (
    build_polytensor,
    damped_interaction_tensor,
    multipole_pair_energy,
    slater_one_center_damp,
    slater_two_center_damp,
)
from rsfff.ff.units import BOHR_ANG
from rsfff.mlip.eem import atomic_dipoles, atomic_quadrupoles
from rsfff.mlip.sqe import atomic_dipole_energy, atomic_quadrupole_energy, sqe_solve

#: The tensors the solver treats as differentiable inputs, in the order it packs them.
PARAMS = (
    "chi", "eta", "q0", "compliance", "chivec", "alpha", "chiquad", "cquad",
    "t_point", "t_ss", "t_1c_i", "t_1c_j", "m_nuc",
)


@pytest.fixture(autouse=True)
def _float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def make_system(n_frag=2, max_rank=2, seed=0, gate=None, n_systems=1):
    """A small coupled system shaped like the one the model actually solves.

    The geometry matters more than it looks. Three atoms per fragment at bonded range, with
    fragment centers ~3.2 Angstrom apart, and the coupling carrying the **same Fermi range
    separation the real model applies** (:data:`rsfff.ff.range_priors.DEFAULT_R0_PRIOR`-ish
    midpoint, ``alpha = 40``). Randomly scattered atoms with every pair coupled at full
    strength is not a harder version of this problem, it is a different one: the functional
    stops being positive definite (measured: min eigenvalue -14 against +0.13 gated), because
    that is the polarization catastrophe. Gating short range off is what prevents it, and it
    is the reason the range separation is reused here rather than a separate damper being
    introduced.

    ``gate=<float>`` overrides the switch with a constant, which the catastrophe test uses.
    """
    g = torch.Generator().manual_seed(seed)
    n_at = 3 * n_frag
    # bonded-length atoms around centers on a line, i.e. a row of loose "molecules"
    centers = torch.zeros(n_frag, 3)
    centers[:, 0] = 3.2 * torch.arange(n_frag, dtype=torch.get_default_dtype())
    offs = torch.randn(n_at, 3, generator=g)
    offs = 0.96 * offs / offs.norm(dim=-1, keepdim=True)
    pos = centers.repeat_interleave(3, dim=0) + offs

    rows, cols = [], []
    for f in range(n_frag):
        o = 3 * f
        for a in range(3):
            for b in range(a + 1, 3):
                rows.append(o + a)
                cols.append(o + b)
    bond_index = torch.tensor([rows, cols], dtype=torch.long)

    pi = [(a, b) for a in range(n_at) for b in range(a + 1, n_at)]
    pair_index = torch.tensor(pi, dtype=torch.long).t().contiguous()

    z = torch.rand(n_at, generator=g) + 0.5
    b_el = torch.rand(n_at, generator=g) + 1.5
    i, j = pair_index[0], pair_index[1]
    dr = (pos[j] - pos[i]) / BOHR_ANG
    r_au = dr.norm(dim=-1)
    r_inv = 1.0 / r_au
    b_ij = (0.5 * (b_el[i].log() + b_el[j].log())).exp()
    r_ang = r_au * BOHR_ANG
    switch = (
        fermi_switch(r_ang, torch.tensor(1.75), torch.tensor(40.0))
        if gate is None else gate * torch.ones_like(r_ang)
    )
    w = switch[:, None, None]

    def tensor(damp):
        return w * damped_interaction_tensor(dr, damp, r_inv, max_rank=max_rank)

    alpha = None
    if max_rank >= 1:
        a = torch.randn(n_at, 3, 3, generator=g) * 0.3
        alpha = a @ a.transpose(-1, -2) + 0.5 * torch.eye(3)

    return CoupledSystem(
        n_systems=n_systems,
        n_atoms=n_at,
        batch_idx=torch.zeros(n_at, dtype=torch.long),
        bond_index=bond_index,
        bond_batch=torch.zeros(bond_index.shape[1], dtype=torch.long),
        pair_index=pair_index,
        max_rank=max_rank,
        chi=torch.randn(n_at, generator=g) * 0.1,
        eta=torch.rand(n_at, generator=g) + 0.5,
        q0=torch.randn(n_at, generator=g) * 0.1,
        compliance=torch.rand(bond_index.shape[1], generator=g) * 0.5,
        chivec=torch.randn(n_at, 3, generator=g) * 0.05 if max_rank >= 1 else None,
        alpha=alpha,
        chiquad=torch.randn(n_at, 5, generator=g) * 0.05 if max_rank >= 2 else None,
        cquad=torch.rand(n_at, generator=g) + 0.5 if max_rank >= 2 else None,
        t_point=tensor(None),
        t_ss=tensor(-slater_two_center_damp(b_ij * r_au, max_rank)),
        t_1c_i=tensor(-slater_one_center_damp(b_el[i] * r_au, max_rank)),
        t_1c_j=tensor(-slater_one_center_damp(b_el[j] * r_au, max_rank)),
        m_nuc=build_polytensor(z, None, None, max_rank=max_rank),
    )


def total_energy(sys, x, d_map):
    """Internal + pair energy at a state, written out independently of ``_grad_state``.

    The pair half is the same four-term contraction
    :func:`rsfff.ff.electrostatics.slater_elec_pair_energy` performs, with the tensors already
    carrying the gate -- so what the solver minimizes is provably the same expression the
    frozen electrostatics channel evaluates.
    """
    q, mu, theta = multipoles_from_state(sys, x, d_map)
    from rsfff.ff.multipole import spherical_to_cartesian_quadrupole

    m_real = build_polytensor(
        q,
        mu if mu.numel() else None,
        spherical_to_cartesian_quadrupole(theta) if theta.numel() else None,
        max_rank=sys.max_rank,
    )
    m_shell = m_real - sys.m_nuc
    i, j = sys.pair_index[0], sys.pair_index[1]
    e_pair = (
        multipole_pair_energy(m_real[i], m_real[j], sys.t_point)
        + multipole_pair_energy(m_shell[i], m_shell[j], sys.t_ss)
        + multipole_pair_energy(m_shell[i], sys.m_nuc[j], sys.t_1c_i)
        + multipole_pair_energy(sys.m_nuc[i], m_shell[j], sys.t_1c_j)
    )
    return coupled_energy(sys, x, d_map).sum() + e_pair.sum()


# ---------------------------------------------------------------------------
# the operator
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("max_rank", [1, 2])
def test_grad_state_is_the_gradient_of_the_energy(max_rank):
    """The load-bearing test: the solver's operator *is* d(reported energy)/dx.

    A transposed interaction tensor or a mis-pulled-back change of variables would leave the
    solver self-consistent and wrong. Autograd of an independently written energy is what
    rules that out.
    """
    sys = make_system(max_rank=max_rank, seed=max_rank)
    d_map = _spherical_to_poly_map(torch.float64, torch.device("cpu"))
    x = tuple(
        torch.randn_like(t) * 0.2 for t in zero_state(sys, torch.float64, torch.device("cpu"))
    )
    leaves = tuple(t.clone().requires_grad_(True) for t in x)
    ref = torch.autograd.grad(total_energy(sys, leaves, d_map), [t for t in leaves if t.numel()])
    got = [t for t in _grad_state(sys, x, d_map) if t.numel()]
    for a, b in zip(ref, got):
        assert torch.allclose(a, b, atol=1e-12), (a - b).abs().max()


@pytest.mark.parametrize("max_rank", [1, 2])
def test_solution_is_a_stationary_point(max_rank):
    sys = make_system(max_rank=max_rank, seed=max_rank + 10)
    d_map = _spherical_to_poly_map(torch.float64, torch.device("cpu"))
    x, _ = coupled_solve(sys, rtol=1e-13, atol=1e-15)
    g = _grad_state(sys, x, d_map)
    assert max(float(t.abs().max()) for t in g if t.numel()) < 1e-12


def _dense_operator(sys, d_map):
    """Materialize ``A`` column by column. Tests only; ``O(n)`` matvecs."""
    zero = zero_state(sys, torch.float64, torch.device("cpu"))
    sizes = [t.numel() for t in zero]
    n = sum(sizes)
    g0 = _grad_state(sys, zero, d_map)

    def un(flat):
        p = torch.split(flat, sizes)
        return p[0], p[1].reshape(-1, 3), p[2].reshape(-1, 5)

    flat = lambda s: torch.cat([t.reshape(-1) for t in s])  # noqa: E731
    eye = torch.eye(n, dtype=torch.float64)
    return torch.stack([flat(_grad_state(sys, un(eye[k]), d_map)) - flat(g0) for k in range(n)], 1)


def test_operator_is_symmetric_and_positive_definite_when_gated():
    """CG's two prerequisites, measured on the geometry the model actually sees."""
    d_map = _spherical_to_poly_map(torch.float64, torch.device("cpu"))
    a = _dense_operator(make_system(max_rank=2, seed=21), d_map)
    assert float((a - a.t()).abs().max()) < 1e-12
    assert float(torch.linalg.eigvalsh(0.5 * (a + a.t())).min()) > 0.0


def test_energy_is_minimized_not_merely_stationary():
    """Random perturbations off the solution can only raise the energy."""
    sys = make_system(max_rank=2, seed=21)
    d_map = _spherical_to_poly_map(torch.float64, torch.device("cpu"))
    x, _ = coupled_solve(sys, rtol=1e-13, atol=1e-15)
    e0 = float(total_energy(sys, x, d_map))
    for k in range(8):
        g = torch.Generator().manual_seed(k)
        pert = tuple(
            t + torch.randn(t.shape, generator=g, dtype=t.dtype) * 0.05 for t in x
        )
        assert float(total_energy(sys, pert, d_map)) > e0


def test_ungating_the_coupling_destroys_positive_definiteness():
    """The polarization catastrophe, made explicit so the guard is not mistaken for free.

    Removing the range separation from the coupling -- coupling *every* pair at full strength,
    including bonded ones -- drives the smallest eigenvalue of the functional strongly
    negative, so the "solution" becomes a saddle and the polarization energy is unbounded
    below. Reusing the electrostatic gate is what prevents this, and it is why the plan
    rejected introducing a separate polarization damper.

    Note what this test does *not* claim: that the solver would notice. CG only reports
    ``pd_fail`` if it happens to probe a direction of negative curvature, so an indefinite
    system can converge to a saddle and look healthy. Positive definiteness is protected
    structurally by the gate, not detected reliably at runtime -- which is why ``E_pol > 0``
    is the metric to watch during training.
    """
    d_map = _spherical_to_poly_map(torch.float64, torch.device("cpu"))
    def min_eig(sys):
        a = _dense_operator(sys, d_map)
        return float(torch.linalg.eigvalsh(0.5 * (a + a.t())).min())

    gated = min_eig(make_system(max_rank=2, seed=21))
    ungated = min_eig(make_system(max_rank=2, seed=21, gate=1.0))
    assert gated > 0.0, f"the gated functional must be convex, got min eigenvalue {gated:.3f}"
    assert ungated < 0.0, (
        f"ungating should make the functional indefinite; got {ungated:.3f}. If this passes, "
        f"the fixture geometry has drifted far enough apart that the catastrophe is out of "
        f"reach and the test no longer demonstrates anything"
    )


# ---------------------------------------------------------------------------
# the solver
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("max_rank", [1, 2])
def test_pcg_matches_the_dense_oracle(max_rank):
    sys = make_system(max_rank=max_rank, seed=max_rank + 20)
    dense = coupled_solve_dense(sys)
    got, info = pcg(sys, rtol=1e-13, atol=1e-15)
    assert bool(info.converged.all())
    assert not bool(info.pd_fail.any())
    for a, b in zip(dense, got):
        if a.numel():
            assert torch.allclose(a, b, atol=1e-10), (a - b).abs().max()


def test_a_frame_is_unaffected_by_the_rest_of_its_batch():
    """Per-frame CG scalars. A shared step size would couple frames through the solver.

    This is the bug a batched CG invites and no physics test would catch: the answer would
    still be a plausible polarization, just one that depended on the minibatch composition.
    """
    a = make_system(n_frag=2, max_rank=2, seed=11)
    b = make_system(n_frag=3, max_rank=2, seed=12)
    alone, _ = coupled_solve(a, rtol=1e-13, atol=1e-15)

    n_a, nb_a = a.n_atoms, a.bond_index.shape[1]
    merged = replace(
        a,
        n_systems=2,
        n_atoms=a.n_atoms + b.n_atoms,
        batch_idx=torch.cat([a.batch_idx, b.batch_idx + 1]),
        bond_index=torch.cat([a.bond_index, b.bond_index + n_a], dim=1),
        bond_batch=torch.cat([a.bond_batch, b.bond_batch + 1]),
        pair_index=torch.cat([a.pair_index, b.pair_index + n_a], dim=1),
        **{n: torch.cat([getattr(a, n), getattr(b, n)]) for n in PARAMS},
    )
    together, _ = coupled_solve(merged, rtol=1e-13, atol=1e-15)
    assert torch.allclose(together[0][:nb_a], alone[0], atol=1e-11)
    assert torch.allclose(together[1][:n_a], alone[1], atol=1e-11)
    assert torch.allclose(together[2][:n_a], alone[2], atol=1e-11)


# ---------------------------------------------------------------------------
# gradients
# ---------------------------------------------------------------------------

def _grads(sys, dense, loss_kind, d_map):
    leaves = {n: getattr(sys, n).clone().requires_grad_(True) for n in PARAMS}
    live = replace(sys, **leaves)
    x = coupled_solve_dense(live) if dense else coupled_solve(live, rtol=1e-14, atol=1e-16)[0]
    if loss_kind == "nonvariational":
        # What the bond correction looks like to the solve: a function of the converged
        # multipoles that is *not* stationary in them.
        q, mu, theta = multipoles_from_state(live, x, d_map)
        loss = q.pow(3).sum() + (mu * mu).sum().sqrt() + theta.tanh().sum()
    else:
        # the *whole* functional, which is what x is stationary in -- E_internal alone is not
        loss = total_energy(live, x, d_map)
    g = torch.autograd.grad(loss, list(leaves.values()), allow_unused=True)
    return dict(zip(PARAMS, g))


@pytest.mark.parametrize("loss_kind", ["nonvariational", "energy"])
def test_adjoint_backward_matches_autograd_through_the_dense_solve(loss_kind):
    """§6.2. ``nonvariational`` is the case that needs the adjoint at all.

    With an energy loss the gradient is Hellmann-Feynman and would survive detaching the
    solve entirely; with a non-variational consumer of ``M*`` it would not, which is exactly
    the failure ``docs/range_separated_mlip.md`` §6.1 names.
    """
    d_map = _spherical_to_poly_map(torch.float64, torch.device("cpu"))
    sys = make_system(max_rank=2, seed=5)
    adjoint = _grads(sys, False, loss_kind, d_map)
    dense = _grads(sys, True, loss_kind, d_map)
    for name in PARAMS:
        if adjoint[name] is None or dense[name] is None:
            continue
        scale = max(float(dense[name].abs().max()), 1e-10)
        err = float((adjoint[name] - dense[name]).abs().max()) / scale
        assert err < 1e-8, f"{name}: relative {err:.2e}"


def test_detaching_the_solve_is_only_safe_for_the_energy():
    """The demonstration behind §6.1, measured rather than asserted.

    For an energy loss, stationarity means the solve may be detached and the forces are still
    right. For anything else it is not, and the difference is not small -- so the adjoint is
    load-bearing the moment the environment features reach the bond head.
    """
    d_map = _spherical_to_poly_map(torch.float64, torch.device("cpu"))
    sys = make_system(max_rank=2, seed=31)

    def detached(loss_kind):
        leaves = {n: getattr(sys, n).clone().requires_grad_(True) for n in PARAMS}
        live = replace(sys, **leaves)
        with torch.no_grad():
            x, _ = pcg(replace(sys), rtol=1e-14, atol=1e-16)
        x = tuple(t.detach() for t in x)
        if loss_kind == "energy":
            loss = total_energy(live, x, d_map)
        else:
            q, mu, theta = multipoles_from_state(live, x, d_map)
            loss = q.pow(3).sum() + (mu * mu).sum().sqrt() + theta.tanh().sum()
        return dict(
            zip(PARAMS, torch.autograd.grad(loss, list(leaves.values()), allow_unused=True))
        )

    exact_e = _grads(sys, True, "energy", d_map)
    got_e = detached("energy")
    worst_e = max(
        float((exact_e[n] - got_e[n]).abs().max()) / max(float(exact_e[n].abs().max()), 1e-10)
        for n in PARAMS
        if exact_e[n] is not None and got_e[n] is not None
    )
    assert worst_e < 1e-8, f"stationarity should make the energy gradient exact, got {worst_e:.2e}"

    exact_n = _grads(sys, True, "nonvariational", d_map)
    got_n = detached("nonvariational")
    worst_n = max(
        float((exact_n[n] - got_n[n]).abs().max()) / max(float(exact_n[n].abs().max()), 1e-10)
        for n in PARAMS
        if exact_n[n] is not None and got_n[n] is not None
    )
    assert worst_n > 1e-3, (
        "detaching should visibly break a non-variational consumer; if this passes the test "
        "has stopped exercising the adjoint"
    )


def test_closed_channels_and_zero_polarizabilities_stay_finite():
    """``alpha -> 0``, ``c -> 0``, ``s -> 0``: the limits the rescaling exists to protect.

    Under the unrescaled form these are ``alpha^-1 -> inf``; here they are a zero row with a
    zero right-hand side, so both the value and every gradient stay finite. This is the
    property force training needs at a closed channel.
    """
    d_map = _spherical_to_poly_map(torch.float64, torch.device("cpu"))
    sys = make_system(max_rank=2, seed=7)
    leaves = {n: getattr(sys, n).clone().requires_grad_(True) for n in PARAMS}
    live = replace(
        sys,
        **{
            **leaves,
            "alpha": leaves["alpha"] * 0.0,
            "cquad": leaves["cquad"] * 0.0,
            "compliance": leaves["compliance"] * 0.0,
        },
    )
    x, _ = coupled_solve(live, rtol=1e-13, atol=1e-15)
    q, mu, theta = multipoles_from_state(live, x, d_map)
    assert bool((mu == 0).all()) and bool((theta == 0).all())
    assert torch.allclose(q, live.q0, atol=1e-14)

    g = torch.autograd.grad(
        coupled_energy(live, x, d_map).sum(),
        [leaves["alpha"], leaves["cquad"], leaves["compliance"]],
        allow_unused=True,
    )
    for name, gi in zip(("alpha", "cquad", "compliance"), g):
        assert gi is None or bool(torch.isfinite(gi).all()), name


# ---------------------------------------------------------------------------
# the reduction that makes `E_pol` a difference of the same functional
# ---------------------------------------------------------------------------

def test_uncoupled_limit_reproduces_fragment_response_exactly():
    """With the gate at zero the coupled solve *is* :class:`FragmentResponse`.

    Not approximately: the multipoles and the internal energy agree to round-off, because the
    functional is the same one with the pair block removed. That identity is what makes
    ``E_pol = E_1 - E_0`` a pure relaxation rather than a difference of two different models.
    """
    d_map = _spherical_to_poly_map(torch.float64, torch.device("cpu"))
    sys = make_system(max_rank=2, seed=3, gate=0.0)
    x, _ = coupled_solve(sys, rtol=1e-14, atol=1e-16)
    q, mu, theta = multipoles_from_state(sys, x, d_map)

    sol = sqe_solve(
        sys.chi, sys.eta, sys.compliance, sys.q0,
        torch.zeros(sys.n_atoms, 3),
        sys.bond_index, sys.batch_idx, sys.bond_batch, sys.n_systems,
        field=None, with_polarizability=False,
    )
    e_ref = (
        sol.energy
        + atomic_dipole_energy(sys.chivec, sys.alpha, sys.batch_idx, sys.n_systems, None)
        + atomic_quadrupole_energy(sys.chiquad, sys.cquad, sys.batch_idx, sys.n_systems, None)
    )
    assert torch.allclose(q, sol.charges, atol=1e-13)
    assert torch.allclose(mu, atomic_dipoles(sys.chivec, sys.alpha, sys.batch_idx, None), atol=1e-13)
    assert torch.allclose(
        theta, atomic_quadrupoles(sys.chiquad, sys.cquad, sys.batch_idx, None), atol=1e-13
    )
    assert torch.allclose(coupled_energy(sys, x, d_map), e_ref, atol=1e-13)


def test_charge_is_conserved_per_channel_component():
    """SQE's structural guarantee survives the coupling, for any compliance."""
    sys = make_system(n_frag=3, max_rank=2, seed=13)
    d_map = _spherical_to_poly_map(torch.float64, torch.device("cpu"))
    x, _ = coupled_solve(sys, rtol=1e-13, atol=1e-15)
    q, _, _ = multipoles_from_state(sys, x, d_map)
    frag = torch.arange(3).repeat_interleave(3)
    got = q.new_zeros(3).index_add_(0, frag, q)
    want = sys.q0.new_zeros(3).index_add_(0, frag, sys.q0)
    assert torch.allclose(got, want, atol=1e-13)


# ---------------------------------------------------------------------------
# The direct parameterization: mu0 instead of chivec
# ---------------------------------------------------------------------------

def _as_direct(sys):
    """The same functional written with permanent multipoles: ``mu0 = -alpha chivec``."""
    from dataclasses import replace

    from rsfff.ff.coupled_solve import _apply_cquad

    mu0 = -torch.einsum("nab,nb->na", sys.alpha, sys.chivec)
    quad0 = -_apply_cquad(sys.cquad, sys.chiquad)
    return replace(sys, chivec=None, chiquad=None, mu0=mu0, quad0=quad0)


def test_direct_grad_state_is_still_the_gradient_of_the_energy():
    """The invariant the whole module rests on has to survive the change of variables."""
    sys = _as_direct(make_system(seed=3, max_rank=2))
    d_map = _spherical_to_poly_map(torch.float64, sys.chi.device)
    g = torch.Generator().manual_seed(11)
    zero = zero_state(sys, torch.float64, sys.chi.device)
    x = tuple(
        (t + 0.05 * torch.randn(t.shape, generator=g)).requires_grad_(True) for t in zero
    )
    auto = torch.autograd.grad(total_energy(sys, x, d_map).sum(), x, allow_unused=True)
    manual = _grad_state(sys, tuple(t.detach() for t in x), d_map)
    for name, a, m in zip("vuw", auto, manual):
        if a is None or not a.numel():
            continue
        assert float((a - m).abs().max()) < 1e-12, name


def test_direct_and_drive_parameterizations_are_the_same_functional():
    """``mu0 = -alpha chivec`` is a change of variables, so the physics must be identical.

    Three things have to agree, and the third is the one that matters for the staged fit:
    the converged multipoles, and ``E_pol`` measured against each form's *own* uncoupled
    reference. That reference is not the same state in the two forms -- under ``chivec`` the
    uncoupled minimum is ``u = -chivec`` while under ``mu0`` it is ``u = 0`` -- and moving it
    to the solver's origin is precisely what empties the on-site sectors out of
    ``internal_energy``.
    """
    drive = make_system(seed=3, max_rank=2)
    direct = _as_direct(drive)
    d_map = _spherical_to_poly_map(torch.float64, drive.chi.device)

    (x_drive, _), (x_direct, _) = coupled_solve(drive), coupled_solve(direct)
    m_drive = multipoles_from_state(drive, x_drive, d_map)
    m_direct = multipoles_from_state(direct, x_direct, d_map)
    for name, a, b in zip(("q", "mu", "Theta"), m_drive, m_direct):
        assert float((a - b).abs().max()) < 1e-9, f"{name} differs between the two forms"

    unc_drive = (torch.zeros_like(x_drive[0]), -drive.chivec, -drive.chiquad)
    unc_direct = zero_state(direct, torch.float64, direct.chi.device)
    pol_drive = float(
        (total_energy(drive, x_drive, d_map) - total_energy(drive, unc_drive, d_map)).sum()
    )
    pol_direct = float(
        (total_energy(direct, x_direct, d_map) - total_energy(direct, unc_direct, d_map)).sum()
    )
    assert abs(pol_drive - pol_direct) < 1e-11, (
        f"E_pol differs: {pol_drive:.12f} vs {pol_direct:.12f}"
    )
    # And the constant that moved: E_internal loses exactly -1/2 chi^T a chi.
    from rsfff.ff.coupled_solve import _apply_cquad

    predicted = float(
        -0.5 * torch.einsum("na,nab,nb->n", drive.chivec, drive.alpha, drive.chivec).sum()
        - 0.5 * (drive.chiquad * _apply_cquad(drive.cquad, drive.chiquad)).sum()
    )
    observed = float(
        (coupled_energy(drive, x_drive, d_map) - coupled_energy(direct, x_direct, d_map)).sum()
    )
    assert abs(predicted - observed) < 1e-11, (
        f"the energy that left the on-site sectors is {observed:.10f}, expected {predicted:.10f}"
    )
