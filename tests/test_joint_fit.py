"""Shared exponents between the permanent and induced models (rsfff.qcgen.joint_fit)."""

import numpy as np
import torch

from rsfff.qcgen import elst_fit as EF, joint_fit as J, pol_fit as PF


def test_induced_exponent_follows_the_outer_permanent_shell():
    cfg = J.JointConfig(perm=EF.FitConfig(max_rank=0, n_shells={17: 2}, z={17: 11.0}, b_min=1.2),
                        pol=PF.PolConfig(n_shape=2), free_ratio=(17,))
    jf = J.JointFit([17], -1.0, np.zeros((1, 3)), cfg, alpha_ref=31.0 * np.eye(3))
    with torch.no_grad():
        jf.perm.rb.add_(torch.randn_like(jf.perm.rb))
    sp = jf.sync()
    b, _ = sp.shells()
    assert torch.allclose(jf.pol.b(), b[:, 0]) and float(b[0, 0]) < float(b[0, 1])
    with torch.no_grad():
        jf.log_ratio.fill_(np.log(1.3))
    jf.sync()
    assert torch.allclose(jf.pol.b(), 1.3 * b[:, 0])
    # the tie is differentiable: a response loss reaches the permanent exponents
    P = torch.tensor(np.random.default_rng(0).normal(size=(20, 3)) * 4)
    phi, _ = PF.response_potential_field(torch.zeros(1, 3, dtype=torch.float64), jf.pol.site_alpha(), jf.pol.b(), jf.pol.w(), P)
    phi.pow(2).sum().backward()
    assert jf.perm.rb.grad is not None and torch.isfinite(jf.perm.rb.grad).all() and float(jf.perm.rb.grad.abs().sum()) > 0
