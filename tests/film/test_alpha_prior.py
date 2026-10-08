"""Per-element alpha priors for the monatomic ions (``film.alpha_prior``).

The atomic-alpha head starts every element at ``softplus(0)`` = 0.693 a0^3 and cannot learn its
way to Cl-'s 31 a0^3 (an Adam step moves its readout by ~lr). With the prior the head's output
multiplies ``alpha_prior / ln 2``, so:

* a fresh NaCl model's lone ions sit near their priors (0.961 and 31.315 a0^3);
* elements without a prior (water, HF, NH3) are bit-identical with the switch on or off;
* a strict load of a checkpoint written before the buffer existed reproduces it (scale 1),
  while a warm start keeps the prior.
"""

from __future__ import annotations

import torch

from rsfff.ff.response import DEFAULT_ALPHA_PRIOR, alpha_scale_prior
from rsfff.ff.units import BOHR_ANG
from rsfff.train.term_loop import warm_start

from test_new_species import _batch, _model, hf_dimer

NACL_REF = [-161.9, -460.3]


def _ion(z: int, q: float):
    return _batch([torch.zeros(1, 3)], [z], [q], [1])


def _alpha_iso(model, batch) -> float:
    out = model(batch, with_polarizability=True, with_induction=False)
    return float(out.polarizability[0].detach().diagonal().mean()) / BOHR_ANG ** 2


def test_scale_table():
    s = alpha_scale_prior([1, 8, 11, 17])
    assert torch.allclose(s[:2], torch.ones(2))
    assert torch.allclose(s[2:] * torch.log(torch.tensor(2.0)), torch.tensor([0.961, 31.315]))


def test_fresh_ions_start_near_their_priors():
    model = _model([11, 17], NACL_REF)
    for z, q in ((11, 1.0), (17, -1.0)):
        a = _alpha_iso(model, _ion(z, q))
        assert abs(a / DEFAULT_ALPHA_PRIOR[z] - 1.0) < 0.25, (z, a)


def test_without_prior_the_ions_start_at_the_attractor():
    model = _model([11, 17], NACL_REF, alpha_prior=False)
    assert torch.equal(model.network.response_heads.alpha_scale, torch.ones(2))
    assert _alpha_iso(model, _ion(17, -1.0)) < 2.0


def test_elements_without_a_prior_are_unchanged():
    on = _model([1, 9], [-0.5, -99.7])
    off = _model([1, 9], [-0.5, -99.7], alpha_prior=False)
    a_on = on(hf_dimer(), with_polarizability=True, with_induction=False).polarizability
    a_off = off(hf_dimer(), with_polarizability=True, with_induction=False).polarizability
    assert torch.equal(a_on, a_off)


def test_strict_load_of_an_old_checkpoint_reproduces_it():
    old = _model([11, 17], NACL_REF, alpha_prior=False)
    state = {k: v for k, v in old.state_dict().items() if not k.endswith("alpha_scale")}
    new = _model([11, 17], NACL_REF)
    new.load_state_dict(state, strict=True)
    assert torch.equal(new.network.response_heads.alpha_scale, torch.ones(2))
    b = _ion(17, -1.0)
    assert _alpha_iso(new, b) == _alpha_iso(old, b)


def test_warm_start_keeps_the_prior(tmp_path):
    old = _model([11, 17], NACL_REF, alpha_prior=False)
    state = {k: v for k, v in old.state_dict().items() if not k.endswith("alpha_scale")}
    path = tmp_path / "old.pt"
    torch.save({"model_state": state}, path)
    new = _model([11, 17], NACL_REF, seed=5)
    warm_start(new, str(path))
    scale = new.network.response_heads.alpha_scale
    assert torch.allclose(scale, alpha_scale_prior([11, 17]).to(scale.dtype))
    a = _alpha_iso(new, _ion(17, -1.0))
    assert abs(a / DEFAULT_ALPHA_PRIOR[17] - 1.0) < 0.25
