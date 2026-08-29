"""Phase-2 tests: the conditioned trunk's initialization and mode dispatch."""

from __future__ import annotations

import pytest
import torch

from rsfff.ff.film.conditioning import CONDITIONING_MODES, ConditionedTrunk, FiLMGenerator


def test_film_is_identity_at_init():
    """Zero-init generators make 'film' bit-equal to 'none' with the same layer weights."""
    torch.manual_seed(3)
    film = ConditionedTrunk(10, 16, 2, d_c=3, mode="film")
    plain = ConditionedTrunk(10, 16, 2, d_c=3, mode="none")
    with torch.no_grad():
        for a, b in zip(plain.layers, film.layers):
            a.linear.weight.copy_(b.linear.weight)
            a.linear.bias.copy_(b.linear.bias)

    x = torch.randn(5, 10)
    c = torch.randn(5, 3)
    assert torch.equal(film(x, c), plain(x, None))

    # And after perturbing a generator readout the conditioning is live.
    with torch.no_grad():
        film.generators[0].net[-1].weight.add_(0.1)
    assert not torch.allclose(film(x, c), film(x, torch.zeros_like(c)))


def test_generators_are_decay_exempt():
    trunk = ConditionedTrunk(4, 8, 2, d_c=2, mode="film")
    for gen in trunk.generators:
        assert getattr(gen.net, "no_weight_decay", False)


def test_mode_dispatch():
    ConditionedTrunk(4, 8, 1, d_c=2, mode="none")
    ConditionedTrunk(4, 8, 1, d_c=2, mode="concatenate")
    ConditionedTrunk(4, 8, 1, d_c=2, mode="film")
    with pytest.raises(NotImplementedError):
        ConditionedTrunk(4, 8, 1, d_c=2, mode="low_rank")
    with pytest.raises(ValueError):
        ConditionedTrunk(4, 8, 1, d_c=2, mode="bogus")
    assert set(CONDITIONING_MODES) == {"none", "concatenate", "film", "low_rank"}


def test_concatenate_reads_conditioning():
    torch.manual_seed(1)
    trunk = ConditionedTrunk(4, 8, 2, d_c=2, mode="concatenate")
    x = torch.randn(6, 4)
    assert not torch.allclose(
        trunk(x, torch.zeros(6, 2)), trunk(x, torch.ones(6, 2))
    )


def test_generator_zero_at_init():
    gen = FiLMGenerator(3, 16, 1, 8)
    dgamma, beta = gen(torch.randn(4, 3))
    assert torch.count_nonzero(dgamma) == 0
    assert torch.count_nonzero(beta) == 0
