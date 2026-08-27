"""The expert bank: composition keys, dispatch, and what the applicability head may not see."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from rsfff.ff.expert import (
    ApplicabilityHead,
    ExpertBank,
    FragmentExpert,
    composition_keys,
)

SYMBOLS = ("H", "O")


def _expert(key: str) -> FragmentExpert:
    """A container full of stand-ins: dispatch does not care what the heads are."""
    # v3: an expert is an *encoder* plus its fragment-state block. The parameter heads it
    # used to hold live once, in the shared `ParameterDecoder`.
    return FragmentExpert(
        key, encoder=nn.Linear(2, 2), fragment_state=nn.Linear(2, 2)
    )


# --- a water dimer, a hydroxide and a hydronium, atoms in species-index order (H=0, O=1) ---
SPECIES = torch.tensor([1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0])
FRAGMENT = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 3])
N_FRAG = 4


def test_composition_keys_are_hill_ordered_and_deduplicated():
    keys, inverse = composition_keys(SPECIES, FRAGMENT, N_FRAG, SYMBOLS)
    assert keys == ["HO", "H2O", "H3O"]
    assert [keys[i] for i in inverse.tolist()] == ["H2O", "H2O", "HO", "H3O"]


def test_single_composition_takes_the_fast_path():
    """One expert covering everything, and nothing gathered."""
    species = torch.tensor([1, 0, 0, 1, 0, 0])
    frag = torch.tensor([0, 0, 0, 1, 1, 1])
    bank = ExpertBank({"H2O": _expert("H2O")}, SYMBOLS)
    (group,) = bank.groups(species, frag, 2)
    assert group.key == "H2O" and group.is_everything
    assert group.atom_index is None and group.fragment_index is None
    assert group.expert is bank.only


def test_multiple_compositions_partition_the_batch():
    bank = ExpertBank(
        {k: _expert(k) for k in ("H2O", "HO", "H3O")}, SYMBOLS
    )
    groups = bank.groups(SPECIES, FRAGMENT, N_FRAG)
    assert {g.key for g in groups} == {"H2O", "HO", "H3O"}
    by_key = {g.key: g for g in groups}
    # every atom claimed exactly once
    claimed = torch.cat([g.atom_index for g in groups]).sort().values
    assert torch.equal(claimed, torch.arange(SPECIES.shape[0]))
    frags = torch.cat([g.fragment_index for g in groups]).sort().values
    assert torch.equal(frags, torch.arange(N_FRAG))
    # and to the right expert
    assert by_key["H2O"].fragment_index.tolist() == [0, 1]
    assert by_key["HO"].fragment_index.tolist() == [2]
    assert by_key["H3O"].fragment_index.tolist() == [3]
    assert by_key["HO"].atom_index.tolist() == [6, 7]


def test_an_unknown_composition_raises_rather_than_falling_back():
    """The failure this prevents is silent: a plausible number from the wrong molecule."""
    bank = ExpertBank({"H2O": _expert("H2O")}, SYMBOLS)
    with pytest.raises(KeyError, match="no expert for fragment composition"):
        bank.groups(SPECIES, FRAGMENT, N_FRAG)


def test_only_refuses_a_multi_expert_bank():
    bank = ExpertBank({k: _expert(k) for k in ("H2O", "HO")}, SYMBOLS)
    with pytest.raises(ValueError, match="single-composition case"):
        _ = bank.only


# ---------------------------------------------------------------------------------------
# applicability
# ---------------------------------------------------------------------------------------

def test_applicability_requires_the_joined_descriptor():
    """The score is about competition between decompositions, so ``eta`` is not optional.

    This head used to *refuse* the joined descriptor, on the reading that applicability was a
    property of one fragment. It is not: the question is whether this decomposition is the
    best description of the system, which an H2O with a proton 1 Angstrom away cannot answer
    from inside its own fragment. Passing the isolated slot is now the width error.
    """
    torch.set_default_dtype(torch.float64)
    head = ApplicabilityHead(6, 4, hidden=8, depth=2).double()
    frag = torch.tensor([0, 0, 0])
    assert head.width == 10
    head(torch.randn(3, 10), frag, 1, None, None)      # joined: accepted
    with pytest.raises(ValueError, match="reads the joined descriptor"):
        head(torch.randn(3, 6), frag, 1, None, None)   # isolated: refused


def test_applicability_sees_the_environment_slot():
    """A score that ignored eta could not tell two fragmentations of one geometry apart.

    ``TwoSlotLinear.w_env`` is zero-initialized, so the score starts *inert* to the
    environment -- a fresh model has no opinion about which decomposition is better, which is
    the right thing to know nothing from. Both halves of that are pinned here: inert at
    initialization, live once the environment weights have moved.
    """
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    head = ApplicabilityHead(6, 4, hidden=8, depth=2).double()
    head.net[-1].weight.data.normal_()      # wake the zero-init readout
    head.net[-1].bias.data.normal_()

    frag = torch.tensor([0, 0, 0])
    joined = torch.randn(3, 10)
    moved = joined.clone()
    moved[:, 6:] += 1.0                     # perturb the environment block only

    base = head(joined, frag, 1, None, None)
    assert torch.allclose(base, head(moved, frag, 1, None, None))

    head.net[0].w_env.data.normal_()        # wake the environment slot
    assert not torch.allclose(
        head(joined, frag, 1, None, None), head(moved, frag, 1, None, None)
    )


def test_applicability_is_per_fragment_and_ignores_other_fragments():
    """Changing one fragment's atoms must not move another fragment's score."""
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    head = ApplicabilityHead(6, hidden=8, depth=2).double()
    head.net[-1].weight.data.normal_()      # wake the zero-init readout
    head.net[-1].bias.data.normal_()

    feats = torch.randn(6, 6)
    frag = torch.tensor([0, 0, 0, 1, 1, 1])
    base = head(feats, frag, 2, None, None)

    moved = feats.clone()
    moved[3:] = torch.randn(3, 6)
    after = head(moved, frag, 2, None, None)
    assert torch.equal(base[:1], after[:1])
    assert not torch.equal(base[1:], after[1:])


def test_applicability_is_invariant_to_atom_order_within_a_fragment():
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    head = ApplicabilityHead(6, hidden=8, depth=2).double()
    head.net[-1].weight.data.normal_()
    feats = torch.randn(3, 6)
    frag = torch.zeros(3, dtype=torch.long)
    a = head(feats, frag, 1, None, None)
    b = head(feats[[2, 0, 1]], frag, 1, None, None)
    assert torch.allclose(a, b, atol=1e-14, rtol=0.0)


def test_applicability_reads_charge_and_multiplicity():
    """A hydroxide and an OH radical share an expert, so ``v_f`` must be able to tell them apart."""
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    head = ApplicabilityHead(6, hidden=8, depth=2).double()
    head.net[-1].weight.data.normal_()
    feats = torch.randn(2, 6)
    frag = torch.zeros(2, dtype=torch.long)
    anion = head(feats, frag, 1, torch.tensor([-1.0]), torch.tensor([0.0]))
    radical = head(feats, frag, 1, torch.tensor([0.0]), torch.tensor([1.0]))
    assert not torch.equal(anion, radical)
