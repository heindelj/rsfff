"""Loading the Q-Chem ALMO-EDA water-cluster labels.

These frames differ from the psi4 labels in three ways the loader has to handle: no
nuclear gradient, a fragment partition supplied as a per-atom column rather than by a
diabatic state library, and per-component EDA energies on the header line.
"""

import pytest
import torch

from rsfff.train.data import concatenate_datasets, load_extxyz

from conftest import DATA_W2, DATA_W3

EDA_PARTS = ("prp", "cls_elec", "mod_pauli", "disp", "pol", "ct")


@pytest.fixture(scope="module")
def w2():
    return load_extxyz(DATA_W2, dtype=torch.float64)


def test_loads_without_forces(w2):
    """EDA single points carry no gradient -- None, not a block of zeros."""
    assert not w2.has_forces
    batch = w2.flat_batch([0, 1, 2])
    assert batch.forces is None
    assert batch.positions.shape == (18, 3)
    assert batch.energy.shape == (3,)


def test_fragment_idx_from_column(w2):
    """A partition with no state library -- and hence no channel graph."""
    assert w2.has_fragments
    assert not w2.has_channels
    assert not w2.has_diabats
    batch = w2.flat_batch([0, 1])
    assert batch.fragment_idx.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert batch.n_fragments == 4
    assert torch.all(batch.fragment_charge == 0)
    assert torch.all(batch.fragment_two_s == 0)   # all singlets
    assert batch.bond_index is None
    assert batch.bond_batch is None


def test_fragment_idx_is_sorted(w2):
    """The dispersion pair list asserts this; check the source data actually honors it."""
    fi = w2.flat_batch(range(20)).fragment_idx
    assert torch.all(fi.diff() >= 0)


def test_eda_components_present(w2):
    batch = w2.flat_batch([0])
    assert batch.eda is not None
    assert set(EDA_PARTS).issubset(batch.eda)
    assert all(v.shape == (1,) for v in batch.eda.values())


def test_eda_sum_rule(w2):
    """int == prp + cls_elec + mod_pauli + disp + pol + ct, to Q-Chem's print precision."""
    batch = w2.flat_batch(range(50))
    total = sum(batch.eda[p] for p in EDA_PARTS)
    assert torch.allclose(total, batch.eda["int"], atol=1e-6)


def test_eda_disp_scale(w2):
    """A units guard: Hartree, not kJ/mol (which would be ~2600x larger)."""
    disp = w2.flat_batch(range(200)).eda["disp"]
    assert 2e-3 < disp.abs().mean().item() < 5e-3
    assert torch.all(disp < 0)      # dispersion is attractive


def test_eda_survives_batch_indexing(w2):
    """Per-frame EDA values follow their frames through flat_batch."""
    all_disp = w2.flat_batch(range(10)).eda["disp"]
    picked = w2.flat_batch([7, 3]).eda["disp"]
    assert picked[0].item() == all_disp[7].item()
    assert picked[1].item() == all_disp[3].item()


def test_concatenate_preserves_eda(w2):
    w3 = load_extxyz(DATA_W3, dtype=torch.float64)
    both = concatenate_datasets([w2, w3])
    assert len(both) == len(w2) + len(w3)
    assert set(both.flat_batch([0]).eda) == set(w2.flat_batch([0]).eda)
    assert not both.has_forces
    # w3 frames still carry 3 fragments after the concatenation re-offsets them
    batch = both.flat_batch([len(w2)])
    assert batch.n_fragments == 3
    assert batch.fragment_idx.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]


def test_concatenate_rejects_mismatched_eda(w2):
    import copy

    trimmed = copy.copy(w2)
    trimmed._eda = {k: v for k, v in w2._eda.items() if k != "disp"}
    with pytest.raises(ValueError, match="EDA component sets differ"):
        concatenate_datasets([w2, trimmed])


def test_to_device_roundtrip(w2):
    """Batch.to enumerates fields by hand; the new ones must be listed."""
    batch = w2.flat_batch([0, 1]).to("cpu")
    assert batch.forces is None
    assert batch.eda is not None and "disp" in batch.eda
    assert batch.fragment_idx is not None
