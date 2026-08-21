"""The archive guard: ``checkpoints/water_staged/best.pt`` still runs and still answers the same.

:mod:`rsfff.ff.v1` is a frozen copy of the unified pair model, kept because that checkpoint is
trained and benchmarked while the live tree has moved to the fragment-expert architecture of
``docs/fff_v2.md``. The copy is only *half* of what running it needs: the parameter heads, the
classical forms, the response solve and the pair list all still come from the live tree, on the
claim that the two-slot refactor was additive and every head builds bit-identical modules under its
``p_env = 0`` default.

**This test is what makes that claim checkable.** It is not a physics test and the numbers below
carry no meaning beyond "what this checkpoint said on the day it was archived". If it fails, a
change that was supposed to be backward compatible was not, and the fix is in the live tree, never
here.

``strict=True`` inside :func:`~rsfff.ff.v1.load_v1_checkpoint` catches the coarse version of that
failure -- a renamed or reshaped parameter. These assertions catch the fine version, where the
tensors still load but the arithmetic around them moved.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rsfff.train.data import load_extxyz

from conftest import DATA_W3

CHECKPOINT = Path("checkpoints/water_staged/best.pt")

#: Frames 0 and 1 of ``DATA_W3``, from the archived checkpoint. Hartree.
EXPECTED_ENERGY = (-229.30360191829055, -229.29688563674537)
EXPECTED_INTERACTION = {
    "elst": (-0.017341420474875427, -0.002144078762320912),
    "pauli": (0.018730494054836504, 0.006145887987739529),
    "disp": (-0.005062063938540719, -0.003110766248181577),
    "induction": (-0.006047653172142259, -0.0009448608881335308),
}
#: Six waters across the two trimers.
EXPECTED_FRAGMENT_ENERGY = (
    -76.43078801315345, -76.42993947756075, -76.43315378404563,
    -76.43268981259696, -76.43114104232201, -76.43300096391552,
)

# 1e-10 Hartree is 2.6e-7 kJ/mol -- tight enough that any real change in the arithmetic trips it,
# loose enough to survive a BLAS or torch version that reassociates a reduction.
TOL = 1.0e-10


@pytest.fixture(scope="module")
def v1_model():
    if not CHECKPOINT.exists():
        pytest.skip(f"{CHECKPOINT} not present")
    from rsfff.ff.v1 import load_v1_checkpoint

    torch.set_default_dtype(torch.float64)
    model, _config, _neighbor_types = load_v1_checkpoint(CHECKPOINT)
    return model


@pytest.fixture(scope="module")
def w3_batch():
    return load_extxyz(DATA_W3, dtype=torch.float64).flat_batch([0, 1])


def test_checkpoint_loads_strictly(v1_model):
    """No missing and no unexpected tensors -- the load itself is the coarse guard."""
    assert v1_model.atomic_energy is not None
    assert v1_model.environment is not None, "the v1 environment residual must still be built"
    assert v1_model.induction is True


def test_total_energy_is_pinned(v1_model, w3_batch):
    out = v1_model(w3_batch)
    expected = torch.tensor(EXPECTED_ENERGY, dtype=torch.float64)
    assert torch.allclose(out.energy, expected, atol=TOL, rtol=0.0)


def test_decomposition_is_pinned(v1_model, w3_batch):
    """Each channel separately: a compensating pair of errors would pass the total alone."""
    out = v1_model(w3_batch)
    assert set(out.interaction) == set(EXPECTED_INTERACTION)
    for name, values in EXPECTED_INTERACTION.items():
        expected = torch.tensor(values, dtype=torch.float64)
        assert torch.allclose(out.interaction[name], expected, atol=TOL, rtol=0.0), name

    expected_frag = torch.tensor(EXPECTED_FRAGMENT_ENERGY, dtype=torch.float64)
    assert torch.allclose(out.fragment_energy, expected_frag, atol=TOL, rtol=0.0)


def test_still_reproduces_its_labels(v1_model, w3_batch):
    """A floor on accuracy, so a checkpoint swapped for a broken one does not just re-pin.

    These bounds are the trained model's reported validation MAEs with room to spare, not
    targets. The point is that the pinned numbers above are a *trained* model's, which the
    numbers alone cannot say.
    """
    out = v1_model(w3_batch)
    kjmol = 2625.4996394799
    assert (out.energy - w3_batch.energy).abs().max() * kjmol < 5.0
    for name, target in (
        ("elst", "cls_elec"), ("pauli", "mod_pauli"), ("disp", "disp"),
    ):
        err = (out.interaction[name] - w3_batch.eda[target]).abs().max() * kjmol
        assert err < 3.0, f"{name}: {err:.3f} kJ/mol"
