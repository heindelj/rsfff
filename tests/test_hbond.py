"""The Kumar/Schmidt/Skinner ``r``-``psi`` hydrogen-bond definition.

The load-bearing tests are the two that pin the *published* numbers:
:func:`test_occupancy_reproduces_paper_mean` (their Eq. 4 must return the reported mean
occupancy at the reported mean geometry) and :func:`test_cutoff_curves_cross_near_30_deg`
(the two cutoffs of their Fig. 8 must cross where the figure shows them crossing). Those
catch a mistyped coefficient, which nothing downstream would -- a wrong constant still
produces a plausible bimodal histogram and a plausible hydrogen-bond count.

The rest pin the geometry: an H-bond count is only meaningful if the descriptors are
rigid-motion invariant and if the ideal cage clusters come out with the counts their
topology demands.
"""

import numpy as np
import pytest
from ase.io import read

from rsfff.ff.hbond import (
    OCCUPANCY_CUTOFF,
    dimer_geometry,
    hbond_labels,
    hbonds_per_molecule,
    pmf_cutoff_radius,
    sigma_star_occupancy,
)

W5 = "data/wb97mv_tzvpd_large/w5_wb97mv_tzvpd.xyz"
W8 = "data/wb97mv_tzvpd_large/w8_wb97mv_tzvpd.xyz"
W2 = "data/wb97mv_tzvpd/w2_wb97mv_tzvpd.xyz"


def frame(path, index=0):
    atoms = read(path, index=index)
    return atoms.positions, atoms.numbers, atoms.arrays["fragment_idx"]


def test_occupancy_reproduces_paper_mean():
    """``N`` at the liquid's typical linear geometry must land on the reported average.

    The paper quotes a mean ``sigma*_OH`` occupancy of 0.021 over 10 000 SPC/E dimers and a
    typical intermolecular OH distance of about 1.75-2.0 A. Eq. (4) at ``r = 2.0``,
    ``psi = 0`` has to reproduce that, and does (0.0208) -- a one-digit typo in the 0.343 A
    decay length or the 7.1 prefactor moves it by tens of percent.
    """
    assert sigma_star_occupancy(2.0, 0.0) == pytest.approx(0.021, abs=0.001)
    # and the distribution's stated span: nothing in the physical range exceeds ~0.06
    assert sigma_star_occupancy(1.6, 0.0) < 0.07


def test_cutoff_curves_cross_near_30_deg():
    """Fig. 8: the occupancy contour sits *inside* the PMF contour at ``psi = 0`` and
    *outside* it at ``psi = 90``, crossing in between.

    This is the qualitative statement the paper makes about the two definitions being
    "reasonably compatible", and it is a joint constraint on Eq. (3) and Eq. (4) -- a sign
    error in either one breaks it while leaving each curve individually plausible.
    """
    def occupancy_radius(psi):
        amplitude = 7.1 - 0.050 * psi + 0.00021 * psi * psi
        return -0.343 * np.log(OCCUPANCY_CUTOFF / amplitude)

    assert occupancy_radius(0.0) < pmf_cutoff_radius(0.0)
    assert occupancy_radius(90.0) > pmf_cutoff_radius(90.0)
    crossing = [p for p in range(0, 91)
                if occupancy_radius(p) > pmf_cutoff_radius(float(p))]
    assert 25 <= min(crossing) <= 40


@pytest.mark.parametrize("psi", [0.0, 25.0, 50.0, 89.0])
def test_psi_folds_about_ninety(psi):
    """Both fits must be symmetric under ``psi -> 180 - psi``.

    The acceptor's out-of-plane normal is ``e1 x e2``, whose sign depends on which O-H the
    code happened to list first. Without the fold, relabelling the acceptor's two hydrogens
    -- a pure bookkeeping change -- would flip a pair's hydrogen-bond status.
    """
    assert pmf_cutoff_radius(psi) == pytest.approx(pmf_cutoff_radius(180.0 - psi))
    assert sigma_star_occupancy(1.9, psi) == pytest.approx(
        sigma_star_occupancy(1.9, 180.0 - psi)
    )


def test_descriptors_are_rigid_motion_invariant():
    positions, z, frag = frame(W5)
    reference = dimer_geometry(positions, z, frag)

    rng = np.random.default_rng(0)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = 0.7
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rotation = np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)
    moved = positions @ rotation.T + np.array([3.1, -7.2, 0.4])

    got = dimer_geometry(moved, z, frag)
    assert got.shape == reference.shape
    for field in ("R", "r", "alpha", "beta", "gamma", "psi"):
        np.testing.assert_allclose(got[field], reference[field], atol=1e-9)


def test_reflection_leaves_psi_unchanged():
    """A mirror image flips the acceptor's normal, and ``psi`` must survive it.

    ``psi`` is measured against ``e1 x e2`` of the acceptor's two O-H bonds, so a reflection
    -- or merely swapping which of two identical hydrogens is listed first -- sends the raw
    angle to its complement. The reported value is folded into ``[0, 90]`` precisely so that
    neither bookkeeping choice can move a pair across a cutoff.
    """
    positions, z, frag = frame(W5)
    reference = dimer_geometry(positions, z, frag)
    got = dimer_geometry(positions * np.array([1.0, 1.0, -1.0]), z, frag)
    assert np.all(reference["psi"] <= 90.0)
    np.testing.assert_allclose(got["psi"], reference["psi"], atol=1e-9)


def test_relabelling_acceptor_hydrogens_changes_nothing():
    """Swapping the two hydrogens of every molecule is pure bookkeeping."""
    positions, z, frag = frame(W5)
    reference = dimer_geometry(positions, z, frag)

    order = np.arange(len(z))
    for f in range(int(frag.max()) + 1):
        h = np.flatnonzero((frag == f) & (z == 1))
        order[h[0]], order[h[1]] = order[h[1]], order[h[0]]
    got = dimer_geometry(positions[order], z[order], frag[order])

    for field in ("R", "r", "alpha", "beta", "gamma", "psi"):
        np.testing.assert_allclose(np.sort(got[field]), np.sort(reference[field]), atol=1e-9)


def test_water_dimer_is_one_near_linear_hydrogen_bond():
    positions, z, frag = frame(W2)
    g = dimer_geometry(positions, z, frag)
    assert len(g) == 1
    assert g["R"][0] == pytest.approx(2.83, abs=0.1)     # the known equilibrium O-O
    assert g["alpha"][0] < 30.0                          # near-linear: alpha = 180 - theta
    assert hbond_labels(g, definition="occupancy")[0]
    assert hbond_labels(g, definition="pmf")[0]


@pytest.mark.parametrize("path, n_frag, expected", [(W5, 5, 2.0), (W8, 8, 3.0)])
def test_ideal_cages_have_the_count_their_topology_demands(path, n_frag, expected):
    """The relaxed cyclic pentamer has 5 bonds and the cubic octamer 12, so ``<n>`` -- twice
    the bonds per molecule -- is exactly 2.0 and 3.0. Any double counting, any missed
    symmetry-equivalent bond, shows up as a non-integer multiple of ``2/n``.
    """
    positions, z, frag = frame(path)
    g = dimer_geometry(positions, z, frag)
    for definition in ("occupancy", "pmf"):
        assert hbonds_per_molecule(g, n_frag, definition=definition) == pytest.approx(expected)


def test_candidate_window_does_not_change_the_labels():
    """``r_max`` bounds the candidate list only; widening it may only add unbonded rows."""
    positions, z, frag = frame(W8)
    narrow = dimer_geometry(positions, z, frag, r_max=3.0)
    wide = dimer_geometry(positions, z, frag, r_max=5.0)
    assert len(wide) > len(narrow)
    assert hbond_labels(narrow).sum() == hbond_labels(wide).sum()


def test_non_water_fragment_raises():
    positions, z, frag = frame(W2)
    z = z.copy()
    z[0] = 7                     # an oxygen turned into nitrogen: no longer water
    with pytest.raises(ValueError, match="water molecules only"):
        dimer_geometry(positions, z, frag)
