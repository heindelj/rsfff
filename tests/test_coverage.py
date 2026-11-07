"""rsfff.coverage: hull distance, support points, the FeatureHull round trip, greedy hull extension, the guard."""
import numpy as np
import pytest

from rsfff.coverage import (INSIDE_TOL, FeatureHull, Guard, Whitener, greedy_extend, hull_distance, hull_support,
                            information_imbalance, maxvol, maxvol_grade)

SQUARE = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])


def test_distance_to_a_square():
    Q = np.array([[0.5, 0.5], [2.0, 0.5], [-1.0, -1.0], [0.5, 1.25], [1.0, 1.0]])
    d, p = hull_distance(SQUARE, Q, return_projection=True)
    np.testing.assert_allclose(d, [0.0, 1.0, np.sqrt(2.0), 0.25, 0.0], atol=1e-6)
    np.testing.assert_allclose(p[1], [1.0, 0.5], atol=1e-6)


def test_support_contains_every_vertex_and_spans_the_hull():
    from scipy.spatial import ConvexHull
    rng = np.random.default_rng(0)
    for dim in (2, 3, 4):
        P = rng.normal(size=(300, dim))
        S = hull_support(P)
        assert set(ConvexHull(P).vertices) <= set(S)
        assert hull_distance(P[S], P).max() < 1e-6             # nothing outside conv(S)
        Sp = hull_support(P, prune=True)
        assert set(Sp) == set(ConvexHull(P).vertices)


def test_whitener_has_unit_variance():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 5)) @ rng.normal(size=(5, 5))
    W = Whitener.fit(X, k=3)
    np.testing.assert_allclose(W(X).std(0, ddof=1), 1.0, rtol=1e-8)
    assert W.k == 3 and abs(W.explained.sum() - 1) < 1e-12


@pytest.fixture
def hull():
    rng = np.random.default_rng(2)
    F = rng.lognormal(size=(400, 8)) * 1e-3          # positive, heavy-tailed, like the overlap invariants
    return FeatureHull.fit(F, F.std(0), k=3, feature_names=[f"f{i}" for i in range(8)], meta={"run": "x"}), F


def test_featurehull_roundtrip_and_inside(hull, tmp_path):
    h, F = hull
    assert h.inside(F).all()
    h.save(tmp_path / "hull.pt")
    g = FeatureHull.load(tmp_path / "hull.pt")
    np.testing.assert_allclose(g.points, h.points)
    np.testing.assert_allclose(g.distance(5 * F[:20]), h.distance(5 * F[:20]))
    assert g.meta["run"] == "x" and g.meta["n_train_atoms"] == 400 and g.feature_names[0] == "f0"


def test_gradient_matches_finite_differences(hull):
    h, F = hull
    x = 4 * F[F.sum(1).argmax()][None]               # well outside
    d, g = h.gradient(x)
    assert d[0] > 0.1
    eps = 1e-6 * h.feat_scale
    fd = np.array([(h.distance(x + np.eye(8)[j] * eps[j])[0] - h.distance(x - np.eye(8)[j] * eps[j])[0])
                   / (2 * eps[j]) for j in range(8)])
    np.testing.assert_allclose(g[0], fd, rtol=1e-4, atol=1e-6 * np.abs(fd).max())
    _, g_in = h.gradient(F[:5])
    assert np.abs(g_in).max() == 0.0


def test_extend_grows_only_from_outside(hull):
    h, F = hull
    n0 = len(h.points)
    assert h.extend(F[:50]) == 0 and len(h.points) == n0
    far = 3 * F[:5]
    assert h.extend(far) > 0
    assert h.inside(far).all()


def test_greedy_extension_picks_far_frames_first_and_stops(hull):
    h, F = hull
    frames = [F[i:i + 4] for i in range(0, 40, 4)]              # inside
    frames += [2.0 * F[:4], -3.0 * F[4:8], 6.0 * F[:4], 6.01 * F[:4]]   # outside; 12 is just inside-of 13
    picked, scores, h2 = greedy_extend(h, frames, n=10)
    assert picked[0] == 13 and np.all(np.diff(scores) <= 1e-12)
    assert picked[1] == 11 and 12 not in picked                 # 13 covers its near-duplicate 12 (and 10)
    assert all(i >= 10 for i in picked)                         # never an inside frame
    assert len(h.points) < len(h2.points)                       # the input hull is untouched (copy)
    for i in picked:
        assert h2.inside(frames[i]).all()


def test_guard():
    g = Guard()
    nacl = (["Na", "Cl"], [[0, 0, 0], [2.4, 0, 0]])
    assert g(*nacl)
    assert g.reason(["Cl", "Cl"], [[0, 0, 0], [1.0, 0, 0]]) == "like contact"
    assert g.reason(["Na", "Cl"], [[0, 0, 0], [1.2, 0, 0]]) == "unlike contact"
    assert g.reason(["Na", "Cl", "Na"], [[0, 0, 0], [2.4, 0, 0], [20, 0, 0]]) == "detached atom"


def test_maxvol_grade_is_at_most_one_on_the_rows():
    rng = np.random.default_rng(3)
    A = rng.normal(size=(200, 4))
    idx = maxvol(A)
    assert maxvol_grade(A[idx], A).max() <= 1.01 + 1e-9


def test_information_imbalance_orders_informative_spaces():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(800, 3))
    full = np.c_[X, X[:, :1] ** 2]
    assert information_imbalance(X, full, n=800) < 0.05          # X determines the full space
    assert information_imbalance(X[:, :1], full, n=800) > 0.2     # one coordinate does not


def test_hull_of_density_features_contains_its_own_frames():
    """End to end on the complete model's featurizer: a hull of NaCl clusters contains them, a compressed copy is
    outside, and its gradient points along increasing contact."""
    from rsfff.coverage import featurizer_for_params, frame_features
    feat = featurizer_for_params("params/ions_wb97mv_tzvpd_pf.json")
    rng = np.random.default_rng(5)
    base = np.array([[0, 0, 0], [2.8, 0, 0], [0, 2.8, 0], [2.8, 2.8, 0]], float)
    Z = [11, 17, 17, 11]
    frames = [frame_features(feat, Z, base * s + rng.normal(scale=0.05, size=base.shape))
              for s in rng.uniform(1.0, 1.3, 40)]
    F = np.concatenate(frames)
    h = FeatureHull.fit(F, F[np.abs(F).sum(1) > 0].std(0).clip(min=1e-6), k=4)   # as train_ionct
    assert h.inside(F).all()
    tight = frame_features(feat, Z, base * 0.85)
    assert h.distance(tight).min() > INSIDE_TOL
