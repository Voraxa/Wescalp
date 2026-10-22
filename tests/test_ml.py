import math

import pytest

np = pytest.importorskip("numpy")

from wescalp import ml  # noqa: E402


def test_rsum_trailing_window_inclusive():
    out = ml.rsum(np.array([1.0, 2.0, np.nan, 4.0]), 2)
    assert math.isnan(out[0])
    assert list(out[1:]) == [3.0, 2.0, 4.0]  # NaN conta como 0


def test_returns_past_and_future_in_bps():
    mid = np.array([100.0, 101.0, 102.0])
    assert ml.past_ret_bps(mid, 1)[1] == pytest.approx(100.0)
    assert math.isnan(ml.past_ret_bps(mid, 1)[0])
    f = ml.fwd_ret_bps(mid, 2)
    assert f[0] == pytest.approx(200.0) and math.isnan(f[1]) and math.isnan(f[2])


def cols(n: int, seed: int = 0) -> dict[str, list]:
    rng = np.random.default_rng(seed)
    mid = 100 + np.cumsum(rng.normal(0, 0.01, n))
    return {
        "t": [1_735_689_600.0 + i + 1 for i in range(n)],
        "mid": list(mid),
        "ofi": list(rng.normal(0, 10, n)),
        "delta": list(rng.normal(0, 5, n)),
        "imb5": list(rng.uniform(-1, 1, n)),
        "micro_bps": list(rng.normal(0, 0.1, n)),
        "spread_bps": [0.01] * n,
    }


def test_features_use_only_the_past():
    n, t = 1500, 1000
    a = cols(n)
    b = {k: v[:t + 1] + cols(n, seed=1)[k][t + 1:] if k != "t" else v for k, v in a.items()}  # futuro trocado
    fa, fb = ml.features(a), ml.features(b)
    assert np.array_equal(fa[:t + 1], fb[:t + 1], equal_nan=True)
    assert not np.array_equal(fa[t + 1:], fb[t + 1:], equal_nan=True)


def test_features_blank_where_bucket_invalid():
    c = cols(1500)
    c["mid"][1200] = None
    f = ml.features(c)
    assert np.isnan(f[1200]).all()
    assert not np.isnan(f[1199]).any()  # depois de 600 s de aquecimento tudo existe
    assert f.shape == (1500, len(ml.FEATURES))


def test_ofi_normalized_by_trailing_mean():
    c = cols(700)
    c["ofi"] = [2.0] * 700
    f = ml.features(c)
    assert f[699, ml.FEATURES.index("ofi_10")] == pytest.approx(1.0)


def test_simulate_does_not_overlap():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, np.nan])
    pred = np.array([1.0, -1.0, 1.0, 1.0, -1.0, 1.0])
    fire = np.array([True, True, True, True, True, True])
    assert ml.simulate(pred, fire, y, 2) == [1.0, 3.0, -5.0]


def test_folds_expanding_window():
    assert ml.folds(16, train=10, test=3) == [(range(0, 10), range(10, 13)), (range(0, 13), range(13, 16))]
    assert ml.folds(14, train=10, test=3)[-1] == (range(0, 13), range(13, 14))


def test_ridge_recovers_linear_relation():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5000, 2))
    y = 3 * x[:, 0] - 2 * x[:, 1] + 1
    p = ml.Ridge(lam=0.0).fit(x, y).predict(np.array([[1.0, 1.0]]))
    assert p[0] == pytest.approx(2.0)
