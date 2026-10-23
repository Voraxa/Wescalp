import math

import pytest

np = pytest.importorskip("numpy")

from wescalp import study  # noqa: E402


def test_bootstrap_resamples_values_of_the_series():
    r = np.arange(100, dtype=float)
    b = study.stationary_bootstrap(r, 5, 10, np.random.default_rng(0))
    assert b.shape == (5, 100) and set(np.unique(b)) <= set(r)
    jumps = (np.diff(b, axis=1) != 1) & (np.diff(b, axis=1) != -99)
    assert 0 < jumps.mean() < 0.3  # blocos contínuos, com emendas de vez em quando (~1 a cada 10)


def test_cagr_drawdown_underwater():
    r = np.array([0.1, -0.5, 0.2, 0.5, 0.5, 0.0])
    assert study.max_dd(r) == pytest.approx(0.5)
    assert study.underwater_days(r) == 3  # cai no dia 2, só passa o topo (1,1) no dia 5 (0,55·1,2·1,5·1,5 = 1,485)
    assert study.cagr(np.full(365, 0.0)) == 0.0
    assert study.cagr(np.concatenate([[1.0], np.zeros(364)])) == pytest.approx(2.0 ** (365.25 / 365) - 1)


def test_by_period_and_profit_factor():
    r = np.array([0.1, 0.1, -0.2, 0.05])
    keys = np.array([1, 1, 2, 3])
    assert np.allclose(study.by_period(r, keys), [0.21, -0.2, 0.05])
    assert study.profit_factor(np.array([0.21, -0.2, 0.05])) == pytest.approx(0.26 / 0.2)
    assert math.isinf(study.profit_factor(np.array([0.1, 0.2])))
