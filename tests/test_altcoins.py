import math

import pytest

np = pytest.importorskip("numpy")

from wescalp import altcoins, intraday, refine, regime  # noqa: E402

H, DAY = 3600, 86400


def test_moves_and_accuracy_needed():
    ts = np.arange(0, 6 * 300, 300, dtype=np.int64)
    close = np.array([100, 101, 100, 101, 100, 101], dtype=float)  # sobe 1% e cai ~0,99% alternando
    up, dn, need = altcoins.moves(ts, close, 5)
    assert up == pytest.approx(0.01) and dn == pytest.approx(1 / 101)
    assert need == pytest.approx((0.002 + dn) / (up + dn))


def test_binary_signal_compares_with_the_open_two_hours_back():
    t0 = intraday.utc("2022-01-03")
    ts = np.arange(t0 - 40 * DAY, t0 + DAY, 300, dtype=np.int64)
    o = np.full(len(ts), 100.0)
    c = np.full(len(ts), 100.0)
    v = np.ones(len(ts))
    d = np.array([t0 + 10 * H], dtype=np.int64)
    o[ts == d[0] - 2 * H] = 101.0  # a hora de antes abriu a 101
    s = altcoins.hourly_signals(d, ts, o, c, v)
    assert s[altcoins.BINARY][0] == 1.0 and s["retorno 1h"][0] == 0.0 and s["volume relativo 1h"][0] == pytest.approx(0.0)


def test_high_vol_needs_history_and_ignores_the_future():
    rng = np.random.default_rng(0)
    t0 = intraday.utc("2021-01-01")
    ts = np.arange(t0, t0 + 420 * DAY, 300, dtype=np.int64)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, len(ts))))
    decision = np.arange(t0 + H, t0 + 419 * DAY, H, dtype=np.int64)
    hv = altcoins.high_vol(decision, ts, c)
    assert not hv[decision < t0 + 300 * DAY].any()  # sem ~300 dias de história, nada é volatilidade alta
    assert 0.1 < hv[decision > t0 + 380 * DAY].mean() < 0.6
    later = c.copy()
    later[ts >= t0 + 400 * DAY] *= np.exp(rng.normal(0, 0.05, (ts >= t0 + 400 * DAY).sum()))
    hv2 = altcoins.high_vol(decision, ts, later)
    past = decision < t0 + 400 * DAY
    assert np.array_equal(hv[past], hv2[past])


def test_binary_test_passes_only_above_cost_with_strong_t():
    rng = np.random.default_rng(1)
    s = (rng.random(20000) < 0.5).astype(float)
    y = np.where(s > 0, 0.004, 0.0) + rng.normal(0, 0.01, 20000)
    assert altcoins.binary_test(s, y)["passes"]
    assert not altcoins.binary_test(s, rng.normal(0, 0.01, 20000))["passes"]


def test_trend_matches_the_research_functions():
    rng = np.random.default_rng(2)
    close = 100 * np.cumprod(1 + rng.normal(0.001, 0.03, 400))
    d = regime.Daily(np.arange(400), close, close, close, close)
    r, b = altcoins.trend(d, 200, 400)
    rets = close[200:] / close[199:-1] - 1
    assert np.allclose(r, regime.returns_from_held(refine.position(close, refine.BASE)[199:-1], rets))
    assert np.allclose(b, regime.returns_from_held(np.ones(200), rets)) and not math.isnan(altcoins.scores(r)["sharpe"])
