import math

import pytest

np = pytest.importorskip("numpy")

from wescalp import intraday, nextcandles as nc  # noqa: E402

STEP, DAY = 300, 86400


def synthetic(days: int = 60, seed: int = 0, drop: int | None = None):
    """Candles de 5 min e séries de fluxo com a mesma cara das da Binance, a partir de 01/01/2020."""
    rng = np.random.default_rng(seed)
    t0 = intraday.utc("2020-01-01")
    ts = np.arange(t0, t0 + days * DAY, STEP, dtype=np.int64)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, len(ts))))
    o = np.concatenate([[100.0], c[:-1]])
    h, l = np.maximum(o, c) * 1.001, np.minimum(o, c) * 0.999
    vol = rng.uniform(1, 2, len(ts))
    flow = {"ts": ts, "volume": vol, "taker_buy": vol * rng.uniform(0.3, 0.7, len(ts)), "trades": rng.integers(10, 99, len(ts)).astype(float)}
    perp = {"ts": ts, "close": c * 1.0005, "volume": vol, "taker_buy": vol * rng.uniform(0.3, 0.7, len(ts))}
    premium = {"ts": ts, "close": rng.normal(0, 1e-4, len(ts))}
    funding = {"ts": ts[::96] + 1, "rate": rng.normal(1e-4, 1e-5, len(ts[::96]))}
    metrics = {"ts": ts, **{k: rng.uniform(1, 2, len(ts)) for k in ("oi", "accounts", "top_positions", "taker_ratio")}}
    eth = (ts, o / 30, h / 30, l / 30, c / 30)
    if drop is not None:
        keep = np.ones(len(ts), dtype=bool)
        keep[drop] = False
        ts, o, h, l, c = ts[keep], o[keep], h[keep], l[keep], c[keep]
    return (ts, o, h, l, c), flow, perp, premium, funding, metrics, eth


def build(days=60, seed=0, drop=None, change_after=None):
    candles, flow, perp, premium, funding, metrics, eth = synthetic(days, seed, drop)
    if change_after is not None:  # muda tudo o que vem depois de um instante
        ts = candles[0]
        later = ts >= change_after
        candles = (ts, *(np.where(later, x * 1.3, x) for x in candles[1:]))
        for s in (flow, perp, metrics):
            for k in s:
                if k != "ts":
                    s[k] = np.where(s["ts"] >= change_after, s[k] * 1.7, s[k])
    t0 = intraday.utc("2020-01-01")
    return nc.build(*candles, flow, perp, premium, funding, metrics, eth, start=t0, end=t0 + days * DAY)


def test_ffill_and_rolling_helpers():
    assert np.array_equal(nc.ffill(np.array([np.nan, 1.0, np.nan, 3.0])), [np.nan, 1, 1, 3], equal_nan=True)
    x = np.random.default_rng(1).normal(size=50)
    assert nc.roll_sum(x, 5)[10] == pytest.approx(x[6:11].sum())
    assert nc.roll_std(x, 7)[20] == pytest.approx(x[14:21].std())


def test_targets_are_the_next_candles():
    d = build(days=10)
    lc = np.log(synthetic(10)[0][4])
    t = 500
    for k in (1, 2, 20):
        assert d.y[t, k - 1] == pytest.approx(lc[t + k] - lc[t + k - 1], abs=1e-6)
        assert d.y[t, nc.K + k - 1] == pytest.approx(lc[t + k] - lc[t], abs=1e-6)
    assert np.isnan(d.y[-1]).all() and np.isfinite(d.y[-21]).all()


def test_a_missing_candle_blanks_the_targets_that_cross_it():
    d = build(days=10, drop=700)
    assert not d.ok[700] and d.ok[699]
    assert np.isnan(d.y[699, 0]) and np.isnan(d.y[690, nc.K + 9]) and np.isfinite(d.y[701, nc.K + 9])
    assert np.isfinite(d.x[701, 0])  # o preço do candle que falta fica parado no anterior


def test_variables_never_use_the_future():
    t0 = intraday.utc("2020-01-01")
    cut = t0 + 40 * DAY + 7 * 3600
    a, b = build(days=50), build(days=50, change_after=cut)
    past = a.grid + STEP <= cut  # candles que já fecharam no instante da mudança
    assert past.sum() > 10000
    assert np.array_equal(a.x[past], b.x[past], equal_nan=True)
    assert not np.array_equal(a.x[~past], b.x[~past], equal_nan=True)


def test_folds_keep_an_embargo_between_train_valid_and_test():
    grid = np.arange(intraday.utc("2020-01-01"), intraday.utc("2022-07-01"), STEP, dtype=np.int64)
    n = len(grid)
    d = nc.Data(grid, np.ones(n, dtype=bool), np.zeros((n, 1), np.float32), np.zeros((n, 1), np.float32),
                np.zeros((n, 2 * nc.K), np.float32), np.zeros(n, np.float32))
    for f in nc.folds(d, (2021, 2022), end=intraday.utc("2022-07-01")):
        start = intraday.utc(f"{f.year}-01-01")
        assert f.train.max() + nc.EMBARGO < f.valid.min()
        assert grid[f.valid.max()] < start - nc.EMBARGO * STEP and grid[f.test.min()] >= start
        assert abs(len(f.valid) / (len(f.train) + len(f.valid)) - nc.VALID_SHARE) < 0.01


def test_measures():
    p = np.array([0.6, 0.4, 0.7, 0.2])
    y = np.array([0.01, 0.0, -0.02, -0.01])
    assert nc.accuracy(p, y) == (pytest.approx(2 / 3), 3)  # o candle igual não conta
    p_va = np.linspace(0, 1, 101)
    assert nc.confident(p_va, np.array([0.5, 0.94, 0.96, 0.03])).tolist() == [False, False, True, True]
    up_dn = np.array([0.01, 0.03, -0.02, -0.02])
    assert nc.needed(up_dn, 0.002) == pytest.approx((0.002 + 0.02) / (0.02 + 0.02))
    r = np.full(100, 0.001) + np.tile([0.0005, -0.0005], 50)
    s1, s5 = nc.trade_stats(r, 1), nc.trade_stats(r, 5)
    assert s1["mean"] == pytest.approx(0.001) and s5["t"] == pytest.approx(s1["t"] / math.sqrt(5))


def tiny_data(n: int = 6000, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3)).astype(np.float32)
    y = np.repeat(np.where(x[:, :1] > 0, 0.01, -0.01), 2 * nc.K, axis=1).astype(np.float32)
    grid = np.arange(n, dtype=np.int64) * STEP
    d = nc.Data(grid, np.ones(n, dtype=bool), x, rng.normal(size=(n, len(nc.CHANNELS))).astype(np.float32), y,
                np.zeros(n, np.float32))
    f = nc.Fold(2021, np.arange(60, 4000), np.arange(4020, 5000), np.arange(5020, n))
    return d, f


def test_dense_network_learns_an_easy_rule(monkeypatch):
    pytest.importorskip("jax")
    monkeypatch.setattr(nc, "BATCH", 256)
    monkeypatch.setattr(nc, "EPOCHS", 3)
    d, f = tiny_data()
    p_va, p_te = nc.nn_fold(d, f, "densa")
    assert p_te.shape == (len(f.test), 2 * nc.K)
    assert nc.accuracy(p_te[:, 0], d.y[f.test, 0])[0] > 0.9


def test_chart_network_runs(monkeypatch):
    pytest.importorskip("jax")
    monkeypatch.setattr(nc, "BATCH", 256)
    monkeypatch.setattr(nc, "EPOCHS", 1)
    d, f = tiny_data(3000)
    f = nc.Fold(2021, np.arange(60, 2000), np.arange(2020, 2500), np.arange(2520, 3000))
    p_va, p_te = nc.nn_fold(d, f, "gráfico")
    assert p_te.shape == (len(f.test), 2 * nc.K) and ((p_te > 0) & (p_te < 1)).all()


def test_lightgbm_fold_learns_an_easy_rule(monkeypatch):
    pytest.importorskip("lightgbm")
    monkeypatch.setattr(nc, "LGBM_PARAMS", {**nc.LGBM_PARAMS, "min_data_in_leaf": 20})
    d, f = tiny_data()
    p_va, p_te, gain = nc.lgbm_fold(d, f)
    assert len(p_te) == 2 * nc.K and np.array_equal(p_te[nc.K], p_te[0])
    assert nc.accuracy(p_te[3], d.y[f.test, 3])[0] > 0.9 and gain.argmax() == 0
