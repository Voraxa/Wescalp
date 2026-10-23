import pytest

np = pytest.importorskip("numpy")

from wescalp import regime  # noqa: E402


def test_daily_bars_from_5min():
    ts = np.array([0, 300, 86100, 86400, 86700], dtype=np.int64)  # dois candles no dia 0 + o último do dia 0, dia 1
    o = np.array([10.0, 11, 12, 20, 21])
    h = np.array([11.0, 15, 13, 22, 25])
    l = np.array([9.0, 10, 8, 19, 20])
    c = np.array([11.0, 12, 13, 21, 24])
    d = regime.daily_bars(ts, o, h, l, c)
    assert list(d.day) == [0, 1]
    assert list(d.open) == [10, 20] and list(d.high) == [15, 25] and list(d.low) == [8, 19] and list(d.close) == [13, 24]


def test_sma_window():
    assert np.allclose(regime.sma(np.array([1.0, 2, 3, 4]), 2)[1:], [1.5, 2.5, 3.5])
    assert np.isnan(regime.sma(np.array([1.0, 2, 3, 4]), 2)[0])


def test_position_from_yesterday_close_and_fees():
    close = np.array([100.0, 100, 110, 121, 121, 108.9])
    pos = np.array([0.0, 1, 1, 0, 0, 0])  # compra no fechamento do dia 1, vende no do dia 3
    r = regime.backtest(close, pos, lo=1, fee=0.001)
    # dias 1..5: comprado nos dias 2 e 3 (decidido nos fechamentos de 1 e 2); taxa no dia 2 (compra) e 4 (venda)
    assert np.allclose(r.daily, [0.0, 0.10 - 0.001, 0.10, -0.001, 0.0])
    (t,) = r.trades
    assert t == pytest.approx(0.999 * 1.1 * 1.1 * 0.999 - 1, rel=1e-3)
    assert r.in_market == pytest.approx(2 / 5)


def test_no_lookahead_rule_change_today_only_affects_tomorrow():
    close = np.array([100.0, 100, 50, 50])
    calm = regime.backtest(close, np.array([1.0, 1, 1, 1]), lo=1, fee=0.0)
    sold_at_crash_close = regime.backtest(close, np.array([1.0, 1, 0, 0]), lo=1, fee=0.0)
    # Vender no fechamento do dia da queda não evita a queda daquele dia.
    assert calm.daily[1] == sold_at_crash_close.daily[1] == pytest.approx(-0.5)


def test_rules_stay_out_until_history():
    close = np.linspace(100, 200, 300)
    r = regime.rules(close)
    assert r["sma 200"][:199].sum() == 0 and r["sma 200"][250] == 1
    assert r["ema 50/200"][:200].sum() == 0
    assert r["mom 180"][:180].sum() == 0 and r["mom 180"][299] == 1


def test_shuffled_timing_keeps_segments():
    held = np.array([0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1], dtype=float)
    rng = np.random.default_rng(1)
    for _ in range(50):
        s = regime.shuffled_timing(held, rng)
        assert len(s) == len(held) and s.sum() == held.sum() and s[0] == held[0]
        runs = lambda x: sorted(len(p) for p in np.split(x, np.flatnonzero(np.diff(x)) + 1) if p[0] == 1)  # noqa: E731
        assert runs(s) == runs(held)  # mesmas durações de operação, mesmo número de operações


def test_luck_test_detects_perfect_timing():
    rng = np.random.default_rng(0)
    rets = rng.normal(0, 0.03, 400)
    held = (rets > 0).astype(float)  # comprado exatamente nos dias de alta: impossível por sorte
    out = regime.luck_test(held, rets, n=200, fee=0.0)
    assert out["p_total"] == 0.0 and out["p_sharpe"] == 0.0


def test_bar_mask_uses_the_day_position():
    day = np.array([10, 11, 12])
    held = np.array([1.0, 0.0])  # dias 11 e 12 (lo = 1)
    ts = np.array([10 * 86400 + 5, 11 * 86400, 11 * 86400 + 86399, 12 * 86400 + 300], dtype=np.int64)
    assert list(regime.bar_mask(ts, day, held, lo=1)) == [False, True, True, False]
