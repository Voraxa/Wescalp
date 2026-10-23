import pytest

np = pytest.importorskip("numpy")

from wescalp import dailydip, regime, spot  # noqa: E402

FEE = 0.001


def bars(prices: list[float]):
    p = np.array(prices, dtype=float)
    o, c = np.r_[p[0], p[:-1]], p
    h, l = np.maximum(o, c), np.minimum(o, c)
    return o, h, l, c, spot.Crossing(h, True), spot.Crossing(l, False)


def test_signal_compares_the_close_with_an_earlier_open():
    opens, closes = np.array([10.0, 12, 11, 13]), np.array([11.0, 11.5, 10.5, 11.5])
    d = regime.Daily(np.arange(4), opens, np.zeros(4), np.zeros(4), closes)
    assert list(dailydip.signal(d, 1)) == [False, False, True, False]  # 10,5 < 12 (a abertura do dia anterior)
    assert list(dailydip.signal(d, 2)) == [False, False, False, True]  # 11,5 < 12 (a abertura de dois dias antes)


def test_target_stop_and_open_trades():
    o, h, l, c, up, dn = bars([100, 100, 101, 102.5, 101, 99, 100])
    day_end = np.array([1, 4])
    t = dailydip.simulate(o, c, up, dn, day_end, np.array([True, True]), 0.02, 0)
    assert [(x.entry, x.exit, x.outcome) for x in t] == [(1, 3, "alvo"), (4, 6, "aberto")]
    assert t[0].exit_px == pytest.approx(102.0) and t[0].ret(FEE) == pytest.approx(1.02 * 0.999 ** 2 - 1)
    o, h, l, c, up, dn = bars([100, 100, 95, 93, 99])
    (s,) = dailydip.simulate(o, c, up, dn, np.array([1]), np.array([True]), 0.02, 3)
    assert s.outcome == "stop" and s.exit_px == pytest.approx(94.0)


def test_equity_marks_the_open_trade():
    o, h, l, c, up, dn = bars([100, 100, 101, 101.5, 102.5])
    trades = dailydip.simulate(o, c, up, dn, np.array([1]), np.array([True]), 0.02, 0)
    eq, in_market = dailydip.daily_equity(c, np.array([1, 3, 4]), trades)
    assert eq[0] == 1.0 and eq[1] == pytest.approx(101.5 / 100 * 0.999 ** 2) and eq[2] == pytest.approx(1 + trades[0].ret())
    assert in_market == pytest.approx(1 / 3)


def test_luck_p_rewards_a_signal_that_knows_the_future():
    rng = np.random.default_rng(0)
    prices = list(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 4000))))
    o, h, l, c, up, dn = bars(prices)
    day_end = np.arange(10, 3990, 20)
    future = np.array([max(c[i + 1:i + 40]) >= c[i] * 1.02 and min(c[i + 1:i + 40]) > c[i] * 0.95 for i in day_end])
    assert dailydip.luck_p(o, c, up, dn, day_end, future, 0.02, 3, n=100) < 0.05
