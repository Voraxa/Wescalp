import pytest

np = pytest.importorskip("numpy")

from wescalp import entries, refine, regime  # noqa: E402

FEE = entries.FEE


def daily(n: int = 400, seed: int = 0) -> regime.Daily:
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.001, 0.03, n))
    return regime.Daily(np.arange(n), close, close, close, close)


def test_window_is_a_backtest_that_starts_in_usdt_on_that_day():
    d, lo = daily(), 200
    pos = refine.position(d.close, refine.BASE)
    s = entries.series(d, lo, 400)
    for i, n in ((0, 50), (37, 50), (120, 80)):
        w = entries.window(s, i, n)
        a = lo + i
        assert w.strat == pytest.approx(np.prod(1 + regime.backtest(d.close[:a + n], pos[:a + n], a).daily))
        c = d.close
        assert w.hold == pytest.approx((c[a] / c[a - 1] - FEE) * c[a + n - 1] / c[a])
        assert w.expo == pytest.approx(pos[a - 1:a + n - 1].mean())
        assert w.same == pytest.approx(w.expo * w.hold + 1 - w.expo)


def test_drawdown_counts_the_starting_capital_as_a_peak():
    assert entries.drawdown(np.array([0.8, 1.2, 1.08])) == pytest.approx(0.2)
    assert entries.drawdown(np.array([1.1, 1.2])) == 0.0


def test_window_counts():
    s = entries.series(daily(), 200, 400)
    assert len(entries.windows(s, 50)) == 151
    ends = entries.to_end(s, min_days=30)
    assert len(ends) == 171 and ends[-1].days == 30 and ends[0].days == 200


def test_basket_averages_the_coins_and_needs_the_same_days():
    a, b = entries.series(daily(seed=1), 200, 400), entries.series(daily(seed=2), 200, 400)
    m = entries.basket([a, b])
    assert np.allclose(m.strat0, (a.strat0 + b.strat0) / 2) and np.allclose(m.expo, (a.expo + b.expo) / 2)
    d = daily(seed=3)
    shifted = entries.series(regime.Daily(d.day + 1, d.close, d.close, d.close, d.close), 200, 400)
    with pytest.raises(AssertionError):
        entries.basket([a, shifted])


def test_series_refuses_a_missing_day():
    d = daily()
    d.day[300:] += 1
    with pytest.raises(AssertionError):
        entries.series(d, 200, 400)


def test_summary_fractions():
    ws = [entries.Window(0, 365, 2.0, 1.5, 1.2, 0.1, 0.3, 0.5), entries.Window(1, 365, 0.9, 1.5, 1.25, 0.2, 0.1, 0.4)]
    r = entries.summary(ws)
    assert r["beats_hold"] == 0.5 and r["beats_same"] == 0.5 and r["less_dd"] == 0.5
    assert r["loss"] == (0.5, 0.0, 0.0) and r["expo"] == pytest.approx(0.45)
    assert r["worst"][0] == pytest.approx(0.9 ** (365.25 / 365) - 1)
