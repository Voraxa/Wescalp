import math

import pytest

np = pytest.importorskip("numpy")

from wescalp import bounce, intraday, regime  # noqa: E402


def make_asset(close, first: int = 0) -> bounce.Asset:
    close = np.asarray(close, dtype=float)
    return bounce.asset(regime.Daily(np.arange(first, first + len(close)), close, close, close, close))


def test_rsi_matches_a_hand_computation():
    r = bounce.rsi(np.array([10, 11, 10.5, 12, 11, 11.5]))
    assert np.isnan(r[:2]).all()
    assert r[2:] == pytest.approx([200 / 3, 800 / 9, 100 - 100 / (1 + 0.5 / 0.5625), 64.0])


def test_drop_z_uses_the_volatility_before_the_drop_and_never_the_future():
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 200)))
    z = bounce.drop_z(close, 3)
    t = 120
    lr = np.diff(np.log(close))
    assert z[t] == pytest.approx(math.log(close[t] / close[t - 3]) / (lr[t - 33:t - 3].std(ddof=1) * math.sqrt(3)))
    assert np.isnan(z[:33]).all() and not np.isnan(z[33])
    later = close.copy()
    later[t + 1:] *= 0.5
    assert np.array_equal(bounce.drop_z(later, 3)[:t + 1], z[:t + 1], equal_nan=True)


def test_every_combo_names_a_signal_and_a_filter():
    a = make_asset(100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.02, 300))))
    cs = bounce.combos()
    assert len(cs) == 108 and len(set(cs)) == 108
    assert all(c.signal in a.signals and c.filter in a.filters for c in cs)


def test_trades_go_one_at_a_time_and_chain_on_the_exit_day():
    a = make_asset(np.linspace(100, 120, 40))
    sig = np.zeros(40, dtype=bool)
    sig[[2, 3, 5, 9, 30]] = True
    a.signals["teste"] = sig
    trs = bounce.trades(a, bounce.Combo("teste", bounce.FILTERS[0], 3), 0, 40, 39)
    assert trs == [(2, 5), (5, 8), (9, 12), (30, 33)]
    h = bounce.held(trs, 40)
    assert h[3:9].all() and h[2] == 0 and h[9] == 0 and h[10:13].all()
    r, _ = bounce.daily(a, trs, 0, 40)
    fees = a.rets() * h - r  # a emenda no dia 5 não paga taxa
    assert np.flatnonzero(np.abs(fees) > 1e-12).tolist() == [3, 9, 10, 13, 31, 34]


def test_exit_on_the_five_day_average_or_after_twenty_days():
    a = make_asset([10, 10, 10, 10, 10, 9, 8, 8.5, 9.5, 11, 12, 12])
    assert bounce.exit_day(a, 6, "média de 5", 11) == 8  # 9,5 acima de (10 + 9 + 8 + 8,5 + 9,5) / 5 = 9
    falling = make_asset(np.linspace(100, 50, 40))
    assert bounce.exit_day(falling, 5, "média de 5", 39) == 25
    assert bounce.exit_day(falling, 30, "média de 5", 39) == 39
    assert bounce.exit_day(falling, 30, 5, 33) == 33


def test_choose_never_looks_past_the_training_end():
    close = 100 * np.exp(np.cumsum(np.random.default_rng(3).normal(0, 0.03, 700)))
    first = bounce.choose(make_asset(close), 200, 500, min_trades=5)
    later = close.copy()
    later[500:] *= np.linspace(1, 3, 200)
    assert bounce.choose(make_asset(later), 200, 500, min_trades=5) == first


def test_apply_carries_an_open_trade_into_the_next_year():
    n = 32
    a = make_asset(np.linspace(100, 110, n), first=intraday.utc("2019-12-20") // 86400)
    one, two = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    one[10] = True  # 30/12/2019
    two[[13, 15]] = True  # 02/01/2020, ignorado (já comprado), e 04/01/2020, dia da venda: emenda
    a.signals.update({"um": one, "dois": two})
    c1, c2 = bounce.Combo("um", bounce.FILTERS[0], 5), bounce.Combo("dois", bounce.FILTERS[0], 3)
    trs = bounce.apply(a, [(2019, c1, 0.0, 0), (2020, c2, 0.0, 0)], intraday.utc("2019-12-20"),
                       intraday.utc("2020-01-21"))
    assert trs == [(10, 15, c1), (15, 18, c2)]


def test_asset_refuses_a_missing_day():
    close = np.linspace(100, 110, 50)
    day = np.arange(50)
    day[30:] += 1
    with pytest.raises(AssertionError):
        bounce.asset(regime.Daily(day, close, close, close, close))
