import math
from datetime import date, datetime, timezone

import pytest

np = pytest.importorskip("numpy")

from wescalp import factors, refine, regime, risk  # noqa: E402

D0 = (date(2019, 1, 1) - factors.EPOCH).days


def daily(n: int, offset_h: int = 0, seed: int = 0) -> regime.Daily:
    close = 100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0.001, 0.03, n)))
    return regime.Daily(np.arange(n) + D0, close, close, close, close, offset_h * 3600)


def weekdays_series(n: int, seed: int = 1):
    days = np.arange(D0 - 400, D0 + n)
    days = days[factors.weekday(days) < 5]
    return days, 15 + np.abs(np.random.default_rng(seed).normal(0, 5, len(days)))


def test_ewma_vol_follows_the_recursion_and_ignores_the_future():
    r = np.random.default_rng(0).normal(0, 0.02, 80)
    close = 100 * np.exp(np.concatenate([[0.0], np.cumsum(r)]))
    s = risk.ewma_vol(close)
    assert np.isnan(s[:30]).all() and s[30] == pytest.approx(math.sqrt(np.mean(r[:30] ** 2)))
    assert s[31] == pytest.approx(math.sqrt(0.94 * s[30] ** 2 + 0.06 * r[30] ** 2))
    later = close.copy()
    later[50:] *= 3
    assert np.array_equal(risk.ewma_vol(later)[:50], s[:50], equal_nan=True)


def test_trailing_median_leaves_out_the_day_itself():
    out = risk.trailing_median(np.arange(10.0), 3)
    assert np.isnan(out[:3]).all() and out[3] == 1.0 and out[9] == 7.0


def test_banded_ignores_small_moves_but_not_zero_or_full():
    assert list(risk.banded(np.array([0.5, 0.52, 0.56, 1.0, 0.97, 0.0]))) == [0.5, 0.5, 0.56, 1.0, 1.0, 0.0]


def test_scaled_never_buys_with_the_frozen_rule_out():
    frozen = np.array([0.0, 1 / 3, 2 / 3, 1.0, 1.0])
    out = risk.scaled(frozen, np.array([5.0, 2.0, 2.0, 0.5, np.nan]))
    assert list(out) == pytest.approx([0.0, 2 / 3, 1.0, 0.5, 1.0])


@pytest.mark.parametrize("offset_h, bar", [(0, 400), (18, 399)])
def test_news_marks_the_decision_before_the_day_with_the_release(offset_h, bar):
    d = daily(500, offset_h)
    t = (D0 + 400) * 86400 + 13 * 3600 + 1800  # 13h30 UTC do dia 400 (CPI no horário de inverno)
    x = risk.inputs(d, weekdays_series(500), np.array([t]))
    assert list(np.flatnonzero(x.news)) == [bar - 1]  # com o dia virando às 18h UTC, a notícia cai na barra anterior
    assert np.array_equal(x.frozen, refine.position(d.close, refine.BASE))


def test_vix_ratio_uses_only_closes_already_known():
    d = daily(500)
    days, vals = weekdays_series(500)
    x = risk.inputs(d, (days, vals), np.array([], dtype=np.int64))
    k = 450
    while factors.weekday(np.array([d.day[k]]))[0] >= 5:
        k += 1
    later = np.where(days > d.day[k], vals * 3, vals)
    assert np.array_equal(risk.inputs(d, (days, later), np.array([], dtype=np.int64)).vix_ratio[:k + 1],
                          x.vix_ratio[:k + 1], equal_nan=True)
    same_day = np.where(days == d.day[k], vals * 3, vals)  # o fechamento do dia k já é conhecido na decisão do dia k
    assert risk.inputs(d, (days, same_day), np.array([], dtype=np.int64)).vix_ratio[k] != x.vix_ratio[k]


def synthetic(n: int = 900, seed: int = 2):
    d = daily(n, seed=seed)
    one = np.ones(n)
    return risk.Inputs(d, one, np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, dtype=bool))


def test_luck_p_rewards_caution_before_the_worst_days_only():
    x = synthetic()
    lo = 400
    rets = x.d.close[lo:] / x.d.close[lo - 1:-1] - 1
    worst = lo + np.argsort(rets)[:40]  # dias de pior retorno no período
    x.news[worst - 1] = True  # a decisão da véspera
    assert risk.luck_p(x, risk.CANDIDATES[1], lo, n=300) < 0.05
    assert risk.luck_p(synthetic(), risk.CANDIDATES[1], lo, n=50) == 1.0  # sem notícia, todo deslocamento é igual


def test_reaction_compares_the_hours_after_the_news_with_neighbor_days():
    start = D0 * 86400
    ts = np.arange(start, start + 40 * 86400, 300)
    close = np.full(len(ts), 100.0)
    t = start + 20 * 86400 + 12 * 3600 + 1800  # notícia no dia 20, 12h30 UTC
    close[ts >= t + 3600] = 102.0  # sobe 2% uma hora depois
    move, ctl = risk.reaction(ts, close, np.array([t]), {D0 + 20})
    assert move[0] == pytest.approx(math.log(1.02)) and ctl[0] == 0.0
    move, _ = risk.reaction(ts, close, np.array([ts[-1]]), set())  # sem candles depois: NaN
    assert np.isnan(move[0])
