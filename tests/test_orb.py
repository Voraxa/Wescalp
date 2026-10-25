from datetime import date, datetime, timedelta, timezone

import pytest

np = pytest.importorskip("numpy")

from wescalp import orb, spot  # noqa: E402

H, DAY = 3600, 86400


def utc(*a) -> int:
    return int(datetime(*a, tzinfo=timezone.utc).timestamp())


def flat_days(first: date, days: int, px: float = 100.5):
    """Candles de 5 min parados em `px`, do começo de `first` (UTC) por `days` dias."""
    t0 = utc(first.year, first.month, first.day)
    ts = np.arange(t0, t0 + days * DAY, 300, dtype=np.int64)
    p = np.full(len(ts), px)
    return ts, p.copy(), p.copy(), p.copy(), p.copy()


def candle(ts, o, h, l, c, t, oo, hh, ll, cc):
    i = int(np.searchsorted(ts, t))
    o[i], h[i], l[i], c[i] = oo, hh, ll, cc


def with_range(ts, o, h, l, c, open_utc: int, minutes: int = 15):
    """Faixa de 100 a 101 nos primeiros `minutes` depois da abertura."""
    for k in range(minutes // 5):
        candle(ts, o, h, l, c, open_utc + 300 * k, 100.5, 101.0, 100.0, 100.5)


def test_range_follows_new_york_daylight_saving():
    ts, o, h, l, c = flat_days(date(2024, 7, 8), 1)
    with_range(ts, o, h, l, c, utc(2024, 7, 8, 13, 30))  # verão: 9h30 em Nova York = 13h30 UTC
    (w,) = orb.windows([date(2024, 7, 8)], ts, h, l, 15, 0, 2 ** 40)
    assert (w.high, w.low) == (101.0, 100.0) and ts[w.first] == utc(2024, 7, 8, 13, 45)
    assert ts[w.last_entry] + 300 == utc(2024, 7, 8, 16) and ts[w.last] + 300 == utc(2024, 7, 8, 20)  # 12h e 16h de NY
    ts, o, h, l, c = flat_days(date(2024, 1, 8), 1)
    with_range(ts, o, h, l, c, utc(2024, 1, 8, 14, 30))  # inverno: 9h30 = 14h30 UTC
    (w,) = orb.windows([date(2024, 1, 8)], ts, h, l, 30, 0, 2 ** 40)
    assert w.high == 101.0 and ts[w.first] == utc(2024, 1, 8, 15) and ts[w.last] + 300 == utc(2024, 1, 8, 21)


def test_day_with_a_missing_range_candle_is_skipped():
    ts, o, h, l, c = flat_days(date(2024, 7, 8), 1)
    keep = ts != utc(2024, 7, 8, 13, 35)
    assert orb.windows([date(2024, 7, 8)], ts[keep], h[keep], l[keep], 15, 0, 2 ** 40) == []


def day_with_breakout(after_breakout):
    """Faixa de 100 a 101; às 10h30 de NY (14h30 UTC) um candle fecha a 101,2, acima da faixa."""
    ts, o, h, l, c = flat_days(date(2024, 7, 8), 1)
    with_range(ts, o, h, l, c, utc(2024, 7, 8, 13, 30))
    candle(ts, o, h, l, c, utc(2024, 7, 8, 14, 30), 100.5, 101.2, 100.5, 101.2)
    after_breakout(ts, o, h, l, c)
    w = orb.windows([date(2024, 7, 8)], ts, h, l, 15, 0, 2 ** 40)
    return ts, o, h, l, c, w


def run(ts, o, h, l, c, w, k):
    return orb.orb_trades(w, o, c, spot.Crossing(h, True), spot.Crossing(l, False), k)[0]


def test_entry_target_and_risk():
    def up(ts, o, h, l, c):
        later = ts >= utc(2024, 7, 8, 14, 40)
        o[later], h[later], l[later], c[later] = 101.5, 101.5, 101.5, 101.5  # fica acima do stop até o fim do dia
        candle(ts, o, h, l, c, utc(2024, 7, 8, 14, 35), 101.2, 102.0, 101.1, 101.8)
    ts, o, h, l, c, w = day_with_breakout(up)
    (t,) = run(ts, o, h, l, c, w, 1)
    assert ts[t.entry] == utc(2024, 7, 8, 14, 30) and t.entry_px == 101.2 and t.risk == pytest.approx(0.7)
    assert t.outcome == "alvo" and t.exit_px == pytest.approx(101.9)  # stop no meio da faixa, 100,5; alvo 1R
    (t2,) = run(ts, o, h, l, c, w, 2)
    assert t2.outcome == "fechamento" and t2.exit_px == c[w[0].last]  # 2R (102,6) não bateu: vende às 16h


def test_stop_comes_before_target_in_the_same_candle():
    def wide(ts, o, h, l, c):
        candle(ts, o, h, l, c, utc(2024, 7, 8, 14, 35), 101.2, 103.0, 100.0, 101.0)
    ts, o, h, l, c, w = day_with_breakout(wide)
    (t,) = run(ts, o, h, l, c, w, 1)
    assert t.outcome == "stop" and t.exit_px == pytest.approx(100.5)


def test_no_entry_after_noon_in_new_york():
    ts, o, h, l, c = flat_days(date(2024, 7, 8), 1)
    with_range(ts, o, h, l, c, utc(2024, 7, 8, 13, 30))
    candle(ts, o, h, l, c, utc(2024, 7, 8, 16, 0), 100.5, 101.5, 100.5, 101.5)  # 12h de NY: já passou da janela
    w = orb.windows([date(2024, 7, 8)], ts, h, l, 15, 0, 2 ** 40)
    assert run(ts, o, h, l, c, w, 1) == []


def test_luck_p_rewards_entering_on_the_breakout():
    """Nos dias de teste, o preço só sobe logo depois do rompimento: entrar em outra hora da janela não pega a alta."""
    days = [date(2024, 7, 8) + timedelta(days=i) for i in range(40)]
    ts, o, h, l, c = flat_days(days[0], 40)
    for d in days:
        t930 = utc(d.year, d.month, d.day, 13, 30)
        with_range(ts, o, h, l, c, t930)
        candle(ts, o, h, l, c, t930 + 3600, 100.5, 101.2, 100.5, 101.2)  # rompe às 10h30 de NY
        i = int(np.searchsorted(ts, t930 + 3900))
        o[i:i + 3], h[i:i + 3], l[i:i + 3], c[i:i + 3] = 101.2, 102.0, 101.2, 101.9  # alvo de 1R logo em seguida
        j = i + 3
        last = int(np.searchsorted(ts, t930 + int(6.5 * H)))
        o[j:last], h[j:last], l[j:last], c[j:last] = 101.9, 101.9, 101.9, 101.9  # depois fica parado
    up, dn = spot.Crossing(h, True), spot.Crossing(l, False)
    trades, used = orb.orb_trades(orb.windows(days, ts, h, l, 15, 0, 2 ** 40), o, c, up, dn, 1)
    assert len(trades) == 40 and all(t.outcome == "alvo" for t in trades)
    assert orb.luck_p(trades, used, o, c, up, dn, 1, n=200) < 0.02
