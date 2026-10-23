import math
from datetime import date, datetime, timezone

import pytest

np = pytest.importorskip("numpy")

from wescalp import sessions  # noqa: E402

DAY, H = 86400, 3600


def utc(*a) -> int:
    return int(datetime(*a, tzinfo=timezone.utc).timestamp())


def market(days: int, start: int):
    ts = np.arange(start, start + days * DAY, 300, dtype=np.int64)
    close = 100 + np.arange(len(ts)) * 0.001
    return ts, close


def test_new_york_close_and_open_follow_daylight_saving():
    assert sessions.ny(date(2024, 1, 5), sessions.time(16)) == utc(2024, 1, 5, 21)  # inverno: 16h = 21h UTC
    assert sessions.ny(date(2024, 7, 5), sessions.time(9, 30)) == utc(2024, 7, 5, 13, 30)  # verão: 9h30 = 13h30 UTC


def test_trades_of_each_rule():
    start = utc(2024, 1, 1)
    ts, close = market(10, start)
    close[ts == utc(2024, 1, 3, 0, 25)] = 50  # o candle que fecha às 0h30 do dia 3 cai: sem momento nesse dia
    tdays = [date(2024, 1, d) for d in (2, 3, 4, 5, 8)]  # terça a sexta e a segunda seguinte
    t = sessions.trades(tdays, ts, close, start, start + 10 * DAY)
    us = t["fora do pregão americano"]
    assert [tuple(x) for x in us[-1:]] == [(utc(2024, 1, 5, 21), utc(2024, 1, 8, 14, 30))]  # sexta até segunda
    late = t["21h às 23h UTC"]
    assert len(late) == 10 and tuple(late[0]) == (utc(2024, 1, 1, 21), utc(2024, 1, 1, 23))
    mom = t["momento intradiário"]
    assert utc(2024, 1, 3, 23, 30) not in mom[:, 0] and utc(2024, 1, 2, 23, 30) in mom[:, 0]
    assert mom[:, 1].max() < start + 10 * DAY  # a última saída cabe no período


def test_returns_with_fee_and_summary():
    start = utc(2024, 1, 1)
    ts, close = market(3, start)
    t = np.array([[start + 21 * H, start + 23 * H]])
    r = sessions.log_returns(ts, close, t, 0.001)
    p0, p1 = close[ts == start + 21 * H - 300][0], close[ts == start + 23 * H - 300][0]
    assert r[0] == pytest.approx(math.log(p1 / p0) + 2 * math.log(0.999))
    s = sessions.summary(t, r, start, start + 3 * DAY)
    assert s["trades"] == 1 and s["total"] == pytest.approx(math.exp(r[0]) - 1)


def test_holdout_answers_both_questions(monkeypatch):
    start, end = utc(2026, 7, 6), utc(2026, 7, 20)
    monkeypatch.setattr(sessions, "HOLDOUT", (start, end))
    ts = np.arange(start - 2 * DAY, end + 2 * DAY, 300, dtype=np.int64)
    ny_open = np.array([sessions.ny(datetime.fromtimestamp(int(t), timezone.utc).date(), sessions.time(9, 30))
                        for t in ts])  # instante da abertura de Nova York no dia de cada candle
    ny_close = ny_open + int(6.5 * H)
    us_hours = (ts >= ny_open) & (ts < ny_close)
    close = 100 * np.exp(np.cumsum(np.where(us_hours, -0.0004, 0.0002)))  # cai no pregão, sobe fora dele
    d = sessions.regime.daily_bars(ts, close, close, close, close)
    days = [date(2026, 7, k) for k in range(6, 21) if date(2026, 7, k).weekday() < 5]
    r = sessions.holdout(ts, close, d, days, luck=50)
    assert r["repeats"] and r["fees"][0.0]["total"] > r["hold"]["total"] and r["p"] < 0.05


def test_luck_p_rewards_the_one_hour_that_rises():
    start = utc(2024, 1, 1)
    ts = np.arange(start, start + 60 * DAY, 300, dtype=np.int64)
    hour = (ts // H) % 24
    rng = np.random.default_rng(0)
    steps = np.where(hour == 21, 0.002, -0.0001) + rng.normal(0, 0.0003, len(ts))
    close = 100 * np.exp(np.cumsum(steps))
    days = np.arange(start // DAY, start // DAY + 59)
    t = np.column_stack([days * DAY + 21 * H, days * DAY + 22 * H])
    assert sessions.luck_p(ts, close, t, n=200) < 0.02
