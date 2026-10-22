from datetime import datetime, timedelta, timezone

import pytest

from wescalp import features
from wescalp.kucoin import Candle
from wescalp.labels import Barriers, label_candles
from wescalp.research import stats, trades_for
from wescalp.setups import breakout, pullback, vwap_reversion

T0 = datetime(2026, 10, 1, tzinfo=timezone.utc)


def candle(i: int, o: float, h: float, l: float, c: float, v: float = 1.0) -> Candle:
    return Candle("futures", "XBTUSDTM", T0 + timedelta(minutes=5 * i), o, h, l, c, v, v * c)


def test_ema_starts_at_first_value():
    assert features.ema([10, 10, 10], 9) == [10, 10, 10]
    e = features.ema([10, 20], 3)
    assert e[1] == pytest.approx(15)


def test_vwap_resets_each_utc_day_and_keeps_previous_day_range():
    day = 288  # candles de 5 min por dia
    cs = [candle(i, 100, 110, 90, 100, v=1) for i in range(day)]
    cs += [candle(day, 200, 202, 198, 200, v=1), candle(day + 1, 200, 203, 197, 200, v=3)]
    f = features.compute(cs)
    assert f["prev_high"][day - 1] is None
    assert (f["prev_high"][day], f["prev_low"][day]) == (110, 90)
    assert f["vwap"][day] == pytest.approx(200)
    assert f["vwap"][day + 1] == pytest.approx(200)
    assert f["vwap_sd"][day + 1] == pytest.approx(0, abs=1e-3)


def test_previous_day_needs_consecutive_day():
    cs = [candle(0, 100, 110, 90, 100), candle(2 * 288, 100, 101, 99, 100)]
    assert features.compute(cs)["prev_high"][1] is None


def test_vol_ratio_uses_only_previous_candles():
    cs = [candle(i, 100, 101, 99, 100, v=2) for i in range(20)] + [candle(20, 100, 101, 99, 100, v=6)]
    f = features.compute(cs, vol_period=20)
    assert f["vol_ratio"][19] is None and f["vol_ratio"][20] == pytest.approx(3)


def cols(**overrides):
    base = {"prev_high": [None, 110], "prev_low": [None, 90], "vol_ratio": [None, 2.0],
            "vwap": [None, 100], "vwap_sd": [None, 1], "atr": [None, 1], "ema9": [None, 100], "ema21": [None, 99]}
    base.update(overrides)
    return base


def test_breakout_needs_close_beyond_level_and_volume():
    cs = [candle(0, 108, 109, 107, 109), candle(1, 109, 112, 109, 111)]
    assert breakout(cs, cols(), 1) == "long"
    assert breakout(cs, cols(vol_ratio=[None, 1.2]), 1) is None
    cs = [candle(0, 92, 93, 91, 91), candle(1, 91, 91, 88, 89)]
    assert breakout(cs, cols(), 1) == "short"


def test_vwap_reversion_needs_close_back_inside_band():
    assert vwap_reversion([None, candle(1, 98.5, 99, 97.5, 98.6)], cols(), 1) == "long"
    assert vwap_reversion([None, candle(1, 98.5, 99, 97.5, 97.9)], cols(), 1) is None
    assert vwap_reversion([None, candle(1, 101.5, 102.5, 101, 101.2)], cols(), 1) == "short"


def test_pullback_in_uptrend_with_more_volume():
    cs = [candle(0, 101, 101, 100.5, 100.8, v=1), candle(1, 100.5, 101, 99.5, 100.6, v=2)]
    assert pullback(cs, cols(), 1) == "long"
    cs[1] = candle(1, 100.5, 101, 99.5, 100.6, v=0.5)
    assert pullback(cs, cols(), 1) is None


def test_trades_do_not_overlap_on_same_side():
    cs = [candle(i, 100, 101, 99, 100) for i in range(20)]
    b = Barriers(max_bars=3, atr_period=2)
    labels = {(lb.ts, lb.side): lb for lb in label_candles(cs, b)}
    trades = trades_for(cs, {}, labels, lambda _cs, _f, _i: "long")
    # Saída por tempo em 3 candles: entra em 1, 4, 7, 10, 13, 16.
    assert [int((t.ts - T0).total_seconds() // 300) for t in trades] == [1, 4, 7, 10, 13, 16]


def test_stats_windows_and_interval():
    cs = [candle(i, 100, 101, 99, 100) for i in range(20)]
    labels = label_candles(cs, Barriers(max_bars=3, atr_period=2))
    s = stats([lb for lb in labels if lb.side == "long"], T0)
    assert s["n"] == 16 and s["win"] == 0 and s["windows"] == 1 and s["windows_pos"] == 0
    assert s["lo_bps"] <= s["net_bps"] <= s["hi_bps"]


def test_report_body_is_json_ready():
    import json

    from wescalp.research import report, report_body

    cs = [candle(i, 100, 101, 99, 100) for i in range(40)]
    b = Barriers(max_bars=3, atr_period=2)
    body = report_body(report(cs, b), cs, b, 16)
    assert body["barriers"] == b.name and body["holdout_weeks"] == 16
    assert {r["setup"] for r in body["rows"]} == {"todos os candles", "breakout", "vwap_reversion", "pullback"}
    json.dumps(body)
