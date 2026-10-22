from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from wescalp.kucoin import Candle
from wescalp.labels import Barriers, label_candles
from wescalp.research import trades_for
from wescalp.sweep import BASELINE, config_returns, grid, week_rows, weekly_returns
from wescalp.validation import walk_forward_splits

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
WEEK = timedelta(weeks=1)


def candle(i: int, px: float = 100.0) -> Candle:
    return Candle("futures", "XBTUSDTM", T0 + timedelta(minutes=5 * i), px, px + 1, px - 1, px, 1.0, 1.0)


def test_grid_has_36_barriers():
    g = grid()
    assert len(g) == 36 == len({b.name for b in g})


def test_weekly_returns_sums_by_entry_week_and_drops_partial_week():
    cs = [candle(i) for i in range(20)]
    labels = [lb for lb in label_candles(cs, Barriers(max_bars=3, atr_period=2)) if lb.side == "long"]
    moved = [replace(lb, ts=T0 + k * WEEK) for k, lb in zip((0, 0, 1, 5), labels)]  # semana 5 > n_weeks
    w = weekly_returns(moved, T0, 3)
    assert w[0] == pytest.approx(2 * labels[0].ret_net) and w[1] == pytest.approx(labels[0].ret_net) and w[2] == 0
    assert weekly_returns(moved, T0, 3, "ret_gross")[0] == pytest.approx(0)  # preço parado: bruto 0


def test_config_returns_baseline_matches_trades_for():
    cs = [candle(i) for i in range(4000)]  # ~2 semanas
    b = Barriers(max_bars=3, atr_period=2)
    sig = [("long" if i % 7 == 0 else None) for i in range(len(cs))]
    rows = config_returns(cs, {"teste": sig}, b, n_weeks=1)
    by = {(r.rule, r.side): r for r in rows}
    assert len(rows) == 4  # linha de base e um setup, dois lados
    labels = {(lb.ts, lb.side): lb for lb in label_candles(cs, b)}
    base = [t for t in trades_for(cs, {}, labels, lambda *_: "long") if t.ts < T0 + WEEK]
    assert by[(BASELINE, "long")].n == len(base)
    assert by[(BASELINE, "long")].net[0] == pytest.approx(sum(t.ret_net for t in base))
    # O candle 0 não tem ATR, então não tem rótulo; 2016 candles = 1 semana.
    assert by[("teste", "short")].n == 0 and by[("teste", "long")].n == sum(s is not None for s in sig[1:2016])


def test_week_rows_follow_splits():
    splits = walk_forward_splits(T0, T0 + 70 * WEEK)
    assert week_rows(splits, T0) == [(0, 60, 60, 64), (4, 64, 64, 68)]
