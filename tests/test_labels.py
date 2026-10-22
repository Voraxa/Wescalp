from datetime import datetime, timedelta, timezone

import pytest

from wescalp.config import MAKER_FEE, TAKER_FEE
from wescalp.kucoin import Candle
from wescalp.labels import Barriers, atr_series, label_candles, run_ends, summarize

T0 = datetime(2026, 10, 1, tzinfo=timezone.utc)


def candle(i: int, o: float, h: float, l: float, c: float) -> Candle:
    return Candle("futures", "XBTUSDTM", T0 + timedelta(minutes=5 * i), o, h, l, c, 1.0, 1.0)


def flat(n: int, start: int = 0, px: float = 100.0) -> list[Candle]:
    """Candles com faixa de 2 (ATR = 2) em torno de px."""
    return [candle(start + i, px, px + 1, px - 1, px) for i in range(n)]


# ATR de 2 com período 2: tp 1.5 ATR = 3, sl 1 ATR = 2, a partir do fechamento 100.
B = Barriers(tp_atr=1.5, sl_atr=1.0, max_bars=3, atr_period=2)


def labels_at(candles, i):
    return {lb.side: lb for lb in label_candles(candles, B) if lb.ts == candles[i].ts}


def test_atr_wilder():
    cs = [candle(0, 10, 12, 10, 11), candle(1, 11, 14, 11, 13), candle(2, 13, 13, 9, 10)]
    # TR: 2, 3 (14-11), 4 (13-9) → ATR(2): None, 2.5, (2.5 + 4) / 2
    assert atr_series(cs, 2) == [None, 2.5, 3.25]


def test_atr_restarts_after_gap():
    cs = flat(2) + flat(2, start=5)
    assert atr_series(cs, 2) == [None, 2.0, None, 2.0]


def test_run_ends():
    cs = flat(3) + flat(2, start=10)
    assert run_ends(cs) == [2, 2, 2, 4, 4]


def test_long_hits_target_short_hits_stop():
    cs = flat(2) + [candle(2, 100, 101, 99.5, 100.5), candle(3, 100.5, 103.2, 100, 103), candle(4, 103, 103, 102, 102)]
    lb = labels_at(cs, 1)
    assert (lb["long"].outcome, lb["long"].bars, lb["long"].exit_px) == ("tp", 2, 103.0)
    assert lb["long"].ret_gross == pytest.approx(0.03)
    assert lb["long"].ret_net == pytest.approx(0.03 - 2 * MAKER_FEE)
    assert (lb["short"].outcome, lb["short"].bars, lb["short"].exit_px) == ("sl", 2, 102.0)
    assert lb["short"].ret_net == pytest.approx(-0.02 - MAKER_FEE - TAKER_FEE)


def test_same_candle_touching_both_counts_as_stop():
    cs = flat(2) + [candle(2, 100, 104, 97, 100)] + flat(2, start=3)
    lb = labels_at(cs, 1)
    assert lb["long"].outcome == "sl" and lb["short"].outcome == "sl"


def test_time_exit_at_close_of_last_bar():
    cs = flat(2) + flat(2, start=2) + [candle(4, 100, 101, 99, 100.7)]
    lb = labels_at(cs, 1)
    assert (lb["long"].outcome, lb["long"].bars, lb["long"].exit_px) == ("time", 3, 100.7)
    assert lb["short"].ret_net == pytest.approx(-0.007 - MAKER_FEE - TAKER_FEE)


def test_stop_beyond_open_exits_at_open():
    # Depois de um candle que fecha em 101, o seguinte abre em 97.5, abaixo do stop de 98.
    cs = flat(2) + [candle(2, 100, 101, 99.5, 101), candle(3, 97.5, 98, 97, 97.5), candle(4, 97, 98, 96, 97)]
    assert labels_at(cs, 1)["long"].exit_px == 97.5


def test_no_label_when_window_crosses_gap_or_end():
    cs = flat(4) + flat(4, start=10)
    labeled = {lb.ts for lb in label_candles(cs, B)}
    # Índice 0 não tem ATR; a janela do 1 (2, 3, 4) atravessa o buraco; depois dele o ATR recomeça.
    assert labeled == set()
    cs = flat(6)
    assert sorted({lb.ts for lb in label_candles(cs, B)}) == [cs[1].ts, cs[2].ts]


def test_summary():
    cs = flat(2) + [candle(2, 100, 101, 99.5, 100.5), candle(3, 100.5, 103.2, 100, 103), candle(4, 103, 103, 102, 102)]
    rows = {r["side"]: r for r in summarize(label_candles(cs, B))}
    assert rows["long"]["n"] == 1 and rows["long"]["tp"] == 1.0 and rows["long"]["win"] == 1.0
    assert rows["short"]["sl"] == 1.0 and rows["short"]["net_bps"] < -200
