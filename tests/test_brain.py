import math
from datetime import datetime, timezone

import pytest

np = pytest.importorskip("numpy")

from wescalp import brain  # noqa: E402
from wescalp.kucoin import Candle  # noqa: E402

N = 20
CY = brain.Cycle(tp=0.75, dca=0.5, sl=2.5, max_bars=10)  # ATR de 1 h = 1: DCA em 99,5 e 99, stop em 97,5


def frame(lows, highs, opens=None, closes=None, ts=None) -> brain.Frame:
    n = len(lows)
    ts = np.arange(n, dtype=np.int64) * 300 if ts is None else np.array(ts, dtype=np.int64)
    o = np.array(opens if opens is not None else [100.0] * n)
    c = np.array(closes if closes is not None else [100.0] * n)
    return brain.Frame(ts, o, np.array(highs, dtype=float), np.array(lows, dtype=float), c, np.full(n, 1.0),
                       np.zeros((n, 1)), ["x"], {})


def flat(v: float) -> list[float]:
    return [v] * N


def test_target_from_first_entry():
    f = frame(flat(99.9), [100.1, 100.1, 100.2, 100.8] + flat(100.1)[4:])
    r = brain.simulate_cycle(f, 0, CY, maker=0.0002, taker=0.0006)
    assert (r.outcome, r.exit_i, r.fills) == ("tp", 3, 1)
    assert r.gross_bps == pytest.approx(75 / 3)  # 0,75% numa parte de 3
    assert r.ret_bps == pytest.approx((75 - 6 - 2) / 3)  # entrada taker, alvo maker


def test_dca_lowers_average_and_target():
    lows = [99.9, 99.9, 98.9] + flat(99.9)[3:]
    highs = [100.1, 100.1, 99.6, 100.3] + flat(100.1)[4:]
    r = brain.simulate_cycle(frame(lows, highs), 0, CY, maker=0.0002, taker=0.0006)
    assert (r.outcome, r.exit_i, r.fills) == ("tp", 3, 3)  # médio 99,5, alvo 100,25
    gross = (100.25 / 100 + 100.25 / 99.5 + 100.25 / 99 - 3) / 3 * 1e4
    assert r.gross_bps == pytest.approx(gross)
    assert r.ret_bps == pytest.approx(gross - (6 + 2 + 2 + 3 * 2) / 3)


def test_no_target_on_a_candle_that_bought():
    # Candle 2 compra em 99,5 (médio 99,75, alvo 100,5) e também chega a 100,6: não dá para saber a ordem.
    lows = [99.9, 99.9, 99.4] + flat(99.9)[3:]
    highs = [100.1, 100.1, 100.6, 100.6] + flat(100.1)[4:]
    r = brain.simulate_cycle(frame(lows, highs), 0, CY)
    assert r.fills == 2 and r.exit_i == 3 and r.outcome == "tp"  # o alvo só conta no candle 3


def test_stop_closes_all_parts():
    lows = [99.9, 99.9, 99.2, 97.4] + flat(99.9)[4:]
    r = brain.simulate_cycle(frame(lows, flat(100.1)), 0, CY, maker=0.0002, taker=0.0006)
    assert (r.outcome, r.exit_i, r.fills) == ("sl", 3, 3)
    gross = (97.5 / 100 + 97.5 / 99.5 + 97.5 / 99 - 3) / 3 * 1e4
    assert r.ret_bps == pytest.approx(gross - (6 + 2 + 2 + 3 * 6) / 3)


def test_gap_below_stop_exits_at_open():
    lows = [99.9, 99.9, 96.0] + flat(99.9)[3:]
    opens = [100.0, 100.0, 97.0] + flat(100.0)[3:]
    r = brain.simulate_cycle(frame(lows, flat(100.1), opens=opens), 0, CY, maker=0.0, taker=0.0)
    assert r.outcome == "sl" and r.fills == 3
    assert r.gross_bps == pytest.approx((97 / 100 + 97 / 97 + 97 / 97 - 3) / 3 * 1e4)  # DCA e stop na abertura


def test_time_exit_at_close():
    closes = flat(100.0)
    closes[10] = 100.3
    r = brain.simulate_cycle(frame(flat(99.9), flat(100.4), closes=closes), 0, CY, maker=0.0, taker=0.0)
    assert (r.outcome, r.exit_i) == ("time", 10)
    assert r.gross_bps == pytest.approx(30 / 3)


def test_no_cycle_across_a_gap():
    ts = [i * 300 for i in range(N)]
    ts[5:] = [t + 300 for t in ts[5:]]  # falta um candle
    assert brain.simulate_cycle(frame(flat(99.9), flat(100.1), ts=ts), 0, CY) is None
    assert brain.simulate_cycle(frame(flat(99.9), flat(100.1)), N - 5, CY) is None  # sem candles à frente


def test_higher_timeframe_only_after_bar_closes():
    n = 36  # três horas de candles de 5 min, alinhadas na hora
    ts = np.arange(n, dtype=np.int64) * 300
    o, c, h, l = np.full(n, 100.0), np.full(n, 100.0), np.full(n, 101.0), np.full(n, 99.0)
    h[3], l[5], c[11] = 110.0, 90.0, 105.0  # hora 0: mínima 90, máxima 110, fecha 105 → posição 0,75
    h[14], l[16], c[23] = 105.0, 95.0, 96.0  # hora 1: posição 0,1
    v = np.ones(n)
    pos = brain.higher(ts, o, h, l, c, v, 60)["pos_range"]
    assert math.isnan(pos[10])  # fecha às 00:55: a hora 0 ainda não fechou
    assert pos[11] == pytest.approx(0.75)  # fecha às 01:00, junto com a hora 0
    assert pos[22] == pytest.approx(0.75)
    assert pos[23] == pytest.approx(0.1)


def test_higher_timeframe_skips_incomplete_bar():
    n = 36
    keep = [i for i in range(n) if i != 20]  # falta um candle na hora 1
    ts = np.arange(n, dtype=np.int64)[keep] * 300
    ones = np.ones(len(keep))
    pos = brain.higher(ts, ones * 100, ones * 101, ones * 99, ones * 100, ones, 60)["pos_range"]
    assert pos[keep.index(11)] == pytest.approx(0.5)
    assert math.isnan(pos[keep.index(23)])


def test_candle_cache_round_trip(tmp_path):
    cs = [Candle("futures", "XBTUSDTM", datetime.fromtimestamp(1_735_689_600 + 300 * i, timezone.utc),
                 100.0 + i, 101.5 + i, 99.25 + i, 100.5 + i, 1234.0, 98765.4321) for i in range(3)]
    path = tmp_path / "c.csv.gz"
    brain.save_candles(path, cs)
    assert brain.load_candles(path) == cs


def R(i: int, exit_i: int, ret: float = 1.0) -> brain.Result:
    return brain.Result(i, exit_i, "tp", 1, ret, ret)


def test_sequence_one_cycle_at_a_time():
    res = {0: R(0, 4), 2: R(2, 3), 4: R(4, 7), 6: R(6, 8), 9: R(9, 10)}
    day = np.zeros(20, dtype=int)
    out = brain.sequence(sorted(res), res.get, lambda i: True, day)
    assert [r.i for r in out] == [0, 4, 9]  # o 4 entra: o sinal é no fechamento do candle em que o 0 saiu


def test_sequence_forces_minimum_per_day_after_hour():
    res = {i: R(i, i) for i in range(6)}
    day = np.zeros(6, dtype=int)
    hour = np.array([10, 15, 16, 17, 18, 19])
    out = brain.sequence(range(6), res.get, lambda i: False, day, hour, lambda i: True)
    assert [r.i for r in out] == [2, 3]  # só depois das 16 h e só até completar 2 no dia


def test_choose_cut_best_total_within_targets():
    idx = list(range(60))
    p = {i: i / 60 for i in idx}
    res = {i: R(i, i, 10.0 if i >= 30 else -10.0) for i in idx}
    day = np.repeat(np.arange(10), 6)
    cut, ok = brain.choose_cut(p, idx, res.get, day, n_days=10)
    # corte na mediana: 30 ciclos (3/dia), 100% de acerto, total 300; mais baixo cai abaixo de 85% de acerto,
    # mais alto dá menos lucro ou menos de 2 por dia
    assert ok and cut == pytest.approx(np.quantile(list(p.values()), 0.5))


def test_choose_cut_flags_when_targets_unreachable():
    idx = list(range(60))
    p = {i: i / 60 for i in idx}
    res = {i: R(i, i, 10.0 if i % 2 else -10.0) for i in idx}  # 50% de acerto em qualquer corte
    cut, ok = brain.choose_cut(p, idx, res.get, np.repeat(np.arange(10), 6), n_days=10)
    assert not ok and math.isfinite(cut)
