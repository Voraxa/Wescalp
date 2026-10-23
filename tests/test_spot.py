import pytest

np = pytest.importorskip("numpy")

from wescalp import brain, spot  # noqa: E402


def test_crossing_matches_brute_force():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=777))
    up, dn = spot.Crossing(x, True), spot.Crossing(x, False)
    for _ in range(500):
        start, level = int(rng.integers(0, 800)), float(rng.normal() * 10)
        want_up = next((k for k in range(start, len(x)) if x[k] >= level), None)
        want_dn = next((k for k in range(start, len(x)) if x[k] <= level), None)
        assert up.first(start, level) == want_up
        assert dn.first(start, level) == want_dn


def frame(opens, highs, lows, closes) -> brain.Frame:
    n = len(opens)
    ts = np.arange(n, dtype=np.int64) * 300
    return brain.Frame(ts, np.array(opens, float), np.array(highs, float), np.array(lows, float),
                       np.array(closes, float), np.full(n, np.nan), np.zeros((n, 1)), ["x"], {})


def flat(n: int, v: float = 100.0) -> list[float]:
    return [v] * n


def test_barrier_up_first_down_first_and_same_candle():
    n = 10
    highs, lows = flat(n, 100.5), flat(n, 99.5)
    highs[4] = 103.1  # +3% no candle 4
    f = frame(flat(n), highs, lows, flat(n))
    up, dn = spot.Crossing(f.high, True), spot.Crossing(f.low, False)
    assert spot.barrier(f, 0, up, dn, 0.03) == (1, 4)
    lows[2] = 96.9  # -3% antes
    f = frame(flat(n), highs, lows, flat(n))
    assert spot.barrier(f, 0, spot.Crossing(f.high, True), spot.Crossing(f.low, False), 0.03) == (0, 2)
    lows[2], lows[4] = 99.5, 96.9  # os dois no mesmo candle: conta como queda
    f = frame(flat(n), highs, lows, flat(n))
    assert spot.barrier(f, 0, spot.Crossing(f.high, True), spot.Crossing(f.low, False), 0.03) == (0, 4)
    f = frame(flat(n), flat(n, 100.5), flat(n, 99.5), flat(n))
    assert spot.barrier(f, 0, spot.Crossing(f.high, True), spot.Crossing(f.low, False), 0.03) is None


def test_part_return_fees_on_both_sides():
    assert spot.part_return([100.0], 103.0, fee=0.0) == pytest.approx(300 / 3)
    r = spot.part_return([100.0, 97.0], 100.0, fee=0.001)
    assert r == pytest.approx(((1.0 * 0.999 - 1.001) + (100 / 97 * 0.999 - 1.001)) / 3 * 1e4)


def test_buy_then_target():
    n = 12
    highs = flat(n, 100.5)
    highs[5] = 102.1
    f = frame(flat(n), highs, flat(n, 99.5), flat(n))
    pol = spot.simulate(f, 0, n, n, lambda k: k == 1, 0.02, fee=0.0)
    (t,) = pol.trades
    assert (t.signal_i, t.exit_i, t.outcome, t.parts) == (1, 5, "tp", 1)
    assert t.ret_bps == pytest.approx(200 / 3)
    assert pol.buys == 1


def test_dca_needs_new_signal_and_distance():
    n = 14
    closes = flat(n)
    closes[3], closes[4] = 98.5, 97.9  # 1,5% (pouco) e 2,1% abaixo da compra
    opens = flat(n)
    opens[5] = 98.0
    highs, lows = flat(n, 100.5), flat(n, 97.5)
    highs[8] = 101.0  # médio (100 + 98) / 2 = 99, alvo 2% acima = 100,98
    f = frame(opens, highs, lows, closes)
    sig = {0, 3, 4}
    pol = spot.simulate(f, 0, n, n, lambda k: k in sig, 0.02, fee=0.0)
    (t,) = pol.trades
    assert t.parts == 2 and t.outcome == "tp" and t.exit_i == 8  # o sinal do candle 3 não compra: perto demais
    assert t.ret_bps == pytest.approx(((100.98 / 100 - 1) + (100.98 / 98 - 1)) / 3 * 1e4)


def test_fall_without_signal_does_not_buy_more():
    n = 14
    closes = flat(n, 95.0)
    f = frame(flat(n), flat(n, 100.5), flat(n, 95.0), closes)
    pol = spot.simulate(f, 0, n, n, lambda k: k == 0, 0.02, fee=0.0)
    assert pol.buys == 1


def test_entry_stop_closes_everything():
    n = 12
    lows = flat(n, 99.5)
    lows[6] = 93.9  # stop 3·2% abaixo da entrada: 94
    f = frame(flat(n), flat(n, 100.5), lows, flat(n))
    (t,) = spot.simulate(f, 0, n, n, lambda k: k == 0, 0.02, fee=0.0).trades
    assert (t.outcome, t.exit_i) == ("sl", 6)
    assert t.ret_bps == pytest.approx(-600 / 3)


def test_open_position_closed_at_end_and_drawdown_marked_to_market():
    n = 6
    closes = [100.0, 100.0, 97.0, 99.0, 98.0, 99.5]
    f = frame(flat(n), flat(n, 100.5), flat(n, 96.5), closes)
    pol = spot.simulate(f, 0, n, n, lambda k: k == 0, 0.05, fee=0.0)
    (t,) = pol.trades
    assert t.outcome == "aberto" and t.exit_i == n - 1
    assert pol.max_dd_bps == pytest.approx(300 / 3)  # do topo (0) ao fechamento de 97 com 1/3 comprado
    assert pol.bars_in_market == n - 1


def test_no_new_decisions_after_hi_but_exits_continue():
    n = 12
    highs = flat(n, 100.5)
    highs[9] = 102.5
    f = frame(flat(n), highs, flat(n, 99.5), flat(n))
    pol = spot.simulate(f, 0, 3, n, lambda k: True, 0.02, fee=0.0)
    assert [(t.signal_i, t.outcome, t.exit_i) for t in pol.trades] == [(0, "tp", 9)]


def test_choose_cuts_profit_cut_and_two_per_day_cut(monkeypatch):
    n = 2880  # 10 dias de candles de 5 min; um candidato a cada 18 candles = 16 por dia
    f = frame(flat(n), flat(n), flat(n), flat(n))
    cand = np.arange(n) % 18 == 0
    p = np.where(cand, (np.arange(n) // 18) / 160, np.nan)

    def ret(j: int) -> float:
        if j >= 144:
            return 1000.0
        if j == 140:
            return -100.0
        return 1.0 if j >= 128 else -10.0

    def fake(f, lo, hi, end, signal, tp, *a, **k):
        trades = [spot.Trade(i, i, "tp", 1, ret(i // 18), None) for i in range(lo, hi) if signal(i)]
        return spot.Policy(trades, 0.0, 0, len(trades))

    monkeypatch.setattr(spot, "simulate", fake)
    (c1, ok1), (c2, ok2) = spot.choose_cuts(f, 0, n, cand, p, 0.01)
    scores = p[cand]
    # Maior lucro: os 16 melhores (1,6/dia). Com ≥ 2/dia: desce até os 32 melhores (3,2/dia, 31 de 32 no lucro).
    assert ok1 and c1 == pytest.approx(np.quantile(scores, 0.9))
    assert ok2 and c2 == pytest.approx(np.quantile(scores, 0.8))


def test_choose_cuts_flags_when_unreachable(monkeypatch):
    n = 2880
    f = frame(flat(n), flat(n), flat(n), flat(n))
    cand = np.arange(n) % 18 == 0
    p = np.where(cand, (np.arange(n) // 18) / 160, np.nan)

    def fake(f, lo, hi, end, signal, tp, *a, **k):  # metade perde em qualquer corte
        trades = [spot.Trade(i, i, "tp", 1, 1.0 if (i // 18) % 2 else -1.0, None) for i in range(lo, hi) if signal(i)]
        return spot.Policy(trades, 0.0, 0, len(trades))

    monkeypatch.setattr(spot, "simulate", fake)
    (c1, ok1), (c2, ok2) = spot.choose_cuts(f, 0, n, cand, p, 0.01)
    assert not ok1 and not ok2 and np.isfinite(c1) and np.isfinite(c2)


def test_yearly_and_hold_by_calendar_year():
    t0 = 1_767_225_000  # 2025-12-31 23:50 UTC
    ts = [t0, t0 + 300, t0 + 600, t0 + 900]
    f = brain.Frame(np.array(ts, dtype=np.int64), np.array([100.0, 101.0, 102.0, 104.0]), np.full(4, 110.0),
                    np.full(4, 90.0), np.array([101.0, 102.0, 103.0, 105.06]), np.full(4, np.nan),
                    np.zeros((4, 1)), ["x"], {})
    hold = spot.hold_yearly(f, 0, 4)
    assert hold[2025] == pytest.approx(2.0) and hold[2026] == pytest.approx(3.0)
    pol = spot.Policy([spot.Trade(0, 1, "tp", 1, 150.0, None), spot.Trade(1, 3, "sl", 1, -50.0, None)], 0, 0, 2)
    assert spot.yearly(pol, f) == {2025: pytest.approx(1.5), 2026: pytest.approx(-0.5)}


def test_recorded_equity_compounds_closed_cycles_and_marks_open_one():
    n = 12
    highs = flat(n, 100.5)
    highs[3] = 102.1
    closes = flat(n)
    closes[10], closes[11] = 99.0, 98.0
    f = frame(flat(n), highs, flat(n, 97.5), closes)
    rec = np.zeros(n, dtype=bool)
    rec[[3, 11]] = True
    pol = spot.simulate(f, 0, n, n, lambda k: k in (0, 8), 0.02, fee=0.0, record=rec)
    first = pol.trades[0].ret_bps / 1e4
    assert pol.equity[0] == pytest.approx(1 + first)  # o 1º ciclo fecha no candle 3 (alvo)
    # o 2º compra na abertura do 9 (100) e é fechado no fim (98): mesma conta marcada a mercado
    assert pol.equity[1] == pytest.approx((1 + first) * (1 + spot.part_return([100.0], 98.0, 0.0) / 1e4))
