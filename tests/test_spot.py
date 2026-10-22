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
