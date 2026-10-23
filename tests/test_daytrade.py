import math

import pytest

np = pytest.importorskip("numpy")

from wescalp import daytrade, spot  # noqa: E402

G = daytrade.Geometry(tp=0.01, step=0.01, stop=0.03, bars=20)
FEE = 0.001


def bars(path: list[tuple[float, float, float, float]]):
    o, h, l, c = (np.array(x, dtype=float) for x in zip(*path))
    return o, h, l, c, spot.Crossing(h, True), spot.Crossing(l, False)


def flat(n: int, px: float = 100.0):
    return [(px, px, px, px)] * n


def test_target_without_dca():
    o, h, l, c, up, dn = bars(flat(1) + [(100, 100.5, 99.8, 100.2), (100.2, 101.3, 100.1, 101.0)] + flat(5, 101))
    cy = daytrade.cycle(o, h, l, c, up, dn, 0, 0.3, G)
    assert (cy.outcome, cy.exit, cy.exit_px, len(cy.fills)) == ("alvo", 2, pytest.approx(101.0), 1)
    assert cy.pnl(FEE) == pytest.approx(0.3 * (1 - FEE) * 1.01 * (1 - FEE) - 0.3)


def test_dca_moves_the_target_to_the_average_price():
    path = flat(1) + [(100, 100, 98.9, 99.0), (99, 100.2, 98.95, 100.0), (100, 100.6, 99.9, 100.5)] + flat(3, 100.5)
    o, h, l, c, up, dn = bars(path)
    cy = daytrade.cycle(o, h, l, c, up, dn, 0, 0.3, G)
    avg = 2 / (1 / 100 + 1 / 99)
    assert [p for _, p in cy.fills] == [100, 99] and cy.outcome == "alvo"
    assert cy.exit == 3 and cy.exit_px == pytest.approx(avg * 1.01)  # o candle 2 não chega no alvo novo


def test_no_target_in_the_bar_where_dca_bought():
    o, h, l, c, up, dn = bars(flat(1) + [(100, 101.5, 98.9, 100), (100, 100.3, 99.9, 100.1)] + flat(25, 100.1))
    cy = daytrade.cycle(o, h, l, c, up, dn, 0, 0.3, G)
    assert len(cy.fills) == 2 and cy.outcome == "tempo" and cy.exit == 20  # o pior primeiro


def test_gap_through_every_level_buys_all_and_stops():
    o, h, l, c, up, dn = bars(flat(1) + [(96, 96.5, 95.0, 95.5)] + flat(5, 95))
    cy = daytrade.cycle(o, h, l, c, up, dn, 0, 0.3, G)
    assert [p for _, p in cy.fills] == [100, 96, 96] and cy.outcome == "stop" and cy.exit_px == 96
    assert cy.pnl(FEE) < -0.3 * 0.03


def test_time_exit_and_mark_to_market():
    o, h, l, c, up, dn = bars(flat(1) + flat(30, 100.4))
    cy = daytrade.cycle(o, h, l, c, up, dn, 0, 0.3, G)
    assert cy.outcome == "tempo" and cy.exit == 20 and cy.exit_px == 100.4
    assert cy.mark(5, 100.4, FEE) == pytest.approx(cy.pnl(FEE))


def market(path, regime=None):
    o, h, l, c, up, dn = bars(path)
    ts = np.arange(len(c), dtype=np.int64) * 300
    return daytrade.Market(ts, o, h, l, c, up, dn, regime or {})


def rising(n_bars: int):
    """Sobe 0,2% a cada 5 min: o alvo de 1% sai em meia hora."""
    return [(100 * 1.002 ** k, 100 * 1.002 ** (k + 1), 100 * 1.002 ** k, 100 * 1.002 ** (k + 1)) for k in range(n_bars)]


def test_forced_entries_guarantee_two_cycles_a_day():
    m = market(rising(2 * 288))
    decision = np.arange(3600, 2 * 86400, 3600, dtype=np.int64)
    cycles = daytrade.run(m, decision, np.zeros(len(decision), dtype=bool), np.zeros((len(decision), 3)),
                          daytrade.Geometry(0.01, 0.01, 0.03, 144), judge=False)
    starts = [(int(m.ts[c.entry]) + 300) // 3600 for c in cycles]
    assert starts[:2] == [12, 18] and all(c.forced for c in cycles)
    assert cycles[0].size == pytest.approx(1 / 3 / 3 * daytrade.FORCED_SIZE)  # 1/3 por parte; sem regime: um terço


def test_without_dca_one_buy_with_the_whole_size():
    o, h, l, c, up, dn = bars(flat(1) + [(100, 100, 98.9, 99.0), (99, 99, 96.9, 97.0)] + flat(5, 97))
    g = daytrade.Geometry(0.01, 0.01, 0.03, 20, parts=1)
    cy = daytrade.cycle(o, h, l, c, up, dn, 0, 0.9, g)
    assert len(cy.fills) == 1 and cy.outcome == "stop" and cy.exit_px == 97.0  # nada de DCA no caminho até o stop
    m = market(rising(2 * 288))
    decision = np.arange(3600, 86400, 3600, dtype=np.int64)
    first = daytrade.run(m, decision, np.zeros(len(decision), dtype=bool), np.zeros((len(decision), 3)), g, False)[0]
    assert first.size == pytest.approx(1 / 3 * daytrade.FORCED_SIZE)  # o capital das 3 partes de uma vez


def test_daily_equity_marks_the_open_cycle():
    path = flat(288 + 12) + flat(400, 101.0)  # o preço sobe para 101 depois do primeiro dia
    m = market(path)
    cy = daytrade.cycle(m.open, m.high, m.low, m.close, m.up, m.dn, 250, 0.3, daytrade.Geometry(0.05, 0.05, 0.15, 144))
    eq = daytrade.daily_equity(m, [cy], np.array([0, 1]), FEE)
    assert eq[0] == pytest.approx(1 + cy.mark(287, 100.0, FEE)) and eq[1] == pytest.approx(1 + cy.pnl(FEE))


def test_luck_p_is_small_for_perfect_timing():
    rng = np.random.default_rng(0)
    path, px = [], 100.0
    for k in range(20 * 288):
        hour = (k // 12) % 24
        drift = 0.0012 if hour in (3, 15) else -0.0002  # sobe forte só nas horas 3 e 15
        new = px * (1 + drift + rng.normal(0, 0.0005))
        path.append((px, max(px, new) * 1.0002, min(px, new) * 0.9998, new))
        px = new
    m = market(path)
    g = daytrade.Geometry(0.004, 0.01, 0.03, 12)
    decision = np.arange(3600, 20 * 86400 - 13 * 3600, 3600, dtype=np.int64)
    signal = ((decision // 3600) % 24 == 3) | ((decision // 3600) % 24 == 15)
    cycles = daytrade.run(m, decision, signal, np.zeros((len(decision), 3)), g, judge=False)
    real = [c for c in cycles if not c.forced]
    assert len(real) >= 30 and np.mean([c.pnl(daytrade.FEE) > 0 for c in real]) > 0.8
    assert daytrade.luck_p(m, cycles, g, np.arange(20), n=100) < 0.05


def test_judge_learns_to_block_losing_signals():
    j = daytrade.Judge(1, FEE)
    for k in range(daytrade.JUDGE_MIN + 50):
        z = np.array([1.0 if k % 2 else -1.0])
        j.learn(z, 0.01 if z[0] > 0 else -0.01)
    assert j.allows(np.array([1.0])) and not j.allows(np.array([-1.0]))
    assert daytrade.Judge(1, FEE).allows(np.array([-1.0]))  # antes de 100 ciclos, não bloqueia
