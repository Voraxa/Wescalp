from decimal import Decimal

import pytest

np = pytest.importorskip("numpy")

from wescalp import guard, refine, regime  # noqa: E402

FEE = 0.001


def test_pair_rules_are_the_ones_read_from_kucoin():
    # GET https://api.kucoin.com/api/v2/symbols, BTC-USDT, 25/10/2026
    p = guard.BTC_USDT
    assert (p.symbol, p.base_min, p.quote_min, p.base_increment, p.price_increment) == (
        "BTC-USDT", Decimal("0.00001"), Decimal("0.1"), Decimal("0.00000001"), Decimal("0.1"))


def test_floor_to_never_rounds_up():
    assert guard.floor_to(0.3, Decimal("0.1")) == Decimal("0.3")
    assert guard.floor_to(0.0312799999, Decimal("0.00000001")) == Decimal("0.03127999")
    assert guard.floor_to(97000.19, Decimal("0.1")) == Decimal("97000.1")


def test_size_by_max_loss_with_fees():
    s = guard.size(10_000, 100_000, 97_000, 0.01, 1 / 3)
    per_btc = 100_000 * 1.001 - 97_000 * 0.999  # 3.197 USDT por BTC
    assert s.limited_by == "perda" and s.qty == guard.floor_to(100 / per_btc, Decimal("0.00000001"))
    assert s.loss_at_stop == pytest.approx(float(s.qty) * per_btc) and s.loss_at_stop <= 100
    assert s.value == pytest.approx(float(s.qty) * 100_000) and s.cost == pytest.approx(s.value * 1.001)


def test_size_capped_and_below_the_pair_minimum():
    s = guard.size(10_000, 100_000, 97_000, 0.05, 1 / 3)
    assert s.limited_by == "teto" and s.qty == Decimal("0.03333333") and s.value <= 10_000 / 3
    tiny = guard.size(1, 100_000, 97_000, 0.01, 1 / 3)  # 0,01 USDT de risco: menos que o mínimo de 0,00001 BTC
    assert tiny.limited_by == "mínimo" and tiny.qty == 0 and tiny.loss_at_stop == 0
    with pytest.raises(ValueError):
        guard.size(10_000, 100_000, 100_000, 0.01, 1 / 3)


def test_stop_goes_to_the_tick_below():
    s = guard.size(10_000, 100_000, 97_000.19, 0.01, 1 / 3)
    assert s.loss_at_stop == pytest.approx(float(s.qty) * guard.loss_per_btc(100_000, 97_000.1))


def daily(closes, spread=0.02):
    c = np.array(closes, dtype=float)
    return regime.Daily(np.arange(len(c)), c, c * (1 + spread), c * (1 - spread), c)


def test_atr_wilder_by_hand():
    d = daily([10, 11, 12, 11, 13], spread=0.0)
    a = guard.atr(d.high, d.low, d.close, n=3)
    tr = [0, 1, 1, 1, 2]  # sem amplitude no dia: só o salto do fechamento anterior
    assert np.isnan(a[:2]).all() and a[2] == pytest.approx(2 / 3)
    assert a[3] == pytest.approx((2 / 3 * 2 + tr[3]) / 3) and a[4] == pytest.approx((a[3] * 2 + tr[4]) / 3)


def wave(n=500, seed=0):
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(0.0005 + 0.02 * np.sin(np.arange(n) / 25) / 5 + rng.normal(0, 0.02, n)))


def test_standard_fractions_are_the_frozen_rule():
    d = daily(wave())
    frac = guard.part_fractions(d, None)
    assert np.allclose(frac.sum(axis=1), refine.position(d.close, refine.BASE))


def test_max_loss_fractions_are_fixed_while_on_and_never_above_the_cap():
    d = daily(wave())
    frac = guard.part_fractions(d, 0.01)
    stops = guard.frozen_stops(d)
    assert (frac <= 1 / 3 + 1e-12).all() and (frac >= 0).all()
    for j in range(3):
        on = frac[:, j] > 0
        starts = np.flatnonzero(on & ~np.r_[False, on[:-1]])
        for t in starts:
            run = frac[t:, j]
            end = np.argmax(run == 0) if (run == 0).any() else len(run)
            assert np.allclose(run[:end], run[0])  # fixa até sair
            risk = guard.loss_per_btc(d.close[t], stops[t, j]) / d.close[t]
            assert run[0] == pytest.approx(min(1 / 3, 0.01 / risk))
    assert (frac[:, 0] > 0).any() and (frac.sum(axis=1) < refine.position(d.close, refine.BASE) - 1e-9).any()


def test_trade_loss_in_capital():
    d = daily([100, 100, 110, 90])
    assert guard.trade_loss(d, (0, 1, 3, 0.2)) == pytest.approx(0.2 * (0.9 - 1) - 2 * FEE * 0.2)
    frac = np.zeros((4, 3))
    frac[1:3, 0] = 0.2
    assert guard.part_trades(d, frac, 1) == [(0, 1, 3, 0.2)]


STATE = guard.State(capital=10_000, cash=10_000, btc=0.0, day_pnl=0.0, trades_today=0)


def buy(qty="0.03", price=100_000.0, stop=97_000.0, target=None):
    return guard.Order("compra", Decimal(qty), price, stop, target)


def test_good_order_passes():
    v = guard.check(buy(qty="0.03"), STATE, guard.Limits(cap=0.5))
    assert v.ok and v.reasons == ()


@pytest.mark.parametrize("order, state, limits, reason", [
    (buy(qty="0.04"), STATE, guard.Limits(cap=1.0), "perda máxima por operação"),
    (buy(), guard.State(10_000, 10_000, 0.0, -250.0, 0), guard.Limits(cap=1.0), "o dia passaria"),
    (buy(), guard.State(10_000, 10_000, 0.0, 0.0, 10), guard.Limits(cap=1.0), "operações por dia"),
    (buy(target=101_000.0), STATE, guard.Limits(cap=1.0), "alvo/risco"),
    (buy(target=99_000.0), STATE, guard.Limits(cap=1.0), "alvo precisa ficar acima"),
    (buy(), STATE, guard.Limits(cap=0.25), "teto"),
    (buy(), guard.State(10_000, 2_000, 0.08, 0.0, 0), guard.Limits(cap=1.0), "caixa"),
    (buy(qty="0.000009", stop=99_000.0), STATE, guard.Limits(), "mínimo do par"),
    (buy(qty="0.000000001"), STATE, guard.Limits(), "incremento"),
    (buy(stop=97_000.05), STATE, guard.Limits(cap=1.0), "tick"),
    (buy(stop=None), STATE, guard.Limits(), "sem stop"),
    (buy(stop=101_000.0), STATE, guard.Limits(), "sem stop"),
])
def test_each_rule_blocks_with_its_reason(order, state, limits, reason):
    v = guard.check(order, state, limits)
    assert not v.ok and any(reason in r for r in v.reasons), v.reasons


def test_sales_only_block_short_selling_and_pair_rules():
    tired = guard.State(10_000, 0.0, 0.05, -900.0, 99)  # dia ruim e cheio de operações: vender ainda pode
    assert guard.check(guard.Order("venda", Decimal("0.05"), 100_000.0), tired).ok
    v = guard.check(guard.Order("venda", Decimal("0.06"), 100_000.0), tired)
    assert not v.ok and "descoberto" in v.reasons[0]
    assert not guard.check(guard.Order("venda", Decimal("0.000001"), 100_000.0), tired).ok


def test_check_is_pure():
    order, state, limits = buy(qty="0.04"), STATE, guard.Limits(cap=1.0)
    first = guard.check(order, state, limits)
    assert guard.check(order, state, limits) == first and state == STATE  # nada guardado, nada mudado
    with pytest.raises(Exception):
        state.cash = 0  # os registros são imutáveis
