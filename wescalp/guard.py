"""Camada de execução para uma fase futura: tamanho da posição pela perda máxima e trava de risco. Só conta.

    python -m wescalp.guard --file data/candles/BINANCE-BTCUSDT-5m.csv.gz      # a pesquisa abaixo

Nada aqui envia ordem nem usa chave privada (Fase 0 do CLAUDE.md): as funções recebem números e devolvem números, sem
rede e sem efeito colateral. A ideia vem do trader-budget e do trader-risk do day-trader-skill
(github.com/Jakeschincariol/day-trader-skill): o tamanho sai de quanto se aceita perder se o stop for atingido, e
cada ordem passa por uma trava antes de existir.

Regras do par BTC-USDT spot da KuCoin, lidas da API pública (GET https://api.kucoin.com/api/v2/symbols) em
25/10/2026 às 14h39 UTC: baseMinSize 0,00001 BTC, quoteMinSize e minFunds 0,1 USDT, baseIncrement 0,00000001 BTC,
priceIncrement 0,1 USDT, feeCategory 1 (Classe A) com coeficientes 1,00, taxa cobrada em USDT. Se a KuCoin mudar,
atualize BTC_USDT e o teste que fixa esses valores.

O stop da congelada: ela não tem stop de preço; cada parte sai quando o fechamento diário cai abaixo da média dela
(50, 100 ou 150 dias). O risco de uma parte é a distância da entrada até a média, mais 1 ATR diário de 14 dias de
folga, porque entre um fechamento e o seguinte o preço pode passar bem da média. Esse "stop equivalente" (média menos
1 ATR, no dia da entrada) é o que vai para o tamanho pela perda máxima.

Pesquisa, fixada em 25/10/2026 antes de rodar, sem ajuste depois: a congelada com o tamanho de cada parte pela perda
máxima, de 11/10/2018 a 30/06/2026 (candles de 5 min da Binance, dia virando às 00h UTC, decisão no fechamento, 0,1%
por lado). Ao entrar, a parte recebe min(1/3, perda máxima / L) do capital, com L = perda por real investido se o
preço for parar no stop equivalente, com as duas taxas; a fração fica fixa até a parte sair. Perda máxima de 1% e de
2% por parte, contra a congelada de sempre (1/3 por parte). Mede retorno ao ano, queda máxima, Sharpe, exposição média
e quantas saídas com perda passaram da perda máxima. Nada é promovido: é para saber o que muda.
"""

import argparse
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import ROUND_FLOOR, Decimal
from pathlib import Path

import numpy as np

from . import refine, regime, study

FEE = 0.001  # Classe A, nível básico, por lado
PARTS = (50, 100, 150)
PART = 1 / 3
ATR_DAYS = 14
MAX_LOSSES = (0.01, 0.02)
EXAMPLE_CAPITAL = 10_000.0


@dataclass(frozen=True)
class Pair:
    symbol: str
    base_min: Decimal  # baseMinSize
    quote_min: Decimal  # quoteMinSize (= minFunds)
    base_increment: Decimal
    price_increment: Decimal


BTC_USDT = Pair("BTC-USDT", Decimal("0.00001"), Decimal("0.1"), Decimal("0.00000001"), Decimal("0.1"))


def floor_to(x: float, step: Decimal) -> Decimal:
    """Múltiplo de `step` logo abaixo de x, em Decimal (para 0,3 não virar 0,29999...)."""
    return (Decimal(repr(float(x))) / step).to_integral_value(rounding=ROUND_FLOOR) * step


def loss_per_btc(entry: float, stop: float, fee: float = FEE) -> float:
    """USDT perdidos por BTC comprado na entrada e vendido no stop, com a taxa nas duas pontas (cobrada em USDT)."""
    return entry * (1 + fee) - stop * (1 - fee)


@dataclass(frozen=True)
class Size:
    qty: Decimal  # BTC, múltiplo do incremento do par
    value: float  # qty × entrada, em USDT
    cost: float  # o que sai do caixa: valor + taxa da compra
    loss_at_stop: float  # perda se vender no stop, com as duas taxas
    limited_by: str  # "perda", "teto" ou "mínimo" (abaixo do mínimo do par: nada a comprar)


def size(capital: float, entry: float, stop: float, max_loss: float, cap: float, fee: float = FEE,
         pair: Pair = BTC_USDT) -> Size:
    """Quanto comprar para que, se o stop sair no preço, a perda seja no máximo `max_loss` do capital, sem passar de
    `cap` do capital em valor. O stop vai para o tick de baixo (onde a ordem ficaria) e a quantidade é arredondada para
    baixo, então a perda nunca passa do limite."""
    if not stop < entry:
        raise ValueError("o stop precisa ficar abaixo da entrada (aqui só se compra)")
    stop = float(floor_to(stop, pair.price_increment))
    per_btc = loss_per_btc(entry, stop, fee)
    by_loss, by_cap = max_loss * capital / per_btc, cap * capital / entry
    qty = floor_to(min(by_loss, by_cap), pair.base_increment)
    if qty < pair.base_min or qty * Decimal(repr(float(entry))) < pair.quote_min:
        return Size(Decimal(0), 0.0, 0.0, 0.0, "mínimo")
    q = float(qty)
    return Size(qty, q * entry, q * entry * (1 + fee), q * per_btc, "perda" if by_loss <= by_cap else "teto")


# ---------- trava de risco ----------

@dataclass(frozen=True)
class Order:
    side: str  # "compra" ou "venda"
    qty: Decimal  # BTC
    price: float  # preço de referência da execução
    stop: float | None = None  # obrigatório na compra
    target: float | None = None


@dataclass(frozen=True)
class State:
    capital: float  # patrimônio simulado: USDT + BTC a preço de mercado
    cash: float  # USDT livre
    btc: float  # BTC na carteira
    day_pnl: float  # resultado do dia até aqui, em USDT (negativo = perda)
    trades_today: int  # compras já feitas hoje


@dataclass(frozen=True)
class Limits:
    max_loss: float = 0.01  # por operação, fração do capital
    max_day_loss: float = 0.03  # no dia, fração do capital, contando o stop da ordem nova
    max_trades_day: int = 10
    min_reward_risk: float = 1.0  # quando a ordem tem alvo
    cap: float = 1.0  # posição depois da ordem, fração do capital
    fee: float = FEE
    pair: Pair = BTC_USDT


@dataclass(frozen=True)
class Verdict:
    ok: bool
    reasons: tuple[str, ...]


def _off_grid(x: Decimal, step: Decimal) -> bool:
    return x % step != 0


def check(order: Order, state: State, limits: Limits = Limits()) -> Verdict:
    """PASSA ou BLOQUEIA, com os motivos. Pura: não mexe em nada e não guarda nada.

    Venda só é barrada pelo que não existe (vender mais BTC do que há na carteira, que seria venda a descoberto) e
    pelas regras do par: uma venda que reduz a posição nunca fica presa por limite de perda ou de operações."""
    p, reasons = limits.pair, []
    qty = order.qty if isinstance(order.qty, Decimal) else Decimal(repr(float(order.qty)))
    if order.side not in ("compra", "venda"):
        return Verdict(False, (f"lado desconhecido: {order.side!r}",))
    if qty <= 0:
        reasons.append("quantidade precisa ser positiva")
    if qty < p.base_min or qty * Decimal(repr(float(order.price))) < p.quote_min:
        reasons.append(f"abaixo do mínimo do par ({p.base_min} BTC e {p.quote_min} USDT)")
    if _off_grid(qty, p.base_increment):
        reasons.append(f"quantidade fora do incremento do par ({p.base_increment} BTC)")
    q = float(qty)
    if order.side == "venda":
        if q > state.btc + 1e-12:
            reasons.append("venda a descoberto: mais BTC do que há na carteira")
        return Verdict(not reasons, tuple(reasons))

    if order.stop is None or not order.stop < order.price:
        reasons.append("compra sem stop abaixo da entrada")
        return Verdict(False, tuple(reasons))
    if _off_grid(Decimal(repr(float(order.stop))), p.price_increment):
        reasons.append(f"stop fora do tick do par ({p.price_increment} USDT)")
    loss = q * loss_per_btc(order.price, order.stop, limits.fee)
    if loss > limits.max_loss * state.capital + 1e-9:
        reasons.append(f"perda no stop de {loss:,.2f} USDT passa da perda máxima por operação "
                       f"({limits.max_loss:.1%} do capital = {limits.max_loss * state.capital:,.2f})")
    if state.day_pnl - loss < -limits.max_day_loss * state.capital - 1e-9:
        reasons.append(f"com o stop desta ordem, o dia passaria da perda máxima de {limits.max_day_loss:.1%}")
    if state.trades_today + 1 > limits.max_trades_day:
        reasons.append(f"limite de {limits.max_trades_day} operações por dia")
    if order.target is not None:
        if not order.target > order.price:
            reasons.append("alvo precisa ficar acima da entrada")
        else:
            reward = q * (order.target * (1 - limits.fee) - order.price * (1 + limits.fee))
            if loss > 0 and reward / loss < limits.min_reward_risk - 1e-12:
                reasons.append(f"alvo/risco de {reward / loss:.2f} abaixo do mínimo de {limits.min_reward_risk:.2f}")
    if (state.btc + q) * order.price > limits.cap * state.capital + 1e-9:
        reasons.append(f"posição depois da ordem passaria do teto de {limits.cap:.0%} do capital")
    if q * order.price * (1 + limits.fee) > state.cash + 1e-9:
        reasons.append("caixa insuficiente (sem alavancagem)")
    return Verdict(not reasons, tuple(reasons))


# ---------- a congelada com o tamanho pela perda máxima ----------

def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = ATR_DAYS) -> np.ndarray:
    """ATR de Wilder no fechamento de cada dia: média (α = 1/n) da maior entre máxima - mínima e as distâncias da
    máxima e da mínima ao fechamento anterior; começa com a média simples dos n primeiros; NaN antes."""
    prev = np.r_[np.nan, close[:-1]]
    tr = np.nanmax(np.column_stack([high - low, np.abs(high - prev), np.abs(low - prev)]), axis=1)
    out = np.full(len(close), np.nan)
    if len(close) < n:
        return out
    a = float(tr[:n].mean())
    out[n - 1] = a
    for t in range(n, len(close)):
        a = (a * (n - 1) + tr[t]) / n
        out[t] = a
    return out


def frozen_stops(d: regime.Daily) -> np.ndarray:
    """(dias × 3) stop equivalente de cada parte no fechamento de cada dia: a média dela menos 1 ATR de 14 dias."""
    a = atr(d.high, d.low, d.close)
    return np.column_stack([regime.sma(d.close, n) - a for n in PARTS])


def part_fractions(d: regime.Daily, max_loss: float | None, cap: float = PART, fee: float = FEE) -> np.ndarray:
    """(dias × 3) fração do capital de cada parte, decidida no fechamento. Fora da média: 0. Ao entrar (o fechamento
    passa para cima da média): 1/3 na congelada de sempre (max_loss None) ou min(cap, max_loss / L), com L a perda por
    real investido se o preço for parar no stop equivalente, com as duas taxas. A fração fica fixa até a parte sair."""
    stops = frozen_stops(d)
    out = np.zeros((len(d.close), len(PARTS)))
    for j, n in enumerate(PARTS):
        on = refine.above(d.close, n)
        frac = 0.0
        for t, c in enumerate(d.close):
            if not on[t]:
                frac = 0.0
            elif frac == 0.0:  # entrou neste fechamento
                if max_loss is None:
                    frac = PART
                else:
                    risk = loss_per_btc(c, stops[t, j], fee) / c if np.isfinite(stops[t, j]) else math.inf
                    frac = min(cap, max_loss / risk) if risk > 0 else cap
            out[t, j] = frac
    return out


def part_trades(d: regime.Daily, frac: np.ndarray, lo: int) -> list[tuple[int, int, int, float]]:
    """(parte, dia da entrada, dia da saída, fração) de cada parte que entrou a partir do fechamento lo-1. A parte
    ainda aberta no fim sai no último fechamento."""
    out = []
    for j in range(frac.shape[1]):
        entry = None
        for t in range(lo - 1, len(d.close)):
            if frac[t, j] > 0 and entry is None:
                entry = t
            elif frac[t, j] == 0 and entry is not None:
                out.append((j, entry, t, float(frac[entry, j])))
                entry = None
        if entry is not None:
            out.append((j, entry, len(d.close) - 1, float(frac[entry, j])))
    return out


def trade_loss(d: regime.Daily, trade: tuple[int, int, int, float], fee: float = FEE) -> float:
    """Resultado da parte em fração do capital da entrada, aproximado (o backtest rebalanceia a fração todo dia)."""
    _, a, b, f = trade
    return f * (d.close[b] / d.close[a] - 1) - 2 * fee * f


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", type=Path, required=True)
    p.add_argument("--start", default=regime.TEST_START)
    args = p.parse_args()
    ts, o, h, l, c = study.load(args.file)
    d = regime.daily_bars(ts, o, h, l, c)
    lo = int(np.searchsorted(d.day, datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc).timestamp() // 86400))
    rets = d.close[lo:] / d.close[lo - 1:-1] - 1
    fmt = lambda k: datetime.fromtimestamp(int(d.day[k]) * 86400, timezone.utc).strftime("%d/%m/%Y")  # noqa: E731
    print(f"congelada com o tamanho pela perda máxima, de {fmt(lo)} a {fmt(len(d.day) - 1)}, {FEE:.1%} por lado")
    print(f"{'variante':<28}{'ao ano':>8}{'queda':>7}{'Sharpe':>8}{'exposição':>11}{'saídas':>8}{'com perda':>11}"
          f"{'passaram de 1%':>16}{'de 2%':>7}{'pior':>8}{'pela perda':>12}")
    variants = {"congelada (1/3 por parte)": None, **{f"perda máx. {m:.0%} por parte": m for m in MAX_LOSSES}}
    fracs = {}
    for name, m in variants.items():
        frac = part_fractions(d, m)
        fracs[name] = frac
        held = regime.held_from(frac.sum(axis=1), lo)
        r = regime.returns_from_held(held, rets)
        trades = part_trades(d, frac, lo)
        losses = np.array([trade_loss(d, t) for t in trades])
        bad = losses[losses < 0]
        entries = [t for t in trades if t[1] >= lo - 1]
        by_loss = np.mean([t[3] < PART - 1e-12 for t in entries]) if entries else math.nan
        print(f"{name:<28}{study.cagr(r):>8.1%}{study.max_dd(r):>7.0%}{r.mean() / r.std() * math.sqrt(365):>8.2f}"
              f"{held.mean():>11.0%}{len(trades):>8}{len(bad):>11}{(bad < -0.01).sum():>16}{(bad < -0.02).sum():>7}"
              f"{bad.min() if len(bad) else 0:>8.1%}{by_loss if m is not None else math.nan:>12.0%}")

    print(f"\nas 6 últimas entradas de parte, com {EXAMPLE_CAPITAL:,.0f} USDT de capital e o par {BTC_USDT.symbol} da KuCoin:")
    print(f"{'dia':<12}{'parte':>7}{'entrada':>11}{'média':>11}{'ATR':>8}{'stop eq.':>11}{'1/3 do capital':>18}"
          f"{'perda 1%':>22}{'perda 2%':>22}")
    stops = frozen_stops(d)
    std = fracs["congelada (1/3 por parte)"]
    last = sorted(((t, j) for j in range(3) for t in range(lo, len(d.close)) if std[t, j] > 0 and std[t - 1, j] == 0))[-6:]
    a = atr(d.high, d.low, d.close)
    for t, j in last:
        e, s = float(d.close[t]), float(stops[t, j])
        third = floor_to(EXAMPLE_CAPITAL * PART / e, BTC_USDT.base_increment)
        cells = []
        for m in MAX_LOSSES:
            z = size(EXAMPLE_CAPITAL, e, s, m, PART)
            cells.append(f"{z.qty} ({z.limited_by})")
        print(f"{fmt(t):<12}{'M' + str(PARTS[j]):>7}{e:>11,.1f}{regime.sma(d.close, PARTS[j])[t]:>11,.1f}{a[t]:>8,.0f}"
              f"{s:>11,.1f}{str(third):>18}" + "".join(f"{x:>22}" for x in cells))


if __name__ == "__main__":
    main()
