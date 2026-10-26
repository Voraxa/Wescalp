"""Comprar depois de queda forte e segurar dias, com os termômetros escolhidos no treino e medidos fora dele.

    python -m wescalp.bounce --file data/candles/BINANCE-BTCUSDT-5m.csv.gz

Fixado antes de rodar (26/10/2026), sem ajuste depois. Nos testes de 1 a 12 h, comprar depois de queda forte rendeu
mais que a média, mas pouco: de 0,15% a 0,35% por operação em 4 a 12 h, perto da ida e volta de 0,2% do spot.
Segurando dias, o repique pode ser maior e a taxa pesa menos. Candles diários (00h UTC) do BTC-USDT da Binance;
decisão, compra e venda no fechamento; 0,1% por lado; o capital inteiro, uma operação por vez.

Termômetros de queda (9):
- RSI de 2 dias (média de Wilder) abaixo de 5, 10 ou 20, como em Connors;
- queda de 1 ou 3 dias medida em desvios, log(C_t / C_t-N) / (σ·√N) ≤ -1,5, -2 ou -2,5, com σ = desvio dos
  log-retornos diários dos 30 dias antes da queda.
Filtro de tendência (2): nenhum, ou fechamento acima da média de 150 dias.
Saída (6): vende depois de 1, 3, 5, 10 ou 20 dias, ou no primeiro fechamento acima da média de 5 dias (até 20 dias).
São 108 combinações. Sinal com operação aberta é ignorado; sinal no dia de uma venda emenda outra operação (a posição
fica, sem taxa).

Walk-forward: para cada ano de 2020 a 2026 (até 30/06), escolhe a combinação de maior Sharpe diário no treino (de
01/02/2018 ao fim do ano anterior, 0,1% por lado), entre as que têm pelo menos 30 operações no treino, e usa no ano.
Cada operação fica com a combinação do dia em que entrou.

Régua, fora da amostra (01/2020 a 06/2026, 0,1% por lado): passa se tiver lucro e p ≤ 0,05 no teste da sorte (os
mesmos trechos comprado e fora em momentos sorteados, 2.000 sorteios; estatística: o patrimônio final). Mostra também
o Sharpe e a queda contra segurar e contra segurar a mesma fração média; operações por ano, acerto e ganho médio por
operação; as taxas de 0,02% e 0%; as 10 melhores combinações no período inteiro (dentro da amostra, só para
comparar); e, sem ajustar nada, as escolhas de cada ano aplicadas a ETH, SOL, XRP, DOGE e ADA (07/2021 a 06/2026).

`--holdout` (só se passar): uma vez, nas 16 semanas guardadas (KuCoin BTC-USDT spot, 04/07 a 21/10/2026), com a
combinação de 2026. São poucas operações, então só se relata; o teste de verdade seria a carteira simulada ao vivo.
"""

import argparse
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import altcoins, brain, intraday, regime, sessions, study

DAY = 86400
FEE = regime.FEE
FEES = sessions.FEES
RSI_LEVELS = (5, 10, 20)
DROP_DAYS = (1, 3)
DROP_Z = (1.5, 2.0, 2.5)
VOL_WIN = 30
TREND = 150
SMA_EXIT = 5
MAX_HOLD = 20
EXITS: tuple[int | str, ...] = (1, 3, 5, 10, 20, "média de 5")
FILTERS = ("nenhum", "acima da média de 150")
TRAIN_START = intraday.utc("2018-02-01")
OOS_YEARS = tuple(range(2020, 2027))
END = intraday.utc("2026-07-01")
ALT_START = altcoins.TREND_START
MIN_TRADES = 30
LUCK = 2000
MAX_P = 0.05


@dataclass(frozen=True)
class Combo:
    signal: str
    filter: str
    exit: int | str  # dias segurando, ou "média de 5"

    def __str__(self) -> str:
        out = self.exit if isinstance(self.exit, str) else f"{self.exit} d"
        return f"{self.signal} | filtro: {self.filter} | saída: {out}"


@dataclass(frozen=True)
class Asset:
    """O que a regra precisa de uma moeda, por dia: fechamento, média de 5, sinais e filtros."""
    day: np.ndarray
    close: np.ndarray
    sma5: np.ndarray
    signals: dict[str, np.ndarray]
    filters: dict[str, np.ndarray]

    def index(self, t: int) -> int:
        """O primeiro dia em t (segundos UTC) ou depois."""
        return int(np.searchsorted(self.day, t // DAY))

    def rets(self) -> np.ndarray:
        """rets[d]: retorno do fechamento d-1 ao d (rets[0] = 0)."""
        return np.concatenate([[0.0], self.close[1:] / self.close[:-1] - 1])


def rsi(close: np.ndarray, n: int = 2) -> np.ndarray:
    """RSI com a média de Wilder; NaN até ter n variações."""
    out = np.full(len(close), np.nan)
    d = np.diff(close)
    if len(d) < n:
        return out
    gain, loss = np.maximum(d, 0.0), np.maximum(-d, 0.0)
    g, lo = gain[:n].mean(), loss[:n].mean()
    for t in range(n, len(close)):
        if t > n:
            g = (g * (n - 1) + gain[t - 1]) / n
            lo = (lo * (n - 1) + loss[t - 1]) / n
        out[t] = 100.0 if lo == 0 else 100 - 100 / (1 + g / lo)
    return out


def drop_z(close: np.ndarray, days: int, win: int = VOL_WIN) -> np.ndarray:
    """log(C_t / C_t-N) / (σ·√N), com σ dos `win` log-retornos diários que terminam no dia t-N (antes da queda)."""
    lr = np.diff(np.log(close))  # lr[k]: do dia k ao k+1
    out = np.full(len(close), np.nan)
    for t in range(days + win, len(close)):
        sigma = lr[t - days - win:t - days].std(ddof=1)
        if sigma > 0:
            out[t] = math.log(close[t] / close[t - days]) / (sigma * math.sqrt(days))
    return out


def signals(close: np.ndarray) -> dict[str, np.ndarray]:
    out = {}
    r = rsi(close)
    for lv in RSI_LEVELS:
        out[f"RSI 2 d < {lv}"] = np.nan_to_num(r, nan=100.0) < lv
    for n in DROP_DAYS:
        z = np.nan_to_num(drop_z(close, n), nan=0.0)
        for k in DROP_Z:
            out[f"queda de {n} d ≤ -{k:.1f}σ".replace(".", ",")] = z <= -k
    return out


def filters(close: np.ndarray) -> dict[str, np.ndarray]:
    m = regime.sma(close, TREND)
    return {FILTERS[0]: np.ones(len(close), dtype=bool), FILTERS[1]: np.nan_to_num(m, nan=np.inf) < close}


def asset(d: regime.Daily) -> Asset:
    assert np.all(np.diff(d.day) == 1), "falta dia na série diária"
    return Asset(d.day, d.close, regime.sma(d.close, SMA_EXIT), signals(d.close), filters(d.close))


def combos() -> list[Combo]:
    names = [*(f"RSI 2 d < {lv}" for lv in RSI_LEVELS),
             *(f"queda de {n} d ≤ -{k:.1f}σ".replace(".", ",") for n in DROP_DAYS for k in DROP_Z)]
    return [Combo(s, f, e) for s in names for f in FILTERS for e in EXITS]


def exit_day(a: Asset, t: int, rule: int | str, last: int) -> int:
    """Dia da venda de quem comprou no fechamento do dia t, sem passar do dia `last`."""
    if rule == "média de 5":
        for j in range(t + 1, min(t + MAX_HOLD, last) + 1):
            if a.close[j] > a.sma5[j]:
                return j
        return min(t + MAX_HOLD, last)
    return min(t + int(rule), last)


def trades(a: Asset, c: Combo, lo: int, hi: int, last: int, busy: int = -1) -> list[tuple[int, int]]:
    """(dia da compra, dia da venda) das compras nos dias lo..hi-1, uma de cada vez, com venda até o dia `last`.

    `busy` é o dia da venda de uma operação anterior: antes dele não compra; nele, emenda."""
    sig = a.signals[c.signal] & a.filters[c.filter]
    out = []
    for t in range(lo, min(hi, last)):
        if t < busy or not sig[t]:
            continue
        busy = exit_day(a, t, c.exit, last)
        out.append((t, busy))
    return out


def held(trs: list[tuple[int, int]], n: int) -> np.ndarray:
    """Posição em cada dia d (o retorno do fechamento d-1 ao d): 1 se uma operação comprou antes de d e vende em d ou
    depois."""
    h = np.zeros(n)
    for a, b in trs:
        h[a + 1:b + 1] = 1.0
    return h


def daily(a: Asset, trs: list[tuple[int, int]], lo: int, hi: int, fee: float = FEE) -> tuple[np.ndarray, np.ndarray]:
    """(retornos diários com taxa, posição) nos dias lo..hi-1; começa sem posição."""
    h = held(trs, len(a.close))[lo:hi]
    return regime.returns_from_held(h, a.rets()[lo:hi], fee), h


def sharpe(r: np.ndarray) -> float:
    return float(r.mean() / r.std() * math.sqrt(365)) if r.std() > 0 else 0.0


def choose(a: Asset, lo: int, hi: int, min_trades: int = MIN_TRADES) -> tuple[Combo, float, int]:
    """A combinação de maior Sharpe nos dias lo..hi-1 (operações vendidas até hi-1), entre as com min_trades ou mais."""
    best: tuple[Combo, float, int] | None = None
    for c in combos():
        trs = trades(a, c, lo, hi, hi - 1)
        if len(trs) < min_trades:
            continue
        s = sharpe(daily(a, trs, lo, hi)[0])
        if best is None or s > best[1]:
            best = (c, s, len(trs))
    assert best is not None, "nenhuma combinação com operações suficientes no treino"
    return best


def year_start(y: int) -> int:
    return intraday.utc(f"{y}-01-01")


def walk_forward(a: Asset) -> list[tuple[int, Combo, float, int]]:
    """(ano, combinação escolhida, Sharpe no treino, operações no treino) de cada ano fora da amostra."""
    lo = a.index(TRAIN_START)
    return [(y, *choose(a, lo, a.index(year_start(y)))) for y in OOS_YEARS]


def apply(a: Asset, picks: list[tuple[int, Combo, float, int]], start: int, end: int) -> list[tuple[int, int, Combo]]:
    """As operações fora da amostra de `start` a `end` (segundos UTC), cada ano com a combinação dele."""
    last = a.index(end) - 1
    out, busy = [], -1
    for y, c, _, _ in picks:
        lo, hi = a.index(max(year_start(y), start)), a.index(min(year_start(y + 1), end))
        if lo >= hi:
            continue
        for t, e in trades(a, c, lo, hi, last, busy):
            out.append((t, e, c))
            busy = e
    return out


def report(a: Asset, trs: list[tuple[int, int, Combo]], start: int, end: int, luck: int) -> dict:
    lo, hi = a.index(start), a.index(end)
    pairs = [(t, e) for t, e, _ in trs]
    years = (hi - lo) / 365.25
    per = {}
    for f in FEES:
        r, h = daily(a, pairs, lo, hi, f)
        per[f] = {"total": float(np.prod(1 + r) - 1), "cagr": study.cagr(r), "dd": study.max_dd(r), "sharpe": sharpe(r)}
    r, h = daily(a, pairs, lo, hi)
    net = np.array([a.close[e] / a.close[t] * (1 - FEE) ** 2 - 1 for t, e in pairs])
    rets = a.rets()[lo:hi]
    hold_eq = np.cumprod(1 + rets)
    f = float(h.mean())
    same = f * hold_eq + 1 - f
    same_r = same / np.concatenate([[1.0], same[:-1]]) - 1
    p = regime.luck_test(h, rets, luck, fee=FEE)["p_total"] if luck and len(pairs) else math.nan
    return {"fees": per, "trades": len(pairs), "per_year": len(pairs) / years,
            "hit": float((net > 0).mean()) if len(net) else math.nan,
            "mean": float(net.mean()) if len(net) else math.nan,
            "days": float(np.mean([e - t for t, e in pairs])) if pairs else math.nan, "expo": f,
            "hold": {"cagr": study.cagr(rets), "dd": study.max_dd(rets), "sharpe": sharpe(rets)},
            "same": {"cagr": study.cagr(same_r), "dd": study.max_dd(same_r), "sharpe": sharpe(same_r)}, "p": p}


def passes(s: dict) -> bool:
    return s["fees"][FEE]["total"] > 0 and s["p"] <= MAX_P


def fmt_day(t: int) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%d/%m/%Y")


def show(name: str, s: dict) -> None:
    fe = s["fees"]
    print(f"{name}: {s['trades']} operações ({s['per_year']:.1f} por ano), acerto {s['hit']:.0%}, média por operação"
          f" {s['mean']:+.2%} com taxa, {s['days']:.1f} dias comprado em média, exposição {s['expo']:.0%}")
    print("  ao ano com taxa de " + ", ".join(f"{f:.2%}: {fe[f]['cagr']:+.1%}" for f in FEES)
          + f"; queda máx. {fe[FEE]['dd']:.0%}, Sharpe {fe[FEE]['sharpe']:.2f}; p da sorte {s['p']:.3f}")
    print(f"  segurar: {s['hold']['cagr']:+.1%} ao ano, queda {s['hold']['dd']:.0%}, Sharpe {s['hold']['sharpe']:.2f};"
          f" mesma exposição: {s['same']['cagr']:+.1%}, queda {s['same']['dd']:.0%}, Sharpe {s['same']['sharpe']:.2f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", type=Path, required=True)
    p.add_argument("--data", type=Path, default=brain.CANDLE_DIR, help="pasta dos candles das altcoins")
    p.add_argument("--luck", type=int, default=LUCK)
    p.add_argument("--holdout", type=Path, help="candles da KuCoin para as 16 semanas guardadas (só se passar)")
    args = p.parse_args()
    btc = asset(regime.daily_bars(*study.load(args.file)))
    picks = walk_forward(btc)
    print("Escolha de cada ano (treino de 01/02/2018 ao fim do ano anterior):")
    for y, c, s, n in picks:
        print(f"  {y}: {c} (Sharpe no treino {s:.2f}, {n} operações)")
    start = year_start(OOS_YEARS[0])
    if args.holdout:
        hk = asset(regime.daily_bars(*study.load(args.holdout)))
        a, b = sessions.HOLDOUT
        s = report(hk, apply(hk, picks[-1:], a, b), a, b, args.luck)
        show(f"\n16 semanas guardadas ({fmt_day(a)} a {fmt_day(b - DAY)}, KuCoin), combinação de 2026", s)
        return
    trs = apply(btc, picks, start, END)
    s = report(btc, trs, start, END, args.luck)
    print()
    show(f"BTC fora da amostra ({fmt_day(start)} a {fmt_day(END - DAY)})", s)
    print(f"  resultado: {'PASSA' if passes(s) else 'não passa'} (lucro com 0,1% por lado e p ≤ {MAX_P})")

    print("\nAs 10 melhores combinações no período inteiro (dentro da amostra, só para comparar):")
    lo, hi = btc.index(TRAIN_START), btc.index(END)
    rows = []
    for c in combos():
        t = trades(btc, c, lo, hi, hi - 1)
        r = daily(btc, t, lo, hi)[0]
        rows.append((sharpe(r), study.cagr(r), study.max_dd(r), len(t) / ((hi - lo) / 365.25), c))
    for sh, cg, dd, n, c in sorted(rows, key=lambda x: -x[0])[:10]:
        print(f"  Sharpe {sh:.2f}, {cg:+.1%} ao ano, queda {dd:.0%}, {n:.1f} operações por ano: {c}")

    print(f"\nAs mesmas escolhas, sem ajuste, nas altcoins ({fmt_day(ALT_START)} a {fmt_day(END - DAY)}):")
    for sym in altcoins.COINS:
        ts, o, h, l, c, _ = altcoins.load(sym, args.data)
        alt = asset(regime.daily_bars(ts, o, h, l, c))
        show(sym[:-4], report(alt, apply(alt, picks, ALT_START, END), ALT_START, END, args.luck))


if __name__ == "__main__":
    main()
