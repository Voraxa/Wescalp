"""A comparação com segurar depende do dia de entrada? A congelada contra segurar em cada data de entrada possível.

    python -m wescalp.entries       # candles da Binance de cada moeda em data/candles (ver wescalp.altcoins)

Fixado antes de rodar (25/10/2026). As comparações do README usam uma janela só, e o resultado de segurar depende
muito do dia em que se compra e do dia em que se mede: comprar no topo de 2021 ou no fundo de 2022 dá resultados
opostos. Aqui, cada dia do período é uma data de entrada:
- quem entra começa em USDT no fechamento do dia anterior e paga 0,1% para comprar: a congelada compra os terços que
  a regra manda naquele dia e daí segue a regra; segurar compra tudo. Nenhum vende no fim: vale o fechamento do último
  dia da janela;
- janelas de 1, 2 e 4 anos (365, 730 e 1.461 dias) e, por último, de cada entrada até o fim dos dados, com pelo menos 1
  ano de janela;
- régua de mesma exposição, para tirar a diferença de tempo comprado: no começo, compra a fração do capital que a
  congelada teve comprada, em média, naquela janela, e segura sem rebalancear; o resto fica parado em USDT;
- BTC de 11/10/2018 a 30/06/2026 (o período das contas do README); cesta das 6 moedas de wescalp.altcoins (BTC, ETH,
  SOL, XRP, DOGE e ADA, 1/6 em cada, sem contar a taxa de rebalancear entre elas) de 01/07/2021 a 30/06/2026, só com as
  janelas de 1 e 2 anos e até o fim;
- em cada caso: a fração das entradas em que a congelada terminou acima de segurar e acima da régua de mesma
  exposição, e em que caiu menos que segurar; a mediana e o pior resultado ao ano de cada uma; a fração das entradas
  que terminaram no prejuízo; e as entradas em que segurar foi melhor e pior.
As janelas se sobrepõem: em 7,7 anos de BTC, janelas de 2 anos têm só 3 ou 4 independentes. Isso mostra o quanto a
conclusão depende do dia de entrada; não é prova nova, porque é o mesmo período em que a regra foi escolhida.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import altcoins, brain, intraday, refine, regime

DAY = 86400
FEE = regime.FEE
BTC_HORIZONS = {"1 ano": 365, "2 anos": 730, "4 anos": 1461}
BASKET_HORIZONS = {"1 ano": 365, "2 anos": 730}
MIN_TO_END = 365
BTC_START, BASKET_START, END = intraday.utc(regime.TEST_START), altcoins.TREND_START, altcoins.END


@dataclass(frozen=True)
class Series:
    """Os dias de um período e, em cada um, o retorno de quem já seguia a regra e o de quem entra nele vindo de USDT."""
    day: np.ndarray
    strat: np.ndarray  # congelada, já seguindo a regra (taxa só quando a posição muda)
    strat0: np.ndarray  # congelada, entrando no dia: paga a taxa da posição inteira
    hold: np.ndarray  # segurar, já comprado
    hold0: np.ndarray  # segurar, entrando no dia
    expo: np.ndarray  # fração do capital comprada pela congelada durante o dia


@dataclass(frozen=True)
class Window:
    start: int  # índice do primeiro dia da janela em Series
    days: int
    strat: float  # patrimônio no fim (1 = o que entrou)
    hold: float
    same: float  # régua de mesma exposição
    strat_dd: float
    hold_dd: float
    expo: float


def series(d: regime.Daily, lo: int, hi: int, fee: float = FEE) -> Series:
    """Os dias lo..hi-1 de `d`; a posição de cada dia é a decidida no fechamento do dia anterior."""
    assert np.all(np.diff(d.day[lo - 1:hi]) == 1), "falta dia na série diária"
    rets = d.close[lo:hi] / d.close[lo - 1:hi - 1] - 1
    held = refine.position(d.close, refine.BASE)[lo - 1:hi - 1]
    return Series(d.day[lo:hi], regime.returns_from_held(held, rets, fee), held * rets - fee * held, rets, rets - fee,
                  held)


def basket(parts: list[Series]) -> Series:
    """1/6 em cada moeda todo dia (a cesta de wescalp.altcoins)."""
    for p in parts[1:]:
        assert np.array_equal(p.day, parts[0].day), "as moedas não têm os mesmos dias"
    return Series(parts[0].day, *(np.mean([getattr(p, k) for p in parts], axis=0)
                                  for k in ("strat", "strat0", "hold", "hold0", "expo")))


def drawdown(eq: np.ndarray) -> float:
    """Maior queda do patrimônio a partir do topo anterior, contando o capital do começo (1) como topo."""
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    return float((1 - eq / peak).max())


def window(s: Series, i: int, n: int) -> Window:
    """Quem entra no dia i e mede no fim do dia i+n-1."""
    st = np.cumprod(np.concatenate([[1 + s.strat0[i]], 1 + s.strat[i + 1:i + n]]))
    ho = np.cumprod(np.concatenate([[1 + s.hold0[i]], 1 + s.hold[i + 1:i + n]]))
    f = float(s.expo[i:i + n].mean())
    return Window(i, n, float(st[-1]), float(ho[-1]), f * float(ho[-1]) + 1 - f, drawdown(st), drawdown(ho), f)


def windows(s: Series, n: int) -> list[Window]:
    return [window(s, i, n) for i in range(len(s.day) - n + 1)]


def to_end(s: Series, min_days: int = MIN_TO_END) -> list[Window]:
    return [window(s, i, len(s.day) - i) for i in range(len(s.day) - min_days + 1)]


def yearly(eq: float, days: int) -> float:
    return eq ** (365.25 / days) - 1


def summary(ws: list[Window]) -> dict:
    st, ho, sa = (np.array([yearly(getattr(w, k), w.days) for w in ws]) for k in ("strat", "hold", "same"))
    return {"n": len(ws),
            "beats_hold": float(np.mean(st > ho)), "beats_same": float(np.mean(st > sa)),
            "less_dd": float(np.mean([w.strat_dd < w.hold_dd for w in ws])),
            "median": tuple(float(np.median(x)) for x in (st, ho, sa)),
            "worst": tuple(float(x.min()) for x in (st, ho, sa)),
            "loss": tuple(float(np.mean(x < 0)) for x in (st, ho, sa)),
            "dd": (float(np.median([w.strat_dd for w in ws])), float(np.median([w.hold_dd for w in ws]))),
            "expo": float(np.median([w.expo for w in ws]))}


def extremes(ws: list[Window]) -> tuple[Window, Window]:
    """(a entrada em que segurar foi melhor, a em que foi pior)."""
    return max(ws, key=lambda w: w.hold), min(ws, key=lambda w: w.hold)


def report(name: str, s: Series, horizons: dict[str, int]) -> dict[str, dict]:
    fmt = lambda i: datetime.fromtimestamp(int(s.day[i]) * DAY, timezone.utc).strftime("%d/%m/%Y")  # noqa: E731
    print(f"\n{name}, entradas de {fmt(0)} em diante, dados até {fmt(len(s.day) - 1)}, 0,1% por lado")
    print(f"{'janela':<10}{'entradas':>9}{'acima de segurar':>18}{'acima da mesma exp.':>21}{'caiu menos':>12}"
          f"{'mediana ao ano: congelada / segurar / mesma exp.':>52}{'pior ao ano: cong. / seg.':>28}"
          f"{'no prejuízo: cong. / seg.':>28}{'queda mediana: cong. / seg.':>30}{'exposição':>11}")
    out = {}
    for label, ws in [*((k, windows(s, n)) for k, n in horizons.items()), ("até o fim", to_end(s))]:
        r = summary(ws)
        out[label] = r
        m, w, lo, dd = r["median"], r["worst"], r["loss"], r["dd"]
        print(f"{label:<10}{r['n']:>9}{r['beats_hold']:>18.0%}{r['beats_same']:>21.0%}{r['less_dd']:>12.0%}"
              f"{f'{m[0]:+.1%} / {m[1]:+.1%} / {m[2]:+.1%}':>52}{f'{w[0]:+.1%} / {w[1]:+.1%}':>28}"
              f"{f'{lo[0]:.0%} / {lo[1]:.0%}':>28}{f'{dd[0]:.0%} / {dd[1]:.0%}':>30}{r['expo']:>11.0%}")
        if label != "até o fim":
            for tag, x in zip(("melhor", "pior"), extremes(ws)):
                print(f"{'':<10}entrada em que segurar foi {tag}: {fmt(x.start)}, segurar {yearly(x.hold, x.days):+.1%}"
                      f" ao ano, congelada {yearly(x.strat, x.days):+.1%}, mesma exposição {yearly(x.same, x.days):+.1%}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=brain.CANDLE_DIR)
    args = p.parse_args()
    btc, parts = None, []
    for sym in (altcoins.REFERENCE, *altcoins.COINS):
        ts, o, h, l, c, _ = altcoins.load(sym, args.data)
        d = regime.daily_bars(ts, o, h, l, c)
        hi = int(np.searchsorted(d.day, END // DAY))
        if sym == altcoins.REFERENCE:
            btc = series(d, int(np.searchsorted(d.day, BTC_START // DAY)), hi)
        parts.append(series(d, int(np.searchsorted(d.day, BASKET_START // DAY)), hi))
    report("BTC", btc, BTC_HORIZONS)
    report("Cesta das 6 moedas", basket(parts), BASKET_HORIZONS)


if __name__ == "__main__":
    main()
