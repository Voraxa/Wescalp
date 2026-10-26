"""A congelada com mais operações: as mesmas três médias em candles mais curtos.

    python -m wescalp.faster --file data/candles/BINANCE-BTCUSDT-5m.csv.gz

Fixado antes de rodar (26/10/2026), sem ajuste depois. A congelada decide uma vez por dia e faz umas 20 operações por
ano. Aqui, a mesma regra (1/3 do capital em cada média, comprado enquanto o fechamento do candle está acima dela, 0,1%
por lado) em candles mais curtos, de 11/10/2018 a 30/06/2026, em duas famílias:
- mais rápida: médias de 50, 100 e 150 candles de 12 h, de 4 h e de 1 h (cobrem de 25 a 75 dias, de 8 a 25 dias e
  de 2 a 6 dias);
- a mesma tendência conferida mais vezes: médias de 50, 100 e 150 dias, com decisão a cada 4 h (300, 600 e 900
  candles) e a cada 1 h (1.200, 2.400 e 3.600 candles).
Régua: a congelada diária no mesmo período. Teste da sorte como em wescalp.refine: cada média com o momento sorteado
(os mesmos trechos comprado e fora, em ordem sorteada), 1.000 sorteios, estatística o Sharpe. Passa a variante com
Sharpe maior que o da congelada diária e p ≤ 0,01 (0,05 dividido pelas 5 variantes). Mostra também a taxa de 0,02%
(ordem limitada nos futuros).
"""

import argparse
import math
from pathlib import Path

import numpy as np

from . import intraday, regime, study

H = 3600
START, END = intraday.utc(regime.TEST_START), intraday.utc("2026-07-01")
FEES = (0.001, 0.0002)
LUCK = 1000
BASE = "congelada diária"
VARIANTS = {  # nome: (horas por candle, médias em candles)
    BASE: (24, (50, 100, 150)),
    "médias de 50/100/150 candles de 12 h": (12, (50, 100, 150)),
    "médias de 50/100/150 candles de 4 h": (4, (50, 100, 150)),
    "médias de 50/100/150 candles de 1 h": (1, (50, 100, 150)),
    "médias de 50/100/150 dias, decisão a cada 4 h": (4, (300, 600, 900)),
    "médias de 50/100/150 dias, decisão a cada 1 h": (1, (1200, 2400, 3600)),
}
MAX_P = 0.05 / (len(VARIANTS) - 1)


def bars(ts: np.ndarray, c: np.ndarray, hours: int) -> tuple[np.ndarray, np.ndarray]:
    """(abertura, fechamento) de cada candle de `hours` horas UTC montado dos candles de 5 min em ordem de tempo."""
    k = ts // (hours * H)
    keys, first = np.unique(k, return_index=True)
    last = np.concatenate([first[1:], [len(ts)]]) - 1
    return keys * hours * H, c[last]


def components(close: np.ndarray, lengths: tuple[int, ...]) -> list[np.ndarray]:
    """Cada média: 1 no fechamento do candle se o fechamento está acima dela (0 enquanto ela não existe)."""
    out = []
    for n in lengths:
        m = regime.sma(close, n)
        out.append(np.where(np.isnan(m), 0.0, (close > m).astype(float)))
    return out


def sharpe(r: np.ndarray, per_year: float) -> float:
    return float(r.mean() / r.std() * math.sqrt(per_year)) if r.std() > 0 else 0.0


def evaluate(t: np.ndarray, close: np.ndarray, hours: int, lengths: tuple[int, ...], luck: int = LUCK,
             start: int = START, end: int = END, seed: int = 0) -> dict:
    """A regra nos candles que abrem em [start, end): a posição decidida no fechamento de um candle vale no seguinte."""
    lo, hi = int(np.searchsorted(t, start)), int(np.searchsorted(t, end))
    per_year = 365.25 * 24 / hours
    years = (hi - lo) / per_year
    rets = close[lo:hi] / close[lo - 1:hi - 1] - 1
    held = [regime.held_from(c[:hi], lo) for c in components(close, lengths)]
    pos = np.mean(held, axis=0)
    out = {"per_year": sum(int((np.diff(np.concatenate([[0.0], h])) > 0).sum()) for h in held) / years,
           "expo": float(pos.mean())}
    for f in FEES:
        r = regime.returns_from_held(pos, rets, f)
        out[f] = {"cagr": float(np.prod(1 + r) ** (1 / years) - 1), "dd": study.max_dd(r), "sharpe": sharpe(r, per_year)}
    target = out[FEES[0]]["sharpe"]
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(luck):
        r = regime.returns_from_held(np.mean([regime.shuffled_timing(h, rng) for h in held], axis=0), rets, FEES[0])
        hits += sharpe(r, per_year) >= target
    out["p"] = hits / luck if luck else math.nan
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", type=Path, required=True)
    p.add_argument("--luck", type=int, default=LUCK)
    args = p.parse_args()
    ts, _, _, _, c = study.load(args.file)
    hold = None
    print(f"{'variante':<48}{'operações/ano':>14}{'exposição':>11}{'ao ano 0,1%':>13}{'0,02%':>9}{'queda':>8}"
          f"{'Sharpe':>8}{'p sorte':>9}  resultado")
    res = {}
    for name, (hours, lengths) in VARIANTS.items():
        t, close = bars(ts, c, hours)
        s = evaluate(t, close, hours, lengths, args.luck)
        res[name] = s
        if hold is None:
            lo, hi = int(np.searchsorted(t, START)), int(np.searchsorted(t, END))
            r = close[lo:hi] / close[lo - 1:hi - 1] - 1
            hold = {"cagr": float(np.prod(1 + r) ** (365.25 / len(r)) - 1), "dd": study.max_dd(r), "sharpe": sharpe(r, 365.25)}
        f = s[FEES[0]]
        verdict = "régua" if name == BASE else (
            "PASSA" if f["sharpe"] > res[BASE][FEES[0]]["sharpe"] and s["p"] <= MAX_P else "não passa")
        print(f"{name:<48}{s['per_year']:>14.1f}{s['expo']:>11.0%}{f['cagr']:>13.1%}{s[FEES[1]]['cagr']:>9.1%}"
              f"{f['dd']:>8.0%}{f['sharpe']:>8.2f}{s['p']:>9.3f}  {verdict}", flush=True)
    print(f"{'segurar BTC':<48}{'':>25}{hold['cagr']:>13.1%}{'':>9}{hold['dd']:>8.0%}{hold['sharpe']:>8.2f}")


if __name__ == "__main__":
    main()
