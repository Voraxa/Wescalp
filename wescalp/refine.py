"""Refinamentos da combinação 50/100/150, com a regra de decisão fixada antes de rodar.

    python -m wescalp.refine --file data/candles/BINANCE-BTCUSDT-5m.csv.gz

Base (campeã de wescalp.study): 1/3 do capital em cada média diária de 50, 100 e 150 dias (comprado se o
fechamento está acima dela). Refinamentos, cada um com os parâmetros fixos:
- volatilidade: a base vezes min(1, alvo / vol), com vol = desvio dos retornos diários dos últimos 30 dias e alvo =
  mediana dessa vol nos 365 dias anteriores (tira exposição nos períodos de pânico; sem alavancagem);
- ampla: média das 13 regras de 50, 60, ..., 170 dias (o platô do estudo);
- mista: as 3 médias da base mais o momento de 30 dias e o cruzamento 20/50 (tipos de sinal diferentes).

Um refinamento substitui a base só se cumprir tudo: Sharpe maior, queda máxima igual ou menor, retorno ao ano na
pior hora de virada do dia (00, 06, 12, 18 h UTC) e na segunda metade do período não menores que os da base, e
p ≤ 0,05 no teste da sorte (cada componente com o momento sorteado por conta própria, 2.000 sorteios).
"""

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import brain, regime, study

VARIANTS = ("base 50/100/150", "volatilidade", "ampla 50..170", "mista")
BASE = VARIANTS[0]
VOL_WIN, TARGET_WIN = 30, 365


def above(close: np.ndarray, n: int) -> np.ndarray:
    m = regime.sma(close, n)
    return np.where(np.isnan(m), 0.0, (close > m).astype(float))


def vol_scale(close: np.ndarray, win: int = VOL_WIN, target_win: int = TARGET_WIN) -> np.ndarray:
    """min(1, alvo / vol) no fechamento de cada dia, só com o passado; 1 enquanto não há histórico."""
    r = np.concatenate([[np.nan], close[1:] / close[:-1] - 1])
    vol = np.full(len(close), np.nan)
    for t in range(win, len(close)):
        vol[t] = np.std(r[t - win + 1:t + 1], ddof=1)
    out = np.ones(len(close))
    for t in range(win + target_win, len(close)):
        target = np.nanmedian(vol[t - target_win:t])  # os 365 dias antes de hoje
        if vol[t] > 0:
            out[t] = min(1.0, target / vol[t])
    return out


def components(close: np.ndarray, variant: str) -> tuple[list[np.ndarray], np.ndarray]:
    """(componentes 0/1 no fechamento de cada dia, escala por dia). A posição é a média dos componentes × escala."""
    base = [above(close, n) for n in (50, 100, 150)]
    one = np.ones(len(close))
    if variant == BASE:
        return base, one
    if variant == "volatilidade":
        return base, vol_scale(close)
    if variant == "ampla 50..170":
        return [above(close, n) for n in range(50, 171, 10)], one
    if variant == "mista":
        r = regime.rules(close)
        return base + [r["mom 30"], r["ema 20/50"]], one
    raise ValueError(variant)


def position(close: np.ndarray, variant: str) -> np.ndarray:
    comps, scale = components(close, variant)
    return np.mean(comps, axis=0) * scale


def luck_p(close: np.ndarray, lo: int, variant: str, n: int = 2000, seed: int = 0) -> float:
    """Fração dos sorteios (cada componente com o momento embaralhado) com Sharpe ≥ o da variante."""
    comps, scale = components(close, variant)
    rets = close[lo:] / close[lo - 1:-1] - 1
    held = [regime.held_from(c, lo) for c in comps]
    s = regime.held_from(scale, lo)
    real = regime.returns_from_held(np.mean(held, axis=0) * s, rets)
    rng = np.random.default_rng(seed)
    target = real.mean() / real.std()
    hits = 0
    for _ in range(n):
        x = regime.returns_from_held(np.mean([regime.shuffled_timing(h, rng) for h in held], axis=0) * s, rets)
        hits += x.mean() / x.std() >= target
    return hits / n


def decide(scores: dict[str, dict]) -> tuple[str, dict[str, list[str]]]:
    """Aplica a regra fixada: devolve a vencedora e, para cada refinamento, os critérios que ele não cumpriu."""
    b = scores[BASE]
    fails = {}
    for v, s in scores.items():
        if v == BASE:
            continue
        f = []
        if not s["sharpe"] > b["sharpe"]:
            f.append("Sharpe")
        if s["dd"] > b["dd"] + 1e-9:
            f.append("queda")
        if s["worst_hour"] < b["worst_hour"] - 1e-9:
            f.append("pior hora")
        if s["second_half"] < b["second_half"] - 1e-9:
            f.append("2ª metade")
        if s["p"] > 0.05:
            f.append("sorte")
        fails[v] = f
    ok = [v for v, f in fails.items() if not f]
    return (max(ok, key=lambda v: scores[v]["sharpe"]) if ok else BASE), fails


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", type=Path, required=True)
    p.add_argument("--start", default=regime.TEST_START)
    args = p.parse_args()
    ts, o, h, l, c = study.load(args.file)
    start_day = int(datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc).timestamp() // 86400)
    split_day = int(datetime.fromisoformat(study.SPLIT).replace(tzinfo=timezone.utc).timestamp() // 86400)
    scores, rows = {}, {}
    for v in VARIANTS:
        by_hour = {}
        for off in study.OFFSETS_H:
            d = regime.daily_bars(ts, o, h, l, c, off * 3600)
            lo = int(np.searchsorted(d.day, start_day))
            rets = d.close[lo:] / d.close[lo - 1:-1] - 1
            held = regime.held_from(position(d.close, v), lo)
            by_hour[off] = (d, lo, rets, held)
        d, lo, rets, held = by_hour[0]
        r = regime.returns_from_held(held, rets)
        when = (d.day[lo:] * 86400).astype("datetime64[s]")
        months, years = when.astype("datetime64[M]").astype(int), when.astype("datetime64[Y]").astype(int) + 1970
        m = study.metrics(r, months, years)
        second = d.day[lo:] >= split_day
        scores[v] = {
            "sharpe": m["sharpe"], "dd": m["dd"], "cagr": m["cagr"], "pf": m["pf"], "worst_month": m["worst_month"],
            "worst_hour": min(study.cagr(regime.returns_from_held(x[3], x[2])) for x in by_hour.values()),
            "second_half": study.cagr(r[second]),
            "fee5": study.cagr(regime.returns_from_held(held, rets, 0.005)),
            "in_market": float(held.mean()), "p": luck_p(d.close, lo, v),
        }
        rows[v] = r
    print(f"{'variante':<18}{'ao ano':>8}{'queda':>7}{'Sharpe':>8}{'fator':>7}{'pior mês':>10}{'pior hora':>11}"
          f"{'2ª metade':>11}{'0,5%/lado':>11}{'exposição':>11}{'p sorte':>9}")
    for v, s in scores.items():
        print(f"{v:<18}{s['cagr']:>8.1%}{s['dd']:>7.0%}{s['sharpe']:>8.2f}{s['pf']:>7.2f}{s['worst_month']:>10.0%}"
              f"{s['worst_hour']:>11.0%}{s['second_half']:>11.0%}{s['fee5']:>11.0%}{s['in_market']:>11.0%}{s['p']:>9.3f}")
    win, fails = decide(scores)
    for v, f in fails.items():
        print(f"{v}: {'cumpre todos os critérios' if not f else 'não cumpre: ' + ', '.join(f)}")
    print(f"\nvencedora pela regra fixada: {win}")


if __name__ == "__main__":
    main()
