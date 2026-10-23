"""Aprendiz: a carteira aprende com os próprios sinais, dando mais peso às médias cujos sinais vêm rendendo.

    python -m wescalp.learn --file data/candles/BINANCE-BTCUSDT-5m.csv.gz

Fixado antes de rodar (23/10/2026), sem ajuste depois:
- componentes: as 13 médias diárias de 50, 60, ..., 170 dias (o platô de wescalp.study); cada uma fica comprada se
  o fechamento das 00h UTC está acima dela;
- nota de cada componente: soma descontada do log do retorno diário dos sinais dele, com 0,1% de taxa em cada
  troca; meia-vida de 180 dias (o que aconteceu há 180 dias vale metade);
- peso: exp(η · nota), normalizado (Hedge, os "especialistas ponderados"), com η = ln 2 / 0,10: 10% a mais de
  retorno descontado dobra o peso relativo; começa com pesos iguais;
- posição: soma dos pesos das médias compradas, decidida no fechamento e valendo para o dia seguinte; a carteira só
  ajusta quando a posição desejada se afasta 5 pontos percentuais ou mais da atual (ou vai a 0% ou 100%), para não
  gerar ordens pequenas todo dia; 0,1% por lado sobre o que muda.
Controles: a combinação congelada (50/100/150, pesos iguais) e as mesmas 13 médias com pesos iguais e a mesma faixa
de ajuste (o aprendiz sem aprender: isola o efeito do aprendizado). A promoção usa a regra de wescalp.refine.
"""

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import regime, refine, study

POOL = tuple(range(50, 171, 10))
HALF_LIFE = 180
GAMMA = 0.5 ** (1 / HALF_LIFE)
ETA = math.log(2) / 0.10
BAND = 0.05
FEE = regime.FEE
LEARNER, EQUAL = "aprendiz", "13 médias, pesos iguais"


def comp_positions(close: np.ndarray, pool=POOL) -> np.ndarray:
    """Matriz (componentes × dias) com 1 onde o fechamento está acima da média."""
    return np.array([refine.above(close, n) for n in pool])


def learn_positions(comps: np.ndarray, close: np.ndarray, eta: float = ETA, gamma: float = GAMMA,
                    band: float = BAND, fee: float = FEE) -> tuple[np.ndarray, np.ndarray]:
    """(posição decidida no fechamento de cada dia, pesos usados nela). Só usa o que já aconteceu até o dia."""
    k, t_len = comps.shape
    r = np.concatenate([[0.0], close[1:] / close[:-1] - 1])
    score = np.zeros(k)
    pos = np.zeros(t_len)
    weights = np.zeros((t_len, k))
    current = 0.0
    zero = np.zeros(k)
    for t in range(t_len):
        held = comps[:, t - 1] if t > 0 else zero  # posição de cada componente durante o dia t
        before = comps[:, t - 2] if t > 1 else zero
        score = gamma * score + np.log1p(held * r[t] - fee * np.abs(held - before))
        w = np.exp(eta * (score - score.max()))
        w /= w.sum()
        target = float(w @ comps[:, t])
        if abs(target - current) >= band - 1e-12 or (target in (0.0, 1.0) and target != current):
            current = target
        pos[t] = current
        weights[t] = w
    return pos, weights


def variant_position(close: np.ndarray, name: str) -> np.ndarray:
    if name == refine.BASE:
        return refine.position(close, refine.BASE)
    comps = comp_positions(close)
    return learn_positions(comps, close, eta=ETA if name == LEARNER else 0.0)[0]


def luck_p(close: np.ndarray, lo: int, name: str, n: int = 1000, seed: int = 0) -> float:
    """Sharpe da variante contra a mesma variante sobre componentes com o momento sorteado (cada um à parte)."""
    rets = close[lo:] / close[lo - 1:-1] - 1
    comps = comp_positions(close) if name != refine.BASE else np.array([refine.above(close, n) for n in (50, 100, 150)])
    eta = ETA if name == LEARNER else 0.0

    def sharpe_of(cs):
        pos = learn_positions(cs, close, eta=eta)[0] if name != refine.BASE else cs.mean(axis=0)
        x = regime.returns_from_held(regime.held_from(pos, lo), rets)
        return x.mean() / x.std()

    real = sharpe_of(comps)
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n):
        shuffled = comps.copy()
        for i in range(len(comps)):
            # só a parte do período de teste é sorteada; o aquecimento fica como está
            shuffled[i, lo - 1:-1] = regime.shuffled_timing(comps[i, lo - 1:-1], rng)
        hits += sharpe_of(shuffled) >= real
    return hits / n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", type=Path, required=True)
    p.add_argument("--start", default=regime.TEST_START)
    p.add_argument("--luck", type=int, default=1000)
    args = p.parse_args()
    ts, o, h, l, c = study.load(args.file)
    start_day = int(datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc).timestamp() // 86400)
    split_day = int(datetime.fromisoformat(study.SPLIT).replace(tzinfo=timezone.utc).timestamp() // 86400)
    names = (refine.BASE, EQUAL, LEARNER)
    scores, series = {}, {}
    for name in names:
        by_hour = {}
        for off in study.OFFSETS_H:
            d = regime.daily_bars(ts, o, h, l, c, off * 3600)
            lo = int(np.searchsorted(d.day, start_day))
            rets = d.close[lo:] / d.close[lo - 1:-1] - 1
            by_hour[off] = (d, lo, rets, regime.held_from(variant_position(d.close, name), lo))
        d, lo, rets, held = by_hour[0]
        r = regime.returns_from_held(held, rets)
        when = (d.day[lo:] * 86400).astype("datetime64[s]")
        months, years = when.astype("datetime64[M]").astype(int), when.astype("datetime64[Y]").astype(int) + 1970
        m = study.metrics(r, months, years)
        changes = int((np.abs(np.diff(np.concatenate([[0.0], held]))) > 1e-12).sum())
        scores[name] = {
            "sharpe": m["sharpe"], "dd": m["dd"], "cagr": m["cagr"], "pf": m["pf"], "worst_month": m["worst_month"],
            "worst_hour": min(study.cagr(regime.returns_from_held(x[3], x[2])) for x in by_hour.values()),
            "second_half": study.cagr(r[d.day[lo:] >= split_day]),
            "fee5": study.cagr(regime.returns_from_held(held, rets, 0.005)),
            "orders_year": changes / (len(r) / 365.25), "p": luck_p(d.close, lo, name, args.luck),
        }
        series[name] = (d, lo, years)
    print(f"{'variante':<24}{'ao ano':>8}{'queda':>7}{'Sharpe':>8}{'fator':>7}{'pior mês':>10}{'pior hora':>11}"
          f"{'2ª metade':>11}{'0,5%/lado':>11}{'ordens/ano':>12}{'p sorte':>9}")
    for name, s in scores.items():
        print(f"{name:<24}{s['cagr']:>8.1%}{s['dd']:>7.0%}{s['sharpe']:>8.2f}{s['pf']:>7.2f}{s['worst_month']:>10.0%}"
              f"{s['worst_hour']:>11.0%}{s['second_half']:>11.0%}{s['fee5']:>11.0%}{s['orders_year']:>12.1f}"
              f"{s['p']:>9.3f}")
    learner_vs_base = {refine.BASE: scores[refine.BASE], LEARNER: scores[LEARNER]}
    win, fails = refine.decide(learner_vs_base)
    print(f"\naprendiz contra a congelada: {'cumpre todos os critérios' if not fails[LEARNER] else 'não cumpre: ' + ', '.join(fails[LEARNER])}")
    print(f"aprendiz contra as 13 médias sem aprender: Sharpe {scores[LEARNER]['sharpe']:.2f} contra"
          f" {scores[EQUAL]['sharpe']:.2f}, ao ano {scores[LEARNER]['cagr']:.1%} contra {scores[EQUAL]['cagr']:.1%}")
    print(f"vencedora pela regra fixada: {win}")

    d, lo, years = series[LEARNER]
    _, w = learn_positions(comp_positions(d.close), d.close)
    print("\no que o aprendiz aprendeu: peso médio de cada média por ano (%)")
    print(f"{'ano':<6}" + "".join(f"{n:>6}" for n in POOL))
    for y in np.unique(years):
        rows = w[lo:][years == y]
        print(f"{y:<6}" + "".join(f"{x * 100:>6.0f}" for x in rows.mean(axis=0)))
    print(f"{'agora':<6}" + "".join(f"{x * 100:>6.0f}" for x in w[-1]))


if __name__ == "__main__":
    main()
