"""Modelo de risco por cima da congelada: quanto ficar comprado, não se o BTC sobe ou desce.

    python -m wescalp.factors && python -m wescalp.events      # antes: VIX e calendário de notícias
    python -m wescalp.risk --file data/candles/BINANCE-BTCUSDT-5m.csv.gz

Fixado antes de rodar (23/10/2026), sem ajuste depois. Base: a combinação congelada 50/100/150 (wescalp.refine).
Cada candidato mexe só no tamanho da posição, entre 0% e 100% do capital (sem alavancagem), e nunca compra com a
congelada fora:
1. volatilidade prevista: σ = EWMA dos quadrados dos log-retornos diários (λ = 0,94, o padrão RiskMetrics);
   posição = min(1, congelada × referência / σ), com referência = mediana de σ nos 365 dias anteriores. Mercado mais
   calmo que o normal aumenta a posição (até 100%); mais agitado, reduz. O refinamento de volatilidade de
   wescalp.refine só reduzia e usava o desvio simples de 30 dias;
2. cautela em notícia marcada: metade da posição da congelada no dia que tem comunicado do FOMC (14h de Nova York,
   só reuniões da agenda) ou divulgação do CPI (8h30 de Nova York), de wescalp.events; a agenda é conhecida antes;
3. VIX: posição = min(1, congelada × mediana do VIX nos 365 dias anteriores / VIX), com o último fechamento do VIX
   já conhecido na decisão (wescalp.factors).
Nos candidatos 1 e 3, a carteira só ajusta quando a posição desejada se afasta 5 p.p. ou mais da atual (ou vai a 0%
ou 100%), como o aprendiz. Período: 11/10/2018 a 30/06/2026, decisão no fechamento do dia, 0,1% por lado.

Teste da sorte de cada candidato: 2.000 deslocamentos circulares, de pelo menos 30 dias, da série do candidato (a
razão nos candidatos 1 e 3, os dias de notícia no 2) dentro do período, com a congelada como está; p = fração com
Sharpe igual ou maior que o do candidato.

Passa para a prova única em 2013-2017 quem cumprir a régua de wescalp.refine contra a congelada: Sharpe maior, queda
máxima igual ou menor, retorno ao ano na pior hora de virada do dia (00, 06, 12, 18 h UTC) e na segunda metade (de
07/2022) não menores, e p ≤ 0,05.

Também mede, sem decidir nada, se o BTC se mexe mais nas 2 h depois de cada notícia do que na mesma hora UTC dos dias
úteis vizinhos sem notícia, e no dia inteiro da notícia contra os outros dias.
"""

import argparse
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import events, factors, learn, refine, regime, study

CANDIDATES = ("volatilidade prevista", "cautela FOMC/CPI", "VIX")
LAMBDA = 0.94
WARMUP = 30
REF_DAYS = 365
CAUTION = 0.5
BAND = learn.BAND
SHIFT_MIN = 30
LUCK = 2000
REACTION_H = 2
CONTROL_DAYS = 14  # dias úteis sem notícia até 14 dias antes e depois


def ewma_vol(close: np.ndarray, lam: float = LAMBDA, warmup: int = WARMUP) -> np.ndarray:
    """σ diário no fechamento de cada dia, só com o passado: EWMA dos quadrados dos log-retornos (RiskMetrics).
    Começa com a média dos quadrados dos `warmup` primeiros retornos; NaN antes disso."""
    r2 = np.concatenate([[np.nan], np.diff(np.log(close)) ** 2])
    out = np.full(len(close), np.nan)
    if len(close) <= warmup:
        return out
    var = float(r2[1:warmup + 1].mean())
    out[warmup] = math.sqrt(var)
    for t in range(warmup + 1, len(close)):
        var = lam * var + (1 - lam) * r2[t]
        out[t] = math.sqrt(var)
    return out


def trailing_median(x: np.ndarray, n: int = REF_DAYS) -> np.ndarray:
    """Mediana dos n dias antes de cada dia, sem ele; NaN enquanto falta histórico."""
    out = np.full(len(x), np.nan)
    if len(x) > n:
        out[n:] = np.median(np.lib.stride_tricks.sliding_window_view(x, n)[:-1], axis=1)
    return out


def banded(target: np.ndarray, band: float = BAND) -> np.ndarray:
    """Só ajusta quando a posição desejada se afasta `band` ou mais da atual, ou vai a 0% ou 100% (como o aprendiz)."""
    pos = np.empty(len(target))
    current = 0.0
    for t, x in enumerate(target):
        if abs(x - current) >= band - 1e-12 or (x in (0.0, 1.0) and x != current):
            current = float(x)
        pos[t] = current
    return pos


def scaled(frozen: np.ndarray, ratio: np.ndarray) -> np.ndarray:
    """min(1, congelada × razão), com a faixa de ajuste; razão 1 onde ainda falta histórico."""
    return banded(np.minimum(1.0, frozen * np.where(np.isfinite(ratio), ratio, 1.0)))


def cautious(frozen: np.ndarray, news: np.ndarray) -> np.ndarray:
    """Metade da congelada nas decisões que valem para um dia com notícia."""
    return frozen * np.where(news, CAUTION, 1.0)


@dataclass
class Inputs:
    d: regime.Daily
    frozen: np.ndarray  # posição da congelada decidida no fechamento de cada dia
    vol_ratio: np.ndarray  # referência / σ no fechamento de cada dia
    vix_ratio: np.ndarray  # mediana do VIX / último VIX conhecido no fechamento de cada dia
    news: np.ndarray  # a decisão do fechamento de cada dia vale para um dia com notícia


def inputs(d: regime.Daily, vix: tuple[np.ndarray, np.ndarray], news_ts: np.ndarray) -> Inputs:
    frozen = refine.position(d.close, refine.BASE)
    sigma = ewma_vol(d.close)
    # O dia k fecha no começo do dia k+1 (mais o deslocamento, menor que um dia): vale o VIX das datas até k.
    v = factors.known("VIXCLS", *vix, d.day + 1)
    has = np.zeros(len(d.day), dtype=bool)
    k = np.searchsorted(d.day, (news_ts - d.offset) // 86400)
    ok = k < len(d.day)
    ok[ok] = d.day[k[ok]] == (news_ts[ok] - d.offset) // 86400
    has[k[ok]] = True
    return Inputs(d, frozen, trailing_median(sigma) / sigma, trailing_median(v) / v, np.concatenate([has[1:], [False]]))


def position(x: Inputs, name: str) -> np.ndarray:
    if name == refine.BASE:
        return x.frozen
    if name == CANDIDATES[0]:
        return scaled(x.frozen, x.vol_ratio)
    if name == CANDIDATES[1]:
        return cautious(x.frozen, x.news)
    if name == CANDIDATES[2]:
        return scaled(x.frozen, x.vix_ratio)
    raise ValueError(name)


def luck_p(x: Inputs, name: str, lo: int, n: int = LUCK, seed: int = 0) -> float:
    """Fração dos deslocamentos circulares da série do candidato (dentro do período) com Sharpe ≥ o real."""
    rets = x.d.close[lo:] / x.d.close[lo - 1:-1] - 1
    caution = name == CANDIDATES[1]
    series = x.news if caution else (x.vol_ratio if name == CANDIDATES[0] else x.vix_ratio)

    def sharpe_of(s):
        pos = cautious(x.frozen, s) if caution else scaled(x.frozen, s)
        r = regime.returns_from_held(regime.held_from(pos, lo), rets)
        return r.mean() / r.std()

    real = sharpe_of(series)
    rng = np.random.default_rng(seed)
    seg = series[lo - 1:]
    hits = 0
    for _ in range(n):
        s = series.copy()
        s[lo - 1:] = np.roll(seg, int(rng.integers(SHIFT_MIN, len(seg) - SHIFT_MIN)))
        hits += sharpe_of(s) >= real
    return hits / n


def reaction(ts: np.ndarray, close: np.ndarray, when: np.ndarray, skip_days: set[int],
             hours: int = REACTION_H, around: int = CONTROL_DAYS) -> tuple[np.ndarray, np.ndarray]:
    """Log-retorno das `hours` horas depois de cada instante de `when` (candles de 5 min) e, para cada um, a média do
    |log-retorno| na mesma hora UTC dos dias úteis a até `around` dias, sem notícia. NaN onde faltam candles."""
    def move(t: int) -> float:
        i, j = np.searchsorted(ts, [t - 300, t + hours * 3600 - 300])  # os candles que fecham em t e t + horas
        if j >= len(ts) or ts[i] != t - 300 or ts[j] != t + hours * 3600 - 300:
            return math.nan
        return math.log(close[j] / close[i])

    ev, ctl = [], []
    for t in when:
        day = int(t) // 86400
        ev.append(move(int(t)))
        others = [abs(move(int(t) + k * 86400)) for k in range(-around, around + 1)
                  if k and factors.weekday(np.array([day + k]))[0] < 5 and day + k not in skip_days]
        ctl.append(np.nanmean(others) if others else math.nan)
    return np.array(ev), np.array(ctl)


def scores_of(by_hour: dict[int, Inputs], name: str, start_day: int, split_day: int, luck: int) -> dict:
    runs = {}
    for off, x in by_hour.items():
        lo = int(np.searchsorted(x.d.day, start_day))
        rets = x.d.close[lo:] / x.d.close[lo - 1:-1] - 1
        runs[off] = (x, lo, rets, regime.held_from(position(x, name), lo))
    x, lo, rets, held = runs[0]
    r = regime.returns_from_held(held, rets)
    when = (x.d.day[lo:] * 86400).astype("datetime64[s]")
    m = study.metrics(r, when.astype("datetime64[M]").astype(int), when.astype("datetime64[Y]").astype(int) + 1970)
    changes = int((np.abs(np.diff(np.concatenate([[0.0], held]))) > 1e-12).sum())
    return {
        "sharpe": m["sharpe"], "dd": m["dd"], "cagr": m["cagr"], "worst_month": m["worst_month"],
        "worst_hour": min(study.cagr(regime.returns_from_held(h, rr)) for _, _, rr, h in runs.values()),
        "second_half": study.cagr(r[x.d.day[lo:] >= split_day]),
        "fee5": study.cagr(regime.returns_from_held(held, rets, 0.005)),
        "exposure": float(held.mean()), "orders_year": changes / (len(r) / 365.25),
        "p": luck_p(x, name, lo, luck) if name != refine.BASE and luck else math.nan,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", type=Path, required=True)
    p.add_argument("--start", default=regime.TEST_START)
    p.add_argument("--factors", type=Path, default=factors.DATA_DIR)
    p.add_argument("--luck", type=int, default=LUCK)
    args = p.parse_args()
    ts, o, h, l, c = study.load(args.file)
    start_day = int(datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc).timestamp() // 86400)
    split_day = int(datetime.fromisoformat(study.SPLIT).replace(tzinfo=timezone.utc).timestamp() // 86400)
    vix = factors.load("VIXCLS", args.factors)
    ev = events.load(args.factors)
    news_ts = np.sort(np.concatenate([ev["fomc"], ev["cpi"]]))
    by_hour = {off: inputs(regime.daily_bars(ts, o, h, l, c, off * 3600), vix, news_ts) for off in study.OFFSETS_H}
    scores = {name: scores_of(by_hour, name, start_day, split_day, args.luck) for name in (refine.BASE, *CANDIDATES)}

    d = by_hour[0].d
    end_day = int(d.day[-1])
    fmt = lambda day: datetime.fromtimestamp(int(day) * 86400, timezone.utc).strftime("%Y-%m-%d")  # noqa: E731
    print(f"período {fmt(start_day)} a {fmt(end_day)}; taxa {regime.FEE:.1%} por lado; dia virando às 00h UTC")
    print(f"{'carteira':<24}{'ao ano':>8}{'queda':>7}{'Sharpe':>8}{'pior mês':>10}{'pior hora':>11}{'2ª metade':>11}"
          f"{'0,5%/lado':>11}{'exposição':>11}{'ordens/ano':>12}{'p sorte':>9}")
    for name, s in scores.items():
        luck = f"{s['p']:>9.3f}" if math.isfinite(s["p"]) else f"{'—':>9}"
        print(f"{name:<24}{s['cagr']:>8.1%}{s['dd']:>7.0%}{s['sharpe']:>8.2f}{s['worst_month']:>10.0%}"
              f"{s['worst_hour']:>11.0%}{s['second_half']:>11.0%}{s['fee5']:>11.0%}{s['exposure']:>11.0%}"
              f"{s['orders_year']:>12.1f}{luck}")
    _, fails = refine.decide(scores)
    for name, f in fails.items():
        print(f"{name}: {'cumpre todos os critérios' if not f else 'não cumpre: ' + ', '.join(f)}")
    ok = [name for name, f in fails.items() if not f]
    print(f"vão para a prova única em 2013-2017: {', '.join(ok) if ok else 'nenhum'}")

    print(f"\nnotícias no período (sem decidir nada): movimento do BTC nas {REACTION_H} h seguintes contra a mesma hora"
          f" UTC dos dias úteis vizinhos sem notícia, e o dia inteiro (00h a 00h UTC) contra os outros dias")
    in_period = lambda t: (t // 86400 >= start_day) & (t // 86400 <= end_day)  # noqa: E731
    skip = {int(t) // 86400 for t in news_ts}
    lo = int(np.searchsorted(d.day, start_day))
    day_ret = np.abs(np.log(d.close[lo:] / d.close[lo - 1:-1]))
    news_day = np.isin(d.day[lo:], list(skip))
    for kind in ("fomc", "cpi"):
        when = ev[kind][in_period(ev[kind])]
        move, ctl = reaction(ts, c, when, skip)
        ok = np.isfinite(move) & np.isfinite(ctl)
        kind_day = np.isin(d.day[lo:], [int(t) // 86400 for t in when])
        print(f"{kind.upper():<5} {ok.sum():>3} notícias: mediana do |movimento| {np.median(np.abs(move[ok])):.2%} contra"
              f" {np.median(ctl[ok]):.2%} nos dias sem notícia ({np.median(np.abs(move[ok])) / np.median(ctl[ok]):.1f}x);"
              f" subiu em {(move[ok] > 0).mean():.0%}; dia inteiro: |retorno| médio {day_ret[kind_day].mean():.2%}"
              f" contra {day_ret[~news_day].mean():.2%}")


if __name__ == "__main__":
    main()
