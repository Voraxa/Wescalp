"""Estudo das estratégias que passaram no teste da sorte: qual é melhor e quão firme é cada uma.

    python -m wescalp.study --file data/candles/BINANCE-BTCUSDT-5m.csv.gz

Candidatas (todas de wescalp.regime, no BTC spot, de 11/10/2018 a 30/06/2026, 0,1% por lado, reinvestindo):
- segurar BTC (referência);
- média de 50, 100 e 150 dias: comprado quando o fechamento diário está acima da média;
- combinação 50/100/150: 1/3 do capital em cada uma das três (não depende de escolher um parâmetro);
- média de 100 dias + ciclos de 3%: os ciclos de wescalp.spot só nos dias em que a média de 100 está comprada.

Para cada uma: retorno ao ano, queda máxima, Sharpe, meses e anos positivos, fator de lucro mensal (soma dos meses
positivos sobre a dos negativos), maior tempo abaixo do topo, faixa de 5% a 95% do retorno ao ano e da queda num
bootstrap estacionário (blocos de ~20 dias), as duas metades do período, taxa de 0,2% e 0,5% por lado e a hora em
que o dia vira (00, 06, 12 e 18 h UTC). Também a vizinhança da média (60 a 200 dias) e o teste da sorte da
combinação (cada componente sorteado por conta própria).
"""

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import brain, regime, spot

SPLIT = "2022-07-01"  # as duas metades do período
OFFSETS_H = (0, 6, 12, 18)
FEES = (0.002, 0.005)
CYCLE_TP = 0.03
BOOT = 1000
BLOCK = 20


def stationary_bootstrap(r: np.ndarray, n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """n reamostras de r (Politis e Romano): blocos de tamanho médio `block`, emendando o fim no começo."""
    t = len(r)
    out = np.empty((n, t))
    for s in range(n):
        new = rng.random(t) < 1 / block
        new[0] = True
        first = np.flatnonzero(new)
        bid = np.cumsum(new) - 1
        starts = rng.integers(0, t, len(first))
        out[s] = r[(starts[bid] + np.arange(t) - first[bid]) % t]
    return out


def cagr(r: np.ndarray) -> float:
    return float(np.prod(1 + r) ** (365.25 / len(r)) - 1)


def max_dd(r: np.ndarray) -> float:
    eq = np.cumprod(1 + r)
    return float((1 - eq / np.maximum.accumulate(eq)).max())


def underwater_days(r: np.ndarray) -> int:
    """Maior sequência de dias abaixo do topo anterior do patrimônio."""
    eq = np.cumprod(1 + r)
    under = eq < np.maximum.accumulate(eq) * (1 - 1e-12)
    best = run = 0
    for u in under:
        run = run + 1 if u else 0
        best = max(best, run)
    return best


def by_period(r: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Retorno composto de cada período (mês, ano) em ordem."""
    return np.array([np.prod(1 + r[keys == k]) - 1 for k in np.unique(keys)])


def profit_factor(x: np.ndarray) -> float:
    loss = -x[x < 0].sum()
    return float(x[x > 0].sum() / loss) if loss > 0 else math.inf


def metrics(r: np.ndarray, months: np.ndarray, years: np.ndarray) -> dict:
    m, y = by_period(r, months), by_period(r, years)
    return {
        "cagr": cagr(r), "dd": max_dd(r), "sharpe": float(r.mean() / r.std() * math.sqrt(365)) if r.std() else 0.0,
        "months_pos": float((m > 0).mean()), "pf": profit_factor(m), "worst_month": float(m.min()),
        "years_pos": int((y > 0).sum()), "years": len(y), "worst_year": float(y.min()), "underwater": underwater_days(r),
    }


def load(path: Path):
    cs = brain.load_candles(path, "BTCUSDT", "spot")
    ts = np.array([int(c.ts.timestamp()) for c in cs], dtype=np.int64)
    o, h, l, c = (np.array([getattr(x, k) for x in cs]) for k in ("open", "high", "low", "close"))
    return ts, o, h, l, c


def daily_returns(ts, o, h, l, c, start_day: int, offset_h: int, fee: float, cycles_tp: float | None = None):
    """Retornos diários das candidatas com o dia virando em `offset_h` horas UTC. Devolve (dias, {nome: r})."""
    d = regime.daily_bars(ts, o, h, l, c, offset_h * 3600)
    lo = int(np.searchsorted(d.day, start_day))
    rets = d.close[lo:] / d.close[lo - 1:-1] - 1
    all_rules = regime.rules(d.close)
    held = {k: regime.held_from(all_rules[k], lo) for k in ("sma 50", "sma 100", "sma 150")}
    out = {"segurar BTC": regime.returns_from_held(np.ones(len(rets)), rets, fee)}
    for k, name in (("sma 50", "média 50"), ("sma 100", "média 100"), ("sma 150", "média 150")):
        out[name] = regime.returns_from_held(held[k], rets, fee)
    out["combinação 50/100/150"] = regime.returns_from_held(np.mean(list(held.values()), axis=0), rets, fee)
    if cycles_tp:
        f = brain.Frame(ts, o, h, l, c, np.full(len(ts), np.nan), np.zeros((len(ts), 1)), ["x"], {})
        mask = regime.bar_mask(ts, d.day, held["sma 100"], lo, d.offset)
        lo_bar = int(np.searchsorted(ts, d.day[lo] * 86400 + d.offset))
        # Fecho de cada dia: o último candle antes de o dia virar.
        day_of = (ts - d.offset) // 86400
        record = np.zeros(len(ts), dtype=bool)
        record[:-1] = day_of[1:] != day_of[:-1]
        record[-1] = True
        record[:lo_bar] = False
        pol = spot.simulate(f, lo_bar, len(ts), len(ts), lambda k: mask[k], cycles_tp, fee=fee, record=record)
        eq = np.array(pol.equity)
        # Cada anotação é o fecho de um dia do período; alinha com os dias lo..fim.
        r = np.concatenate([[eq[0] - 1], eq[1:] / eq[:-1] - 1])
        out[f"média 100 + ciclos {cycles_tp:.0%}"] = r[:len(rets)] if len(r) >= len(rets) else np.pad(r, (0, len(rets) - len(r)))
    return d.day[lo:], out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", type=Path, required=True)
    p.add_argument("--start", default=regime.TEST_START)
    args = p.parse_args()
    ts, o, h, l, c = load(args.file)
    start_day = int(datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc).timestamp() // 86400)
    split_day = int(datetime.fromisoformat(SPLIT).replace(tzinfo=timezone.utc).timestamp() // 86400)

    days, base = daily_returns(ts, o, h, l, c, start_day, 0, regime.FEE, CYCLE_TP)
    when = (days * 86400).astype("datetime64[s]")
    months, years = when.astype("datetime64[M]").astype(int), when.astype("datetime64[Y]").astype(int) + 1970
    print(f"período {when[0].astype(datetime):%Y-%m-%d} a {when[-1].astype(datetime):%Y-%m-%d}, {len(days)} dias;"
          f" taxa {regime.FEE:.1%} por lado; tudo reinvestido; fator de lucro = soma dos meses positivos / negativos")

    head = (f"{'estratégia':<26}{'ao ano':>8}{'queda':>7}{'Sharpe':>8}{'meses+':>8}{'fator':>7}{'pior mês':>10}"
            f"{'anos+':>7}{'pior ano':>10}{'dias abaixo do topo':>21}")
    print("\n" + head)
    for name, r in base.items():
        s = metrics(r, months, years)
        print(f"{name:<26}{s['cagr']:>8.1%}{s['dd']:>7.0%}{s['sharpe']:>8.2f}{s['months_pos']:>8.0%}{s['pf']:>7.2f}"
              f"{s['worst_month']:>10.0%}{s['years_pos']:>4}/{s['years']}{s['worst_year']:>10.0%}{s['underwater']:>21}")

    print(f"\nbootstrap estacionário ({BOOT} reamostras, blocos de ~{BLOCK} dias): faixa de 5% a 95%")
    print(f"{'estratégia':<26}{'ao ano':>20}{'queda máxima':>20}{'P(bate segurar)':>18}")
    boots = {name: stationary_bootstrap(r, BOOT, BLOCK, np.random.default_rng(1)) for name, r in base.items()}
    for name, b in boots.items():  # mesma semente: as reamostras pegam os mesmos dias em todas as estratégias
        cg = np.array([cagr(x) for x in b])
        dd = np.array([max_dd(x) for x in b])
        hold = np.array([cagr(x) for x in boots["segurar BTC"]])
        print(f"{name:<26}{np.quantile(cg, .05):>9.0%} a {np.quantile(cg, .95):>4.0%}"
              f"{np.quantile(dd, .05):>11.0%} a {np.quantile(dd, .95):>4.0%}{(cg > hold).mean():>18.0%}")

    print(f"\nas duas metades (dia que vira: 00h UTC), ao ano / queda máxima:")
    first, second = days < split_day, days >= split_day
    print(f"{'estratégia':<26}{'até 06/2022':>18}{'de 07/2022':>18}")
    for name, r in base.items():
        print(f"{name:<26}{cagr(r[first]):>9.0%} / {max_dd(r[first]):>4.0%}{cagr(r[second]):>11.0%} / {max_dd(r[second]):>4.0%}")

    print("\ntaxa mais alta (ao ano):")
    print(f"{'estratégia':<26}" + "".join(f"{f:>10.1%}" for f in (regime.FEE, *FEES)))
    fee_runs = {f: daily_returns(ts, o, h, l, c, start_day, 0, f, CYCLE_TP)[1] for f in FEES}
    for name in base:
        print(f"{name:<26}{cagr(base[name]):>10.0%}" + "".join(f"{cagr(fee_runs[f][name]):>10.0%}" for f in FEES))

    print("\nhora em que o dia vira (ao ano / queda máxima):")
    print(f"{'estratégia':<26}" + "".join(f"{f'{x:02d}h UTC':>16}" for x in OFFSETS_H))
    hour_runs = {x: daily_returns(ts, o, h, l, c, start_day, x, regime.FEE, CYCLE_TP)[1] for x in OFFSETS_H}
    for name in base:
        print(f"{name:<26}" + "".join(f"{cagr(hour_runs[x][name]):>8.0%} / {max_dd(hour_runs[x][name]):>4.0%}"
                                      for x in OFFSETS_H))

    d = regime.daily_bars(ts, o, h, l, c)
    lo = int(np.searchsorted(d.day, start_day))
    rets = d.close[lo:] / d.close[lo - 1:-1] - 1
    print("\nvizinhança da média (ao ano / queda máxima / Sharpe):")
    for n in range(60, 201, 10):
        m = regime.sma(d.close, n)
        pos = np.where(np.isnan(m), 0.0, (d.close > m).astype(float))
        r = regime.returns_from_held(regime.held_from(pos, lo), rets)
        print(f"  média {n:>3}: {cagr(r):>5.0%} / {max_dd(r):>4.0%} / {r.mean() / r.std() * math.sqrt(365):.2f}")

    all_rules = regime.rules(d.close)
    held = [regime.held_from(all_rules[k], lo) for k in ("sma 50", "sma 100", "sma 150")]
    ens = regime.returns_from_held(np.mean(held, axis=0), rets)
    rng = np.random.default_rng(0)
    sims = []
    for _ in range(2000):
        hs = np.mean([regime.shuffled_timing(x, rng) for x in held], axis=0)
        sims.append(regime.returns_from_held(hs, rets))
    tot = np.array([np.prod(1 + s) for s in sims])
    sh = np.array([s.mean() / s.std() for s in sims])
    dd = np.array([max_dd(s) for s in sims])
    print(f"\nteste da sorte da combinação (2.000 sorteios, cada componente sorteado por conta própria):"
          f" p do total {(tot >= np.prod(1 + ens)).mean():.3f}, p do Sharpe {(sh >= ens.mean() / ens.std()).mean():.3f},"
          f" p da queda {(dd <= max_dd(ens)).mean():.3f}")


if __name__ == "__main__":
    main()
