"""A chance de cada virada da congelada virar tendência: um filtro que só compra nas viradas com mais chance.

    python -m wescalp.flips       # candles da Binance de cada moeda em data/candles (ver wescalp.altcoins)

Fixado antes de rodar (26/10/2026), sem ajuste depois. A congelada perde em cerca de 80% das operações (o preço fura a
média e volta) e ganha nas poucas que viram tendência. Dá para saber, no dia da compra, quais têm mais chance?
- Compras: cada vez que o fechamento diário passa para cima da média de 50, 100 ou 150 dias (o primeiro dia de cada
  média não conta), no BTC desde 2017 e em ETH, SOL, XRP, DOGE e ADA desde 2021, para ter mais exemplos.
- Resultado: a operação vai até a venda seguinte da mesma média, com 0,2% de ida e volta; deu certo se terminou no lucro.
- Variáveis no dia da compra, só com o passado (8): distância do fechamento à média; inclinação da média em 10 dias;
  viradas para cima dessa média nos 60 dias anteriores; volatilidade diária de 30 dias; retorno de 30 dias; quantas das
  outras duas médias estão compradas; e qual é a média (duas colunas, 50 e 100).
- Modelo: regressão logística com ridge (variáveis padronizadas, encolhimento 1), treinada para cada ano de 2021 a
  2026 com as compras de todas as moedas cujas operações terminaram antes de 1º de janeiro daquele ano.
- Filtro: no ano de teste, a compra só acontece se a chance prevista for pelo menos a mediana das chances previstas no
  treino; a média pulada fica em dólar até a compra seguinte dela.
- Régua, no BTC, de 01/2021 a 06/2026, 0,1% por lado: passa se o Sharpe com filtro for maior que o da congelada sem
  filtro e p ≤ 0,05 no teste da sorte (2.000 filtros que pulam ao acaso o mesmo número de compras; p = fração com
  Sharpe igual ou maior).
- Mostra também quantas das compras aceitas e das puladas deram certo, os pesos do último modelo e, só para comparar,
  o mesmo nas altcoins de 2022 a 06/2026.
"""

import argparse
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import altcoins, brain, intraday, regime, study

DAY = 86400
FEE = regime.FEE
LENGTHS = (50, 100, 150)
TEST_YEARS = tuple(range(2021, 2027))
END = intraday.utc("2026-07-01")
ALT_FROM = 2022
SLOPE_DAYS, FLIP_WINDOW, VOL_DAYS, RET_DAYS = 10, 60, 30, 30
LAMBDA = 1.0
LUCK = 2000
VARIABLES = ("distância à média", "inclinação da média 10 d", "viradas nos 60 d", "volatilidade 30 d", "retorno 30 d",
             "outras médias compradas", "média de 50", "média de 100")


@dataclass(frozen=True)
class Event:
    asset: str
    n: int  # a média
    day: int  # índice do dia da compra (fechamento acima da média)
    exit: int | None  # índice do dia da venda (fechamento abaixo), None se ainda aberta no fim dos dados
    ret: float  # resultado com taxa (até o último fechamento, se aberta)
    x: tuple[float, ...]


def above(close: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(1 se o fechamento está acima da média de n dias, a média); 0 enquanto a média não existe."""
    m = regime.sma(close, n)
    return np.where(np.isnan(m), 0.0, (close > m).astype(float)), m


def events(asset: str, close: np.ndarray, fee: float = FEE) -> list[Event]:
    comps = {n: above(close, n) for n in LENGTHS}
    lr = np.concatenate([[np.nan], np.diff(np.log(close))])
    out = []
    for n, (on, m) in comps.items():
        ups = np.flatnonzero((on[1:] == 1) & (on[:-1] == 0)) + 1
        ups = ups[~np.isnan(m[ups - 1])]  # o dia em que a média passa a existir não conta
        for t in ups:
            if t < max(SLOPE_DAYS, FLIP_WINDOW, VOL_DAYS, RET_DAYS):
                continue
            off = np.flatnonzero(on[t + 1:] == 0)
            u = int(t + 1 + off[0]) if len(off) else None
            px = close[u] if u is not None else close[-1]
            others = sum(comps[k][0][t] for k in LENGTHS if k != n)
            past_ups = int(((on[t - FLIP_WINDOW + 1:t] == 1) & (on[t - FLIP_WINDOW:t - 1] == 0)).sum())
            x = (close[t] / m[t] - 1, m[t] / m[t - SLOPE_DAYS] - 1, past_ups, float(np.std(lr[t - VOL_DAYS + 1:t + 1])),
                 close[t] / close[t - RET_DAYS] - 1, others, float(n == 50), float(n == 100))
            if not np.all(np.isfinite(x)):  # média com menos de 10 dias de vida
                continue
            out.append(Event(asset, n, int(t), u, float(px / close[t] * (1 - fee) ** 2 - 1), x))
    return out


def fit_logit(x: np.ndarray, y: np.ndarray, lam: float = LAMBDA, iters: int = 100):
    """Regressão logística com ridge (sem encolher o intercepto), por Newton; devolve a função de probabilidade e os pesos."""
    mu, sd = x.mean(axis=0), x.std(axis=0)
    sd[sd == 0] = 1.0
    z = np.column_stack([np.ones(len(x)), (x - mu) / sd])
    w = np.zeros(z.shape[1])
    reg = lam * np.eye(z.shape[1])
    reg[0, 0] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-z @ w))
        step = np.linalg.solve(z.T @ (z * (p * (1 - p))[:, None]) + reg, z.T @ (p - y) + reg @ w)
        w -= step
        if np.abs(step).max() < 1e-10:
            break
    return (lambda q: 1 / (1 + np.exp(-(np.column_stack([np.ones(len(q)), (q - mu) / sd]) @ w)))), w


def filtered(on: np.ndarray, skipped: list[Event]) -> np.ndarray:
    """A posição da média com as compras puladas: fica em dólar do dia da compra até a venda."""
    out = on.copy()
    for e in skipped:
        out[e.day:e.exit if e.exit is not None else len(out)] = 0.0
    return out


def day_index(days: np.ndarray, t: int) -> int:
    return int(np.searchsorted(days, t // DAY))


@dataclass
class Asset:
    name: str
    day: np.ndarray
    close: np.ndarray
    events: list[Event]


def decisions(assets: list[Asset], years=TEST_YEARS) -> dict[tuple[str, int, int], tuple[bool, float, int]]:
    """Para cada compra fora da amostra, (aceita, chance prevista, ano), com um modelo por ano."""
    out = {}
    for y in years:
        cut = intraday.utc(f"{y}-01-01")
        train = [e for a in assets for e in a.events if e.exit is not None and a.day[e.exit] * DAY < cut]
        prob, _ = fit_logit(np.array([e.x for e in train]), np.array([float(e.ret > 0) for e in train]))
        threshold = float(np.median(prob(np.array([e.x for e in train]))))
        nxt = intraday.utc(f"{y + 1}-01-01")
        for a in assets:
            test = [e for e in a.events if cut <= a.day[e.day] * DAY < nxt]
            if test:
                p = prob(np.array([e.x for e in test]))
                for e, q in zip(test, p):
                    out[(a.name, e.n, e.day)] = (bool(q >= threshold), float(q), y)
    return out


def run(a: Asset, dec: dict, start: int, end: int, luck: int, seed: int = 0) -> dict:
    """A congelada com e sem filtro em [start, end), e o teste da sorte com filtros ao acaso."""
    lo, hi = day_index(a.day, start), day_index(a.day, end)
    rets = a.close[lo:hi] / a.close[lo - 1:hi - 1] - 1
    ons = {n: above(a.close, n)[0] for n in LENGTHS}
    test = [e for e in a.events if (a.name, e.n, e.day) in dec and lo <= e.day < hi]
    skip = [e for e in test if not dec[(a.name, e.n, e.day)][0]]

    def returns(skipped: list[Event]) -> np.ndarray:
        held = np.mean([regime.held_from(filtered(ons[n], [e for e in skipped if e.n == n])[:hi], lo) for n in LENGTHS], axis=0)
        return regime.returns_from_held(held, rets)

    score = lambda r: float(r.mean() / r.std() * math.sqrt(365)) if r.std() > 0 else 0.0  # noqa: E731
    plain, filt = returns([]), returns(skip)
    rng = np.random.default_rng(seed)
    hits = sum(score(returns([test[i] for i in rng.choice(len(test), len(skip), replace=False)])) >= score(filt)
               for _ in range(luck)) if luck and test else 0
    stats = lambda r: {"cagr": study.cagr(r), "dd": study.max_dd(r), "sharpe": score(r)}  # noqa: E731
    years = (hi - lo) / 365.25
    acc, sk = [e for e in test if e not in skip], skip
    win = lambda es: float(np.mean([e.ret > 0 for e in es])) if es else math.nan  # noqa: E731
    return {"plain": stats(plain), "filtered": stats(filt), "p": hits / luck if luck else math.nan,
            "buys": len(test), "accepted": len(acc), "per_year": (len(acc) / years, len(test) / years),
            "win": (win(acc), win(sk)), "mean": (float(np.mean([e.ret for e in acc])) if acc else math.nan,
                                                 float(np.mean([e.ret for e in sk])) if sk else math.nan)}


def load_asset(sym: str, data_dir: Path) -> Asset:
    ts, o, h, l, c, _ = altcoins.load(sym, data_dir)
    d = regime.daily_bars(ts, o, h, l, c)
    hi = int(np.searchsorted(d.day, END // DAY))
    day, close = d.day[:hi], d.close[:hi]
    assert np.all(np.diff(day) == 1), f"{sym}: falta dia"
    return Asset(sym[:-4], day, close, events(sym[:-4], close))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=brain.CANDLE_DIR)
    p.add_argument("--luck", type=int, default=LUCK)
    args = p.parse_args()
    assets = [load_asset(s, args.data) for s in (altcoins.REFERENCE, *altcoins.COINS)]
    dec = decisions(assets)
    print("Compras por ano fora da amostra e quantas o filtro aceitou:")
    for y in TEST_YEARS:
        ys = [k for k, v in dec.items() if v[2] == y]
        print(f"  {y}: {len(ys)} compras, {sum(dec[k][0] for k in ys)} aceitas")
    fmt = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%m/%Y")  # noqa: E731
    for a in assets:
        start = intraday.utc(f"{TEST_YEARS[0] if a.name == 'BTC' else ALT_FROM}-01-01")
        s = run(a, dec, start, END, args.luck if a.name == "BTC" else 0)
        pl, fl = s["plain"], s["filtered"]
        tag = "" if a.name != "BTC" else (" PASSA" if fl["sharpe"] > pl["sharpe"] and s["p"] <= 0.05 else " não passa")
        print(f"\n{a.name} ({fmt(start)} a 06/2026): {s['buys']} compras, {s['accepted']} aceitas"
              f" ({s['per_year'][0]:.1f} contra {s['per_year'][1]:.1f} por ano)")
        print(f"  deram certo: aceitas {s['win'][0]:.0%} (média {s['mean'][0]:+.1%}), puladas {s['win'][1]:.0%}"
              f" (média {s['mean'][1]:+.1%})")
        print(f"  sem filtro: {pl['cagr']:+.1%} ao ano, queda {pl['dd']:.0%}, Sharpe {pl['sharpe']:.2f};"
              f" com filtro: {fl['cagr']:+.1%} ao ano, queda {fl['dd']:.0%}, Sharpe {fl['sharpe']:.2f}"
              + (f"; p da sorte {s['p']:.3f}{tag}" if a.name == "BTC" else ""))
    cut = intraday.utc(f"{TEST_YEARS[-1]}-01-01")
    train = [e for a in assets for e in a.events if e.exit is not None and a.day[e.exit] * DAY < cut]
    _, w = fit_logit(np.array([e.x for e in train]), np.array([float(e.ret > 0) for e in train]))
    print(f"\nPesos do modelo de 2026 (variáveis padronizadas; {len(train)} compras, {np.mean([e.ret > 0 for e in train]):.0%}"
          " deram certo):")
    for name, coef in zip(VARIABLES, w[1:]):
        print(f"  {name}: {coef:+.2f}")


if __name__ == "__main__":
    main()
