"""Exposição por regime no BTC spot: comprado quando a tendência diária é de alta, em USDT quando não.

    python -m wescalp.regime --file data/candles/BINANCE-BTCUSDT-5m.csv.gz

A pista dos testes de wescalp.spot: o que mais pesou no juiz foram as variáveis do diário, e ele ganhava ficando
fora nas quedas. Aqui a regra é só essa, sem ciclos curtos. Cada regra olha o fechamento diário (UTC) e decide a
posição do dia seguinte inteiro (sem olhar o futuro). Taxa de 0,1% em cada compra e venda; USDT parado não rende.

Regras, fixadas antes de rodar (as clássicas da literatura de seguir tendência, sem ajuste):
- sma N: comprado se o fechamento está acima da média simples de N dias (N = 50, 100, 150, 200);
- ema a/b: comprado se a média exponencial de a dias está acima da de b dias (20/50 e 50/200);
- mom N: comprado se o fechamento está acima do de N dias atrás (N = 30, 90, 180).
"""

import argparse
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import brain, spot
from .validation import deflated_sharpe, sharpe

FEE = spot.FEE
TEST_START = "2018-10-11"  # o mesmo começo do teste de wescalp.spot na Binance, para comparar


@dataclass
class Daily:
    day: np.ndarray  # dias desde a época (UTC)
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray


def daily_bars(ts: np.ndarray, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> Daily:
    """Barras diárias (UTC) a partir dos candles de 5 min em ordem de tempo."""
    day = ts // 86400
    keys, first = np.unique(day, return_index=True)
    last = np.concatenate([first[1:], [len(ts)]]) - 1
    return Daily(keys, o[first], np.maximum.reduceat(h, first), np.minimum.reduceat(l, first), c[last])


def sma(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        c = np.cumsum(np.concatenate([[0.0], x]))
        out[n - 1:] = (c[n:] - c[:-n]) / n
    return out


def rules(close: np.ndarray) -> dict[str, np.ndarray]:
    """Posição desejada (0 ou 1) no fechamento de cada dia; 0 enquanto o indicador não tem histórico."""
    out = {}
    for n in (50, 100, 150, 200):
        m = sma(close, n)
        out[f"sma {n}"] = np.where(np.isnan(m), 0.0, (close > m).astype(float))
    for a, b in ((20, 50), (50, 200)):
        ea, eb = brain.ema(close, a), brain.ema(close, b)
        ok = np.arange(len(close)) >= b  # a média lenta precisa de b dias para valer
        out[f"ema {a}/{b}"] = np.where(ok, (ea > eb).astype(float), 0.0)
    for n in (30, 90, 180):
        prev = brain.lagged(close, n)
        out[f"mom {n}"] = np.where(np.isnan(prev), 0.0, (close > prev).astype(float))
    return out


@dataclass
class Result:
    daily: np.ndarray  # retorno de cada dia do período, já com taxa
    trades: list[float]  # resultado de cada compra até a venda, com taxa
    in_market: float


def backtest(close: np.ndarray, pos: np.ndarray, lo: int, fee: float = FEE) -> Result:
    """A posição decidida no fechamento do dia d vale para o dia d+1. Período: retornos dos dias lo..fim.

    Começa em USDT no fechamento do dia lo-1: se a regra estiver comprada ali, compra (e paga a taxa).
    """
    held = pos[lo - 1:-1]  # posição durante cada dia lo..fim
    rets = close[lo:] / close[lo - 1:-1] - 1
    prev = np.concatenate([[0.0], held[:-1]])
    daily = held * rets - fee * np.abs(held - prev)  # taxa de 0,1% do capital em cada compra e venda
    trades, entry = [], None
    for k, (h, p) in enumerate(zip(held, prev)):
        if h > p:
            entry = k
        if h < p and entry is not None:
            trades.append(float(np.prod(1 + daily[entry:k + 1]) - 1))
            entry = None
    if entry is not None:
        trades.append(float(np.prod(1 + daily[entry:]) * (1 - fee) - 1))  # vende no fim do período
    return Result(daily, trades, float(held.mean()))


def held_from(pos: np.ndarray, lo: int) -> np.ndarray:
    """Posição durante cada dia lo..fim (a decidida no fechamento do dia anterior)."""
    return pos[lo - 1:-1]


def returns_from_held(held: np.ndarray, rets: np.ndarray, fee: float = FEE) -> np.ndarray:
    prev = np.concatenate([[0.0], held[:-1]])
    return held * rets - fee * np.abs(held - prev)


def shuffled_timing(held: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """O "macaco com moeda": os mesmos trechos comprado e fora, com os mesmos comprimentos, em ordem sorteada.

    Mantém o número de operações, o tempo comprado e a duração de cada operação; só o momento muda.
    """
    edges = np.flatnonzero(np.diff(held)) + 1
    parts = np.split(held, edges)
    first = parts[0][0]
    same = [len(p) for p in parts if p[0] == first]
    other = [len(p) for p in parts if p[0] != first]
    rng.shuffle(same)
    rng.shuffle(other)
    out = []
    for i in range(len(parts)):
        n = same[i // 2] if i % 2 == 0 else other[i // 2]
        out.append(np.full(n, first if i % 2 == 0 else 1 - first))
    return np.concatenate(out)


def luck_test(held: np.ndarray, rets: np.ndarray, n: int = 2000, seed: int = 0, fee: float = FEE) -> dict:
    """Fração das estratégias sorteadas (mesmos trechos, momento ao acaso) que igualam ou batem a regra."""
    rng = np.random.default_rng(seed)

    def score(h):
        d = returns_from_held(h, rets, fee)
        eq = np.cumprod(1 + d)
        return eq[-1], d.mean() / d.std() if d.std() > 0 else 0.0, (1 - eq / np.maximum.accumulate(eq)).max()

    total, sh, dd = score(held)
    sims = np.array([score(shuffled_timing(held, rng)) for _ in range(n)])
    return {"p_total": float((sims[:, 0] >= total).mean()), "p_sharpe": float((sims[:, 1] >= sh).mean()),
            "p_dd": float((sims[:, 2] <= dd).mean()), "median_total": float(np.median(sims[:, 0]) - 1)}


def bar_mask(ts: np.ndarray, day: np.ndarray, held: np.ndarray, lo: int) -> np.ndarray:
    """Para cada candle de 5 min, se a regra estava comprada naquele dia (held cobre os dias lo..fim)."""
    idx = np.searchsorted(day, ts // 86400) - lo
    out = np.zeros(len(ts), dtype=bool)
    ok = (idx >= 0) & (idx < len(held))
    out[ok] = held[idx[ok]] > 0
    return out


_CYC: dict = {}


def _cycle_total(args) -> tuple[float, float, int]:
    """Ciclos de wescalp.spot só nos dias em que `held` está comprado: (soma em % do capital, acerto, ciclos)."""
    held, tp = args
    c = _CYC
    mask = bar_mask(c["f"].ts, c["day"], held, c["lo_day"])
    pol = spot.simulate(c["f"], c["lo_bar"], c["end"], c["end"], lambda k: mask[k], tp)
    r = np.array([t.ret_bps for t in pol.trades]) / 100
    return float(r.sum()), float((r > 0).mean()) if len(r) else math.nan, len(r)


def cycles_luck(f: brain.Frame, d: Daily, held: np.ndarray, lo_day: int, tp: float, n: int, jobs: int = 4,
                seed: int = 0) -> dict:
    """A regra com ciclos contra a mesma regra com o momento sorteado (mesmos trechos), também com ciclos."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    _CYC.update(f=f, day=d.day, lo_day=lo_day, lo_bar=int(np.searchsorted(f.ts, d.day[lo_day] * 86400)),
                end=len(f.ts))
    rng = np.random.default_rng(seed)
    work = [(held, tp)] + [(shuffled_timing(held, rng), tp) for _ in range(n)]
    with ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context("fork")) as pool:
        out = list(pool.map(_cycle_total, work, chunksize=4))
    total, win, cycles = out[0]
    sims = np.array([o[0] for o in out[1:]])
    days = (f.ts[-1] - f.ts[_CYC["lo_bar"]]) / 86400
    return {"total": total, "win": win, "per_day": cycles / days, "p": float((sims >= total).mean()),
            "median": float(np.median(sims))}


def summary(r: Result, years: np.ndarray) -> dict:
    eq = np.cumprod(1 + r.daily)
    dd = float((1 - eq / np.maximum.accumulate(eq)).max())
    n_years = len(r.daily) / 365.25
    weekly = [float(np.prod(1 + r.daily[i:i + 7]) - 1) for i in range(0, len(r.daily) - 6, 7)]
    t = np.array(r.trades)
    by_year = {int(y): float(np.prod(1 + r.daily[years == y]) - 1) for y in np.unique(years)}
    return {
        "total": float(eq[-1] - 1), "cagr": float(eq[-1] ** (1 / n_years) - 1), "dd": dd,
        "sharpe": float(r.daily.mean() / r.daily.std() * math.sqrt(365)) if r.daily.std() > 0 else 0.0,
        "trades_year": len(t) / n_years, "win": float((t > 0).mean()) if len(t) else math.nan,
        "in_market": r.in_market, "weekly": weekly, "by_year": by_year,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", type=Path, required=True, help="candles de 5 min (ex.: o de wescalp.binance)")
    p.add_argument("--start", default=TEST_START, help="primeiro dia do período (AAAA-MM-DD)")
    p.add_argument("--luck", type=int, default=2000, help="sorteios por regra no teste da sorte")
    p.add_argument("--cycles", type=float, nargs="*", default=[], help="alvos dos ciclos de wescalp.spot dentro da regra")
    p.add_argument("--cycle-rules", nargs="*", default=["sma 50", "sma 100"], help="regras para os ciclos")
    p.add_argument("--cycle-luck", type=int, default=200, help="sorteios por regra e alvo no teste dos ciclos")
    args = p.parse_args()
    cs = brain.load_candles(args.file, "BTCUSDT", "spot")
    ts = np.array([int(c.ts.timestamp()) for c in cs], dtype=np.int64)
    d = daily_bars(ts, *(np.array([getattr(c, k) for c in cs]) for k in ("open", "high", "low", "close")))
    lo = int(np.searchsorted(d.day, datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc).timestamp() // 86400))
    years = (d.day[lo:] * 86400).astype("datetime64[s]").astype("datetime64[Y]").astype(int) + 1970
    fmt = lambda k: datetime.fromtimestamp(int(d.day[k]) * 86400, timezone.utc).strftime("%Y-%m-%d")  # noqa: E731
    print(f"{args.file.name}: {len(d.day)} dias; período {fmt(lo)} a {fmt(len(d.day) - 1)}; taxa {FEE:.1%} por lado")
    res = {"segurar BTC": backtest(d.close, np.ones(len(d.close)), lo)}
    res.update({k: backtest(d.close, v, lo) for k, v in rules(d.close).items()})
    sums = {k: summary(r, years) for k, r in res.items()}
    print(f"{'regra':<13}{'total':>10}{'ao ano':>9}{'queda máx':>11}{'Sharpe':>8}{'trades/ano':>12}{'acerto':>8}"
          f"{'comprado':>10}")
    for k, s in sums.items():
        print(f"{k:<13}{s['total']:>10.0%}{s['cagr']:>9.1%}{s['dd']:>11.1%}{s['sharpe']:>8.2f}{s['trades_year']:>12.1f}"
              f"{s['win']:>8.0%}{s['in_market']:>10.0%}")
    ys = sorted(sums["segurar BTC"]["by_year"])
    print("\npor ano:")
    print(f"{'regra':<13}" + "".join(f"{y:>8}" for y in ys))
    for k, s in sums.items():
        print(f"{k:<13}" + "".join(f"{s['by_year'][y]:>8.0%}" for y in ys))
    rets = d.close[lo:] / d.close[lo - 1:-1] - 1
    print(f"\nteste da sorte ({args.luck} sorteios por regra): os mesmos trechos comprado e fora em ordem sorteada;"
          " p = fração dos sorteios que iguala ou bate a regra (p pequeno = não é sorte)")
    print(f"{'regra':<13}{'p do total':>12}{'p do Sharpe':>13}{'p da queda':>12}{'total mediano sorteado':>24}")
    for k, v in rules(d.close).items():
        lt = luck_test(held_from(v, lo), rets, args.luck)
        print(f"{k:<13}{lt['p_total']:>12.3f}{lt['p_sharpe']:>13.3f}{lt['p_dd']:>12.3f}{lt['median_total']:>23.0%}")
    if args.cycles:
        f = brain.Frame(ts, *(np.array([getattr(c, k) for c in cs]) for k in ("open", "high", "low", "close")),
                        np.full(len(ts), np.nan), np.zeros((len(ts), 1)), ["x"], {})
        print(f"\nciclos de wescalp.spot (alvo fixo, DCA com novo sinal, stop 3 alvos abaixo) só nos dias em que a"
              f" regra está comprada, contra {args.cycle_luck} sorteios do momento da regra (mesmos trechos)")
        print(f"{'regra':<13}{'alvo':>6}{'ciclos/dia':>12}{'acerto':>8}{'soma':>9}{'sorteado (mediana)':>20}{'p':>7}")
        all_rules = rules(d.close)
        for k in args.cycle_rules:
            for tp in args.cycles:
                c = cycles_luck(f, d, held_from(all_rules[k], lo), lo, tp, args.cycle_luck)
                print(f"{k:<13}{tp:>6.0%}{c['per_day']:>12.2f}{c['win']:>8.0%}{c['total']:>8.0f}%"
                      f"{c['median']:>19.0f}%{c['p']:>7.3f}", flush=True)
    trials = [sharpe(s["weekly"]) for k, s in sums.items() if k != "segurar BTC"]
    best = max((k for k in sums if k != "segurar BTC"), key=lambda k: sums[k]["sharpe"])
    print(f"\nmelhor Sharpe: {best} ({sums[best]['sharpe']:.2f} ao ano); DSR entre {len(trials)} regras:"
          f" {deflated_sharpe(sums[best]['weekly'], trials):.3f}; segurar BTC: Sharpe {sums['segurar BTC']['sharpe']:.2f}")


if __name__ == "__main__":
    main()
