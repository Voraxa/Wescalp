"""O vale no candle diário com alvo de lucro: compra quando o dia fecha abaixo da abertura do dia anterior (ou de dois
dias antes) e vende quando o preço sobe o alvo sobre a compra.

    python -m wescalp.dailydip --file data/candles/BINANCE-BTCUSDT-5m.csv.gz

Fixado antes de rodar (23/10/2026), sem ajuste depois:
- sinal no fechamento diário (00h UTC): o fechamento do dia abaixo da abertura do dia anterior, ou de dois dias antes;
- compra no fechamento do dia do sinal, com o capital inteiro, uma operação por vez;
- venda com ordem limitada no alvo de 2%, 3% ou 5% sobre a compra (os alvos escolhidos em 23/10), conferida nos
  candles de 5 min;
- duas saídas: só o alvo (fica comprado até chegar lá) ou o alvo com stop a mercado 3 alvos abaixo da compra (como em
  wescalp.spot); no mesmo candle de 5 min, o stop vem antes do alvo;
- 0,1% por lado, de 11/10/2018 a 30/06/2026; o que estiver aberto no fim é vendido no último fechamento;
- teste da sorte: 1.000 sorteios com a mesma regra e o sinal em dias ao acaso, na mesma frequência do sinal real;
  p = fração com retorno total igual ou maior;
- passa a variante com lucro, Sharpe maior que o de segurar o BTC e p ≤ 0,05 / 12 (as 12 variantes).
"""

import argparse
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import intraday, refine, regime, spot, study

TARGETS = (0.02, 0.03, 0.05)
BACKS = {"anterior": 1, "2 antes": 2}
EXITS = {"só alvo": 0, "alvo e stop": 3}  # stop em quantos alvos abaixo da compra (0 = sem stop)
FEE = 0.001
LUCK = 1000
START, END = intraday.utc("2018-10-11"), intraday.utc("2026-07-01")
MAX_P = 0.05 / 12


@dataclass(frozen=True)
class Trade:
    entry: int  # candle de 5 min da compra (o último do dia do sinal)
    exit: int
    entry_px: float
    exit_px: float
    outcome: str  # alvo | stop | aberto

    def ret(self, fee: float = FEE) -> float:
        return self.exit_px / self.entry_px * (1 - fee) ** 2 - 1


def signal(d: regime.Daily, back: int) -> np.ndarray:
    """No fechamento de cada dia: fechou abaixo da abertura de `back` dias antes?"""
    s = np.zeros(len(d.close), dtype=bool)
    s[back:] = d.close[back:] < d.open[:-back]
    return s


def simulate(o, c, up: spot.Crossing, dn: spot.Crossing, day_end: np.ndarray, sig: np.ndarray, target: float,
             stop_mult: int) -> list[Trade]:
    """Uma operação por vez. `day_end[k]`: último candle de 5 min do dia k; `sig[k]`: sinal no fechamento do dia k."""
    trades: list[Trade] = []
    free_from, last = 0, len(c) - 1
    for k, i in enumerate(day_end):
        if i < free_from or not sig[k] or i >= last:
            continue
        px = c[i]
        tgt, stop = px * (1 + target), px * (1 - stop_mult * target)
        kt = up.first(i + 1, tgt)
        ks = dn.first(i + 1, stop) if stop_mult else None
        if ks is not None and (kt is None or ks <= kt):
            trades.append(Trade(i, ks, px, min(stop, o[ks]), "stop"))
        elif kt is not None:
            trades.append(Trade(i, kt, px, max(tgt, o[kt]), "alvo"))
        else:
            trades.append(Trade(i, last, px, c[last], "aberto"))
        free_from = trades[-1].exit + 1
    return trades


def daily_equity(c: np.ndarray, day_end: np.ndarray, trades: list[Trade], fee: float = FEE) -> tuple[np.ndarray, float]:
    """(patrimônio no fechamento de cada dia, fração dos dias com operação aberta), marcando a mercado."""
    eq = np.empty(len(day_end))
    wealth, j, open_days = 1.0, 0, 0
    for k, b in enumerate(day_end):
        while j < len(trades) and trades[j].exit <= b:
            wealth *= 1 + trades[j].ret(fee)
            j += 1
        t = trades[j] if j < len(trades) and trades[j].entry <= b else None
        if t is not None and t.entry < b:
            open_days += 1
            eq[k] = wealth * c[b] / t.entry_px * (1 - fee) ** 2
        else:
            eq[k] = wealth
    return eq, open_days / len(day_end)


def summary(c, day_end, trades, fee: float = FEE) -> dict:
    eq, in_market = daily_equity(c, day_end, trades, fee)
    r = np.diff(np.concatenate([[1.0], eq])) / np.concatenate([[1.0], eq[:-1]])
    years = len(day_end) / 365.25
    rets = np.array([t.ret(fee) for t in trades])
    days = np.array([(t.exit - t.entry) * 300 / 86400 for t in trades])
    return {"trades": len(trades), "per_year": len(trades) / years, "hit": float((rets > 0).mean()) if len(rets) else math.nan,
            "days": float(np.median(days)) if len(days) else math.nan, "max_days": float(days.max()) if len(days) else 0.0,
            "in_market": in_market, "total": float(eq[-1] - 1), "cagr": float(eq[-1] ** (1 / years) - 1),
            "dd": study.max_dd(r), "sharpe": float(r.mean() / r.std() * math.sqrt(365)) if r.std() > 0 else 0.0,
            "outcomes": {k: sum(t.outcome == k for t in trades) for k in ("alvo", "stop", "aberto")}}


def luck_p(o, c, up, dn, day_end, sig, target, stop_mult, n: int = LUCK, seed: int = 0) -> float:
    real = math.prod(1 + t.ret() for t in simulate(o, c, up, dn, day_end, sig, target, stop_mult))
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n):
        fake = rng.random(len(sig)) < sig.mean()
        hits += math.prod(1 + t.ret() for t in simulate(o, c, up, dn, day_end, fake, target, stop_mult)) >= real
    return hits / n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", type=Path, required=True)
    p.add_argument("--luck", type=int, default=LUCK)
    args = p.parse_args()
    ts, o, h, l, c = study.load(args.file)
    d = regime.daily_bars(ts, o, h, l, c)
    up, dn = spot.Crossing(h, True), spot.Crossing(l, False)
    lo, hi = int(np.searchsorted(d.day, START // 86400)), int(np.searchsorted(d.day, END // 86400))
    day_end = (np.searchsorted(ts, (d.day + 1) * 86400) - 1)  # último candle de 5 min de cada dia
    # A primeira decisão é no fechamento da véspera do primeiro dia do período.
    sl = slice(lo - 1, hi)

    rets = d.close[lo:hi] / d.close[lo - 1:hi - 1] - 1
    bench = {}
    for name, pos in (("segurar BTC", np.ones(len(d.close))), ("congelada 50/100/150", refine.position(d.close, refine.BASE))):
        r = regime.returns_from_held(pos[lo - 1:hi - 1], rets)
        bench[name] = {"cagr": study.cagr(r), "dd": study.max_dd(r), "sharpe": float(r.mean() / r.std() * math.sqrt(365))}

    fmt = lambda k: datetime.fromtimestamp(int(d.day[k]) * 86400, timezone.utc).strftime("%d/%m/%Y")  # noqa: E731
    print(f"vale no candle diário com alvo, de {fmt(lo)} a {fmt(hi - 1)}, capital inteiro por operação, {FEE:.1%} por lado")
    print(f"{'sinal: fecha abaixo da abertura de':<36}{'alvo':>5}{'operações/ano':>15}{'acerto':>8}{'dias (mediana/máx)':>20}"
          f"{'comprado':>10}{'ao ano':>9}{'queda':>7}{'Sharpe':>8}{'p sorte':>9}  resultado")
    hold = bench["segurar BTC"]
    for sname, back in BACKS.items():
        sig = signal(d, back)[sl]
        for ename, mult in EXITS.items():
            for tp in TARGETS:
                trades = simulate(o, c, up, dn, day_end[sl], sig, tp, mult)
                s = summary(c, day_end[sl][1:], [t for t in trades], FEE)
                pv = luck_p(o, c, up, dn, day_end[sl], sig, tp, mult, args.luck) if args.luck else math.nan
                fails = [k for k, bad in (("lucro", s["total"] <= 0), ("Sharpe", s["sharpe"] <= hold["sharpe"]),
                                          ("sorte", not pv <= MAX_P)) if bad]
                label = f"o dia {sname}, {ename}"
                print(f"{label:<36}{tp:>5.0%}{s['per_year']:>15.1f}{s['hit']:>8.0%}"
                      f"{s['days']:>10.1f} / {s['max_days']:>6.0f}{s['in_market']:>10.0%}{s['cagr']:>9.1%}{s['dd']:>7.0%}"
                      f"{s['sharpe']:>8.2f}{pv:>9.3f}  {'PASSA' if not fails else 'não passa: ' + ', '.join(fails)}",
                      flush=True)
    for name, b in bench.items():
        print(f"{name:<36}{'':>5}{'':>15}{'':>8}{'':>20}{'':>10}{b['cagr']:>9.1%}{b['dd']:>7.0%}{b['sharpe']:>8.2f}")


if __name__ == "__main__":
    main()
