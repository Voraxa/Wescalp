"""Moedas mais voláteis que o BTC: o day trade fica mais perto de pagar a taxa? E a tendência diária numa cesta?

    python -m wescalp.altcoins       # depois de baixar os candles de cada moeda com wescalp.binance (ver o README)

Fixado antes de rodar (23/10/2026), sem ajuste depois. Moedas: ETH, SOL, XRP, DOGE e ADA contra USDT, as grandes mais
conhecidas, todas na Classe A da KuCoin (a mesma taxa do BTC, 0,1% por lado, conferida na API pública), com os candles
de 5 min da Binance de 01/2021 a 06/2026; o BTC entra para comparar. São as moedas que existem hoje: as que morreram no
caminho ficam de fora, o que favorece quem só compra.

1. Tamanho do movimento: o movimento médio de um candle de 5 min a 1 dia e o acerto de direção necessário para empatar
   com a ida e volta de 0,2%, como na tabela do BTC.
2. Previsão no curto prazo, de hora em hora: retorno de 1 h e de 24 h, volume da última hora sobre a média de 30 dias e
   "o candle de 1 h fechou abaixo da abertura do anterior"; alvo nas 1, 4 e 12 h seguintes, em todas as horas e só nas
   de volatilidade alta (σ EWMA dos retornos de 1 h, λ = 0,94, no terço de cima dos 365 dias anteriores). Lado e corte
   do decil saem do treino (2021 a 2023; no sinal de sim ou não, vale o sim); passa o que render, fora da amostra (2024 a
   06/2026), mais que 0,2% por operação com t ≥ 3,4 (o limiar de Bonferroni para os 120 testes das 5 moedas).
3. Tendência diária: a congelada (50/100/150) em cada moeda e numa cesta das 6 com 1/6 do capital em cada (sem contar a
   taxa de rebalancear entre elas), de 07/2021 a 06/2026, contra segurar; teste da sorte da combinação em cada moeda
   (2.000 sorteios). Passa a moeda com Sharpe maior, queda menor que segurar e p ≤ 0,05/6; a cesta, com Sharpe maior e
   queda menor que a cesta comprada.
"""

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import binance, brain, intraday, refine, regime, study

COINS = ("ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT")
REFERENCE = "BTCUSDT"
TIMEFRAMES = (("5 min", 5), ("15 min", 15), ("1 h", 60), ("4 h", 240), ("1 dia", 1440))
H, DAY = 3600, 86400
START, SPLIT, END = intraday.utc("2021-01-01"), intraday.utc("2024-01-01"), intraday.utc("2026-07-01")
TREND_START = intraday.utc("2021-07-01")
COST = 0.002
HORIZONS = (1, 4, 12)
MIN_T = 3.4
SIGNALS = ("retorno 1h", "retorno 24h", "volume relativo 1h", "fechou abaixo da abertura anterior")
BINARY = SIGNALS[3]
LAMBDA = 0.94
HIGH_VOL = 2 / 3
LUCK = 2000
MAX_P = 0.05 / 6


def load(symbol: str, data_dir: Path = brain.CANDLE_DIR):
    cs = brain.load_candles(binance.path(data_dir, symbol), symbol, "spot")
    ts = np.array([int(x.ts.timestamp()) for x in cs], dtype=np.int64)
    return (ts, *(np.array([getattr(x, k) for x in cs]) for k in ("open", "high", "low", "close", "volume")))


def moves(ts: np.ndarray, close: np.ndarray, minutes: int) -> tuple[float, float, float]:
    """(alta média, queda média, acerto de direção para empatar com a taxa) num candle de `minutes`."""
    k = ts // (minutes * 60)
    _, first = np.unique(k, return_index=True)
    c = close[np.r_[first[1:], len(ts)] - 1]
    r = c[1:] / c[:-1] - 1
    up, dn = r[r > 0].mean(), -r[r < 0].mean()
    return float(up), float(dn), float((COST + dn) / (up + dn))


def hourly_signals(decision: np.ndarray, ts, o, c, v) -> dict[str, np.ndarray]:
    price = lambda t: intraday.at(ts, c, t - 300)  # noqa: E731  fechamento do candle que termina em t
    p0 = price(decision)
    vol1 = intraday.window_sum(ts, v, decision - H, decision)
    vol30 = intraday.window_sum(ts, v, decision - 30 * 24 * H, decision) / (30 * 24)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = {"retorno 1h": np.log(p0 / price(decision - H)), "retorno 24h": np.log(p0 / price(decision - 24 * H)),
               "volume relativo 1h": np.log(vol1 / vol30),
               # a hora que acabou fechou abaixo da abertura da hora anterior a ela
               BINARY: (p0 < intraday.at(ts, o, decision - 2 * H)).astype(float)}
    return {k: np.where(np.isfinite(x), x, np.nan) for k, x in out.items()}


def high_vol(decision: np.ndarray, ts, c, lam: float = LAMBDA) -> np.ndarray:
    """Volatilidade da hora (EWMA dos retornos de 1 h) no terço de cima dos 365 dias anteriores, só com o passado."""
    p = intraday.at(ts, c, decision - 300)
    r = np.r_[np.nan, np.log(p[1:] / p[:-1])]
    sigma = np.full(len(decision), np.nan)
    var = None
    for t, x in enumerate(r):
        if np.isfinite(x):
            var = x * x if var is None else lam * var + (1 - lam) * x * x
        sigma[t] = math.sqrt(var) if var is not None else np.nan
    days = decision // DAY
    out = np.zeros(len(decision), dtype=bool)
    for d in np.unique(days):
        past = sigma[(days >= d - 365) & (days < d)]
        past = past[np.isfinite(past)]
        if len(past) >= 300 * 24:  # ao menos ~300 dias de história
            today = days == d
            out[today] = sigma[today] >= np.quantile(past, HIGH_VOL)
    return out


def binary_test(s_out: np.ndarray, y_out: np.ndarray) -> dict:
    ok = np.isfinite(s_out) & np.isfinite(y_out)
    y = y_out[ok & (s_out > 0.5)]
    r = intraday.summary(y, "sim", math.nan, y_out[ok].mean())
    return r | {"passes": bool(r["n"] > 2 and r["out"] > COST and r["t"] >= MIN_T)}


def information(symbol_data, decision: np.ndarray) -> list[dict]:
    """Os testes da parte 2 para uma moeda: uma linha por sinal, horizonte e regime."""
    ts, o, h, l, c, v = symbol_data
    sig = hourly_signals(decision, ts, o, c, v)
    hv = high_vol(decision, ts, c)
    rows = []
    for hours in HORIZONS:
        y = intraday.forward({"ts": ts, "close": c}, decision, hours)
        step = (decision // H) % hours == 0
        for regime_name, mask in (("todas as horas", np.ones(len(decision), dtype=bool)), ("volatilidade alta", hv)):
            i_in = step & mask & (decision >= START) & (decision < SPLIT)
            i_out = step & mask & (decision >= SPLIT)
            for k in SIGNALS:
                if k == BINARY:
                    r = binary_test(sig[k][i_out], y[i_out])
                else:
                    r = intraday.tail_test(sig[k][i_in], y[i_in], sig[k][i_out], y[i_out])
                    r["passes"] = bool(r["n"] > 2 and r["out"] > COST and r["t"] >= MIN_T)
                rows.append({"signal": k, "hours": hours, "regime": regime_name} | r)
    return rows


def trend(d: regime.Daily, lo: int, hi: int, fee: float = regime.FEE) -> tuple[np.ndarray, np.ndarray]:
    """(retornos diários da congelada, retornos de segurar) nos dias lo..hi-1."""
    rets = d.close[lo:hi] / d.close[lo - 1:hi - 1] - 1
    held = refine.position(d.close, refine.BASE)[lo - 1:hi - 1]
    return regime.returns_from_held(held, rets, fee), regime.returns_from_held(np.ones(hi - lo), rets, fee)


def scores(r: np.ndarray) -> dict:
    return {"cagr": study.cagr(r), "dd": study.max_dd(r), "sharpe": float(r.mean() / r.std() * math.sqrt(365))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=brain.CANDLE_DIR)
    p.add_argument("--luck", type=int, default=LUCK)
    args = p.parse_args()
    data = {s: load(s, args.data) for s in (REFERENCE, *COINS)}

    print("1. movimento médio de um candle e acerto de direção necessário para empatar com a ida e volta de 0,2%")
    print(f"{'moeda':<10}" + "".join(f"{name:>22}" for name, _ in TIMEFRAMES))
    for s, (ts, o, h, l, c, v) in data.items():
        keep = (ts >= START) & (ts < END)
        cells = []
        for _, minutes in TIMEFRAMES:
            up, dn, need = moves(ts[keep], c[keep], minutes)
            cells.append(f"{(up + dn) / 2:.2%} / {'impossível' if need > 1 else f'{need:.0%}'}")
        print(f"{s[:-4]:<10}" + "".join(f"{x:>22}" for x in cells))

    print("\n2. previsão no curto prazo (decil escolhido no treino, 2021-2023; medido fora, 2024-06/2026)")
    decision = np.arange(START, END - 12 * H, H, dtype=np.int64)
    passed = []
    print(f"{'moeda':<6}{'sinal':<38}{'h':>4}{'regime':>19}{'lado':>7}{'fora':>9}{'base':>8}{'acerto':>8}{'n':>7}{'t':>7}")
    for s in (REFERENCE, *COINS):
        rows = information(data[s], decision)
        best = sorted(rows, key=lambda r: -(r["out"] if math.isfinite(r["out"]) else -9))[:4]
        for r in best:
            print(f"{s[:-4]:<6}{r['signal']:<38}{r['hours']:>3}h{r['regime']:>19}{r['side']:>7}{r['out']:>9.3%}"
                  f"{r['base']:>8.3%}{r['hit']:>8.0%}{r['n']:>7}{r['t']:>7.1f}")
        passed += [f"{s[:-4]} {r['signal']} {r['hours']} h ({r['regime']})" for r in rows if r["passes"] and s != REFERENCE]
    print(f"passam (melhores 4 de cada moeda acima; régua: mais que 0,2% com t ≥ {MIN_T}): {', '.join(passed) or 'nenhum'}")

    print("\n3. tendência diária (congelada 50/100/150), 07/2021 a 06/2026, 0,1% por lado")
    print(f"{'moeda':<10}{'congelada: ao ano / queda / Sharpe':>36}{'segurar: ao ano / queda / Sharpe':>36}{'p sorte':>9}  resultado")
    strat, hold = [], []
    for s, (ts, o, h, l, c, v) in data.items():
        d = regime.daily_bars(ts, o, h, l, c)
        lo, hi = int(np.searchsorted(d.day, TREND_START // DAY)), int(np.searchsorted(d.day, END // DAY))
        r, b = trend(d, lo, hi)
        strat.append(r)
        hold.append(b)
        a, z = scores(r), scores(b)
        pv = refine.luck_p(d.close[:hi], lo, refine.BASE, args.luck) if args.luck else math.nan
        ok = a["sharpe"] > z["sharpe"] and a["dd"] < z["dd"] and pv <= MAX_P
        print(f"{s[:-4]:<10}{a['cagr']:>14.1%} / {a['dd']:>4.0%} / {a['sharpe']:>5.2f}{z['cagr']:>17.1%} / {z['dd']:>4.0%} /"
              f" {z['sharpe']:>5.2f}{pv:>9.3f}  {'PASSA' if ok else 'não passa'}", flush=True)
    n = min(len(x) for x in strat)
    basket, basket_hold = np.mean([x[:n] for x in strat], axis=0), np.mean([x[:n] for x in hold], axis=0)
    a, z = scores(basket), scores(basket_hold)
    ok = a["sharpe"] > z["sharpe"] and a["dd"] < z["dd"]
    print(f"{'cesta das 6':<10}{a['cagr']:>14.1%} / {a['dd']:>4.0%} / {a['sharpe']:>5.2f}{z['cagr']:>17.1%} / {z['dd']:>4.0%} /"
          f" {z['sharpe']:>5.2f}{'':>9}  {'PASSA' if ok else 'não passa'}")


if __name__ == "__main__":
    main()
