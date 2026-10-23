"""Efeitos de horário publicados, testados depois da publicação: a data do artigo vira o corte entre aprender e testar.

    python -m wescalp.sessions --file data/candles/BINANCE-BTCUSDT-5m.csv.gz

Fixado antes de rodar (23/10/2026), sem ajuste depois. Três regras de day trade tiradas da literatura, cada uma com no
máximo uma ida e volta por dia e o capital inteiro em cada operação:
- fora do pregão americano (Bespoke, citado pela Bloomberg em 06/2022: a maior parte da alta do BTC na pandemia veio
  com a bolsa fechada): compra no fechamento de Nova York (16h de lá) de cada dia de pregão e vende na abertura do
  pregão seguinte (9h30), ficando comprado no fim de semana. Dias de pregão: os que têm fechamento do Nasdaq no FRED;
- 21h às 23h UTC (Quantpedia, "Are There Seasonal Intraday or Overnight Anomalies in Bitcoin?"): compra às 21h e vende
  às 23h UTC, todo dia;
- momento intradiário (Shen, Urquhart e Wang, "Bitcoin intraday time-series momentum", Financial Review, 2022): se a
  primeira meia hora do dia UTC subiu, compra às 23h30 UTC e vende à meia-noite; senão, fica fora. O artigo define o
  dia pelo volume; aqui é o dia UTC.
Teste: 01/07/2022 a 30/06/2026, depois das publicações; o trecho de 11/10/2018 a 30/06/2022 aparece só para mostrar se o
efeito existia antes. Preço: o fechamento do candle de 5 min que termina na hora da ordem; 0,1% por lado, e 0,02% e 0%
para ver o tamanho bruto. Teste da sorte: 2.000 sorteios com cada operação deslocada para uma hora cheia sorteada do
mesmo dia, com a mesma duração; p = fração com soma dos log-retornos igual ou maior. Passa se, no teste, com 0,1% por
lado, tiver lucro, Sharpe maior que o de segurar o BTC e p ≤ 0,0167 (0,05 dividido pelas 3 regras).
"""

import argparse
import math
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from . import factors, intraday, regime, study

H, DAY = 3600, 86400
NEW_YORK = ZoneInfo("America/New_York")
RULES = ("fora do pregão americano", "21h às 23h UTC", "momento intradiário")
PRE_START, TEST_START, END = intraday.utc("2018-10-11"), intraday.utc("2022-07-01"), intraday.utc("2026-07-01")
FEES = (0.001, 0.0002, 0.0)
FEE = FEES[0]
LUCK = 2000
MAX_P = 0.05 / 3


def ny(day: date, hour: time) -> int:
    return int(datetime.combine(day, hour, NEW_YORK).timestamp())


def trades(trading_days: list[date], ts: np.ndarray, close: np.ndarray, start: int, end: int) -> dict[str, np.ndarray]:
    """(entrada, saída) em segundos UTC das operações de cada regra que começam e terminam em [start, end)."""
    price = lambda t: intraday.at(ts, close, np.asarray(t) - 300)  # noqa: E731  o candle que fecha em t
    us = np.array([(ny(a, time(16)), ny(b, time(9, 30))) for a, b in zip(trading_days, trading_days[1:])])
    days = np.arange(start // DAY, end // DAY)
    late = np.column_stack([days * DAY + 21 * H, days * DAY + 23 * H])
    up = price(days * DAY + 1800) > price(days * DAY)  # a primeira meia hora subiu
    momentum = np.column_stack([days * DAY + 23 * H + 1800, (days + 1) * DAY])[up]
    out = {}
    for name, t in zip(RULES, (us, late, momentum)):
        t = t[(t[:, 0] >= start) & (t[:, 1] < end)]
        ok = np.isfinite(price(t[:, 0])) & np.isfinite(price(t[:, 1]))
        out[name] = t[ok]
    return out


def log_returns(ts: np.ndarray, close: np.ndarray, t: np.ndarray, fee: float) -> np.ndarray:
    price = lambda x: intraday.at(ts, close, x - 300)  # noqa: E731
    return np.log(price(t[:, 1]) / price(t[:, 0])) + 2 * math.log(1 - fee)


def summary(t: np.ndarray, r: np.ndarray, start: int, end: int) -> dict:
    """Retorno total, ao ano e Sharpe diário (cada operação conta no dia em que fecha)."""
    days = np.arange(start // DAY, end // DAY)
    daily = np.zeros(len(days))
    np.add.at(daily, (t[:, 1] - 1) // DAY - days[0], r)  # a saída à meia-noite conta no dia que termina
    simple = np.expm1(daily)
    years = len(days) / 365.25
    total = math.exp(r.sum())
    return {"trades": len(t), "total": total - 1, "cagr": total ** (1 / years) - 1, "dd": study.max_dd(simple),
            "sharpe": float(simple.mean() / simple.std() * math.sqrt(365)) if simple.std() > 0 else 0.0,
            "hit": float((r > 0).mean()) if len(r) else math.nan}


def luck_p(ts: np.ndarray, close: np.ndarray, t: np.ndarray, n: int = LUCK, seed: int = 0) -> float:
    """Fração dos sorteios (cada operação numa hora cheia sorteada do mesmo dia, mesma duração) com soma igual ou maior."""
    real = log_returns(ts, close, t, 0.0).sum()
    rng = np.random.default_rng(seed)
    dur = t[:, 1] - t[:, 0]
    day = t[:, 0] // DAY * DAY
    hits, done = 0, 0
    while done < n:
        k = min(200, n - done)
        start = day[None, :] + rng.integers(0, 24, (k, len(t))) * H
        a = intraday.at(ts, close, (start - 300).ravel()).reshape(k, -1)
        b = intraday.at(ts, close, (start + dur[None, :] - 300).ravel()).reshape(k, -1)
        hits += int((np.nansum(np.log(b / a), axis=1) >= real).sum())
        done += k
    return hits / n


def hold(d: regime.Daily, start: int, end: int) -> dict:
    lo, hi = np.searchsorted(d.day, start // DAY), np.searchsorted(d.day, end // DAY)
    r = d.close[lo:hi] / d.close[lo - 1:hi - 1] - 1
    return {"total": float(np.prod(1 + r) - 1), "cagr": study.cagr(r), "dd": study.max_dd(r),
            "sharpe": float(r.mean() / r.std() * math.sqrt(365))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", type=Path, required=True)
    p.add_argument("--factors", type=Path, default=factors.DATA_DIR)
    p.add_argument("--luck", type=int, default=LUCK)
    args = p.parse_args()
    ts, o, h, l, c = study.load(args.file)
    nd, _ = factors.load("NASDAQCOM", args.factors)
    trading_days = [factors.EPOCH + timedelta(days=int(x)) for x in nd if PRE_START - 7 * DAY <= x * DAY < END]
    d = regime.daily_bars(ts, o, h, l, c)
    for label, start, end in (("antes da publicação (só para ver se existia)", PRE_START, TEST_START),
                              ("TESTE, depois da publicação", TEST_START, END)):
        fmt = lambda x: datetime.fromtimestamp(x, timezone.utc).strftime("%d/%m/%Y")  # noqa: E731
        print(f"\n{label}: {fmt(start)} a {fmt(end - DAY)}")
        print(f"{'regra':<28}{'operações':>10}{'acerto':>8}" + "".join(f"{f'ao ano {f:.2%}':>17}" for f in FEES)
              + f"{'queda':>8}{'Sharpe':>8}{'p sorte':>9}")
        b = hold(d, start, end)
        for name, t in trades(trading_days, ts, c, start, end).items():
            s = {f: summary(t, log_returns(ts, c, t, f), start, end) for f in FEES}
            luck = luck_p(ts, c, t, args.luck) if start == TEST_START and args.luck else math.nan
            print(f"{name:<28}{s[FEE]['trades']:>10}{s[FEE]['hit']:>8.0%}"
                  + "".join(f"{s[f]['cagr']:>17.1%}" for f in FEES)
                  + f"{s[FEE]['dd']:>8.0%}{s[FEE]['sharpe']:>8.2f}" + (f"{luck:>9.3f}" if math.isfinite(luck) else ""))
            if start == TEST_START:
                fails = [k for k, bad in (("lucro", s[FEE]["total"] <= 0), ("Sharpe", s[FEE]["sharpe"] <= b["sharpe"]),
                                          ("sorte", not luck <= MAX_P)) if bad]
                print(f"{'':<28}{'PASSA' if not fails else 'não passa: ' + ', '.join(fails)}")
        print(f"{'segurar BTC':<28}{'':>10}{'':>8}{b['cagr']:>17.1%}{'':>34}{b['dd']:>8.0%}{b['sharpe']:>8.2f}")


if __name__ == "__main__":
    main()
