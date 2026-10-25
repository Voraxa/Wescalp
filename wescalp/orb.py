"""Rompimento da faixa de abertura (ORB) no BTC, com o início do pregão americano no lugar da abertura.

    python -m wescalp.orb --file data/candles/BINANCE-BTCUSDT-5m.csv.gz

É o setup principal do day-trader-skill (Jakeschincariol), feito para ações dos EUA. O BTC negocia 24 h, então a
"abertura" aqui é a da bolsa de Nova York. Fixado antes de rodar (25/10/2026), sem ajuste depois:
- faixa de abertura: máxima e mínima dos candles de 5 min do BTC nos primeiros 15 ou 30 min depois das 9h30 de Nova
  York (com o horário de verão de lá), em cada dia de pregão (os dias com fechamento do Nasdaq no FRED); dia em que
  falta algum desses candles fica de fora;
- entrada: só compra, no spot, com o capital inteiro, no fechamento do primeiro candle de 5 min que fechar acima da
  máxima da faixa, até o candle que fecha às 12h de Nova York; uma operação por dia;
- saída: stop a mercado no meio da faixa (como no exemplo do autor), R = entrada - stop; alvo com ordem limitada 1R ou
  2R acima da entrada; se nada bater, venda no fechamento do candle das 16h de Nova York. No mesmo candle, o stop vem
  antes do alvo; o stop sai no nível ou na abertura do candle, o que for pior, e o alvo no nível ou na abertura, o que
  for melhor;
- 4 variantes (faixa de 15 ou 30 min × alvo de 1R ou 2R); 0,1% por lado, e 0,02% e 0% para ver o tamanho bruto; de
  11/10/2018 a 30/06/2026; régua: segurar o BTC no mesmo período;
- teste da sorte: nos mesmos dias em que a regra operou, a entrada vai para o fechamento de um candle sorteado da mesma
  janela (do fim da faixa ao candle das 12h), com o mesmo R em preço daquele dia, o mesmo alvo em R e a mesma saída;
  2.000 sorteios; p = fração com soma dos log-retornos igual ou maior. A hipótese nula é que entrar no rompimento não
  vale mais que entrar a qualquer hora da mesma janela, com o mesmo risco;
- passa a variante com lucro, Sharpe maior que segurar e p ≤ 0,0125 (0,05 dividido pelas 4 variantes).

`--holdout` (fixado junto, antes de qualquer rodada): só as variantes que passarem (`--variants`), uma vez, nas 16
semanas guardadas (04/07 a 21/10/2026, candles da KuCoin, `--file data/candles/BTC-USDT-5m.csv.gz`). Duas perguntas: o
efeito se repete (sem taxa, total acima de zero e p ≤ 0,05 no teste da sorte)? E dá lucro com a taxa do spot (0,1% por
lado, total acima de zero e Sharpe acima do de segurar no mesmo período)? Se nenhuma passar, a prova única não roda.

Nota depois da rodada (25/10/2026, sem mudar o método): a hipótese nula ficou viesada a favor dos sorteios. Os dias
são os que tiveram rompimento, e a entrada sorteada pode cair antes dele, num dia que já se sabe que vai fechar acima da
faixa; por isso todos os sorteios renderam mais e o p deu 1,000, o que não mede o setup. O veredito não depende disso:
sem taxa a regra rende perto de zero (de -1,4% a +2,1% ao ano) e, com a do spot, perde de 22% a 28% ao ano.
"""

import argparse
import math
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np

from . import factors, intraday, regime, sessions, spot, study

DAY = 86400
RANGES = (15, 30)
TARGETS = (1, 2)
VARIANTS = {f"faixa {w} min, alvo {k}R": (w, k) for w in RANGES for k in TARGETS}
START, END = intraday.utc("2018-10-11"), intraday.utc("2026-07-01")
FEES = sessions.FEES
FEE = FEES[0]
LUCK = 2000
MAX_P = 0.05 / len(VARIANTS)


@dataclass(frozen=True)
class Window:
    """Um dia de pregão: a faixa de abertura e os candles em que dá para entrar e sair."""
    first: int  # primeiro candle que pode dar entrada (o logo depois da faixa)
    last_entry: int  # último candle que pode dar entrada (o que fecha às 12h de Nova York)
    last: int  # candle da venda no fechamento (o que fecha às 16h)
    high: float  # máxima da faixa
    low: float  # mínima da faixa


@dataclass(frozen=True)
class Trade:
    entry: int  # candle em cujo fechamento comprou
    exit: int
    entry_px: float
    exit_px: float
    outcome: str  # alvo | stop | fechamento
    risk: float  # R em preço: entrada - stop


def windows(trading_days: list[date], ts: np.ndarray, h: np.ndarray, l: np.ndarray, minutes: int, start: int,
            end: int) -> list[Window]:
    """A faixa dos primeiros `minutes` depois das 9h30 de Nova York em cada dia de pregão com abertura em [start, end)."""
    out = []
    n = minutes // 5
    for d in trading_days:
        t0 = sessions.ny(d, time(9, 30))
        if not start <= t0 < end:
            continue
        i0 = int(np.searchsorted(ts, t0))
        if i0 + n > len(ts) or not np.array_equal(ts[i0:i0 + n], t0 + 300 * np.arange(n)):
            continue  # falta candle na faixa
        last_entry = int(np.searchsorted(ts, sessions.ny(d, time(12)) - 300, side="right")) - 1
        last = int(np.searchsorted(ts, sessions.ny(d, time(16)) - 300, side="right")) - 1
        if last_entry < i0 + n:
            continue
        out.append(Window(i0 + n, last_entry, last, float(h[i0:i0 + n].max()), float(l[i0:i0 + n].min())))
    return out


def exit_of(o, c, up: spot.Crossing, dn: spot.Crossing, i: int, stop: float, target: float, last: int):
    """Compra no fechamento do candle i: (candle da saída, preço, motivo), com o stop antes do alvo no mesmo candle."""
    ks, kt = dn.first(i + 1, stop), up.first(i + 1, target)
    ks = ks if ks is not None and ks <= last else None
    kt = kt if kt is not None and kt <= last else None
    if ks is not None and (kt is None or ks <= kt):
        return ks, min(stop, o[ks]), "stop"
    if kt is not None:
        return kt, max(target, o[kt]), "alvo"
    return last, c[last], "fechamento"


def orb_trades(wins: list[Window], o, c, up: spot.Crossing, dn: spot.Crossing, k: int) -> tuple[list[Trade], list[Window]]:
    """As operações da regra e a janela de cada uma (para o teste da sorte)."""
    trades, used = [], []
    for w in wins:
        above = np.flatnonzero(c[w.first:w.last_entry + 1] > w.high)
        if not len(above):
            continue
        i = w.first + int(above[0])
        stop = (w.high + w.low) / 2
        risk = c[i] - stop
        e, px, why = exit_of(o, c, up, dn, i, stop, c[i] + k * risk, w.last)
        trades.append(Trade(i, e, float(c[i]), float(px), why, float(risk)))
        used.append(w)
    return trades, used


def as_times(ts: np.ndarray, trades: list[Trade]) -> np.ndarray:
    """(entrada, saída) em segundos UTC, no fechamento dos candles, no formato de wescalp.sessions."""
    return np.array([(ts[t.entry] + 300, ts[t.exit] + 300) for t in trades], dtype=np.int64).reshape(-1, 2)


def log_returns(trades: list[Trade], fee: float) -> np.ndarray:
    return np.array([math.log(t.exit_px / t.entry_px) + 2 * math.log(1 - fee) for t in trades])


def luck_p(trades: list[Trade], wins: list[Window], o, c, up, dn, k: int, n: int = LUCK, seed: int = 0) -> float:
    """Fração dos sorteios (entrada num candle sorteado da mesma janela, mesmo R e alvo) com soma igual ou maior."""
    real = float(log_returns(trades, 0.0).sum())
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n):
        total = 0.0
        for t, w in zip(trades, wins):
            i = int(rng.integers(w.first, w.last_entry + 1))
            _, px, _ = exit_of(o, c, up, dn, i, c[i] - t.risk, c[i] + k * t.risk, w.last)
            total += math.log(px / c[i])
        hits += total >= real
    return hits / n


def evaluate(ts, o, h, l, c, trading_days, start: int, end: int, names, luck: int) -> dict[str, dict]:
    up, dn = spot.Crossing(h, True), spot.Crossing(l, False)
    out = {}
    for name in names:
        minutes, k = VARIANTS[name]
        trades, used = orb_trades(windows(trading_days, ts, h, l, minutes, start, end), o, c, up, dn, k)
        t = as_times(ts, trades)
        s = {f: sessions.summary(t, log_returns(trades, f), start, end) for f in FEES}
        s["p"] = luck_p(trades, used, o, c, up, dn, k, luck) if luck and trades else math.nan
        s["outcomes"] = {x: sum(tr.outcome == x for tr in trades) for x in ("alvo", "stop", "fechamento")}
        out[name] = s
    return out


def verdict(s: dict, hold: dict) -> list[str]:
    return [k for k, bad in (("lucro", s[FEE]["total"] <= 0), ("Sharpe", s[FEE]["sharpe"] <= hold["sharpe"]),
                             ("sorte", not s["p"] <= MAX_P)) if bad]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", type=Path, required=True)
    p.add_argument("--factors", type=Path, default=factors.DATA_DIR)
    p.add_argument("--luck", type=int, default=LUCK)
    p.add_argument("--holdout", action="store_true", help="a prova única nas 16 semanas guardadas (ver o cabeçalho)")
    p.add_argument("--variants", nargs="*", default=list(VARIANTS), help="só com --holdout: as variantes que passaram")
    args = p.parse_args()
    ts, o, h, l, c = study.load(args.file)
    nd, _ = factors.load("NASDAQCOM", args.factors)
    trading_days = [factors.EPOCH + timedelta(days=int(x)) for x in nd]
    d = regime.daily_bars(ts, o, h, l, c)
    start, end = sessions.HOLDOUT if args.holdout else (START, END)
    names = args.variants if args.holdout else list(VARIANTS)
    hold = sessions.hold(d, start, end)
    res = evaluate(ts, o, h, l, c, trading_days, start, end, names, args.luck)
    fmt = lambda x: datetime.fromtimestamp(x, timezone.utc).strftime("%d/%m/%Y")  # noqa: E731
    print(f"ORB na abertura de Nova York, {fmt(start)} a {fmt(end - DAY)}, capital inteiro por operação")
    print(f"{'variante':<24}{'operações':>10}{'acerto':>8}" + "".join(f"{f'ao ano {f:.2%}':>16}" for f in FEES)
          + f"{'queda':>8}{'Sharpe':>8}{'p sorte':>9}   alvo/stop/fechamento")
    for name, s in res.items():
        oc = s["outcomes"]
        print(f"{name:<24}{s[FEE]['trades']:>10}{s[FEE]['hit']:>8.0%}" + "".join(f"{s[f]['cagr']:>16.1%}" for f in FEES)
              + f"{s[FEE]['dd']:>8.0%}{s[FEE]['sharpe']:>8.2f}{s['p']:>9.3f}   {oc['alvo']}/{oc['stop']}/{oc['fechamento']}")
    print(f"{'segurar BTC':<24}{'':>18}{hold['cagr']:>16.1%}{'':>32}{hold['dd']:>8.0%}{hold['sharpe']:>8.2f}")
    print()
    for name, s in res.items():
        if args.holdout:
            repeats = s[0.0]["total"] > 0 and s["p"] <= 0.05
            pays = s[FEE]["total"] > 0 and s[FEE]["sharpe"] > hold["sharpe"]
            print(f"{name}: o efeito se repete? {'SIM' if repeats else 'não'}; dá lucro com a taxa do spot?"
                  f" {'SIM' if pays else 'não'}")
        else:
            fails = verdict(s, hold)
            print(f"{name}: {'PASSA' if not fails else 'não passa: ' + ', '.join(fails)}")


if __name__ == "__main__":
    main()
