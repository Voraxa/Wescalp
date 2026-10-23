"""Cérebro de entrada no spot BTC-USDT: compra boa, DCA só com novo sinal, alvo fixo e stop de entrada.

    python -m wescalp.spot fetch     # 100 semanas de candles de 5 min do BTC-USDT em data/candles/
    python -m wescalp.spot run       # walk-forward do juiz e relatório para alvos de 2%, 3% e 5%

Tudo simulado, só compra e venda no spot. O ciclo, com alvo `tp` (fração):
1. Compra: no fechamento de um candle de 5 min em que algum gatilho de wescalp.brain dispara e o juiz dá nota
   acima do corte, compra 1/3 do capital na abertura do candle seguinte.
2. DCA com inteligência: com a posição aberta, compra mais 1/3 (até 2 vezes) só quando há um novo sinal (gatilho
   e nota acima do corte) e o fechamento está pelo menos `tp` abaixo da última compra. Queda sozinha não compra.
3. Venda: tudo no alvo, `tp` acima do preço médio (ordem limitada); no stop definido na entrada, STOP_MULT·`tp`
   abaixo dela (abaixo dos dois aportes possíveis); ou em MAX_DAYS dias.
No mesmo candle, stop antes do alvo (o conservador); abriu além de um dos dois, sai na abertura. Taxa do spot no
nível básico: 0,1% em cada compra e venda. Um ciclo por vez.

O juiz (LightGBM) aprende a "boa entrada": subir `tp` antes de cair `tp` a partir da abertura seguinte, sem
precisar de DCA. Numa entrada ao acaso isso dá perto de 50%. Usa as variáveis de wescalp.brain com contexto de
15 min, 1 h, 4 h e 1 dia.

Walk-forward: treina 60 semanas (aprende nas primeiras 45; o corte sai das 15 seguintes: o de maior lucro com
pelo menos 85% dos ciclos no lucro), testa 4, anda 4; as últimas 16 semanas ficam intocadas. O teste é uma
simulação contínua das 24 semanas, cada janela com o seu modelo. Comparação: os gatilhos sem juiz, o "DCA cego"
(compra em qualquer candle e aporta a cada queda de `tp`) e segurar o BTC.
"""

import argparse
import asyncio
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import brain

log = logging.getLogger("wescalp.spot")

SYMBOL = "BTC-USDT"
FEE = 0.001  # spot, nível básico, por lado
TPS = (0.02, 0.03, 0.05)
STOP_MULT = 3  # stop 3·tp abaixo da entrada: cabem dois aportes de tp em tp
MAX_PARTS = 3
MAX_DAYS = 30
BARS_PER_DAY = 86400 // brain.BAR
CONTEXTS = (15, 60, 240, 1440)
MIN_CYCLE_WIN = 0.85  # 80% pedido + folga
MIN_CAL_CYCLES = 10
MIN_PER_DAY = 2  # pedido: pelo menos 2 ciclos por dia
LGBM_ROUNDS = 200
LGBM_PARAMS = dict(
    objective="binary", learning_rate=0.03, num_leaves=7, min_data_in_leaf=300, feature_fraction=0.7,
    bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, seed=0, num_threads=1, verbose=-1,
)
JUIZ, JUIZ2, GATILHOS, CEGO = "juiz", "juiz ≥ 2/dia", "gatilhos sem juiz", "DCA cego (qualquer candle)"


class Crossing:
    """Primeiro índice ≥ start em que a série chega a um nível, em O(log n) por consulta (tabela esparsa).

    up=True procura x ≥ nível (máximas); up=False procura x ≤ nível (mínimas).
    """

    def __init__(self, x: np.ndarray, up: bool):
        self.x, self.up = x, up
        op = np.maximum if up else np.minimum
        self.t = [x]
        while (1 << len(self.t)) <= len(x):
            prev, half = self.t[-1], 1 << (len(self.t) - 1)
            self.t.append(op(prev[:-half], prev[half:]))  # t[j][p] = op(x[p : p + 2^j])

    def first(self, start: int, level: float) -> int | None:
        p = start
        for j in range(len(self.t) - 1, -1, -1):
            t = self.t[j]
            if p < len(t) and (t[p] < level if self.up else t[p] > level):
                p += 1 << j  # o bloco inteiro não chega ao nível
        if p < len(self.x) and (self.x[p] >= level if self.up else self.x[p] <= level):
            return p
        return None


def barrier(f: brain.Frame, i: int, up: Crossing, dn: Crossing, tp: float,
            horizon: int = MAX_DAYS * BARS_PER_DAY) -> tuple[int, int] | None:
    """Entrada na abertura do candle i+1: (1, saída) se sobe tp antes de cair tp, (0, saída) se cai antes ou
    no mesmo candle; None se nenhum dos dois em `horizon` candles."""
    if i + 1 >= len(f.ts):
        return None
    e, last = f.open[i + 1], i + horizon
    ku, kd = up.first(i + 1, e * (1 + tp)), dn.first(i + 1, e * (1 - tp))
    ku = ku if ku is not None and ku <= last else None
    kd = kd if kd is not None and kd <= last else None
    if ku is None and kd is None:
        return None
    if kd is not None and (ku is None or kd <= ku):
        return 0, kd
    return 1, ku


def part_return(entries: list[float], px: float, fee: float = FEE) -> float:
    """Resultado em bps do capital do ciclo (MAX_PARTS partes iguais), com taxa na compra e na venda."""
    return sum((px / e) * (1 - fee) - 1 - fee for e in entries) / MAX_PARTS * 1e4


@dataclass(frozen=True, slots=True)
class Trade:
    signal_i: int  # candle em cujo fechamento saiu o sinal da primeira compra
    exit_i: int
    outcome: str  # tp | sl | time | aberto (fechado a mercado no fim da simulação)
    parts: int
    ret_bps: float
    good_entry: int | None  # a primeira compra subiu tp antes de cair tp?


@dataclass
class Policy:
    trades: list[Trade]
    max_dd_bps: float  # maior queda do patrimônio marcado a mercado (realizado + posição aberta)
    bars_in_market: int
    buys: int
    equity: list[float] = field(default_factory=list)  # patrimônio reinvestido (começa em 1) nos candles de `record`


def simulate(f: brain.Frame, lo: int, hi: int, end: int, signal, tp: float, good=lambda i: None,
             fee: float = FEE, horizon: int = MAX_DAYS * BARS_PER_DAY, record: np.ndarray | None = None) -> Policy:
    """Um ciclo por vez. Decide nos fechamentos dos candles [lo, hi), executa na abertura seguinte e deixa as
    saídas correrem até o candle end-1, onde o que estiver aberto é fechado a mercado ("aberto").

    `record`: máscara dos candles em cujo fechamento anotar o patrimônio com reinvestimento (cada ciclo usa o
    patrimônio do início dele), marcado a mercado com a taxa de venda."""
    trades: list[Trade] = []
    parts: list[float] = []
    pending, entry_k, signal_i, stop = False, 0, 0, 0.0
    realized = peak = max_dd = 0.0
    in_market = buys = 0
    wealth, marks = 1.0, []
    for k in range(lo, end):
        o, h, l, c = f.open[k], f.high[k], f.low[k], f.close[k]
        if pending:
            if not parts:
                entry_k, stop = k, o * (1 - STOP_MULT * tp)
            parts.append(o)
            buys += 1
            pending = False
        if parts:
            in_market += 1
            target = sum(parts) / len(parts) * (1 + tp)
            out = None
            if l <= stop:
                out, px = "sl", min(stop, o)
            elif h >= target:
                out, px = "tp", max(target, o)
            elif k - entry_k >= horizon:
                out, px = "time", c
            elif k == end - 1:
                out, px = "aberto", c
            if out:
                r = part_return(parts, px, fee)
                realized += r
                wealth *= 1 + r / 1e4
                trades.append(Trade(signal_i, k, out, len(parts), r, good(signal_i)))
                parts = []
        if k < hi and k + 1 < end:
            if not parts:
                if signal(k):
                    pending, signal_i = True, k
            elif len(parts) < MAX_PARTS and c <= parts[-1] * (1 - tp) and signal(k):
                pending = True
        open_r = part_return(parts, c, fee) if parts else 0.0
        equity = realized + open_r
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
        if record is not None and record[k]:
            marks.append(wealth * (1 + open_r / 1e4))
    return Policy(trades, max_dd, in_market, buys, marks)


def fit_lgbm(x: np.ndarray, y: np.ndarray):
    import lightgbm

    return lightgbm.train(LGBM_PARAMS, lightgbm.Dataset(x, y.astype(float)), num_boost_round=LGBM_ROUNDS)


def choose_cuts(f: brain.Frame, lo: int, hi: int, cand: np.ndarray, p: np.ndarray,
                tp: float) -> tuple[tuple[float, bool], tuple[float, bool]]:
    """Dois cortes de nota na calibração [lo, hi), cada um com True se cumpriu a meta:
    1. o de maior lucro com ≥ MIN_CYCLE_WIN dos ciclos no lucro e ≥ MIN_CAL_CYCLES ciclos; sem nenhum assim, o de
       maior acerto;
    2. o mesmo exigindo também ≥ MIN_PER_DAY ciclos por dia; sem nenhum assim, o de maior lucro entre os que dão
       ≥ MIN_PER_DAY por dia, ou o mais frequente.
    """
    days = max((f.ts[hi - 1] - f.ts[lo]) / 86400, 1e-9)
    scores = p[lo:hi][cand[lo:hi]]
    tried = []  # (corte, acerto, total, ciclos, ciclos por dia)
    for t in sorted(set(np.quantile(scores, brain.CUT_LEVELS))) if len(scores) else []:
        pol = simulate(f, lo, hi, hi, lambda k: cand[k] and p[k] >= t, tp)
        if pol.trades:
            r = np.array([x.ret_bps for x in pol.trades])
            tried.append((t, float((r > 0).mean()), float(r.sum()), len(r), len(r) / days))
    if not tried:
        return (math.inf, False), (math.inf, False)
    ok = [x for x in tried if x[1] >= MIN_CYCLE_WIN and x[3] >= MIN_CAL_CYCLES]
    if ok:
        first = (max(ok, key=lambda x: x[2])[0], True)
    else:
        first = (max([x for x in tried if x[3] >= MIN_CAL_CYCLES] or tried, key=lambda x: (x[1], x[2]))[0], False)
    ok2 = [x for x in ok if x[4] >= MIN_PER_DAY]
    if ok2:
        return first, (max(ok2, key=lambda x: x[2])[0], True)
    frequent = [x for x in tried if x[4] >= MIN_PER_DAY]
    second = max(frequent, key=lambda x: x[2]) if frequent else max(tried, key=lambda x: x[4])
    return first, (second[0], False)


@dataclass
class TpRun:
    tp: float
    policies: dict[str, Policy]
    lo: int
    end: int
    folds: int
    folds_met: int
    folds_met2: int
    base_good: float  # fração de boas entradas entre todos os candles de teste (o acaso)
    importance: list[tuple[str, float]]


def walk_forward(f: brain.Frame, x: np.ndarray, names: list[str], tp: float,
                 holdout_weeks: int = brain.HOLDOUT_WEEKS) -> TpRun:
    week = (f.ts - f.ts[0]) // brain.WEEK
    dev_weeks = int(week[-1]) + 1 - holdout_weeks
    end = int(np.searchsorted(week, dev_weeks))
    up, dn = Crossing(f.high, True), Crossing(f.low, False)
    cand = brain.any_candidate(f)
    cand[end:] = False
    labels = {}
    for i in np.flatnonzero(cand):
        b = barrier(f, int(i), up, dn, tp)
        if b is not None and b[1] < end:
            labels[int(i)] = b
    cache: dict[int, int | None] = {}

    def good(i: int) -> int | None:
        if i not in cache:
            b = labels.get(i) or barrier(f, i, up, dn, tp)
            cache[i] = b[0] if b is not None and b[1] < end else None
        return cache[i]

    p = np.full(len(f.ts), np.nan)
    cut, cut2 = np.full(len(f.ts), np.inf), np.full(len(f.ts), np.inf)
    folds = met = met2 = 0
    importance: list[tuple[str, float]] = []
    test_lo = None
    for s in range(0, dev_weeks - brain.TRAIN_WEEKS - brain.TEST_WEEKS + 1, brain.TEST_WEEKS):
        cal_lo, te_lo, te_hi = (int(np.searchsorted(week, w)) for w in
                                (s + brain.FIT_WEEKS, s + brain.TRAIN_WEEKS, s + brain.TRAIN_WEEKS + brain.TEST_WEEKS))
        fit = [i for i, (y, k) in labels.items() if week[i] >= s and i < cal_lo and k < cal_lo]
        if len(fit) < 1000:
            continue
        booster = fit_lgbm(x[fit], np.array([labels[i][0] for i in fit]))
        for a, b in ((cal_lo, te_lo), (te_lo, te_hi)):
            idx = np.flatnonzero(cand[a:b]) + a
            if len(idx):
                p[idx] = booster.predict(x[idx])
        (t, ok), (t2, ok2) = choose_cuts(f, cal_lo, te_lo, cand, p, tp)
        cut[te_lo:te_hi], cut2[te_lo:te_hi] = t, t2
        test_lo = te_lo if test_lo is None else test_lo
        folds, met, met2 = folds + 1, met + ok, met2 + ok2
        gain = booster.feature_importance("gain")
        importance = sorted(zip(names, gain / gain.sum()), key=lambda r: -r[1])

    policies = {
        JUIZ: simulate(f, test_lo, end, end, lambda k: cand[k] and p[k] >= cut[k], tp, good),
        JUIZ2: simulate(f, test_lo, end, end, lambda k: cand[k] and p[k] >= cut2[k], tp, good),
        GATILHOS: simulate(f, test_lo, end, end, lambda k: cand[k], tp, good),
        CEGO: simulate(f, test_lo, end, end, lambda k: True, tp, good),
    }
    sample = range(test_lo, end, 12)  # o acaso: uma entrada por hora
    goods = [g for g in (good(i) for i in sample) if g is not None]
    return TpRun(tp, policies, test_lo, end, folds, met, met2, float(np.mean(goods)) if goods else math.nan,
                 importance)


def stats(pol: Policy, f: brain.Frame, lo: int, end: int) -> dict:
    days = (f.ts[end - 1] - f.ts[lo]) / 86400
    r = np.array([t.ret_bps for t in pol.trades])
    n = len(r)
    out = {"n": n, "per_day": n / days, "buys_day": pol.buys / days, "dd": pol.max_dd_bps,
           "in_market": pol.bars_in_market / (end - lo)}
    if not n:
        return out
    week = (f.ts - f.ts[lo]) // brain.WEEK
    weekly = np.zeros(int(week[end - 1]) + 1)
    for t in pol.trades:
        weekly[int(week[t.exit_i])] += t.ret_bps
    goods = [t.good_entry for t in pol.trades if t.good_entry is not None]
    held = np.array([(f.ts[t.exit_i] - f.ts[t.signal_i]) / 86400 for t in pol.trades])
    out.update({
        "good": float(np.mean(goods)) if goods else math.nan, "win": float((r > 0).mean()),
        "avg_win": float(r[r > 0].mean()) if (r > 0).any() else 0.0,
        "avg_loss": float(r[r <= 0].mean()) if (r <= 0).any() else 0.0,
        "mean": float(r.mean()), "total": float(r.sum()), "worst": float(r.min()),
        "stops": sum(t.outcome == "sl" for t in pol.trades) / n,
        "dca": sum(t.parts > 1 for t in pol.trades) / n,
        "open_end": sum(t.outcome == "aberto" for t in pol.trades),
        "held": float(np.median(held)), "held90": float(np.quantile(held, 0.9)),
        "weekly": list(weekly), "weeks_pos": int((weekly > 0).sum()), "weeks": len(weekly),
    })
    return out


def years_of(f: brain.Frame, idx) -> np.ndarray:
    return f.ts[idx].astype("datetime64[s]").astype("datetime64[Y]").astype(int) + 1970


def yearly(pol: Policy, f: brain.Frame) -> dict[int, float]:
    """Resultado por ano (pelo dia da saída), em % do capital."""
    out: dict[int, float] = {}
    for t in pol.trades:
        y = int(years_of(f, t.exit_i))
        out[y] = out.get(y, 0.0) + t.ret_bps / 100
    return out


def hold_yearly(f: brain.Frame, lo: int, end: int) -> dict[int, float]:
    """Segurar o BTC dentro de cada ano do período [lo, end), em %."""
    years = years_of(f, np.arange(lo, end))
    out = {}
    for y in np.unique(years):
        idx = np.flatnonzero(years == y) + lo
        out[int(y)] = (f.close[idx[-1]] / f.open[idx[0]] - 1) * 100
    return out


def format_run(run: TpRun, f: brain.Frame) -> str:
    head = (f"{'regra':<28}{'ciclos':>7}{'/dia':>6}{'compras/dia':>12}{'entrada boa':>12}{'acerto':>8}"
            f"{'ganho':>8}{'perda':>8}{'líq/ciclo':>10}{'total':>9}{'pior':>8}{'queda':>8}{'stops':>7}"
            f"{'c/ DCA':>8}{'dias':>6}{'dias90':>7}{'posic.':>7}{'sem+':>9}")
    lines = [f"\nalvo {run.tp:.1%}, stop {STOP_MULT * run.tp:.1%} abaixo da entrada: {run.folds} rodadas; meta da"
             f" calibração cumprida pelo juiz em {run.folds_met} e pelo juiz ≥ 2/dia em {run.folds_met2};"
             f" entrada boa ao acaso {run.base_good:.1%}", head]
    for name, pol in run.policies.items():
        s = stats(pol, f, run.lo, run.end)
        if not s["n"]:
            lines.append(f"{name:<28}{0:>7}")
            continue
        lines.append(
            f"{name:<28}{s['n']:>7}{s['per_day']:>6.2f}{s['buys_day']:>12.2f}{s['good']:>12.1%}{s['win']:>8.1%}"
            f"{s['avg_win'] / 100:>7.2f}%{s['avg_loss'] / 100:>7.2f}%{s['mean'] / 100:>9.2f}%"
            f"{s['total'] / 100:>8.1f}%{s['worst'] / 100:>7.1f}%{s['dd'] / 100:>7.1f}%{s['stops']:>7.1%}"
            f"{s['dca']:>8.1%}{s['held']:>6.1f}{s['held90']:>7.1f}{s['in_market']:>7.0%}"
            f"{s['weeks_pos']:>5}/{s['weeks']}"
        )
    hold = hold_yearly(f, run.lo, run.end)
    per = {name: yearly(pol, f) for name, pol in run.policies.items()}
    lines.append("por ano, em % do capital (segurar = BTC comprado na abertura do ano e vendido no fechamento):")
    lines.append(f"{'ano':<6}" + "".join(f"{n[:22]:>24}" for n in per) + f"{'segurar BTC':>14}")
    for y in sorted(hold):
        lines.append(f"{y:<6}" + "".join(f"{per[n].get(y, 0.0):>23.1f}%" for n in per) + f"{hold[y]:>13.1f}%")
    return "\n".join(lines)


_F: brain.Frame | None = None
_X: np.ndarray | None = None
_NAMES: list[str] = []
_HOLDOUT = brain.HOLDOUT_WEEKS


def _init(f, x, names, holdout) -> None:
    global _F, _X, _NAMES, _HOLDOUT
    _F, _X, _NAMES, _HOLDOUT = f, x, names, holdout


def _work(tp: float) -> TpRun:
    return walk_forward(_F, _X, _NAMES, tp, _HOLDOUT)


def run(args) -> None:
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    from .validation import deflated_sharpe, sharpe

    path = args.file or brain.candle_path(SYMBOL, args.data)
    candles = brain.load_candles(path, SYMBOL, "spot")
    if not candles:
        raise SystemExit(f"sem candles em {path}; rode antes: python -m wescalp.spot fetch")
    f = brain.build(candles, CONTEXTS)
    x, names = brain.model_matrix(f)
    x = x.astype(np.float32)
    fmt = lambda t: datetime.fromtimestamp(int(t), timezone.utc).strftime("%Y-%m-%d")  # noqa: E731
    print(f"{path.name}: {len(candles)} candles de 5 min de {fmt(f.ts[0])} a {fmt(f.ts[-1])};"
          f" intocadas as últimas {args.holdout} semanas do arquivo")
    print(f"taxa {FEE:.1%} por compra e por venda; DCA só com novo sinal e {MAX_PARTS - 1} aportes no máximo;"
          f" {MAX_DAYS} dias no máximo por ciclo")
    print("entrada boa = a 1ª compra subiu o alvo antes de cair o alvo; acerto = ciclos com líquido > 0;"
          " ganho/perda/líq/total/pior/queda em % do capital do ciclo; queda = maior queda marcada a mercado;"
          " dias = mediana do ciclo; posic. = tempo comprado", flush=True)
    _init(f, x, names, args.holdout)  # os processos herdam os dados (fork) em vez de copiá-los
    with ProcessPoolExecutor(len(args.tps), mp_context=multiprocessing.get_context("fork")) as pool:
        runs = list(pool.map(_work, args.tps))
    lo, end = runs[0].lo, runs[0].end
    print(f"\nperíodo de teste: {fmt(f.ts[lo])} a {fmt(f.ts[end - 1])};"
          f" segurar BTC: {f.close[end - 1] / f.open[lo] - 1:+.1%}")
    for r in runs:
        print(format_run(r, f), flush=True)
    juizes = {(r.tp, name): stats(r.policies[name], f, r.lo, r.end) for r in runs for name in (JUIZ, JUIZ2)}
    trials = [sharpe(s["weekly"]) for s in juizes.values() if s["n"]]
    done = [k for k, s in juizes.items() if s["n"]]
    if done:
        tp, name = max(done, key=lambda k: juizes[k]["total"])
        s = juizes[tp, name]
        print(f"\nmelhor juiz por total: {name}, alvo {tp:.1%}, total {s['total'] / 100:+.1f}%, Sharpe semanal"
              f" {sharpe(s['weekly']):.2f}, DSR entre {len(trials)} juízes {deflated_sharpe(s['weekly'], trials):.3f}")
        imp = next(r for r in runs if r.tp == tp).importance
        print("importância no último juiz desse alvo: " + ", ".join(f"{n} {g:.0%}" for n, g in imp[:8]))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=brain.CANDLE_DIR, help="pasta dos candles")
    sub = p.add_subparsers(dest="cmd", required=True)
    fe = sub.add_parser("fetch", help="baixa ou completa os candles de 5 min do BTC-USDT da KuCoin")
    fe.add_argument("--weeks", type=int, default=brain.WEEKS)
    r = sub.add_parser("run", help="walk-forward do juiz e relatório")
    r.add_argument("--file", type=Path, help="arquivo de candles (padrão: o da KuCoin; ex.: o de wescalp.binance)")
    r.add_argument("--tps", type=float, nargs="+", default=list(TPS), help="alvos (fração), ex.: 0.01 0.015")
    r.add_argument("--holdout", type=int, default=brain.HOLDOUT_WEEKS, help="últimas semanas do arquivo intocadas")
    args = p.parse_args()
    if args.cmd == "fetch":
        cs = asyncio.run(brain.fetch(SYMBOL, args.weeks, brain.candle_path(SYMBOL, args.data), "spot"))
        log.info("%d candles de %s a %s", len(cs), cs[0].ts, cs[-1].ts)
    else:
        run(args)


if __name__ == "__main__":
    main()
