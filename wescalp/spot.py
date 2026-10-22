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
from dataclasses import dataclass
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
LGBM_ROUNDS = 200
LGBM_PARAMS = dict(
    objective="binary", learning_rate=0.03, num_leaves=7, min_data_in_leaf=300, feature_fraction=0.7,
    bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, seed=0, num_threads=1, verbose=-1,
)
JUIZ, GATILHOS, CEGO = "juiz", "gatilhos sem juiz", "DCA cego (qualquer candle)"


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


def simulate(f: brain.Frame, lo: int, hi: int, end: int, signal, tp: float, good=lambda i: None,
             fee: float = FEE, horizon: int = MAX_DAYS * BARS_PER_DAY) -> Policy:
    """Um ciclo por vez. Decide nos fechamentos dos candles [lo, hi), executa na abertura seguinte e deixa as
    saídas correrem até o candle end-1, onde o que estiver aberto é fechado a mercado ("aberto")."""
    trades: list[Trade] = []
    parts: list[float] = []
    pending, entry_k, signal_i, stop = False, 0, 0, 0.0
    realized = peak = max_dd = 0.0
    in_market = buys = 0
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
                trades.append(Trade(signal_i, k, out, len(parts), r, good(signal_i)))
                parts = []
        if k < hi and k + 1 < end:
            if not parts:
                if signal(k):
                    pending, signal_i = True, k
            elif len(parts) < MAX_PARTS and c <= parts[-1] * (1 - tp) and signal(k):
                pending = True
        equity = realized + (part_return(parts, c, fee) if parts else 0.0)
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    return Policy(trades, max_dd, in_market, buys)


def fit_lgbm(x: np.ndarray, y: np.ndarray):
    import lightgbm

    return lightgbm.train(LGBM_PARAMS, lightgbm.Dataset(x, y.astype(float)), num_boost_round=LGBM_ROUNDS)


def choose_cut(f: brain.Frame, lo: int, hi: int, cand: np.ndarray, p: np.ndarray, tp: float) -> tuple[float, bool]:
    """Corte de nota na calibração [lo, hi): o de maior lucro com ≥ MIN_CYCLE_WIN dos ciclos no lucro e pelo
    menos MIN_CAL_CYCLES ciclos. Sem nenhum assim, o de maior acerto (marcado False)."""
    scores = p[lo:hi][cand[lo:hi]]
    tried = []
    for t in sorted(set(np.quantile(scores, brain.CUT_LEVELS))) if len(scores) else []:
        pol = simulate(f, lo, hi, hi, lambda k: cand[k] and p[k] >= t, tp)
        if len(pol.trades) >= MIN_CAL_CYCLES:
            r = np.array([x.ret_bps for x in pol.trades])
            tried.append((t, float((r > 0).mean()), float(r.sum())))
    ok = [x for x in tried if x[1] >= MIN_CYCLE_WIN]
    if ok:
        return max(ok, key=lambda x: x[2])[0], True
    if not tried:
        return math.inf, False
    return max(tried, key=lambda x: (x[1], x[2]))[0], False


@dataclass
class TpRun:
    tp: float
    policies: dict[str, Policy]
    lo: int
    end: int
    folds: int
    folds_met: int
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
    cut = np.full(len(f.ts), np.inf)
    folds = met = 0
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
        t, ok = choose_cut(f, cal_lo, te_lo, cand, p, tp)
        cut[te_lo:te_hi] = t
        test_lo = te_lo if test_lo is None else test_lo
        folds, met = folds + 1, met + ok
        gain = booster.feature_importance("gain")
        importance = sorted(zip(names, gain / gain.sum()), key=lambda r: -r[1])

    policies = {
        JUIZ: simulate(f, test_lo, end, end, lambda k: cand[k] and p[k] >= cut[k], tp, good),
        GATILHOS: simulate(f, test_lo, end, end, lambda k: cand[k], tp, good),
        CEGO: simulate(f, test_lo, end, end, lambda k: True, tp, good),
    }
    sample = range(test_lo, end, 12)  # o acaso: uma entrada por hora
    goods = [g for g in (good(i) for i in sample) if g is not None]
    return TpRun(tp, policies, test_lo, end, folds, met, float(np.mean(goods)) if goods else math.nan, importance)


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


def format_run(run: TpRun, f: brain.Frame) -> str:
    head = (f"{'regra':<28}{'ciclos':>7}{'/dia':>6}{'compras/dia':>12}{'entrada boa':>12}{'acerto':>8}"
            f"{'ganho':>8}{'perda':>8}{'líq/ciclo':>10}{'total':>9}{'pior':>8}{'queda':>8}{'stops':>7}"
            f"{'c/ DCA':>8}{'dias':>6}{'dias90':>7}{'posic.':>7}{'sem+':>7}")
    lines = [f"\nalvo {run.tp:.0%}, stop {STOP_MULT * run.tp:.0%} abaixo da entrada: {run.folds} rodadas, juiz"
             f" dentro da meta na calibração em {run.folds_met}; entrada boa ao acaso {run.base_good:.1%}", head]
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
            f"{s['weeks_pos']:>4}/{s['weeks']}"
        )
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
    from concurrent.futures import ProcessPoolExecutor

    from .validation import deflated_sharpe, sharpe

    candles = brain.load_candles(brain.candle_path(SYMBOL, args.data), SYMBOL, "spot")
    if not candles:
        raise SystemExit("sem candles; rode antes: python -m wescalp.spot fetch")
    f = brain.build(candles, CONTEXTS)
    x, names = brain.model_matrix(f)
    fmt = lambda t: datetime.fromtimestamp(int(t), timezone.utc).strftime("%Y-%m-%d")  # noqa: E731
    print(f"{SYMBOL} spot 5 min: {len(candles)} candles de {fmt(f.ts[0])} a {fmt(f.ts[-1])};"
          f" intocadas as últimas {args.holdout} semanas")
    print(f"taxa {FEE:.1%} por compra e por venda; DCA só com novo sinal e {MAX_PARTS - 1} aportes no máximo;"
          f" {MAX_DAYS} dias no máximo por ciclo")
    print("entrada boa = a 1ª compra subiu o alvo antes de cair o alvo; acerto = ciclos com líquido > 0;"
          " ganho/perda/líq/total/pior/queda em % do capital do ciclo; queda = maior queda marcada a mercado;"
          " dias = mediana do ciclo; posic. = tempo comprado", flush=True)
    with ProcessPoolExecutor(len(TPS), initializer=_init, initargs=(f, x, names, args.holdout)) as pool:
        runs = list(pool.map(_work, TPS))
    lo, end = runs[0].lo, runs[0].end
    print(f"\nperíodo de teste: {fmt(f.ts[lo])} a {fmt(f.ts[end - 1])};"
          f" segurar BTC: {f.close[end - 1] / f.open[lo] - 1:+.1%}")
    for r in runs:
        print(format_run(r, f), flush=True)
    weekly = {r.tp: stats(r.policies[JUIZ], f, r.lo, r.end) for r in runs}
    trials = [sharpe(s["weekly"]) for s in weekly.values() if s["n"]]
    best = max((r for r in runs if weekly[r.tp]["n"]), key=lambda r: weekly[r.tp]["total"], default=None)
    if best is not None:
        s = weekly[best.tp]
        print(f"\nmelhor juiz por total: alvo {best.tp:.0%}, total {s['total'] / 100:+.1f}%, Sharpe semanal"
              f" {sharpe(s['weekly']):.2f}, DSR entre {len(trials)} alvos {deflated_sharpe(s['weekly'], trials):.3f}")
        print("importância no último juiz: " + ", ".join(f"{n} {g:.0%}" for n, g in best.importance[:8]))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=brain.CANDLE_DIR, help="pasta dos candles")
    sub = p.add_subparsers(dest="cmd", required=True)
    fe = sub.add_parser("fetch", help="baixa ou completa os candles de 5 min do BTC-USDT")
    fe.add_argument("--weeks", type=int, default=brain.WEEKS)
    r = sub.add_parser("run", help="walk-forward do juiz e relatório")
    r.add_argument("--holdout", type=int, default=brain.HOLDOUT_WEEKS, help="últimas semanas intocadas")
    args = p.parse_args()
    if args.cmd == "fetch":
        cs = asyncio.run(brain.fetch(SYMBOL, args.weeks, brain.candle_path(SYMBOL, args.data), "spot"))
        log.info("%d candles de %s a %s", len(cs), cs[0].ts, cs[-1].ts)
    else:
        run(args)


if __name__ == "__main__":
    main()
