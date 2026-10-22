"""Cérebro de compra: sinais de buy com DCA, julgados por LightGBM, em walk-forward sobre os candles do perpétuo.

    python -m wescalp.brain fetch            # baixa/atualiza 100 semanas de candles de 5 min em data/candles/
    python -m wescalp.brain run              # walk-forward e relatório

Só compra, tudo simulado (nada envia ordem). Um ciclo:
1. Candidato: no fechamento de um candle de 5 min, uma heurística gráfica propõe compra (lista em CANDIDATES).
2. Juiz: o LightGBM dá a probabilidade de o ciclo fechar no lucro, com variáveis de 5 min, 15 min, 1 h e 4 h.
   Só passa quem fica acima do corte, tirado do treino para dar acerto alto.
3. Entrada 1 na abertura do candle seguinte, a mercado (taker). Duas ordens de DCA limitadas (maker) ficam
   `dca` e 2·`dca` ATRs de 1 h abaixo dela: é a zona planejada de entrada fracionada, cada parte 1/3 do capital.
4. Saída: alvo `tp` ATRs de 1 h acima do preço médio (maker); um só stop para a posição inteira `sl` ATRs abaixo
   da entrada 1 (taker); ou tempo máximo (taker, no fechamento). Só um ciclo aberto por vez.

O ATR de 5 min (~14 bps) é menor que a taxa, por isso a geometria do ciclo usa o ATR de 1 h (~40 a 50 bps).
Dentro de um candle não se sabe a ordem dos preços: primeiro as compras na mínima e o stop, e o alvo só conta em
candle sem compra nova e sem stop (o conservador). Abriu além do stop: sai na abertura.

Walk-forward como wescalp.research: treina 60 semanas, testa as 4 seguintes, anda 4. Dentro do treino, o modelo
aprende nas primeiras 3/4 e o corte sai do último 1/4, que ele não viu. As últimas 16 semanas ficam intocadas.
Resultado do ciclo em bps do capital do ciclo (as 3 partes), depois das taxas de config (maker 0,02%, taker 0,06%).
"""

import argparse
import asyncio
import csv
import gzip
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from .config import CANDLE_SECONDS, FUTURES_SYMBOL, MAKER_FEE, TAKER_FEE
from .features import compute as candle_features
from .kucoin import Candle, KucoinRest

log = logging.getLogger("wescalp.brain")

CANDLE_DIR = Path("data/candles")
WEEKS = 100
HOLDOUT_WEEKS = 16
TRAIN_WEEKS, TEST_WEEKS = 60, 4
MIN_WIN_RATE = 0.80
CUT_MARGIN = 0.05  # o corte mira 85% no trecho de calibração, para sobrar folga fora da amostra
BAR = CANDLE_SECONDS


# ---------- candles ----------

def candle_path(symbol: str = FUTURES_SYMBOL, data_dir: Path = CANDLE_DIR) -> Path:
    return data_dir / f"{symbol}-5m.csv.gz"


def save_candles(path: Path, candles: list[Candle]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    with gzip.open(tmp, "wt", newline="") as f:
        w = csv.writer(f)
        for c in candles:
            w.writerow((int(c.ts.timestamp()), c.open, c.high, c.low, c.close, c.volume, c.turnover))
    tmp.rename(path)


def load_candles(path: Path, symbol: str = FUTURES_SYMBOL) -> list[Candle]:
    if not path.exists():
        return []
    with gzip.open(path, "rt", newline="") as f:
        return [
            Candle("futures", symbol, datetime.fromtimestamp(int(t), timezone.utc), *map(float, rest))
            for t, *rest in csv.reader(f)
        ]


async def fetch(symbol: str, weeks: int, path: Path) -> list[Candle]:
    """Completa o arquivo até o último candle fechado; retoma de onde parou."""
    have = {c.ts: c for c in load_candles(path, symbol)}
    end = int(time.time()) // BAR * BAR
    start = end - weeks * 7 * 86400
    if have:
        start = max(start, int(max(have).timestamp()) + BAR)
    rest = KucoinRest()
    try:
        step = 200 * BAR
        for a in range(start, end, step):
            for c in await rest.futures_candles(symbol, a, min(a + step, end)):
                have[c.ts] = c
    finally:
        await rest.aclose()
    candles = [have[k] for k in sorted(have) if k.timestamp() >= end - weeks * 7 * 86400]
    save_candles(path, candles)
    return candles


# ---------- variáveis ----------

def ema(x: np.ndarray, n: int) -> np.ndarray:
    k, out, e = 2 / (n + 1), np.empty(len(x)), x[0]
    for i, v in enumerate(x):
        e = v * k + e * (1 - k)
        out[i] = e
    return out


def rsi(close: np.ndarray, n: int = 14) -> np.ndarray:
    """RSI de Wilder; NaN nos primeiros n."""
    d = np.diff(close, prepend=close[0])
    up, dn = np.clip(d, 0, None), np.clip(-d, 0, None)
    out = np.full(len(close), np.nan)
    if len(close) <= n:
        return out
    au, ad = up[1:n + 1].mean(), dn[1:n + 1].mean()
    for i in range(n, len(close)):
        if i > n:
            au, ad = (au * (n - 1) + up[i]) / n, (ad * (n - 1) + dn[i]) / n
        out[i] = 100.0 if ad == 0 else 100 - 100 / (1 + au / ad)
    return out


def lagged(x: np.ndarray, k: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    out[k:] = x[:-k]
    return out


@dataclass
class Frame:
    ts: np.ndarray  # abertura do candle de 5 min, segundos
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    atr_h1: np.ndarray  # ATR de 1 h conhecido no fechamento do candle (só barras de 1 h completas)
    x: np.ndarray  # variáveis no fechamento do candle
    names: list[str]
    candidates: dict[str, np.ndarray]  # nome → máscara booleana


def higher(ts: np.ndarray, o, h, l, c, v, minutes: int) -> dict[str, np.ndarray]:
    """Barras de `minutes` montadas dos candles de 5 min e, para cada candle, os valores da última barra maior
    já fechada no fechamento dele. Barra incompleta (faltou candle) não conta."""
    width = minutes * 60
    key = ts // width
    keys, first, counts = np.unique(key, return_index=True, return_counts=True)
    last = first + counts - 1
    H = {
        "open": o[first], "close": c[last],
        "high": np.maximum.reduceat(h, first), "low": np.minimum.reduceat(l, first),
    }
    complete = counts == width // BAR
    cl = H["close"]
    prev = np.concatenate([[cl[0]], cl[:-1]])
    tr = np.maximum(H["high"] - H["low"], np.maximum(abs(H["high"] - prev), abs(H["low"] - prev)))
    atr = np.full(len(cl), np.nan)
    if len(tr) >= 14:
        atr[13] = tr[:14].mean()
        for i in range(14, len(tr)):
            atr[i] = (atr[i - 1] * 13 + tr[i]) / 14
    e20, e50 = ema(cl, 20), ema(cl, 50)
    vals = {
        "atr": atr,
        "rsi": rsi(cl),
        "ema20_dist": (cl - e20) / atr,
        "ema50_dist": (cl - e50) / atr,
        "trend": np.sign(e20 - e50),
        "ret3": (cl / lagged(cl, 3) - 1) * 1e4,
        "pos_range": (cl - H["low"]) / np.where(H["high"] > H["low"], H["high"] - H["low"], np.nan),
    }
    for k in vals:
        vals[k] = np.where(complete, vals[k], np.nan)
    # Candle i fecha em ts+BAR; a última barra maior fechada é a de chave (ts+BAR)//width - 1.
    want = (ts + BAR) // width - 1
    idx = np.searchsorted(keys, want)
    found = (idx < len(keys)) & (keys[np.minimum(idx, len(keys) - 1)] == want)
    idx = np.minimum(idx, len(keys) - 1)
    return {k: np.where(found, a[idx], np.nan) for k, a in vals.items()}


def build(candles: list[Candle]) -> Frame:
    ts = np.array([int(c.ts.timestamp()) for c in candles], dtype=np.int64)
    o, h, l, c, v = (np.array([getattr(x, f) for x in candles]) for f in ("open", "high", "low", "close", "volume"))
    f5 = candle_features(candles)
    nan = lambda xs: np.array([np.nan if x is None else x for x in xs], dtype=float)  # noqa: E731
    atr5, vwap, vsd = nan(f5["atr"]), nan(f5["vwap"]), nan(f5["vwap_sd"])
    ph, pl, vr = nan(f5["prev_high"]), nan(f5["prev_low"]), nan(f5["vol_ratio"])
    e9, e21, e50, e200 = ema(c, 9), ema(c, 21), ema(c, 50), ema(c, 200)
    r5 = rsi(c)
    sma20 = np.convolve(c, np.ones(20) / 20, mode="full")[:len(c)]
    sd20 = np.sqrt(np.maximum(np.convolve(c * c, np.ones(20) / 20, mode="full")[:len(c)] - sma20 ** 2, 0))
    sma20[:19] = np.nan
    bb_lo, bb_hi = sma20 - 2 * sd20, sma20 + 2 * sd20
    rng = np.where(h > l, h - l, np.nan)

    cols = {
        "ret1": (c / lagged(c, 1) - 1) * 1e4, "ret3": (c / lagged(c, 3) - 1) * 1e4,
        "ret12": (c / lagged(c, 12) - 1) * 1e4, "ret48": (c / lagged(c, 48) - 1) * 1e4,
        "atr5_bps": atr5 / c * 1e4, "rsi5": r5,
        "ema9_dist": (c - e9) / atr5, "ema21_dist": (c - e21) / atr5,
        "ema50_dist": (c - e50) / atr5, "ema200_dist": (c - e200) / atr5,
        "bb_pos": (c - bb_lo) / np.where(bb_hi > bb_lo, bb_hi - bb_lo, np.nan),
        "bb_width_bps": (bb_hi - bb_lo) / c * 1e4,
        "vwap_z": (c - vwap) / np.where(vsd > 0, vsd, np.nan),
        "prev_high_dist": (c - ph) / atr5, "prev_low_dist": (c - pl) / atr5,
        "vol_ratio": vr,
        "body": (c - o) / rng, "lower_wick": (np.minimum(o, c) - l) / rng, "upper_wick": (h - np.maximum(o, c)) / rng,
        "hour": (ts % 86400) // 3600, "weekday": ((ts // 86400) + 3) % 7,  # 0 = segunda
    }
    ctx = {m: higher(ts, o, h, l, c, v, m) for m in (15, 60, 240)}
    for m, vals in ctx.items():
        for k, a in vals.items():
            if k != "atr":
                cols[f"h{m}_{k}"] = a
    atr_h1 = ctx[60]["atr"]
    cols["atr_h1_bps"] = atr_h1 / c * 1e4
    names = list(cols)
    x = np.column_stack([cols[k] for k in names]).astype(float)
    x[~np.isfinite(x)] = np.nan

    up = c > o
    cand = {
        # Pullback na tendência de 1 h: toca a EMA 21 de 5 min e fecha acima dela, candle de alta.
        "pullback": (ctx[60]["trend"] > 0) & (l <= e21) & (c > e21) & up,
        # Volta para dentro da 2ª banda inferior da VWAP do dia.
        "vwap_low": (l <= vwap - 2 * vsd) & (c > vwap - 2 * vsd) & up,
        # Sobrevendido no 5 min e candle de alta.
        "rsi_oversold": (lagged(r5, 1) < 30) & up,
        # Toque na banda de Bollinger inferior com fechamento de volta para dentro.
        "bb_low": (l <= bb_lo) & (c > bb_lo) & up,
        # Varre a mínima do dia anterior e recupera.
        "prev_low_reclaim": (l < pl) & (c > pl),
        # Rompe a máxima do dia anterior com volume forte.
        "breakout": (lagged(c, 1) <= ph) & (c > ph) & (vr >= 1.5),
        # Pavio inferior longo (rejeição) com tendência de 4 h para cima.
        "rejection": (ctx[240]["trend"] > 0) & ((np.minimum(o, c) - l) / rng >= 0.6),
    }
    cand = {k: np.nan_to_num(m, nan=0).astype(bool) for k, m in cand.items()}
    return Frame(ts, o, h, l, c, atr_h1, x, names, cand)


# ---------- ciclo com DCA ----------

@dataclass(frozen=True)
class Cycle:
    tp: float = 0.75  # alvo em ATR de 1 h acima do preço médio
    dca: float = 0.5  # distância entre entradas, em ATR de 1 h
    sl: float = 2.5  # stop em ATR de 1 h abaixo da entrada 1 (abaixo das duas ordens de DCA)
    max_bars: int = 144  # 12 h

    @property
    def name(self) -> str:
        return f"tp{self.tp:g}-dca{self.dca:g}-sl{self.sl:g}-t{self.max_bars}"


@dataclass(frozen=True, slots=True)
class Result:
    i: int  # candle do sinal
    exit_i: int  # candle da saída
    outcome: str  # tp | sl | time
    fills: int  # entradas executadas (1 a 3)
    ret_bps: float  # líquido, em bps do capital do ciclo (3 partes)
    gross_bps: float


def simulate_cycle(f: Frame, i: int, cy: Cycle, maker: float = MAKER_FEE, taker: float = TAKER_FEE) -> Result | None:
    """Ciclo a partir do sinal no fechamento do candle i; None se faltar ATR de 1 h, candles ou houver buraco."""
    a = f.atr_h1[i]
    n = len(f.ts)
    if not (a > 0) or i + 1 + cy.max_bars >= n or f.ts[i + 1 + cy.max_bars] - f.ts[i] != (cy.max_bars + 1) * BAR:
        return None
    e1 = f.open[i + 1]
    levels = [e1 - cy.dca * a, e1 - 2 * cy.dca * a]
    stop = e1 - cy.sl * a
    fills = [e1]
    fee = taker  # a entrada 1, em fração de uma parte
    for k in range(i + 1, i + 1 + cy.max_bars):
        lo, hi, op = f.low[k], f.high[k], f.open[k]
        bought = False
        while len(fills) < 3 and lo <= levels[len(fills) - 1]:
            px = min(levels[len(fills) - 1], op)
            fills.append(px)
            fee += maker
            bought = True
        avg = sum(fills) / len(fills)
        if lo <= stop:
            px = min(stop, op)
            return _close(i, k, "sl", fills, px, fee + taker * len(fills))
        target = avg + cy.tp * a
        if not bought and hi >= target:  # abriu acima do alvo: a ordem limitada sai na abertura
            return _close(i, k, "tp", fills, max(target, op), fee + maker * len(fills))
    k = i + cy.max_bars
    return _close(i, k, "time", fills, f.close[k], fee + taker * len(fills))


def _close(i: int, k: int, outcome: str, fills: list[float], px: float, fees_parts: float) -> Result:
    gross = sum(px / e - 1 for e in fills) / 3  # cada parte é 1/3 do capital; parte não usada não rende
    net = gross - fees_parts / 3
    return Result(i, k, outcome, len(fills), net * 1e4, gross * 1e4)


# ---------- juiz e walk-forward ----------

CONFIGS = tuple(Cycle(tp, dca, sl) for tp in (0.5, 0.75, 1.0) for dca in (0.5, 1.0) for sl in (2.5, 4.0))
WEEK, DAY = 7 * 86400, 86400
FIT_WEEKS = 45  # das 60 de treino: o juiz aprende nas primeiras 45 e o corte sai das 15 seguintes
MIN_PER_DAY = 2
FORCE_HOUR = 16  # UTC: daqui em diante, dia com menos de 2 ciclos aceita candidato com nota menor
CUT_LEVELS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99)
LGBM_ROUNDS = 300
LGBM_PARAMS = dict(
    objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=100, feature_fraction=0.8,
    bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, seed=0, num_threads=1, verbose=-1,
)
JUIZ, FORCADO, TODOS, QUALQUER = "juiz", "juiz + mínimo de 2/dia", "todos os gatilhos", "qualquer candle"


def model_matrix(f: Frame) -> tuple[np.ndarray, list[str]]:
    """Variáveis do candle mais uma coluna 0/1 por gatilho (um candle pode disparar vários)."""
    flags = np.column_stack([f.candidates[k] for k in f.candidates]).astype(float)
    return np.hstack([f.x, flags]), f.names + [f"gatilho_{k}" for k in f.candidates]


def any_candidate(f: Frame) -> np.ndarray:
    m = np.zeros(len(f.ts), dtype=bool)
    for v in f.candidates.values():
        m |= v
    return m


def label(f: Frame, idx, cy: Cycle, end: int, maker: float, taker: float) -> dict[int, Result]:
    """Ciclo de cada candidato que termina antes do candle `end` (o primeiro das semanas intocadas)."""
    out = {}
    for i in idx:
        if i + 1 + cy.max_bars < end and (r := simulate_cycle(f, int(i), cy, maker, taker)) is not None:
            out[int(i)] = r
    return out


def sequence(idx, result, take, day_of, hour_of=None, force=None) -> list[Result]:
    """Um ciclo por vez, em ordem de tempo: candidato com o ciclo anterior ainda aberto é ignorado.

    `take(i)` é a regra normal. `force(i)` só vale depois de FORCE_HOUR, em dia com menos de MIN_PER_DAY ciclos.
    O sinal no fechamento do candle em que o ciclo anterior saiu já pode entrar (a entrada é no candle seguinte).
    """
    out, busy, opened = [], -1, {}
    for i in idx:
        if i < busy:
            continue
        ok = take(i)
        if not ok and force is not None:
            ok = opened.get(day_of[i], 0) < MIN_PER_DAY and hour_of[i] >= FORCE_HOUR and force(i)
        if not ok or (r := result(i)) is None:
            continue
        out.append(r)
        busy = r.exit_i
        opened[day_of[i]] = opened.get(day_of[i], 0) + 1
    return out


def choose_cut(p: dict[int, float], idx: list[int], result, day_of, n_days: int) -> tuple[float, bool]:
    """Corte de nota para o trecho de calibração: o de maior lucro com acerto ≥ 80% + folga e ≥ 2 ciclos/dia.

    Sem corte que cumpra os dois, fica o de maior acerto entre os que dão ≥ 2 ciclos/dia, e a rodada é marcada
    como fora da meta (False).
    """
    tried = []
    for t in sorted(set(np.quantile([p[i] for i in idx], CUT_LEVELS))):
        rs = sequence(idx, result, lambda i: p[i] >= t, day_of)
        if rs:
            r = np.array([x.ret_bps for x in rs])
            tried.append((t, float((r > 0).mean()), len(rs) / n_days, float(r.sum())))
    ok = [x for x in tried if x[1] >= MIN_WIN_RATE + CUT_MARGIN and x[2] >= MIN_PER_DAY]
    if ok:
        return max(ok, key=lambda x: x[3])[0], True
    enough = [x for x in tried if x[2] >= MIN_PER_DAY] or tried
    if not enough:
        return math.inf, False
    return max(enough, key=lambda x: (x[1], x[3]))[0], False


def fit_lgbm(x: np.ndarray, y: np.ndarray):
    import lightgbm

    return lightgbm.train(LGBM_PARAMS, lightgbm.Dataset(x, y.astype(float)), num_boost_round=LGBM_ROUNDS)


@dataclass
class ConfigRun:
    cycle: Cycle
    rows: dict[str, list[Result]]  # regra → ciclos nas semanas de teste, todas as rodadas
    test_weeks: list[int]
    test_days: int
    folds_met: int
    folds: int
    importance: list[tuple[str, float]]


def walk_forward(f: Frame, x: np.ndarray, names: list[str], cy: Cycle, maker: float, taker: float,
                 holdout_weeks: int = HOLDOUT_WEEKS) -> ConfigRun:
    week = (f.ts - f.ts[0]) // WEEK
    dev_weeks = int(week[-1]) + 1 - holdout_weeks
    end = int(np.searchsorted(week, dev_weeks))
    labels = label(f, np.flatnonzero(any_candidate(f)[:end]), cy, end, maker, taker)
    day_of, hour_of = f.ts // DAY, (f.ts % DAY) // 3600
    cache: dict[int, Result | None] = {}

    def any_bar(i: int) -> Result | None:  # "qualquer candle": simula sob demanda
        if i not in cache:
            cache[i] = simulate_cycle(f, i, cy, maker, taker) if i + 1 + cy.max_bars < end else None
        return cache[i]

    rows: dict[str, list[Result]] = {}
    test_weeks, test_days, met, n_folds, importance = [], 0, 0, 0, []
    for s in range(0, dev_weeks - TRAIN_WEEKS - TEST_WEEKS + 1, TEST_WEEKS):
        cal_lo, te_lo, te_hi = s + FIT_WEEKS, s + TRAIN_WEEKS, s + TRAIN_WEEKS + TEST_WEEKS
        fit = [i for i, r in labels.items() if s <= week[i] < cal_lo and week[r.exit_i] < cal_lo]
        cal = [i for i, r in labels.items() if cal_lo <= week[i] < te_lo and week[r.exit_i] < te_lo]
        test = [i for i in labels if te_lo <= week[i] < te_hi]
        if len(fit) < 500 or not cal:
            continue
        booster = fit_lgbm(x[fit], np.array([labels[i].ret_bps > 0 for i in fit]))
        p_cal = dict(zip(cal, booster.predict(x[cal])))
        cal_days = len(np.unique(day_of[(week >= cal_lo) & (week < te_lo)]))
        cut, ok = choose_cut(p_cal, cal, labels.get, day_of, cal_days)
        low = float(np.median(list(p_cal.values())))
        p = dict(zip(test, booster.predict(x[test]))) if test else {}

        add = lambda name, rs: rows.setdefault(name, []).extend(rs)  # noqa: E731
        add(JUIZ, sequence(test, labels.get, lambda i: p[i] >= cut, day_of))
        add(FORCADO, sequence(test, labels.get, lambda i: p[i] >= cut, day_of, hour_of, lambda i: p[i] >= low))
        add(TODOS, sequence(test, labels.get, lambda i: True, day_of))
        for k, m in f.candidates.items():
            add(k, sequence([i for i in test if m[i]], labels.get, lambda i: True, day_of))
        lo, hi = int(np.searchsorted(week, te_lo)), int(np.searchsorted(week, te_hi))
        add(QUALQUER, sequence(range(lo, hi), any_bar, lambda i: True, day_of))

        test_weeks += list(range(te_lo, te_hi))
        test_days += len(np.unique(day_of[lo:hi]))
        met += ok
        n_folds += 1
        gain = booster.feature_importance("gain")
        importance = sorted(zip(names, gain / gain.sum()), key=lambda r: -r[1])
    return ConfigRun(cy, rows, test_weeks, test_days, met, n_folds, importance)


def stats(rs: list[Result], f: Frame, test_weeks: list[int], test_days: int) -> dict:
    r = np.array([x.ret_bps for x in rs])
    n = len(r)
    if not n:
        return {"n": 0}
    cum = np.concatenate([[0.0], np.cumsum(r)])
    week = (f.ts - f.ts[0]) // WEEK
    by_week = {w: 0.0 for w in test_weeks}
    by_day: dict[int, int] = {}
    for x in rs:
        by_week[int(week[x.i])] = by_week.get(int(week[x.i]), 0.0) + x.ret_bps
        d = int(f.ts[x.i] // DAY)
        by_day[d] = by_day.get(d, 0) + 1
    wins, losses = r[r > 0], r[r <= 0]
    return {
        "n": n, "per_day": n / test_days, "days2": sum(v >= MIN_PER_DAY for v in by_day.values()) / test_days,
        "win": len(wins) / n, "avg_win": wins.mean() if len(wins) else 0.0,
        "avg_loss": losses.mean() if len(losses) else 0.0, "mean": r.mean(), "total": r.sum(),
        "worst": r.min(), "dd": float((np.maximum.accumulate(cum) - cum).max()),
        "stops": sum(x.outcome == "sl" for x in rs) / n,
        "weeks_pos": sum(v > 0 for v in by_week.values()), "weekly": [by_week[w] for w in test_weeks],
    }


def format_config(run: ConfigRun, f: Frame) -> str:
    head = (f"{'regra':<24}{'ciclos':>7}{'/dia':>6}{'dias≥2':>8}{'acerto':>8}{'ganho':>8}{'perda':>8}"
            f"{'líq/ciclo':>10}{'total':>9}{'pior':>8}{'queda':>8}{'stops':>7}{'sem+':>7}")
    lines = [f"\n{run.cycle.name}: {run.folds} rodadas, juiz dentro da meta na calibração em {run.folds_met}",
             head]
    order = [JUIZ, FORCADO, TODOS, *f.candidates, QUALQUER]
    for name in order:
        s = stats(run.rows.get(name, []), f, run.test_weeks, run.test_days)
        if not s["n"]:
            lines.append(f"{name:<24}{0:>7}")
            continue
        lines.append(
            f"{name:<24}{s['n']:>7}{s['per_day']:>6.1f}{s['days2']:>8.0%}{s['win']:>8.1%}{s['avg_win']:>8.1f}"
            f"{s['avg_loss']:>8.1f}{s['mean']:>10.2f}{s['total']:>9.0f}{s['worst']:>8.0f}{s['dd']:>8.0f}"
            f"{s['stops']:>7.1%}{s['weeks_pos']:>4}/{len(run.test_weeks)}"
        )
    return "\n".join(lines)


_F: Frame | None = None
_X: np.ndarray | None = None
_NAMES: list[str] = []
_FEES: tuple[float, float] = (MAKER_FEE, TAKER_FEE)
_HOLDOUT = HOLDOUT_WEEKS


def _init(f: Frame, x: np.ndarray, names: list[str], fees: tuple[float, float], holdout: int) -> None:
    global _F, _X, _NAMES, _FEES, _HOLDOUT
    _F, _X, _NAMES, _FEES, _HOLDOUT = f, x, names, fees, holdout


def _work(cy: Cycle) -> ConfigRun:
    return walk_forward(_F, _X, _NAMES, cy, *_FEES, _HOLDOUT)


def run(args) -> None:
    from concurrent.futures import ProcessPoolExecutor

    from .validation import deflated_sharpe, sharpe

    candles = load_candles(candle_path(data_dir=args.data))
    if not candles:
        raise SystemExit("sem candles; rode antes: python -m wescalp.brain fetch")
    f = build(candles)
    x, names = model_matrix(f)
    week = (f.ts - f.ts[0]) // WEEK
    dev_end = int(np.searchsorted(week, int(week[-1]) + 1 - args.holdout))
    fmt = lambda t: datetime.fromtimestamp(int(t), timezone.utc).strftime("%Y-%m-%d")  # noqa: E731
    print(f"{FUTURES_SYMBOL} 5 min: {len(candles)} candles de {fmt(f.ts[0])} a {fmt(f.ts[-1])};"
          f" intocadas as últimas {args.holdout} semanas (desde {fmt(f.ts[dev_end])})")
    print(f"taxas: maker {args.maker:.2%}, taker {args.taker:.2%}; ATR de 1 h mediano"
          f" {np.nanmedian(f.atr_h1[:dev_end] / f.close[:dev_end] * 1e4):.0f} bps")
    print("gatilhos no período de pesquisa: " + ", ".join(
        f"{k} {int(m[:dev_end].sum())}" for k, m in f.candidates.items()))
    print("ganho/perda/líq/total/pior/queda em bps do capital do ciclo (as 3 partes); queda = maior queda do"
          " acumulado; sem+ = semanas de teste no positivo; acerto = ciclos com líquido > 0")
    print(f"gerado em {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC", flush=True)

    with ProcessPoolExecutor(args.jobs, initializer=_init,
                             initargs=(f, x, names, (args.maker, args.taker), args.holdout)) as pool:
        runs = list(pool.map(_work, CONFIGS))
    for r in runs:
        print(format_config(r, f), flush=True)

    weekly = {r.cycle.name: stats(r.rows.get(JUIZ, []), f, r.test_weeks, r.test_days) for r in runs}
    trials = [sharpe(s["weekly"]) for s in weekly.values() if s["n"]]
    best = max((r for r in runs if weekly[r.cycle.name]["n"]), key=lambda r: weekly[r.cycle.name]["total"],
               default=None)
    if best is not None:
        s = weekly[best.cycle.name]
        print(f"\nmelhor juiz por total: {best.cycle.name}, acerto {s['win']:.1%}, {s['per_day']:.1f} ciclos/dia,"
              f" total {s['total']:.0f} bps, Sharpe semanal {sharpe(s['weekly']):.2f},"
              f" DSR entre {len(trials)} configurações {deflated_sharpe(s['weekly'], trials):.3f}")
        print("importância no último juiz dessa configuração: "
              + ", ".join(f"{n} {g:.0%}" for n, g in best.importance[:8]))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=CANDLE_DIR, help="pasta dos candles")
    sub = p.add_subparsers(dest="cmd", required=True)
    fe = sub.add_parser("fetch", help="baixa ou completa os candles de 5 min do perpétuo")
    fe.add_argument("--weeks", type=int, default=WEEKS)
    r = sub.add_parser("run", help="walk-forward do juiz e relatório")
    r.add_argument("--maker", type=float, default=MAKER_FEE, help="taxa maker (fração)")
    r.add_argument("--taker", type=float, default=TAKER_FEE, help="taxa taker (fração)")
    r.add_argument("--holdout", type=int, default=HOLDOUT_WEEKS, help="últimas semanas intocadas")
    r.add_argument("--jobs", type=int, default=4, help="configurações em paralelo")
    args = p.parse_args()
    if args.cmd == "fetch":
        cs = asyncio.run(fetch(FUTURES_SYMBOL, args.weeks, candle_path(data_dir=args.data)))
        gaps = sum(1 for a, b in zip(cs, cs[1:]) if (b.ts - a.ts) != timedelta(seconds=BAR))
        log.info("%d candles de %s a %s, %d buracos", len(cs), cs[0].ts, cs[-1].ts, gaps)
    else:
        run(args)


if __name__ == "__main__":
    main()
