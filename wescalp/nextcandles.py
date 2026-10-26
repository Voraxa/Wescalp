"""Quanto dá para acertar dos próximos 20 candles de 5 min com LightGBM e redes neurais, usando tudo o que temos.

    python -m wescalp.nextcandles --file data/candles/BINANCE-BTCUSDT-5m.csv.gz    # pip install -e '.[ml,bayes]'

Fixado antes de rodar (26/10/2026), sem ajuste depois.

Dados: candles de 5 min do BTCUSDT spot da Binance; as séries de wescalp.flowdata (fluxo agressor do spot e do
perpétuo, prêmio, funding e, desde 12/2021, posições abertas e relações compradas/vendidas); e os candles de 5 min do
ETH (desde 2021). No fechamento de cada candle t, de 01/01/2020 a 30/06/2026, para k = 1 a 20, duas perguntas:
- o candle k: o candle t+k fecha acima do fechamento anterior?
- daqui a k candles: o fechamento do candle t+k fica acima do de agora?
Para k = 1 as duas são a mesma. Candle que fechou igual não conta no acerto.

Armas, todas só com o que já tinha acontecido no fechamento do candle t (56 variáveis, em VARIABLES): retornos de 1 a
288 candles e dos 2 candles anteriores; corpo, pavios e amplitude do candle; volatilidade de 12, 48 e 288 candles;
distância às EMAs de 9, 21, 50 e 200 candles; RSI de 14 e de 2; %b de Bollinger; distância à VWAP do dia; volume e
negócios relativos; fluxo agressor do spot e do perpétuo; base do perpétuo, prêmio e funding; posições abertas e
relações compradas/vendidas (5 min de atraso); hora, minuto e dia da semana; a posição da congelada, a distância à
média de 50 dias e a volatilidade diária; retornos do ETH.

Modelos, com parâmetros fixos:
- LightGBM: um por pergunta e horizonte (39), binário, taxa 0,05, 31 folhas, mínimo de 2.000 por folha, até 400
  árvores com parada antecipada (30) na validação;
- rede densa: 56 → 128 → 64 → 40 saídas (as duas perguntas nos 20 horizontes), GELU, AdamW (taxa 0,001,
  decaimento 0,0001), lotes de 2.048, até 8 épocas, parada com paciência 2 na validação;
- rede com o gráfico: os últimos 48 candles em 8 canais (retorno, corpo, pavio de cima, pavio de baixo, volume
  relativo, fluxo agressor do spot e do perpétuo, base do perpétuo) passam por 3 convoluções (24 filtros, janela 5,
  dilatação 1, 2 e 4) e se juntam às 56 variáveis (64 neurônios) numa camada de 64; o mesmo treino da rede densa;
- todas juntas: a média das probabilidades dos três;
- referências: chutar sempre a direção mais comum no treino; repetir o último candle; inverter o último candle.

Walk-forward: cada ano de 2021 a 2026 (até 30/06) é testado com modelos treinados de 01/01/2020 até o ano anterior.
Os últimos 10% do treino são a validação, e há 20 candles de intervalo entre treino, validação e teste, para o alvo
mais longo não vazar.

Medidas, fora da amostra, juntando os anos:
- acerto: fração das previsões na direção certa;
- acerto nos 10% mais confiantes: |p - 0,5| acima do corte dos 10% de cima na validação;
- acerto necessário para empatar com a ida e volta, na pergunta "daqui a k candles": (custo + queda média) / (alta
  média + queda média) no teste, com 0,2% (spot) e 0,04% (ordem limitada nos futuros);
- útil para operar: as compras nos 10% de previsão mais alta (corte da validação), segurando k candles, com retorno
  médio acima do custo e t ≥ 3, contando n/k casos independentes (os alvos se sobrepõem); com 0,2% e com 0,04%.
Mostra também o acerto de cada ano (k = 1, 5 e 20) e as 10 variáveis que mais pesaram no LightGBM em k = 1.

Nota depois da rodada (26/10/2026, sem mudar o método): a régua de "útil para operar" conferiu o t do retorno bruto,
e não o da sobra depois do custo. O LightGBM passou com 0,04% em k = 16, 19 e 20 (ganho médio de 0,041% a 0,050%),
mas a sobra, de 0,001% a 0,010% por operação, tem t abaixo de 1.
"""

import argparse
import math
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import altcoins, brain, flowdata, intraday, refine, regime, study

STEP, H, DAY = 300, 3600, 86400
K = 20
GRID_START = intraday.utc("2019-01-01")  # aquecimento das variáveis
START, END = intraday.utc("2020-01-01"), intraday.utc("2026-07-01")
TEST_YEARS = tuple(range(2021, 2027))
EMBARGO = K
VALID_SHARE = 0.10
TOP = 0.10
COSTS = (0.002, 0.0004)
MIN_T = 3.0
METRIC_LAG = intraday.METRIC_LAG
SEQ = 48
DILATIONS = (1, 2, 4)
WIDTH = 24
BATCH = 2048
EPOCHS = 8
PATIENCE = 2
LR, DECAY = 1e-3, 1e-4
LGBM_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=2000, feature_fraction=0.8,
                   bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, max_bin=63, seed=0, verbose=-1, num_threads=4)
LGBM_ROUNDS, LGBM_STOP = 400, 30
VARIABLES = (
    *(f"retorno {n}" for n in (1, 2, 3, 6, 12, 24, 48, 96, 288)), "retorno do candle -1", "retorno do candle -2",
    "corpo", "pavio de cima", "pavio de baixo", "amplitude", "vol 12", "vol 48", "vol 288", "vol 12 / vol 288",
    *(f"distância EMA {n}" for n in (9, 21, 50, 200)), "RSI 14", "RSI 2", "%b Bollinger 20", "distância VWAP do dia",
    "volume relativo 1", "volume relativo 12", "negócios relativos 1", "fluxo spot 1", "fluxo spot 12",
    "fluxo spot 48", "fluxo perp 1", "fluxo perp 12", "fluxo perp 288", "base do perpétuo", "prêmio", "prêmio Δ12",
    "funding", "posições Δ12", "posições Δ288", "contas compradas", "grandes traders", "relação agressora",
    "hora sen", "hora cos", "dia da semana sen", "dia da semana cos", "minuto sen", "minuto cos", "congelada",
    "distância média 50 d", "vol diária 30 d", "ETH retorno 1", "ETH retorno 12")
CHANNELS = ("retorno", "corpo", "pavio de cima", "pavio de baixo", "volume relativo", "fluxo spot", "fluxo perp",
            "base do perpétuo")
TARGETS = (*(f"candle {k}" for k in range(1, K + 1)), *(f"daqui a {k}" for k in range(1, K + 1)))
MODELS = ("LightGBM", "rede densa", "rede com o gráfico", "todas juntas")


@dataclass
class Data:
    grid: np.ndarray  # abertura de cada candle de 5 min, sem buracos
    ok: np.ndarray  # o candle existe na Binance
    x: np.ndarray  # (n, 56) variáveis
    ch: np.ndarray  # (n, 8) canais da rede com o gráfico
    y: np.ndarray  # (n, 40) retornos-alvo (log), NaN quando falta candle
    first_ret: np.ndarray  # retorno do candle t (para as referências)


# ---------------------------------------------------------------- variáveis


def ffill(x: np.ndarray) -> np.ndarray:
    """Repete o último valor conhecido; antes do primeiro, NaN."""
    ok = np.isfinite(x)
    idx = np.where(ok, np.arange(len(x)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = x[idx]
    if ok.any():
        out[:int(np.argmax(ok))] = np.nan
    return out


def roll_sum(x: np.ndarray, n: int) -> np.ndarray:
    c = np.concatenate([[0.0], np.cumsum(x)])
    out = np.full(len(x), np.nan)
    out[n - 1:] = c[n:] - c[:-n]
    return out


def roll_std(x: np.ndarray, n: int) -> np.ndarray:
    m, m2 = roll_sum(x, n) / n, roll_sum(x * x, n) / n
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


def lag(x: np.ndarray, k: int) -> np.ndarray:
    return brain.lagged(x, k) if k else x


def zero_nan(x: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(x), x, 0.0)


def ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return a / b


def daily_context(grid: np.ndarray, d: regime.Daily) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(posição da congelada, média de 50 dias, volatilidade diária de 30 dias) do último dia fechado na decisão."""
    pos = refine.position(d.close, refine.BASE)
    sma50 = regime.sma(d.close, 50)
    lr = np.concatenate([[np.nan], np.diff(np.log(d.close))])
    vol = np.full(len(d.close), np.nan)
    vol[30:] = roll_std(lr[1:], 30)[29:]
    last_day = (grid + STEP) // DAY - 1  # o dia fechado mais recente quando o candle t fecha
    i = np.searchsorted(d.day, last_day, side="right") - 1
    ok = (i >= 0) & (d.day[np.maximum(i, 0)] == last_day)
    take = lambda v: np.where(ok, v[np.maximum(i, 0)], np.nan)  # noqa: E731
    return take(pos), take(sma50), take(vol)


def build(ts, o, h, l, c, flow: dict, perp: dict, premium: dict, funding: dict, metrics: dict, eth: tuple | None,
          start: int = GRID_START, end: int = END) -> Data:
    """Põe tudo numa grade de 5 min sem buracos e calcula variáveis, canais e alvos."""
    grid = np.arange(start, end, STEP, dtype=np.int64)
    raw = intraday.at(ts, c, grid)
    ok = np.isfinite(raw)
    close = ffill(raw)
    assert ok[0], "a grade tem de começar num candle que existe"
    get = lambda col: np.where(ok, intraday.at(ts, col, grid), close)  # noqa: E731  candle faltando: parado no último
    op, hi, lo = get(o), get(h), get(l)
    at = lambda s, col: intraday.at(s["ts"], s[col], grid)  # noqa: E731
    vol, buy, trades = (zero_nan(at(flow, k)) for k in ("volume", "taker_buy", "trades"))
    pv, pbuy, pclose = zero_nan(at(perp, "volume")), zero_nan(at(perp, "taker_buy")), at(perp, "close")
    prem = at(premium, "close")
    decision = grid + STEP
    known = lambda col, lag_s=0: intraday.last_known(metrics["ts"] + METRIC_LAG, metrics[col], decision - lag_s, H)  # noqa: E731

    lc = np.log(close)
    r = np.concatenate([[np.nan], np.diff(lc)])
    r0 = zero_nan(r)
    ema = {n: brain.ema(close, n) for n in (9, 21, 50, 200)}
    sma20, sd20 = roll_sum(close, 20) / 20, roll_std(close, 20)
    day = grid // DAY
    first = np.searchsorted(day, day)  # primeiro candle do mesmo dia
    tp = (hi + lo + close) / 3
    cpv, cv = np.concatenate([[0.0], np.cumsum(tp * vol)]), np.concatenate([[0.0], np.cumsum(vol)])
    vwap = ratio(cpv[1:] - cpv[first], cv[1:] - cv[first])
    vol288, tr288 = roll_sum(vol, 288) / 288, roll_sum(trades, 288) / 288
    flow_n = lambda b, v, n: ratio(roll_sum(b, n), roll_sum(v, n)) - 0.5  # noqa: E731
    v12, v288 = roll_std(r0, 12), roll_std(r0, 288)
    dctx = daily_context(grid, regime.daily_bars(ts, o, h, l, c))
    hour = (decision // H) % 24
    minute = (decision // STEP) % 12
    weekday = (decision // DAY + 3) % 7
    oi = known("oi")
    if eth is not None:
        eraw = intraday.at(eth[0], eth[4], grid)
        eclose = ffill(eraw)
        eth1, eth12 = np.log(eclose / lag(eclose, 1)), np.log(eclose / lag(eclose, 12))
    else:
        eth1 = eth12 = np.full(len(grid), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        cols = [
            *(lc - lag(lc, n) for n in (1, 2, 3, 6, 12, 24, 48, 96, 288)), lag(r, 1), lag(r, 2),
            np.log(close / op), np.log(hi / np.maximum(op, close)), np.log(np.minimum(op, close) / lo), np.log(hi / lo),
            v12, roll_std(r0, 48), v288, ratio(v12, v288),
            *(np.log(close / ema[n]) for n in (9, 21, 50, 200)), brain.rsi(close, 14), brain.rsi(close, 2),
            ratio(close - (sma20 - 2 * sd20), 4 * sd20), np.log(close / vwap),
            np.log1p(ratio(vol, vol288)), np.log1p(ratio(roll_sum(vol, 12) / 12, vol288)), np.log1p(ratio(trades, tr288)),
            flow_n(buy, vol, 1), flow_n(buy, vol, 12), flow_n(buy, vol, 48),
            flow_n(pbuy, pv, 1), flow_n(pbuy, pv, 12), flow_n(pbuy, pv, 288),
            np.log(pclose / close), prem, prem - lag(prem, 12),
            intraday.last_known(funding["ts"] + 1, funding["rate"], decision, 9 * H),
            np.log(oi / known("oi", H)), np.log(oi / known("oi", 24 * H)), np.log(known("accounts")),
            np.log(known("top_positions")), np.log(known("taker_ratio")),
            np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * weekday / 7), np.cos(2 * np.pi * weekday / 7),
            np.sin(2 * np.pi * minute / 12), np.cos(2 * np.pi * minute / 12),
            dctx[0], np.log(close / dctx[1]), dctx[2], eth1, eth12,
        ]
        ch = [r, np.log(close / op), np.log(hi / np.maximum(op, close)), np.log(np.minimum(op, close) / lo),
              np.log1p(ratio(vol, vol288)), flow_n(buy, vol, 1), flow_n(pbuy, pv, 1), np.log(pclose / close)]
    x = np.column_stack([np.where(np.isfinite(v), v, np.nan) for v in cols]).astype(np.float32)
    assert x.shape[1] == len(VARIABLES)
    lr_raw = np.log(raw)
    y = np.full((len(grid), 2 * K), np.nan)
    for k in range(1, K + 1):
        y[:-k, k - 1] = lr_raw[k:] - lr_raw[k - 1:-1]  # o candle t+k
        y[:-k, K + k - 1] = lr_raw[k:] - lr_raw[:-k]  # daqui a k candles
    return Data(grid, ok, x, np.column_stack(ch).astype(np.float32), y.astype(np.float32), r.astype(np.float32))


def load_all(file: Path, flow_dir: Path, candle_dir: Path) -> Data:
    ts, o, h, l, c = study.load(file)
    flows = {k: flowdata.load(k, flow_dir) for k in ("spot-flow", "perp", "premium", "funding", "metrics")}
    try:
        e = altcoins.load("ETHUSDT", candle_dir)
        eth = (e[0], e[1], e[2], e[3], e[4])
    except FileNotFoundError:
        eth = None
    return build(ts, o, h, l, c, flows["spot-flow"], flows["perp"], flows["premium"], flows["funding"],
                 flows["metrics"], eth)


# ---------------------------------------------------------------- dobras


@dataclass(frozen=True)
class Fold:
    year: int
    train: np.ndarray  # índices da grade
    valid: np.ndarray
    test: np.ndarray


def folds(d: Data, years=TEST_YEARS, start: int = START, end: int = END) -> list[Fold]:
    out = []
    usable = d.ok & (d.grid >= start) & (d.grid < end)
    for y in years:
        a, b = intraday.utc(f"{y}-01-01"), min(intraday.utc(f"{y + 1}-01-01"), end)
        test = np.flatnonzero(usable & (d.grid >= a) & (d.grid < b))
        before = np.flatnonzero(usable & (d.grid < a - EMBARGO * STEP))
        cut = int(len(before) * (1 - VALID_SHARE))
        train, valid = before[:cut - EMBARGO], before[cut:]
        out.append(Fold(y, train, valid, test))
    return out


# ---------------------------------------------------------------- modelos


def labels(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(1 se subiu, peso 1 se o alvo existe e não é zero)."""
    ok = np.isfinite(y) & (y != 0)
    return (zero_nan(y) > 0).astype(np.float32), ok.astype(np.float32)


def lgbm_fold(d: Data, f: Fold) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], np.ndarray]:
    """Probabilidades na validação e no teste de cada alvo (0..39) e o ganho de cada variável em k = 1."""
    import lightgbm as lgb

    warnings.filterwarnings("ignore", message="Overriding the parameters")
    base = lgb.Dataset(d.x[f.train], free_raw_data=False, params=LGBM_PARAMS).construct()
    va = lgb.Dataset(d.x[f.valid], reference=base, free_raw_data=False, params=LGBM_PARAMS).construct()
    p_va, p_te, gain = {}, {}, None
    for j in range(2 * K):
        if j == K:  # "daqui a 1" é o mesmo que "candle 1"
            p_va[j], p_te[j] = p_va[0], p_te[0]
            continue
        (ytr, wtr), (yva, wva) = labels(d.y[f.train, j]), labels(d.y[f.valid, j])
        base.set_label(ytr)
        base.set_weight(wtr)
        va.set_label(yva)
        va.set_weight(wva)
        b = lgb.train(LGBM_PARAMS, base, LGBM_ROUNDS, valid_sets=[va], callbacks=[lgb.early_stopping(LGBM_STOP, verbose=False)])
        p_va[j], p_te[j] = b.predict(d.x[f.valid]), b.predict(d.x[f.test])
        if j == 0:
            gain = b.feature_importance("gain")
    return p_va, p_te, gain


def standardize(train: np.ndarray, *others: np.ndarray) -> list[np.ndarray]:
    mean, std = np.nanmean(train, axis=0), np.nanstd(train, axis=0)
    std[~(std > 0)] = 1.0
    return [np.clip(zero_nan((a - mean) / std), -10, 10).astype(np.float32) for a in (train, *others)]


def nn_fold(d: Data, f: Fold, kind: str, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Treina a rede (densa ou com o gráfico) nas 40 saídas e devolve as probabilidades na validação e no teste."""
    import jax
    import jax.numpy as jnp

    x_tr, x_va, x_te = standardize(d.x[f.train], d.x[f.valid], d.x[f.test])
    xs = np.zeros((len(d.grid), d.x.shape[1]), np.float32)
    xs[f.train], xs[f.valid], xs[f.test] = x_tr, x_va, x_te
    ch = standardize(d.ch[f.train], d.ch)[1]
    off = np.arange(-SEQ + 1, 1)
    seq = (lambda rows: ch[rows[:, None] + off]) if kind == "gráfico" else (lambda rows: np.zeros((len(rows), 1, 1), np.float32))
    yl, ym = labels(d.y)
    yl[:, K], ym[:, K] = yl[:, 0], ym[:, 0]

    key = jax.random.PRNGKey(seed)

    def dense(key, n_in, n_out):
        lim = math.sqrt(6 / (n_in + n_out))
        return [jax.random.uniform(key, (n_in, n_out), minval=-lim, maxval=lim), jnp.zeros(n_out)]

    keys = jax.random.split(key, 8)
    nx = d.x.shape[1]
    if kind == "densa":
        params = {"layers": [dense(keys[0], nx, 128), dense(keys[1], 128, 64), dense(keys[2], 64, 2 * K)]}
    else:
        conv, cin = [], len(CHANNELS)
        for i in range(len(DILATIONS)):
            lim = math.sqrt(6 / (5 * cin + 5 * WIDTH))
            conv.append([jax.random.uniform(keys[i], (5, cin, WIDTH), minval=-lim, maxval=lim), jnp.zeros(WIDTH)])
            cin = WIDTH
        params = {"conv": conv, "feat": dense(keys[4], nx, 64), "mix": dense(keys[5], 2 * WIDTH + 64, 64),
                  "out": dense(keys[6], 64, 2 * K)}

    def forward(p, x, s):
        if kind == "densa":
            z = x
            for w, b in p["layers"][:-1]:
                z = jax.nn.gelu(z @ w + b)
            w, b = p["layers"][-1]
            return z @ w + b
        z = s
        for (w, b), dil in zip(p["conv"], DILATIONS):
            z = jax.nn.gelu(jax.lax.conv_general_dilated(z, w, (1,), "VALID", rhs_dilation=(dil,),
                                                         dimension_numbers=("NWC", "WIO", "NWC")) + b)
        z = jnp.concatenate([z.mean(axis=1), z[:, -1, :], jax.nn.gelu(x @ p["feat"][0] + p["feat"][1])], axis=1)
        z = jax.nn.gelu(z @ p["mix"][0] + p["mix"][1])
        return z @ p["out"][0] + p["out"][1]

    def loss(p, x, s, y, m):
        z = forward(p, x, s)
        return ((jax.nn.softplus(z) - y * z) * m).sum() / jnp.maximum(m.sum(), 1.0)

    @jax.jit
    def step(p, opt, x, s, y, m):
        g = jax.grad(loss)(p, x, s, y, m)
        mu, nu, t = opt
        t = t + 1
        mu = jax.tree_util.tree_map(lambda a, b: 0.9 * a + 0.1 * b, mu, g)
        nu = jax.tree_util.tree_map(lambda a, b: 0.999 * a + 0.001 * b * b, nu, g)
        p = jax.tree_util.tree_map(
            lambda w, a, b: w - LR * (a / (1 - 0.9 ** t) / (jnp.sqrt(b / (1 - 0.999 ** t)) + 1e-8) + DECAY * w), p, mu, nu)
        return p, (mu, nu, t)

    batch_loss = jax.jit(loss)
    prob = jax.jit(lambda p, x, s: jax.nn.sigmoid(forward(p, x, s)))

    def predict(p, rows):
        return np.concatenate([np.asarray(prob(p, xs[rows[i:i + 8192]], seq(rows[i:i + 8192])))
                               for i in range(0, len(rows), 8192)])

    def valid_loss(p):
        tot = n = 0.0
        for i in range(0, len(f.valid), 8192):
            r = f.valid[i:i + 8192]
            w = float(ym[r].sum())
            tot += float(batch_loss(p, xs[r], seq(r), yl[r], ym[r])) * w
            n += w
        return tot / n

    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    opt = (zeros, zeros, 0)
    rng = np.random.default_rng(seed)
    best, best_p, bad = math.inf, params, 0
    for epoch in range(EPOCHS):
        perm = f.train[rng.permutation(len(f.train))]
        for i in range(0, len(perm) - BATCH + 1, BATCH):
            r = np.sort(perm[i:i + BATCH])
            params, opt = step(params, opt, xs[r], seq(r), yl[r], ym[r])
        v = valid_loss(params)
        print(f"    {kind} {f.year} época {epoch + 1}: perda na validação {v:.5f}", file=sys.stderr, flush=True)
        if v < best - 1e-6:
            best, best_p, bad = v, params, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    return predict(best_p, f.valid), predict(best_p, f.test)


# ---------------------------------------------------------------- medidas


def accuracy(p: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    ok = np.isfinite(y) & (y != 0)
    return float(((p[ok] >= 0.5) == (y[ok] > 0)).mean()) if ok.any() else math.nan, int(ok.sum())


def confident(p_va: np.ndarray, p_te: np.ndarray, share: float = TOP) -> np.ndarray:
    """Previsões do teste com |p - 0,5| acima do corte dos `share` mais confiantes da validação."""
    return np.abs(p_te - 0.5) >= np.quantile(np.abs(p_va - 0.5), 1 - share)


def needed(y: np.ndarray, cost: float) -> float:
    """Acerto de direção necessário para empatar com o custo da ida e volta."""
    y = y[np.isfinite(y) & (y != 0)]
    up, dn = y[y > 0].mean(), -y[y < 0].mean()
    return (cost + dn) / (up + dn)


def trade_stats(r: np.ndarray, k: int) -> dict:
    """Retorno médio das compras que seguram k candles, com t contando n/k casos independentes."""
    n = len(r)
    mean = float(r.mean()) if n else math.nan
    t = mean / (r.std(ddof=1) / math.sqrt(max(n / k, 1))) if n > 2 and r.std() > 0 else math.nan
    return {"n": n, "mean": mean, "t": t, **{c: bool(n > 2 and mean > c and t >= MIN_T) for c in COSTS}}


def evaluate(d: Data, fs: list[Fold], preds: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]) -> dict:
    """Junta os anos de teste e mede cada modelo em cada alvo."""
    test = np.concatenate([f.test for f in fs])
    out = {}
    for m, by_year in preds.items():
        rows = []
        for j in range(2 * K):
            p_va = [by_year[f.year][0][:, j] for f in fs]
            p_te = np.concatenate([by_year[f.year][1][:, j] for f in fs])
            y = d.y[test, j]
            acc, n = accuracy(p_te, y)
            conf = np.concatenate([confident(pv, by_year[f.year][1][:, j]) for f, pv in zip(fs, p_va)])
            acc_top, n_top = accuracy(p_te[conf], y[conf])
            buys = None
            if j >= K:  # compras nos 10% de previsão mais alta, com o corte de cada ano tirado da validação
                picks = np.concatenate([by_year[f.year][1][:, j] >= np.quantile(pv, 1 - TOP) for f, pv in zip(fs, p_va)])
                buys = trade_stats(np.expm1(y[picks & np.isfinite(y)]), j - K + 1)
            rows.append({"acc": acc, "n": n, "acc_top": acc_top, "n_top": n_top, "buys": buys})
        out[m] = rows
    return out


def references(d: Data, fs: list[Fold]) -> dict[str, list[float]]:
    """Acerto de chutar a direção mais comum do treino, de repetir e de inverter o último candle."""
    test = np.concatenate([f.test for f in fs])
    last = d.first_ret[test]
    out = {"chute": [], "repetir o último": [], "inverter o último": []}
    for j in range(2 * K):
        guess = np.concatenate([np.full(len(f.test), float((d.y[f.train, j][np.isfinite(d.y[f.train, j])] > 0).mean() >= 0.5))
                                for f in fs])
        y = d.y[test, j]
        out["chute"].append(accuracy(guess, y)[0])
        out["repetir o último"].append(accuracy((last > 0).astype(float), y)[0])
        out["inverter o último"].append(accuracy((last < 0).astype(float), y)[0])
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", type=Path, required=True)
    p.add_argument("--flow", type=Path, default=flowdata.DATA_DIR)
    p.add_argument("--data", type=Path, default=brain.CANDLE_DIR, help="pasta dos candles do ETH")
    p.add_argument("--years", type=int, nargs="*", default=list(TEST_YEARS))
    args = p.parse_args()
    t0 = time.time()
    d = load_all(args.file, args.flow, args.data)
    fs = folds(d, tuple(args.years))
    print(f"grade pronta em {time.time() - t0:.0f} s: {len(d.grid)} candles, {d.x.shape[1]} variáveis", file=sys.stderr)
    preds: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {m: {} for m in MODELS}
    gains = []
    for f in fs:
        print(f"{f.year}: treino {len(f.train)}, validação {len(f.valid)}, teste {len(f.test)}", file=sys.stderr, flush=True)
        pv, pt, g = lgbm_fold(d, f)
        preds["LightGBM"][f.year] = (np.column_stack([pv[j] for j in range(2 * K)]),
                                     np.column_stack([pt[j] for j in range(2 * K)]))
        gains.append(g / g.sum())
        print(f"  LightGBM pronto ({time.time() - t0:.0f} s)", file=sys.stderr, flush=True)
        preds["rede densa"][f.year] = nn_fold(d, f, "densa")
        preds["rede com o gráfico"][f.year] = nn_fold(d, f, "gráfico")
        preds["todas juntas"][f.year] = tuple(np.mean([preds[m][f.year][i] for m in MODELS[:3]], axis=0)
                                              for i in (0, 1))
        print(f"  redes prontas ({time.time() - t0:.0f} s)", file=sys.stderr, flush=True)
    res = evaluate(d, fs, preds)
    ref = references(d, fs)
    test = np.concatenate([f.test for f in fs])
    print(f"Fora da amostra: {args.years[0]} a {args.years[-1]} (2026 até 30/06), {len(test)} candles de 5 min\n")
    print(f"{'k':>3}  {'modelo':<20}{'candle k':>10}{'daqui a k':>11}{'10% mais confiantes':>21}"
          f"{'chute':>8}{'repetir':>9}{'inverter':>10}{'necessário 0,2%':>17}{'0,04%':>8}")
    for k in range(1, K + 1):
        ya = d.y[test, K + k - 1]
        need = [needed(ya, c) for c in COSTS]
        for m in MODELS:
            ra, rb = res[m][k - 1], res[m][K + k - 1]
            extra = (f"{ref['chute'][K + k - 1]:>8.2%}{ref['repetir o último'][K + k - 1]:>9.2%}"
                     f"{ref['inverter o último'][K + k - 1]:>10.2%}"
                     + "".join(f"{(f'{x:.1%}' if x <= 1 else 'impossível'):>{w}}" for x, w in zip(need, (17, 8)))
                     if m == MODELS[0] else "")
            print(f"{k:>3}  {m:<20}{ra['acc']:>10.2%}{rb['acc']:>11.2%}{rb['acc_top']:>14.2%} ({rb['n_top'] / rb['n']:.0%})"
                  + extra)
    print("\nÚtil para operar (compras nos 10% de previsão mais alta, segurando k candles; t com n/k casos):")
    for m in MODELS:
        for k in range(1, K + 1):
            b = res[m][K + k - 1]["buys"]
            flags = ", ".join(f"paga {c:.2%}" for c in COSTS if b[c]) or "não paga"
            print(f"  {m:<20} k={k:>2}: {b['n']:>7} compras, média {b['mean']:+.3%}, t {b['t']:.1f}: {flags}")
    print("\nAcerto de cada ano, todas juntas, 'daqui a k candles':")
    for f in fs:
        line = []
        for k in (1, 5, 20):
            j = K + k - 1
            line.append(f"k={k}: {accuracy(preds['todas juntas'][f.year][1][:, j], d.y[f.test, j])[0]:.2%}")
        print(f"  {f.year}: " + ", ".join(line))
    g = np.mean(gains, axis=0)
    print("\nAs 10 variáveis que mais pesaram no LightGBM (candle 1, ganho médio nos anos):")
    for i in np.argsort(-g)[:10]:
        print(f"  {VARIABLES[i]}: {g[i]:.1%}")
    print(f"\n({time.time() - t0:.0f} s)", file=sys.stderr)


if __name__ == "__main__":
    main()
