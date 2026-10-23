"""Teste de informação do day trade: algum sinal prevê o BTC nas próximas horas mais do que a taxa?

    python -m wescalp.flowdata                                                 # antes: fluxo, prêmio, posições
    python -m wescalp.intraday --file data/candles/BINANCE-BTCUSDT-5m.csv.gz

Fixado antes de rodar (23/10/2026), sem ajuste depois:
- decisão de hora em hora, no fechamento do candle de 5 min que termina na hora cheia, de 01/01/2020 a 30/06/2026;
- alvo: retorno do BTC spot nas 1, 4 e 12 h seguintes; em cada horizonte, só decisões sem sobreposição (de 1 em 1 h,
  de 4 em 4 h, de 12 em 12 h);
- sinais, só com o que já tinha acontecido na decisão (métricas com 5 min de atraso, funding já liquidado):
  retorno de 1 h e de 24 h, volume da última hora sobre a média dos 30 dias, fluxo agressor do spot e do perpétuo
  (compra agressora / volume - 0,5) em 1 h e do perpétuo em 24 h, prêmio do perpétuo e a variação dele em 1 h,
  funding, variação das posições abertas em 1 h e em 24 h, relação compradas/vendidas das contas e das posições dos
  grandes traders (log) e "liquidação de comprados" (queda do preço × queda das posições na última hora);
- dentro da amostra (até 31/12/2022; até 31/12/2023 para os sinais de posições, que começam em 12/2021) saem os cortes
  dos decis e o lado (decil de cima ou de baixo) de maior retorno médio; fora da amostra (2023 ou 2024 a 06/2026)
  mede-se o retorno médio desse decil;
- passa o sinal cujo decil escolhido, fora da amostra, render em média mais que 0,2% (a ida e volta no spot) com
  t ≥ 3 (o limiar de Harvey, Liu e Zhu para muitos testes);
- confluência: ridge e LightGBM (parâmetros fixos) com todos os sinais e a hora do dia, treinados de 12/2021 a 12/2023;
  o mesmo teste no decil de cima da previsão (corte tirado do treino), de 2024 a 06/2026.
"""

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import flowdata, study

SIGNALS = ("retorno 1h", "retorno 24h", "volume relativo 1h", "fluxo spot 1h", "fluxo perp 1h", "fluxo perp 24h",
           "prêmio", "prêmio Δ1h", "funding", "posições Δ1h", "posições Δ24h", "contas compradas", "grandes traders",
           "liquidação de comprados")
POSITION_SIGNALS = frozenset(SIGNALS[9:])
HORIZONS = (1, 4, 12)
COST = 0.002
MIN_T = 3.0
H = 3600


def utc(s: str) -> int:
    return int(datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp())


START, START_POS = utc("2020-01-01"), utc("2021-12-01")
SPLIT, SPLIT_POS = utc("2023-01-01"), utc("2024-01-01")
END = utc("2026-07-01")  # exclusivo: o alvo precisa terminar até 30/06/2026
METRIC_LAG = 300
LGBM_PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=15, min_data_in_leaf=500,
                   feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, seed=0, verbose=-1)
LGBM_ROUNDS = 300
RIDGE_SHRINK = 0.01


def at(ts: np.ndarray, values: np.ndarray, when: np.ndarray) -> np.ndarray:
    """Valor da barra que abre exatamente em `when` (NaN se ela não existe)."""
    i = np.searchsorted(ts, when)
    ok = i < len(ts)
    ok[ok] = ts[i[ok]] == when[ok]
    out = np.full(len(when), np.nan)
    out[ok] = values[i[ok]]
    return out


def last_known(ts: np.ndarray, values: np.ndarray, when: np.ndarray, max_age: int) -> np.ndarray:
    """Último valor com tempo ≤ `when`, se tiver no máximo `max_age` segundos; senão NaN."""
    i = np.searchsorted(ts, when, side="right") - 1
    ok = i >= 0
    ok[ok] = when[ok] - ts[i[ok]] <= max_age
    out = np.full(len(when), np.nan)
    out[ok] = values[i[ok]]
    return out


def window_sum(ts: np.ndarray, values: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Soma dos valores das barras que abrem em [start, end)."""
    c = np.concatenate([[0.0], np.cumsum(values)])
    return c[np.searchsorted(ts, end)] - c[np.searchsorted(ts, start)]


def signals(decision: np.ndarray, spot: dict, flow: dict, perp: dict, premium: dict, funding: dict,
            metrics: dict) -> dict[str, np.ndarray]:
    """Cada sinal em cada decisão (instante da hora cheia), só com o passado."""
    price = lambda t: at(spot["ts"], spot["close"], t - 300)  # noqa: E731  o candle que fecha em t
    p0 = price(decision)
    ret1, ret24 = np.log(p0 / price(decision - H)), np.log(p0 / price(decision - 24 * H))
    vol1 = window_sum(flow["ts"], flow["volume"], decision - H, decision)
    vol30 = window_sum(flow["ts"], flow["volume"], decision - 30 * 24 * H, decision) / (30 * 24)

    def aggressor(s, hours):
        v = window_sum(s["ts"], s["volume"], decision - hours * H, decision)
        with np.errstate(invalid="ignore", divide="ignore"):
            return window_sum(s["ts"], s["taker_buy"], decision - hours * H, decision) / v - 0.5

    prem = lambda t: at(premium["ts"], premium["close"], t - 300)  # noqa: E731
    known = lambda col, t: last_known(metrics["ts"] + METRIC_LAG, metrics[col], t, H)  # noqa: E731
    oi = known("oi", decision)
    with np.errstate(invalid="ignore", divide="ignore"):
        oi1, oi24 = np.log(oi / known("oi", decision - H)), np.log(oi / known("oi", decision - 24 * H))
        out = {
            "retorno 1h": ret1, "retorno 24h": ret24, "volume relativo 1h": np.log(vol1 / vol30),
            "fluxo spot 1h": aggressor(flow, 1), "fluxo perp 1h": aggressor(perp, 1), "fluxo perp 24h": aggressor(perp, 24),
            "prêmio": prem(decision), "prêmio Δ1h": prem(decision) - prem(decision - H),
            "funding": last_known(funding["ts"] + 1, funding["rate"], decision, 9 * H),  # liquidado antes da decisão
            "posições Δ1h": oi1, "posições Δ24h": oi24,
            "contas compradas": np.log(known("accounts", decision)), "grandes traders": np.log(known("top_positions", decision)),
            "liquidação de comprados": np.maximum(0, -ret1) * np.maximum(0, -oi1),
        }
    return {k: np.where(np.isfinite(v), v, np.nan) for k, v in out.items()}


def forward(spot: dict, decision: np.ndarray, hours: int) -> np.ndarray:
    price = lambda t: at(spot["ts"], spot["close"], t - 300)  # noqa: E731
    return price(decision + hours * H) / price(decision) - 1


def summary(y: np.ndarray, side: str, in_mean: float, base: float) -> dict:
    """Retorno médio, t e acerto das decisões escolhidas fora da amostra; passa acima da taxa com t ≥ MIN_T."""
    n = len(y)
    mean = float(y.mean()) if n else math.nan
    t = mean / (y.std(ddof=1) / math.sqrt(n)) if n > 2 and y.std() > 0 else math.nan
    return {"side": side, "in": float(in_mean), "out": mean, "n": n, "t": t,
            "hit": float((y > 0).mean()) if n else math.nan, "base": float(base),
            "passes": bool(n > 2 and mean > COST and t >= MIN_T)}


def tail_test(s_in, y_in, s_out, y_out) -> dict:
    """Decil escolhido dentro da amostra (o lado de maior média) e o retorno dele fora da amostra."""
    ok_in, ok_out = np.isfinite(s_in) & np.isfinite(y_in), np.isfinite(s_out) & np.isfinite(y_out)
    s_in, y_in, s_out, y_out = s_in[ok_in], y_in[ok_in], s_out[ok_out], y_out[ok_out]
    lo, hi = np.quantile(s_in, [0.1, 0.9])
    top, bottom = y_in[s_in >= hi].mean(), y_in[s_in <= lo].mean()
    side = "cima" if top >= bottom else "baixo"
    pick = s_out >= hi if side == "cima" else s_out <= lo
    return summary(y_out[pick], side, max(top, bottom), y_out.mean())


def features(sig: dict[str, np.ndarray], decision: np.ndarray) -> np.ndarray:
    hour = (decision // H) % 24
    return np.column_stack([sig[k] for k in SIGNALS] + [np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)])


def ridge_fit(x: np.ndarray, y: np.ndarray):
    mean, std = np.nanmean(x, axis=0), np.nanstd(x, axis=0)
    std[std == 0] = 1
    z = np.nan_to_num((x - mean) / std)
    beta = np.linalg.solve(z.T @ z + RIDGE_SHRINK * len(y) * np.eye(z.shape[1]), z.T @ (y - y.mean()))
    return lambda q: y.mean() + np.nan_to_num((q - mean) / std) @ beta


def lgbm_fit(x: np.ndarray, y: np.ndarray):
    import lightgbm as lgb

    booster = lgb.train(LGBM_PARAMS, lgb.Dataset(x, y), LGBM_ROUNDS)
    return booster.predict


def predict(fit, x_in, y_in, x_out, y_out):
    """Treina com as linhas de treino que têm alvo e devolve (previsão no treino, alvo, previsão fora, alvo fora)."""
    ok_in, ok_out = np.isfinite(y_in), np.isfinite(y_out)
    model = fit(x_in[ok_in], y_in[ok_in])
    return model(x_in[ok_in]), y_in[ok_in], model(x_out[ok_out]), y_out[ok_out]


def top_test(p_in, y_in, p_out, y_out) -> dict:
    """Só o decil de cima da previsão (a carteira só compra), com o corte tirado do treino."""
    hi = np.quantile(p_in, 0.9)
    return summary(y_out[p_out >= hi], "cima", y_in[p_in >= hi].mean(), y_out.mean())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", type=Path, required=True)
    p.add_argument("--data", type=Path, default=flowdata.DATA_DIR)
    args = p.parse_args()
    ts, o, h, l, c = study.load(args.file)
    spot = {"ts": ts, "close": c}
    flow, perp, premium, funding, metrics = (flowdata.load(k, args.data) for k in
                                             ("spot-flow", "perp", "premium", "funding", "metrics"))
    decision = np.arange(START, END, H, dtype=np.int64)
    sig = signals(decision, spot, flow, perp, premium, funding, metrics)
    x = features(sig, decision)

    print(f"decisões de hora em hora de {datetime.fromtimestamp(START, timezone.utc):%Y-%m-%d} a 30/06/2026;"
          f" passa quem render mais que {COST:.1%} no decil escolhido fora da amostra, com t ≥ {MIN_T:.0f}")
    print(f"{'sinal':<26}{'h':>4}{'lado':>7}{'dentro':>9}{'fora':>9}{'base':>8}{'acerto':>8}{'n':>7}{'t':>7}  passa")
    passed = []
    for hours in HORIZONS:
        y = forward(spot, decision, hours)
        step = (decision // H) % hours == 0
        rows = {}
        for k in SIGNALS:
            split, start = (SPLIT_POS, START_POS) if k in POSITION_SIGNALS else (SPLIT, START)
            i_in, i_out = step & (decision >= start) & (decision < split), step & (decision >= split)
            rows[k] = tail_test(sig[k][i_in], y[i_in], sig[k][i_out], y[i_out])
        i_in = (decision >= START_POS) & (decision < SPLIT_POS)  # o treino usa todas as horas
        i_out = step & (decision >= SPLIT_POS)
        for name, fit in (("confluência ridge", ridge_fit), ("confluência LightGBM", lgbm_fit)):
            rows[name] = top_test(*predict(fit, x[i_in], y[i_in], x[i_out], y[i_out]))
        for k, r in rows.items():
            print(f"{k:<26}{hours:>3}h{r['side']:>7}{r['in']:>9.3%}{r['out']:>9.3%}{r['base']:>8.3%}{r['hit']:>8.0%}"
                  f"{r['n']:>7}{r['t']:>7.1f}  {'SIM' if r['passes'] else 'não'}")
            if r["passes"]:
                passed.append(f"{k} ({hours} h)")
        print()
    print(f"passam: {', '.join(passed) if passed else 'nenhum'}")


if __name__ == "__main__":
    main()
