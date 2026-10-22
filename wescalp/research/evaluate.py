"""Avaliação dos setups: walk-forward por semanas, métricas por trade, posterior Beta e Sharpe deflacionado."""

import json
import math
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import stats

from .labels import CostModel, triple_barrier

SAIDA = ("tp_atr", "sl_atr", "max_bars")
EULER_GAMMA = 0.5772156649015329


class WalkForward(NamedTuple):
    trades: pd.DataFrame  # trades fora da amostra, com colunas `fold` e `params`
    folds: pd.DataFrame  # uma linha por dobra, inclusive as puladas
    n_trials: int  # combinações distintas testadas (para o Sharpe deflacionado)
    trial_sharpes: pd.Series  # sharpe_trade de cada combinação na amostra toda, índice = params


def _chave(params: dict) -> str:
    return json.dumps(params, sort_keys=True)


def _vazio(trades: pd.DataFrame) -> pd.DataFrame:
    return trades.iloc[0:0]


def metrics(trades: pd.DataFrame, dias: float | None = None) -> dict:
    """Métricas por trade. `dias` é o tamanho da janela para `trades_por_dia`; sem ele, usa o
    intervalo entre a primeira e a última entrada (mínimo de 1 dia)."""
    r = trades["ret_net"].astype(float).to_numpy() if len(trades) else np.array([])
    n = len(r)
    if n == 0:
        return {"n": 0, "win_rate": np.nan, "avg_net_bps": np.nan, "median_net_bps": np.nan,
                "profit_factor": np.nan, "sharpe_trade": np.nan, "max_dd": 0.0, "trades_por_dia": 0.0}
    ganho, perda = r[r > 0].sum(), -r[r < 0].sum()
    pf = ganho / perda if perda > 0 else (np.inf if ganho > 0 else np.nan)
    sd = r.std(ddof=1) if n > 1 else np.nan
    sharpe = r.mean() / sd if n > 1 and sd > 0 else np.nan
    acum = np.concatenate([[0.0], np.cumsum(r)])
    max_dd = float((np.maximum.accumulate(acum) - acum).max())
    if dias is None:
        ts = pd.to_datetime(trades["entry_ts"])
        dias = max((ts.max() - ts.min()) / pd.Timedelta(days=1), 1.0)
    return {
        "n": n,
        "win_rate": float((r > 0).mean()),
        "avg_net_bps": float(r.mean() * 1e4),
        "median_net_bps": float(np.median(r) * 1e4),
        "profit_factor": float(pf),
        "sharpe_trade": float(sharpe),
        "max_dd": max_dd,
        "trades_por_dia": n / dias if dias > 0 else np.nan,
    }


def beta_posterior(wins: int, n: int, prior_wins: float = 1, prior_n: float = 2) -> tuple[float, float, float]:
    """Posterior Beta(prior_wins + wins, prior_n - prior_wins + n - wins) da taxa de acerto:
    (média, p05, p95). O padrão é a priori uniforme Beta(1, 1)."""
    a = prior_wins + wins
    b = (prior_n - prior_wins) + (n - wins)
    d = stats.beta(a, b)
    return float(d.mean()), float(d.ppf(0.05)), float(d.ppf(0.95))


def expected_max_sharpe(n_trials: int, sr_var_trials: float) -> float:
    """E[max SR] de `n_trials` estratégias sem habilidade (SR verdadeiro 0), aproximação de
    Bailey & López de Prado com a constante de Euler–Mascheroni γ:

        SR0 = sqrt(V) · [(1 − γ) · Φ⁻¹(1 − 1/N) + γ · Φ⁻¹(1 − 1/(N·e))]
    """
    if n_trials <= 1 or not sr_var_trials > 0:
        return 0.0
    z1 = stats.norm.ppf(1 - 1 / n_trials)
    z2 = stats.norm.ppf(1 - 1 / (n_trials * math.e))
    return float(math.sqrt(sr_var_trials) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe(sr: float, n_obs: int, n_trials: int, skew: float, kurt: float,
                    sr_var_trials: float) -> float:
    """Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

    Probabilidade de o Sharpe verdadeiro superar o máximo esperado por acaso entre `n_trials`
    tentativas:

        DSR = Φ( (SR − SR0) · sqrt(T − 1) / sqrt(1 − γ3·SR + (γ4 − 1)/4 · SR²) )

    com SR0 = `expected_max_sharpe(n_trials, sr_var_trials)`. Tudo na mesma unidade (por trade, não
    anualizado): `sr` é o Sharpe observado, T = `n_obs`, γ3 = `skew`, γ4 = `kurt` (curtose NÃO
    excedente: normal = 3), `sr_var_trials` é a variância dos Sharpes das tentativas. Com
    `n_trials` = 1 vira o PSR contra zero.
    """
    if n_obs < 2 or not np.isfinite(sr):
        return float("nan")
    sr0 = expected_max_sharpe(n_trials, sr_var_trials)
    den = 1 - skew * sr + (kurt - 1) / 4 * sr**2
    if den <= 0:
        return float("nan")
    return float(stats.norm.cdf((sr - sr0) * math.sqrt(n_obs - 1) / math.sqrt(den)))


def breakdown(trades: pd.DataFrame, by: str) -> pd.DataFrame:
    """Métricas por grupo (`regime`, `hour`, ...), com o intervalo Beta da taxa de acerto."""
    linhas = {}
    for k, g in trades.groupby(by, sort=True):
        m = metrics(g)
        _, p05, p95 = beta_posterior(int((g["ret_net"] > 0).sum()), len(g))
        linhas[k] = {**m, "wr_p05": p05, "wr_p95": p95}
    out = pd.DataFrame.from_dict(linhas, orient="index")
    out.index.name = by
    return out


def _janelas(idx: pd.DatetimeIndex, train_weeks: int, test_weeks: int, step_weeks: int):
    # semanas de calendário começando na segunda 00:00 UTC
    inicio = idx.min().normalize() - pd.Timedelta(days=idx.min().weekday())
    fim = idx.max()
    sem = pd.Timedelta(weeks=1)
    k = 0
    while True:
        t0 = inicio + (train_weeks + k * step_weeks) * sem
        if t0 > fim:
            return
        yield k, t0 - train_weeks * sem, t0, t0 + test_weeks * sem
        k += 1


def walk_forward(df: pd.DataFrame, setup, train_weeks: int = 60, test_weeks: int = 4,
                 step_weeks: int = 4, min_trades: int = 30, cost: CostModel | None = None) -> WalkForward:
    """Walk-forward por semanas de calendário. Os sinais são causais, então cada combinação do grid
    roda uma vez na série toda e os trades são fatiados por `entry_ts`:

    - treino [train_start, test_start): só trades que também SAEM antes de test_start (o resultado
      não pode usar barras do teste);
    - teste [test_start, test_end): trades que entram na janela, com a combinação de maior `ret_net`
      médio no treino entre as que têm ≥ `min_trades` (empate: a primeira do grid). Sem nenhuma, a
      dobra é pulada (`status` = "sem_combo").

    Uma posição por vez também entre dobras: trade do teste que entra antes da saída do último
    trade fora da amostra já aceito é descartado.
    """
    if "atr14" not in df.columns:
        from .features import add_features
        df = add_features(df)
    cost = cost or CostModel()

    # sinais uma vez por conjunto de parâmetros de sinal; trades uma vez por combinação
    combos: dict[str, tuple[dict, pd.DataFrame]] = {}
    sinais: dict[str, pd.Series] = {}
    for p in setup.grid:
        chave = _chave(p)
        if chave in combos:
            continue
        sig_p = {k: v for k, v in p.items() if k not in SAIDA}
        sk = _chave(sig_p)
        if sk not in sinais:
            sinais[sk] = setup.fn(df, **sig_p)
        t = triple_barrier(df, sinais[sk], tp_atr=p["tp_atr"], sl_atr=p["sl_atr"],
                           max_bars=int(p["max_bars"]), cost=cost)
        t = t.sort_values("entry_ts").reset_index(drop=True)
        combos[chave] = (p, t)

    trial_sharpes = pd.Series({k: metrics(t)["sharpe_trade"] for k, (_, t) in combos.items()},
                              name="sharpe_trade", dtype=float)

    oos, linhas = [], []
    ultima_saida = None
    for fold, tr0, te0, te1 in _janelas(df.index, train_weeks, test_weeks, step_weeks):
        linha = {"fold": fold, "train_start": tr0, "train_end": te0, "test_start": te0, "test_end": te1}
        melhor = None
        n_validas = 0
        for chave, (p, t) in combos.items():
            tr = t[(t["entry_ts"] >= tr0) & (t["entry_ts"] < te0) & (t["exit_ts"] < te0)]
            if len(tr) < min_trades:
                continue
            n_validas += 1
            media = tr["ret_net"].mean()
            if melhor is None or media > melhor[0]:
                melhor = (media, chave, tr)
        linha["n_combos_validas"] = n_validas
        if melhor is None:
            linhas.append({**linha, "status": "sem_combo", "params": None})
            continue

        _, chave, tr = melhor
        t = combos[chave][1]
        te = t[(t["entry_ts"] >= te0) & (t["entry_ts"] < te1)]
        if ultima_saida is not None:
            te = te[te["entry_ts"] > ultima_saida]
        if len(te):
            ultima_saida = te["exit_ts"].max()
            oos.append(te.assign(fold=fold, params=chave))

        dias_treino = (te0 - max(tr0, df.index.min())) / pd.Timedelta(days=1)
        dias_teste = (min(te1, df.index.max()) - te0) / pd.Timedelta(days=1)
        linha.update({"status": "ok", "params": chave})
        linha.update({f"train_{k}": v for k, v in metrics(tr, dias_treino).items()})
        linha.update({f"test_{k}": v for k, v in metrics(te, max(dias_teste, 1 / 288)).items()})
        linhas.append(linha)

    if oos:
        trades = pd.concat(oos, ignore_index=True)
    else:
        base = next(iter(combos.values()))[1] if combos else pd.DataFrame(columns=["entry_ts", "exit_ts", "ret_net"])
        trades = _vazio(base).assign(fold=pd.Series(dtype=int), params=pd.Series(dtype=object))
    return WalkForward(trades, pd.DataFrame(linhas), len(combos), trial_sharpes)
