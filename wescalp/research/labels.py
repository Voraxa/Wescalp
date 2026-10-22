"""Rótulo por tripla barreira (alvo, stop, tempo) com custos da KuCoin, uma posição por vez."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

COLUNAS = ["entry_ts", "exit_ts", "side", "entry_px", "exit_px", "outcome", "bars",
           "ret_gross", "cost", "ret_net", "regime", "hour"]


@dataclass(frozen=True)
class CostModel:
    maker: float = 0.0002
    taker: float = 0.0006
    slippage: float = 0.0001  # por lado, só em ordem a mercado

    def total(self, outcome: str) -> float:
        # entrada sempre limitada (maker); alvo sai maker, stop/tempo sai a mercado
        if outcome == "tp":
            return 2 * self.maker
        return self.maker + self.taker + self.slippage


def triple_barrier(df: pd.DataFrame, signals: pd.Series, tp_atr: float, sl_atr: float, max_bars: int,
                   cost: CostModel = CostModel()) -> pd.DataFrame:
    n = len(df)
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    atr = df["atr14"].to_numpy(float)
    sig = signals.reindex(df.index).fillna(0).to_numpy()

    # candidatos: sinal ≠ 0, ATR válido e existe barra seguinte
    cand = np.flatnonzero((sig != 0) & np.isfinite(atr) & (atr > 0))
    cand = cand[cand < n - 1]

    linhas = []
    livre = 0  # primeiro índice de sinal aceito (barra de saída do trade anterior)
    k = 0
    while k < len(cand):
        t = cand[k]
        if t < livre:
            # pula direto para o próximo sinal depois da saída
            k = int(np.searchsorted(cand, livre))
            continue
        lado = 1 if sig[t] > 0 else -1
        e = t + 1
        px = o[e]
        fim = min(e + max_bars, n)
        hh, ll = h[e:fim], l[e:fim]
        if lado > 0:
            alvo, stop = px + tp_atr * atr[t], px - sl_atr * atr[t]
            toca_tp, toca_sl = hh >= alvo, ll <= stop
        else:
            alvo, stop = px - tp_atr * atr[t], px + sl_atr * atr[t]
            toca_tp, toca_sl = ll <= alvo, hh >= stop
        toque = toca_tp | toca_sl
        if toque.any():
            j = int(toque.argmax())
            x = e + j
            if toca_sl[j]:  # alvo e stop na mesma barra conta stop
                res = "sl"
                # se a barra abre além do stop, sai na abertura
                saida = min(stop, o[x]) if lado > 0 else max(stop, o[x])
            else:
                res, saida = "tp", alvo
        elif fim - e == max_bars:
            x = fim - 1
            res, saida = "time", c[x]
        else:
            break  # dados acabam antes do fim da janela: trade incompleto, descartado

        bruto = lado * (saida / px - 1)
        custo = cost.total(res)
        linhas.append((t, e, x, lado, px, saida, res, x - e + 1, bruto, custo))
        livre = x  # um sinal no fechamento da barra de saída já pode abrir outro trade
        k += 1

    if not linhas:
        return pd.DataFrame({k: pd.Series(dtype=object) for k in COLUNAS})

    t, e, x, lado, px, saida, res, barras, bruto, custo = map(np.array, zip(*linhas))
    idx = df.index
    out = pd.DataFrame({
        "entry_ts": idx[e],
        "exit_ts": idx[x],
        "side": lado.astype(int),
        "entry_px": px.astype(float),
        "exit_px": saida.astype(float),
        "outcome": res.astype(object),
        "bars": barras.astype(int),
        "ret_gross": bruto.astype(float),
        "cost": custo.astype(float),
    })
    out["ret_net"] = out["ret_gross"] - out["cost"]
    out["regime"] = df["regime"].to_numpy()[t] if "regime" in df else None
    out["hour"] = df["hour"].to_numpy()[t] if "hour" in df else idx[t].hour
    return out
