"""Setups da fase 1: funções de sinal sobre a saída de `add_features` e o registro `SETUPS`.

Cada função devolve uma Series no índice de `df` com valores em {-1, 0, +1}. O sinal na linha `t`
usa só colunas calculadas até o fechamento de `t` (e, quando precisa, a linha `t-1`); a entrada
acontece na abertura de `t+1` (ver `labels.triple_barrier`).

Escolha das grades (candles de 5 min de BTC):
- O `atr14` de 5 min do BTC costuma ficar em ~0,1–0,25% do preço. A ida e volta custa ~0,04% (alvo,
  maker nos dois lados) a ~0,09% (stop/tempo, taker + slippage na saída). Alvo de 0,5 ATR (~0,05–0,1%)
  some nos custos, por isso `tp_atr` começa em 1,0 ATR e vai até 2,0.
- `sl_atr` entre 0,75 e 1,5 ATR: abaixo disso o ruído de uma barra de 5 min derruba o stop à toa
  (e o stop paga taker); acima, a perda média fica grande demais para um alvo de scalp.
- `max_bars` entre 6 e 24 barras (30 min a 2 h): rompimento e reversão à VWAP se resolvem rápido
  (6–18), pullback a favor da tendência pode precisar de mais tempo (12–24).
- Cada grade tem 24 combinações (teto do contrato ~30), para não inflar o número de tentativas
  que o Sharpe deflacionado penaliza.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from itertools import product

import pandas as pd


@dataclass(frozen=True)
class Setup:
    name: str
    fn: Callable[..., pd.Series]
    grid: list[dict] = field(default_factory=list)


def _sinal(df: pd.DataFrame, compra: pd.Series, venda: pd.Series, horas: Iterable[int] | None) -> pd.Series:
    # Comparações com NaN dão False, então o aquecimento sai como 0.
    s = pd.Series(0, index=df.index, dtype="int64")
    s[compra.fillna(False).astype(bool)] = 1
    s[venda.fillna(False).astype(bool)] = -1
    if horas is not None:
        s[~df["hour"].isin(list(horas))] = 0
    return s


def rompimento(df: pd.DataFrame, vol_z_min: float = 1.0, horas: Iterable[int] | None = None, **_) -> pd.Series:
    """Close acima de `hh20` (compra) / abaixo de `ll20` (venda) com `vol_z` >= `vol_z_min`."""
    volume_ok = df["vol_z"] >= vol_z_min
    return _sinal(df, (df["close"] > df["hh20"]) & volume_ok, (df["close"] < df["ll20"]) & volume_ok, horas)


def vwap_reversao(df: pd.DataFrame, k: float = 2.0, horas: Iterable[int] | None = None, **_) -> pd.Series:
    """Barra anterior esticada além de ±k desvios da VWAP e a barra atual fecha de volta para dentro.

    Venda: `vwap_z[t-1] >= k`, `close[t] < close[t-1]` e o close ainda acima da VWAP (há caminho até ela).
    Compra: espelhado. Na primeira hora do dia UTC a VWAP tem poucas barras e o `vwap_z` explode; fica de fora.
    """
    z, z_ant = df["vwap_z"], df["vwap_z"].shift(1)
    close, close_ant = df["close"], df["close"].shift(1)
    dia_formado = df["hour"] >= 1
    venda = dia_formado & (z_ant >= k) & (close < close_ant) & (z > 0)
    compra = dia_formado & (z_ant <= -k) & (close > close_ant) & (z < 0)
    return _sinal(df, compra, venda, horas)


def pullback_ema(df: pd.DataFrame, toque: str = "ema9", horas: Iterable[int] | None = None, **_) -> pd.Series:
    """Tendência pela `ema9` vs `ema21`; a barra recua até a média de `toque` e fecha de volta além da `ema9`.

    `toque="ema9"`: basta o low (high, na venda) encostar na `ema9` (inclui os que vão até a `ema21`).
    `toque="ema21"`: só pullbacks fundos, que chegam à `ema21`.
    """
    if toque not in ("ema9", "ema21"):
        raise ValueError(f"toque inválido: {toque}")
    ema9, ema21, nivel = df["ema9"], df["ema21"], df[toque]
    compra = (ema9 > ema21) & (df["low"] <= nivel) & (df["close"] > ema9)
    venda = (ema9 < ema21) & (df["high"] >= nivel) & (df["close"] < ema9)
    return _sinal(df, compra, venda, horas)


def _grade(sinal: dict[str, list], tp: list[float], sl: list[float], max_bars: list[int]) -> list[dict]:
    chaves = [*sinal, "tp_atr", "sl_atr", "max_bars"]
    return [dict(zip(chaves, v)) for v in product(*sinal.values(), tp, sl, max_bars)]


SETUPS: dict[str, Setup] = {
    s.name: s
    for s in (
        Setup("rompimento", rompimento,
              _grade({"vol_z_min": [1.0, 2.0]}, tp=[1.0, 1.5, 2.0], sl=[0.75, 1.0], max_bars=[6, 18])),
        Setup("vwap_reversao", vwap_reversao,
              _grade({"k": [2.0, 2.5, 3.0]}, tp=[1.0, 1.5], sl=[1.0, 1.5], max_bars=[6, 12])),
        Setup("pullback_ema", pullback_ema,
              _grade({"toque": ["ema9", "ema21"]}, tp=[1.0, 1.5, 2.0], sl=[1.0, 1.5], max_bars=[12, 24])),
    )
}
