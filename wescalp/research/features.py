"""Features dos candles de 5 min. Tudo causal: a linha t só usa dados até o fechamento de t."""

import numpy as np
import pandas as pd

VOL_JANELA = 288  # 1 dia de barras de 5 min
CANAL = 20
REGIME_BARRAS = 12
REGIME_LIMIAR = 0.5


def _wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def _indexar(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "ts" in out.columns:
        out = out.set_index("ts")
    out.index = pd.DatetimeIndex(out.index, name="ts")
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    out = out[~out.index.duplicated(keep="last")]
    return out.sort_index()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = _indexar(df)
    o, h, l, c, v = (out[k].astype(float) for k in ("open", "high", "low", "close", "volume"))

    out["ema9"] = c.ewm(span=9, adjust=False).mean()
    out["ema21"] = c.ewm(span=21, adjust=False).mean()

    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    out["atr14"] = _wilder(tr, 14)
    out["atr_pct"] = out["atr14"] / c

    # VWAP do dia UTC; o preço típico é centrado no primeiro do dia para a variância não perder precisão
    dia = out.index.floor("D")
    tp = (h + l + c) / 3
    base = tp.groupby(dia).transform("first")
    x = tp - base
    g = pd.DataFrame({"v": v, "xv": x * v, "x2v": x * x * v}).groupby(dia).cumsum()
    with np.errstate(divide="ignore", invalid="ignore"):
        media = g["xv"] / g["v"]
        var = g["x2v"] / g["v"] - media**2
    out["vwap"] = base + media
    sd = np.sqrt(var.clip(lower=0))
    sd = sd.where(sd > 1e-9 * out["vwap"].abs(), 0.0).where(var.notna())
    out["vwap_sd"] = sd
    out["vwap_z"] = (c - out["vwap"]) / sd.replace(0.0, np.nan)

    d = c.diff()
    ganho = _wilder(d.clip(lower=0), 14)
    perda = _wilder((-d).clip(lower=0), 14)
    with np.errstate(divide="ignore", invalid="ignore"):
        rsi = 100 - 100 / (1 + ganho / perda)
    out["rsi14"] = rsi.where(perda != 0, 100.0).where(ganho.notna() & perda.notna())

    # média e desvio das 288 barras anteriores, sem a atual
    ant = v.shift(1).rolling(VOL_JANELA, min_periods=VOL_JANELA)
    vsd = ant.std()
    out["vol_z"] = (v - ant.mean()) / vsd.where(vsd > 0)

    out["hh20"] = h.rolling(CANAL, min_periods=CANAL).max().shift(1)
    out["ll20"] = l.rolling(CANAL, min_periods=CANAL).min().shift(1)

    # dia anterior pelo calendário (se o dia anterior não tem dados, fica NaN)
    diario = pd.DataFrame({"h": h.groupby(dia).max(), "l": l.groupby(dia).min()})
    diario.index = diario.index + pd.Timedelta(days=1)
    out["pdh"] = diario["h"].reindex(dia).to_numpy()
    out["pdl"] = diario["l"].reindex(dia).to_numpy()

    out["hour"] = out.index.hour.astype(int)

    incl = (out["ema21"] - out["ema21"].shift(REGIME_BARRAS)) / out["atr14"]
    regime = np.where(incl > REGIME_LIMIAR, "alta", np.where(incl < -REGIME_LIMIAR, "baixa", "lateral"))
    out["regime"] = pd.Series(regime, index=out.index, dtype=object).where(incl.notna())
    return out
