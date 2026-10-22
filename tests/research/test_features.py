import numpy as np
import pandas as pd
import pytest

from wescalp.research.features import add_features


def candles(n=700, inicio="2025-01-01 20:00", seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range(inicio, periods=n, freq="5min", tz="UTC")
    c = 100_000 + np.cumsum(rng.normal(0, 50, n))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) + rng.uniform(0, 30, n)
    l = np.minimum(o, c) - rng.uniform(0, 30, n)
    v = rng.uniform(1000, 5000, n)
    return pd.DataFrame({"ts": ts, "open": o, "high": h, "low": l, "close": c, "volume": v, "turnover": v * c / 1000})


COLS = ["ema9", "ema21", "atr14", "atr_pct", "vwap", "vwap_sd", "vwap_z", "rsi14", "vol_z",
        "hh20", "ll20", "pdh", "pdl", "hour", "regime"]


def test_colunas_e_indice():
    raw = candles().sample(frac=1, random_state=1)  # embaralhado
    f = add_features(raw)
    assert f.index.name == "ts" and f.index.is_monotonic_increasing
    for col in COLS:
        assert col in f.columns
    assert "ts" not in raw.columns or len(f) == len(raw)
    assert f["hour"].dtype.kind == "i"
    assert set(f["regime"].dropna()) <= {"alta", "baixa", "lateral"}


def test_sem_lookahead():
    raw = candles()
    base = add_features(raw)
    k = 500
    mexido = raw.copy()
    for col in ("open", "high", "low", "close", "volume"):
        mexido.loc[k:, col] = mexido.loc[k:, col] * 1.5
    alt = add_features(mexido)
    num = [c for c in COLS if c != "regime"]
    pd.testing.assert_frame_equal(base[num].iloc[:k], alt[num].iloc[:k])
    pd.testing.assert_series_equal(base["regime"].iloc[:k], alt["regime"].iloc[:k])


def test_vwap_zera_a_meia_noite():
    raw = candles(n=100, inicio="2025-01-01 23:00")  # 12 barras no dia 1, resto no dia 2
    f = add_features(raw)
    meia_noite = pd.Timestamp("2025-01-02", tz="UTC")
    primeira = f.loc[meia_noite]
    tp = (primeira["high"] + primeira["low"] + primeira["close"]) / 3
    assert primeira["vwap"] == pytest.approx(tp)
    assert primeira["vwap_sd"] == 0
    assert np.isnan(primeira["vwap_z"])
    # duas barras do dia: média ponderada à mão
    d2 = f.loc[meia_noite:].iloc[:2]
    tps = (d2["high"] + d2["low"] + d2["close"]) / 3
    esperado = (tps * d2["volume"]).sum() / d2["volume"].sum()
    assert d2["vwap"].iloc[1] == pytest.approx(esperado)
    sd = np.sqrt((d2["volume"] * (tps - esperado) ** 2).sum() / d2["volume"].sum())
    assert d2["vwap_sd"].iloc[1] == pytest.approx(sd)
    assert d2["vwap_z"].iloc[1] == pytest.approx((d2["close"].iloc[1] - esperado) / sd)


def test_hh_ll_excluem_barra_atual():
    raw = candles(n=60)
    raw.loc[30, "high"] = 1e6
    raw.loc[30, "low"] = 1.0
    f = add_features(raw).reset_index()
    assert f.loc[30, "hh20"] < 1e6 and f.loc[30, "ll20"] > 1.0
    assert f.loc[31, "hh20"] == 1e6 and f.loc[31, "ll20"] == 1.0
    assert f.loc[50, "hh20"] == 1e6 and f.loc[51, "hh20"] < 1e6
    assert f.loc[25, "hh20"] == raw.loc[5:24, "high"].max()
    assert np.isnan(f.loc[19, "hh20"]) and not np.isnan(f.loc[20, "hh20"])


def test_vol_z_exclui_barra_atual():
    raw = candles(n=400)
    f = add_features(raw).reset_index()
    t = 350
    ant = raw.loc[t - 288:t - 1, "volume"]
    assert f.loc[t, "vol_z"] == pytest.approx((raw.loc[t, "volume"] - ant.mean()) / ant.std())
    assert np.isnan(f.loc[287, "vol_z"]) and not np.isnan(f.loc[288, "vol_z"])
    # pico na barra atual não entra na própria média
    raw.loc[t, "volume"] = 1e9
    f2 = add_features(raw).reset_index()
    assert f2.loc[t, "vol_z"] == pytest.approx((1e9 - ant.mean()) / ant.std())


def test_pdh_pdl_e_hora():
    raw = candles(n=600, inicio="2025-01-01 00:00")
    f = add_features(raw)
    d1 = f.loc["2025-01-01"]
    d2 = f.loc["2025-01-02"]
    assert d1["pdh"].isna().all()
    assert (d2["pdh"] == d1["high"].max()).all() and (d2["pdl"] == d1["low"].min()).all()
    assert f.loc["2025-01-01 13:55", "hour"] == 13


def test_atr_rsi_regime_basicos():
    n = 200
    ts = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
    c = 100_000 + np.arange(n) * 100.0  # alta constante
    raw = pd.DataFrame({"ts": ts, "open": c - 50, "high": c + 10, "low": c - 60, "close": c,
                        "volume": 1000.0, "turnover": 1.0})
    f = add_features(raw)
    assert f["rsi14"].iloc[-1] == 100
    assert f["atr14"].iloc[-1] == pytest.approx(110, rel=1e-6)  # max(h-l=70, h-pc=110)
    assert np.isnan(f["atr14"].iloc[12]) and not np.isnan(f["atr14"].iloc[13])
    assert f["regime"].iloc[-1] == "alta"
    raw["close"] = c[::-1]
    raw["open"], raw["high"], raw["low"] = raw["close"] + 50, raw["close"] + 60, raw["close"] - 10
    f = add_features(raw)
    assert f["regime"].iloc[-1] == "baixa"
    assert f["rsi14"].iloc[-1] == pytest.approx(0)
