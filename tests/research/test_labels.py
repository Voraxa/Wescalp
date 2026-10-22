import numpy as np
import pandas as pd
import pytest

from wescalp.research.labels import CostModel, triple_barrier

C = CostModel()
CUSTO_TP = 2 * C.maker
CUSTO_MERCADO = C.maker + C.taker + C.slippage


def barras(n=30, atr=10.0):
    # preço parado em 1000, barras estreitas (±1): nada toca barreira a 1 ATR
    ts = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC", name="ts")
    df = pd.DataFrame({"open": 1000.0, "high": 1001.0, "low": 999.0, "close": 1000.0,
                       "atr14": atr, "regime": "lateral", "hour": ts.hour}, index=ts)
    return df


def sinais(df, marcas):
    s = pd.Series(0, index=df.index)
    for i, v in marcas.items():
        s.iloc[i] = v
    return s


def test_long_alvo_entrada_no_open_seguinte():
    df = barras()
    df.iloc[3, df.columns.get_loc("open")] = 1002.0  # entrada na barra 3
    df.iloc[5, df.columns.get_loc("high")] = 1013.0
    df.iloc[2, df.columns.get_loc("regime")] = "alta"
    tr = triple_barrier(df, sinais(df, {2: 1}), tp_atr=1.0, sl_atr=1.0, max_bars=10)
    assert len(tr) == 1
    r = tr.iloc[0]
    assert r.entry_ts == df.index[3] and r.exit_ts == df.index[5]
    assert r.entry_px == 1002 and r.exit_px == 1012 and r.outcome == "tp" and r.bars == 3
    assert r.ret_gross == pytest.approx(1012 / 1002 - 1)
    assert r.cost == pytest.approx(CUSTO_TP)
    assert r.ret_net == pytest.approx(1012 / 1002 - 1 - CUSTO_TP)
    assert r.regime == "alta" and r.hour == df.index[2].hour
    assert list(tr.columns) == ["entry_ts", "exit_ts", "side", "entry_px", "exit_px", "outcome", "bars",
                                "ret_gross", "cost", "ret_net", "regime", "hour"]


def test_mesma_barra_conta_stop():
    df = barras()
    df.iloc[4, df.columns.get_loc("high")] = 1020.0
    df.iloc[4, df.columns.get_loc("low")] = 980.0
    tr = triple_barrier(df, sinais(df, {2: 1}), 1.0, 1.0, 10)
    r = tr.iloc[0]
    assert r.outcome == "sl" and r.exit_px == 990 and r.bars == 2
    assert r.cost == pytest.approx(CUSTO_MERCADO)
    assert r.ret_net == pytest.approx(990 / 1000 - 1 - CUSTO_MERCADO)


def test_barra_de_entrada_conta():
    df = barras()
    df.iloc[3, df.columns.get_loc("high")] = 1011.0
    tr = triple_barrier(df, sinais(df, {2: 1}), 1.0, 1.0, 10)
    assert tr.iloc[0].outcome == "tp" and tr.iloc[0].exit_ts == df.index[3] and tr.iloc[0].bars == 1


def test_saida_por_tempo():
    df = barras()
    df.iloc[7, df.columns.get_loc("close")] = 1004.0  # última barra da janela (3..7)
    tr = triple_barrier(df, sinais(df, {2: 1}), 1.0, 1.0, 5)
    r = tr.iloc[0]
    assert r.outcome == "time" and r.exit_ts == df.index[7] and r.bars == 5 and r.exit_px == 1004
    assert r.ret_net == pytest.approx(0.004 - CUSTO_MERCADO)


def test_short_sinal_do_lado():
    df = barras()
    df.iloc[5, df.columns.get_loc("low")] = 985.0  # alvo da venda a 990
    tr = triple_barrier(df, sinais(df, {2: -1}), 1.0, 2.0, 10)
    r = tr.iloc[0]
    assert r.side == -1 and r.outcome == "tp" and r.exit_px == 990
    assert r.ret_gross == pytest.approx(0.01)
    assert r.ret_net == pytest.approx(0.01 - CUSTO_TP)
    df.iloc[5, df.columns.get_loc("low")] = 999.0
    df.iloc[6, df.columns.get_loc("high")] = 1025.0  # stop da venda a 1020
    r = triple_barrier(df, sinais(df, {2: -1}), 1.0, 2.0, 10).iloc[0]
    assert r.outcome == "sl" and r.exit_px == 1020
    assert r.ret_gross == pytest.approx(-0.02)
    assert r.ret_net == pytest.approx(-0.02 - CUSTO_MERCADO)


def test_sinais_sobrepostos_ignorados():
    df = barras()
    # trade 1: sinal 2, entrada 3, sai por tempo na barra 7 (max_bars 5)
    tr = triple_barrier(df, sinais(df, {2: 1, 3: -1, 6: 1, 7: -1, 9: 1}), 1.0, 1.0, 5)
    assert list(tr.entry_ts) == [df.index[3], df.index[8]]
    assert list(tr.side) == [1, -1]
    # o sinal 9 caiu dentro do trade 2 (entrada 8, saída 12)
    assert tr.iloc[1].exit_ts == df.index[12]


def test_atr_invalido_e_ultima_barra_e_janela_incompleta():
    df = barras(n=10)
    df.iloc[1, df.columns.get_loc("atr14")] = np.nan
    tr = triple_barrier(df, sinais(df, {1: 1, 9: 1}), 1.0, 1.0, 3)
    assert tr.empty and "ret_net" in tr.columns
    # janela passa do fim sem toque: descartado
    assert triple_barrier(df, sinais(df, {7: 1}), 1.0, 1.0, 5).empty
    # mas com toque antes do fim, conta
    df.iloc[9, df.columns.get_loc("high")] = 1020.0
    assert triple_barrier(df, sinais(df, {7: 1}), 1.0, 1.0, 5).iloc[0].outcome == "tp"


def test_stop_com_gap_sai_na_abertura():
    df = barras()
    df.iloc[5, df.columns.get_loc("open")] = 980.0
    df.iloc[5, df.columns.get_loc("low")] = 975.0
    r = triple_barrier(df, sinais(df, {2: 1}), 1.0, 1.0, 10).iloc[0]
    assert r.outcome == "sl" and r.exit_px == 980


def test_custo_personalizado():
    df = barras()
    df.iloc[5, df.columns.get_loc("high")] = 1013.0
    r = triple_barrier(df, sinais(df, {2: 1}), 1.0, 1.0, 10, cost=CostModel(maker=0, taker=0, slippage=0)).iloc[0]
    assert r.cost == 0 and r.ret_net == r.ret_gross
