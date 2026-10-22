import numpy as np
import pandas as pd
import pytest

from wescalp.research.setups import SETUPS, Setup, pullback_ema, rompimento, vwap_reversao

COLUNAS = ["open", "high", "low", "close", "volume", "ema9", "ema21", "atr14", "atr_pct", "vwap", "vwap_sd",
           "vwap_z", "rsi14", "vol_z", "hh20", "ll20", "pdh", "pdl", "hour", "regime"]


def _frame(linhas: list[dict]) -> pd.DataFrame:
    # Preenche colunas não usadas com valores neutros; índice de 5 min em UTC.
    base = {c: 0.0 for c in COLUNAS} | {"hour": 10, "regime": "lateral"}
    idx = pd.date_range("2026-01-01 10:00", periods=len(linhas), freq="5min", tz="UTC", name="ts")
    return pd.DataFrame([base | l for l in linhas], index=idx)


def _aleatorio(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 60000 + np.cumsum(rng.normal(0, 30, n))
    df = pd.DataFrame({
        "open": close + rng.normal(0, 10, n),
        "close": close,
        "high": close + rng.uniform(0, 40, n),
        "low": close - rng.uniform(0, 40, n),
        "ema9": close + rng.normal(0, 20, n),
        "ema21": close + rng.normal(0, 20, n),
        "vwap_z": rng.normal(0, 2, n),
        "vol_z": rng.normal(0.5, 1.5, n),
        "hh20": close + rng.normal(0, 30, n),
        "ll20": close + rng.normal(-30, 30, n),
        "hour": rng.integers(0, 24, n),
    }, index=pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC", name="ts"))
    df.iloc[:5, df.columns.get_loc("ema9")] = np.nan  # aquecimento
    return df


def test_rompimento_compra_e_venda():
    df = _frame([
        {"close": 101, "hh20": 100, "ll20": 90, "vol_z": 2.5},  # rompe para cima com volume
        {"close": 101, "hh20": 100, "ll20": 90, "vol_z": 0.5},  # rompe sem volume
        {"close": 89, "hh20": 100, "ll20": 90, "vol_z": 3.0},   # rompe para baixo
        {"close": 95, "hh20": 100, "ll20": 90, "vol_z": 3.0},   # dentro do canal
        {"close": 101, "hh20": np.nan, "ll20": np.nan, "vol_z": 3.0},  # aquecimento
    ])
    s = rompimento(df, vol_z_min=2.0)
    assert s.index.equals(df.index)
    assert s.tolist() == [1, 0, -1, 0, 0]
    assert rompimento(df, vol_z_min=0.0).tolist() == [1, 1, -1, 0, 0]


def test_vwap_reversao_compra_e_venda():
    df = _frame([
        {"close": 110, "vwap_z": 2.6},
        {"close": 108, "vwap_z": 2.1},   # venda: anterior esticada, fecha de volta para dentro
        {"close": 90, "vwap_z": -2.7},
        {"close": 91, "vwap_z": -2.4},   # compra
        {"close": 92, "vwap_z": -1.0},   # anterior em -2,4 < k=2,5 → nada com k=2,5
        {"close": 120, "vwap_z": 3.0},
        {"close": 121, "vwap_z": 3.2},   # ainda esticando: nada
        {"close": 99, "vwap_z": -0.1},   # voltou e passou da VWAP: nada (sem caminho até ela)
    ])
    s = vwap_reversao(df, k=2.5)
    assert s.index.equals(df.index)
    assert s.tolist() == [0, -1, 0, 1, 0, 0, 0, 0]
    assert vwap_reversao(df, k=2.0).tolist() == [0, -1, 0, 1, 1, 0, 0, 0]
    assert vwap_reversao(df, k=3.0).tolist() == [0, 0, 0, 0, 0, 0, 0, 0]


def test_pullback_ema_compra_e_venda():
    df = _frame([
        # alta: low encosta na ema9 e fecha acima
        {"ema9": 100, "ema21": 98, "low": 99.5, "high": 102, "close": 101},
        # alta: pullback fundo até a ema21
        {"ema9": 100, "ema21": 98, "low": 97.5, "high": 102, "close": 100.5},
        # alta mas fecha abaixo da ema9
        {"ema9": 100, "ema21": 98, "low": 97.5, "high": 101, "close": 99},
        # baixa: high encosta na ema9 e fecha abaixo
        {"ema9": 100, "ema21": 102, "low": 98, "high": 100.5, "close": 99},
        # baixa: high nem chega na ema9
        {"ema9": 100, "ema21": 102, "low": 97, "high": 99.5, "close": 98},
    ])
    s = pullback_ema(df, toque="ema9")
    assert s.index.equals(df.index)
    assert s.tolist() == [1, 1, 0, -1, 0]
    assert pullback_ema(df, toque="ema21").tolist() == [0, 1, 0, 0, 0]
    with pytest.raises(ValueError):
        pullback_ema(df, toque="ema50")


def test_filtro_de_hora():
    df = _frame([{"close": 101, "hh20": 100, "ll20": 90, "vol_z": 3, "hour": h} for h in (3, 14)])
    assert rompimento(df, vol_z_min=1, horas=[14]).tolist() == [0, 1]


@pytest.mark.parametrize("nome", sorted(SETUPS))
def test_nao_olha_o_futuro(nome):
    setup = SETUPS[nome]
    df = _aleatorio()
    for params in setup.grid[:: max(1, len(setup.grid) // 4)]:
        base = setup.fn(df, **params)
        assert base.index.equals(df.index)
        assert set(base.unique()) <= {-1, 0, 1}
        assert (base != 0).sum() > 0  # o teste só vale se houver sinais
        for corte in (50, 200, 350):
            mexido = df.copy()
            ruido = _aleatorio(len(df) - corte, seed=corte)
            mexido.iloc[corte:] = ruido.to_numpy()
            depois = setup.fn(mexido, **params)
            pd.testing.assert_series_equal(base.iloc[:corte], depois.iloc[:corte])


@pytest.mark.parametrize("nome", sorted(SETUPS))
def test_grade(nome):
    setup = SETUPS[nome]
    assert isinstance(setup, Setup) and setup.name == nome
    assert 0 < len(setup.grid) <= 30
    assert len({tuple(sorted(p.items())) for p in setup.grid}) == len(setup.grid)
    for p in setup.grid:
        assert {"tp_atr", "sl_atr", "max_bars"} <= p.keys()
        assert p["tp_atr"] >= 1.0 and 0.5 <= p["sl_atr"] <= 2.0 and 3 <= p["max_bars"] <= 24
        setup.fn(_aleatorio(), **p)  # todo combo é aceito pela função


def test_registro_tem_os_tres_setups():
    assert set(SETUPS) == {"rompimento", "vwap_reversao", "pullback_ema"}


def test_vwap_reversao_ignora_primeira_hora_do_dia():
    linhas = [{"close": 110, "vwap_z": 2.6}, {"close": 108, "vwap_z": 2.1}]
    assert vwap_reversao(_frame([r | {"hour": 0} for r in linhas]), k=2.5).tolist() == [0, 0]
    assert vwap_reversao(_frame([r | {"hour": 1} for r in linhas]), k=2.5).tolist() == [0, -1]
