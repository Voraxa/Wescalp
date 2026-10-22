import importlib
import math
import sys
import types
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from scipy import stats

# labels.py é escrito em paralelo; se ainda não existir, um substituto basta (o teste troca a função)
try:
    importlib.import_module("wescalp.research.labels")
except ImportError:
    _m = types.ModuleType("wescalp.research.labels")

    @dataclass
    class CostModel:
        maker: float = 0.0002
        taker: float = 0.0006
        slippage: float = 0.0001

    _m.CostModel = CostModel
    _m.triple_barrier = None
    sys.modules["wescalp.research.labels"] = _m

from wescalp.research import evaluate as ev  # noqa: E402


def fake_barrier(df, signals, tp_atr, sl_atr, max_bars, cost=None):
    # um trade por sinal: entra na barra seguinte, sai max_bars depois; ret = lado × df["r"] da entrada
    idx = df.index
    pos = np.flatnonzero(signals.to_numpy() != 0)
    rows, livre = [], 0
    for i in pos:
        e = i + 1
        x = e + max_bars - 1
        if e < livre or x >= len(idx):
            continue
        s = int(signals.iloc[i])
        r = s * float(df["r"].iloc[e])
        rows.append({"entry_ts": idx[e], "exit_ts": idx[x], "side": s, "ret_net": r,
                     "regime": "lateral", "hour": idx[i].hour})
        livre = x + 1
    return pd.DataFrame(rows)


@dataclass
class FakeSetup:
    name: str
    fn: object
    grid: list


def _lado(df, lado):
    return pd.Series(lado, index=df.index)


SETUP = FakeSetup("fake", _lado, [
    {"lado": 1, "tp_atr": 1.0, "sl_atr": 1.0, "max_bars": 1},
    {"lado": -1, "tp_atr": 1.0, "sl_atr": 1.0, "max_bars": 1},
    {"lado": -1, "tp_atr": 1.0, "sl_atr": 1.0, "max_bars": 1},  # repetida: conta uma vez
    {"lado": 1, "tp_atr": 2.0, "sl_atr": 1.0, "max_bars": 3},
])


def _df(semanas=14, r=None):
    # barras de 1 h a partir de uma quarta (a primeira semana fica parcial)
    idx = pd.date_range("2025-01-01", periods=semanas * 7 * 24, freq="1h", tz="UTC", name="ts")
    df = pd.DataFrame({"atr14": 1.0, "r": 0.001 if r is None else r}, index=idx)
    return df


@pytest.fixture(autouse=True)
def _stub(monkeypatch):
    monkeypatch.setattr(ev, "triple_barrier", fake_barrier)


def test_folds_sem_sobreposicao_e_sem_lookahead():
    res = ev.walk_forward(_df(), SETUP, train_weeks=4, test_weeks=2, step_weeks=2, min_trades=30)
    f = res.folds
    assert res.n_trials == 3
    assert (f["train_start"].dt.day_of_week == 0).all() and (f["test_start"].dt.day_of_week == 0).all()
    assert (f["train_end"] == f["test_start"]).all()
    ok = f[f["status"] == "ok"]
    # janelas de teste encadeadas, sem sobreposição
    assert (ok["test_start"].iloc[1:].to_numpy() >= ok["test_end"].iloc[:-1].to_numpy()).all()
    t = res.trades.merge(f[["fold", "test_start", "test_end"]], on="fold")
    assert (t["entry_ts"] >= t["test_start"]).all() and (t["entry_ts"] < t["test_end"]).all()
    # uma posição por vez em toda a série fora da amostra
    assert (t["entry_ts"].iloc[1:].to_numpy() > t["exit_ts"].iloc[:-1].to_numpy()).all()
    assert not t.duplicated("entry_ts").any()


def test_params_escolhidos_so_com_o_treino():
    df = _df()
    corte = pd.Timestamp("2025-02-24", tz="UTC")  # segunda; início do 2º teste com treino de 4 semanas
    # compra ganha até o corte e perde depois
    df["r"] = np.where(df.index < corte, 0.001, -0.002)
    res = ev.walk_forward(df, SETUP, train_weeks=4, test_weeks=2, step_weeks=2, min_trades=30)
    f = res.folds.set_index("test_start")
    d = f.loc[corte]
    # o teste que começa no corte ainda escolhe a compra (o treino não vê a virada) e perde
    assert '"lado": 1' in d["params"]
    assert d["test_avg_net_bps"] < 0
    # mexer só no futuro do corte não muda a escolha das dobras que testam a partir do corte
    df2 = df.copy()
    df2.loc[df2.index >= corte, "r"] = 0.05
    res2 = ev.walk_forward(df2, SETUP, train_weeks=4, test_weeks=2, step_weeks=2, min_trades=30)
    f2 = res2.folds.set_index("test_start")
    ate = f.index[f.index <= corte]
    assert (f.loc[ate, "params"] == f2.loc[ate, "params"]).all()
    # depois que o treino passa a ver a virada, a venda vence
    assert '"lado": -1' in f.iloc[-1]["params"]


def test_trades_do_treino_que_saem_no_teste_ficam_de_fora(monkeypatch):
    # max_bars enorme: todo trade sai muito depois; nenhum treino tem trades válidos
    setup = FakeSetup("fake", _lado, [{"lado": 1, "tp_atr": 1.0, "sl_atr": 1.0, "max_bars": 24 * 7 * 30}])
    res = ev.walk_forward(_df(semanas=40), setup, train_weeks=4, test_weeks=2, step_weeks=2, min_trades=1)
    assert (res.folds["status"] == "sem_combo").all()
    assert len(res.trades) == 0 and {"fold", "params"} <= set(res.trades.columns)


def test_dobra_pulada_com_poucos_trades():
    res = ev.walk_forward(_df(), SETUP, train_weeks=4, test_weeks=2, step_weeks=2, min_trades=10_000)
    assert (res.folds["status"] == "sem_combo").all()
    assert res.folds["params"].isna().all()
    assert len(res.trades) == 0


def test_metrics_conhecidas():
    ts = pd.date_range("2025-01-01", periods=5, freq="12h", tz="UTC")
    r = [0.002, -0.001, 0.003, -0.004, 0.001]
    m = ev.metrics(pd.DataFrame({"entry_ts": ts, "ret_net": r}))
    assert m["n"] == 5
    assert m["win_rate"] == pytest.approx(0.6)
    assert m["avg_net_bps"] == pytest.approx(2.0)
    assert m["median_net_bps"] == pytest.approx(10.0)
    assert m["profit_factor"] == pytest.approx(0.006 / 0.005)
    assert m["sharpe_trade"] == pytest.approx(np.mean(r) / np.std(r, ddof=1))
    # acumulado: 0.002, 0.001, 0.004, 0.000, 0.001 → pior queda 0.004
    assert m["max_dd"] == pytest.approx(0.004)
    assert m["trades_por_dia"] == pytest.approx(5 / 2)
    assert ev.metrics(pd.DataFrame({"entry_ts": ts, "ret_net": r}), dias=10)["trades_por_dia"] == 0.5
    assert ev.metrics(pd.DataFrame({"entry_ts": [], "ret_net": []}))["n"] == 0


def test_breakdown():
    t = pd.DataFrame({"entry_ts": pd.date_range("2025-01-01", periods=4, freq="1h", tz="UTC"),
                      "ret_net": [0.001, -0.001, 0.002, 0.002], "regime": ["alta", "alta", "baixa", "baixa"]})
    b = ev.breakdown(t, "regime")
    assert list(b.index) == ["alta", "baixa"]
    assert b.loc["alta", "win_rate"] == 0.5 and b.loc["baixa", "win_rate"] == 1.0
    assert b.loc["baixa", "wr_p05"] < 1.0


def test_beta_posterior():
    # Beta(1+7, 1+3): média 8/12
    media, p05, p95 = ev.beta_posterior(7, 10)
    assert media == pytest.approx(8 / 12)
    assert p05 == pytest.approx(stats.beta(8, 4).ppf(0.05))
    assert p05 < media < p95
    # sem dados = a priori uniforme
    assert ev.beta_posterior(0, 0) == pytest.approx((0.5, 0.05, 0.95))


def test_expected_max_sharpe_proximo_do_exato():
    # E[max de N normais padrão]: 1.5388 (N=10), 2.5076 (N=100)
    assert ev.expected_max_sharpe(10, 1.0) == pytest.approx(1.5388, abs=0.05)
    assert ev.expected_max_sharpe(100, 1.0) == pytest.approx(2.5076, abs=0.05)
    assert ev.expected_max_sharpe(1, 1.0) == 0.0
    assert ev.expected_max_sharpe(100, 0.04) == pytest.approx(0.2 * ev.expected_max_sharpe(100, 1.0))


def test_deflated_sharpe_valor_conhecido():
    # N=1 → PSR contra 0: Φ(0,1·√99 / √(1 + 2/4·0,01)) = Φ(0,99252) ≈ 0,8395
    assert ev.deflated_sharpe(0.1, 100, 1, 0.0, 3.0, 0.01) == pytest.approx(0.8395, abs=1e-3)
    # à mão: SR0 = 0,1·[(1−γ)Φ⁻¹(0,9) + γΦ⁻¹(1 − 1/(10e))]
    g = 0.5772156649015329
    sr0 = 0.1 * ((1 - g) * stats.norm.ppf(0.9) + g * stats.norm.ppf(1 - 1 / (10 * math.e)))
    esperado = stats.norm.cdf((0.2 - sr0) * math.sqrt(499) / math.sqrt(1 + 0.5 * 0.2 + 1.5 * 0.04))
    assert ev.deflated_sharpe(0.2, 500, 10, -0.5, 7.0, 0.01) == pytest.approx(esperado)


def test_deflated_sharpe_cai_com_mais_tentativas():
    vals = [ev.deflated_sharpe(0.1, 1000, n, 0.0, 3.0, 0.002) for n in (1, 2, 5, 10, 50, 100, 1000)]
    assert all(a > b for a, b in zip(vals, vals[1:]))
    # cauda gorda e assimetria negativa também derrubam
    assert ev.deflated_sharpe(0.1, 1000, 10, -1.0, 10.0, 0.002) < ev.deflated_sharpe(0.1, 1000, 10, 0.0, 3.0, 0.002)
