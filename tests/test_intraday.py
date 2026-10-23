import math

import pytest

np = pytest.importorskip("numpy")

from wescalp import intraday  # noqa: E402

H = 3600
T0 = intraday.utc("2022-01-03")


def market(n_hours: int = 60, seed: int = 0):
    """Barras de 5 min de tudo, com números ao acaso, começando 30 dias antes de T0."""
    rng = np.random.default_rng(seed)
    ts = np.arange(T0 - 30 * 24 * H, T0 + n_hours * H, 300, dtype=np.int64)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, len(ts))))
    vol = rng.uniform(1, 2, len(ts))
    spot = {"ts": ts, "close": close}
    flow = {"ts": ts, "volume": vol, "taker_buy": vol * rng.uniform(0.3, 0.7, len(ts))}
    perp = {"ts": ts, "volume": vol * 3, "taker_buy": vol * 3 * rng.uniform(0.3, 0.7, len(ts))}
    premium = {"ts": ts, "close": rng.normal(0, 1e-4, len(ts))}
    funding = {"ts": np.arange(ts[0], ts[-1], 8 * H, dtype=np.int64)}
    funding["rate"] = rng.normal(0, 1e-4, len(funding["ts"]))
    metrics = {"ts": ts, "oi": 1000 * np.exp(np.cumsum(rng.normal(0, 0.001, len(ts)))),
               "accounts": rng.uniform(0.8, 1.2, len(ts)), "top_positions": rng.uniform(0.8, 1.2, len(ts))}
    return spot, flow, perp, premium, funding, metrics


def test_helpers():
    ts = np.array([0, 300, 900])
    v = np.array([1.0, 2.0, 3.0])
    assert list(intraday.at(ts, v, np.array([300, 600]))[:1]) == [2.0] and math.isnan(intraday.at(ts, v, np.array([600]))[0])
    assert list(intraday.last_known(ts, v, np.array([600, 900, 5000]), 600)[:2]) == [2.0, 3.0]
    assert math.isnan(intraday.last_known(ts, v, np.array([5000]), 600)[0])  # velho demais
    assert list(intraday.window_sum(ts, v, np.array([0, 300]), np.array([900, 1000]))) == [3.0, 5.0]


def test_signals_and_target_use_the_right_side_of_the_decision():
    m = market()
    decision = np.array([T0 + 10 * H], dtype=np.int64)
    s = intraday.signals(decision, *m)
    spot, flow = m[0], m[1]
    price = lambda t: spot["close"][np.searchsorted(spot["ts"], t - 300)]  # noqa: E731
    assert s["retorno 1h"][0] == pytest.approx(math.log(price(decision[0]) / price(decision[0] - H)))
    last_hour = (flow["ts"] >= decision[0] - H) & (flow["ts"] < decision[0])
    assert last_hour.sum() == 12
    assert s["fluxo spot 1h"][0] == pytest.approx(flow["taker_buy"][last_hour].sum() / flow["volume"][last_hour].sum() - 0.5)
    y = intraday.forward(spot, decision, 4)
    assert y[0] == pytest.approx(price(decision[0] + 4 * H) / price(decision[0]) - 1)

    # Mudar tudo o que só fica conhecido depois da decisão não mexe em nenhum sinal.
    def change(cut):
        return [{k: (v if k == "ts" else np.where(d["ts"] >= cut, v * 7, v)) for k, v in d.items()} for d in m]

    s2 = intraday.signals(decision, *change(decision[0]))
    assert all(np.array_equal(s[k], s2[k], equal_nan=True) for k in s)
    # A foto das posições de 5 min antes já vale na decisão (5 min de atraso, não mais).
    s3 = intraday.signals(decision, *change(decision[0] - 300))
    assert s3["posições Δ1h"][0] != s["posições Δ1h"][0]


def test_tail_test_picks_the_side_inside_and_judges_outside():
    rng = np.random.default_rng(1)
    s_in, s_out = rng.normal(size=5000), rng.normal(size=5000)
    y_in, y_out = 0.004 * (s_in < -1.28) + rng.normal(0, 0.01, 5000), 0.004 * (s_out < -1.28) + rng.normal(0, 0.01, 5000)
    r = intraday.tail_test(s_in, y_in, s_out, y_out)
    assert r["side"] == "baixo" and r["passes"] and r["out"] == pytest.approx(0.004, abs=0.0015)
    flat = intraday.tail_test(s_in, rng.normal(0, 0.01, 5000), s_out, rng.normal(0, 0.01, 5000))
    assert not flat["passes"]  # sem efeito, o decil não paga a taxa


def test_combinations_run_and_keep_the_cut_from_training():
    rng = np.random.default_rng(2)
    x_in, x_out = rng.normal(size=(3000, 3)), rng.normal(size=(1000, 3))
    y_in, y_out = 0.003 * x_in[:, 0] + rng.normal(0, 0.01, 3000), 0.003 * x_out[:, 0] + rng.normal(0, 0.01, 1000)
    r = intraday.top_test(*intraday.predict(intraday.ridge_fit, x_in, y_in, x_out, y_out))
    assert r["side"] == "cima" and 50 < r["n"] < 200 and r["out"] > y_out.mean()
