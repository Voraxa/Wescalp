from datetime import date

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("numpyro")
from scipy import stats  # noqa: E402

from wescalp import bayes, factors, refine, regime  # noqa: E402

START = date(2019, 1, 7)  # uma segunda


def inputs(n: int = 400):
    rng = np.random.default_rng(0)
    close = 100 * np.cumprod(1 + rng.normal(0.001, 0.03, n))
    days = np.arange(n) + (START - factors.EPOCH).days
    d = regime.Daily(days, close, close * 1.01, close * 0.99, close)
    weekdays = days[factors.weekday(days) < 5]

    def walk(ds, seed):
        return ds, 100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.01, len(ds))))

    return d, {"fng": walk(days, 1), "NASDAQCOM": walk(weekdays, 2), "DTWEXBGS": walk(weekdays, 3)}


def test_knot_weights_interpolate_between_evenly_spaced_knots():
    w = bayes.knot_weights(105)  # 104 / 26 = 4 intervalos, 5 knots
    assert w.shape == (105, 5) and np.allclose(w.sum(axis=1), 1)
    assert w[0, 0] == 1 and w[-1, -1] == 1 and w[26, 1] == pytest.approx(1)
    assert w[13, 0] == pytest.approx(0.5) and w[13, 1] == pytest.approx(0.5)
    assert bayes.knot_weights(10).shape == (10, 2)  # treino curto: pelo menos 2 knots


def test_weekly_frame_uses_only_what_was_public():
    d, fac = inputs()
    wk = bayes.weekly(d, fac)
    assert (factors.weekday(wk.monday) == 0).all() and (np.diff(wk.monday) == 7).all()
    j = len(wk.y) // 2
    m, i = wk.monday[j], wk.first[j]
    sun = i - 1
    assert d.day[i] == m and wk.x[j, 0] == refine.position(d.close, refine.BASE)[sun]
    fd, fv = fac["fng"]
    assert wk.x[j, 1] == fv[fd == m - 1][0] and wk.x[j, 2] == fv[fd == m - 1][0] - fv[fd == m - 8][0]  # domingos
    nd, nv = fac["NASDAQCOM"]
    assert wk.x[j, 3] == pytest.approx(np.log(nv[nd == m - 3][0] / nv[nd == m - 31][0]))  # sextas
    xd, xv = fac["DTWEXBGS"]
    assert wk.x[j, 4] == pytest.approx(np.log(xv[xd == m - 10][0] / xv[xd == m - 38][0]))  # sexta retrasada
    assert wk.x[j, 5] == pytest.approx(np.log(d.close[sun] / d.close[sun - 28]))
    assert wk.x[j, 6] == pytest.approx(np.std(np.diff(np.log(d.close[sun - 30:sun + 1]))))
    assert wk.y[j] == pytest.approx(np.log(d.close[i + 6] / d.close[sun]))

    # Troca tudo o que só fica público depois da decisão de `m`: os fatores daquela semana e de antes não mudam.
    later = d.day >= m
    d2 = regime.Daily(d.day, d.open, d.high, d.low, np.where(later, d.close * 1.5, d.close))
    fac2 = {k: (ds, np.where(ds >= (m - 7 if k in factors.WEEKLY_RELEASE else m), vs * 2, vs)) for k, (ds, vs) in fac.items()}
    wk2 = bayes.weekly(d2, fac2)
    assert np.array_equal(wk2.x[:j + 1], wk.x[:j + 1]) and wk2.y[j] != wk.y[j]


def test_weekly_frame_stops_at_end_with_full_weeks():
    d, fac = inputs()
    end = int(d.day[300])
    wk = bayes.weekly(d, fac, end)
    assert wk.monday[-1] + 6 <= end < wk.monday[-1] + 13


def test_evaluate_chains_weeks_day_by_day():
    d, fac = inputs()
    wk = bayes.weekly(d, fac)
    rows = np.arange(5, 15)
    out = bayes.evaluate(d, wk, rows, np.full(10, 0.7), luck=0)
    lo, hi = wk.first[rows[0]], wk.first[rows[-1]] + 7
    rets = d.close[lo:hi] / d.close[lo - 1:hi - 1] - 1
    assert np.prod(1 + rets) == pytest.approx(np.exp(wk.y[rows].sum()))
    hold = (1 + rets[0] - regime.FEE) * np.prod(1 + rets[1:]) - 1  # compra no primeiro dia e paga a taxa
    assert out["segurar BTC"]["total"] == pytest.approx(hold)
    assert out[bayes.FORMS[0]]["total"] == pytest.approx(hold)  # chance sempre acima de 50%: comprado o tempo todo
    assert out[bayes.FORMS[1]]["exposure"] == pytest.approx(0.7)
    with pytest.raises(ValueError):
        bayes.evaluate(d, wk, np.array([1, 3]), np.array([0.6, 0.6]), luck=0)


def test_luck_p():
    rng = np.random.default_rng(0)
    rets = rng.normal(0, 0.02, 7 * 60)
    week = (1 + rets.reshape(60, 7)).prod(axis=1) - 1
    assert bayes.luck_p((week > 0).astype(float), rets, 200) < 0.02  # sabe o futuro: o sorteio quase nunca iguala
    assert bayes.luck_p(np.clip(0.5 + 5 * week, 0, 1), rets, 200) < 0.02
    assert bayes.luck_p(np.ones(60), rets, 50) == 1.0  # sempre comprado: todo sorteio é igual
    assert bayes.luck_p(np.zeros(60), rets, 50) == 1.0  # nunca comprado: não tem o que testar


def test_p_up_uses_the_last_knot_and_training_scale():
    f = bayes.Fit({"knots": np.array([[0.5, 0.0]] * 4), "beta": np.array([[0.01]] * 4), "sigma": np.full(4, 0.05),
                   "nu": np.full(4, 5.0)}, np.array([2.0]), np.array([4.0]), 1.0, 1000.0, 0)
    p = bayes.p_up(f, np.array([[2.0], [6.0], [-2.0]]))
    assert p[0] == pytest.approx(0.5)  # fator na média do treino: só o último knot (zero) conta
    assert p[1] == pytest.approx(stats.t.cdf(0.01 / 0.05, 5)) and p[2] == pytest.approx(1 - p[1])


def test_decide_needs_every_criterion():
    a, b, base = bayes.FORMS[0], bayes.FORMS[1], bayes.BENCH[0]
    wf = {base: {"sharpe": 1.0, "dd": 0.4}, a: {"sharpe": 1.2, "dd": 0.3, "p": 0.01},
          b: {"sharpe": 1.2, "dd": 0.5, "p": 0.03}}
    test = {base: {"sharpe": 0.8}, a: {"sharpe": 0.8}, b: {"sharpe": 0.5}}
    fails = bayes.decide(wf, test, valid=True)
    assert fails[a] == [] and fails[b] == ["queda no walk-forward", "sorte", "Sharpe no teste de 13%"]
    assert bayes.decide(wf, test, valid=False)[a] == ["ajuste"]


def test_fit_recovers_a_planted_factor():
    rng = np.random.default_rng(0)
    t = 160
    x = rng.normal(0, 1, (t, 2))
    y = 0.02 * x[:, 0] + 0.04 * rng.standard_t(5, t)
    f = bayes.fit(x, y, seeds=(0, 1), warmup=300, samples=300, chain_method="sequential")
    beta = f.samples["beta"]
    assert np.quantile(beta[:, 0], 0.05) > 0.005 and abs(beta[:, 1].mean()) < 0.008
    assert f.r_hat < 1.05 and f.samples["knots"].shape == (600, bayes.knot_weights(t).shape[1])
