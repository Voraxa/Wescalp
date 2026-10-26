import pytest

np = pytest.importorskip("numpy")

from wescalp import flips, intraday  # noqa: E402

DAY = 86400


def walk(n: int = 900, seed: int = 0) -> np.ndarray:
    return 100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0.0005, 0.03, n)))


def test_events_are_the_buys_and_sells_of_each_average():
    close = walk()
    evs = flips.events("X", close)
    assert {e.n for e in evs} == {50, 100, 150}
    for e in evs:
        on, m = flips.above(close, e.n)
        assert on[e.day] == 1 and on[e.day - 1] == 0 and not np.isnan(m[e.day - 1])
        if e.exit is not None:
            assert on[e.exit] == 0 and on[e.day:e.exit].all()
            assert e.ret == pytest.approx(close[e.exit] / close[e.day] * (1 - flips.FEE) ** 2 - 1)
        assert np.all(np.isfinite(e.x)) and len(e.x) == len(flips.VARIABLES)


def test_variables_use_only_the_past():
    close = walk()
    later = close.copy()
    later[600:] *= np.linspace(1, 2, 300)
    a = {(e.n, e.day): e.x for e in flips.events("X", close) if e.day < 600}
    b = {(e.n, e.day): e.x for e in flips.events("X", later) if e.day < 600}
    assert a == b and len(a) > 10


def test_logit_finds_the_variable_that_matters():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2000, 3))
    y = (x[:, 0] + rng.normal(0, 0.5, 2000) > 0).astype(float)
    prob, w = flips.fit_logit(x, y)
    assert w[1] > 2 and abs(w[2]) < 0.3 and abs(w[3]) < 0.3
    assert prob(np.array([[2.0, 0, 0]]))[0] > 0.9


def test_filtered_keeps_a_skipped_average_in_dollars_until_its_next_buy():
    on = np.array([0, 1, 1, 1, 0, 0, 1, 1, 0], dtype=float)
    e = flips.Event("X", 50, 1, 4, 0.0, (0.0,) * 8)
    assert flips.filtered(on, [e]).tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 0]
    still_open = flips.Event("X", 50, 6, None, 0.0, (0.0,) * 8)
    assert flips.filtered(on, [still_open]).tolist() == [0, 1, 1, 1, 0, 0, 0, 0, 0]


def test_decisions_train_only_on_closed_trades_and_run_counts():
    first = intraday.utc("2017-01-01") // DAY
    close = walk(2000, seed=2)
    day = np.arange(first, first + len(close))
    a = flips.Asset("BTC", day, close, flips.events("BTC", close))
    dec = flips.decisions([a], years=(2020, 2021))
    for (_, n, t), (ok, q, y) in dec.items():
        assert intraday.utc(f"{y}-01-01") <= day[t] * DAY < intraday.utc(f"{y + 1}-01-01") and 0 < q < 1
    s = flips.run(a, dec, intraday.utc("2020-01-01"), intraday.utc("2022-01-01"), luck=50)
    assert s["buys"] == len(dec) and 0 < s["accepted"] < s["buys"] and 0 <= s["p"] <= 1
