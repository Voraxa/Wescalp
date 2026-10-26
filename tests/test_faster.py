import pytest

np = pytest.importorskip("numpy")

from wescalp import faster, intraday, refine, regime  # noqa: E402

DAY = 86400


def five_min(days: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    t0 = intraday.utc("2019-01-01")
    ts = np.arange(t0, t0 + days * DAY, 300, dtype=np.int64)
    return ts, 100 * np.exp(np.cumsum(rng.normal(0.00002, 0.002, len(ts))))


def test_bars_take_the_last_close_of_each_period():
    ts, c = five_min(3)
    t, close = faster.bars(ts, c, 4)
    assert len(t) == 18 and t[1] - t[0] == 4 * 3600
    assert close[0] == c[47] and close[-1] == c[-1]
    d = regime.daily_bars(ts, c, c, c, c)
    assert np.array_equal(faster.bars(ts, c, 24)[1], d.close)


def test_daily_variant_is_the_frozen_rule():
    ts, c = five_min(400)
    t, close = faster.bars(ts, c, 24)
    start, end = int(t[200]), int(t[-1]) + DAY
    s = faster.evaluate(t, close, 24, (50, 100, 150), luck=0, start=start, end=end)
    rets = close[200:] / close[199:-1] - 1
    r = regime.returns_from_held(refine.position(close, refine.BASE)[199:-1], rets)
    assert s[0.001]["cagr"] == pytest.approx(np.prod(1 + r) ** (365.25 / len(r)) - 1)
    assert s[0.001]["sharpe"] == pytest.approx(r.mean() / r.std() * np.sqrt(365.25))


def test_operations_count_each_entry_of_each_average():
    close = np.array([10.0] * 5 + [11, 12, 9, 8, 12, 13, 7, 6, 14])
    t = np.arange(len(close)) * 3600
    s = faster.evaluate(t, close, 1, (2,), luck=20, start=int(t[3]), end=int(t[-1]) + 3600)
    held = regime.held_from(faster.components(close, (2,))[0], 3)
    entries = int((np.diff(np.concatenate([[0.0], held])) > 0).sum())
    assert s["per_year"] == pytest.approx(entries / ((len(close) - 3) / (365.25 * 24)))
    assert 0 <= s["p"] <= 1
