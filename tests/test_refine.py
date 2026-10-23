import pytest

np = pytest.importorskip("numpy")

from wescalp import refine  # noqa: E402


def test_vol_scale_uses_only_the_past():
    rng = np.random.default_rng(0)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.02, 600))
    a = refine.vol_scale(close)
    future = close.copy()
    future[500:] *= np.cumprod(1 + rng.normal(0, 0.2, 100))  # futuro muito mais volátil
    b = refine.vol_scale(future)
    assert np.array_equal(a[:500], b[:500])
    assert (a <= 1).all() and b[550] < 1  # sem alavancagem; corta exposição no pânico
    assert (a[:30 + 365] == 1).all()  # sem histórico, não mexe


def test_positions_are_component_averages():
    close = np.concatenate([np.full(200, 100.0), np.linspace(100, 300, 200)])
    pos = refine.position(close, refine.BASE)
    assert set(np.unique(pos)) <= {0.0, 1 / 3, 2 / 3, 1.0}
    assert pos[-1] == 1.0
    assert refine.position(close, "ampla 50..170")[-1] == 1.0
    assert len(refine.components(close, "mista")[0]) == 5


def test_decide_requires_every_criterion():
    base = {"sharpe": 1.0, "dd": 0.4, "worst_hour": 0.5, "second_half": 0.3, "p": 0.01}
    better = {"sharpe": 1.1, "dd": 0.35, "worst_hour": 0.52, "second_half": 0.31, "p": 0.02}
    deeper = dict(better, dd=0.45)
    lucky = dict(better, p=0.2)
    win, fails = refine.decide({refine.BASE: base, "a": better, "b": deeper, "c": lucky})
    assert win == "a" and fails == {"a": [], "b": ["queda"], "c": ["sorte"]}
    win, _ = refine.decide({refine.BASE: base, "b": deeper})
    assert win == refine.BASE
