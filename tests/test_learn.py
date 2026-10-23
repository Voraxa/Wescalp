import pytest

np = pytest.importorskip("numpy")

from wescalp import learn  # noqa: E402


def test_weights_start_equal_and_move_to_the_component_that_works():
    t = 600
    close = 100 * np.cumprod(np.full(t, 1.002))
    comps = np.zeros((2, t))
    comps[0, 100:] = 1  # o componente 0 fica comprado numa alta; o 1 fica fora
    pos, w = learn.learn_positions(comps, close, band=0.0)
    assert np.allclose(w[0], 0.5)
    assert w[-1, 0] > 0.9 and pos[-1] == pytest.approx(w[-1, 0])


def test_no_lookahead():
    rng = np.random.default_rng(0)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.03, 500))
    comps = (rng.random((4, 500)) > 0.5).astype(float)
    pos, w = learn.learn_positions(comps, close)
    future = close.copy()
    future[400:] *= 3
    pos2, w2 = learn.learn_positions(comps, future)
    assert np.array_equal(pos[:400], pos2[:400]) and np.array_equal(w[:400], w2[:400])


def test_band_skips_small_rebalances_but_reaches_extremes():
    close = np.full(50, 100.0)
    comps = np.zeros((20, 50))
    comps[:1, 10:] = 1  # 1 de 20 componentes com pesos iguais: alvo de 5%
    comps[1:, 30:40] = 1  # depois todos (alvo 100%) e de volta a 1 de 20
    pos, _ = learn.learn_positions(comps, close, eta=0.0, band=0.06)
    assert pos[15] == 0.0  # 5% < faixa de 6%: não ajusta
    assert pos[35] == 1.0  # foi a 100% (extremo: ajusta sempre)
    assert pos[45] == pytest.approx(0.05)  # de 100% para 5%: passa da faixa


def test_equal_weights_without_learning():
    close = 100 * np.cumprod(np.full(300, 1.001))
    comps = np.array([np.r_[np.zeros(100), np.ones(200)], np.zeros(300)])
    pos, w = learn.learn_positions(comps, close, eta=0.0, band=0.0)
    assert np.allclose(w, 0.5) and pos[-1] == pytest.approx(0.5)
