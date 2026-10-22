import math
import random
from datetime import datetime, timedelta, timezone

import pytest

from wescalp.validation import (
    deflated_sharpe,
    expected_max_sharpe,
    moments,
    pbo,
    probabilistic_sharpe,
    psr,
    sharpe,
    walk_forward,
    walk_forward_splits,
)

T0 = datetime(2024, 11, 23, tzinfo=timezone.utc)
WEEK = timedelta(weeks=1)


def test_splits_do_not_overlap_and_stay_before_end():
    end = T0 + 83.7 * WEEK
    splits = walk_forward_splits(T0, end)
    assert len(splits) == 5  # começos em 0, 4, 8, 12, 16 semanas; o de 20 passaria do fim
    for sp in splits:
        assert sp.train_start < sp.train_end == sp.test_start < sp.test_end <= end
        assert sp.train_end - sp.train_start == 60 * WEEK and sp.test_end - sp.test_start == 4 * WEEK
    # Os testes andam 4 semanas e não se sobrepõem entre si.
    for a, b in zip(splits, splits[1:]):
        assert b.test_start == a.test_end
    assert walk_forward_splits(T0, T0 + 63 * WEEK) == []


def test_sharpe_and_moments():
    assert sharpe([1, 1, 1]) == 0.0 and sharpe([1]) == 0.0
    assert sharpe([1, 3]) == pytest.approx(2 / math.sqrt(2))
    skew, kurt = moments([-1, 1] * 50)
    assert skew == pytest.approx(0) and kurt == pytest.approx(1)


def test_psr_known_values():
    assert psr(0.0, 100) == 0.5
    assert psr(0.0, 100, skew=-2, kurt=10) == 0.5
    # SR 0,1 com T = 101: √(T-1) = 10. Com g3 = 0,05 e g4 = 3 o denominador é √(1 - 0,005 + 0,005) = 1,
    # então z = 0,1 · 10 = 1 e PSR = Φ(1).
    assert psr(0.1, 101, skew=0.05, kurt=3) == pytest.approx(0.8413447, abs=1e-6)
    # Normal: denominador √(1 + 0,5 · 0,01) = 1,0024969, z = 0,997509 e Φ(z) ≈ Φ(1) - φ(1) · 0,002491 = 0,840742.
    assert psr(0.1, 101) == pytest.approx(0.840742, abs=1e-5)
    # Contra o próprio Sharpe dá 0,5; assimetria negativa piora o PSR de Sharpe positivo.
    assert psr(0.1, 101, sr_benchmark=0.1) == 0.5
    assert psr(0.1, 101, skew=-1) < psr(0.1, 101)


def test_expected_max_sharpe():
    assert expected_max_sharpe(1.0, 1) == 0.0
    # Máximo de 1000 normais padrão: ≈ 3,241; a aproximação do paper dá ≈ 3,25.
    assert expected_max_sharpe(1.0, 1000) == pytest.approx(3.241, rel=0.01)
    assert expected_max_sharpe(0.04, 1000) == pytest.approx(0.2 * expected_max_sharpe(1.0, 1000))
    assert expected_max_sharpe(1.0, 10) < expected_max_sharpe(1.0, 100)


def test_dsr_is_below_psr_with_many_trials():
    rng = random.Random(7)
    rets = [rng.gauss(0.1, 1) for _ in range(200)]
    trials = [rng.gauss(0, 0.07) for _ in range(50)]
    p, d = probabilistic_sharpe(rets), deflated_sharpe(rets, trials)
    assert d < p
    assert deflated_sharpe(rets, [sharpe(rets)]) == pytest.approx(p)  # N = 1 não deflaciona


def test_walk_forward_picks_best_in_train_and_reports_test():
    # Coluna 0 boa no treino, coluna 1 boa no teste.
    m = [[1 + (i % 2), -1 + (i % 2)] for i in range(6)] + [[-1, 1], [-2, 2]]
    (step,) = walk_forward(m, [(0, 6, 6, 8)])
    assert step.best == 0 and step.test_returns == [-1, -2] and step.train_sharpe > 0


def noise(rng, t, n):
    return [[rng.gauss(0, 1) for _ in range(n)] for _ in range(t)]


def test_pbo_near_zero_when_one_config_dominates():
    rng = random.Random(1)
    m = noise(rng, 160, 20)
    for row in m:
        row[3] += 2.0
    r = pbo(m, n_blocks=16)
    assert r.pbo == 0.0 and r.prob_oos_loss == 0.0
    assert len(r.logits) == math.comb(16, 8)


def test_pbo_near_half_with_pure_noise():
    rng = random.Random(42)
    vals = [pbo(noise(rng, 120, 30), n_blocks=8).pbo for _ in range(20)]
    assert sum(vals) / len(vals) == pytest.approx(0.5, abs=0.12)


def test_pbo_uneven_blocks_and_validation():
    rng = random.Random(3)
    r = pbo(noise(rng, 83, 5), n_blocks=16)  # blocos de 5 ou 6 linhas
    assert 0 <= r.pbo <= 1
    with pytest.raises(ValueError):
        pbo(noise(rng, 83, 5), n_blocks=15)
